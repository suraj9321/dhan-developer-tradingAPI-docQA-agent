"""Self-checking RAG agent as a LangGraph state machine.

    retrieve -> grade_evidence -+-> answer -> verify -+-> END
        ^                       |                     |
        +---- rewrite_query <---+ (weak, 1 retry)     +-> revise -> verify (once)
                                                      +-> abstain -> END

Design notes
  * Every LLM step uses the answer model (open-weight Llama by default); the
    eval judge is a different model family, so the self-check is not graded
    by the model that performs it.
  * One query rewrite at most. Each retry costs a grader call and a
    retrieval; a second rewrite rarely finds evidence the first missed.
  * Retried retrieval merges with the first round instead of replacing it,
    so a rewrite can only add evidence, never lose it.
  * A failed citation check triggers one revision that drops the unsupported
    claims; only if that also fails does the agent abstain. Abstaining on the
    first failure throws away answers whose core is supported.

Verifier policy (`AgentConfig`)
  v1 ("strict")  claims must appear in the excerpts; two failed checks abstain.
                 Measured: the open-weight verifier rejected verbatim quotes
                 and simple arithmetic, abstaining on ~19% of answerable
                 questions and costing 14 points of correctness.
  v2 ("lenient") paraphrase and direct derivation count as support; only
                 absent or contradicted claims are flagged. Two failed checks
                 abstain only when the evidence was graded weak; with strong
                 evidence the revised answer (flagged claims removed) stands.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, StateGraph

from docsqa.config import settings
from docsqa.ingest.chunk import Chunk
from docsqa.llm import LLMClient, Usage, parse_json
from docsqa.rag import (
    ANSWER_SYSTEM,
    REFUSAL,
    Answer,
    build_citations,
    format_context,
    generate_answer,
    is_refusal,
)
from docsqa.retrieve import Retriever

MAX_RETRIES = 1
MAX_CONTEXT_CHUNKS = 8

GRADE_PROMPT = """Question about the DhanHQ trading API:
{question}

Retrieved documentation excerpts:
{context}

Do these excerpts contain the information needed to answer the question fully and for the right API (e.g. Indian market vs Global Stocks vs Python SDK)?

Return JSON only: {{"sufficient": true or false, "missing": "what information is missing, in a few words (empty if sufficient)"}}"""

REWRITE_PROMPT = """A search over the DhanHQ trading API documentation did not find enough information.

Original question: {question}
Missing information: {missing}

Write one new search query that is more likely to retrieve the missing documentation. Use the vocabulary the docs would use (endpoint paths, parameter names, page titles). Return only the query text."""

VERIFY_PROMPT = """Check whether an answer is supported by the documentation excerpts it cites.

Question: {question}

Answer:
{answer}

Excerpts (numbered as cited):
{context}

For each factual claim in the answer, check that the cited excerpt actually states it. Identifiers (endpoints, parameter names, enum values, codes, numbers) must appear in the excerpts exactly.

Return JSON only: {{"supported": true or false, "unsupported_claims": ["claim", "..."]}}"""

REVISE_INSTRUCTION = """Your previous answer contained claims that the excerpts do not support:
{claims}

Rewrite the answer keeping only what the excerpts state, with [n] citations. If nothing supported remains, reply with exactly: {refusal}"""


VERIFY_PROMPT_LENIENT = """Check whether an answer is supported by the documentation excerpts it cites.

Question: {question}

Answer:
{answer}

Excerpts (numbered as cited):
{context}

A claim is SUPPORTED if an excerpt states it, paraphrases it, or it follows directly from excerpt content (e.g. simple arithmetic on stated values, or quoting an identifier exactly as the excerpt spells it, even if the spelling looks odd). A claim is UNSUPPORTED only if no excerpt contains it or an excerpt contradicts it (wrong endpoint, wrong value, wrong API). Do not flag citation formatting or the presence of a refusal sentence.

Return JSON only: {{"supported": true or false, "unsupported_claims": ["claim", "..."]}}"""


@dataclass(frozen=True)
class AgentConfig:
    lenient_verifier: bool = False
    abstain_only_if_weak: bool = False


AGENT_V1 = AgentConfig()
AGENT_V2 = AgentConfig(lenient_verifier=True, abstain_only_if_weak=True)


class AgentState(TypedDict, total=False):
    question: str
    query: str
    chunks: list[Chunk]
    sufficient: bool
    missing: str
    retries: int
    answer: str
    supported: bool
    unsupported: list[str]
    revised: bool
    abstained: bool
    trace: list[str]
    usage: Usage


def _merge(old: list[Chunk], new: list[Chunk]) -> list[Chunk]:
    seen = {c.chunk_id for c in old}
    merged = list(old) + [c for c in new if c.chunk_id not in seen]
    return merged[:MAX_CONTEXT_CHUNKS]


class AgentPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient | None = None, model: str | None = None, config: AgentConfig = AGENT_V2):
        self.cfg = config
        self.retriever = retriever
        self.llm = llm or LLMClient()
        self.model = model or settings.answer_model
        self.graph = self._build()

    def _json_call(self, prompt: str, usage: Usage, max_tokens: int = 300) -> dict:
        r = self.llm.chat([{"role": "user", "content": prompt}], model=self.model, max_tokens=max_tokens)
        usage.add(r)
        try:
            out = parse_json(r.text)
            return out if isinstance(out, dict) else {}
        except ValueError:
            return {}

    # --- nodes ---------------------------------------------------------------

    def retrieve(self, s: AgentState) -> AgentState:
        hits = [h.chunk for h in self.retriever.retrieve(s["query"])]
        chunks = _merge(s.get("chunks", []), hits)
        return {"chunks": chunks, "trace": s["trace"] + [f"retrieve({s['query']!r}) -> {len(chunks)} chunks"]}

    def grade(self, s: AgentState) -> AgentState:
        v = self._json_call(GRADE_PROMPT.format(question=s["question"], context=format_context(s["chunks"])), s["usage"])
        # An unparseable grade counts as sufficient: proceeding to answer is
        # the cheaper failure mode, and the verifier still runs afterwards.
        ok = bool(v.get("sufficient", True))
        return {"sufficient": ok, "missing": v.get("missing", ""), "trace": s["trace"] + [f"grade -> {'sufficient' if ok else 'weak: ' + str(v.get('missing', ''))}"]}

    def rewrite(self, s: AgentState) -> AgentState:
        r = self.llm.chat(
            [{"role": "user", "content": REWRITE_PROMPT.format(question=s["question"], missing=s.get("missing") or "unspecified")}],
            model=self.model,
            max_tokens=80,
        )
        s["usage"].add(r)
        query = r.text.strip().strip('"').splitlines()[0] if r.text.strip() else s["question"]
        return {"query": query, "retries": s.get("retries", 0) + 1, "trace": s["trace"] + [f"rewrite -> {query!r}"]}

    def answer(self, s: AgentState) -> AgentState:
        text = generate_answer(s["question"], s["chunks"], self.llm, self.model, s["usage"])
        return {"answer": text, "trace": s["trace"] + ["answer" + (" (refused)" if is_refusal(text) else "")]}

    def verify(self, s: AgentState) -> AgentState:
        cited = build_citations(s["answer"], s["chunks"])
        if not cited:
            return {"supported": False, "unsupported": ["answer has no citations"], "trace": s["trace"] + ["verify -> no citations"]}
        # Show the verifier the excerpts under their original numbers so it can
        # match [n] markers in the answer.
        ctx = "\n\n".join(f"[{c.n}] {s['chunks'][c.n - 1].text}" for c in cited)
        prompt = VERIFY_PROMPT_LENIENT if self.cfg.lenient_verifier else VERIFY_PROMPT
        v = self._json_call(prompt.format(question=s["question"], answer=s["answer"], context=ctx), s["usage"], max_tokens=400)
        ok = bool(v.get("supported", True))
        claims = [str(c) for c in v.get("unsupported_claims", [])] if not ok else []
        return {"supported": ok, "unsupported": claims, "trace": s["trace"] + [f"verify -> {'supported' if ok else 'unsupported: ' + '; '.join(claims)[:200]}"]}

    def revise(self, s: AgentState) -> AgentState:
        r = self.llm.chat(
            [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": f"Documentation excerpts:\n\n{format_context(s['chunks'])}\n\nQuestion: {s['question']}"},
                {"role": "assistant", "content": s["answer"]},
                {"role": "user", "content": REVISE_INSTRUCTION.format(claims="\n".join(f"- {c}" for c in s["unsupported"]), refusal=REFUSAL)},
            ],
            model=self.model,
            max_tokens=700,
        )
        s["usage"].add(r)
        return {"answer": r.text.strip(), "revised": True, "trace": s["trace"] + ["revise"]}

    def abstain(self, s: AgentState) -> AgentState:
        return {"answer": REFUSAL, "abstained": True, "trace": s["trace"] + ["abstain"]}

    # --- routing -------------------------------------------------------------

    def after_grade(self, s: AgentState) -> str:
        if s["sufficient"] or s.get("retries", 0) >= MAX_RETRIES:
            return "answer"
        return "rewrite"

    def after_answer(self, s: AgentState) -> str:
        return END if is_refusal(s["answer"]) else "verify"

    def after_verify(self, s: AgentState) -> str:
        if s["supported"] or is_refusal(s["answer"]):
            return END
        if not s.get("revised"):
            return "revise"
        if self.cfg.abstain_only_if_weak and s.get("sufficient"):
            return END  # strong evidence: keep the revised answer
        return "abstain"

    def _build(self):
        g = StateGraph(AgentState)
        for name in ("retrieve", "grade", "rewrite", "answer", "verify", "revise", "abstain"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self.after_grade, {"answer": "answer", "rewrite": "rewrite"})
        g.add_edge("rewrite", "retrieve")
        g.add_conditional_edges("answer", self.after_answer, {END: END, "verify": "verify"})
        g.add_conditional_edges("verify", self.after_verify, {END: END, "revise": "revise", "abstain": "abstain"})
        g.add_edge("revise", "verify")
        g.add_edge("abstain", END)
        return g.compile()

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        usage = Usage()
        s = self.graph.invoke({"question": question, "query": question, "chunks": [], "retries": 0, "trace": [], "usage": usage})
        chunks = s["chunks"]
        refused = is_refusal(s["answer"])
        return Answer(
            question=question,
            answer=REFUSAL if refused else s["answer"],
            refused=refused,
            citations=[] if refused else build_citations(s["answer"], chunks),
            retrieved=[c.chunk_id for c in chunks],
            retrieved_sections=[c.section_id for c in chunks],
            contexts=[c.text for c in chunks],
            latency_s=time.perf_counter() - t0,
            usage=usage,
            trace=s["trace"],
        )
