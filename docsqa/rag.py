"""Baseline RAG: retrieve top-k chunks, answer with numbered citations."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from docsqa.config import settings
from docsqa.ingest.chunk import Chunk
from docsqa.llm import LLMClient, Usage
from docsqa.retrieve import Retriever

REFUSAL = "I don't know based on the DhanHQ documentation."

ANSWER_SYSTEM = f"""You answer questions about the DhanHQ trading API using ONLY the numbered documentation excerpts provided.

Rules:
- Every factual sentence must end with the citation number(s) of the excerpt(s) that support it, like [1] or [2][3].
- Copy endpoint paths, parameter names, enum values and error codes exactly as written in the excerpts.
- Excerpts can come from different APIs (e.g. the Indian Trading API vs the Global Stocks API) that share page names. Use the excerpt that matches the API the question is about, and say which API you are describing when it matters.
- If the excerpts do not contain the answer, reply with exactly: {REFUSAL}
- Do not use outside knowledge. Be concise."""


@dataclass
class Citation:
    n: int
    chunk_id: str
    section_id: str
    citation: str
    sub_doc: str
    heading_path: list[str]


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    citations: list[Citation]  # only the excerpts the answer actually cites
    retrieved: list[str]  # chunk_ids in rank order, for retrieval metrics
    retrieved_sections: list[str]
    contexts: list[str]  # texts of retrieved chunks, for the faithfulness judge
    latency_s: float
    usage: Usage = field(default_factory=Usage)
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usage"] = self.usage.to_dict()
        return d


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, 1))


def is_refusal(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith("i don't know") or t.startswith("i do not know")


def cited_numbers(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})


def build_citations(text: str, chunks: list[Chunk]) -> list[Citation]:
    out = []
    for n in cited_numbers(text):
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            out.append(Citation(n, c.chunk_id, c.section_id, c.citation, c.sub_doc, c.heading_path))
    return out


def generate_answer(question: str, chunks: list[Chunk], llm: LLMClient, model: str, usage: Usage) -> str:
    r = llm.chat(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Documentation excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"},
        ],
        model=model,
        max_tokens=700,
    )
    usage.add(r)
    return r.text.strip()


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient | None = None, model: str | None = None):
        self.retriever = retriever
        self.llm = llm or LLMClient()
        self.model = model or settings.answer_model

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        usage = Usage()
        hits = self.retriever.retrieve(question)
        chunks = [h.chunk for h in hits]
        text = generate_answer(question, chunks, self.llm, self.model, usage)
        refused = is_refusal(text)
        return Answer(
            question=question,
            answer=REFUSAL if refused else text,
            refused=refused,
            citations=[] if refused else build_citations(text, chunks),
            retrieved=[c.chunk_id for c in chunks],
            retrieved_sections=[c.section_id for c in chunks],
            contexts=[c.text for c in chunks],
            latency_s=time.perf_counter() - t0,
            usage=usage,
        )
