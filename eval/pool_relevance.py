"""Pooled relevance judgments (TREC-style) for retrieval metrics.

A synthetic question is written from one section, but the corpus repeats
facts: the API reference, the developer guides, release notes and the
agent-skill pages often state the same limit or enum. Counting only the
source section as relevant penalises a retriever for returning an equally
good duplicate.

For every question the top sections from several retrievers are pooled, and
a model marks which of them fully answer the question on their own. Metrics
report both views:
  strict  only the source section(s) count
  pooled  source section(s) plus pooled sections judged fully answering

For disambiguation questions the distractor section is never added, even if
the model says it answers; it belongs to the wrong API by construction.

    python -m eval.pool_relevance
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from docsqa.config import settings
from docsqa.ingest.chunk import n_tokens
from docsqa.llm import LLMClient, parse_json
from docsqa.retrieve import Retriever, RetrieverConfig
from eval.common import DATA, read_jsonl, section_texts, write_jsonl

POOL_DEPTH = 10

PROMPT = """Question about the DhanHQ trading API:
{question}

Reference answer:
{answer}

Below are numbered documentation sections. List the numbers of sections that, ON THEIR OWN, state the core fact the question asks for (the reference answer may include extra detail that is not required). A developer reading only that section should be able to answer the question correctly. Sections that are merely on a related topic, or describe a different API (e.g. Global Stocks vs the Indian market), do not count.

{sections}

Return JSON only: {{"answering": [numbers]}}"""


def _trim(text: str, limit: int = 700) -> str:
    return text if n_tokens(text) <= limit else text[: limit * 4] + "\n[...]"


def main():
    rows = [q for q in read_jsonl(DATA / "questions_audited.jsonl") if q["audit"]["verdict"] != "drop"]
    texts = section_texts()
    retrievers = [Retriever(RetrieverConfig(mode=m, chunk_tokens=500)) for m in ("vector", "bm25")]
    llm = LLMClient(max_cost=2.0)

    def pool(q: dict) -> dict:
        cands: list[str] = []
        for r in retrievers:
            for h in r.retrieve(q["question"], k=POOL_DEPTH):
                sid = h.chunk.section_id
                if sid not in cands and sid not in q["gold_sections"] and sid != q.get("distractor_section"):
                    cands.append(sid)
        q = dict(q)
        if not cands:
            q["pooled_sections"] = []
            return q
        listing = "\n\n".join(f"[{i}]\n{_trim(texts[s])}" for i, s in enumerate(cands, 1))
        r = llm.chat(
            [{"role": "user", "content": PROMPT.format(question=q["question"], answer=q["reference_answer"], sections=listing)}],
            model=settings.gen_model,
            max_tokens=200,
            effort="low",  # without reasoning the model over-includes topically related pages
        )
        try:
            nums = parse_json(r.text).get("answering", [])
        except (ValueError, AttributeError):
            nums = []
        q["pooled_sections"] = [cands[n - 1] for n in nums if isinstance(n, int) and 1 <= n <= len(cands)]
        return q

    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(pool, rows))
    write_jsonl(DATA / "questions_pooled.jsonl", out)
    extra = sum(len(q["pooled_sections"]) for q in out)
    with_extra = sum(bool(q["pooled_sections"]) for q in out)
    print(f"{len(out)} questions; {with_extra} gained pooled sections ({extra} total); cost ${llm.usage.cost:.3f}")


if __name__ == "__main__":
    main()
