"""Long-context baseline: the whole documentation in one prompt, no retrieval.

The corpus is ~110K tokens, which fits a 128K-context model. This answers
the obvious question "why retrieve at all?" with measured accuracy, cost and
latency instead of an argument. Sections are numbered so the answer can be
cited and judged for faithfulness exactly like the RAG pipelines.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache

from docsqa.config import settings
from docsqa.ingest.chunk import Chunk
from docsqa.llm import LLMClient, Usage
from docsqa.rag import ANSWER_SYSTEM, REFUSAL, Answer, build_citations, is_refusal


@lru_cache(maxsize=1)
def _sections_as_chunks() -> list[Chunk]:
    chunks = []
    with (settings.processed_dir / "sections.jsonl").open() as f:
        for line in f:
            s = json.loads(line)
            prefix = " > ".join([s["sub_doc"], *s["heading_path"]])
            if s["endpoint"]:
                prefix += f"\nEndpoint: {s['method']} {s['endpoint']}"
            chunks.append(
                Chunk(
                    chunk_id=f"{s['section_id']}-all",
                    section_id=s["section_id"],
                    sub_doc=s["sub_doc"],
                    heading_path=s["heading_path"],
                    method=s["method"],
                    endpoint=s["endpoint"],
                    citation=s["citation"],
                    text=f"{prefix}\n\n{s['body']}",
                    n_tokens=0,
                )
            )
    return chunks


class LongContextPipeline:
    def __init__(self, llm: LLMClient | None = None, model: str | None = None):
        self.llm = llm or LLMClient()
        self.model = model or settings.answer_model
        self.chunks = _sections_as_chunks()
        self.context = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(self.chunks, 1))

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        usage = Usage()
        r = self.llm.chat(
            [
                {"role": "system", "content": ANSWER_SYSTEM},
                # Document first, question last: the instruction-following
                # position for long prompts, and a stable prefix for any
                # provider-side prompt caching.
                {"role": "user", "content": f"Documentation excerpts:\n\n{self.context}\n\nQuestion: {question}"},
            ],
            model=self.model,
            max_tokens=700,
        )
        usage.add(r)
        text = r.text.strip()
        refused = is_refusal(text)
        cites = [] if refused else build_citations(text, self.chunks)
        cited = {c.chunk_id for c in cites}
        return Answer(
            question=question,
            answer=REFUSAL if refused else text,
            refused=refused,
            citations=cites,
            # No retrieval step: record cited sections so the run file has
            # something to inspect, but retrieval metrics are not reported.
            retrieved=[c.chunk_id for c in self.chunks if c.chunk_id in cited],
            retrieved_sections=[c.section_id for c in self.chunks if c.chunk_id in cited],
            contexts=[c.text for c in self.chunks if c.chunk_id in cited],
            latency_s=time.perf_counter() - t0,
            usage=usage,
        )
