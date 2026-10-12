"""Generate (query, passage) training pairs for embedding fine-tuning.

Pairs come only from chunks whose section is NOT reserved by the eval split
(eval/data/reserved_sections.jsonl), plus the held-out synthetic questions.
Each chunk gets a few queries in different registers (terse keyword search,
natural-language question, task description) because real queries to a docs
assistant mix all three, and the base model is weakest on the terse,
identifier-heavy kind.

    python -m finetune.gen_pairs --per-chunk 3
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor

from docsqa.config import ROOT, settings
from docsqa.ingest.chunk import chunks_path, load_chunks
from docsqa.llm import LLMClient, parse_json
from eval.common import DATA, read_jsonl, write_jsonl

OUT = ROOT / "finetune" / "data" / "pairs.jsonl"

PROMPT = """Here is one passage from the DhanHQ trading API documentation:

<<<
{text}
>>>

Write {n} different search queries a developer might type when this passage is exactly what they need. Vary the style:
- a terse keyword query (identifiers, endpoint words)
- a natural-language question
- a task description ("I want to ...")
Each query must be answerable from this passage. Mention the API context (e.g. Global Stocks, Python SDK, MCP, Dhan Cloud) only when the passage is specific to it. Do not copy the heading verbatim.

Return JSON only: {{"queries": ["...", "..."]}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-chunk", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    reserved = {r["section_id"] for r in read_jsonl(DATA / "reserved_sections.jsonl")}
    chunks = [c for c in load_chunks(chunks_path(settings.processed_dir, args.chunk)) if c.section_id not in reserved]
    # Skip near-empty chunks (index pages that are a heading and one line)
    chunks = [c for c in chunks if len(c.text.split()) >= 30]

    llm = LLMClient(max_cost=2.0)

    def gen(c):
        r = llm.chat(
            [{"role": "user", "content": PROMPT.format(text=c.text[:6000], n=args.per_chunk)}],
            model=settings.gen_model,
            temperature=0.8,
            max_tokens=400,
        )
        try:
            qs = parse_json(r.text).get("queries", [])
        except (ValueError, AttributeError):
            return []
        return [{"query": q.strip(), "chunk_id": c.chunk_id, "section_id": c.section_id, "source": "generated"} for q in qs if isinstance(q, str) and q.strip()]

    with ThreadPoolExecutor(8) as ex:
        pairs = [p for ps in ex.map(gen, chunks) for p in ps]

    # Held-out synthetic questions are also valid training queries; their
    # sections are disjoint from the eval set by construction of the split.
    by_section = {}
    for c in load_chunks(chunks_path(settings.processed_dir, args.chunk)):
        by_section.setdefault(c.section_id, c.chunk_id)
    for q in read_jsonl(DATA / "questions_train_heldout.jsonl"):
        for sid in q["gold_sections"]:
            pairs.append({"query": q["question"], "chunk_id": by_section[sid], "section_id": sid, "source": "heldout"})

    assert not {p["section_id"] for p in pairs} & reserved, "training pairs touch eval-reserved sections"
    random.Random(args.seed).shuffle(pairs)
    write_jsonl(OUT, pairs)
    print(f"{len(pairs)} pairs from {len(chunks)} chunks ({len({p['section_id'] for p in pairs})} sections); cost ${llm.usage.cost:.3f}")


if __name__ == "__main__":
    main()
