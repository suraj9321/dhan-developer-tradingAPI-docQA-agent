"""Generate synthetic eval questions grounded in specific sections.

Each question records the section(s) it was written from; those are the gold
retrieval targets. Relevance is judged at section level rather than chunk
level so that the same question set is valid across chunk-size experiments
(a 300-token index splits sections that a 800-token index keeps whole).

Question types and why each exists:
  lookup          "which endpoint does X" - tests routing to the right page
  exact_fact      enum values, limits, error codes - checkable answers
  disambiguation  same page title in two APIs; the question names the API,
                  and only one of the two sections is correct
  multi_step      needs two related sections from the same API

    python -m eval.gen_questions --n 240 --seed 7
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from docsqa.config import settings
from docsqa.llm import LLMClient, parse_json
from docsqa.ingest.chunk import n_tokens
from eval.common import DATA, load_sections, section_texts, write_jsonl

MIX = {"lookup": 0.30, "exact_fact": 0.35, "disambiguation": 0.20, "multi_step": 0.15}

TYPE_GUIDE = {
    "lookup": "Ask which endpoint, page, method or feature handles a task described in the excerpt. Describe the task in your own words; do not quote the page title.",
    "exact_fact": "Ask for one precise, checkable fact stated in the excerpt: allowed enum values, a parameter's type or whether it is required, a limit, an error code meaning, a status code, a default value.",
    "disambiguation": "Two excerpts from DIFFERENT APIs share a page name. Write a question that is only correctly answered by excerpt A. The question must name A's API context naturally (e.g. 'for US stocks', 'in the Python SDK', 'via the MCP server') without copying the heading verbatim.",
    "multi_step": "Write a question whose full answer needs information from BOTH excerpts (e.g. a prerequisite in one and the call in the other, or comparing two related calls).",
}

GEN_PROMPT = """You are writing evaluation questions for a question-answering system over the DhanHQ trading API documentation.

Question type: {qtype}
Instructions: {guide}

{excerpts}

Write the question a developer would realistically ask. Avoid yes/no questions. The answer must be fully supported by the excerpt(s) above.

Return JSON only:
{{"question": "...", "answer": "reference answer in at most 60 words, quoting exact identifiers", "usable": true}}
Set "usable" to false if the excerpt has no meaningful content to ask about."""


def _fmt(label: str, text: str, limit: int = 1800) -> str:
    if n_tokens(text) > limit:
        text = text[: limit * 4]
    return f"Excerpt {label}:\n<<<\n{text}\n>>>"


def build_tasks(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sections = load_sections()
    texts = section_texts()

    # Sections with too little content produce trivia like "what is the title of this page"
    eligible = [sid for sid, s in sections.items() if len(s["body"].split()) >= 25 and sid in texts]

    by_title: dict[str, list[str]] = defaultdict(list)
    for sid in eligible:
        by_title[sections[sid]["title"]].append(sid)
    dup_pairs = [
        (a, b)
        for sids in by_title.values()
        for a in sids
        for b in sids
        if a != b and sections[a]["sub_doc"] != sections[b]["sub_doc"]
    ]

    # Neighbouring sections within the same sub-doc are the natural multi-step pairs
    ordered = sorted(eligible)
    neighbour_pairs = [
        (a, b)
        for a, b in zip(ordered, ordered[1:])
        if sections[a]["sub_doc"] == sections[b]["sub_doc"]
    ]

    counts = {t: round(n * w) for t, w in MIX.items()}
    tasks = []
    singles = rng.sample(eligible, min(len(eligible), counts["lookup"] + counts["exact_fact"]))
    for i, sid in enumerate(singles):
        qtype = "lookup" if i < counts["lookup"] else "exact_fact"
        tasks.append({"type": qtype, "gold": [sid], "excerpts": _fmt("A", texts[sid])})
    for a, b in rng.sample(dup_pairs, min(len(dup_pairs), counts["disambiguation"])):
        ex = _fmt("A (the correct one)", texts[a]) + "\n\n" + _fmt("B (the distractor, different API)", texts[b])
        tasks.append({"type": "disambiguation", "gold": [a], "distractor": b, "excerpts": ex})
    for a, b in rng.sample(neighbour_pairs, min(len(neighbour_pairs), counts["multi_step"])):
        tasks.append({"type": "multi_step", "gold": [a, b], "excerpts": _fmt("A", texts[a]) + "\n\n" + _fmt("B", texts[b])})
    rng.shuffle(tasks)
    return tasks


def run_task(llm: LLMClient, task: dict) -> dict | None:
    prompt = GEN_PROMPT.format(qtype=task["type"], guide=TYPE_GUIDE[task["type"]], excerpts=task["excerpts"])
    r = llm.chat([{"role": "user", "content": prompt}], model=settings.gen_model, temperature=0.7, max_tokens=900)
    try:
        out = parse_json(r.text)
    except ValueError:
        return None
    if not out.get("usable", True) or not out.get("question"):
        return None
    row = {
        "question": out["question"].strip(),
        "reference_answer": out.get("answer", "").strip(),
        "type": task["type"],
        "gold_sections": task["gold"],
        "answerable": True,
        "origin": "synthetic",
        "gen_model": settings.gen_model,
    }
    if "distractor" in task:
        row["distractor_section"] = task["distractor"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(DATA / "questions_raw.jsonl"))
    args = ap.parse_args()

    llm = LLMClient(max_cost=5.0)
    tasks = build_tasks(args.n, args.seed)
    with ThreadPoolExecutor(8) as pool:
        rows = [r for r in pool.map(lambda t: run_task(llm, t), tasks) if r]
    for i, r in enumerate(rows):
        r["qid"] = f"q{i:04d}"
    write_jsonl(args.out, rows)
    by_type = defaultdict(int)
    for r in rows:
        by_type[r["type"]] += 1
    print(f"{len(rows)} questions -> {args.out}  {dict(by_type)}  cost ${llm.usage.cost:.3f}")


if __name__ == "__main__":
    main()
