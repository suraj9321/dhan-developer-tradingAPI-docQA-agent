"""Merge offline grades into a run and recompute its summary row.

    python -m eval.apply_grades hybrid --grader claude-opus-5.5

Grades file: eval/results/grades/<run>.jsonl with {qid, correct, faithful, reason},
produced by applying the eval/judge.py rubric to <run>.to_grade.jsonl.
"""

from __future__ import annotations

import argparse
import json

from docsqa.config import settings
from eval.common import RESULTS, RUNS, read_jsonl
from eval.eval import save, summarize

GRADES = RESULTS / "grades"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--grader", required=True, help="who/what produced the grades, recorded in the summary")
    args = ap.parse_args()

    rows = read_jsonl(RUNS / f"{args.run}.jsonl")
    grades = {g["qid"]: g for g in read_jsonl(GRADES / f"{args.run}.jsonl")}
    for r in rows:
        g = grades.get(r["qid"])
        if g and r["answerable"] and not r["refused"]:
            r.update({"correct": int(g["correct"]), "faithful": int(g["faithful"]), "judge_reason": g.get("reason", "")})
    prev = json.loads((RESULTS / "summary.json").read_text())[args.run]
    meta = {k: prev.get(k) for k in ("name", "system", "mode", "chunk", "embed", "rerank_model", "answer_model")}
    meta["judge"] = args.grader
    summary = summarize(rows, meta, settings.top_k)
    for k in ("errors", "error_samples", "wall_s"):
        if k in prev:
            summary[k] = prev[k]
    summary["judge_usd"] = 0.0
    save(args.run, rows, summary)


if __name__ == "__main__":
    main()
