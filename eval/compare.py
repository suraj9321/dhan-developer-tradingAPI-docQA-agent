"""Paired comparison of two runs on the same questions.

With 150 questions one question moves a rate by 0.67 points, so a gain of a
few points may be noise. This resamples questions with replacement (paired:
both runs see the same resample) and reports the 95% interval of the
difference, plus the count of questions each run wins.

    python -m eval.compare vector-c500 finetuned --metric recall@5
    python -m eval.compare hybrid agent --metric correct
"""

from __future__ import annotations

import argparse
import random

from docsqa.config import settings
from eval.common import RUNS, read_jsonl
from eval.metrics import mrr, recall_at_k


def per_question(rows: list[dict], metric: str) -> dict[str, float]:
    out = {}
    for r in rows:
        if metric in ("recall@5", "mrr"):
            if not r["answerable"] or r.get("retrieved_sections") is None:
                continue
            gold = set(r["gold_sections"])
            out[r["qid"]] = recall_at_k(r["retrieved_sections"], gold, settings.top_k) if metric == "recall@5" else mrr(r["retrieved_sections"], gold | set(r["pooled_sections"]))
        elif metric == "refusal_acc":
            if not r["answerable"]:
                out[r["qid"]] = float(r["refused"])
        elif r["answerable"] and r.get(metric) is not None:
            out[r["qid"]] = float(r[metric])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--metric", default="recall@5")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a = per_question(read_jsonl(RUNS / f"{args.a}.jsonl"), args.metric)
    b = per_question(read_jsonl(RUNS / f"{args.b}.jsonl"), args.metric)
    qids = sorted(set(a) & set(b))
    diffs = [b[q] - a[q] for q in qids]
    rng = random.Random(args.seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(args.iters))
    lo, hi = boots[int(0.025 * args.iters)], boots[int(0.975 * args.iters)]
    wins = sum(d > 0 for d in diffs)
    losses = sum(d < 0 for d in diffs)
    print(
        f"{args.metric}: {args.a}={sum(a[q] for q in qids)/n:.3f}  {args.b}={sum(b[q] for q in qids)/n:.3f}  "
        f"diff={sum(diffs)/n:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={n}  {args.b} better on {wins}, worse on {losses}"
    )


if __name__ == "__main__":
    main()
