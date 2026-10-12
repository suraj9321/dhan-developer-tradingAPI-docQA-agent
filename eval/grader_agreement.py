"""Agreement between two graders on the same answers.

Used to cross-check offline grades against the validated LLM judge on the
baseline run (the judge's verdicts live in the run file; the offline grades
were produced blind to them).

    python -m eval.grader_agreement baseline
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from eval.common import RESULTS, RUNS, read_jsonl
from eval.metrics import cohen_kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    judged = {r["qid"]: r for r in read_jsonl(RUNS / f"{args.run}.jsonl") if r.get("correct") is not None and not r["refused"]}
    offline = {g["qid"]: g for g in read_jsonl(RESULTS / "grades" / f"{args.run}.jsonl")}
    qids = sorted(set(judged) & set(offline))
    report = {"run": args.run, "n": len(qids)}
    for dim in ("correct", "faithful"):
        a = [int(judged[q][dim]) for q in qids]
        b = [int(offline[q][dim]) for q in qids]
        report[dim] = {
            "agreement": round(sum(x == y for x, y in zip(a, b)) / len(a), 3),
            "kappa": round(cohen_kappa(a, b), 3),
            "judge_rate": round(sum(a) / len(a), 3),
            "offline_rate": round(sum(b) / len(b), 3),
            "confusion(judge,offline)": {f"{k[0]}{k[1]}": v for k, v in sorted(Counter(zip(a, b)).items())},
        }
    report["disagreements"] = [
        {"qid": q, "dim": d, "judge": judged[q][d], "offline": offline[q][d], "judge_reason": judged[q]["judge_reason"], "offline_reason": offline[q]["reason"]}
        for q in qids
        for d in ("correct", "faithful")
        if int(judged[q][d]) != int(offline[q][d])
    ]
    (RESULTS / f"grader_agreement_{args.run}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}, indent=2))
    for d in report["disagreements"]:
        print(f"- {d['qid']} {d['dim']}: judge={d['judge']} offline={d['offline']} | J: {d['judge_reason'][:110]} | O: {d['offline_reason'][:110]}")


if __name__ == "__main__":
    main()
