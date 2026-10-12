"""Measure how often the LLM judge agrees with reference labels.

Labels live in eval/data/judge_labels.jsonl. Each row carries the labelled
answer and the chunk ids it cited, so the judge is re-applied to exactly the
answer that was labelled, independent of later eval runs. Excerpt text is
rebuilt from the chunk store rather than stored with the labels. Reported per dimension: raw agreement,
Cohen's kappa (agreement corrected for chance, which matters because most
answers are correct), and the confusion counts.

    python -m eval.validate_judge
"""

from __future__ import annotations

import json
from collections import Counter

from docsqa.config import settings
from docsqa.ingest.chunk import chunks_path, load_chunks
from docsqa.llm import LLMClient
from eval.common import DATA, RESULTS, read_jsonl
from eval.judge import judge_answer
from eval.metrics import cohen_kappa


def main():
    labels = read_jsonl(DATA / "judge_labels.jsonl")
    # Grade against the current (reviewed) reference answers, not the ones
    # stored in the run file at answer time.
    # Questions dropped from the eval set as ambiguous keep their audited reference here:
    # judge validation is about the judge, not about the question set.
    refs = {q["qid"]: q["reference_answer"] for q in read_jsonl(DATA / "questions_audited.jsonl")}
    refs.update({q["qid"]: q["reference_answer"] for q in read_jsonl(DATA / "questions_reviewed.jsonl")})
    texts = {c.chunk_id: c.text for c in load_chunks(chunks_path(settings.processed_dir, 500))}
    llm = LLMClient()
    pairs = []
    for lab in labels:
        excerpt_ids = lab["cited_chunks"] or lab["retrieved_chunks"]
        j = judge_answer(llm, lab["question"], refs[lab["qid"]], lab["answer"], [texts[c] for c in excerpt_ids])
        pairs.append((lab, j))

    report = {"n": len(pairs), "label_status": sorted({lab["label_status"] for lab, _ in pairs})}
    for dim in ("correct", "faithful"):
        a = [lab[dim] for lab, j in pairs if j[dim] is not None]
        b = [j[dim] for lab, j in pairs if j[dim] is not None]
        report[dim] = {
            "agreement": sum(x == y for x, y in zip(a, b)) / len(a),
            "kappa": cohen_kappa(a, b),
            "label_positive_rate": sum(a) / len(a),
            "confusion(label,judge)": {f"{k[0]}{k[1]}": v for k, v in sorted(Counter(zip(a, b)).items())},
        }
    report["disagreements"] = [
        {"qid": lab["qid"], "dim": d, "label": lab[d], "judge": j[d], "judge_reason": j["reason"]}
        for lab, j in pairs
        for d in ("correct", "faithful")
        if j[d] is not None and lab[d] != j[d]
    ]
    report["judge_usd"] = llm.usage.cost
    (RESULTS / "judge_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
