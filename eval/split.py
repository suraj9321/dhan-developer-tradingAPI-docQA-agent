"""Apply manual review edits and split questions into eval / train-heldout.

The split is by section, not by question: once a question is in the eval
set, every section relevant to it (source and pooled) is reserved for eval,
and no training question may touch a reserved section. Otherwise a
fine-tuned embedder could memorise the exact passages the eval asks about.

Questions that straddle the boundary are discarded, not moved, so the eval
set is fixed by the seed alone.

    python -m eval.split --n-eval 150 --seed 13
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter

from docsqa.config import settings
from eval.common import DATA, read_jsonl, write_jsonl

DOC_LINK_RE = re.compile(r"\[`(/[^`\]]+)`\]\((/api/v2/[^)\s]+)\)")


def docs_slug_to_path() -> dict[str, str]:
    """Map docs-site page slugs to REST paths using the docs' own endpoint tables.

    Overview tables render as [`/ledger`](/api/v2/statements/get-ledger-report),
    and question generation sometimes copied the page slug into the reference
    answer as if it were the endpoint. A correct answer naming `/ledger` would
    then be graded wrong.
    """
    text = settings.raw_doc.read_text(encoding="utf-8")
    return {slug: path for path, slug in DOC_LINK_RE.findall(text)}


def normalize_reference(ref: str, mapping: dict[str, str]) -> str:
    return re.sub(r"/api/v2/[a-z0-9-]+/[a-z0-9-]+", lambda m: mapping.get(m.group(0), m.group(0)), ref)


def apply_edits(rows: list[dict], edits: list[dict]) -> list[dict]:
    by_qid = {e["qid"]: e for e in edits}
    mapping = docs_slug_to_path()
    out = []
    for r in rows:
        e = by_qid.get(r["qid"])
        r = dict(r)
        r["label_status"] = "draft"
        fixed = normalize_reference(r["reference_answer"], mapping)
        if fixed != r["reference_answer"]:
            r["reference_answer_original"] = r["reference_answer"]
            r["reference_answer"] = fixed
        if e:
            r["review_note"] = e["note"]
            if e["action"] == "drop":
                continue
            if e["action"] == "retype":
                r["type"] = e["type"]
            if e["action"] == "edit":
                r.update({k: e[k] for k in ("question", "reference_answer") if k in e})
        out.append(r)
    return out


def relevant(q: dict) -> set[str]:
    return set(q["gold_sections"]) | set(q.get("pooled_sections", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rows = [q for q in read_jsonl(DATA / "questions_pooled.jsonl") if q["audit"]["verdict"] != "drop"]
    rows = apply_edits(rows, read_jsonl(DATA / "manual_edits.jsonl"))
    write_jsonl(DATA / "questions_reviewed.jsonl", rows)
    print(f"{sum('reference_answer_original' in r for r in rows)} references had docs-page slugs mapped to REST paths")
    random.Random(args.seed).shuffle(rows)

    eval_qs = rows[: args.n_eval]
    reserved = set().union(*(relevant(q) for q in eval_qs))
    # A disambiguation distractor is the passage the eval expects NOT to win;
    # training on it would also leak.
    reserved |= {q["distractor_section"] for q in eval_qs if q.get("distractor_section")}
    train = [q for q in rows[args.n_eval :] if not (relevant(q) & reserved)]
    dropped = len(rows) - len(eval_qs) - len(train)

    write_jsonl(DATA / "questions_eval.jsonl", eval_qs)
    write_jsonl(DATA / "questions_train_heldout.jsonl", train)
    write_jsonl(DATA / "reserved_sections.jsonl", [{"section_id": s} for s in sorted(reserved)])
    print(f"eval {len(eval_qs)} {dict(Counter(q['type'] for q in eval_qs))}")
    print(f"train-heldout {len(train)}, discarded (overlap) {dropped}, reserved sections {len(reserved)}")


if __name__ == "__main__":
    main()
