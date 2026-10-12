"""Filter synthetic questions with a stronger model before any manual review.

Synthetic questions fail in predictable ways: the reference answer drifts
from the excerpt, the question is answerable from general knowledge, or (for
disambiguation) both APIs would give the same answer, so the question does
not actually test disambiguation. Each question gets a keep / fix / drop
verdict; fixes replace the question or reference answer in place.

    python -m eval.validate_questions
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from docsqa.config import settings
from docsqa.llm import LLMClient, parse_json
from eval.common import DATA, read_jsonl, section_texts, write_jsonl

PROMPT = """You are auditing an evaluation question for a documentation QA system over the DhanHQ trading API.

Question type: {qtype}
Question: {question}
Reference answer: {answer}

Gold excerpt(s) the question was written from:
{gold}
{distractor}
Check:
1. Is the question fully answerable from the gold excerpt(s) alone?
2. Is the reference answer correct and complete according to the excerpt(s)? Identifiers must match exactly.
3. Is the question specific enough that a developer would recognise what it asks, without seeing the excerpt?
4. For disambiguation: does the question clearly point to the gold excerpt's API, and would the distractor excerpt give a different answer?

Verdict:
- "keep" if all checks pass
- "fix" if a small edit to the question and/or reference answer makes all checks pass (supply the edited versions)
- "drop" otherwise

Return JSON only:
{{"verdict": "keep|fix|drop", "question": "...", "answer": "...", "reason": "one sentence"}}"""


def audit(llm: LLMClient, q: dict, texts: dict[str, str], max_tokens: int = 600) -> dict:
    gold = "\n\n".join(f"<<<\n{texts[s]}\n>>>" for s in q["gold_sections"])
    distractor = ""
    if q.get("distractor_section"):
        distractor = f"\nDistractor excerpt (different API, should NOT be the answer):\n<<<\n{texts[q['distractor_section']]}\n>>>\n"
    r = llm.chat(
        [{"role": "user", "content": PROMPT.format(qtype=q["type"], question=q["question"], answer=q["reference_answer"], gold=gold, distractor=distractor)}],
        model=settings.judge_model,
        max_tokens=max_tokens,
    )
    try:
        v = parse_json(r.text)
    except ValueError:
        v = {"verdict": "drop", "reason": "unparseable audit"}
    q = dict(q)
    q["audit"] = {"verdict": v.get("verdict", "drop"), "reason": v.get("reason", ""), "model": settings.judge_model}
    if v.get("verdict") == "fix":
        q["question"] = v.get("question") or q["question"]
        q["reference_answer"] = v.get("answer") or q["reference_answer"]
    return q


def main():
    rows = read_jsonl(DATA / "questions_raw.jsonl")
    texts = section_texts()
    llm = LLMClient(max_cost=5.0)
    with ThreadPoolExecutor(8) as pool:
        audited = list(pool.map(lambda q: audit(llm, q, texts), rows))
        # Verbose audits can overrun the reply budget; give those a second, longer attempt.
        retry = [i for i, q in enumerate(audited) if q["audit"]["reason"] == "unparseable audit"]
        for i, q in zip(retry, pool.map(lambda i: audit(llm, rows[i], texts, max_tokens=1500), retry)):
            audited[i] = q
    kept = [q for q in audited if q["audit"]["verdict"] in ("keep", "fix")]
    write_jsonl(DATA / "questions_audited.jsonl", audited)
    from collections import Counter

    print(Counter(q["audit"]["verdict"] for q in audited), Counter((q["type"], q["audit"]["verdict"]) for q in audited if q["audit"]["verdict"] == "drop"))
    print(f"kept {len(kept)}/{len(rows)}  cost ${llm.usage.cost:.3f}")


if __name__ == "__main__":
    main()
