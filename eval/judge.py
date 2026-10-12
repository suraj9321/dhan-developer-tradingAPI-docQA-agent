"""LLM judge for answer correctness and faithfulness.

Binary labels rather than 1-5 scales: they are easier to validate against
reference labels (agreement and Cohen's kappa are well defined) and the
rubric for each is a single, checkable sentence.

  correct   the answer gives the key facts of the reference answer and
            contradicts nothing in it
  faithful  every factual claim is supported by the excerpts the answer
            cites (or, if it cites nothing, by the excerpts it was given)

Refusals are not sent to the judge: a refusal on an answerable question is
scored incorrect and trivially faithful.
"""

from __future__ import annotations

from docsqa.config import settings
from docsqa.llm import LLMClient, Usage, parse_json

JUDGE_PROMPT = """You are grading an answer produced by a documentation QA system for the DhanHQ trading API.

Question:
{question}

Reference answer (written from the documentation):
{reference}

System answer:
{answer}

Excerpts the system answer cites:
{excerpts}

Grade two things independently.

correct = 1 if the system answer states the key facts the question asks for, consistent with the reference answer (exact identifiers such as endpoints, enum values and codes must match). Extra correct detail is fine. Missing the core fact, giving a wrong identifier, or describing the wrong API (e.g. Global Stocks instead of the Indian market) is 0.

faithful = 1 if every factual claim in the system answer is supported by the cited excerpts above. Any claim that is not in the excerpts (even if true) makes it 0. Judge faithfulness only against the excerpts, not against the reference answer.

Return JSON only:
{{"correct": 0 or 1, "faithful": 0 or 1, "reason": "one short sentence"}}"""


def judge_answer(
    llm: LLMClient,
    question: str,
    reference: str,
    answer: str,
    cited_texts: list[str],
    usage: Usage | None = None,
) -> dict:
    excerpts = "\n\n".join(f"<<<\n{t}\n>>>" for t in cited_texts) or "(none)"
    messages = [{"role": "user", "content": JUDGE_PROMPT.format(question=question, reference=reference, answer=answer, excerpts=excerpts)}]
    text = ""
    for _ in range(2):
        r = llm.chat(messages, model=settings.judge_model, max_tokens=300)
        if usage is not None:
            usage.add(r)
        text = r.text
        try:
            v = parse_json(text)
            return {"correct": int(v["correct"]), "faithful": int(v["faithful"]), "reason": v.get("reason", "")}
        except (ValueError, KeyError, TypeError):
            # The judge occasionally reasons in prose first; ask once more for the object alone.
            messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Now return only the JSON object."},
            ]
    return {"correct": None, "faithful": None, "reason": f"unparseable: {text[:120]}"}
