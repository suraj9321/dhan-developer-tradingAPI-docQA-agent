"""Run the evaluation for one system configuration and update the results table.

    python -m eval.eval --system baseline
    python -m eval.eval --system vector --chunk 300 --retrieval-only
    python -m eval.eval --system rerank
    python -m eval.eval --system longctx
    python -m eval.eval --system finetuned --embed models/bge-small-dhan
    python -m eval.eval --system agent

Outputs
  eval/results/runs/<name>.jsonl   per-question answers and judgments (gitignored:
                                   contains documentation text)
  eval/results/summary.json        one row of metrics per run
  eval/results/results.md          the same rows as a Markdown table

Latency is the end-to-end time of `ask()`, with any LLM call served from the
disk cache charged at its originally measured latency, so reruns report the
same numbers as the first run.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from docsqa.config import settings
from docsqa.llm import BudgetExceeded, LLMClient, Usage
from docsqa.retrieve import Retriever, RetrieverConfig
from eval.common import DATA, RESULTS, RUNS, read_jsonl, write_jsonl
from eval.judge import judge_answer
from eval.metrics import hit_at_k, mean, mrr, percentile, recall_at_k

PRESETS = {
    "baseline": {"mode": "vector"},
    "vector": {"mode": "vector"},
    "bm25": {"mode": "bm25"},
    "hybrid": {"mode": "hybrid"},
    "rerank": {"mode": "rerank"},
    "finetuned": {"mode": "vector"},
    "finetuned-rerank": {"mode": "rerank"},
    "longctx": {"mode": None},
    "agent": {"mode": "hybrid"},
    "agent-v2": {"mode": "hybrid"},
}


def build_pipeline(system: str, mode: str | None, chunk: int, embed: str, llm: LLMClient, answer_model: str, rerank_model: str = settings.rerank_model):
    if system == "longctx":
        from docsqa.longctx import LongContextPipeline

        return None, LongContextPipeline(llm=llm, model=answer_model)
    retriever = Retriever(RetrieverConfig(mode=mode, chunk_tokens=chunk, embed_model=embed, rerank_model=rerank_model))
    if system in ("agent", "agent-v2"):
        from docsqa.agent import AGENT_V1, AGENT_V2, AgentPipeline

        cfg = AGENT_V1 if system == "agent" else AGENT_V2
        return retriever, AgentPipeline(retriever, llm=llm, model=answer_model, config=cfg)
    from docsqa.rag import RAGPipeline

    return retriever, RAGPipeline(retriever, llm=llm, model=answer_model)


def retrieval_metrics(rows: list[dict], k: int) -> dict:
    ans = [r for r in rows if r["answerable"] and r.get("retrieved_sections") is not None]
    if not ans:
        return {}
    return {
        "recall@5": mean(recall_at_k(r["retrieved_sections"], set(r["gold_sections"]), k) for r in ans),
        "hit@5_pooled": mean(hit_at_k(r["retrieved_sections"], set(r["gold_sections"]) | set(r["pooled_sections"]), k) for r in ans),
        "mrr": mean(mrr(r["retrieved_sections"], set(r["gold_sections"]) | set(r["pooled_sections"])) for r in ans),
    }


def by_type(rows: list[dict], key: str) -> dict:
    out = {}
    for t in sorted({r["type"] for r in rows if r["answerable"]}):
        vals = [r[key] for r in rows if r["type"] == t and r.get(key) is not None]
        out[t] = round(mean(vals), 3) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=PRESETS)
    ap.add_argument("--mode", default=None, help="override retrieval mode")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--embed", default=settings.embed_model)
    ap.add_argument("--rerank-model", default=settings.rerank_model)
    ap.add_argument("--answer-model", default=settings.answer_model)
    ap.add_argument("--name", default=None, help="row label in the results table")
    ap.add_argument("--retrieval-only", action="store_true", help="no LLM calls; retrieval metrics only")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cost", type=float, default=2.0, help="abort if this run's LLM spend exceeds this ($)")
    ap.add_argument("--judge", choices=["llm", "offline"], default="llm", help="offline: export answers for grading instead of calling the judge model")
    ap.add_argument("--fresh", action="store_true", help="bypass the response cache for the system under test (judge stays cached)")
    args = ap.parse_args()

    mode = args.mode or PRESETS[args.system]["mode"]
    name = args.name or args.system
    k = settings.top_k

    questions = read_jsonl(DATA / "questions_eval.jsonl") + read_jsonl(DATA / "unanswerable.jsonl")
    assert not any(q["qid"].startswith("q") and q.get("audit", {}).get("verdict") == "drop" for q in questions)
    if args.limit:
        questions = questions[: args.limit]

    llm = LLMClient(max_cost=args.max_cost, use_cache=not args.fresh)
    retriever, pipeline = build_pipeline(args.system, mode, args.chunk, args.embed, llm, args.answer_model, args.rerank_model)

    if args.retrieval_only:
        if retriever is None:
            raise SystemExit("retrieval-only is not defined for longctx")
        rows = []
        for q in questions:
            if q["answerable"]:
                hits = retriever.retrieve(q["question"], k=k)
                rows.append({**q, "retrieved_sections": [h.chunk.section_id for h in hits]})
        summary = {"name": name, "system": args.system, "mode": mode, "chunk": args.chunk, "embed": args.embed, "rerank_model": args.rerank_model if mode == "rerank" else None, "n": len(rows), **retrieval_metrics(rows, k)}
        summary["recall@5_by_type"] = by_type([{**r, "_r": recall_at_k(r["retrieved_sections"], set(r["gold_sections"]), k)} for r in rows], "_r")
        save(name, rows, summary, retrieval_only=True)
        return

    def run(q: dict) -> dict:
        try:
            ans = pipeline.ask(q["question"])
        except BudgetExceeded:
            raise
        except Exception as e:  # one provider error must not discard the whole run
            return {**q, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        u = ans.usage
        latency = ans.latency_s - u.live_s + u.latency_s
        return {
            **q,
            "answer": ans.answer,
            "refused": ans.refused,
            "citations": [c.__dict__ for c in ans.citations],
            "retrieved_sections": None if args.system == "longctx" else ans.retrieved_sections,
            "cited_texts": [ans.contexts[ans.retrieved.index(c.chunk_id)] for c in ans.citations if c.chunk_id in ans.retrieved],
            "contexts_if_uncited": ans.contexts if not ans.citations else [],
            "latency_s": latency,
            "usage": u.to_dict(),
            "trace": ans.trace,
        }

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run, questions))

    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    judge_usage = Usage()
    if args.judge == "llm":
        judge_llm = LLMClient(max_cost=args.max_cost)
        with ThreadPoolExecutor(4) as ex:
            rows = list(ex.map(lambda r: grade_with_llm(r, judge_llm, judge_usage), rows))
        judge_name = settings.judge_model
    else:
        rows = [grade_trivial(r) for r in rows]
        write_grading_packets(name, rows)
        judge_name = "offline"

    meta = {
        "name": name,
        "system": args.system,
        "mode": mode,
        "chunk": args.chunk,
        "embed": args.embed,
        "rerank_model": args.rerank_model if mode == "rerank" else None,
        "answer_model": args.answer_model,
        "judge": judge_name,
    }
    summary = summarize(rows, meta, k)
    summary.update({"judge_usd": judge_usage.cost, "errors": len(errors), "error_samples": [e["error"] for e in errors[:3]], "wall_s": time.time() - t0})
    save(name, rows, summary)


def grade_trivial(r: dict) -> dict:
    """Rows that need no judgment: unanswerable questions and refusals."""
    if not r["answerable"]:
        r["refusal_correct"] = int(r["refused"])
    elif r["refused"]:
        r.update({"correct": 0, "faithful": 1, "judge_reason": "refused"})
    return r


def grade_with_llm(r: dict, llm: LLMClient, usage: Usage) -> dict:
    r = grade_trivial(r)
    if r["answerable"] and not r["refused"]:
        texts = r["cited_texts"] or r["contexts_if_uncited"]
        j = judge_answer(llm, r["question"], r["reference_answer"], r["answer"], texts, usage)
        r.update({"correct": j["correct"], "faithful": j["faithful"], "judge_reason": j["reason"]})
    return r


def write_grading_packets(name: str, rows: list[dict]) -> None:
    """Export answers that need a correctness/faithfulness judgment.

    Offline grading applies the same rubric as eval/judge.py (JUDGE_PROMPT)
    and writes eval/results/grades/<name>.jsonl with {qid, correct, faithful,
    reason}; `python -m eval.apply_grades <name>` merges them back.
    """
    out = [
        {
            "qid": r["qid"],
            "question": r["question"],
            "reference_answer": r["reference_answer"],
            "answer": r["answer"],
            "excerpts": r["cited_texts"] or r["contexts_if_uncited"],
        }
        for r in rows
        if r["answerable"] and not r["refused"]
    ]
    write_jsonl(RUNS / f"{name}.to_grade.jsonl", out)
    print(f"{len(out)} answers need grading -> {RUNS / f'{name}.to_grade.jsonl'}")


def summarize(rows: list[dict], meta: dict, k: int) -> dict:
    ans_rows = [r for r in rows if r["answerable"]]
    una_rows = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans_rows if not r["refused"]]
    lat = [r["latency_s"] for r in rows]
    cost = [r["usage"]["cost"] for r in rows]
    return {
        **meta,
        "n": len(rows),
        **({} if meta["system"] == "longctx" else retrieval_metrics(rows, k)),
        "correct": mean(r.get("correct") for r in ans_rows),
        "faithful": mean(r.get("faithful") for r in answered),
        "refusal_acc": mean(r["refusal_correct"] for r in una_rows),
        "false_refusal": mean(int(r["refused"]) for r in ans_rows),
        "p50_s": percentile(lat, 50),
        "p95_s": percentile(lat, 95),
        "usd_per_1k": 1000 * mean(cost),
        "tokens_in_avg": mean(r["usage"]["input_tokens"] for r in rows),
        "llm_calls_avg": mean(r["usage"]["calls"] for r in rows),
        "ungraded": sum(1 for r in answered if r.get("correct") is None),
        "correct_by_type": by_type(rows, "correct"),
    }


def save(name: str, rows: list[dict], summary: dict, retrieval_only: bool = False) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    write_jsonl(RUNS / f"{name}.jsonl", rows)
    path = RESULTS / "summary.json"
    allrows = json.loads(path.read_text()) if path.exists() else {}
    prev = allrows.get(name, {})
    # A retrieval-only rerun must not erase answer metrics recorded earlier.
    allrows[name] = {**prev, **summary} if retrieval_only else summary
    path.write_text(json.dumps(allrows, indent=2))
    write_table(allrows)
    print(json.dumps(summary, indent=2))


def _f(v, digits=3):
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{v:.{digits}f}"


def write_table(allrows: dict) -> None:
    """Two tables: retrieval experiments (no LLM) and end-to-end runs."""
    ret = [
        "| Run | Retrieval | Embedding | Chunk | recall@5 | hit@5 (pooled) | MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    e2e = [
        "| Run | Retrieval | Answer model | Grader | Correct | Faithful | Refusal acc | False refusal | LLM calls / q | p50 s | p95 s | $ / 1K q |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in allrows.items():
        emb = (r.get("embed") or "").split("/")[-1]
        if r.get("recall@5") is not None and r.get("refusal_acc") is None:  # retrieval-only runs
            ret.append(f"| {name} | {r.get('mode')} | {emb} | {r.get('chunk')} | {_f(r.get('recall@5'))} | {_f(r.get('hit@5_pooled'))} | {_f(r.get('mrr'))} |")
        if r.get("refusal_acc") is not None:
            pending = r.get("ungraded")
            corr = "pending" if pending else _f(r.get("correct"))
            faith = "pending" if pending else _f(r.get("faithful"))
            e2e.append(
                f"| {name} | {r.get('mode') or 'none (full doc)'} | {(r.get('answer_model') or '').split('/')[-1]} | {r.get('judge') or '–'} "
                f"| {corr} | {faith} | {_f(r.get('refusal_acc'))} | {_f(r.get('false_refusal'))} | {_f(r.get('llm_calls_avg'), 1)} "
                f"| {_f(r.get('p50_s'), 2)} | {_f(r.get('p95_s'), 2)} | {_f(r.get('usd_per_1k'), 2)} |"
            )
    text = "## Retrieval (150 answerable questions, section-level)\n\n" + "\n".join(ret) + "\n\n## End to end (150 answerable + 20 unanswerable)\n\n" + "\n".join(e2e) + "\n"
    (RESULTS / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
