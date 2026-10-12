"""Fine-tune the embedding model on (query, passage) pairs.

MultipleNegativesRankingLoss treats every other passage in the batch as a
negative, so no negative mining is needed and there is no risk of mining a
"negative" that actually states the same fact (the corpus repeats facts
across sub-docs). The NO_DUPLICATES batch sampler keeps two queries for the
same passage out of one batch, where one would otherwise be scored as a
negative for the other.

Passages are the same prefixed chunk text that is indexed, so the model
learns the sub-doc/heading prefix as a signal. Queries get the BGE query
instruction, matching how the retriever embeds them.

    python -m finetune.train --epochs 3
    python -m scripts.build_index --model models/bge-small-dhan --chunk 500
    python -m eval.eval --system finetuned --embed models/bge-small-dhan
"""

from __future__ import annotations

import argparse
import json
import random

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from docsqa.config import ROOT, settings
from docsqa.embed import BGE_QUERY_INSTRUCTION, best_device
from docsqa.ingest.chunk import chunks_path, load_chunks
from eval.common import DATA, read_jsonl
from finetune.gen_pairs import OUT as PAIRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=settings.embed_model)
    ap.add_argument("--out", default=str(ROOT / "models" / "bge-small-dhan"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    reserved = {r["section_id"] for r in read_jsonl(DATA / "reserved_sections.jsonl")}
    eval_questions = {q["question"].strip().lower() for q in read_jsonl(DATA / "questions_eval.jsonl")}
    texts = {c.chunk_id: c.text for c in load_chunks(chunks_path(settings.processed_dir, 500))}

    pairs = read_jsonl(PAIRS)
    # Leakage guards: no eval-relevant passage, and no eval question text.
    assert not {p["section_id"] for p in pairs} & reserved, "pairs include eval-reserved sections"
    assert not {p["query"].strip().lower() for p in pairs} & eval_questions, "pairs include eval questions"

    prefix = BGE_QUERY_INSTRUCTION if "bge" in args.base.lower() else ""
    ds = Dataset.from_dict(
        {
            "anchor": [prefix + p["query"] for p in pairs],
            "positive": [texts[p["chunk_id"]] for p in pairs],
        }
    )

    model = SentenceTransformer(args.base, device=best_device())
    model.max_seq_length = 512
    loss = MultipleNegativesRankingLoss(model)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(ROOT / ".cache" / "ft-checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=10,
        seed=args.seed,
        report_to=[],
    )
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss).train()
    model.save(args.out)
    meta = {"base": args.base, "pairs": len(pairs), "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": args.seed}
    with open(f"{args.out}/docsqa_train.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {args.out}  {meta}")


if __name__ == "__main__":
    main()
