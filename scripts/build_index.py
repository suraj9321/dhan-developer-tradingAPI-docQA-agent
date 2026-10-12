"""Rebuild the index from the raw export: parse -> chunk -> embed -> Qdrant.

    python -m scripts.build_index                 # 500-token chunks, base model
    python -m scripts.build_index --chunk 300 800 # several chunk sizes
    python -m scripts.build_index --model models/bge-small-dhan
"""

import argparse
import json
import time

from docsqa import store
from docsqa.config import settings
from docsqa.embed import get_embedder
from docsqa.ingest.chunk import chunk_sections, chunks_path, save_chunks
from docsqa.ingest.parse import load_sections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, nargs="+", default=[settings.chunk_tokens])
    ap.add_argument("--model", default=settings.embed_model)
    ap.add_argument("--no-index", action="store_true", help="only write chunk files")
    args = ap.parse_args()

    if not settings.raw_doc.exists():
        raise SystemExit(f"Missing {settings.raw_doc}; see README 'Getting the docs'.")

    sections = load_sections(settings.raw_doc)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    with (settings.processed_dir / "sections.jsonl").open("w") as f:
        for s in sections:
            f.write(json.dumps(s.to_dict()) + "\n")
    print(f"{len(sections)} sections")

    client = None if args.no_index else store.get_client()
    embedder = None if args.no_index else get_embedder(args.model)
    for size in args.chunk:
        chunks = chunk_sections(sections, size)
        save_chunks(chunks, chunks_path(settings.processed_dir, size))
        if args.no_index:
            print(f"c{size}: {len(chunks)} chunks (not indexed)")
            continue
        name = store.collection_name(args.model, size)
        t0 = time.time()
        store.build_collection(client, name, chunks, embedder)
        print(f"c{size}: {len(chunks)} chunks -> {name} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
