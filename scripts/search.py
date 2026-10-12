"""Print the top-k chunks for a query.

    python -m scripts.search "cancel a global stocks order" --mode hybrid
"""

import argparse

from docsqa.config import settings
from docsqa.retrieve import MODES, Retriever, RetrieverConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", choices=MODES, default="vector")
    ap.add_argument("--chunk", type=int, default=settings.chunk_tokens)
    ap.add_argument("--model", default=settings.embed_model)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    r = Retriever(RetrieverConfig(mode=args.mode, chunk_tokens=args.chunk, embed_model=args.model, k=args.k))
    for i, h in enumerate(r.retrieve(args.query), 1):
        c = h.chunk
        print(f"[{i}] {h.score:.4f}  {c.chunk_id}  {c.citation}")
        print("    " + c.text.split("\n\n", 1)[-1][:160].replace("\n", " "))


if __name__ == "__main__":
    main()
