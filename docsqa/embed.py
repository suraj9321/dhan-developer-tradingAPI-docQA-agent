from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# BGE models are trained with an instruction on the query side only; passages
# are embedded as-is. Dropping the instruction costs a few points of recall.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, model_name_or_path: str):
        self.name = model_name_or_path
        self.model = SentenceTransformer(model_name_or_path, device=best_device())
        self.dim = self.model.get_sentence_embedding_dimension()
        self.query_prefix = BGE_QUERY_INSTRUCTION if "bge" in model_name_or_path.lower() else ""

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            [self.query_prefix + t for t in texts], batch_size=64, normalize_embeddings=True
        )


@lru_cache(maxsize=4)
def get_embedder(model_name_or_path: str) -> Embedder:
    return Embedder(model_name_or_path)
