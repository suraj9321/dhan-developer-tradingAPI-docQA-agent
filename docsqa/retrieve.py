"""Retrieval strategies behind one interface.

  vector  dense search in Qdrant
  bm25    lexical search (in-process; the corpus is ~700 chunks)
  hybrid  both lists merged with reciprocal rank fusion
  rerank  hybrid top-N rescored by a cross-encoder, cut to top-k

RRF is used instead of score interpolation because BM25 and cosine scores
live on unrelated scales; ranks are comparable without any tuning.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Okapi

from docsqa import store
from docsqa.config import settings
from docsqa.embed import _LOCK as EMBED_LOCK
from docsqa.embed import get_embedder
from docsqa.ingest.chunk import Chunk, bm25_tokens, chunks_path, load_chunks

MODES = ("vector", "bm25", "hybrid", "rerank")


@dataclass(frozen=True)
class RetrieverConfig:
    mode: str = "vector"
    chunk_tokens: int = 500
    embed_model: str = settings.embed_model
    k: int = 5
    candidates: int = 20  # depth fed to RRF / the reranker
    rrf_k: int = 60  # standard RRF constant from Cormack et al. (2009)


@dataclass
class Hit:
    chunk: Chunk
    score: float


def rrf_merge(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


_CORPUS_LOCK = threading.Lock()


def _corpus(chunk_tokens: int) -> tuple[list[Chunk], BM25Okapi]:
    with _CORPUS_LOCK:
        return _load_corpus(chunk_tokens)


@lru_cache(maxsize=8)
def _load_corpus(chunk_tokens: int) -> tuple[list[Chunk], BM25Okapi]:
    chunks = load_chunks(chunks_path(settings.processed_dir, chunk_tokens))
    return chunks, BM25Okapi([bm25_tokens(c.text) for c in chunks])


@lru_cache(maxsize=1)
def _load_cross_encoder():
    from sentence_transformers import CrossEncoder

    from docsqa.embed import best_device

    return CrossEncoder(settings.rerank_model, device=best_device())


def _cross_encoder():
    with EMBED_LOCK:
        return _load_cross_encoder()


class Retriever:
    def __init__(self, cfg: RetrieverConfig = RetrieverConfig()):
        if cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.cfg = cfg
        self.chunks, self.bm25 = _corpus(cfg.chunk_tokens)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self._client = None
        self.collection = store.collection_name(cfg.embed_model, cfg.chunk_tokens)

    @property
    def client(self):
        if self._client is None:
            self._client = store.get_client()
        return self._client

    def _vector(self, query: str, n: int) -> list[Hit]:
        vec = get_embedder(self.cfg.embed_model).embed_queries([query])[0]
        return [Hit(c, s) for c, s in store.search(self.client, self.collection, vec, n)]

    def _bm25(self, query: str, n: int) -> list[Hit]:
        scores = self.bm25.get_scores(bm25_tokens(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]

    def _hybrid(self, query: str, n: int) -> list[Hit]:
        depth = max(n, self.cfg.candidates) * 2
        dense = [h.chunk.chunk_id for h in self._vector(query, depth)]
        sparse = [h.chunk.chunk_id for h in self._bm25(query, depth)]
        merged = rrf_merge([dense, sparse], self.cfg.rrf_k)[:n]
        return [Hit(self.by_id[cid], s) for cid, s in merged]

    def retrieve(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or self.cfg.k
        mode = self.cfg.mode
        if mode == "vector":
            return self._vector(query, k)
        if mode == "bm25":
            return self._bm25(query, k)
        if mode == "hybrid":
            return self._hybrid(query, k)
        cands = self._hybrid(query, max(k, self.cfg.candidates))
        with EMBED_LOCK:
            scores = _cross_encoder().predict([(query, h.chunk.text) for h in cands], show_progress_bar=False)
        ranked = sorted(zip(cands, scores), key=lambda t: float(t[1]), reverse=True)[:k]
        return [Hit(h.chunk, float(s)) for h, s in ranked]
