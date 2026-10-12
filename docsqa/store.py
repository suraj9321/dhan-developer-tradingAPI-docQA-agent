from __future__ import annotations

import re
import uuid
from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from docsqa.config import settings
from docsqa.embed import Embedder
from docsqa.ingest.chunk import Chunk


class QdrantUnavailable(RuntimeError):
    pass


def get_client() -> QdrantClient:
    client = QdrantClient(url=settings.qdrant_url, timeout=30)
    try:
        client.get_collections()
    except Exception as e:  # connection refused, DNS, timeout
        raise QdrantUnavailable(
            f"Cannot reach Qdrant at {settings.qdrant_url}. Start it with `docker compose up -d qdrant`."
        ) from e
    return client


def collection_name(embed_model: str, chunk_tokens: int) -> str:
    """One collection per (embedding model, chunk size) so experiments never overwrite each other."""
    tag = re.sub(r"[^a-z0-9]+", "-", embed_model.lower().split("/")[-1]).strip("-")
    return f"dhan_{tag}_c{chunk_tokens}"


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def build_collection(client: QdrantClient, name: str, chunks: list[Chunk], embedder: Embedder) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config=qm.VectorParams(size=embedder.dim, distance=qm.Distance.COSINE),
    )
    client.create_payload_index(name, "sub_doc", qm.PayloadSchemaType.KEYWORD)
    vectors = embedder.embed_passages([c.text for c in chunks])
    batch = 128
    for i in range(0, len(chunks), batch):
        client.upsert(
            name,
            points=[
                qm.PointStruct(id=_point_id(c.chunk_id), vector=v.tolist(), payload=asdict(c))
                for c, v in zip(chunks[i : i + batch], vectors[i : i + batch])
            ],
        )


def search(client: QdrantClient, name: str, query_vec, limit: int, sub_doc: str | None = None) -> list[tuple[Chunk, float]]:
    flt = None
    if sub_doc:
        flt = qm.Filter(must=[qm.FieldCondition(key="sub_doc", match=qm.MatchValue(value=sub_doc))])
    res = client.query_points(name, query=list(map(float, query_vec)), limit=limit, query_filter=flt, with_payload=True)
    return [(Chunk(**p.payload), p.score) for p in res.points]
