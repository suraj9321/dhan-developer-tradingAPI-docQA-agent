"""HTTP API.

    uvicorn docsqa.api:app --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question": "What values can productType take?"}'

Read-only by design: the service only reads documentation. It holds no Dhan
credentials and has no code path that calls Dhan's trading endpoints.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from docsqa.config import settings
from docsqa.llm import LLMClient
from docsqa.retrieve import Retriever, RetrieverConfig

System = Literal["baseline", "hybrid", "rerank", "agent"]
DEFAULT_SYSTEM: System = os.environ.get("DOCSQA_SYSTEM", "baseline")  # type: ignore[assignment]
EMBED_MODEL = os.environ.get("DOCSQA_EMBED_MODEL", settings.embed_model)

app = FastAPI(title="DocsQA", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    system: System | None = None


class CitationOut(BaseModel):
    n: int
    citation: str
    sub_doc: str
    heading_path: list[str]
    chunk_id: str
    text: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[CitationOut]
    system: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    trace: list[str] = []


@lru_cache(maxsize=1)
def _llm() -> LLMClient:
    # The response cache is for reproducible evals; live traffic should not
    # be served stale answers after a re-index.
    return LLMClient(use_cache=False)


@lru_cache(maxsize=8)
def _pipeline(system: str):
    from docsqa.rag import RAGPipeline

    mode = {"baseline": "vector", "hybrid": "hybrid", "rerank": "rerank", "agent": "rerank"}[system]
    retriever = Retriever(RetrieverConfig(mode=mode, embed_model=EMBED_MODEL))
    if system == "agent":
        from docsqa.agent import AgentPipeline

        return AgentPipeline(retriever, llm=_llm())
    return RAGPipeline(retriever, llm=_llm())


@app.get("/health")
def health():
    return {"status": "ok", "answer_model": settings.answer_model, "default_system": DEFAULT_SYSTEM}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    system = req.system or DEFAULT_SYSTEM
    try:
        pipe = _pipeline(system)
        ans = pipe.ask(req.question)
    except Exception as e:  # surface upstream failures (LLM/Qdrant) as 503s
        raise HTTPException(status_code=503, detail=f"{type(e).__name__}: {e}") from e
    texts = dict(zip(ans.retrieved, ans.contexts))
    return AskResponse(
        answer=ans.answer,
        refused=ans.refused,
        citations=[
            CitationOut(
                n=c.n,
                citation=c.citation,
                sub_doc=c.sub_doc,
                heading_path=c.heading_path,
                chunk_id=c.chunk_id,
                text=texts.get(c.chunk_id, ""),
            )
            for c in ans.citations
        ],
        system=system,
        model=settings.answer_model,
        latency_ms=int(ans.latency_s * 1000),
        input_tokens=ans.usage.input_tokens,
        output_tokens=ans.usage.output_tokens,
        cost_usd=round(ans.usage.cost, 6),
        trace=ans.trace,
    )
