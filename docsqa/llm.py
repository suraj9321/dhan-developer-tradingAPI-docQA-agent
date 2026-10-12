"""Single entry point for every LLM call (answering, agent steps, judging, generation).

All calls go through LangChain's ChatOpenAI pointed at OpenRouter, so any
hosted model (open-weight Llama, Claude, Qwen, ...) is one config string away
and the LangGraph agent can use the same client.

Two concerns live here rather than at call sites:
  * Cost/latency accounting. Every result carries token counts and dollar
    cost from OpenRouter's published per-token pricing, so evals can report
    cost per query without guessing.
  * A content-addressed disk cache. Re-running an eval after changing only
    the metrics code must not re-bill thousands of calls. Cached results keep
    their original latency/cost so reported numbers stay those of a real call.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from docsqa.config import settings


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_s: float = 0.0
    cached: bool = False


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_s: float = 0.0

    def add(self, r: LLMResult) -> None:
        self.calls += 1
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.cost += r.cost
        self.latency_s += r.latency_s

    def merge(self, other: "Usage") -> None:
        for k in ("calls", "input_tokens", "output_tokens", "cost", "latency_s"):
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def to_dict(self) -> dict:
        return asdict(self)


class BudgetExceeded(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _pricing() -> dict[str, tuple[float, float]]:
    """$/token (prompt, completion) for every OpenRouter model, cached on disk for a day."""
    path = settings.cache_dir / "openrouter_models.json"
    if not path.exists() or time.time() - path.stat().st_mtime > 86400:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = httpx.get(f"{settings.openrouter_base_url}/models", timeout=30).json()["data"]
        path.write_text(json.dumps(data))
    data = json.loads(path.read_text())
    return {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])) for m in data}


def known_model(model: str) -> bool:
    return model in _pricing()


@lru_cache(maxsize=16)
def chat_model(model: str, temperature: float = 0.0, max_tokens: int = 1024) -> ChatOpenAI:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example)")
    return ChatOpenAI(
        model=model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=4,
        timeout=180,
    )


_ROLE = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}


class LLMClient:
    def __init__(self, use_cache: bool = True, max_cost: float | None = None):
        self.use_cache = use_cache
        self.max_cost = max_cost
        self.usage = Usage()
        self._lock = threading.Lock()
        self._cache_dir = settings.cache_dir / "llm"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model, messages, temperature, max_tokens) -> str:
        blob = json.dumps([model, messages, temperature, max_tokens], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResult:
        key = self._key(model, messages, temperature, max_tokens)
        path = self._cache_dir / f"{key}.json"
        if self.use_cache and path.exists():
            r = LLMResult(**json.loads(path.read_text()))
            r.cached = True
            with self._lock:
                self.usage.add(r)
            return r

        if self.max_cost is not None and self.usage.cost >= self.max_cost:
            raise BudgetExceeded(f"spent ${self.usage.cost:.2f} of ${self.max_cost:.2f} budget")

        lc_messages = [_ROLE[m["role"]](content=m["content"]) for m in messages]
        t0 = time.perf_counter()
        msg = chat_model(model, temperature, max_tokens).invoke(lc_messages)
        latency = time.perf_counter() - t0

        um = msg.usage_metadata or {}
        in_tok, out_tok = um.get("input_tokens", 0), um.get("output_tokens", 0)
        p_in, p_out = _pricing().get(model, (0.0, 0.0))
        r = LLMResult(
            text=msg.content if isinstance(msg.content, str) else str(msg.content),
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost=in_tok * p_in + out_tok * p_out,
            latency_s=latency,
        )
        if self.use_cache:
            path.write_text(json.dumps(asdict(r)))
        with self._lock:
            self.usage.add(r)
        return r


def parse_json(text: str) -> dict | list:
    """Extract the first JSON object/array from a model reply.

    Not every OpenRouter provider honours `response_format`, so prompts ask
    for JSON and this tolerates code fences or prose around it.
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError(f"no JSON in reply: {text[:200]!r}")
    start = min(starts)
    closer = "}" if text[start] == "{" else "]"
    end = text.rfind(closer)
    return json.loads(text[start : end + 1])
