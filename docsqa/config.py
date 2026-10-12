from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # The answering/agent model is open-weight. Judging and question generation
    # use a different model family so answers are never graded by the model
    # that wrote them; generation is high-volume, so it uses the cheaper tier.
    answer_model: str = "meta-llama/llama-3.3-70b-instruct"
    judge_model: str = "anthropic/claude-sonnet-5.5"
    gen_model: str = "anthropic/claude-haiku-5.5"

    qdrant_url: str = "http://localhost:6333"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    raw_doc: Path = ROOT / "data" / "raw" / "dhan-api-docs.md"
    processed_dir: Path = ROOT / "data" / "processed"
    cache_dir: Path = ROOT / ".cache"

    chunk_tokens: int = 500
    top_k: int = 5


settings = Settings()
