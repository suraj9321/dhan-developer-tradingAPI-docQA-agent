# One image for the indexer, API and UI. Models and the tokenizer encoding
# are fetched at build time so containers start without downloading anything.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    HF_HOME=/opt/hf TIKTOKEN_CACHE_DIR=/opt/tiktoken
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
# torch resolves to the CPU-only wheel on Linux (see [tool.uv.sources]),
# keeping the image ~2GB smaller than the default CUDA build.
RUN uv sync --frozen --no-install-project --extra ui
COPY docsqa ./docsqa
COPY scripts ./scripts
COPY app ./app
RUN uv sync --frozen --extra ui
# The encoding host can be slow and drop connections; retry rather than fail the build.
RUN for i in 1 2 3 4 5; do \
      uv run python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" && break; \
      echo "retrying tokenizer download ($i)"; sleep 5; \
    done && test -n "$(ls /opt/tiktoken)"
RUN uv run python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501
CMD ["uvicorn", "docsqa.api:app", "--host", "0.0.0.0", "--port", "8000"]
