# DocsQA

Question answering over the DhanHQ trading API documentation, with cited answers, abstention when the docs don't cover a question, and an evaluation harness that measures both.

## Getting the docs

The source documentation is not redistributed in this repository. Export the DhanHQ API docs as a single Markdown file and place it at `data/raw/dhan-api-docs.md`.

## Build the index

```bash
docker compose up -d qdrant
uv sync --extra ui
uv run python -m scripts.build_index --chunk 300 500 800
uv run python -m scripts.search "cancel an order on global stocks" --mode hybrid
```
