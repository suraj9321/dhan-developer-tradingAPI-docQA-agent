from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from docsqa.config import ROOT, settings
from docsqa.ingest.chunk import chunks_path, load_chunks

DATA = ROOT / "eval" / "data"
RESULTS = ROOT / "eval" / "results"
RUNS = RESULTS / "runs"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path | str, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_sections() -> dict[str, dict]:
    return {s["section_id"]: s for s in read_jsonl(settings.processed_dir / "sections.jsonl")}


def section_texts(chunk_tokens: int = 500) -> dict[str, str]:
    """Full prefixed text per section, reassembled from its chunks."""
    parts: dict[str, list[str]] = defaultdict(list)
    for c in load_chunks(chunks_path(settings.processed_dir, chunk_tokens)):
        parts[c.section_id].append(c.text)
    return {sid: "\n\n".join(ts) for sid, ts in parts.items()}
