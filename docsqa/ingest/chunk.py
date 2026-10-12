"""Heading-aware chunking.

Design choices, driven by the shape of the corpus (median section ~66 words,
a handful of sections over 800 words):

  * One section is one chunk whenever it fits the token budget. Most sections
    are tiny, so splitting further would only scatter a parameter table away
    from the endpoint it describes.
  * Every chunk is prefixed with its sub-doc, heading path and endpoint. The
    body of "Cancel Order" is nearly identical in the Indian and Global Stocks
    APIs; the prefix is what makes the two embeddings (and BM25 terms) differ.
  * Markdown tables and fenced code blocks are atomic. A parameter table split
    in half produces chunks that answer "which values can X take?" wrongly.
  * Oversized sections are split on `###` first, then packed greedily, so a
    chunk boundary falls on a sub-heading rather than mid-paragraph.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import tiktoken

from docsqa.ingest.parse import Section


@lru_cache(maxsize=1)
def _enc():
    return tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(_enc().encode(text, disallowed_special=()))


@dataclass
class Chunk:
    chunk_id: str
    section_id: str
    sub_doc: str
    heading_path: list[str]
    method: str | None
    endpoint: str | None
    citation: str
    text: str  # prefix + body, what gets embedded and shown to the LLM
    n_tokens: int


def _blocks(body: str) -> list[str]:
    """Split a body into atomic blocks: paragraphs, whole tables, whole code fences."""
    blocks: list[str] = []
    cur: list[str] = []
    in_fence = False
    in_table = False

    def push():
        nonlocal cur
        if cur and any(l.strip() for l in cur):
            blocks.append("\n".join(cur).strip("\n"))
        cur = []

    for line in body.splitlines():
        s = line.lstrip()
        if s.startswith("```") or s.startswith("~~~"):
            if not in_fence:
                push()
                in_fence = True
                cur.append(line)
            else:
                cur.append(line)
                in_fence = False
                push()
            continue
        if in_fence:
            cur.append(line)
            continue
        is_row = s.startswith("|")
        if is_row and not in_table:
            in_table = True
        elif not is_row and in_table:
            in_table = False
            push()
        if not line.strip() and not in_table:
            push()
            continue
        cur.append(line)
    push()

    # A heading on its own carries no answer; glue it to the block it introduces.
    merged: list[str] = []
    for b in blocks:
        if merged and merged[-1].lstrip().startswith("#") and "\n" not in merged[-1]:
            merged[-1] = merged[-1] + "\n\n" + b
        else:
            merged.append(b)
    return merged


def _split_oversized(block: str, budget: int) -> list[str]:
    """Break a block that alone exceeds the budget.

    Tables are split by rows with the header repeated, so each piece is still
    a self-describing table (the conditional-order request tables run to ~6K
    tokens). Code fences are split by lines and re-fenced. Anything else is
    left whole.
    """
    lines = block.splitlines()
    if lines[0].lstrip().startswith("|") and len(lines) > 2:
        header, rows = lines[:2], lines[2:]
        fixed = n_tokens("\n".join(header))
    elif lines[0].lstrip().startswith(("```", "~~~")) and len(lines) > 2:
        header, rows = [lines[0]], lines[1:-1]
        fixed = n_tokens(lines[0]) + 2
        rows_tail = [lines[-1]]
    else:
        return [block]
    is_code = len(header) == 1
    pieces, buf, used = [], [], fixed
    for row in rows:
        rt = n_tokens(row) + 1
        if buf and used + rt > budget:
            pieces.append(buf)
            buf, used = [], fixed
        buf.append(row)
        used += rt
    if buf:
        pieces.append(buf)
    tail = rows_tail if is_code else []
    return ["\n".join(header + p + tail) for p in pieces]


def _split_h3(body: str) -> list[tuple[str | None, str]]:
    """Split a body on `### ` headings (outside code fences)."""
    parts: list[tuple[str | None, list[str]]] = [(None, [])]
    in_fence = False
    for line in body.splitlines():
        s = line.lstrip()
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = not in_fence
        if not in_fence and line.startswith("### "):
            parts.append((line[4:].strip(), [line]))
        else:
            parts[-1][1].append(line)
    return [(h, "\n".join(ls).strip()) for h, ls in parts if "\n".join(ls).strip()]


def _prefix(section: Section, sub_heading: str | None = None) -> str:
    path = [section.sub_doc, *section.heading_path]
    if sub_heading:
        path.append(sub_heading)
    lines = [" > ".join(path)]
    if section.endpoint:
        lines.append(f"Endpoint: {section.method} {section.endpoint}")
    return "\n".join(lines) + "\n\n"


def chunk_section(section: Section, max_tokens: int) -> list[Chunk]:
    out: list[tuple[str, str]] = []  # (prefix, body)
    full_prefix = _prefix(section)
    if n_tokens(full_prefix + section.body) <= max_tokens:
        out.append((full_prefix, section.body))
    else:
        for h3, part in _split_h3(section.body):
            prefix = _prefix(section, h3)
            budget = max_tokens - n_tokens(prefix)
            buf: list[str] = []
            buf_tok = 0
            blocks = []
            for block in _blocks(part):
                blocks.extend(_split_oversized(block, budget) if n_tokens(block) > budget else [block])
            for block in blocks:
                bt = n_tokens(block) + 1
                if buf and buf_tok + bt > budget:
                    out.append((prefix, "\n\n".join(buf)))
                    buf, buf_tok = [], 0
                buf.append(block)
                buf_tok += bt
            if buf:
                out.append((prefix, "\n\n".join(buf)))

    chunks = []
    for i, (prefix, body) in enumerate(out):
        text = prefix + body
        chunks.append(
            Chunk(
                chunk_id=f"{section.section_id}-{i:02d}",
                section_id=section.section_id,
                sub_doc=section.sub_doc,
                heading_path=section.heading_path,
                method=section.method,
                endpoint=section.endpoint,
                citation=section.citation,
                text=text,
                n_tokens=n_tokens(text),
            )
        )
    return chunks


def chunk_sections(sections: list[Section], max_tokens: int) -> list[Chunk]:
    return [c for s in sections for c in chunk_section(s, max_tokens)]


def chunks_path(processed_dir: Path, max_tokens: int) -> Path:
    return processed_dir / f"chunks_c{max_tokens}.jsonl"


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open() as f:
        return [Chunk(**json.loads(line)) for line in f]


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def bm25_tokens(text: str) -> list[str]:
    """Lexical tokens for BM25.

    Identifiers such as `DH-904`, `NSE_FNO` and `productType` are the terms
    users actually search for, so each identifier is kept whole and also
    broken into its camelCase / snake_case parts.
    """
    out: list[str] = []
    for tok in TOKEN_RE.findall(text):
        low = tok.lower()
        out.append(low)
        parts = [p.lower() for piece in re.split(r"[-_]", tok) for p in CAMEL_RE.findall(piece)]
        if len(parts) > 1:
            out.extend(parts)
    return out
