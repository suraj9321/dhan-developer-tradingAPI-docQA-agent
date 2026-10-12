"""Parse the DhanHQ Markdown export into sections.

The export concatenates several documentation sites (the Indian REST API, the
Global Stocks API, developer guides, the Python SDK, agent skills, Dhan Cloud
and the MCP server) without any explicit marker saying which site a page came
from. Page titles such as "Cancel Order" or "Get Holdings" repeat across those
sites, so recovering the sub-doc is what lets retrieval and citations tell
them apart.

Rules:
  * `---` is used both as a page separator and as a horizontal rule, so it is
    not trusted on its own. A `## ` heading directly after `---` starts a page;
    any other `## ` heading starts a section inside the current page.
  * Lines inside fenced code blocks are never headings (bash comments such as
    `# Install` appear in curl examples).
  * Sub-docs are assigned by a sequence of landmark page titles, checked in
    order, because the export keeps each site's pages contiguous.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

ENDPOINT_RE = re.compile(r"^`(GET|POST|PUT|DELETE|PATCH)`\s+(https?://\S+)")

INITIAL_SUBDOC = "Indian Trading & Data API"

# (page title that opens the next site, sub-doc name). Matched strictly in
# order so that a later repeat of the same title cannot trigger a switch early.
LANDMARKS: list[tuple[str, str]] = [
    ("Global Stocks", "Global Stocks API"),
    ("Historical Data", "Indian Trading & Data API"),
    ("Annexure", "Annexure"),
    ("Browser Based Login", "Authentication"),
    ("CAS Live Feed", "Developer Guides"),
    ("Python SDK", "Python SDK"),
    ("Statement", "Developer Guides"),
    ("Backtesting", "Agent Skills"),
    ("Compute Tiers", "Dhan Cloud"),
    ("First Query", "Dhan MCP"),
]


@dataclass
class Section:
    section_id: str
    sub_doc: str
    page_title: str
    title: str
    heading_path: list[str]
    method: str | None
    endpoint: str | None
    body: str
    line_start: int
    extra: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        if self.endpoint:
            return f"{self.method} {self.endpoint}"
        return " > ".join([self.sub_doc, *self.heading_path])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation"] = self.citation
        return d


def _is_fence(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("```") or s.startswith("~~~")


def parse_markdown(text: str) -> list[Section]:
    lines = text.splitlines()
    sections: list[Section] = []

    in_fence = False
    landmark_idx = 0
    sub_doc = INITIAL_SUBDOC
    page_title = ""
    prev_nonblank = ""

    cur: dict | None = None

    def flush():
        if cur is None:
            return
        body = "\n".join(cur["lines"]).strip()
        method = endpoint = None
        for ln in body.splitlines()[:6]:
            m = ENDPOINT_RE.match(ln.strip())
            if m:
                method, endpoint = m.group(1), m.group(2)
                break
        if not body:
            return
        path = [cur["page_title"]] if cur["title"] == cur["page_title"] else [cur["page_title"], cur["title"]]
        sections.append(
            Section(
                section_id=f"s{len(sections):04d}",
                sub_doc=cur["sub_doc"],
                page_title=cur["page_title"],
                title=cur["title"],
                heading_path=path,
                method=method,
                endpoint=endpoint,
                body=body,
                line_start=cur["line_start"],
            )
        )

    for i, line in enumerate(lines, start=1):
        if _is_fence(line):
            in_fence = not in_fence
        heading = not in_fence and line.startswith("## ")

        if heading:
            title = line[3:].strip()
            new_page = prev_nonblank == "---" or cur is None
            if new_page:
                if landmark_idx < len(LANDMARKS) and title == LANDMARKS[landmark_idx][0]:
                    sub_doc = LANDMARKS[landmark_idx][1]
                    landmark_idx += 1
                page_title = title
            flush()
            cur = {
                "title": title,
                "page_title": page_title,
                "sub_doc": sub_doc,
                "line_start": i,
                "lines": [],
            }
        elif cur is not None and not (not in_fence and line.strip() == "---"):
            cur["lines"].append(line)

        if line.strip():
            prev_nonblank = "<code>" if (in_fence or _is_fence(line)) else line.strip()
    flush()

    # The URL is a stronger signal than position: any page whose endpoint is
    # under /globalstocks/ belongs to the Global Stocks API.
    for s in sections:
        if s.endpoint and "/globalstocks/" in s.endpoint:
            s.sub_doc = "Global Stocks API"
    return sections


def load_sections(path: Path) -> list[Section]:
    return parse_markdown(path.read_text(encoding="utf-8"))
