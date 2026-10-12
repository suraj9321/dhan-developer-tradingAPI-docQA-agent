import re

from docsqa.ingest.chunk import bm25_tokens, chunk_section, n_tokens
from docsqa.ingest.parse import parse_markdown
from docsqa.retrieve import rrf_merge

DOC = """# Export title

---

## Place Order

`POST` https://api.dhan.co/v2/orders

Places an order.

```bash
# Not a heading
## Also not a heading
curl -X POST https://api.dhan.co/v2/orders
```

### Request Body Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| productType | enum | Yes | Values: `CNC`, `INTRADAY` |

---

## Global Stocks

Overview of international trading.

---

## Cancel Order

`DELETE` https://api.dhan.co/v2/globalstocks/orders/{order-id}

Cancels a global order.

---

## Historical Data

Back in the Indian API.
"""


def test_headings_inside_code_fences_are_ignored():
    sections = parse_markdown(DOC)
    titles = [s.title for s in sections]
    assert "Not a heading" not in titles
    assert "Also not a heading" not in titles
    assert "## Also not a heading" in sections[0].body


def test_endpoint_and_subdoc_assignment():
    by_title = {s.title: s for s in parse_markdown(DOC)}
    assert by_title["Place Order"].endpoint == "https://api.dhan.co/v2/orders"
    assert by_title["Place Order"].sub_doc == "Indian Trading & Data API"
    assert by_title["Cancel Order"].sub_doc == "Global Stocks API"
    assert by_title["Historical Data"].sub_doc == "Indian Trading & Data API"
    assert by_title["Cancel Order"].citation.startswith("DELETE https://api.dhan.co/v2/globalstocks/")


def test_chunks_carry_prefix_and_respect_budget():
    section = parse_markdown(DOC)[0]
    whole = chunk_section(section, 500)
    assert len(whole) == 1
    assert whole[0].text.startswith("Indian Trading & Data API > Place Order\nEndpoint: POST")

    # 90 tokens is just above the smallest indivisible unit here (prefix +
    # heading + table header + one row), forcing a split everywhere else.
    small = chunk_section(section, 90)
    assert len(small) > 1
    assert all(c.n_tokens <= 90 for c in small)
    assert not any(c.text.rstrip().endswith("### Request Body Parameters") for c in small)
    table_chunks = [c for c in small if "| productType" in c.text]
    assert table_chunks and all("|-----------" in c.text for c in table_chunks), "table header must travel with rows"


def test_large_tables_split_with_repeated_header():
    rows = "\n".join(f"| p{i} | string | No | parameter number {i} with a description |" for i in range(80))
    body = "| Parameter | Type | Required | Description |\n|---|---|---|---|\n" + rows
    md = f"## Big\n\n{body}\n"
    chunks = chunk_section(parse_markdown(md)[0], 200)
    assert len(chunks) > 1
    for c in chunks:
        assert "| Parameter | Type |" in c.text
        assert c.n_tokens <= 200
    assert sum(len(re.findall(r"^\| p\d+ \|", c.text, re.M)) for c in chunks) == 80


def test_bm25_tokens_keep_identifiers_and_parts():
    toks = bm25_tokens("DH-904 productType NSE_FNO")
    assert {"dh-904", "producttype", "product", "type", "nse_fno", "fno"} <= set(toks)


def test_rrf_rewards_agreement():
    merged = dict(rrf_merge([["a", "b", "c"], ["b", "a", "d"]]))
    assert merged["a"] == merged["b"] > merged["c"] > 0
    assert n_tokens("hello world") == 2
