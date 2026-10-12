from docsqa.ingest.chunk import Chunk
from docsqa.llm import parse_json
from docsqa.rag import build_citations, cited_numbers, is_refusal


def _chunk(i):
    return Chunk(f"s{i:04d}-00", f"s{i:04d}", "Global Stocks API", ["Cancel Order"], "DELETE", "u", f"cite{i}", "t", 1)


def test_citations_map_to_retrieved_chunks_and_ignore_out_of_range():
    chunks = [_chunk(0), _chunk(1)]
    text = "Use DELETE [2]. Also see [1][2] and a hallucinated [7]."
    assert cited_numbers(text) == [1, 2, 7]
    cites = build_citations(text, chunks)
    assert [c.n for c in cites] == [1, 2]
    assert cites[1].chunk_id == "s0001-00"


def test_refusal_detection():
    assert is_refusal("I don't know based on the DhanHQ documentation.")
    assert not is_refusal("The endpoint is /v2/orders [1].")


def test_parse_json_tolerates_fences_and_prose():
    assert parse_json('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('result: [1, 2] done') == [1, 2]
