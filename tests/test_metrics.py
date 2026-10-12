import math

from eval.metrics import cohen_kappa, dedupe, hit_at_k, mrr, percentile, recall_at_k
from eval.split import normalize_reference


def test_retrieval_metrics_are_section_level_and_deduplicated():
    retrieved = ["s1", "s1", "s2", "s3", "s4", "s5", "s6"]
    assert dedupe(retrieved) == ["s1", "s2", "s3", "s4", "s5", "s6"]
    assert recall_at_k(retrieved, {"s5"}, 5) == 1.0  # duplicates don't consume top-k slots
    assert recall_at_k(retrieved, {"s6"}, 5) == 0.0
    assert recall_at_k(retrieved, {"s1", "s9"}, 5) == 0.5
    assert hit_at_k(retrieved, {"s9", "s3"}, 5) == 1.0
    assert mrr(retrieved, {"s3"}) == 1 / 3


def test_percentile_and_kappa():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert math.isnan(percentile([], 50))
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) < 0


def test_reference_slug_normalization():
    mapping = {"/api/v2/statements/get-ledger-report": "/ledger"}
    assert normalize_reference("Use GET /api/v2/statements/get-ledger-report.", mapping) == "Use GET /ledger."
    assert normalize_reference("See /api/v2/unknown/page", mapping) == "See /api/v2/unknown/page"
