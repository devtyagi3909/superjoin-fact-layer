import json
import tempfile

from core.parser import Fact, FactEvidence, FactLayer, _evidence_failure


def test_evidence_rejects_excerpt_page_and_numeric_mismatches():
    chunk = "\n--- Page 2 ---\nThe service handled 12 requests."
    base = {"text": "requests", "value": "12", "page": 2}
    assert "missing" in _evidence_failure({**base, "excerpt": ""}, chunk)
    assert "not present" in _evidence_failure({**base, "excerpt": "handled 13", "subject": "service"}, chunk)
    assert "page" in _evidence_failure({**base, "excerpt": "handled 12", "page": 1, "subject": "service"}, chunk)
    assert "Numeric" in _evidence_failure({**base, "excerpt": "handled 12", "value": "13", "subject": "service"}, chunk)


def _fact(identifier, document, value, **metadata):
    return Fact(
        id=identifier,
        text="Service handled requests",
        value=value,
        evidence=[FactEvidence(document_name=document, page=1, excerpt=f"handled {value} requests")],
        subject="service",
        predicate="handled requests",
        normalized_value=metadata.pop("normalized_value", value),
        normalized_unit=metadata.pop("normalized_unit", "count"),
        **metadata,
    )


def test_relationship_gate_correlates_normalized_equivalent_values():
    layer = FactLayer(client=None)
    left = _fact("a", "a.txt", "1,000", normalized_value="1000")
    right = _fact("b", "b.txt", "1000", normalized_value="1000")
    result = layer._relationship_gate(left, right)
    assert result["type"] == "corroboration"
    assert result["metadata"]["gate"] == "exact_normalized_match"


def test_relationship_gate_explains_different_periods_and_scopes():
    layer = FactLayer(client=None)
    left = _fact("a", "a.txt", "10", time_start="2024-01-01", time_end="2024-01-31")
    right = _fact("b", "b.txt", "10", time_start="2024-02-01", time_end="2024-02-29")
    result = layer._relationship_gate(left, right)
    assert result["type"] == "explained_by_context"
    assert result["metadata"]["gate"] == "time_context"


def test_schema_fields_are_optional_for_legacy_facts():
    fact = Fact(
        id="legacy",
        text="A",
        value="1",
        evidence=[FactEvidence(document_name="a.txt", page=1, excerpt="A 1")],
    )
    assert fact.subject is None
    assert fact.model_dump()["evidence"][0]["excerpt"] == "A 1"
