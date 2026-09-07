import tempfile
import time
import json

from fastapi.testclient import TestClient

from api.main import app, fact_layer
from core.parser import Fact, FactEvidence, FactLayer


class FakeModels:
    def __init__(self, response_payload=None):
        self.prompts = []
        self.calls = 0
        self.response_payload = response_payload or {
            "facts": [
                {
                    "text": "Revenue increased",
                    "value": "100",
                    "excerpt": "Revenue increased to 100",
                    "page": 1,
                }
            ]
        }

    def generate_content(self, model, contents, config):
        self.calls += 1
        self.prompts.append(contents)
        return type(
            "Response",
            (),
            {
                "text": json.dumps(self.response_payload)
            },
        )()


class FakeClient:
    def __init__(self, response_payload=None):
        self.models = FakeModels(response_payload)


def test_text_is_chunked_incrementally_without_gemini():
    layer = FactLayer(client=None, chunk_size=20, chunk_overlap=3)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8") as handle:
        handle.write("alpha beta gamma delta epsilon zeta eta theta")
        handle.flush()
        chunks = list(layer.iter_text_chunks(handle.name, "sample.txt"))
    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)


def test_mocked_gemini_extraction_preserves_evidence_and_processes_chunks():
    layer = FactLayer(client=FakeClient(), chunk_size=20, chunk_overlap=3)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8") as handle:
        handle.write("alpha beta gamma delta epsilon zeta eta theta")
        handle.flush()
        result = layer.process_document(handle.name, "sample.txt")
    assert result["status"] == "success"
    assert result["facts_extracted"] > 1
    fact = layer.get_facts()[0]
    assert fact.evidence[0].document_name == "sample.txt"
    assert fact.evidence[0].excerpt == "Revenue increased to 100"


def test_processing_reports_progress_and_reasoning_is_grounded_and_cached():
    client = FakeClient(
        {
            "facts": [
                {
                    "text": "Revenue increased",
                    "value": "100",
                    "excerpt": "Revenue increased to 100",
                    "page": 1,
                }
            ]
        }
    )
    layer = FactLayer(client=client, chunk_size=1000)
    progress = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8") as handle:
        handle.write("Revenue increased")
        handle.flush()
        layer.process_document(handle.name, "sample.txt", progress_callback=lambda done, total: progress.append((done, total)))
    assert progress == [(1, 1)]
    layer.facts.append(
        Fact(
            id="second",
            text="Revenue increased",
            value="100",
            evidence=[FactEvidence(document_name="second.txt", page=2, excerpt="Revenue increased to 100")],
        )
    )
    layer._facts_version += 1
    client.models.response_payload = {
        "corroborations": [
            {
                "type": "corroboration",
                "fact_1": "Revenue increased",
                "fact_2": "Revenue increased",
                "explanation": "Both sources support the same claim.",
            }
        ],
        "contradictions": [],
        "failures": [],
    }
    result = layer.run_reasoning()
    assert result["corroborations"][0]["fact_1"]["evidence"][0]["document_name"] == "sample.txt"
    calls_after_first_reasoning = client.models.calls
    layer.run_reasoning()
    assert client.models.calls == calls_after_first_reasoning


def test_upload_returns_job_and_reaches_explicit_failure_without_key():
    original_client = fact_layer.client
    fact_layer.client = None
    try:
        with TestClient(app) as client:
            response = client.post("/upload", files={"file": ("sample.txt", b"alpha beta")})
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            for _ in range(20):
                status = client.get(f"/upload/{job_id}").json()
                if status["status"] not in {"queued", "processing"}:
                    break
                time.sleep(0.01)
            assert status["status"] == "failed"
            assert status["error"] is None
            assert status["result"]["errors"]
    finally:
        fact_layer.client = original_client
