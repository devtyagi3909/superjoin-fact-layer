import tempfile
import time
import json
from pathlib import Path
import pytest

from fastapi.testclient import TestClient

from api.main import app, fact_layer
from core.parser import Fact, FactEvidence, FactLayer, ProviderQuotaError, classify_provider_error


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
        handle.write("alpha beta Revenue increased to 100 gamma delta epsilon zeta eta theta")
        handle.flush()
        result = layer.process_document(handle.name, "sample.txt")
    assert result["status"] == "success"
    assert result["facts_extracted"] >= 1
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


def test_upload_returns_job_and_reaches_explicit_failure_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    original_client = fact_layer._provided_client
    fact_layer._provided_client = None
    try:
        with TestClient(app) as client:
            response = client.post("/upload", files={"file": ("sample.txt", b"Revenue increased")})
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            for _ in range(20):
                status = client.get(f"/upload/{job_id}").json()
                if status["status"] not in {"queued", "processing"}:
                    break
                time.sleep(0.01)
            assert status["status"] == "failed"
            assert status["result"]["errors"]
    finally:
        fact_layer._provided_client = original_client


def test_multiple_uploads_are_accepted_and_each_gets_a_job(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    original_client = fact_layer._provided_client
    fact_layer._provided_client = None
    try:
        with TestClient(app) as client:
            response = client.post(
                "/uploads",
                files=[
                    ("files", ("first.txt", b"Revenue increased")),
                    ("files", ("second.txt", b"gamma delta")),
                ],
            )
            assert response.status_code == 202
            jobs = response.json()["jobs"]
            assert len(jobs) == 2
            assert {job["filename"] for job in jobs} == {"first.txt", "second.txt"}
    finally:
        fact_layer._provided_client = original_client


def test_upload_rejects_unsupported_extensions():
    with TestClient(app) as client:
        response = client.post("/upload", files={"file": ("notes.docx", b"not supported")})
    assert response.status_code == 415


def test_health_exposes_provider_configuration_without_secret():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert {"status", "provider", "model", "base_url", "configured", "sdk_available", "error"} <= set(payload)
    assert "test-key" not in str(payload)


def test_provider_selection_supports_openai_compatible_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    layer = FactLayer(client=None)
    status = layer.provider_status()
    assert status["provider"] == "openai_compatible"
    assert status["model"] == "test-model"
    assert status["error"] in (None, "openai SDK is not installed. Install requirements.txt.")


def test_provider_budget_defaults_are_sequential_and_smoke_test_is_safe(monkeypatch):
    for name in (
        "LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL",
        "LLM_MAX_WORKERS", "LLM_MAX_PROVIDER_CALLS",
    ):
        monkeypatch.delenv(name, raising=False)
    layer = FactLayer(client=None)
    assert layer.max_workers == 1
    assert layer.max_chunks == 8
    result = layer.provider_smoke_test()
    assert result["status"] == "not_ready"
    assert "test-key" not in json.dumps(result).lower()


def test_provider_smoke_endpoint_never_returns_credentials():
    with TestClient(app) as client:
        response = client.post("/provider-smoke-test")
    assert response.status_code == 200
    assert "LLM_API_KEY" not in response.text


def test_empty_document_is_reported_as_failed_instead_of_success():
    layer = FactLayer(client=None)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8") as handle:
        handle.flush()
        result = layer.process_document(handle.name, "empty.txt")
    assert result["status"] == "failed"
    assert result["facts_extracted"] == 0
    assert result["errors"] == ["No extractable text found"]


def test_relationship_reasoning_preserves_four_case_contract():
    class ReasoningModels:
        def generate_content(self, model, contents, config):
            return type(
                "Response",
                (),
                {
                    "text": json.dumps(
                        {
                            "corroborations": [],
                            "contradictions": [
                                {
                                    "type": "genuine_contradiction",
                                    "fact_1": {"text": "A", "value": "1", "evidence": []},
                                    "fact_2": {"text": "A", "value": "2", "evidence": []},
                                    "explanation": "Same scope, different values.",
                                },
                                {
                                    "type": "explained_by_context",
                                    "fact_1": {"text": "A", "value": "1", "evidence": []},
                                    "fact_2": {"text": "A", "value": "2", "evidence": []},
                                    "explanation": "Different reporting periods.",
                                },
                            ],
                            "failures": [],
                        }
                    )
                },
            )()

    layer = FactLayer(client=type("Client", (), {"models": ReasoningModels()})())
    layer.facts = [
        Fact(
            id="1",
            text="Revenue",
            value="1",
            evidence=[FactEvidence(document_name="a.txt", page=1, excerpt="Revenue 1")],
        ),
        Fact(
            id="2",
            text="Revenue",
            value="2",
            evidence=[FactEvidence(document_name="b.txt", page=1, excerpt="Revenue 2")],
        ),
    ]
    result = layer.run_reasoning()
    assert {item["type"] for item in result["contradictions"]} == {
        "genuine_contradiction",
        "explained_by_context",
    }


def test_quota_errors_are_structured_and_retry_after_is_concise():
    error = classify_provider_error(
        RuntimeError("429 RESOURCE_EXHAUSTED: RetryInfo retryDelay: 17s")
    )
    assert error == {
        "code": "provider_quota",
        "message": "Gemini quota is temporarily exhausted. Retry after about 17 seconds.",
        "retry_after_seconds": 17,
        "retryable": True,
    }


def test_provider_errors_are_actionable_without_provider_payloads():
    assert classify_provider_error(RuntimeError("401 invalid api key"), "openai_compatible")["code"] == "provider_auth"
    assert classify_provider_error(RuntimeError("404 model not found"), "openai_compatible")["code"] == "provider_model"
    assert classify_provider_error(RuntimeError("404 endpoint not found"), "openai_compatible")["code"] == "provider_endpoint"
    assert classify_provider_error(RuntimeError("400 response_format json_object unsupported"), "openai_compatible")["code"] == "provider_structured_output"
    assert classify_provider_error(TimeoutError("request timed out"), "openai_compatible")["code"] == "provider_timeout"


def test_request_too_large_errors_are_classified_without_payloads():
    error = type("HTTPError", (RuntimeError,), {"status_code": 413})(
        "413 Request Entity Too Large secret request body"
    )
    result = classify_provider_error(error, "openai_compatible")
    assert result["code"] == "provider_request_too_large"
    assert "No facts were fabricated" in result["message"]
    assert "secret request body" not in result["message"]


def test_request_too_large_call_is_not_retried_and_keeps_call_index():
    class TooLargeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **_kwargs):
            self.calls += 1
            raise type("HTTPError", (RuntimeError,), {"status_code": 413})(
                "413 request too large internal payload"
            )

    models = TooLargeModels()
    layer = FactLayer(
        client=type("Client", (), {"models": models})(),
        max_workers=1,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8") as handle:
        handle.write("short source is here")
        handle.flush()
        result = layer.process_document(handle.name, "sample.txt")
    assert models.calls == 1
    assert result["status"] == "failed"
    assert result["provider_calls_total"] == 1
    assert result["provider_calls_succeeded"] == 0
    assert result["failed_provider_calls"] == [1]
    assert result["errors"][0]["code"] == "provider_request_too_large"


def test_openai_compatible_retries_without_unsupported_structured_output(monkeypatch):
    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if "response_format" in kwargs:
                raise RuntimeError("400 response_format json_object is not supported")
            return type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": '{"facts": []}'})()})()]})()

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    layer = FactLayer(client=client)
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    assert layer._extract_chunk("source text") == {"facts": []}
    assert len(completions.calls) == 2
    assert "response_format" not in completions.calls[1]


def test_json_parser_accepts_fenced_json_but_not_non_objects():
    assert FactLayer._parse_json("```json\n{\"facts\": []}\n```") == {"facts": []}
    assert FactLayer._parse_json("Here is the result:\n{\"facts\": []}\nDone.") == {"facts": []}
    with pytest.raises((json.JSONDecodeError, ValueError)):
        FactLayer._parse_json("not JSON")
    with pytest.raises(ValueError, match="invalid JSON"):
        FactLayer._parse_json('{"facts": []} {"facts": []}')


def test_invalid_chunk_cap_is_corrected_and_28_source_chunks_use_at_most_8_calls(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_CHUNKS", "-1")

    class EmptyModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **_kwargs):
            self.calls += 1
            return type("Response", (), {"text": '{"facts": []}'})()

    layer = FactLayer(
        client=type("Client", (), {"models": EmptyModels()})(),
        max_workers=1,
    )
    monkeypatch.setattr(layer, "iter_text_chunks", lambda *_args: iter(["chunk"] * 28))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8") as handle:
        handle.write("source")
        handle.flush()
        result = layer.process_document(handle.name, "sample.txt")
    assert layer.max_chunks == 8
    assert layer.config_warnings
    assert result["source_chunks_total"] == 28
    assert result["chunks_total"] <= 8
    assert layer.client.models.calls == result["chunks_total"]


def test_provider_budget_groups_28_source_chunks_without_oversized_prompts(monkeypatch):
    monkeypatch.setenv("LLM_MAX_INPUT_CHARS", "2000")

    class RecordingModels:
        def __init__(self):
            self.prompts = []

        def generate_content(self, **kwargs):
            self.prompts.append(kwargs["contents"])
            return type("Response", (), {"text": '{"facts": []}'})()

    models = RecordingModels()
    layer = FactLayer(
        client=type("Client", (), {"models": models})(),
        chunk_size=1000,
        max_workers=1,
    )
    source_chunks = [f"\n--- Page {page} ---\n" + ("evidence is " * 20) for page in range(1, 29)]
    monkeypatch.setattr(layer, "iter_text_chunks", lambda *_args: iter(source_chunks))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", encoding="utf-8") as handle:
        handle.write("source")
        handle.flush()
        result = layer.process_document(handle.name, "sample.pdf")
    assert result["source_chunks_total"] == 28
    assert result["provider_calls_total"] <= layer.max_chunks
    assert len(models.prompts) == result["provider_calls_total"]
    assert all(len(prompt) <= layer.max_input_chars for prompt in models.prompts)
    assert all(f"--- Page {page} ---" in "\n".join(models.prompts) for page in range(1, 29))


def test_provider_diagnostics_exposes_request_budget_and_call_limits(monkeypatch):
    monkeypatch.setenv("LLM_MAX_INPUT_CHARS", "18000")
    with TestClient(app) as client:
        response = client.get("/provider-diagnostics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["effective_max_input_chars"] == fact_layer.max_input_chars
    assert payload["max_chunks"] == fact_layer.max_chunks
    assert payload["max_workers"] == fact_layer.max_workers
    assert payload["model"] == fact_layer.model


def test_quota_retry_honors_retry_after_without_raw_provider_blob(monkeypatch):
    class RateLimitedModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED retryDelay: 1s secret provider payload")
            return type("Response", (), {"text": '{"facts": []}'})()

    client = type("Client", (), {"models": RateLimitedModels()})()
    layer = FactLayer(client=client, max_workers=1)
    monkeypatch.setattr("core.parser.time.sleep", lambda seconds: None)
    result = layer._extract_chunk("text")
    assert result == {"facts": []}
    assert client.models.calls == 2


def test_demo_path_is_deterministic_and_marks_simulated_results():
    layer = FactLayer(client=None)
    result = layer.load_demo()
    assert result["demo"] is True
    assert result["corroborations"]
    assert {item["type"] for item in result["contradictions"]} == {
        "genuine_contradiction",
        "explained_by_context",
    }
    assert result["failures"][0]["type"] == "extraction_failure"


def test_graph_data_contains_grounded_nodes_colored_relationships_and_failure_node():
    layer = FactLayer(client=None)
    reasoning = layer.load_demo()
    graph = layer.build_graph(reasoning)
    assert len(graph["nodes"]) == 6
    assert {edge["type"] for edge in graph["edges"]} == {
        "corroboration",
        "genuine_contradiction",
        "explained_by_context",
        "extraction_failure",
    }
    assert graph["legend"]["genuine_contradiction"]["color"] == "#dc2626"
    assert any(node["kind"] == "failure" for node in graph["nodes"])


def test_source_preview_preserves_pdf_and_rejects_path_traversal():
    with TestClient(app) as client:
        response = client.post(
            "/upload", files={"file": ("grounding.txt", b"Evidence on page one")}
        )
        assert response.status_code == 202
        preview = client.get("/source-preview", params={"document_name": "grounding.txt"})
        assert preview.status_code == 200
        assert preview.json()["text"] == "Evidence on page one"
        assert client.get(
            "/source-preview", params={"document_name": "../grounding.txt"}
        ).status_code == 404


def test_api_failure_shape_does_not_expose_provider_blob(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED RetryInfo retryDelay: 12s internal token dump")

    monkeypatch.setattr(fact_layer, "process_document", fail)
    with TestClient(app) as client:
        response = client.post("/upload", files={"file": ("sample.txt", b"Revenue increased")})
        job_id = response.json()["job_id"]
        for _ in range(20):
            status = client.get(f"/upload/{job_id}").json()
            if status["status"] not in {"queued", "processing"}:
                break
            time.sleep(0.01)
    assert status["error"]["code"] == "provider_quota"
    assert status["error"]["retry_after_seconds"] == 12
    assert "internal token dump" not in str(status)


def test_failed_chunk_reports_safe_provider_context_and_progress():
    class FailingModels:
        def generate_content(self, **_kwargs):
            raise RuntimeError("401 invalid api key")

    layer = FactLayer(
        client=type("Client", (), {"models": FailingModels()})(),
        chunk_size=10,
        chunk_overlap=0,
        max_workers=1,
    )
    progress = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8") as handle:
        handle.write("one two three four five six")
        handle.flush()
        result = layer.process_document(
            handle.name,
            "sample.txt",
            progress_callback=lambda done, total: progress.append((done, total)),
        )
    assert result["status"] == "failed"
    assert progress == [(index, len(progress)) for index in range(1, len(progress) + 1)]
    assert result["errors"][0] == {
        "code": "provider_auth",
        "message": "Gemini authentication failed. Check the configured API key.",
        "retry_after_seconds": None,
        "retryable": False,
        "provider": "gemini",
        "model": layer.model,
        "status": 401,
        "chunk": 1,
        "chunks_total": len(progress),
        "provider_call": 1,
        "provider_calls_total": len(progress),
    }


def test_ui_css_forces_light_main_and_dark_sidebar_text():
    css = Path("ui/app.py").read_text(encoding="utf-8")
    assert "color-scheme: light" in css
    assert "[data-testid=\"stMain\"]" in css
    assert "color: #111827 !important" in css
    assert "[data-testid=\"stSidebar\"] { background: #101827" in css
    assert "prefers-color-scheme" not in css
