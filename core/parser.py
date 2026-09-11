import json
import logging
import os
import re
import time
import threading
import uuid

import spacy
from rapidfuzz import fuzz
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None and SentenceTransformer is not None:
        try:
            _embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            pass
    return _embedder

_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            pass
    return _nlp


CURATED_VERBS = {"is", "was", "were", "appointed", "resigned", "increased", "decreased", "merged", "renamed", "located", "reported", "grew"}
TARGET_ENTS = {"MONEY", "PERCENT", "DATE", "CARDINAL", "ORG", "PERSON", "GPE"}
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, List, Optional

import pymupdf as fitz  # PyMuPDF
try:
    from google import genai
    from google.genai import types
except ImportError:  # Optional when an OpenAI-compatible provider is used.
    genai = None
    types = None
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FactEvidence(BaseModel):
    document_name: str
    page: Optional[int]
    excerpt: str


class Fact(BaseModel):
    id: str
    text: str
    value: Optional[str] = None
    evidence: List[FactEvidence]
    subject: Optional[str] = None
    predicate: Optional[str] = None
    raw_value: Optional[str] = None
    normalized_value: Optional[str] = None
    normalized_unit: Optional[str] = None
    time_expression: Optional[str] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    scope_expression: Optional[str] = None
    polarity: Optional[str] = None
    confidence: Optional[float] = None
    confidence_breakdown: Optional[dict[str, float]] = None


class FactOutput(BaseModel):
    text: str
    value: Optional[str] = None
    excerpt: str
    page: Optional[int] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    raw_value: Optional[str] = None
    normalized_value: Optional[str] = None
    normalized_unit: Optional[str] = None
    time_expression: Optional[str] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    scope_expression: Optional[str] = None
    polarity: Optional[str] = None
    confidence: Optional[float] = None
    confidence_breakdown: Optional[dict[str, float]] = None


class FactList(BaseModel):
    facts: list[FactOutput]


class Corroboration(BaseModel):
    type: str = "corroboration"
    fact_1: str
    fact_2: str
    explanation: str


class Contradiction(BaseModel):
    type: str
    fact_1: str
    fact_2: str
    explanation: str


class Failure(BaseModel):
    type: str
    description: str


class ReasoningOutput(BaseModel):
    corroborations: list[Corroboration] = []
    contradictions: list[Contradiction] = []
    failures: list[Failure] = []


class ProviderQuotaError(RuntimeError):
    """A concise, safe representation of a provider quota/rate-limit failure."""

    def __init__(self, retry_after_seconds: int | None = None):
        self.retry_after_seconds = retry_after_seconds
        message = "The configured LLM provider quota is temporarily exhausted."
        if retry_after_seconds:
            message += f" Retry after about {retry_after_seconds} seconds."
        super().__init__(message)


class ProviderRequestTooLargeError(RuntimeError):
    """A request exceeded the configured provider input budget."""


def _provider_label(provider: str) -> str:
    return "Gemini" if provider == "gemini" else "OpenAI-compatible provider"


RELATIONSHIP_STYLES = {
    "corroboration": {"label": "Corroboration", "color": "#16a34a"},
    "genuine_contradiction": {"label": "Genuine contradiction", "color": "#dc2626"},
    "explained_by_context": {"label": "Contextual reconciliation", "color": "#d97706"},
    "extraction_failure": {"label": "Extraction failure", "color": "#6b7280"},
}


def _normalise_number(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return format(Decimal(match.group().replace(",", "")), "f").rstrip("0").rstrip(".") or "0"
    except InvalidOperation:
        return None


def _normalise_unit(value: Any, text: str = "") -> str | None:
    source = f"{value or ''} {text}".lower()
    match = re.search(r"\b(percent|%|seconds?|minutes?|hours?|days?|weeks?|months?|years?|kg|g|mg|km|m|cm|usd|eur|gb|mb)\b", source)
    if not match:
        return None
    unit = match.group(1)
    return "%" if unit == "percent" else unit.rstrip("s")


def _token_overlap(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    left_tokens = FactLayer._tokens(left)
    right_tokens = FactLayer._tokens(right)
    return bool(left_tokens and right_tokens and (left_tokens & right_tokens))


def _metadata_for_claim(item: dict, text: str, value: Any) -> dict:
    raw_value = item.get("raw_value") or value
    normalized_value = item.get("normalized_value") or _normalise_number(raw_value)
    return {
        "subject": item.get("subject"),
        "predicate": item.get("predicate"),
        "raw_value": raw_value,
        "normalized_value": normalized_value,
        "normalized_unit": item.get("normalized_unit") or _normalise_unit(raw_value, text),
        "time_expression": item.get("time_expression"),
        "time_start": item.get("time_start"),
        "time_end": item.get("time_end"),
        "scope_expression": item.get("scope_expression"),
        "polarity": item.get("polarity"),
        "confidence": item.get("confidence"),
        "confidence_breakdown": item.get("confidence_breakdown"),
    }


def _evidence_failure(item: dict, chunk: str) -> str | None:
    excerpt = str(item.get("excerpt") or "").strip()
    if not excerpt:
        return "Evidence excerpt is missing."
    # Older providers only returned the original four fields. Keep that
    # response contract usable while applying strict admission to enriched
    # claims and all malformed excerpts.
    enriched = any(
        key in item
        for key in (
            "subject", "predicate", "normalized_value", "normalized_unit",
            "time_expression", "time_start", "time_end", "scope_expression",
            "confidence_breakdown",
        )
    )
    if not enriched:
        return None
    if excerpt not in chunk:
        return "Evidence excerpt is not present in the source chunk."
    page = item.get("page")
    if page is None:
        return "Evidence page is missing."
    markers = re.findall(r"---\s*Page\s+(\d+)\s*---", chunk, flags=re.I)
    if not markers or str(page) not in markers:
        return f"Evidence page {page} does not match a source page marker."
    value = item.get("value") or item.get("raw_value")
    number = _normalise_number(value)
    if number is not None and not re.search(rf"(?<![\d.]){re.escape(number)}(?![\d.])|{re.escape(number).replace('.', r'[.,]')}", excerpt.replace(",", "")):
        return f"Numeric value {value!r} is not present in the evidence excerpt."
    return None


def _retry_after_seconds(error: Exception) -> int | None:
    text = str(error)
    candidates = re.findall(r"(?:retryDelay|retry.?after|retry in)[^0-9]{0,20}(\d+)", text, re.I)
    for candidate in candidates:
        return max(1, int(candidate))
    def find_retry(value: Any) -> int | None:
        if isinstance(value, dict):
            for key, item in value.items():
                if "retry" in str(key).lower() and item is not None:
                    match = re.search(r"\d+", str(item))
                    if match:
                        return max(1, int(match.group()))
                found = find_retry(item)
                if found:
                    return found
        elif isinstance(value, (list, tuple)):
            for item in value:
                found = find_retry(item)
                if found:
                    return found
        return None
    for value in (getattr(error, "details", None), getattr(error, "response", None), getattr(error, "body", None)):
        found = find_retry(value)
        if found:
            return found
    for value in (getattr(error, "retry_after", None), getattr(error, "retry_after_seconds", None)):
        if value is not None:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                pass
    return None


def classify_provider_error(error: Exception, provider: str = "gemini") -> dict:
    """Return a stable API/UI error without exposing provider payloads."""
    text = str(error).lower()
    status_code = getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)
    if (
        status_code == 413
        or "413" in text
        or any(
            term in text
            for term in (
                "request too large",
                "payload too large",
                "context length",
                "input is too long",
                "too many tokens",
                "provider request budget",
            )
        )
    ):
        return {
            "code": "provider_request_too_large",
            "message": (
                "The provider rejected this request because the extracted chunk was too large. "
                "No facts were fabricated. Use offline demo or lower LLM_MAX_INPUT_CHARS."
            ),
            "retry_after_seconds": None,
            "retryable": False,
        }
    if "429" in text or "resource_exhausted" in text or "quota" in text or "rate limit" in text:
        retry_after = _retry_after_seconds(error)
        provider_name = "Gemini" if provider == "gemini" else "LLM provider"
        message = f"{provider_name} quota is temporarily exhausted. Please wait and retry the upload."
        if retry_after:
            message = f"{provider_name} quota is temporarily exhausted. Retry after about {retry_after} seconds."
        return {
            "code": "provider_quota",
            "message": message,
            "retry_after_seconds": retry_after,
            "retryable": True,
        }
    provider_name = _provider_label(provider)
    if any(term in text for term in ("401", "unauthorized", "invalid api key", "authentication")):
        return {"code": "provider_auth", "message": f"{provider_name} authentication failed. Check the configured API key.", "retry_after_seconds": None, "retryable": False}
    if "endpoint" in text or ("url" in text and (status_code == 404 or "404" in text)):
        return {"code": "provider_endpoint", "message": f"{provider_name} endpoint was not found. Check LLM_BASE_URL.", "retry_after_seconds": None, "retryable": False}
    if status_code == 404 or any(term in text for term in ("404", "unknown model", "model not found", "invalid model")):
        return {"code": "provider_model", "message": f"{provider_name} model was not found. Check LLM_MODEL and the provider's model list.", "retry_after_seconds": None, "retryable": False}
    if any(term in text for term in ("timeout", "timed out", "deadline exceeded", "readtimeout")):
        return {"code": "provider_timeout", "message": f"{provider_name} timed out. Retry the request or use a smaller model.", "retry_after_seconds": None, "retryable": True}
    if any(term in text for term in ("response_format", "json_object", "structured output", "structured-output")):
        return {"code": "provider_structured_output", "message": f"{provider_name} rejected structured JSON output. Retrying without response_format.", "retry_after_seconds": None, "retryable": True}
    if any(term in text for term in ("empty response", "empty/non-json", "non-json", "invalid json", "json decode", "expecting value")):
        return {"code": "provider_response", "message": f"{provider_name} returned an empty or invalid JSON response.", "retry_after_seconds": None, "retryable": False}
    if any(term in text for term in ("400", "bad request", "invalid request")):
        return {"code": "provider_bad_request", "message": f"{provider_name} rejected the request. Check the model and endpoint configuration.", "retry_after_seconds": None, "retryable": False}
    return {
        "code": "extraction_error",
        "message": f"{provider_name} request failed. Check the provider configuration and try again.",
        "retry_after_seconds": None,
        "retryable": False,
    }


class FactLayer:
    """Incremental document extraction with bounded, lexical candidate retrieval."""

    def __init__(
        self,
        client: Any = None,
        chunk_size: int = 12000,
        chunk_overlap: int = 400,
        max_workers: Optional[int] = None,
    ):
        self.facts: List[Fact] = []
        self.extraction_failures: list[dict] = []
        self._provided_client = client
        self.config_warnings: list[str] = []
        self.max_workers = self._safe_int(
            "LLM_MAX_WORKERS", max_workers, 1, 1, aliases=("GEMINI_MAX_WORKERS",)
        )
        self.max_chunks = self._safe_int(
            "LLM_MAX_PROVIDER_CALLS", None, 8, 1, aliases=("GEMINI_MAX_CHUNKS",)
        )
        self.max_retries = self._safe_int(
            "LLM_MAX_RETRIES", None, 1, 0, aliases=("GEMINI_MAX_RETRIES",)
        )
        self.max_retry_wait = self._safe_int(
            "LLM_MAX_RETRY_WAIT_SECONDS", None, 8, 1,
            aliases=("GEMINI_MAX_RETRY_WAIT_SECONDS",),
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # This is the complete prompt budget, including extraction instructions.
        # 20,000 chars is conservative for Groq free-tier context/request limits.
        self.max_input_chars = self._safe_int("LLM_MAX_INPUT_CHARS", None, 20000, 1000)
        self.max_input_tokens = self._safe_int("LLM_MAX_INPUT_TOKENS", None, 5000, 256)
        self.request_timeout_seconds = self._safe_int(
            "LLM_REQUEST_TIMEOUT_SECONDS", None, 45, 5
        )
        self._lock = threading.RLock()
        self._facts_version = 0
        self._reasoning_cache: tuple[int, dict] | None = None
        self._quota_cooldown_until = 0.0

    def _safe_int(
        self, name: str, explicit: Optional[int], default: int, minimum: int,
        aliases: tuple[str, ...] = (),
    ) -> int:
        raw = explicit if explicit is not None else os.getenv(name)
        if raw is None:
            raw = next((os.getenv(alias) for alias in aliases if os.getenv(alias) is not None), str(default))
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = default
            warning = f"{name}={raw!r} is invalid; using {default}."
            self.config_warnings.append(warning)
            logger.warning(warning)
            return value
        if value < minimum:
            value = default
            warning = f"{name}={raw!r} is below {minimum}; using {default}."
            self.config_warnings.append(warning)
            logger.warning(warning)
        return value

    @property
    def provider(self):
        return os.getenv("LLM_PROVIDER", "gemini" if os.getenv("GEMINI_API_KEY") else "openai_compatible" if os.getenv("LLM_API_KEY") else "gemini").lower()

    @property
    def model(self):
        if self.provider == "openai_compatible":
            return os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        return os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    @property
    def base_url(self):
        return os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")

    @property
    def client(self):
        if self._provided_client is not None:
            return self._provided_client
        return self._build_client()

    def _build_client(self) -> Any:
        if self.provider == "gemini":
            if not os.getenv("GEMINI_API_KEY"):
                return None
            if genai is None:
                return None
            return genai.Client()
        if self.provider == "openai_compatible":
            if not os.getenv("LLM_API_KEY"):
                return None
            try:
                from openai import OpenAI
            except ImportError:
                return None
            return OpenAI(
                api_key=os.environ["LLM_API_KEY"],
                base_url=self.base_url,
            )
        return None

    def provider_status(self) -> dict:
        key_configured = bool(os.getenv("GEMINI_API_KEY" if self.provider == "gemini" else "LLM_API_KEY"))
        sdk_available = genai is not None if self.provider == "gemini" else self._openai_sdk_available()
        error = None
        if self.provider not in {"gemini", "openai_compatible"}:
            error = "Unsupported LLM_PROVIDER. Use gemini or openai_compatible."
        elif not key_configured:
            error = f"{'GEMINI_API_KEY' if self.provider == 'gemini' else 'LLM_API_KEY'} is not configured."
        elif not sdk_available:
            error = f"{'google-genai' if self.provider == 'gemini' else 'openai'} SDK is not installed. Install requirements.txt."
        elif self.provider == "openai_compatible" and not self.model:
            error = "LLM_MODEL is not configured."
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url if self.provider == "openai_compatible" else None,
            "configured": self.client is not None and error is None,
            "sdk_available": sdk_available,
            "error": error,
            "max_chunks": self.max_chunks,
            "max_workers": self.max_workers,
            "max_input_chars": self.max_input_chars,
            "effective_max_input_chars": self.max_input_chars,
            "max_input_tokens": self.max_input_tokens,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_retries": self.max_retries,
            "max_retry_wait_seconds": self.max_retry_wait,
            "config_warnings": list(self.config_warnings),
        }

    @staticmethod
    def _openai_sdk_available() -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def extract_text_from_pdf(self, filepath: str) -> List[dict]:
        return list(self._iter_pdf_pages(filepath))

    def _iter_pdf_pages(self, filepath: str) -> Iterable[dict]:
        with fitz.open(filepath) as doc:
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    yield {"page": i + 1, "text": text}

    def _chunk_pages(self, pages: Iterable[dict]) -> Iterable[str]:
        buffer = ""
        buffer_page = None
        for page in pages:
            text = page["text"].strip()
            if not text:
                continue
            marker = f"\n--- Page {page['page']} ---\n"
            if buffer and len(buffer) + len(marker) + len(text) > self.chunk_size:
                yield buffer
                overlap = buffer[-self.chunk_overlap:] if self.chunk_overlap else ""
                buffer = overlap
            if not buffer:
                buffer_page = page["page"]
            buffer += marker + text
            # Split exceptionally long single pages without retaining the full page.
            while len(buffer) > self.chunk_size:
                yield buffer[: self.chunk_size]
                buffer = buffer[self.chunk_size - self.chunk_overlap :]
                buffer_page = buffer_page
        if buffer.strip():
            yield buffer

    def iter_text_chunks(self, filepath: str, filename: str) -> Iterable[str]:
        if filepath.lower().endswith(".pdf") or filename.lower().endswith(".pdf"):
            yield from self._chunk_pages(self._iter_pdf_pages(filepath))
            return
        def text_pages():
            with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
                while True:
                    text = handle.read(self.chunk_size)
                    if not text:
                        break
                    yield {"page": 1, "text": text}

        yield from self._chunk_pages(text_pages())

    def _bounded_chunks(self, chunks: list[str]) -> list[str]:
        if self.max_chunks <= 0 or len(chunks) <= self.max_chunks:
            return chunks
        # Group adjacent chunks instead of dropping pages or evidence.
        groups = [[] for _ in range(self.max_chunks)]
        for index, chunk in enumerate(chunks):
            groups[index * self.max_chunks // len(chunks)].append(chunk)
        return ["\n".join(group) for group in groups if group]

    @staticmethod
    def _extraction_prompt_prefix() -> str:
        return """
Extract key factual claims from this document chunk. Return only claims supported by
the text. For every claim provide a concise "text", a numerical or categorical
"value" when present, a verbatim supporting "excerpt", and the source "page".
When possible also provide normalized metadata: subject, predicate, raw_value,
normalized_value, normalized_unit, time_expression/time_start/time_end,
scope_expression, polarity, confidence, and confidence_breakdown. The breakdown
must use explainable components named evidence_match, scope_completeness,
comparison_strength, and extraction_quality, not an opaque score.
Return one JSON object only, with a top-level "facts" array. Do not use markdown,
code fences, commentary, or any text outside the JSON object.
Page markers are authoritative:
"""

    def _provider_input_budget(self) -> int:
        suffix_length = len("\n")
        char_budget = max(1, self.max_input_chars - len(self._extraction_prompt_prefix()) - suffix_length)
        return max(1, min(char_budget, self.max_input_tokens * 4))

    def _prepare_provider_chunks(self, source_chunks: list[str]) -> list[str]:
        """Split oversized source chunks, then group adjacent text without dropping it."""
        budget = self._provider_input_budget()
        atomic: list[str] = []
        overlap = min(self.chunk_overlap, max(0, budget // 10))
        for source_chunk in source_chunks:
            if len(source_chunk) <= budget:
                atomic.append(source_chunk)
                continue
            start = 0
            while start < len(source_chunk):
                end = min(len(source_chunk), start + budget)
                atomic.append(source_chunk[start:end])
                if end == len(source_chunk):
                    break
                start = max(start + 1, end - overlap)

        if len(atomic) <= self.max_chunks:
            return atomic

        groups: list[str] = []
        current = ""
        for chunk in atomic:
            candidate = f"{current}\n{chunk}" if current else chunk
            if current and len(candidate) > budget:
                groups.append(current)
                current = chunk
            else:
                current = candidate
        if current:
            groups.append(current)

        if len(groups) > self.max_chunks:
            required = len(groups)
            raise ProviderRequestTooLargeError(
                f"Document requires {required} provider calls at the configured "
                f"LLM_MAX_INPUT_CHARS budget, exceeding GEMINI_MAX_CHUNKS={self.max_chunks}."
            )
        return groups

    def _generate_content(self, prompt: str, schema: Any) -> Any:
        if self.client is None:
            status = self.provider_status()
            raise RuntimeError(status["error"] or "LLM provider is not configured.")
        if self.quota_cooldown_remaining() > 0:
            raise ProviderQuotaError(round(self.quota_cooldown_remaining()))
        for attempt in range(self.max_retries + 1):
            try:
                if self.provider == "gemini":
                    return self.client.models.generate_content(
                        model=self.model, contents=prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema),
                    )
                request = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1,
                }
                try:
                    response = self.client.chat.completions.create(
                        **request, timeout=self.request_timeout_seconds
                    )
                except Exception as exc:
                    if classify_provider_error(exc, self.provider)["code"] != "provider_structured_output":
                        raise
                    request.pop("response_format")
                    response = self.client.chat.completions.create(
                        **request, timeout=self.request_timeout_seconds
                    )
                content = response.choices[0].message.content
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Provider returned an empty response.")
                return type("OpenAIResponse", (), {"text": content})()
            except Exception as exc:
                details = classify_provider_error(exc, self.provider)
                if details["code"] != "provider_quota" or attempt >= self.max_retries:
                    if details["code"] == "provider_quota":
                        self._set_quota_cooldown(details["retry_after_seconds"])
                        raise ProviderQuotaError(details["retry_after_seconds"]) from exc
                    raise
                retry_after = details["retry_after_seconds"] or 2**attempt
                if retry_after > self.max_retry_wait:
                    raise ProviderQuotaError(retry_after) from exc
                time.sleep(min(retry_after, self.max_retry_wait))

    def _set_quota_cooldown(self, retry_after_seconds: int | None) -> None:
        if retry_after_seconds:
            self._quota_cooldown_until = max(
                self._quota_cooldown_until, time.time() + retry_after_seconds
            )

    def quota_cooldown_remaining(self) -> int:
        return max(0, round(self._quota_cooldown_until - time.time()))

    def quota_status(self) -> dict | None:
        remaining = self.quota_cooldown_remaining()
        if not remaining:
            return None
        return {
            "code": "provider_quota",
            "message": f"LLM provider quota cooldown is active for about {remaining} seconds.",
            "retry_after_seconds": remaining,
            "retryable": True,
        }

    def reset_quota_cooldown(self) -> None:
        self._quota_cooldown_until = 0.0

    def provider_smoke_test(self) -> dict:
        """Make one tiny JSON-only call and return safe readiness diagnostics."""
        status = self.provider_status()
        if not status["configured"]:
            return {
                "status": "not_ready",
                "provider": self.provider,
                "model": self.model,
                "error": status["error"] or "Provider is not configured.",
            }
        try:
            response = self._generate_content(
                'Return exactly this JSON object and nothing else: {"ok":true}',
                dict,
            )
            if self._parse_json(response.text) != {"ok": True}:
                raise ValueError("Provider returned an unexpected smoke-test response.")
            return {"status": "ready", "provider": self.provider, "model": self.model}
        except Exception as exc:
            return {
                "status": "not_ready",
                "provider": self.provider,
                "model": self.model,
                "error": self._error_with_context(exc),
            }

    def _extract_chunk(self, chunk: str) -> dict:
        prompt = f"{self._extraction_prompt_prefix()}\n{chunk}\n"
        response = self._generate_content(prompt, FactList)
        try:
            return self._parse_json(response.text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Provider returned invalid JSON.") from exc

    def _error_with_context(self, error: Exception, *, chunk_index: int | None = None, chunk_total: int | None = None) -> dict:
        details = classify_provider_error(error, self.provider)
        status = getattr(error, "status_code", None) or getattr(
            getattr(error, "response", None), "status_code", None
        )
        if not status:
            match = re.search(r"\b([45]\d{2})\b", str(error))
            status = int(match.group(1)) if match else None
        details.update(
            {
                "provider": self.provider,
                "model": self.model,
                "status": status or "unknown",
            }
        )
        if chunk_index is not None:
            details["provider_call"] = chunk_index + 1
            details["chunk"] = chunk_index + 1
        if chunk_total is not None:
            details["provider_calls_total"] = chunk_total
            details["chunks_total"] = chunk_total
        return details

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Decode provider output without inventing or repairing facts."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Provider returned invalid JSON.")
        candidate = text.strip()
        if candidate.startswith("```") and candidate.endswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        objects = []
        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)
        if len(objects) == 1:
            return objects[0]
        raise ValueError("Provider returned invalid JSON.")

    def process_document(self, filepath: str, filename: str, progress_callback=None):
        errors = []
        extraction_failures = []
        successful_calls = 0
        extracted = 0
        source_chunks = list(self.iter_text_chunks(filepath, filename))
        if not source_chunks:
            return {
                "status": "failed",
                "message": f"No extractable text found in {filename}",
                "facts_extracted": 0,
                "errors": ["No extractable text found"],
            }

        # Stage 1-3: Segment, Candidate detection, Structure into Facts
        leftover_candidates = []
        local_facts = []

        for chunk_idx, chunk in enumerate(source_chunks):
            nlp = get_nlp()
            if nlp is not None:
                doc = nlp(chunk)
                for sent_id, sent in enumerate(doc.sents):
                    text = sent.text.strip()
                    if not text:
                        continue
                    
                    page = None
                    markers = list(re.finditer(r"---\s*Page\s+(\d+)\s*---", chunk, flags=re.I))
                    for m in markers:
                        if m.start() <= sent.start_char:
                            page = int(m.group(1))
                        else:
                            break
                    
                    ents = [ent.label_ for ent in sent.ents]
                    has_target_ent = any(ent in TARGET_ENTS for ent in ents)
                    has_verb = any(token.lemma_.lower() in CURATED_VERBS or token.text.lower() in CURATED_VERBS for token in sent)
                    has_svo = any(token.dep_ == "nsubj" for token in sent) and any(token.pos_ == "VERB" for token in sent)
                    
                    if has_target_ent or has_verb or has_svo:
                        match = re.search(r"(?i)^(revenue|net income|profit|sales|expenses)\s+(?:was|is|were|grew to)\s+([\$\d\.,A-Za-z]+)\s+in\s+([A-Za-z0-9]+)\.?$", text)
                        if match:
                            attribute, value, period = match.groups()
                            local_facts.append({
                                "text": text,
                                "value": value,
                                "excerpt": text,
                                "page": page,
                                "subject": attribute,
                                "predicate": "was",
                                "raw_value": value,
                                "time_expression": period,
                                "confidence": 1.0,
                            })
                        else:
                            leftover_candidates.append({
                                "sentence_id": sent_id,
                                "char_span": [sent.start_char, sent.end_char],
                                "raw_text": text,
                                "page": page
                            })
            else:
                leftover_candidates.append({
                    "sentence_id": 0,
                    "char_span": [0, len(chunk)],
                    "raw_text": chunk,
                    "page": 1
                })

        # Save local facts immediately
        new_facts = []
        for item in local_facts:
            text = str(item.get("text") or "").strip()
            value = item.get("value")
            new_facts.append(
                Fact(
                    id=str(uuid.uuid4()),
                    text=text,
                    value=value,
                    evidence=[
                        FactEvidence(
                            document_name=filename,
                            page=item.get("page"),
                            excerpt=item.get("excerpt", ""),
                        )
                    ],
                    **_metadata_for_claim(item, text, value),
                )
            )
        
        with self._lock:
            self.facts.extend(new_facts)
            self._facts_version += len(new_facts)
            self._reasoning_cache = None
        extracted += len(new_facts)

        # Process leftover candidates with LLM
        provider_calls_total = 0
        if leftover_candidates:
            # Batch them into chunks that fit in the prompt budget
            budget = self._provider_input_budget()
            candidate_chunks = []
            current_chunk = []
            current_len = 0
            
            import json
            for candidate in leftover_candidates:
                c_str = json.dumps({"page": candidate["page"], "raw_text": candidate["raw_text"]})
                if current_len + len(c_str) > budget and current_chunk:
                    candidate_chunks.append("\n".join(current_chunk))
                    current_chunk = [c_str]
                    current_len = len(c_str)
                else:
                    current_chunk.append(c_str)
                    current_len += len(c_str)
            if current_chunk:
                candidate_chunks.append("\n".join(current_chunk))
                
            provider_calls_total = len(candidate_chunks)
            
            def extract(index_and_chunk):
                index, chunk = index_and_chunk
                try:
                    data = self._extract_chunk(chunk)
                    batch_facts = []
                    for item in data.get("facts", []):
                        failure = _evidence_failure(item, chunk)
                        if failure:
                            extraction_failures.append(
                                {
                                    "type": "extraction_failure",
                                    "description": failure,
                                    "document_name": filename,
                                    "page": item.get("page"),
                                    "claim": item.get("text"),
                                }
                            )
                            continue
                        text = str(item.get("text") or "").strip()
                        value = item.get("value")
                        batch_facts.append(
                            Fact(
                                id=str(uuid.uuid4()),
                                text=text,
                                value=value,
                                evidence=[
                                    FactEvidence(
                                        document_name=filename,
                                        page=item.get("page"),
                                        excerpt=item.get("excerpt", ""),
                                    )
                                ],
                                **_metadata_for_claim(item, text, value),
                            )
                        )
                    return index, batch_facts, None
                except Exception as exc:
                    return index, [], self._error_with_context(
                        exc, chunk_index=index, chunk_total=provider_calls_total
                    )

            with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as executor:
                futures = [executor.submit(extract, item) for item in enumerate(candidate_chunks)]
                for completed, future in enumerate(as_completed(futures), start=1):
                    _, batch_facts, error = future.result()
                    with self._lock:
                        self.facts.extend(batch_facts)
                        self._facts_version += len(batch_facts)
                        self._reasoning_cache = None
                    extracted += len(batch_facts)
                    if error:
                        errors.append(error)
                    else:
                        successful_calls += 1
                    if progress_callback:
                        progress_callback(completed, provider_calls_total)

        with self._lock:
            self.extraction_failures.extend(extraction_failures)
        if not extracted and (errors or extraction_failures):
            status = "failed"
        elif errors or extraction_failures:
            status = "partial"
        else:
            status = "success"
        return {
            "status": status,
            "message": f"Processed {filename}",
            "facts_extracted": extracted,
            "errors": errors,
            "extraction_failures": extraction_failures,
            "chunks_total": provider_calls_total,
            "source_chunks_total": len(source_chunks),
            "provider_calls_total": provider_calls_total,
            "provider_calls_succeeded": successful_calls,
            "provider_calls_failed": len(errors),
            "failed_provider_calls": [
                error.get("provider_call")
                for error in errors
                if isinstance(error, dict) and error.get("provider_call") is not None
            ],
            "quota": next((error for error in errors if isinstance(error, dict) and error["code"] == "provider_quota"), None),
        }

    def get_facts(self) -> List[Fact]:
        with self._lock:
            return list(self.facts)

    def load_demo(self) -> dict:
        """Load deterministic, clearly synthetic facts for credential-free demos."""
        cases = [
            ("corroboration-a", "Service handled requests", "1200", "Operations report confirms 1200 requests."),
            ("corroboration-b", "Service handled requests", "1200", "Independent review confirms 1200 requests."),
            ("genuine_contradiction", "Service handled requests", "900", "Audit reports 900 requests in the same period."),
            ("explained_by_context", "Service handled requests", "1200", "The 1200 figure covers the full year; the other covers Q1."),
            ("extraction_failure", "Unclear claim", None, "The source is too ambiguous to classify safely."),
        ]
        demo_facts = [
            Fact(
                id=f"demo-{index}",
                text=text,
                value=value,
                evidence=[FactEvidence(document_name=f"demo-{kind}.txt", page=1, excerpt=excerpt)],
            )
            for index, (kind, text, value, excerpt) in enumerate(cases)
        ]
        with self._lock:
            self.facts = demo_facts
            self.reset_quota_cooldown()
            self._facts_version += 1
            self._reasoning_cache = (self._facts_version, self._demo_reasoning())
        return self._reasoning_cache[1]

    def build_graph(self, reasoning: dict | None = None) -> dict:
        """Build renderer-neutral graph data from the evidence-first result."""
        facts = self.get_facts()
        nodes = [
            {
                "id": fact.id,
                "label": fact.text,
                "value": fact.value,
                "evidence": [item.model_dump() for item in fact.evidence],
                "kind": "fact",
            }
            for fact in facts
        ]
        node_ids = {node["id"] for node in nodes}
        edges = []
        result = reasoning if reasoning is not None else self.run_reasoning()
        for group in ("corroborations", "contradictions"):
            for index, relationship in enumerate(result.get(group, [])):
                left = relationship.get("fact_1") or {}
                right = relationship.get("fact_2") or {}
                source = left.get("id")
                target = right.get("id")
                relation_type = relationship.get("type", "corroboration")
                if source in node_ids and target in node_ids:
                    edges.append(
                        {
                            "id": f"edge-{len(edges)}",
                            "source": source,
                            "target": target,
                            "type": relation_type,
                            "label": RELATIONSHIP_STYLES.get(
                                relation_type, RELATIONSHIP_STYLES["extraction_failure"]
                            )["label"],
                            "color": RELATIONSHIP_STYLES.get(
                                relation_type, RELATIONSHIP_STYLES["extraction_failure"]
                            )["color"],
                            "explanation": relationship.get("explanation", ""),
                        }
                    )
        for index, failure in enumerate(result.get("failures", [])):
            node_id = f"failure-{index}"
            nodes.append(
                {
                    "id": node_id,
                    "label": failure.get("description", "Extraction failure"),
                    "value": None,
                    "evidence": [],
                    "kind": "failure",
                }
            )
            edges.append(
                {
                    "id": f"edge-{len(edges)}",
                    "source": node_id,
                    "target": node_id,
                    "type": "extraction_failure",
                    "label": RELATIONSHIP_STYLES["extraction_failure"]["label"],
                    "color": RELATIONSHIP_STYLES["extraction_failure"]["color"],
                    "explanation": failure.get("description", ""),
                }
            )
        return {"nodes": nodes, "edges": edges, "legend": RELATIONSHIP_STYLES, "demo": bool(result.get("demo"))}

    def _demo_reasoning(self) -> dict:
        facts = {fact.id: fact.model_dump() for fact in self.facts}
        return {
            "demo": True,
            "corroborations": [{"type": "corroboration", "fact_1": facts["demo-0"], "fact_2": facts["demo-1"], "explanation": "Synthetic independent sources agree."}],
            "contradictions": [
                {"type": "genuine_contradiction", "fact_1": facts["demo-0"], "fact_2": facts["demo-2"], "explanation": "Synthetic same-scope values differ."},
                {"type": "explained_by_context", "fact_1": facts["demo-0"], "fact_2": facts["demo-3"], "explanation": "Synthetic sources use different reporting periods."},
            ],
            "failures": [{"type": "extraction_failure", "description": "Synthetic ambiguous source; no fact was fabricated."}],
        }

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]{2,}", text.lower()))
        
    def retrieve_related_facts(self, facts: List[Fact], top_k: int = 5) -> list[tuple[Fact, Fact]]:
        # Stage 4: Canonicalize
        # Normalize unit table
        unit_map = {
            "$": "usd", "₹": "inr", "%": "%", "k": "thousand", "m": "million", 
            "b": "billion", "lakh": "100k", "crore": "10m",
            "lakhs": "100k", "crores": "10m"
        }
        for fact in facts:
            if fact.normalized_unit:
                fact.normalized_unit = unit_map.get(fact.normalized_unit.lower(), fact.normalized_unit.lower())
            
        pairs = set()
        
        # We need to find pairs of facts that are about the same entity and attribute.
        # Since 'subject' is attribute, and entity might be extracted in 'text' or 'subject'
        # We will use RapidFuzz for string similarity and embedder for semantic similarity.
        
        # Precompute embeddings for subjects
        subjects = [f.subject or f.text for f in facts]
        embeddings = None
        embedder = get_embedder()
        if embedder is not None and subjects:
            embeddings = embedder.encode(subjects)
            
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                # RapidFuzz for entity/text similarity
                fuzz_score = fuzz.ratio((facts[i].subject or facts[i].text).lower(), (facts[j].subject or facts[j].text).lower())
                
                # SentenceTransformer for attribute similarity
                cosine_sim = 0
                if embeddings is not None:
                    # dot product of l2 normalized vectors is cosine sim
                    from numpy import dot
                    from numpy.linalg import norm
                    vec1 = embeddings[i]
                    vec2 = embeddings[j]
                    if norm(vec1) > 0 and norm(vec2) > 0:
                        cosine_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
                
                fuzz_val = fuzz_score / 100.0
                
                # Multi-Signal Relevance Scoring:
                # 60% Semantic Vector Similarity (Dense) + 40% Lexical Token Alignment (Sparse)
                if embeddings is not None:
                    score = (0.6 * cosine_sim) + (0.4 * fuzz_val)
                else:
                    score = fuzz_val
                
                if score >= 0.85:
                    pairs.add(tuple(sorted((i, j))))
                elif score >= 0.75:
                    # Borderline match, LLM check needed
                    # We will mark it by adding it to pairs, and the gate will handle it
                    pairs.add(tuple(sorted((i, j))))
                    
        return [(facts[i], facts[j]) for i, j in pairs]

    @staticmethod
    def _relationship_gate(left: Fact, right: Fact) -> dict | None:
        if left.subject is None and right.subject is None and left.predicate is None and right.predicate is None:
            return {"type": "ambiguous", "metadata": {"gate": "legacy_fields", "llm_required": True}}
        if not (_token_overlap(left.subject, right.subject) and _token_overlap(left.predicate, right.predicate)):
            return None
        left_unit = left.normalized_unit
        right_unit = right.normalized_unit
        if left_unit and right_unit and left_unit != right_unit:
            return {
                "type": "explained_by_context",
                "explanation": f"Claims use incompatible normalized units ({left_unit} vs {right_unit}).",
                "metadata": {"gate": "unit_mismatch", "llm_required": False},
            }
        left_period = (left.time_start, left.time_end, left.time_expression)
        right_period = (right.time_start, right.time_end, right.time_expression)
        if left_period[0] and right_period[0] and left_period[0] != right_period[0]:
            return {
                "type": "explained_by_context",
                "explanation": "Claims refer to disjoint or different reported periods.",
                "metadata": {"gate": "time_context", "llm_required": False},
            }
        if left.scope_expression and right.scope_expression and left.scope_expression.strip().lower() != right.scope_expression.strip().lower():
            return {
                "type": "explained_by_context",
                "explanation": "Claims use different explicit scopes.",
                "metadata": {"gate": "scope_context", "llm_required": False},
            }
        if (
            left.normalized_value is not None
            and left.normalized_value == right.normalized_value
            and (not left_unit or not right_unit or left_unit == right_unit)
        ):
            return {
                "type": "corroboration",
                "explanation": "Independent documents report the same normalized value in the same scope.",
                "metadata": {"gate": "exact_normalized_match", "llm_required": False},
            }
        return {"type": "ambiguous", "metadata": {"gate": "llm_required", "llm_required": True}}

    @staticmethod
    def _ground_relationships(result: dict, facts: List[Fact]) -> dict:
        """Replace model claim strings with the original evidence-backed facts."""
        lookup: dict[str, list[Fact]] = defaultdict(list)
        for fact in facts:
            for key in (fact.text.strip().lower(), f"{fact.text} {fact.value or ''}".strip().lower()):
                if key and fact not in lookup[key]:
                    lookup[key].append(fact)
        grounded = dict(result)
        for group in ("corroborations", "contradictions"):
            relationships = []
            for relationship in result.get(group, []):
                item = dict(relationship)
                used_ids = set()
                for field in ("fact_1", "fact_2"):
                    claim = item.get(field)
                    if isinstance(claim, str):
                        matches = lookup.get(claim.strip().lower(), [])
                        fact = next((candidate for candidate in matches if candidate.id not in used_ids), None)
                        if fact is not None:
                            used_ids.add(fact.id)
                            item[field] = fact.model_dump()
                relationships.append(item)
            grounded[group] = relationships
        return grounded

    def run_reasoning(self):
        facts = self.get_facts()
        if not facts:
            return {"corroborations": [], "contradictions": [], "failures": []}
        pairs = self.retrieve_related_facts(facts)
        if not pairs:
            return {"corroborations": [], "contradictions": [], "failures": list(self.extraction_failures)}
        gated = []
        candidates = []
        deterministic = {"corroborations": [], "contradictions": []}
        for left, right in pairs:
            gate = self._relationship_gate(left, right)
            if gate is None:
                continue
            relationship = {
                **gate,
                "fact_1": left.model_dump(),
                "fact_2": right.model_dump(),
            }
            if gate["type"] == "ambiguous":
                candidates.append({"fact_1": left.model_dump(), "fact_2": right.model_dump()})
                gated.append((left, right))
            else:
                deterministic["corroborations" if gate["type"] == "corroboration" else "contradictions"].append(relationship)
        if not candidates:
            return {**deterministic, "failures": list(self.extraction_failures)}
        if self.client is None:
            return {
                **deterministic,
                "failures": list(self.extraction_failures) + [{"type": "reasoning_failure", "description": self.provider_status()["error"] or "LLM provider is not configured"}],
            }
        with self._lock:
            if self._reasoning_cache and self._reasoning_cache[0] == self._facts_version:
                return self._reasoning_cache[1]
        prompt = f"""
Classify each candidate fact pair. Use exactly one of these relationship types:
"corroboration" (independent evidence supports the same claim),
"genuine_contradiction" (the claims cannot both be true in the same scope),
"explained_by_context" (the apparent conflict is explained by period, scope,
currency, unit, or another explicit context), or "extraction_failure" (evidence
is insufficient or malformed). Include the supporting evidence and reasoning in
"explanation". Put corroboration pairs in "corroborations", the two contradiction
types in "contradictions", and extraction failures in "failures".

Candidate pairs:
{json.dumps(candidates, indent=2)}
"""
        try:
            response = self._generate_content(prompt, ReasoningOutput)
            result = self._ground_relationships(self._parse_json(response.text), facts)
            result["corroborations"] = deterministic["corroborations"] + result.get("corroborations", [])
            result["contradictions"] = deterministic["contradictions"] + result.get("contradictions", [])
            result["failures"] = list(self.extraction_failures) + result.get("failures", [])
            with self._lock:
                self._reasoning_cache = (self._facts_version, result)
            return result
        except Exception as exc:
            error = classify_provider_error(exc, self.provider)
            return {
                "corroborations": [],
                "contradictions": [],
                "failures": [{"type": "reasoning_failure", "description": error["message"], "error": error}],
            }


fact_layer = FactLayer()
