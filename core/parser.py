import json
import os
import re
import time
import threading
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, List, Optional

import fitz  # PyMuPDF
from google import genai
from google.genai import types
from pydantic import BaseModel


class FactEvidence(BaseModel):
    document_name: str
    page: Optional[int]
    excerpt: str


class Fact(BaseModel):
    id: str
    text: str
    value: Optional[str]
    evidence: List[FactEvidence]


class FactOutput(BaseModel):
    text: str
    value: Optional[str] = None
    excerpt: str
    page: Optional[int] = None


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
        message = "Gemini quota is temporarily exhausted."
        if retry_after_seconds:
            message += f" Retry after about {retry_after_seconds} seconds."
        super().__init__(message)


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


def classify_provider_error(error: Exception) -> dict:
    """Return a stable API/UI error without exposing provider payloads."""
    text = str(error).lower()
    if "429" in text or "resource_exhausted" in text or "quota" in text or "rate limit" in text:
        retry_after = _retry_after_seconds(error)
        message = "Gemini quota is temporarily exhausted. Please wait and retry the upload."
        if retry_after:
            message = f"Gemini quota is temporarily exhausted. Retry after about {retry_after} seconds."
        return {
            "code": "provider_quota",
            "message": message,
            "retry_after_seconds": retry_after,
            "retryable": True,
        }
    return {
        "code": "extraction_error",
        "message": "The document could not be processed. Check the file and try again.",
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
        self.client = client if client is not None else self._build_client()
        self.model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        self.max_workers = max_workers or int(os.getenv("GEMINI_MAX_WORKERS", "2"))
        self.max_chunks = int(os.getenv("GEMINI_MAX_CHUNKS", "8"))
        self.max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "1"))
        self.max_retry_wait = int(os.getenv("GEMINI_MAX_RETRY_WAIT_SECONDS", "8"))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._lock = threading.RLock()
        self._facts_version = 0
        self._reasoning_cache: tuple[int, dict] | None = None

    @staticmethod
    def _build_client() -> Any:
        # Importing the layer remains useful for local tests and API health checks.
        if not os.getenv("GEMINI_API_KEY"):
            return None
        return genai.Client()

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

    def _generate_content(self, prompt: str, schema: Any) -> Any:
        if self.client is None:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
            except Exception as exc:
                details = classify_provider_error(exc)
                if details["code"] != "provider_quota" or attempt >= self.max_retries:
                    if details["code"] == "provider_quota":
                        raise ProviderQuotaError(details["retry_after_seconds"]) from exc
                    raise
                retry_after = details["retry_after_seconds"] or 2**attempt
                if retry_after > self.max_retry_wait:
                    raise ProviderQuotaError(retry_after) from exc
                time.sleep(min(retry_after, self.max_retry_wait))

    def _extract_chunk(self, chunk: str) -> dict:
        prompt = f"""
Extract key factual claims from this document chunk. Return only claims supported by
the text. For every claim provide a concise "text", a numerical or categorical
"value" when present, a verbatim supporting "excerpt", and the source "page".
Page markers are authoritative:
{chunk}
"""
        response = self._generate_content(prompt, FactList)
        return json.loads(response.text)

    def process_document(self, filepath: str, filename: str, progress_callback=None):
        errors = []
        extracted = 0
        chunks = self._bounded_chunks(list(self.iter_text_chunks(filepath, filename)))
        total_chunks = len(chunks)
        if not chunks:
            return {
                "status": "failed",
                "message": f"No extractable text found in {filename}",
                "facts_extracted": 0,
                "errors": ["No extractable text found"],
            }

        def extract(index_and_chunk):
            index, chunk = index_and_chunk
            try:
                data = self._extract_chunk(chunk)
                new_facts = []
                for item in data.get("facts", []):
                    new_facts.append(
                        Fact(
                            id=str(uuid.uuid4()),
                            text=item.get("text", ""),
                            value=item.get("value"),
                            evidence=[
                                FactEvidence(
                                    document_name=filename,
                                    page=item.get("page"),
                                    excerpt=item.get("excerpt", ""),
                                )
                            ],
                        )
                    )
                return index, new_facts, None
            except Exception as exc:
                return index, [], classify_provider_error(exc)

        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as executor:
            futures = [executor.submit(extract, item) for item in enumerate(chunks)]
            for completed, future in enumerate(as_completed(futures), start=1):
                _, new_facts, error = future.result()
                with self._lock:
                    self.facts.extend(new_facts)
                    self._facts_version += len(new_facts)
                    self._reasoning_cache = None
                extracted += len(new_facts)
                if error:
                    errors.append(error)
                if progress_callback:
                    progress_callback(completed, total_chunks)
        if not extracted and errors:
            status = "failed"
        elif errors:
            status = "partial"
        else:
            status = "success"
        return {
            "status": status,
            "message": f"Processed {filename}",
            "facts_extracted": extracted,
            "errors": errors,
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
            self._facts_version += 1
            self._reasoning_cache = (self._facts_version, self._demo_reasoning())
        return self._reasoning_cache[1]

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
        """Use an inverted lexical index, avoiding an all-pairs comparison."""
        index: dict[str, set[int]] = defaultdict(set)
        tokens = []
        for i, fact in enumerate(facts):
            fact_tokens = FactLayer._tokens(f"{fact.text} {fact.value or ''}")
            tokens.append(fact_tokens)
            for token in fact_tokens:
                index[token].add(i)
        pairs: set[tuple[int, int]] = set()
        for i, fact_tokens in enumerate(tokens):
            candidates = Counter(j for token in fact_tokens for j in index[token] if j != i)
            for j, _ in candidates.most_common(top_k):
                pairs.add(tuple(sorted((i, j))))
        return [(facts[i], facts[j]) for i, j in pairs]

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
            return {"corroborations": [], "contradictions": [], "failures": []}
        candidates = [
            {"fact_1": left.model_dump(), "fact_2": right.model_dump()}
            for left, right in pairs
        ]
        if self.client is None:
            return {
                "corroborations": [],
                "contradictions": [],
                "failures": [{"type": "reasoning_failure", "description": "GEMINI_API_KEY is not configured"}],
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
            result = self._ground_relationships(json.loads(response.text), facts)
            with self._lock:
                self._reasoning_cache = (self._facts_version, result)
            return result
        except Exception as exc:
            error = classify_provider_error(exc)
            return {
                "corroborations": [],
                "contradictions": [],
                "failures": [{"type": "reasoning_failure", "description": error["message"], "error": error}],
            }


fact_layer = FactLayer()
