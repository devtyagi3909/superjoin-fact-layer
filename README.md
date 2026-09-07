# Superjoin Fact Knowledge Layer

An evidence-first document intelligence demo: upload one or more PDFs (or text
fixtures), extract generic claims with page-level evidence, retrieve related
claims, and classify their relationship as corroboration, genuine contradiction,
contextual reconciliation, or an explicit failure. The implementation does not
assume a company, filename, or financial vocabulary.

## Run it locally

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="..."
export GEMINI_MODEL="gemini-3.5-flash"       # optional
export GEMINI_MAX_WORKERS="2"                # optional; keep low on free tier
export GEMINI_MAX_CHUNKS="8"                 # optional; groups pages, never drops them
export GEMINI_MAX_RETRIES="1"                # optional; bounded 429 retry count
export GEMINI_MAX_RETRY_WAIT_SECONDS="8"     # optional; no indefinite waits
export LLM_MAX_INPUT_CHARS="20000"           # optional; full prompt budget, Groq-safe
python api/main.py                            # terminal 1
streamlit run ui/app.py                       # terminal 2
```

Gemini is retained for backwards compatibility. To use a low-cost
OpenAI-compatible provider instead, install the same requirements and configure
one of these examples:

```bash
# Groq (OpenAI-compatible endpoint; models and limits vary by account)
export LLM_PROVIDER=openai_compatible
export LLM_API_KEY="gsk_..."
export LLM_BASE_URL="https://api.groq.com/openai/v1"
export LLM_MODEL="openai/gpt-oss-120b"
export LLM_MAX_INPUT_CHARS="20000"           # keep 18000-24000 on Groq free tier

# OpenRouter (many free or low-cost models; availability and limits change)
export LLM_PROVIDER=openai_compatible
export LLM_API_KEY="sk-or-..."
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_MODEL="google/gemini-2.0-flash-exp:free"
```

`LLM_MODEL` and `LLM_BASE_URL` are required in practice to select the model and
endpoint you want. Free tiers are subject to provider rate limits, model
availability, credit requirements, and changing policies; they are not
guaranteed. `LLM_API_KEY` is never returned by the API. If `LLM_PROVIDER` is
omitted, `GEMINI_API_KEY` selects Gemini and `LLM_API_KEY` selects the
OpenAI-compatible path.

Open <http://localhost:8501>. The API is at <http://localhost:8000>; `/health`
is safe to use as a readiness check and never returns the key.

For Groq, use this configuration first. `openai/gpt-oss-120b` is recommended
when available on your account; the model catalog can change:

```bash
export LLM_PROVIDER=openai_compatible
export LLM_API_KEY="gsk_..."
export LLM_BASE_URL="https://api.groq.com/openai/v1"
export LLM_MODEL="openai/gpt-oss-120b"

curl -sS --oauth2-bearer "$LLM_API_KEY" \
  https://api.groq.com/openai/v1/models
```

Run a tiny provider-only extraction check without exposing the key:

```bash
curl -sS "$LLM_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $LLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-120b","temperature":0.1,"messages":[{"role":"user","content":"Return JSON only: {\"facts\":[{\"text\":\"The service handled 12 requests.\",\"value\":\"12\",\"excerpt\":\"handled 12 requests\",\"page\":1}]}"}]}'
```

The default `GEMINI_MAX_CHUNKS=8` is a provider-call cap, not a page drop:
even a 28-page document is grouped into at most 8 provider calls. The
independent `LLM_MAX_INPUT_CHARS` budget applies to the complete generated
prompt (instructions plus extracted text), so adjacent source chunks are grouped
only while they fit. An individual long page is split with overlap; text is never
silently discarded. If the document cannot fit both limits, processing fails
clearly rather than fabricating facts.

`/provider-diagnostics` exposes the active provider, model, endpoint, and a
safe configuration hint, including effective input-character budget, maximum
provider calls, and worker count. Never put a key in a debug script or commit it; revoke
and rotate any key that was previously exposed that way.

After changing provider settings, restart the API so the process reloads them:

```bash
kill "$API_PID"                # set API_PID to the API process PID
python api/main.py
```

For a tiny live smoke test, use a short prompt and the configured model:

```bash
curl -sS "$LLM_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$LLM_MODEL\",\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Return JSON only: {\\\"facts\\\":[]}\"}]}"
```

## Deterministic demo path

The UI accepts multiple files in one upload. For a repeatable credential-free
smoke test, use the included plain-text fixtures (the same parser contract is
used for PDFs):

```bash
curl -F "file=@demo/corroboration-a.txt" http://localhost:8000/upload
curl -F "file=@demo/corroboration-b.txt" http://localhost:8000/upload
curl http://localhost:8000/upload/<job_id>
curl http://localhost:8000/facts
curl http://localhost:8000/corroborations
```

With Gemini configured, prepare four small documents that state: (1) the same
claim and value from independent sources (**corroboration**), (2) different
values for the same scope (**genuine contradiction**), (3) different periods,
units, or scopes (**explained by context**), and (4) an unreadable or
unsupported claim (**extraction/reasoning failure**). Upload them together,
wait for each job to reach `success`, `partial`, or `failed`, then select
**Run comparison**. The four tabs make each outcome and its evidence visible.
Without a key, extraction jobs fail honestly with a structured error instead of
fabricating facts; chunking, lifecycle, validation, and UI/API contracts remain
testable offline.

The sidebar's **Load offline demo** button calls `POST /demo` and loads synthetic,
clearly labeled facts for all four relationship cases without contacting Gemini.
This is the recommended evaluator path when no provider quota is available.
It also clears stale upload jobs and quota messages. **Use last successful
results** restores the most recent comparison without another provider call.

## API contract

| Endpoint | Purpose |
| --- | --- |
| `POST /upload` | Queue one `.pdf` or `.txt` file; returns `202` and `{job_id, filename, status}` immediately. |
| `POST /uploads` | Queue repeated `files` parts for batch processing; returns `202` and a `jobs` array. |
| `GET /upload/{job_id}` | Poll `queued`, `processing`, `success`, `partial`, or `failed`; includes progress, chunk counts, result, and error. |
| `GET /facts` | Return extracted claims and source/page excerpts currently held in memory. |
| `GET /graph` | Return fact nodes, relationship edges, status colors, and failure nodes. |
| `GET /source-preview` | Return safe text/PDF page metadata and extracted preview text. |
| `GET /source-file/{document_name}` | Stream a retained uploaded source for PDF preview. |
| `POST /demo` | Load deterministic synthetic facts and four relationship cases without Gemini calls. |
| `GET /corroborations` | Retrieve bounded related pairs and classify relationships. |
| `GET /health` | Return readiness plus active provider, model, SDK availability, and configuration status (never credentials). |
| `GET /provider-diagnostics` | Return safe provider configuration hints and model/endpoint metadata (never credentials). |

Uploads are capped at 25 MB by default (`MAX_UPLOAD_BYTES` can override it).
State is intentionally in memory for the assignment demo and is cleared on
restart.

## Architecture and tradeoffs

`core/parser.py` uses PyMuPDF for page-aware extraction, bounded overlapping
chunks, a configurable total call budget, structured provider output, and a
limited thread pool. Each fact keeps backwards-compatible document, page,
value, and verbatim excerpt fields plus optional normalized subject/predicate,
value/unit, time, scope, polarity, and explainable confidence breakdown
metadata. Before a fact enters the ledger, its excerpt must occur in the
source chunk, its page must match an authoritative page marker, and numeric
values must occur in the excerpt; rejected claims are explicit
`extraction_failure` records. An inverted lexical index selects candidate pairs
without an all-pairs explosion. Deterministic gates require subject/predicate
overlap, classify clear period/unit/scope differences as
`explained_by_context`, and classify exact normalized same-scope values from
independent documents as `corroboration`. Only ambiguous pairs and genuine
contradiction reasoning reach the LLM. Provider quota errors are normalized to a short
code/message/retry-after shape; only one short retry is attempted by default,
and a session-level cooldown blocks new uploads before another provider call.
The offline demo explicitly resets that cooldown. `api/main.py` owns validation and
asynchronous job state, while `ui/app.py` is a small evidence-first Streamlit review workspace
with progress, empty/error states, search/filtering through the fact table,
expandable evidence, a self-contained SVG relationship graph, and side-by-side
source grounding. PDFs are streamed from an API-owned temporary source registry;
the UI displays cited page metadata and requests the cited page where browser
PDF viewers support it, while retaining page navigation as the reliable fallback.

The tradeoff is deliberate: lexical retrieval and deterministic gates are
transparent and dependency-light, but synonyms and genuinely ambiguous context
can still require the LLM. In-memory state is easy to review locally, but a
production deployment would use durable job storage and a vector index.
Scanned PDFs without an OCR layer may yield no text and are reported as a
failure rather than silently producing unsupported claims. The four-case demo
is explicit: same normalized value and scope is corroboration; different
values with the same scope are a genuine contradiction; disjoint periods,
units, or scopes are explained by context; malformed or unsupported evidence is
an extraction/reasoning failure rather than a fabricated fact.

## Validation

```bash
python3 -m pytest -q
python3 -m compileall -q core api ui
```

Tests cover incremental chunking, evidence preservation and grounding, empty
inputs, batch uploads, extension validation, async lifecycle, health/API shape,
quota classification/retry handling, the offline demo, and the four-case relationship response contract. Live Gemini classification requires
`GEMINI_API_KEY`; all other checks run offline.
