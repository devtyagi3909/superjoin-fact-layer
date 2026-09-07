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
python api/main.py                            # terminal 1
streamlit run ui/app.py                       # terminal 2
```

Open <http://localhost:8501>. The API is at <http://localhost:8000>; `/health`
is safe to use as a readiness check and never returns the key.

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

## API contract

| Endpoint | Purpose |
| --- | --- |
| `POST /upload` | Queue one `.pdf` or `.txt` file; returns `202` and `{job_id, filename, status}` immediately. |
| `POST /uploads` | Queue repeated `files` parts for batch processing; returns `202` and a `jobs` array. |
| `GET /upload/{job_id}` | Poll `queued`, `processing`, `success`, `partial`, or `failed`; includes progress, chunk counts, result, and error. |
| `GET /facts` | Return extracted claims and source/page excerpts currently held in memory. |
| `POST /demo` | Load deterministic synthetic facts and four relationship cases without Gemini calls. |
| `GET /corroborations` | Retrieve bounded related pairs and classify relationships. |
| `GET /health` | Return service readiness and whether a Gemini client is configured. |

Uploads are capped at 25 MB by default (`MAX_UPLOAD_BYTES` can override it).
State is intentionally in memory for the assignment demo and is cleared on
restart.

## Architecture and tradeoffs

`core/parser.py` uses PyMuPDF for page-aware extraction, bounded overlapping
chunks, a configurable total call budget, structured Gemini output, and a limited thread pool. Each fact keeps a
stable ID plus document, page, and verbatim excerpt. An inverted lexical index
selects candidate pairs without an all-pairs explosion; Gemini then reasons
only over those candidates. Provider quota errors are normalized to a short
code/message/retry-after shape; only one short retry is attempted by default,
and the UI never retries automatically. `api/main.py` owns validation and
asynchronous job state, while `ui/app.py` is a small evidence-first Streamlit review workspace
with progress, empty/error states, search/filtering through the fact table, and
expandable evidence.

The tradeoff is deliberate: lexical retrieval is transparent and dependency
light, but synonyms can be missed. In-memory state is easy to review locally,
but a production deployment would use durable job storage and a vector index.
Scanned PDFs without an OCR layer may yield no text and are reported as a
failure rather than silently producing unsupported claims.

## Validation

```bash
python3 -m pytest -q
python3 -m compileall -q core api ui
```

Tests cover incremental chunking, evidence preservation and grounding, empty
inputs, batch uploads, extension validation, async lifecycle, health/API shape,
quota classification/retry handling, the offline demo, and the four-case relationship response contract. Live Gemini classification requires
`GEMINI_API_KEY`; all other checks run offline.
