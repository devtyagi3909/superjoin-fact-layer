# Superjoin Fact Knowledge Layer

This project implements a Fact Knowledge Layer designed to ingest financial documents (PDFs), extract meaningful semantic and numerical facts using a large language model (LLM), and perform cross-document reasoning to identify corroborations, genuine contradictions, and context-explained contradictions.

Built as an exploration into AI agents for finance for the Superjoin Engineering Intern Hiring Assignment.

## Setup and Run Instructions

**Prerequisites:**
- Python 3.10+
- A Google Gemini API Key (`GEMINI_API_KEY` environment variable). The system uses Gemini 2.5 Pro for intelligent, dynamic fact extraction via Google GenAI SDK.

1.  **Clone the repository & navigate to the folder:**
    ```bash
    git clone https://github.com/yourusername/superjoin-fact-layer.git
    cd superjoin-fact-layer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set your API Key:**
    ```bash
    export GEMINI_API_KEY="your-gemini-api-key"
    # Optional: override the model available in your Google GenAI project
    export GEMINI_MODEL="gemini-3.5-flash"
    # Optional: parallel Gemini requests per document (default: 4)
    export GEMINI_MAX_WORKERS="4"
    ```

5.  **Run the application (Requires two terminal windows):**
    
    *Terminal 1 - Start the FastAPI backend:*
    ```bash
    python api/main.py
    ```
    
    *Terminal 2 - Start the Streamlit UI:*
    ```bash
    streamlit run ui/app.py
    ```
    
    The UI will be accessible at `http://localhost:8501`.

### Try the included starter data

The assignment bundle contains curated PDFs under `../delhivery/` and
`../india-macroeconomy/`. Upload at least two documents from the same dataset to
exercise cross-document reasoning. The API returns a `job_id` immediately:

```bash
curl -F "file=@../delhivery/01-delhivery-prospectus-2022-excerpt.pdf" \
  http://localhost:8000/upload
curl http://localhost:8000/upload/<job_id>
curl http://localhost:8000/facts
curl http://localhost:8000/corroborations
```

## Video Demo

Add an unlisted video link before submission. It should be no longer than three
minutes and show one upload, job polling, the facts table, and all four required
outcomes. The repository is intentionally credential-free; reviewers can use
the included sample PDFs with their own Gemini key.

*The video demonstrates uploading documents, viewing the extracted facts linked to source evidence, and the reasoning engine correctly categorizing relationships.*

## Approach

The system is designed with a **separation of concerns** representing modern AI product architectures:

1.  **Core Parser (`core/parser.py`)**: Uses `PyMuPDF (fitz)` to accurately extract text from documents, maintaining page structures. It chunks the text, applies explicit `Pydantic` schemas, and sends it to the configured Gemini model using the `google-genai` structured outputs feature.
    - *Why this matters*: Structured JSON schemas keep chunk results robust while bounded page-aware chunks avoid sending an entire large PDF in one request. Chunk requests run concurrently with a bounded worker pool (`GEMINI_MAX_WORKERS`) so large reports do not wait on dozens of sequential API calls. The model is configurable with `GEMINI_MODEL`.
2.  **Retrieval and Reasoning**: An in-memory inverted lexical index selects a small set of related fact pairs before the LLM evaluates them. This avoids all-pairs comparisons and degrades with a clear failure when `GEMINI_API_KEY` is unavailable. Relationships are classified as `corroboration`, `genuine_contradiction`, `explained_by_context`, or `extraction_failure`.
3.  **API (`api/main.py`)**: A `FastAPI` layer serves as the backbone. This means the knowledge layer isn't just a script—it's a microservice ready to be integrated into a larger IPO readiness platform.
4.  **UI (`ui/app.py`)**: An evidence-first `Streamlit` dashboard for merchant bankers to inspect source excerpts, live job progress, and four explicit relationship cases.

## Limitations and Next Steps

**What doesn't work perfectly yet:**
- Facts and job state are currently in memory and are lost when the server restarts.
- The retrieval layer is deterministic lexical retrieval rather than a hosted vector database, so domain-specific synonyms may not match.
- The four-case examples are data-dependent: without a Gemini key, the local
  smoke tests verify extraction and job failure handling but cannot produce
  live relationship classifications.

### Representative output shape

Each fact includes a stable ID and source evidence:

```json
{
  "id": "generated-uuid",
  "text": "Revenue increased during the reported period",
  "value": "₹X crore",
  "evidence": [{
    "document_name": "annual-report.pdf",
    "page": 42,
    "excerpt": "verbatim supporting text"
  }]
}
```

Reasoning returns corroborations, contradictions (including contextual
reconciliation), and explicit failures with explanations. No company-specific
terms, filenames, or facts are hard-coded.

**What I would build next (Next Steps):**
- **Persistent Retrieval**: Replace the in-memory index with ChromaDB or Qdrant when durable storage and embeddings are required.
- **Source Highlighting**: Pass exact bounding boxes from `PyMuPDF` to the frontend UI so users can click a fact and see the exact highlight on the original PDF.

## Additional Notes

- The system handles the 4 required edge cases entirely dynamically. The LLM is capable of realizing that a "Q4 Loss" and "FY24 Profit" might not be a contradiction (explained by context/time period).
- The choice of FastAPI + Streamlit demonstrates backend robustness while shipping the frontend fast, simulating how an AI startup needs to operate.
