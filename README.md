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

## Video Demo

[Link to Demo Video] *(Replace with your unlisted YouTube or Loom link)*

*The video demonstrates uploading documents, viewing the extracted facts linked to source evidence, and the reasoning engine correctly categorizing relationships.*

## Approach

The system is designed with a **separation of concerns** representing modern AI product architectures:

1.  **Core Parser (`core/parser.py`)**: Uses `PyMuPDF (fitz)` to accurately extract text from documents, maintaining page structures. It chunks the text, applies explicit `Pydantic` schemas, and sends it to `gemini-2.5-pro` using the `google-genai` structured outputs feature.
    - *Why this matters*: Using structured JSON schemas directly at the API level (instead of prompt-hacking) guarantees robust output that won't break the application pipeline.
2.  **Reasoning Engine**: Instead of comparing facts via hard-coded keyword matching, the engine takes the aggregated universe of facts and uses the LLM to contextually evaluate relationships, outputting categorizations like `corroboration`, `genuine_contradiction`, and `explained_by_context`. 
3.  **API (`api/main.py`)**: A `FastAPI` layer serves as the backbone. This means the knowledge layer isn't just a script—it's a microservice ready to be integrated into a larger IPO readiness platform.
4.  **UI (`ui/app.py`)**: A fast, responsive `Streamlit` dashboard for merchant bankers to inspect results. 

## Limitations and Next Steps

**What doesn't work perfectly yet:**
- *Token Limits on Massive Documents*: Currently, documents are passed entirely in one context window. While Gemini handles 2M tokens, 1000+ page S-1 filings might degrade reasoning quality or hit limits.
- *Vector Database Missing*: All facts are stored in memory. If the server restarts, knowledge is lost. 

**What I would build next (Next Steps):**
- **Vector Retrieval (RAG)**: Integrate `ChromaDB` or `Qdrant`. When extracting facts from a new document, we would query the vector DB for semantically similar facts first, and only run the Reasoning Engine on that subset. This makes the system scalable to hundreds of documents (O(1) reasoning vs O(N)).
- **Chunking Strategy**: Implement hierarchical chunking for giant PDFs (e.g., LlamaIndex node parsers) so we only process relevant sections.
- **Source Highlighting**: Pass exact bounding boxes from `PyMuPDF` to the frontend UI so users can click a fact and see the exact highlight on the original PDF.

## Additional Notes

- The system handles the 4 required edge cases entirely dynamically. The LLM is capable of realizing that a "Q4 Loss" and "FY24 Profit" might not be a contradiction (explained by context/time period).
- The choice of FastAPI + Streamlit demonstrates backend robustness while shipping the frontend fast, simulating how an AI startup needs to operate.
