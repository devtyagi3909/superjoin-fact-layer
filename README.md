# Superjoin Fact Knowledge Layer

This project implements a Fact Knowledge Layer for processing financial documents, extracting semantic/numerical facts, and reasoning about their corroboration and contradiction.

## Architecture

*   **API (`api/main.py`)**: FastAPI application exposing endpoints for uploading documents, fetching facts, and triggering the reasoning engine.
*   **UI (`ui/app.py`)**: Streamlit interface to interact with the API.
*   **Core Logic (`core/`)**: Contains the modules for parsing PDFs/Spreadsheets, extracting facts using LLMs, and performing conflict resolution reasoning.

## Setup

1.  Create a virtual environment: `python -m venv venv`
2.  Activate it: `source venv/bin/activate`
3.  Install dependencies: `pip install -r requirements.txt`

## Running

1.  Start the API: `python api/main.py`
2.  Start the UI: `streamlit run ui/app.py`
