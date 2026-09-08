import os
import re
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
import uvicorn

from core.parser import classify_provider_error, fact_layer

app = FastAPI(title="Fact Knowledge Layer API")
_jobs: dict[str, dict] = {}
_jobs_lock = threading.RLock()
_source_files: dict[str, str] = {}
_source_lock = threading.RLock()
_source_dir = Path(tempfile.mkdtemp(prefix="fact-layer-sources-"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def _set_job(job_id: str, **updates):
    with _jobs_lock:
        _jobs[job_id].update(updates, updated_at=datetime.now(timezone.utc).isoformat())


def _process_job(job_id: str, filepath: str, filename: str):
    _set_job(job_id, status="processing")
    try:
        def update_progress(completed, total):
            _set_job(
                job_id,
                chunks_completed=completed,
                chunks_total=total,
                provider_calls_completed=completed,
                provider_calls_total=total,
                progress=round(completed / total * 100) if total else 100,
            )

        result = fact_layer.process_document(filepath, filename, progress_callback=update_progress)
        _set_job(job_id, status=result["status"], result=result, error=None)
    except Exception as exc:
        error = classify_provider_error(exc)
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        if not status:
            match = re.search(r"\b([45]\d{2})\b", str(exc))
            status = int(match.group(1)) if match else None
        error.update(
            {
                "provider": fact_layer.provider,
                "model": fact_layer.model,
                "status": status or "unknown",
            }
        )
        _set_job(
            job_id,
            status="failed",
            error=error,
            result={
                "status": "failed",
                "errors": [error],
                "provider_calls_total": None,
                "provider_calls_succeeded": 0,
                "provider_calls_failed": 1,
                "quota": error if error["code"] == "provider_quota" else None,
            },
        )


async def _queue_upload(background_tasks: BackgroundTasks, file: UploadFile) -> dict:
    filename = file.filename or "uploaded-document"
    suffix = os.path.splitext(filename)[1]
    if suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only PDF and plain-text documents are supported")
    quota = fact_layer.quota_status()
    if quota:
        raise HTTPException(status_code=429, detail=quota)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                os.remove(temp_file.name)
                raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
            temp_file.write(chunk)
        temp_filepath = temp_file.name
    source_path = _source_dir / f"{uuid.uuid4()}{suffix.lower()}"
    os.replace(temp_filepath, source_path)
    with _source_lock:
        _source_files[filename] = str(source_path)
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "status": "queued",
            "result": None,
            "error": None,
            "chunks_completed": 0,
            "chunks_total": None,
            "provider_calls_completed": 0,
            "provider_calls_total": None,
            "progress": 0,
            "created_at": now,
            "updated_at": now,
        }
    background_tasks.add_task(_process_job, job_id, str(source_path), filename)
    return {"job_id": job_id, "filename": filename, "status": "queued"}


@app.post("/upload", status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    return await _queue_upload(background_tasks, file)


@app.post("/uploads", status_code=202)
async def upload_documents(
    background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one document is required")
    invalid = [
        file.filename or "uploaded-document"
        for file in files
        if os.path.splitext(file.filename or "")[1].lower() not in ALLOWED_EXTENSIONS
    ]
    if invalid:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {', '.join(invalid)}",
        )
    jobs = [await _queue_upload(background_tasks, file) for file in files]
    return {"jobs": jobs}


@app.get("/upload/{job_id}")
async def get_upload_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


@app.get("/facts")
async def get_facts():
    return {"facts": [fact.model_dump() for fact in fact_layer.get_facts()]}


@app.get("/graph")
async def get_graph():
    return fact_layer.build_graph()


def _source_path(document_name: str) -> Path:
    safe_name = os.path.basename(document_name)
    with _source_lock:
        path = _source_files.get(safe_name)
    if not path or safe_name != document_name:
        raise HTTPException(status_code=404, detail="Source file is not available")
    resolved = Path(path).resolve()
    if resolved.parent != _source_dir.resolve() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Source file is not available")
    return resolved


@app.get("/source-preview")
async def source_preview(
    document_name: str = Query(...),
    page: int = Query(1, ge=1),
):
    path = _source_path(document_name)
    if path.suffix.lower() == ".pdf":
        import pymupdf as fitz

        with fitz.open(path) as document:
            if page > len(document):
                raise HTTPException(status_code=404, detail="Requested source page is unavailable")
            return {
                "document_name": document_name,
                "kind": "pdf",
                "page": page,
                "page_count": len(document),
                "text": document[page - 1].get_text(),
                "url": f"/source-file/{document_name}",
            }
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "document_name": document_name,
        "kind": "text",
        "page": 1,
        "page_count": 1,
        "text": text,
        "url": None,
    }


@app.get("/source-file/{document_name:path}")
async def source_file(document_name: str):
    path = _source_path(document_name)
    media_type = "application/pdf" if path.suffix.lower() == ".pdf" else "text/plain"
    return FileResponse(path, media_type=media_type, filename=document_name)


@app.post("/demo")
async def load_demo():
    """Load deterministic synthetic facts without calling Gemini."""
    result = fact_layer.load_demo()
    return {"status": "success", "demo": True, "result": result}


@app.get("/health")
async def health():
    provider = fact_layer.provider_status()
    return {
        "status": "ok",
        "gemini_configured": provider["provider"] == "gemini" and provider["configured"],
        "max_chunks": fact_layer.max_chunks,
        "effective_max_chunks": fact_layer.max_chunks,
        "max_retries": fact_layer.max_retries,
        "config_warnings": fact_layer.config_warnings,
        **provider,
    }


@app.get("/provider-diagnostics")
async def provider_diagnostics():
    """Expose safe configuration hints; credentials and provider responses stay private."""
    provider = fact_layer.provider_status()
    return {
        **provider,
        "hint": (
            "For Groq, use https://api.groq.com/openai/v1 and a model returned by "
            "GET https://api.groq.com/openai/v1/models."
            if provider["provider"] == "openai_compatible"
            else None
        ),
        "effective_max_input_chars": fact_layer.max_input_chars,
        "max_chunks": fact_layer.max_chunks,
        "max_workers": fact_layer.max_workers,
        "max_input_tokens": fact_layer.max_input_tokens,
        "request_timeout_seconds": fact_layer.request_timeout_seconds,
        "model": fact_layer.model,
    }


@app.post("/provider-smoke-test")
async def provider_smoke_test():
    """Run one tiny provider call; never return keys or raw provider content."""
    return fact_layer.provider_smoke_test()


@app.get("/corroborations")
async def check_corroborations():
    return fact_layer.run_reasoning()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
