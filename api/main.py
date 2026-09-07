import os
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
import uvicorn

from core.parser import fact_layer

app = FastAPI(title="Fact Knowledge Layer API")
_jobs: dict[str, dict] = {}
_jobs_lock = threading.RLock()
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
                progress=round(completed / total * 100) if total else 100,
            )

        result = fact_layer.process_document(filepath, filename, progress_callback=update_progress)
        _set_job(job_id, status=result["status"], result=result, error=None)
    except Exception as exc:
        _set_job(job_id, status="failed", error=str(exc), result=None)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


async def _queue_upload(background_tasks: BackgroundTasks, file: UploadFile) -> dict:
    filename = file.filename or "uploaded-document"
    suffix = os.path.splitext(filename)[1]
    if suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only PDF and plain-text documents are supported")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                os.remove(temp_file.name)
                raise HTTPException(status_code=413, detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
            temp_file.write(chunk)
        temp_filepath = temp_file.name
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
            "progress": 0,
            "created_at": now,
            "updated_at": now,
        }
    background_tasks.add_task(_process_job, job_id, temp_filepath, filename)
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


@app.get("/health")
async def health():
    return {"status": "ok", "gemini_configured": fact_layer.client is not None}


@app.get("/corroborations")
async def check_corroborations():
    return fact_layer.run_reasoning()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
