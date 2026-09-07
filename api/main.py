import os
import shutil
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
import uvicorn

from core.parser import fact_layer

app = FastAPI(title="Fact Knowledge Layer API")
_jobs: dict[str, dict] = {}
_jobs_lock = threading.RLock()


def _set_job(job_id: str, **updates):
    with _jobs_lock:
        _jobs[job_id].update(updates, updated_at=datetime.now(timezone.utc).isoformat())


def _process_job(job_id: str, filepath: str, filename: str):
    _set_job(job_id, status="processing")
    try:
        result = fact_layer.process_document(filepath, filename)
        _set_job(job_id, status=result["status"], result=result, error=None)
    except Exception as exc:
        _set_job(job_id, status="failed", error=str(exc))
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.post("/upload", status_code=202)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    filename = file.filename or "uploaded-document"
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
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
            "created_at": now,
            "updated_at": now,
        }
    background_tasks.add_task(_process_job, job_id, temp_filepath, filename)
    return {"job_id": job_id, "filename": filename, "status": "queued"}


@app.get("/upload/{job_id}")
async def get_upload_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(job)


@app.get("/facts")
async def get_facts():
    return {"facts": [fact.model_dump() for fact in fact_layer.get_facts()]}


@app.get("/corroborations")
async def check_corroborations():
    return fact_layer.run_reasoning()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
