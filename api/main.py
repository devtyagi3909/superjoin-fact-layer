import sys
import os
import shutil
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File
import uvicorn
from typing import List
from core.parser import fact_layer

app = FastAPI(title="Fact Knowledge Layer API")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_filepath = temp_file.name

    try:
        result = fact_layer.process_document(temp_filepath, file.filename)
    finally:
        os.remove(temp_filepath)

    return {"filename": file.filename, "status": result["status"]}

@app.get("/facts")
async def get_facts():
    facts = fact_layer.get_facts()
    return {"facts": [f.model_dump() for f in facts]}

@app.get("/corroborations")
async def check_corroborations():
    return fact_layer.run_reasoning()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
