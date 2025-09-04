from fastapi import APIRouter, UploadFile
from .schemas import QueryRequest

router = APIRouter()

@router.post("/upload")
def upload_invoice(file: UploadFile):
    return {"filename": file.filename, "status": "uploaded"}

@router.post("/query")
def query_invoice(request: QueryRequest):
    return {"question": request.question, "answer": "Fake answer"}
