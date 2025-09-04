from fastapi import FastAPI, UploadFile
from .routes import router

app = FastAPI(title="Invoice QA API", version="0.1")
app.include_router(router)

@app.get("/")
def root():
    return {"status": "Invoice QA API running"}