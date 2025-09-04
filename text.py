import os
import time
from typing import List, Dict, Any

import torch
from fastapi import FastAPI, BackgroundTasks, File, HTTPException, UploadFile
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

from langchain_huggingface import HuggingFaceEmbeddings  # use pip install -U langchain-huggingface
from functions import create_vector_index, build_cross_encoder_retriever

# Force CPU usage and limit threading for consistent local behavior
setattr(torch, "get_default_device", lambda: "cpu")
torch.set_num_threads(2)

app = FastAPI()

# Globals to track app state
index = None
qa = None
processing = False
current_stage = None
raw_docs_count = 0
processing_start_time: float 
processing_end_time: float 

def process_bg(paths: List[str]):
    """Background job to process uploaded documents and set up the QA system."""
    global index, qa, processing, current_stage, processing_start_time, processing_end_time, raw_docs_count

    processing = True
    processing_start_time = time.time()
    current_stage = "Creating index"

    index, raw_docs_count = create_vector_index(paths)

    current_stage = "Building retriever"
    retriever = build_cross_encoder_retriever(index)

    llm = OllamaLLM(model="llama3", temperature=0.0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    processing_end_time = time.time()
    current_stage = "Complete"
    processing = False

@app.post("/load-docs")
async def load_docs(bg: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload and load files (PDF, CSV, PNG, etc). Starts background processing."""
    paths = []
    for f in files:
        temp_path = f"./tmp_{f.filename}"
        with open(temp_path, "wb") as out:
            out.write(await f.read())
        paths.append(temp_path)

    bg.add_task(process_bg, paths)
    return {"status": "processing", "message": "Indexing started – check /status"}

@app.get("/status")
async def status():
    """Check if system is ready to answer questions."""
    if processing:
        return {"status": "processing", "stage": current_stage}
    if qa is None:
        return {"status": "not_ready"}
    return {"status": "ready", "docs_loaded": raw_docs_count}

@app.post("/ask")
async def ask(payload: Dict[str, Any]):
    """Ask a question and get an answer from the document set."""
    if qa is None:
        raise HTTPException(400, "Index not ready – load documents first")

    query = (payload.get("q") or "").strip()
    if not query:
        raise HTTPException(422, "Missing 'q' field")

    result = qa.invoke({"query": query})
    answer = result.get("result", str(result)).strip()
    return {"q": query, "a": answer}

@app.post("/debug")
async def debug(payload: Dict[str, Any]):
    """Return the top 5 similar document chunks to a given query (no LLM used)."""
    if index is None:
        raise HTTPException(400, "Index not built yet")

    query = payload.get("q", "")
    docs = index.similarity_search(query, k=5)
    return {"chunks": [d.page_content for d in docs]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("text:app", host="0.0.0.0", port=8000, reload=True)
