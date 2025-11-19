from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict, Any
from pipeline.index_build import PineconeIndexer
from pipeline.loader import load_documents_from_file
from pipeline.search import pinecone_search, pinecone_retrieve
from config import API_KEY
import uvicorn
import tempfile
import os
import shutil
import asyncio

app = FastAPI(title="RAG Service (Pinecone)")
indexer = PineconeIndexer()

def auth_check(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

# -----------------------------
# Models
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class RetrieveRequest(BaseModel):
    ids: List[str]


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/upsert", dependencies=[Depends(auth_check)])
async def upsert_file(file: UploadFile = File(...)):
    """
    Upload a file, split into chunks and upsert to Pinecone
    """
    # save to temp file
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        docs = await load_documents_from_file(path)
        # You will need a text splitter; here assume docs list contains page_content
        chunks = []
        for d_idx, d in enumerate(docs):
            text = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
            # naive chunking: split by 1000 chars. Replace with your splitter strategy.
            chunk_size = 1000
            for i in range(0, len(text), chunk_size):
                chunks.append({
                    "id": file.filename,
                    "chunk_id": i // chunk_size,
                    "text": text[i:i+chunk_size],
                    "metadata": {"filename": file.filename}
                })
        # upsert (run in threadpool)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, indexer.upsert_documents, chunks)
        return {"status": "ok", "chunks_indexed": len(chunks)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/search", dependencies=[Depends(auth_check)])
async def search_endpoint(req: SearchRequest, x_api_key: str = Header(...)):
    """
    pinecone_search
    """
    results = await pinecone_search(req.query, top_k=req.top_k)
    return {"matches": results}


@app.post("/retrieve", dependencies=[Depends(auth_check)])
async def retrieve_endpoint(req: RetrieveRequest, x_api_key: str = Header(...)):
    """
    Retrieve exact vector items by ID.
    Used by LangGraph for confirmatory lookups,
    or when your application stores known chunk IDs.
    """
    res = await pinecone_retrieve(req.ids)
    return {"records": res}


@app.get("/health")
async def health():
    return { "status": "ok" }


