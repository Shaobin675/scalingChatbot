# rag-indexer/main.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from pipeline.loader import load_pdf_or_text
from pipeline.splitter import split_docs
from pipeline.index_build import pinecone_upsert
from pipeline.search import pinecone_search, pinecone_retrieve

app = FastAPI(title="RAG Indexer Service (Pinecone)")

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
@app.post("/upsert")
async def upsert_file(file: UploadFile = File(...)):
    """
    Upload a PDF/TXT file -> extract text -> split -> embed -> upsert to Pinecone
    """
    try:
        raw_text = await load_pdf_or_text(file)
        docs = split_docs(raw_text)

        # Push chunks to Pinecone
        upsert_result = await pinecone_upsert(docs)

        return {
            "status": "ok",
            "chunks_indexed": len(docs),
            "index_response": upsert_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


@app.post("/search")
async def search(req: SearchRequest):
    """
    Standard semantic similarity search.
    """
    try:
        results = await pinecone_search(req.query, top_k=req.top_k)
        return { "matches": results }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    """
    Retrieve exact vector items by ID.
    Used by LangGraph for confirmatory lookups,
    or when your application stores known chunk IDs.
    """
    try:
        records = await pinecone_retrieve(req.ids)
        return { "records": records }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieve failed: {e}")


@app.get("/health")
async def health():
    return { "status": "ok" }


