# embedding-service/main.py
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from config import API_KEY
from embedder import Embedder  # your existing logic

app = FastAPI(title="Embedding Service")
embedder = Embedder()

# --------------------- Auth check ---------------------
def auth_check(request: Request):
    api_key = request.headers.get("x-api-key")
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# --------------------- Request models ---------------------
class EmbedRequest(BaseModel):
    content: str

class LLMRequest(BaseModel):
    prompt: str

# --------------------- Endpoints ---------------------
@app.post("/llm_rag")
async def llm_rag(req: LLMRequest, request: Request):
    auth_check(request)
    # This endpoint can be used for RAG-specific completions
    # Here we simply call fallback_llm for demonstration (could be specialized)
    out = embedder.fallback_llm(req.prompt)
    return {"llm_output": out}

@app.post("/fallback_llm")
async def fallback_llm(req: LLMRequest, request: Request):
    auth_check(request)
    out = embedder.fallback_llm(req.prompt)
    return {"output": out}

@app.post("/embed")
async def embed(req: EmbedRequest, request: Request):
    auth_check(request)
    emb = embedder.embed_text(req.content)
    return {"embedding": emb}
