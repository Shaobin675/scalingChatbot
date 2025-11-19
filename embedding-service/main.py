# embedding-service/main.py
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from config import API_KEY
from embedder import Embedder  # your existing logic

app = FastAPI(title="Embedding Service")
embedder = Embedder()

# --------------------- Auth check ---------------------
def auth_check(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

# --------------------- Request models ---------------------
class EmbedRequest(BaseModel):
    content: str

class FallbackRequest(BaseModel):
    prompt: str

# --------------------- Endpoints ---------------------
@app.post("/llm_rag", dependencies=[Depends(auth_check)])
async def llm_rag(req: EmbedRequest):
    """
    Generate embeddings or call LLM on content.
    """
    emb = await embedder.embed_content(req.content)
    return {"embedding": emb}

@app.post("/fallback_llm", dependencies=[Depends(auth_check)])
async def fallback_llm(req: FallbackRequest):
    """
    Fallback LLM call (summarization or default processing).
    """
    output = await embedder.fallback(req.prompt)
    return {"output": output}
