# langgraph-service/main.py
import asyncio
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from nodes.langgraph_nodes import LangGraphNodes
from config import LANGGRAPH_API_KEY

app = FastAPI(title="LangGraph Service")

class RunGraphRequest(BaseModel):
    session_id: str
    type: str  # "user_message" or "file_uploaded"
    message: Optional[str] = None
    file_meta: Optional[Dict[str, Any]] = None
    history: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None

class RunGraphResponse(BaseModel):
    events: List[str]
    llm_output: Optional[str] = None
    state: Optional[Dict[str, Any]] = None

def auth_check(request: Request):
    api_key = request.headers.get("x-api-key")
    if not api_key or api_key != LANGGRAPH_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/run_graph", response_model=RunGraphResponse)
async def run_graph(req: RunGraphRequest, request: Request):
    """
    LangGraph orchestration endpoint.
    Expects header: x-api-key
    """
    auth_check(request)

    # Build initial state (you can extend as needed)
    state = {
        "session_id": req.session_id,
        "type": req.type,
        "user_message": req.message or "",
        "file_meta": req.file_meta,
        "history": req.history or [],
        "events": [],
        "llm_output": None,
        "use_rag": False,
        "rag_answer": "",
        "confidence": 0.0,
        "summary": ""
    }

    nodes = LangGraphNodes()  # creates httpx clients internally
    try:
        result_state = await nodes.run_graph(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangGraph run failed: {e}")

    return RunGraphResponse(
        events=result_state.get("events", []),
        llm_output=result_state.get("llm_output"),
        state=result_state
    )
