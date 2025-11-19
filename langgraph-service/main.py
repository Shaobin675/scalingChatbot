# langgraph-service/main.py
import asyncio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any
from nodes.langgraph_nodes import LangGraphNodes
from config import LANGGRAPH_SERVICE_HOST, LANGGRAPH_API_KEY  # Add API key here

app = FastAPI(title="LangGraph Service")

class RunGraphRequest(BaseModel):
    session_id: str
    type: str  # "user_message" or "file_uploaded"
    message: Optional[str] = None
    file_meta: Optional[Dict[str, Any]] = None
    history: Optional[list] = None   # optional chat history passed from orchestrator
    extra: Optional[Dict[str, Any]] = None

class RunGraphResponse(BaseModel):
    events: list
    llm_output: Optional[str] = None
    state: Optional[Dict[str, Any]] = None

def auth_check(request: Request):
    """
    Simple API key auth for inter-service calls.
    """
    api_key = request.headers.get("x-api-key")
    if not api_key or api_key != LANGGRAPH_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.on_event("startup")
async def startup():
    """
    Pre-warm or create a LangGraphNodes instance if needed.
    Currently, we create per-request to avoid cross-request state.
    """
    pass

@app.post("/run_graph", response_model=RunGraphResponse)
async def run_graph(req: RunGraphRequest, request: Request):
    """
    Run the LangGraph orchestration for a given session event.
    Authenticated with API key in headers.
    """
    auth_check(request)  # <-- enforce API key

    nodes = LangGraphNodes()  # lightweight; internal httpx client created inside

    # build initial state
    state = {
        "session_id": req.session_id,
        "type": req.type,
        "user_message": req.message or "",
        "file_meta": req.file_meta,
        "history": req.history or [],
        "events": [],        # collect WS-style events to send back to orchestrator
        "llm_output": None,
        "use_rag": False,
        "rag_answer": "",
        "confidence": 0.0,
        "summary": ""
    }

    try:
        # run the LangGraph compiled graph
        result_state = await nodes.run_graph(state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangGraph run failed: {e}")

    # Compose events and llm_output
    events = result_state.get("events", [])
    llm_output = result_state.get("llm_output", None)
    return RunGraphResponse(events=events, llm_output=llm_output, state=result_state)
