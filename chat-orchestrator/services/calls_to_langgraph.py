# chat-orchestrator/services/calls_to_langgraph.py
import httpx
from typing import Dict, Any, Optional
from config import LANGGRAPH_SERVICE_URL, API_KEY

class LangGraphClient:
    """
    Lightweight HTTP client for talking to LangGraph service.
    Use orchestrator (WebSocketManager) to send the returned events to WS clients
    and to persist memory if present in the returned state.
    """

    def __init__(self, timeout: float = 60.0):
        self._client = httpx.AsyncClient(timeout=timeout, headers={"x-api-key": API_KEY})

    async def run_graph(self, session_id: str, message: Optional[str] = None,
                        file_meta: Optional[Dict[str, Any]] = None,
                        history: Optional[list] = None, msg_type: str = "user_message") -> Dict[str, Any]:
        """
        Call LangGraph service /run_graph and return the parsed JSON response.
        """
        payload = {
            "session_id": session_id,
            "type": msg_type,
            "message": message,
            "file_meta": file_meta,
            "history": history or []
        }
        resp = await self._client.post(f"{LANGGRAPH_SERVICE_URL}/run_graph", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._client.aclose()
