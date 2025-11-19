# langgraph-service/nodes/langgraph_nodes.py
import asyncio
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from config import RAG_SERVICE_URL, EMBEDDING_SERVICE_URL, OUTBOUND_API_KEY, HTTPX_TIMEOUT
import httpx

class LangGraphNodes:
    """
    Nodes for LangGraph orchestration.
    Each node calls RAG or Embedding microservices as needed via httpx.
    Events are appended to state['events'] for orchestrator to forward to WebSocket.
    """

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=HTTPX_TIMEOUT, headers={"x-api-key": OUTBOUND_API_KEY})

    async def retrieve_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call RAG /search for relevant docs and store rag_answer/confidence.
        """
        query = state.get("user_message", "")
        try:
            resp = await self._client.post(f"{RAG_SERVICE_URL}/search", json={"query": query, "top_k": 5})
            resp.raise_for_status()
            data = resp.json()
            # data['matches'] expected shape from RAG service
            matches = data.get("matches") or data.get("results") or []
            # combine page content if present into rag_answer
            context = "\n\n".join([m.get("text", m.get("metadata", {}).get("text", "")) for m in matches])
            state["retrieved_docs"] = matches
            state["rag_answer"] = context
            # crude confidence – average of scores if provided
            scores = [m.get("score", 0.0) for m in matches]
            state["confidence"] = round(sum(scores) / len(scores), 3) if scores else 0.0
            state.setdefault("events", []).append("WS:retrieval:done")
        except Exception as e:
            state.setdefault("events", []).append(f"WS:retrieval:error:{e}")
            state["rag_answer"] = ""
            state["confidence"] = 0.0
        return state

    async def decide_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide whether to use RAG or fallback based on state['confidence'].
        """
        use_rag = bool(state.get("rag_answer")) and float(state.get("confidence", 0)) >= 0.35
        state["use_rag"] = use_rag
        state.setdefault("events", []).append("WS:rag:using" if use_rag else "WS:fallback:using")
        return state

    async def rag_generate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answer using RAG context by calling embedding service's llm_rag or fallback endpoint.
        """
        context = state.get("rag_answer", "")
        user_msg = state.get("user_message", "")
        prompt = f"Context:\n{context}\n\nQuestion: {user_msg}\nAnswer (use only the context):"

        try:
            # Try a dedicated RAG LLM endpoint first
            resp = await self._client.post(f"{EMBEDDING_SERVICE_URL}/llm_rag", json={"prompt": prompt})
            resp.raise_for_status()
            out = resp.json().get("llm_output") or resp.json().get("output") or resp.json().get("text")
            state["llm_output"] = out or "⚠️ RAG generation returned empty."
            state.setdefault("events", []).append("WS:generated:rag")
        except Exception as e:
            # fallback to fallback_llm if error
            try:
                resp2 = await self._client.post(f"{EMBEDDING_SERVICE_URL}/fallback_llm", json={"prompt": prompt})
                resp2.raise_for_status()
                out2 = resp2.json().get("output") or resp2.json().get("text")
                state["llm_output"] = out2 or "⚠️ RAG fallback returned empty."
                state.setdefault("events", []).append("WS:generated:rag-fallback")
            except Exception as ex2:
                state["llm_output"] = f"⚠️ RAG generation failed: {e} / {ex2}"
                state.setdefault("events", []).append(f"WS:generated:error:{e}")

        return state

    async def fallback_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a reply using the fallback LLM (embedding service).
        """
        user_msg = state.get("user_message", "")
        short_context = (state.get("summary") or (state.get("rag_answer")[:300] if state.get("rag_answer") else ""))
        prompt = f"{short_context}\nUser: {user_msg}\nRespond conversationally."

        try:
            resp = await self._client.post(f"{EMBEDDING_SERVICE_URL}/fallback_llm", json={"prompt": prompt})
            resp.raise_for_status()
            out = resp.json().get("output") or resp.json().get("text")
            state["llm_output"] = out or "⚠️ Fallback LLM returned empty."
            state.setdefault("events", []).append("WS:generated:fallback")
        except Exception as e:
            state["llm_output"] = f"⚠️ Fallback LLM error: {e}"
            state.setdefault("events", []).append(f"WS:generated:error:{e}")
        return state

    async def memory_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return memory payload for orchestrator to persist.
        """
        reply = state.get("llm_output", "")
        if reply:
            state.setdefault("events", []).append("WS:memory:ready")
            state.setdefault("memory", {})["bot_reply"] = reply
        return state

    async def run_graph(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile and run a simple LangGraph StateGraph using the nodes above.
        """
        graph = StateGraph(dict)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("decide", self.decide_node)
        graph.add_node("rag_generate", self.rag_generate_node)
        graph.add_node("fallback", self.fallback_node)
        graph.add_node("memory", self.memory_node)

        graph.add_edge("retrieve", "decide")
        graph.add_conditional_edges("decide", lambda s: "rag_generate" if s.get("use_rag") else "fallback")
        graph.add_edge("rag_generate", "memory")
        graph.add_edge("fallback", "memory")
        graph.add_edge("memory", END)

        graph.set_entry_point("retrieve")
        compiled = graph.compile()

        final_state = await compiled.ainvoke(initial_state)
        final_state.setdefault("events", final_state.get("events", []))
        return final_state

    async def close(self):
        await self._client.aclose()
