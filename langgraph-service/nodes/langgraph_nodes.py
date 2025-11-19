# langgraph-service/nodes/langgraph_nodes.py
import asyncio
from typing import Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END
import httpx
from config import RAG_SERVICE_URL, EMBEDDING_SERVICE_URL, EMBEDDING_FALLBACK_ENABLED

# TIMEOUTS
HTTP_TIMEOUT = 30.0

class LangGraphNodes:
    """
    Implements LangGraph nodes using the LangGraph StateGraph API.
    Each node is an async function that calls RAG / Embedding microservices via HTTP.
    Events (strings) are appended to state['events'] so the orchestrator can forward them to WebSocket clients.
    """

    def __init__(self):
        # Reusable httpx client
        self._client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)

    # -------------------- Node implementations --------------------

    async def retrieve_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call RAG microservice to retrieve relevant context.
        Expects RAG service to expose /retrieve which accepts {"query": ...} and returns {"docs": [...], "rag_answer": "...", "confidence": 0.88}
        """
        query = state.get("user_message", "")
        # if file uploaded, we may set query to filename or file content preview
        if state.get("type") == "file_uploaded" and state.get("file_meta"):
            # file_meta might include text snippet or filename; pass helpful prompt
            query = f"Summarize file: {state['file_meta'].get('filename')}"

        try:
            resp = await self._client.post(f"{RAG_SERVICE_URL}/retrieve", json={"query": query})
            resp.raise_for_status()
            data = resp.json()
            # update state
            state["retrieved_docs"] = data.get("docs", [])
            state["rag_answer"] = data.get("rag_answer", "")
            state["confidence"] = float(data.get("confidence", 0.0))
            # event: notify orchestrator that retrieval finished
            state.setdefault("events", []).append("WS:retrieval:done")
        except Exception as e:
            # retrieval failed — keep state but register the error event
            state.setdefault("events", []).append(f"WS:retrieval:error:{str(e)}")
            state["rag_answer"] = ""
            state["confidence"] = 0.0
        return state

    async def decide_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide whether to use RAG or fallback LLM based on confidence and presence of rag_answer.
        Appends WS:rag:using or WS:fallback:using to state['events'].
        """
        rag_answer = state.get("rag_answer", "")
        confidence = float(state.get("confidence", 0.0))
        use_rag = bool(rag_answer) and (confidence >= 0.4)
        state["use_rag"] = use_rag
        if use_rag:
            state.setdefault("events", []).append("WS:rag:using")
        else:
            state.setdefault("events", []).append("WS:fallback:using")
        return state

    async def rag_generate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a final answer using retrieved context. Calls embedding-service's LLM endpoint
        or a dedicated LLM endpoint if you have one. We call EMBEDDING_SERVICE_URL/fallback_llm
        (or /llm_rag) with a composed prompt.
        """
        context = state.get("rag_answer", "")
        user_msg = state.get("user_message", "")
        prompt = f"Context:\n{context}\n\nQuestion: {user_msg}\nAnswer succinctly using only the context."

        try:
            # prefer a dedicated LLM endpoint for RAG completions if available
            # fallback to embedding-service fallback_llm if no dedicated endpoint
            # try /llm_rag first
            try:
                resp = await self._client.post(f"{EMBEDDING_SERVICE_URL}/llm_rag", json={"prompt": prompt})
                resp.raise_for_status()
                out = resp.json().get("llm_output")
            except httpx.HTTPStatusError:
                # try fallback endpoint
                resp2 = await self._client.post(f"{EMBEDDING_SERVICE_URL}/fallback_llm", json={"prompt": prompt})
                resp2.raise_for_status()
                out = resp2.json().get("output")
            state["llm_output"] = out or "⚠️ RAG generation returned empty."
            state.setdefault("events", []).append("WS:generated:rag")
        except Exception as e:
            state["llm_output"] = f"⚠️ RAG generation failed: {e}"
            state.setdefault("events", []).append(f"WS:generated:error:{str(e)}")
        return state

    async def fallback_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a reply using fallback LLM when RAG is not suitable.
        Calls embedding-service /fallback_llm.
        """
        user_msg = state.get("user_message", "")
        # include short retrieval summary if available (helps LLM)
        short_context = state.get("summary") or (state.get("rag_answer")[:200] if state.get("rag_answer") else "")
        prompt = f"{short_context}\nUser: {user_msg}\nRespond conversationally."

        try:
            resp = await self._client.post(f"{EMBEDDING_SERVICE_URL}/fallback_llm", json={"prompt": prompt})
            resp.raise_for_status()
            out = resp.json().get("output")
            state["llm_output"] = out or "⚠️ Fallback LLM returned empty."
            state.setdefault("events", []).append("WS:generated:fallback")
        except Exception as e:
            state["llm_output"] = f"⚠️ Fallback LLM error: {e}"
            state.setdefault("events", []).append(f"WS:generated:error:{str(e)}")
        return state

    async def memory_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph writes memory back to state (state['memory']) for orchestrator to persist if required.
        We append an event indicating that memory is ready.
        """
        # LangGraph service does not write to Postgres directly in this design.
        # Orchestrator (chat-orchestrator) persists messages; so we just indicate what to store.
        reply = state.get("llm_output", "")
        if reply:
            state.setdefault("events", []).append("WS:memory:ready")
            # also put memory payload in state for orchestrator
            state.setdefault("memory", {}).update({"bot_reply": reply})
        return state

    # -------------------- Graph runner --------------------
    async def run_graph(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build and run a LangGraph StateGraph using the node functions above.
        Returns the final state dict (with events and llm_output).
        """
        # Create the graph
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

        # Execute and return final state
        final_state = await compiled.ainvoke(initial_state)
        # ensure events is present
        final_state.setdefault("events", final_state.get("events", []))
        return final_state

    # -------------------- cleanup --------------------
    async def close(self):
        try:
            await self._client.aclose()
        except Exception:
            pass
