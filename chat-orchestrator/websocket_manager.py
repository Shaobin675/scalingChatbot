# chat-orchestrator/websocket_manager.py
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from services.calls_to_langgraph import LangGraphClient
from rw_lock import RWLock

class WebSocketManager:
    IDLE_TIMEOUT = timedelta(seconds=180)
    WARNING_SECONDS = 30
    WARNING_INTERVAL = 5

    def __init__(self, db=None):
        self.db = db
        self.langgraph = LangGraphClient()
        self.active_connections: Dict[str, WebSocket] = {}
        self.last_active: Dict[str, datetime] = {}
        self._last_warning_sent: Dict[str, int] = {}
        self._lock = RWLock()

    def setup_routes(self, app: FastAPI):
        @app.websocket("/ws/{session_id}")
        async def ws_endpoint(ws: WebSocket, session_id: str):
            await ws.accept()
            self.active_connections[session_id] = ws
            self.last_active[session_id] = datetime.now(timezone.utc)

            try:
                while True:
                    data = await ws.receive_text()
                    self.last_active[session_id] = datetime.now(timezone.utc)
                    self._last_warning_sent.pop(session_id, None)

                    # Expect JSON payloads for structured actions
                    try:
                        payload = json.loads(data)
                        p_type = payload.get("type")
                    except Exception:
                        payload = None
                        p_type = "user_message"

                    # Acquire write lock for session to ensure sequential processing
                    await self._lock.acquire_write()
                    try:
                        # ---------------- FILE UPLOAD ----------------
                        if p_type == "file_upload":
                            await ws.send_text("📁 Received file, forwarding to LangGraph...")
                            lg_resp = await self.langgraph.run_graph(
                                session_id=session_id,
                                file_meta=payload,
                                msg_type="file_uploaded"
                            )
                        # ---------------- USER MESSAGE ----------------
                        else:
                            message = payload.get("message") if payload else data
                            # persist user message
                            if self.db:
                                await self.db.insert_chat(session_id, message, "User")

                            lg_resp = await self.langgraph.run_graph(
                                session_id=session_id,
                                message=message,
                                msg_type="user_message"
                            )

                        # Forward LangGraph events
                        for ev in lg_resp.get("events", []):
                            await ws.send_text(ev)

                        output = lg_resp.get("llm_output")
                        if output:
                            if self.db:
                                await self.db.insert_chat(session_id, output, "Bot")
                            await ws.send_text(output)

                    finally:
                        self._lock.release_write()

            except WebSocketDisconnect:
                self._cleanup_session(session_id)
            except Exception as e:
                try:
                    await ws.send_text(f"⚠️ Connection error: {e}")
                except Exception:
                    pass
                self._cleanup_session(session_id)

    async def monitor_idle_sessions(self):
        while True:
            now = datetime.now(timezone.utc)
            for sid, ws in list(self.active_connections.items()):
                idle = now - self.last_active.get(sid, now)
                secs = int((self.IDLE_TIMEOUT - idle).total_seconds())
                if secs <= 0:
                    try: await ws.close()
                    except Exception: pass
                    self._cleanup_session(sid)
                elif secs <= self.WARNING_SECONDS:
                    last_warned = self._last_warning_sent.get(sid, -999)
                    if secs % self.WARNING_INTERVAL == 0 and secs != last_warned:
                        try:
                            await ws.send_text(f"⚠️ Idle timeout in {secs} seconds")
                        except Exception: pass
                        self._last_warning_sent[sid] = secs
            await asyncio.sleep(1)

    def _cleanup_session(self, session_id: str):
        self.active_connections.pop(session_id, None)
        self.last_active.pop(session_id, None)
        self._last_warning_sent.pop(session_id, None)
