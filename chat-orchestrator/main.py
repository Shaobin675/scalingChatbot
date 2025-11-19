import os
import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from websocket_manager import WebSocketManager
from db_postgres import AsyncPostgresDB
from rag_engine import RAGEngine
from config import POSTGRES_DSN, REDIS_URL
import uvicorn

class AppServer:
    def __init__(self, db_dsn=None, redis_url=REDIS_URL):
        self.app = FastAPI(lifespan=self._lifespan)
        self.db = AsyncPostgresDB(dsn=db_dsn or POSTGRES_DSN)
        self.rag = RAGEngine(redis_url=redis_url)
        self.websocket_manager = WebSocketManager(db=self.db, rag=self.rag)
        self.websocket_manager.setup_routes(self.app)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        await self.db.init_db()
        try:
            await self.rag.init_redis()
            await self.rag.load_index()
        except FileNotFoundError:
            print("⚠️ No existing RAG index found.")
        asyncio.create_task(self.websocket_manager.monitor_idle_sessions())
        yield
        await self.db.close()

    def run(self, host="0.0.0.0", port=8000):
        
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    server = AppServer(db_dsn=POSTGRES_DSN)
    server.run()
