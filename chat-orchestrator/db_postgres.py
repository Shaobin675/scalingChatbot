# chat-orchestrator/db_postgres.py
import asyncio
import os
from typing import List

try:
    import asyncpg
except Exception:
    asyncpg = None

class AsyncPostgresDB:
    """
    Simple async Postgres wrapper. If asyncpg is not available,
    fallback to an in-memory store (for local/dev usage).
    """

    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool = None
        self._in_memory = []  # list of tuples (session_id, text, role, timestamp)

    async def init_db(self):
        if asyncpg:
            self.pool = await asyncpg.create_pool(dsn=self.dsn)
        else:
            # no-op for in-memory fallback
            await asyncio.sleep(0)

    async def close(self):
        if self.pool:
            await self.pool.close()
        else:
            await asyncio.sleep(0)

    async def insert_chat(self, session_id: str, text: str, role: str = "User", timestamp=None):
        timestamp = timestamp or None
        if self.pool:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO chat_logs(session_id, role, message) VALUES($1, $2, $3)",
                    session_id, role, text
                )
        else:
            self._in_memory.append((session_id, role, text))

    async def get_history(self, session_id: str, limit: int = 100) -> List[str]:
        if self.pool:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT role, message FROM chat_logs WHERE session_id=$1 ORDER BY id DESC LIMIT $2",
                    session_id, limit
                )
                return [f"{r['role']}: {r['message']}" for r in reversed(rows)]
        else:
            entries = [t for t in self._in_memory if t[0] == session_id]
            # return last `limit` messages
            return [f"{role}: {text}" for (_sid, role, text) in entries[-limit:]]
