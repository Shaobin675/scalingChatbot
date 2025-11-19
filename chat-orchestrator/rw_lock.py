# rw_lock.py
# Async Reader–Writer Lock for RAG & LangGraph concurrency

import asyncio
from contextlib import asynccontextmanager


class RWLock:
    """Efficient async Reader/Writer lock.
    - Many readers allowed concurrently.
    - Writers get exclusive access.
    """
    def __init__(self):
        self._rlock = asyncio.Lock()
        self._wlock = asyncio.Lock()
        self._readers = 0

    async def acquire_read(self):
        async with self._rlock:
            self._readers += 1
            if self._readers == 1:
                await self._wlock.acquire()

    async def release_read(self):
        async with self._rlock:
            self._readers -= 1
            if self._readers == 0 and self._wlock.locked():
                self._wlock.release()

    async def acquire_write(self):
        await self._wlock.acquire()

    def release_write(self):
        if self._wlock.locked():
            self._wlock.release()

