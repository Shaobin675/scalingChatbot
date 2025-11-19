# rag-service/pipeline/search.py
from .index_build import PineconeIndexer
from typing import List, Dict, Any
from config import API_KEY

indexer = PineconeIndexer()

async def pinecone_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Async wrapper for Pinecone query; run blocking call in thread if necessary.
    """
    loop = __import__("asyncio").get_event_loop()
    res = await loop.run_in_executor(None, indexer.query, query, top_k)
    matches = res.get("matches", []) if isinstance(res, dict) else []
    formatted = []
    for m in matches:
        formatted.append({
            "id": m.get("id"),
            "score": m.get("score"),
            "metadata": m.get("metadata", {}),
            "text": m.get("metadata", {}).get("snippet", "")
        })
    return formatted

async def pinecone_retrieve(ids: List[str]):
    loop = __import__("asyncio").get_event_loop()
    res = await loop.run_in_executor(None, indexer.retrieve_by_ids, ids)
    return res
