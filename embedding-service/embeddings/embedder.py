# embedding-service/embeddings/embedder.py
import os
import base64
import asyncio
from typing import Dict, Any

# Try to use OpenAI via openai package if available, otherwise provide a mock
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

class Embedder:
    def __init__(self):
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            openai.api_key = OPENAI_API_KEY

    async def embed_text(self, text: str):
        """
        Compute and return embedding vector as a list.
        Uses OpenAI if available, else returns a simple hash-based vector (fallback).
        """
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            # using text-embedding-3-small or whichever you prefer
            resp = await asyncio.to_thread(
                lambda: openai.Embedding.create(input=text, model="text-embedding-3-small")
            )
            return resp["data"][0]["embedding"]
        # fallback: simple deterministic pseudo-embedding
        vec = [float(ord(c) % 97) for c in text[:256]]
        return vec

    async def embed_file(self, payload: Dict[str, Any]):
        """
        Decode base64 file and compute document-level embedding.
        Stores nothing here — rag-indexer will keep content/embeddings.
        """
        data_b64 = payload.get("data", "")
        filename = payload.get("filename", "uploaded")
        if not data_b64:
            return {"status": "error", "detail": "no data"}

        try:
            content = base64.b64decode(data_b64)
            # best-effort: decode as text
            try:
                text = content.decode("utf-8")
            except Exception:
                text = str(content[:1000])

            vec = await self.embed_text(text)
            return {"status": "ok", "filename": filename, "embedding_len": len(vec)}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    async def fallback_llm(self, prompt: str) -> str:
        """
        A simple wrapper to call the LLM if available. Returns generated text.
        """
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            try:
                resp = await asyncio.to_thread(
                    lambda: openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512
                    )
                )
                # depending on API response format:
                return resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                return f"⚠️ Fallback LLM error: {e}"
        # fallback deterministic reply
        return f"(fallback) Could not generate answer for prompt: {prompt[:200]}"
