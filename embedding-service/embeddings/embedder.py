# embedding-service/embedder.py
import os
import openai
from config import OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL
openai.api_key = OPENAI_API_KEY

class Embedder:
    def __init__(self):
        pass

    def embed_text(self, text: str):
        """
        Synchronous embedding (wraps OpenAI). For large batches, implement batching.
        """
        resp = openai.embeddings.create(model=EMBEDDING_MODEL, input=text)
        emb = resp["data"][0]["embedding"]
        return emb

    def embed_documents(self, texts: list):
        # NOTE: this calls OpenAI in a loop; consider using batch embeddings if available
        return [self.embed_text(t) for t in texts]

    def fallback_llm(self, prompt: str):
        """
        A simple fallback LLM call (synchronous).
        Adjust to your async flow if needed.
        """
        resp = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        # generic extraction
        choices = resp.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""
