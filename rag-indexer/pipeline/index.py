# rag-service/pipeline/index_build.py
import os
import pinecone
import math
from typing import List, Dict, Any
from config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME, PINECONE_NAMESPACE, BATCH_SIZE
from httpx import TimeoutException

# Replace `OpenAIEmbeddings` with your embedder if you have a wrapper
try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    # fallback to OpenAI API directly or custom embedder
    OpenAIEmbeddings = None

class PineconeIndexer:
    def __init__(self, embedder=None):
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set")
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.index_name = PINECONE_INDEX_NAME
        self.namespace = PINECONE_NAMESPACE or None

        # create index if not exists
        if self.index_name not in pinecone.list_indexes():
            if embedder is None and OpenAIEmbeddings is not None:
                embedder_temp = OpenAIEmbeddings()
                dim = len(embedder_temp.embed_query("test"))
            else:
                # fallback dimension, assume 1536 (OpenAI text-embedding-3-small / 1536)
                dim = int(os.getenv("EMBED_DIM", "1536"))
            pinecone.create_index(self.index_name, dimension=dim)
        self.index = pinecone.Index(self.index_name)
        self.embedder = embedder or (OpenAIEmbeddings() if OpenAIEmbeddings is not None else None)

    def _make_id(self, source_id: str, chunk_id: int):
        return f"{source_id}~{chunk_id}"

    def upsert_documents(self, docs: List[Dict[str, Any]]):
        """
        docs: list of dicts: {id: source_id, chunk_id: int, text: str, metadata: dict}
        """
        # embed in batches
        texts = [d["text"] for d in docs]
        all_vectors = []
        # compute embeddings in batches
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            if self.embedder:
                batch_embs = self.embedder.embed_documents(batch_texts)
            else:
                # Placeholder: raise if no embedder
                raise RuntimeError("No embedder configured for PineconeIndexer")
            vectors = []
            for j, emb in enumerate(batch_embs):
                d = docs[i+j]
                vec_id = self._make_id(d["id"], d.get("chunk_id", j))
                metadata = d.get("metadata", {})
                # keep a short snippet in metadata for retrieval convenience
                metadata.setdefault("snippet", d["text"][:500])
                metadata.update({"source_id": d["id"], "chunk_id": d.get("chunk_id", j)})
                vectors.append((vec_id, emb, metadata))
            # upsert this batch
            self.index.upsert(vectors=vectors, namespace=self.namespace)

    def query(self, query_text: str, top_k: int = 5):
        if self.embedder:
            q_emb = self.embedder.embed_query(query_text)
        else:
            raise RuntimeError("No embedder configured for query")
        res = self.index.query(vector=q_emb, top_k=top_k, include_metadata=True, namespace=self.namespace)
        return res

    def retrieve_by_ids(self, ids: List[str]):
        # fetch vectors by ids
        res = self.index.fetch(ids=ids, namespace=self.namespace)
        return res
