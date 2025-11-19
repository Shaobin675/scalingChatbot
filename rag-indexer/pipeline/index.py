# rag-indexer/pipeline/index.py
import os
import math
import time
import uuid
import pinecone
from typing import List, Dict, Any
from config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME, PINECONE_NAMESPACE
from langchain_openai import OpenAIEmbeddings  

BATCH_SIZE = 100  # tune: 100-500 works well

class PineconeIndexer:
    def __init__(self, embedding_model=None):
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set in environment/config")

        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        self.index_name = PINECONE_INDEX_NAME
        self.namespace = PINECONE_NAMESPACE or None
        if self.index_name not in pinecone.list_indexes():
            # Create index with dimension derived from embeddings
            embedder = embedding_model or OpenAIEmbeddings()
            # get embedding vector dim by a single call
            v = embedder.embed_query("test")
            dim = len(v)
            pinecone.create_index(self.index_name, dimension=dim)
        self.index = pinecone.Index(self.index_name)
        self.embedder = embedding_model or OpenAIEmbeddings()

    def _make_id(self, source_id: str, chunk_id: int) -> str:
        # deterministic id for easy update/delete
        return f"{source_id}~{chunk_id}"

    def upsert_documents(self, docs: List[Dict[str, Any]]):
        """
        docs: list of dicts with keys:
          - id (unique source id, e.g. filename or generated uuid)
          - text (chunk text)
          - metadata (dict)
        """
        vectors = []
        texts = [d["text"] for d in docs]
        # compute embeddings in batches
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            embeddings = self.embedder.embed_documents(batch_texts)  # sync; if async, adapt
            for j, emb in enumerate(embeddings):
                d = docs[i+j]
                chunk_id = d.get("chunk_id", j)
                vec_id = self._make_id(d["id"], chunk_id)
                metadata = d.get("metadata", {})
                metadata.update({"source_id": d["id"], "chunk_id": chunk_id})
                vectors.append((vec_id, emb, metadata))

            # upsert this batch
            self.index.upsert(vectors=vectors, namespace=self.namespace)
            vectors = []

    def query(self, query_text: str, top_k: int = 5):
        # embed query
        q_emb = self.embedder.embed_query(query_text)
        result = self.index.query(vector=q_emb, top_k=top_k, include_metadata=True, namespace=self.namespace)
        # result['matches'] is list of dicts with 'id', 'score', 'metadata'
        return result

    def delete_namespace(self):
        # delete all vectors in namespace
        if self.namespace:
            self.index.delete(delete_all=True, namespace=self.namespace)
        else:
            self.index.delete(delete_all=True)
