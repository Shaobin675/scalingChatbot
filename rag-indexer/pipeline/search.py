# rag-indexer/pipeline/search.py
import numpy as np
from typing import List, Tuple

class SimpleVectorSearch:
    """
    In-memory cosine similarity search for document embeddings.
    """
    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def search(query_embedding: list,
               docs: List[dict],
               top_k: int = 3) -> List[Tuple[dict, float]]:
        """
        Returns list of (doc, score)
        """
        scores = []
        for d in docs:
            score = SimpleVectorSearch.cosine_similarity(query_embedding, d["embedding"])
            scores.append((d, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
