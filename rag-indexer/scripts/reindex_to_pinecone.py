# rag-service/scripts/reindex_to_pinecone.py
import asyncio
import os
from pipeline.loader import load_documents_from_file
from pipeline.index_build import PineconeIndexer

async def reindex_folder(folder_path: str):
    indexer = PineconeIndexer()
    for fname in os.listdir(folder_path):
        path = os.path.join(folder_path, fname)
        docs = await load_documents_from_file(path)
        chunks = []
        for d in docs:
            text = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
            chunk_size = 1000
            for i in range(0, len(text), chunk_size):
                chunks.append({
                    "id": fname,
                    "chunk_id": i // chunk_size,
                    "text": text[i:i+chunk_size],
                    "metadata": {"filename": fname}
                })
        indexer.upsert_documents(chunks)
        print(f"Indexed {len(chunks)} chunks from {fname}")

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "/data/uploads"
    asyncio.run(reindex_folder(folder))
