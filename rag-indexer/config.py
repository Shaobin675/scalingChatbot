# rag-service/config.py
import os

RAG_SERVICE_HOST = "0.0.0.0"
RAG_SERVICE_PORT = int(os.environ.get("RAG_SERVICE_PORT", 8001))

# Pinecone configuration
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "default-index")

SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "default-rag-key")
