# langgraph-service/config.py
import os

LANGGRAPH_SERVICE_HOST = "0.0.0.0"
LANGGRAPH_SERVICE_PORT = int(os.environ.get("LANGGRAPH_SERVICE_PORT", 8003))

SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "default-langgraph-key")

# Endpoints for RAG + Embedding microservices
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://rag-service:8001")
EMBEDDING_SERVICE_URL = os.environ.get("EMBEDDING_SERVICE_URL", "http://embedding-service:8002")
