# config.py (public-safe, project root)
ORCHESTRATOR_URL = "http://orchestrator:8000"
RAG_SERVICE_URL = "http://rag-service:8001"
EMBEDDING_SERVICE_URL = "http://embedding-service:8002"
LANGGRAPH_SERVICE_URL = "http://langgraph-service:8003"

POSTGRES_HOST = "postgres"
POSTGRES_PORT = 5432
REDIS_HOST = "redis"
REDIS_PORT = 6379

# Idle timeout for frontend
IDLE_TIMEOUT_SECONDS = 180
