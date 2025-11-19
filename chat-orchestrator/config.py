# chat-orchestrator/config.py
import os

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN",
    "postgresql+asyncpg://postgres:GET_YOUR_OWN@postgres:5432/chatdb"
)
REDIS_URL = f"redis://{os.environ.get('REDIS_HOST', 'redis')}:{os.environ.get('REDIS_PORT', 6379)}/0"

LANGGRAPH_SERVICE_URL = os.environ.get("LANGGRAPH_SERVICE_URL", "http://langgraph-service:8003")
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "default-orchestrator-key")  # used to call LangGraph
