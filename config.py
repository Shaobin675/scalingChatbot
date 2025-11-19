import os

# ---------------- URLs for microservices ----------------
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-indexer:8002")
LANGGRAPH_SERVICE_URL = os.getenv("LANGGRAPH_SERVICE_URL", "http://langgraph-service:8003")

# ---------------- Redis ----------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# ---------------- PostgreSQL (used only by Orchestrator) ----------------
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "topsecret")
POSTGRES_DB = os.getenv("POSTGRES_DB", "chatdb")

POSTGRES_DSN = (
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ---------------- LLM / OpenAI ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ----------------- API Keys for inter-service calls -----------------
API_KEY = os.getenv("API_KEY", "GET_YOUR_OWN")  # Default key for local testing

# config.py (add)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "GET_YOUR_OWN_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "")            # e.g. "us-west1-gcp"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-index")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")  # optional

