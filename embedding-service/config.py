# embedding-service/config.py
import os

EMBEDDING_SERVICE_HOST = "0.0.0.0"
EMBEDDING_SERVICE_PORT = int(os.environ.get("EMBEDDING_SERVICE_PORT", 8002))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "default-embedding-key")

# Optional fallback LLM configuration
FALLBACK_LLM_MODEL = os.environ.get("FALLBACK_LLM_MODEL", "gpt-4.0-mini")
