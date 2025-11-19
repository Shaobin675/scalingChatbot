# chat-orchestrator/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables (example, override in docker-compose)
ENV POSTGRES_DSN=postgresql://postgres:topsecret@postgres:5432/chatdb
ENV REDIS_URL=redis://redis:6379/0
ENV LANGGRAPH_SERVICE_URL=http://langgraph-service:8001
ENV RAG_SERVICE_URL=http://rag-service:8002
ENV EMBEDDING_SERVICE_URL=http://embedding-service:8003
ENV API_KEY=topsecret

CMD ["uvicorn", "main:server.app", "--host", "0.0.0.0", "--port", "8000"]
