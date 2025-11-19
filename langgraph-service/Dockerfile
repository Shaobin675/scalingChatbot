# langgraph-service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

ENV API_KEY=topsecret 
ENV RAG_SERVICE_URL=http://rag-service:8002
ENV EMBEDDING_SERVICE_URL=http://embedding-service:8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
