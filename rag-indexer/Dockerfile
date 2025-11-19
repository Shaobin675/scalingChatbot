# rag-service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8002

ENV REDIS_URL=redis://redis:6379/0
ENV API_KEY=topsecret
ENV PINECONE_API_KEY=GET_YOUR_OWN
ENV PINECONE_ENV=GET_YOUR_OWN
ENV PINECONE_INDEX_NAME=GET_YOUR_OWN

EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
