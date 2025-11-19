# rag-service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8002

ENV REDIS_URL=redis://redis:6379/0
ENV API_KEY=topsecret

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
