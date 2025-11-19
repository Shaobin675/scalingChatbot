FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV LANGGRAPH_API_KEY=GET_YOUR_OWN
ENV RAG_SERVICE_URL=http://rag-service:8002
ENV EMBEDDING_SERVICE_URL=http://embedding-service:8003
ENV OUTBOUND_API_KEY=GET_YOUR_OWN

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

