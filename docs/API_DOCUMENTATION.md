# API Documentation

Base URL (local): `http://127.0.0.1:8000`  
Interactive Swagger UI: `http://127.0.0.1:8000/docs`  
ReDoc: `http://127.0.0.1:8000/redoc`  
Docker Compose backend: `http://backend:8000`

All responses are JSON. Errors use a consistent envelope:

```json
{
  "error": "validation_error",
  "message": "Human-readable message",
  "details": {}
}
```

## 1. Feedback

### `POST /feedback/upload`
Upload and process a CSV file.

**Request:** `multipart/form-data`, field `file`.  
**Query:** `source` (default `csv`).

**Response (201):**
```json
{
  "accepted": 75,
  "rejected": 0,
  "total": 75,
  "message": "Processed 75 feedback record(s).",
  "errors": []
}
```

Pipeline: clean → deduplicate → sentiment → topic → embed → persist (SQLite +
vector index).

### `GET /feedback`
List feedback with filters and pagination.

**Query:** `page`, `page_size`, `sentiment`, `topic`, `channel`, `source`, `product`.

**Response:**
```json
{
  "records": [...],
  "total": 75,
  "page": 1,
  "page_size": 20
}
```

### `GET /feedback/{feedback_id}`
Get one record by ID. `404` if not found.

### `DELETE /feedback/{feedback_id}`
Delete one record. `204` on success, `404` if not found.

## 2. Semantic Search

### `POST /feedback/search`
Retrieve top-K feedback most similar to a natural-language query.

**Body:**
```json
{ "query": "billing is slow", "top_k": 5 }
```

**Response:**
```json
{
  "query": "billing is slow",
  "top_k": 5,
  "results": [
    { "feedback": { ... }, "score": 0.637 }
  ]
}
```

## 3. RAG Question Answering

### `POST /rag/ask`  (alias: `POST /rag/chat`)
Ask a natural-language question. Retrieves relevant feedback, builds a grounded
prompt, and answers via Ollama.

**Body:**
```json
{ "question": "What are customers complaining about?", "top_k": 5 }
```

**Response:**
```json
{
  "question": "...",
  "answer": "...",
  "sources": [
    { "id": "...", "feedback_text": "...", "sentiment": "negative", "score": 0.63 }
  ]
}
```

Answers are grounded in retrieved feedback; if evidence is insufficient the
model states so.

## 4. Analytics

### `GET /analytics/summary`
Total, average rating, sentiment/topic/source/product/rating distributions.

### `GET /analytics/overview`
High-level overview: totals, average rating, top topics.

### `GET /analytics/sentiment`
Sentiment distribution with percentages.

### `GET /analytics/topics`
Topic distribution with percentages.

### `GET /analytics/trends`
Feedback volume trend over N days (default 30), bucketed by day.

## 5. Evaluation

### `POST /evaluation/run`
Run the RAG evaluation suite against `data/evaluation_dataset.jsonl`.

**Query:** `top_k` (default 5).

**Response:** aggregate Precision@K, Recall@K, MRR, and answer-quality metrics,
plus per-case results.

## 6. Health

### `GET /health`
```json
{
  "status": "ok",
  "version": "0.1.0",
  "ollama_configured": true,
  "ollama_ready": true,
  "vector_backend": "numpy",
  "database_backend": "sqlite"
}
```

### `GET /health/ollama`
Ollama readiness and model configuration.
