# Architecture Document

Customer Feedback Intelligence Platform — Enterpret-inspired AI-native application.

## 1. Overview

The platform transforms unstructured customer feedback (CSV uploads) into
actionable insights. It cleans and validates feedback, runs sentiment analysis
and topic classification, generates embeddings, stores them in a vector index,
retrieves relevant feedback via semantic search, and answers natural-language
questions using a Retrieval-Augmented Generation (RAG) pipeline.

## 2. Design Goals

| Goal | How it's achieved |
|------|-------------------|
| Modular | One service per feature (preprocessing, sentiment, topic, embeddings, vector store, retriever, RAG, analytics, evaluation). |
| Maintainable | Pydantic schemas, typed exceptions, consistent JSON error responses, structured logging. |
| Testable | 45 Pytest tests; Ollama/FAISS mocked in unit tests; FastAPI TestClient for API tests. |
| Deployable | Docker Compose with backend, frontend, and Ollama services; persistent volumes. |
| Explainable | Rule-based sentiment & topic (lexicon/keywords), not black-box; source citations in RAG answers. |
| No hardcoded secrets | All configuration via environment variables / `.env`. |
| No fabricated metrics | Evaluation metrics derived from a manually labeled dataset. |

## 3. High-Level Data Flow

```
Customer Feedback CSV
        │
        ▼
  FastAPI Backend  (POST /feedback/upload)
        │
        ▼
 PreprocessingService   clean → validate → deduplicate → FeedbackRecord
        │
        ├─▶ SentimentService   (rule-based lexicon, score [-1,1])
        │
        └─▶ TopicService       (keyword buckets → topic)
                │
                ▼
         EmbeddingsService   (Ollama nomic-embed-text → 768-dim vectors)
                │
                ▼
        VectorStore   (FAISS IndexFlatIP on L2-normalised vectors =
                      cosine similarity; NumPy fallback on Python 3.15)
                │
                ▼
          Retriever   (top-K semantic search)
                │
                ▼
              RAG   (Ollama llama3.2:3b → grounded answer + sources)
                │
                ▼
        AnalyticsService   (SQLite aggregations)
                │
                ▼
          Streamlit Dashboard
```

## 4. Layered Architecture

### 4.1 API layer (`backend/api/`)
FastAPI routers with Pydantic request/response validation:
- `feedback.py` — upload, list, get, delete, semantic search
- `analytics.py` — overview, sentiment, topics, trends, summary
- `rag.py` — `/ask` and `/chat` (RAG question answering)
- `evaluation.py` — `/run` (RAG evaluation suite)

### 4.2 Service layer (`backend/services/`)
One class per feature:
- `PreprocessingService` — CSV ingest, cleaning, validation, dedup
- `SentimentService` — polarity scoring
- `TopicService` — keyword topic classification
- `EmbeddingsService` / `OllamaClient` — Ollama embeddings + ping
- `VectorStore` — FAISS index with NumPy cosine fallback, disk persistence
- `RetrieverService` — semantic search wrapper
- `RAGService` / `OllamaProvider` — provider-independent LLM abstraction
- `AnalyticsService` — aggregations
- `EvaluationService` — Precision@K, Recall@K, MRR, answer metrics

### 4.3 Core (`backend/core/`)
- `config.py` — pydantic-settings config from `.env`
- `database.py` — SQLite persistence layer
- `logging.py` — structured logging
- `exceptions.py` — typed `AppError`, `ValidationError`, `NotFoundError`, `ProviderError`

### 4.4 Models (`backend/models/schemas.py`)
Pydantic `FeedbackRecord` and request/response schemas.

### 4.5 Frontend (`frontend/app.py`)
Streamlit dashboard with 5 tabs:
A. Overview — totals, sentiment/topic/source/product distributions, trends
B. Feedback Explorer — search, filters, delete
C. Customer Insights — top topics, sentiment %, volume trends
D. AI Assistant — RAG Q&A with source citations
E. Evaluation — Precision@K, Recall@K, MRR, answer-quality metrics

### 4.6 Data persistence
- **SQLite** (`data/feedback.db`) — durable feedback records with indexes on
  sentiment, topic, product.
- **FAISS / NumPy** (`backend/index/`) — vector index persisted to disk and
  restored on startup.
- **Evaluation dataset** (`data/evaluation_dataset.jsonl`) — manually labeled
  ground truth for metrics.

## 5. Vector Search Backend

FAISS (`IndexFlatIP` on L2-normalised vectors = cosine similarity) is used
when available. On Python 3.15 no `faiss-cpu` wheel is published, so the app
transparently falls back to an exact NumPy cosine-similarity search with an
identical API. This is documented, not hidden.

## 6. LLM / Provider Independence

The RAG service talks to an `LLMProvider` interface. The shipped
implementation is `OllamaProvider` (local `llama3.2:3b`). Selection happens
through environment variables (`OLLAMA_BASE_URL`, `OLLAMA_LLM_MODEL`,
`OLLAMA_EMBED_MODEL`). No API keys are ever hardcoded. If Ollama is
unavailable, endpoints return a clear 502/error instead of crashing.

## 7. Evaluation Framework

Metrics (computed from a manually labeled dataset in
`data/evaluation_dataset.jsonl`, never fabricated):

- Retrieval: Precision@K, Recall@K, Mean Reciprocal Rank (MRR)
- Answer: Faithfulness, Context Relevance, Answer Relevance, Evidence Coverage

Relevance is judged by case-insensitive substring matching of retrieved
feedback text against expected substrings.

## 8. Docker Architecture

Three containers on a dedicated bridge network (`app-network`):

- **ollama** — `ollama/ollama:latest`, port 11434, persistent model volume.
- **backend** — FastAPI, port 8000, connects to Ollama at `http://ollama:11434`.
- **frontend** — Streamlit, port 8501, connects to backend at `http://backend:8000`.

Persistent volumes: `ollama_models`, `feedback_data`, `vector_index`, `uploads`.

## 9. Limitations

- Sentiment and topic classification are rule-based baselines (documented), not
  trained deep-learning models. Replace with transformers/zero-shot as needed.
- In-memory record cache during a server session; durable store is SQLite.
- `faiss-cpu` has no Python 3.15 wheel → NumPy fallback used on 3.15.
- No authentication/authorization (internal/demo use).
- Single Ollama instance handles both embeddings and chat; large CSV uploads
  may be slow (no embedding batching).
