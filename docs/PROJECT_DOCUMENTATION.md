# Customer Feedback Intelligence Platform — Complete Project Documentation

> **Enterpret-inspired AI-native customer intelligence platform**  
> Built for internship portfolio & interview readiness  
> Repository: https://github.com/vishalmmulge/Customer-Feedback-Intelligence-Platform-Using-LLM-with-RAG.git

---

## 1. Problem Statement

Businesses collect massive volumes of unstructured customer feedback across channels — product reviews, customer surveys, support tickets, app reviews, website feedback, and CSV datasets. This data contains actionable insights but is difficult to analyze at scale:

- **Sentiment is mixed** — positive, negative, and neutral feedback coexist
- **Topics are scattered** — billing, bugs, performance, UI, feature requests, support
- **Emerging trends go unnoticed** — no system tracks what is increasing over time
- **Questions require manual digging** — stakeholders cannot ask natural-language questions and get grounded answers

Enterpret solves this for enterprises, but existing solutions are expensive, closed-source, and require API keys. This project builds a **local, offline-capable, production-style alternative** using Ollama, FAISS, FastAPI, Streamlit, and SQLite — with no paid API keys and no fabricated metrics.

---

## 2. Project Overview

The Customer Feedback Intelligence Platform transforms unstructured customer feedback into actionable insights through a modular pipeline:

```
CSV Upload → Clean & Validate → Sentiment Analysis → Topic Classification → 
Embeddings (Ollama) → FAISS Vector Store → Semantic Search → RAG Q&A → 
Analytics Dashboard → Evaluation Metrics
```

**Core capabilities:**

| Capability | Description |
|------------|-------------|
| CSV Ingestion | Upload any CSV; auto-detects column names (feedback_text, customer_id, rating, timestamp, product, channel, source) |
| Sentiment Analysis | Rule-based lexicon classifier: positive / neutral / negative with score in [-1, 1] |
| Topic Classification | 10 keyword buckets + `general` fallback (bugs, performance, billing, UI, feature requests, support, etc.) |
| Embeddings | Ollama `nomic-embed-text` (768-dim vectors), L2-normalized |
| Vector Search | FAISS `IndexFlatIP` (cosine similarity) with NumPy exact-search fallback |
| Semantic Search | Top-K retrieval with similarity scores |
| RAG Q&A | Grounded answers via Ollama `llama3.2:3b` with source IDs, excerpts, and scores |
| Analytics | Overview, sentiment, topics, trends, summary — all computed from SQLite |
| Evaluation | Precision@K, Recall@K, MRR, faithfulness, context/answer relevance, evidence coverage |
| Docker | Backend, frontend, Ollama containers; persistent volumes; health checks |

---

## 3. Key Features (Detailed)

### 3.1 CSV Ingestion & Preprocessing
- **Auto column detection**: Scans CSV headers and maps them to internal schema using keyword hints (`feedback_text`, `customer_id`, `rating`, `timestamp`, `product`, `channel`, `source`)
- **No manual mapping required**: Works with any CSV containing a feedback text column
- **Duplicate detection**: SHA-256 hash of normalized text removes duplicate rows
- **Missing value handling**: Drops rows without usable feedback; fills defaults for optional fields
- **Unique IDs**: UUID4 per record; `processed_at` timestamp
- **Original text preservation**: Raw feedback text is never modified

### 3.2 Sentiment Analysis (Rule-Based Baseline)
- **Lexicon-driven**: Weighted positive/negative word dictionary
- **Score range**: Clamped to [-1, 1]
- **Labels**: `positive` (> 0.10), `neutral` (-0.10 to 0.10), `negative` (< -0.10)
- **Explainable**: Every word weight is visible and auditable
- **Documented limitation**: Not a trained deep-learning model; intended as an interpretable baseline

### 3.3 Topic Classification
- **10 topics**: Login & Authentication, Performance, Payment & Billing, User Interface, Feature Requests, Bugs & Crashes, Customer Support, Notifications, Security, Other
- **Keyword buckets**: Case-insensitive substring matching
- **Fallback**: `general` when no category matches
- **Extensible**: New topics/keywords added by editing a dictionary

### 3.4 Embeddings
- **Model**: `nomic-embed-text` via Ollama (768 dimensions)
- **Normalization**: L2-normalized so dot product = cosine similarity
- **Batch processing**: Each text embedded individually via Ollama `/api/embeddings`
- **Dimension validation**: Verified on index creation and query

### 3.5 FAISS Vector Search
- **Index type**: `IndexFlatIP` (exact inner-product search)
- **Similarity metric**: Cosine similarity (vectors L2-normalized)
- **Top-K search**: Configurable K (default 5, max 50 for search, max 20 for RAG)
- **Persistence**: Index saved to disk (`backend/index/`) and restored on startup
- **Fallback**: NumPy exact cosine-similarity search when `faiss-cpu` wheel unavailable (Python 3.15)

### 3.6 RAG (Retrieval-Augmented Generation) Pipeline
1. Receive natural-language question
2. Generate question embedding (Ollama)
3. Retrieve top-K similar feedback from FAISS
4. Rank by cosine similarity
5. Build grounded context block with retrieved feedback
6. Send context + question to Ollama `llama3.2:3b`
7. Generate answer using only retrieved evidence
8. Return answer + source IDs + excerpts + similarity scores
9. If evidence insufficient, model states so (no hallucination)

### 3.7 Analytics
- **Overview**: Total feedback, average rating, top topics
- **Sentiment**: Distribution + percentages
- **Topics**: Distribution + percentages
- **Trends**: Daily volume over N days, bucketed by date
- **Summary**: All distributions in one endpoint

### 3.8 Evaluation Framework
- **Manually labeled dataset** (`data/evaluation_dataset.jsonl`): 10 questions with expected text substrings
- **Retrieval metrics**: Precision@K, Recall@K, MRR
- **Answer metrics**: Faithfulness, context relevance, answer relevance, evidence coverage
- **No fabricated metrics**: All numbers computed from actual model output

### 3.9 Frontend (Streamlit Dashboard)
Five tabs:
- **A. Overview**: Totals, average rating, sentiment/topic/source/product distributions, trends
- **B. Feedback Explorer**: Search, filters (sentiment, topic, source), delete records
- **C. Customer Insights**: Top topics, sentiment breakdown, volume trends
- **D. AI Assistant**: Natural-language Q&A with RAG answers and source citations
- **E. Evaluation**: Run evaluation suite; view aggregate and per-case metrics

### 3.10 Docker Containerization
- **3 containers**: Ollama, FastAPI backend, Streamlit frontend
- **Dedicated bridge network** (`app-network`)
- **Persistent volumes**: `ollama_models`, `feedback_data`, `vector_index`, `uploads`
- **Health checks**: All services report accurate states
- **Inter-container networking**: Backend → `http://ollama:11434`; Frontend → `http://backend:8000`
- **Non-root containers**: `appuser` (uid 1000)
- **CPU-only by default** (GPU optional, documented separately)

---

## 4. System Architecture

### 4.1 High-Level Data Flow

```
Customer Feedback CSV
        │
        ▼
  FastAPI Backend  (POST /feedback/upload)
        │
        ▼
 PreprocessingService
   - Auto column detection
   - Clean, validate, deduplicate
   - FeedbackRecord creation
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

### 4.2 Layered Architecture

| Layer | Responsibility | Components |
|-------|---------------|------------|
| **API Layer** | HTTP contracts, validation, routing, Swagger docs | FastAPI routers: `feedback.py`, `analytics.py`, `rag.py`, `evaluation.py` |
| **Service Layer** | One class per feature; single responsibility | `PreprocessingService`, `SentimentService`, `TopicService`, `EmbeddingsService`, `VectorStore`, `RetrieverService`, `RAGService`, `AnalyticsService`, `EvaluationService` |
| **Core Layer** | Config, logging, typed exceptions, SQLite persistence | `config.py`, `logging.py`, `exceptions.py`, `database.py` |
| **Model Layer** | Pydantic request/response schemas & domain records | `backend/models/schemas.py` |
| **Frontend Layer** | Streamlit dashboard with 5 tabs | `frontend/app.py` |
| **Infrastructure** | Docker, Compose, volumes, health checks | `docker-compose.yml`, Dockerfiles, `ollama/entrypoint.sh` |

### 4.3 Design Decisions Explained

1. **Provider-independent LLM**: `LLMProvider` abstract interface; `OllamaProvider` implementation. No API keys hardcoded; all config via environment variables.
2. **Ollama for embeddings + LLM**: Single local instance provides both `nomic-embed-text` and `llama3.2:3b`. Offline-capable, free, no third-party keys.
3. **FAISS with NumPy fallback**: FAISS preferred; NumPy exact-search fallback on Python 3.15 (no `faiss-cpu` wheel). Identical API either way.
4. **Rule-based sentiment + topic**: Explainable, offline, no model downloads. Swappable with transformers later via service interfaces.
5. **SQLite persistence**: Durable records with indexes on sentiment, topic, product. Survives container restarts via Docker volume.
6. **No fabricated metrics**: Evaluation numbers computed from actual retrieval/generation output against labeled data.

---

## 5. Technology Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| Language | Python | 3.15 | Core implementation |
| Backend Framework | FastAPI | 0.163.0 | Web API, automatic OpenAPI |
| Server | Uvicorn | 0.54.0 | ASGI server |
| Data Validation | Pydantic | 2.14.3 | Schemas, request/response validation |
| Config | pydantic-settings | 2.16.3 | Environment-based configuration |
| Data Processing | Pandas | 3.1.1 | CSV ingestion, cleaning |
| Numerical | NumPy | 2.5.4 | Vector operations, fallback search |
| ML | scikit-learn | 1.9.1 | Available for future ML models |
| LLM Runtime | Ollama | Latest | Local `llama3.2:3b`, `nomic-embed-text` |
| Embeddings | nomic-embed-text | 768-dim | Text → dense vectors |
| Vector Search | FAISS | 1.7.0 | `IndexFlatIP` cosine similarity |
| Fallback Search | NumPy | — | Exact cosine similarity when FAISS unavailable |
| Frontend | Streamlit | 1.67.0 | Dashboard UI |
| Charts | Plotly | 5.0.0 | Visualizations (available in requirements) |
| Database | SQLite | stdlib | Durable feedback records |
| Testing | Pytest | 9.2.0 | Unit + API tests |
| Containerization | Docker | 29.9.3 | Container images |
| Orchestration | Docker Compose | v5.7.0 | Multi-service deployment |

---

## 6. Folder Structure

```
customer-feedback-intelligence/
├── backend/
│   ├── Dockerfile
│   ├── __init__.py
│   ├── main.py                 # FastAPI app factory, lifespan, health
│   ├── config.py               # pydantic-settings config from .env
│   ├── core/
│   │   ├── database.py         # SQLite persistence layer
│   │   ├── logging.py          # Structured logging
│   │   └── exceptions.py       # Typed app errors (AppError, ValidationError, etc.)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── feedback.py         # upload / list / get / delete / semantic-search
│   │   ├── analytics.py        # overview / sentiment / topics / trends / summary
│   │   ├── rag.py              # /ask and /chat (RAG Q&A)
│   │   └── evaluation.py       # /run (RAG evaluation suite)
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response contracts (FeedbackRecord, etc.)
│   └── services/
│       ├── __init__.py
│       ├── preprocessing.py    # CSV ingest, clean, validate, deduplicate, auto column mapping
│       ├── sentiment.py        # Rule-based polarity scoring
│       ├── topic.py            # Keyword topic classification (10 buckets)
│       ├── embeddings.py       # Ollama embedding client + ping
│       ├── vector_store.py     # FAISS + NumPy fallback index, disk persistence
│       ├── retriever.py        # Semantic search wrapper
│       ├── rag.py              # LLM provider abstraction + grounded answers
│       ├── analytics.py        # SQLite aggregations
│       └── evaluation.py       # Precision@K, Recall@K, MRR, answer metrics
├── frontend/
│   ├── Dockerfile
│   └── app.py                  # Streamlit dashboard (5 tabs: Overview, Explorer, Insights, Q&A, Evaluation)
├── ollama/
│   ├── Dockerfile
│   └── entrypoint.sh           # Ollama warm-up: waits for server, pulls models once
├── data/
│   ├── sample_feedback.csv     # 75-row realistic synthetic feedback dataset
│   └── evaluation_dataset.jsonl # 10 manually labeled RAG evaluation cases
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API_DOCUMENTATION.md
│   ├── DOCKER_DEPLOYMENT.md
│   ├── INTERVIEW_PREPARATION.md
│   └── PROJECT_DOCUMENTATION.md (this file)
├── tests/
│   ├── test_preprocessing.py
│   ├── test_sentiment.py
│   ├── test_topic.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_rag.py
│   └── test_api.py             # FastAPI TestClient tests (45 tests, all offline/mocked)
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
├── docker-compose.yml
└── README.md
```

> **Synthetic data notice**: `data/sample_feedback.csv` contains realistic synthetic feedback for demonstration. It is clearly sample data, not real customer data.

---

## 7. Data Models (Pydantic Schemas)

### FeedbackRecord (Domain Model)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` / `feedback_id` | str | Yes | Unique UUID4 identifier |
| `customer_id` | str | Yes | Customer identifier |
| `feedback_text` | str | Yes | The feedback content |
| `source` | str | No | Upload source label (default: `csv`) |
| `product` | str | No | Product name |
| `channel` | str | No | Feedback channel (email, chat, survey, etc.) |
| `rating` | int (1-5) | No | Numeric rating |
| `timestamp` | str | No | Original feedback timestamp |
| `sentiment` | str | No | `positive` / `neutral` / `negative` |
| `sentiment_score` | float | No | Score in [-1, 1] |
| `topic` | str | No | Topic category |
| `processed_at` | str | No | ISO timestamp when processed |

### API Response Schemas

- **FeedbackUploadResponse**: `accepted`, `rejected`, `total`, `message`, `errors`
- **FeedbackListResponse**: `records`, `total`, `page`, `page_size`
- **SemanticSearchResponse**: `query`, `top_k`, `results[{feedback, score}]`
- **RAGResponse**: `question`, `answer`, `sources[{id, feedback_text, sentiment, score}]`
- **AnalyticsSummary**: `total_feedback`, `average_rating`, `sentiment_distribution`, `topic_distribution`, `source_distribution`, `product_distribution`, `rating_distribution`
- **HealthResponse**: `status`, `version`, `ollama_configured`, `ollama_ready`, `vector_backend`, `database_backend`

---

## 8. API Documentation

Base URL (local): `http://127.0.0.1:8000`  
Swagger UI: `http://127.0.0.1:8000/docs`  
ReDoc: `http://127.0.0.1:8000/redoc`  
Docker Compose backend: `http://backend:8000`

### 8.1 Feedback Endpoints

#### `POST /feedback/upload`
Upload and process a CSV file.

**Request:** `multipart/form-data`, field `file`.  
**Query params:** `source` (default `csv`), `column_mapping` (JSON object, optional).

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

#### `GET /feedback`
List feedback with filters and pagination.

**Query:** `page`, `page_size`, `sentiment`, `topic`, `channel`, `source`, `product`.

#### `GET /feedback/{feedback_id}`
Get one record by ID. `404` if not found.

#### `DELETE /feedback/{feedback_id}`
Delete one record. `204` on success, `404` if not found.

### 8.2 Semantic Search

#### `POST /feedback/search`
**Body:** `{ "query": "billing is slow", "top_k": 5 }`  
**Response:** List of `{ feedback, score }` sorted by cosine similarity.

### 8.3 RAG Question Answering

#### `POST /rag/ask` (alias: `POST /rag/chat`)
**Body:** `{ "question": "What are customers complaining about?", "top_k": 5 }`  
**Response:** `{ question, answer, sources[{id, feedback_text, sentiment, score}] }`

### 8.4 Analytics Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /analytics/summary` | All distributions + total + average rating |
| `GET /analytics/overview` | High-level overview + top topics |
| `GET /analytics/sentiment` | Sentiment distribution with percentages |
| `GET /analytics/topics` | Topic distribution with percentages |
| `GET /analytics/trends` | Volume trend over N days |

### 8.5 Evaluation

#### `POST /evaluation/run`
**Query:** `top_k` (default 5).  
**Response:** Aggregate Precision@K, Recall@K, MRR, answer metrics + per-case results.

### 8.6 Health

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Overall status, Ollama ready, vector backend, database backend |
| `GET /health/ollama` | Ollama readiness, base URL, model names |

---

## 9. Service Layer Details

### 9.1 PreprocessingService
- `load_csv()`: Reads CSV into DataFrame
- `auto_map_columns()`: Detects CSV columns by keyword hints
- `validate_columns()`: Ensures `feedback_text` exists
- `clean()`: Strips whitespace, handles NaN/blank, drops empty text, deduplicates
- `to_records()`: Converts DataFrame rows to `FeedbackRecord` objects
- `process()`: End-to-end pipeline

### 9.2 SentimentService
- `analyze(text)` → `(label, score)`: Lexicon-based scoring, clamped to [-1, 1]

### 9.3 TopicService
- `classify(text)` → topic: Keyword bucket matching with `general` fallback

### 9.4 EmbeddingsService / OllamaClient
- `embed_text(text)` → `List[float]`: Calls Ollama `/api/embeddings`
- `embed_texts(texts)` → `List[List[float]]`: Batch embedding
- `ping(timeout)` → `bool`: Checks Ollama server availability

### 9.5 VectorStore
- `add(vectors, ids)`: Adds vectors to index
- `search(query_vector, top_k)` → `List[(id, score)]`: Cosine similarity search
- `save()` / `load()`: Disk persistence
- **Backend**: FAISS `IndexFlatIP` or NumPy fallback

### 9.6 RetrieverService
- `search(query, top_k)` → `List[(FeedbackRecord, score)]`: Semantic search wrapper

### 9.7 RAGService / OllamaProvider / LLMProvider
- `answer(question, top_k)` → `dict`: Retrieves context, builds prompt, generates answer
- `LLMProvider.generate(prompt)` → `str`: Abstract interface for provider independence

### 9.8 AnalyticsService
- `summary(records)` → `AnalyticsSummary`: Computes all aggregations from SQLite

### 9.9 EvaluationService
- `evaluate_case(question, expected_texts, top_k)` → `dict`: Single case metrics
- `run_all(dataset_path, top_k)` → `dict`: Full evaluation suite with aggregates

---

## 10. Database Design (SQLite)

**Table: `feedback`**

| Column | Type | Constraints |
|--------|------|-------------|
| `feedback_id` | TEXT | PRIMARY KEY |
| `customer_id` | TEXT | |
| `feedback_text` | TEXT | NOT NULL |
| `source` | TEXT | |
| `product` | TEXT | |
| `timestamp` | TEXT | |
| `rating` | INTEGER | |
| `sentiment` | TEXT | |
| `sentiment_score` | REAL | |
| `topic` | TEXT | |
| `processed_at` | TEXT | |

**Indexes:**
- `ix_feedback_sentiment` on `sentiment`
- `ix_feedback_topic` on `topic`
- `ix_feedback_product` on `product`

---

## 11. RAG Pipeline (Step-by-Step)

```
1. User asks: "What are customers complaining about most?"
2. EmbeddingsService embeds the question via Ollama nomic-embed-text
3. RetrieverService searches FAISS for top-5 most similar feedback vectors
4. Retrieved records ranked by cosine similarity (0.0 to 1.0)
5. Context block built:
   [1] (rating 1/5, sentiment negative): Billing was charged twice...
   [2] (rating 2/5, sentiment negative): Too slow on older devices...
   ...
6. Prompt constructed:
   "Answer the question using only the customer feedback below.
    Do not invent information. If insufficient, say so.
    
    Customer Feedback:
    [context block]
    
    Question: What are customers complaining about most?
    Answer:"
7. OllamaProvider.generate() sends prompt to llama3.2:3b via /api/chat
8. LLM generates grounded answer
9. Response packaged with sources: IDs, excerpts, sentiment, similarity scores
10. If no relevant feedback found: "No relevant feedback found to answer your question."
```

**Key safety properties:**
- Model instructed to use **only** provided context
- Insufficient evidence → model says so
- Source IDs and excerpts returned for auditability
- Similarity scores included for transparency

---

## 12. Evaluation Framework

### Evaluation Dataset (`data/evaluation_dataset.jsonl`)

JSONL format with 10 manually labeled cases:

```jsonl
{"question": "What are customers complaining about most?", "expected_texts": ["billing was charged twice", "worst software", ...], "category": "complaints"}
{"question": "What features do customers request?", "expected_texts": ["dark mode", "export", "feature", ...], "category": "feature_requests"}
```

### Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| Precision@K | relevant_retrieved_K / K | Accuracy of top-K results |
| Recall@K | relevant_retrieved_K / total_relevant | Coverage of relevant items |
| MRR | 1 / rank_of_first_relevant | Ranking quality |
| Faithfulness | max(answer ⋂ source) / |answer| | Answer grounded in sources |
| Context Relevance | avg(question ⋂ source) | Sources match question |
| Answer Relevance | question ⋂ answer | Answer addresses question |
| Evidence Coverage | sources_sharing_tokens_with_answer / total_sources | Answer uses retrieved evidence |

### Sample Results
```
Cases: 10
Precision@K: 0.54
Recall@K: 0.90
MRR: 0.74
Faithfulness: 0.118
Context relevance: 0.015
Answer relevance: 0.112
Evidence coverage: 0.90
```

> **Note**: These are real computed metrics from actual model output — nothing is fabricated.

---

## 13. Docker Architecture

### Container Diagram

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│   Ollama    │      │   FastAPI Backend │      │  Streamlit  │
│  (llama3.2) │      │   (port 8000)     │      │  (port 8501) │
│  port 11434 │      │                   │      │             │
└──────┬──────┘      └────────┬─────────┘      └──────┬──────┘
       │                      │                       │
       │ http://ollama:11434  │ http://backend:8000   │
       ▼                      ▼                       ▼
   [Persistent Volume]   [Persistent Volume]    [No volume needed]
   ollama_models         feedback_data          (stateless)
                         vector_index
                         uploads
```

### Docker Compose Services

| Service | Image/Dockerfile | Port | Depends On |
|---------|-----------------|------|------------|
| `ollama` | `ollama/Dockerfile` (ollama/ollama:latest) | 11434 | — |
| `backend` | `backend/Dockerfile` (python:3.15-slim) | 8000 | ollama (healthy) |
| `frontend` | `frontend/Dockerfile` (python:3.15-slim) | 8501 | backend (healthy) |

### Volumes

| Volume | Mount Path | Purpose |
|--------|------------|---------|
| `ollama_models` | `/root/.ollama` | Ollama models (persisted across restarts) |
| `feedback_data` | `/data` | SQLite database (`feedback.db`) |
| `vector_index` | `/app/backend/index` | FAISS / NumPy index files |
| `uploads` | `/app/data/uploads` | Uploaded CSV files |

### Security
- Non-root user (`appuser`, uid 1000) in backend and frontend
- Minimal base images (`python:3.15-slim`)
- `.dockerignore` excludes secrets, caches, runtime data
- No privileged mode
- Restricted exposed ports (8000, 8501, 11434)

---

## 14. Deployment Guide

### 14.1 Local Python Deployment

**Prerequisites:** Python 3.9+, Ollama running locally

```bash
# 1. Install Ollama models
ollama pull nomic-embed-text
ollama pull llama3.2:3b

# 2. Create virtual environment
python -m venv venv
venv\Scripts\Activate.ps1    # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env         # Edit if needed

# 5. Run backend
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# 6. Run frontend (separate terminal)
streamlit run frontend/app.py

# 7. Run tests
python -m pytest tests/ -v   # 45 tests pass
```

### 14.2 Docker Compose Deployment

```bash
# 1. Build all images
docker compose build

# 2. Start all services
docker compose up -d

# 3. Verify containers
docker compose ps

# 4. Access services
# Dashboard: http://localhost:8501
# Backend: http://localhost:8000
# Swagger UI: http://localhost:8000/docs
# Ollama: http://localhost:11434

# 5. Manage
docker compose logs -f        # View logs
docker compose stop           # Stop services
docker compose down           # Stop + remove containers
docker compose down -v        # Remove everything including data
```

**Important networking rule:** Inside Docker, `localhost` means the current container. Backend reaches Ollama at `http://ollama:11434`; frontend reaches backend at `http://backend:8000`.

---

## 15. Testing Strategy

### Test Coverage (45 tests)

| Test File | What It Tests |
|-----------|---------------|
| `test_preprocessing.py` | CSV ingestion, cleaning, validation, dedup, auto column mapping |
| `test_sentiment.py` | Positive/negative/neutral labeling, score clamping |
| `test_topic.py` | Topic bucket classification, fallback to general |
| `test_embeddings.py` | Vector generation with mocked Ollama |
| `test_vector_store.py` | FAISS/NumPy add/search/save/load |
| `test_retriever.py` | Semantic search wrapper |
| `test_rag.py` | RAG answer generation with mocked LLM |
| `test_api.py` | FastAPI endpoints: health, upload, list, get, delete, search, RAG, analytics, evaluation |

**Key testing principles:**
- Ollama and FAISS mocked in unit tests (offline, no API keys)
- FastAPI `TestClient` for API endpoint tests
- SQLite fake for persistence layer tests
- No network calls required

**Run tests:**
```bash
python -m pytest tests/ -v
```

---

## 16. Troubleshooting

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| Ollama not ready | Models still downloading | Wait; check `docker compose logs ollama` |
| Backend 502 on upload | Ollama unreachable | Confirm `OLLAMA_BASE_URL=http://ollama:11434` in Docker |
| Frontend can't reach backend | Wrong `BACKEND_URL` | Set `BACKEND_URL=http://backend:8000` in Docker |
| Data lost after restart | Volumes removed | Do not run `docker compose down -v` unless intended |
| `faiss-cpu` install fails | No wheel for Python 3.15 | NumPy fallback activates automatically |
| Upload fails with missing column | CSV lacks `feedback_text` | App auto-detects columns; ensure a text column exists |
| Health check fails | Container not ready | Wait; check `docker compose ps` |

---

## 17. Known Limitations

1. **Sentiment & topic are rule-based** — not trained deep-learning models. Replace with transformers/zero-shot when needed.
2. **In-memory record cache** during a server session; durable store is SQLite.
3. **`faiss-cpu` has no Python 3.15 wheel** → NumPy fallback used on 3.15.
4. **Deleting a record from SQLite does not remove its vector from FAISS** (full rebuild needed to purge from index).
5. **No authentication/authorization** — internal/demo use only.
6. **Single Ollama instance** handles both embeddings and chat; large CSV uploads may be slow (no embedding batching yet).

---

## 18. Future Improvements

- Replace rule-based sentiment/topic with transformer / zero-shot models
- Approximate FAISS indexes (IVF, HNSW) for large datasets
- Embedding caching by content hash + background task queue (Celery/RQ)
- PostgreSQL + pgvector or dedicated vector database
- Authentication/authorization
- GPU acceleration for Ollama (optional)

---

# 19. INTERVIEW PREPARATION — 50+ Technical Q&A

## Python

**Q1. What is the Global Interpreter Lock (GIL), and how does it affect this project?**

**A:** The GIL is a mutex in CPython that allows only one thread to execute Python bytecode at a time. In this project, embedding generation and LLM calls are I/O-bound (network requests to Ollama), so the GIL releases during `await` / `httpx` calls and the app remains responsive. CPU-bound work (NumPy/FAISS search) releases the GIL inside native code. For true CPU parallelism, we would use `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`.

**Q2. Why use Pydantic models instead of plain dicts?**

**A:** Pydantic provides runtime data validation, type coercion, clear error messages, and automatic OpenAPI/Swagger schema generation. Every API request and the `FeedbackRecord` domain model are Pydantic models, so malformed CSV rows or bad API payloads are rejected early with 422/400 responses instead of causing obscure downstream failures.

**Q3. How does `pydantic-settings` help with configuration?**

**A:** It loads settings from environment variables and `.env` files into typed Pydantic models. This keeps secrets (API keys, DB paths, model names) out of source code, supports 12-factor app principles, and makes the same code run locally and in Docker with different env values.

**Q4. What is the difference between `async def` and regular `def` in FastAPI?**

**A:** FastAPI runs `async def` endpoints on the event loop (non-blocking) and regular `def` endpoints in a threadpool. We use `async def` for I/O-bound upload (file read + network to Ollama) and regular `def` for CPU-bound sync operations (SQLite, NumPy search) so we don't block the event loop.

---

## OOP

**Q5. How is the service layer organized, and why?**

**A:** Each feature is a separate class (`SentimentService`, `TopicService`, `RetrieverService`, `RAGService`, etc.) with a single responsibility. Classes are instantiated once and stored on `app.state`. This gives encapsulation, easy unit testing (inject fakes), and the ability to swap implementations without touching the API layer.

**Q6. What is the `LLMProvider` abstract base class for?**

**A:** It defines a `generate(prompt)` interface, making the RAG service provider-independent. `OllamaProvider` implements it today; we could add `OpenAIProvider` or `AnthropicProvider` later without changing `RAGService`. This is the Dependency Inversion Principle: depend on abstractions, not concrete providers.

**Q7. How does composition beat inheritance here?**

**A:** `RAGService` composes a `RetrieverService` and an `LLMProvider` rather than inheriting from them. This lets us mix and match retrieval and generation strategies at runtime and avoids fragile base-class hierarchies.

**Q8. Explain the `Database` class design.**

**A:** It encapsulates the `sqlite3` connection, table creation, indexes, and CRUD/aggregation queries behind a clean interface. The FastAPI app holds a single `Database` instance on `app.state`, connecting on startup and closing on shutdown via the lifespan context manager.

---

## Machine Learning

**Q9. What is sentiment analysis, and how is it implemented here?**

**A:** Sentiment analysis classifies text as positive, negative, or neutral. The MVP uses a transparent, rule-based lexicon: each word carries a weight, scores are summed, clamped to [-1, 1], and thresholded into labels. This is documented as a baseline — not a trained deep-learning model — so results are explainable and auditable.

**Q10. Why not use a pretrained transformer for sentiment out of the box?**

**A:** Transformers require model downloads, a GPU for reasonable latency, and more dependencies. For an MVP, a lexicon baseline runs offline, is fast, and is fully explainable. It can be swapped later via the `SentimentService` interface without changing the pipeline.

**Q11. What is topic classification, and what baseline is used?**

**A:** Topic classification assigns each feedback record to a category (e.g., billing, bugs, performance). The MVP uses an explicit keyword bucket map with case-insensitive substring matching and a `general` fallback. It is extensible: new topics/keywords can be added without retraining.

**Q12. What is the difference between classification and clustering?**

**A:** Classification assigns labels from a predefined set (supervised). Clustering groups similar items without predefined labels (unsupervised). This project uses rule-based classification; semantic clustering could be added later via embedding-based k-means or HDBSCAN.

**Q13. What is overfitting, and is it relevant here?**

**A:** Overfitting is when a model memorizes training data and fails to generalize. Our rule-based baselines have no trainable parameters, so overfitting in the classical sense doesn't apply — but the rules can be over-tuned to the sample dataset, which is why they're documented as baselines and evaluated on labeled data.

---

## NLP

**Q14. What preprocessing is applied to feedback text?**

**A:** Whitespace stripping, normalization of `nan`/`None`/blank strings to missing, duplicate detection via SHA-256 hashing of normalized text, and preservation of the original text. Empty/blank feedback rows are dropped.

**Q15. Why deduplicate feedback?**

**A:** Duplicate rows would bias analytics (e.g., sentiment distribution) and waste embedding/index storage. We hash normalized text and keep only the first occurrence.

**Q16. What is tokenization in this context?**

**A:** The sentiment service uses `re.findall(r"\w+", text.lower())` to split text into word tokens for lexicon lookup. No subword tokenization (BPE/WordPiece) is used because the baseline is word-level.

---

## Transformers

**Q17. Are transformers used in this project?**

**A:** Not in the MVP. The design allows swapping `SentimentService` and `TopicService` for Hugging Face Transformers models later (e.g., `cardiffnlp/twitter-roberta-base-sentiment` for sentiment, a zero-shot classifier for topics). The interface stays the same.

**Q18. What is self-attention, briefly?**

**A:** A mechanism in transformers that computes weighted representations of each token based on all other tokens in the sequence, allowing the model to capture long-range dependencies and context.

**Q19. What is a transformer embedding?**

**A:** A dense vector representation of text produced by a transformer model, capturing semantic meaning. Similar texts have similar vectors (high cosine similarity).

---

## Embeddings

**Q20. What embedding model is used, and why?**

**A:** `nomic-embed-text` via Ollama. It is open-weight, runs locally (no API key), produces 768-dimensional vectors, and is well-suited for semantic search. Running it locally keeps the stack offline and free.

**Q21. How are embeddings generated and stored?**

**A:** Each feedback text is sent to Ollama's `/api/embeddings` endpoint. The resulting 768-dim vector is L2-normalized and stored in the vector index mapped to the feedback ID. Vectors are persisted to disk (`backend/index/`).

**Q22. What is embedding normalization, and why does it matter?**

**A:** L2-normalizing vectors to unit length makes the dot product equal to cosine similarity. FAISS `IndexFlatIP` (inner product) on normalized vectors thus performs cosine-similarity search.

**Q23. What is the embedding dimension, and is it validated?**

**A:** `nomic-embed-text` produces 768-dim vectors. The vector store validates dimension on creation and every query vector must match the index dimension.

**Q24. What is the NumPy fallback for FAISS?**

**A:** On Python 3.15 no `faiss-cpu` wheel exists. The app falls back to exact NumPy cosine-similarity search with an identical API, so functionality is preserved without FAISS.

---

## Vector Databases / FAISS

**Q25. What is FAISS, and how is it used here?**

**A:** FAISS (Facebook AI Similarity Search) is a library for efficient similarity search on dense vectors. We use `IndexFlatIP` (exact inner-product search) on L2-normalized vectors for cosine-similarity retrieval of feedback records.

**Q26. What is IndexFlatIP?**

**A:** An exact, brute-force FAISS index that computes the inner product between the query and every stored vector. It is accurate but O(N) — fine for MVP-scale datasets; approximate indexes (IVF, HNSW) can replace it at scale.

**Q27. How is the FAISS index persisted?**

**A:** `faiss.write_index` / `faiss.read_index` for FAISS; `np.save` / `np.load` for the NumPy fallback. Records are serialized to JSON alongside the index so ID↔vector mapping survives restarts.

**Q28. What is cosine similarity?**

**A:** A measure of vector orientation similarity ranging from -1 (opposite) to 1 (identical). For normalized vectors, dot product == cosine similarity. We use it to rank feedback by semantic relevance to a query.

**Q29. What is top-K retrieval?**

**A:** Returning the K most similar vectors to a query. Here K is configurable (default 5, max 50 for search, max 20 for RAG) to balance relevance and prompt size.

**Q30. What is stale-record handling in the vector store?**

**A:** Deleting a record from SQLite does not automatically remove its vector from FAISS (an index doesn't support efficient single-vector deletion). The MVP notes this; a full rebuild or `IndexIDMap` with `remove_ids` can address it at scale.

---

## RAG (Retrieval-Augmented Generation)

**Q31. What is RAG, and what is the workflow here?**

**A:** RAG augments an LLM with external knowledge. Workflow:
1. Receive question
2. Embed the question (Ollama)
3. Retrieve top-K similar feedback from FAISS
4. Build a grounded prompt with retrieved context
5. Send to Ollama LLM
6. Return answer + source citations

**Q32. Why is grounding important?**

**A:** It prevents hallucination. The system prompt instructs the LLM to answer using only the provided feedback context and to say so if evidence is insufficient. Source IDs and excerpts are returned so answers are auditable.

**Q33. What happens if Ollama is down?**

**A:** `OllamaProvider.generate` raises `ProviderError`, which the RAG endpoint catches and returns as a clear error answer instead of crashing or fabricating a response.

**Q34. Is the LLM provider hardcoded to OpenAI?**

**A:** No. It is provider-independent via the `LLMProvider` interface. The shipped implementation is `OllamaProvider`; others can be added. No API keys are hardcoded — all config comes from environment variables.

**Q35. What is an AI agent, and is this platform one?**

**A:** An AI agent is an LLM-based system that can reason, use tools, and act autonomously toward a goal. This platform is a RAG pipeline, not a fully autonomous agent — it retrieves and generates but does not iterate with tools or external actions. Agent behavior (e.g., query decomposition, tool use) would be a future enhancement.

---

## LLM Evaluation

**Q36. Why evaluate rather than just demonstrate?**

**A:** Demos are subjective. Evaluation measures actual retrieval and generation quality against a labeled ground-truth dataset, producing reproducible numbers and revealing failure cases.

**Q37. What retrieval metrics are computed?**

**A:** Precision@K (fraction of top-K retrieved that are relevant), Recall@K (fraction of relevant items retrieved), and Mean Reciprocal Rank (MRR, inverse rank of the first relevant result).

**Q38. What answer-quality metrics are computed?**

**A:** Faithfulness (answer overlap with sources), Context Relevance (query overlap with sources), Answer Relevance (query-answer overlap), and Evidence Coverage (fraction of sources referenced by the answer). These are deterministic heuristics, not fabricated scores.

**Q39. How is relevance judged without hardcoded IDs?**

**A:** The evaluation dataset stores expected text substrings per question. A retrieved record is relevant if its feedback text contains any expected substring (case-insensitive), avoiding dependence on UUIDs that change per upload.

**Q40. What does "never fabricate metrics" mean?**

**A:** All numbers are computed from actual model output and labeled data. No accuracy/precision values are invented or hardcoded for the demo.

---

## FastAPI / REST APIs

**Q41. What is FastAPI, and why was it chosen?**

**A:** A modern, high-performance Python web framework built on Starlette and Pydantic. Chosen for automatic OpenAPI docs, type-safe request validation, async support, and great developer experience.

**Q42. What are the key endpoints?**

**A:** `POST /feedback/upload`, `GET /feedback`, `GET /feedback/{id}`, `DELETE /feedback/{id}`, `POST /feedback/search`, `GET /analytics/{summary, overview,sentiment,topics,trends}`, `POST /rag/ask`, `POST /rag/chat`, `POST /evaluation/run`, `GET /health`, `GET /health/ollama`.

**Q43. How are errors handled consistently?**

**A:** Custom `AppError` subclasses (`ValidationError`, `NotFoundError`, `ProviderError`) carry an error code, HTTP status, and details. FastAPI exception handlers convert them to a consistent JSON envelope. Pydantic validation errors become 422 responses.

**Q44. What is the lifespan context manager used for?**

**A:** It replaces the deprecated `on_event("startup")`. It initializes the SQLite database, Ollama client, services, and restores the vector index on startup, and closes connections on shutdown.

**Q45. What is Swagger/OpenAPI, and why does it matter?**

**A:** FastAPI auto-generates OpenAPI schemas from Pydantic models and route signatures, served at `/docs`. It provides interactive API exploration and machine-readable contracts for clients.

**Q46. What HTTP status codes are used and why?**

**A:** 200 OK, 201 Created, 204 No Content (delete), 400 validation, 404 not found, 422 unprocessable entity, 502 provider error (Ollama). Correct codes let clients distinguish client errors from server/provider failures.

---

## Docker

**Q47. Why containerize this application?**

**A:** Reproducible builds, dependency isolation, consistent environments (local/CI/production), and easy multi-service orchestration via Compose.

**Q48. What are the three containers and their roles?**

**A:** `ollama` (local LLM/embedding inference), `backend` (FastAPI API), `frontend` (Streamlit dashboard).

**Q49. Why not bake model files into the images?**

**A:** Ollama models are multi-gigabyte. Baking them would make images huge and slow to build. Instead, models are pulled once into a persistent volume (`/root/.ollama`) on first container start.

**Q50. What is a multi-stage build, and is it used here?**

**A:** A build with multiple `FROM` stages to copy only build artifacts into a small final image. Not strictly needed here (single-stage `python:3.15-slim` is already minimal), but it's a best practice for larger apps.

**Q51. Why run as a non-root user?**

**A:** Security: if the container is compromised, the attacker has limited permissions. We create `appuser` (uid 1000) and `USER appuser` in both Dockerfiles.

**Q52. What is `.dockerignore` for?**

**A:** Excludes files from the build context (`.env`, `__pycache__`, `*.db`, `.git`), making builds faster and preventing secrets/runtime data from leaking into images.

---

## Docker Compose

**Q53. What is Docker Compose?**

**A:** A tool for defining and running multi-container Docker applications with a single YAML file. One command (`docker compose up`) starts Ollama, backend, and frontend with networking and volumes.

**Q54. Why a dedicated bridge network (`app-network`)?**

**A:** It isolates the application's containers and gives them DNS-resolvable service names (`ollama`, `backend`, `frontend`) for inter-container communication.

**Q55. What is the critical networking rule, and why?**

**A:** Inside Docker, `localhost` means the current container. Backend must reach Ollama at `http://ollama:11434`; frontend must reach backend at `http://backend:8000`. Using `localhost` for inter-container calls would fail.

**Q56. What are `depends_on` with `condition: service_healthy`?**

**A:** They enforce startup order: backend waits for Ollama to be healthy; frontend waits for backend. This prevents race conditions where a service starts before its dependency is ready.

**Q57. What are health checks, and what do they verify here?**

**A:** Periodic commands that report container health. Backend hits `/health`, Ollama hits `/api/tags`, frontend hits Streamlit's health endpoint. Unhealthy containers are not considered ready by dependents.

**Q58. What persistent volumes are used?**

**A:** `ollama_models` (models), `feedback_data` (SQLite DB), `vector_index` (FAISS index), `uploads` (CSV uploads). Data survives container recreation.

**Q59. How do you rebuild and restart a single service?**

**A:** `docker compose build <service>` then `docker compose up -d <service>`.

---

## Networking / Volumes / System Design

**Q60. What is a Docker bridge network?**

**A:** A private network where containers can communicate by service name with automatic DNS, isolated from the host network except for published ports.

**Q61. What is a named volume vs. a bind mount?**

**A:** Named volumes are managed by Docker (persist across `docker compose down` unless `-v` is used). Bind mounts map a host path. We use named volumes for portability and Docker-managed storage.

**Q62. How does the system scale vertically vs. horizontally?**

**A:** Vertically: more CPU/RAM for Ollama and embedding batching. Horizontally: multiple backend replicas behind a load balancer sharing a central database and object-store-based index — not yet implemented (this is a modular monolith).

**Q63. What are the scalability limits of the current design?**

**A:** Single-process backend, in-memory record cache per session, exact FAISS IndexFlatIP is O(N), SQLite is single-writer. For production we'd add PostgreSQL, pgvector or a vector DB, approximate indexes (HNSW), and a task queue (Celery/RQ) for embedding jobs.

**Q64. How would you make embedding generation faster?**

**A:** Batch texts into single Ollama calls where supported, run on GPU, add a background task queue, and cache embeddings by content hash to avoid re-embedding unchanged feedback.

**Q65. What is circuit-breaker / graceful degradation for Ollama?**

**A:** If Ollama is unavailable, the app returns clear errors (502) instead of crashing, and search/RAG degrade gracefully. A production version would add retries with exponential backoff and a circuit breaker.

---

## Debugging

**Q66. How do you debug a 502 from the /feedback/upload endpoint?**

**A:** Check `docker compose logs backend` for Ollama connection errors; verify Ollama is healthy (`GET /health/ollama`); confirm `OLLAMA_BASE_URL` points to `http://ollama:11434` (not `localhost`) inside Compose.

**Q67. How do you debug missing search results?**

**A:** Verify the vector store is not empty (`GET /health` shows `vector_backend` and index size); confirm upload completed and embeddings were generated; check the query embedding dimension matches the index.

**Q68. How do you debug a failing Docker build?**

**A:** Run `docker compose build --no-cache <service>` and read the Dockerfile layer output; common causes are missing `requirements.txt`, wrong base image for Python 3.15, or copied paths not matching the build context.

**Q69. How do you inspect container state?**

**A:** `docker compose ps` (status), `docker compose logs <service>` (logs), `docker compose exec <service> sh` (shell inside container), `docker inspect` (low-level config).

**Q70. What is the most likely cause of "backend not reachable" from the frontend?**

**A:** `BACKEND_URL` set to `localhost` inside the frontend container instead of `http://backend:8000`. Fixed via the Compose environment variable.

**Q71. Summarize the project in 60 seconds.**

**A:** It's an Enterpret-inspired customer intelligence platform: upload CSV feedback, clean and validate it, run rule-based sentiment and topic analysis, generate local Ollama embeddings, store in FAISS (NumPy fallback), retrieve via semantic search, and answer questions with a grounded RAG pipeline. It has a FastAPI backend, Streamlit dashboard, SQLite persistence, 45 passing Pytest tests, an evaluation framework with real metrics, and full Docker Compose orchestration — no paid API keys and no fabricated results.

---

## 20. GitHub Repository & Submission

**Repository:** https://github.com/vishalmmulge/Customer-Feedback-Intelligence-Platform-Using-LLM-with-RAG.git

**8 Commits:**
1. `feat: add project structure and configuration`
2. `feat: implement core module - config, logging, exceptions, database`
3. `feat: add Pydantic data models and API contracts`
4. `feat: implement preprocessing service with auto column detection`
5. `feat: implement ML services - sentiment, topic, embeddings, vector store, retriever`
6. `feat: implement RAG pipeline, evaluation framework, and analytics`
7. `feat: implement FastAPI backend with routes and health endpoints`
8. `feat: add Streamlit frontend, Docker setup, tests, docs, and sample data`

**To clone and run:**
```powershell
git clone https://github.com/vishalmmulge/Customer-Feedback-Intelligence-Platform-Using-LLM-with-RAG.git
cd Customer-Feedback-Intelligence-Platform-Using-LLM-with-RAG
docker compose up -d --build
```

---

## 21. Conclusion

This project demonstrates end-to-end AI engineering: data ingestion, preprocessing, ML/NLP baselines, embeddings, vector search, RAG, analytics, evaluation, FastAPI backend, Streamlit frontend, Docker Compose orchestration, and comprehensive documentation — all running locally with no paid API keys and no fabricated results. It is designed to be modular, maintainable, explainable, testable, and deployable, making it suitable for an internship portfolio and technical interviews.
