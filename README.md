# Customer Feedback Intelligence Platform

An **Enterpret-inspired, AI-native customer intelligence platform** built as an
internship portfolio project. Upload CSV customer feedback, clean and validate
it, run sentiment + topic analysis, generate embeddings with local Ollama,
store vectors in FAISS, retrieve relevant feedback via semantic search, and
answer natural-language questions with a grounded RAG pipeline.

Designed to be **modular, maintainable, explainable, testable, and deployable**
with Docker Compose. No paid API keys required — all inference runs locally.

---

## Features

| Capability | Implementation |
|------------|----------------|
| CSV ingestion | Upload via API or Streamlit UI; column validation, missing-value handling, duplicate detection, timestamp handling, unique IDs, original-text preservation |
| Preprocessing | Pandas cleaning; row-level error reporting; source tagging; `processed_at` timestamps |
| Sentiment analysis | Rule-based lexicon baseline (positive / neutral / negative, score in [-1, 1]); documented limitations |
| Topic classification | 10 keyword buckets (bugs, performance, billing, UI, feature requests, support, etc.) + `general` fallback |
| Embeddings | Ollama `nomic-embed-text` (768-dim), L2-normalized, batched |
| Vector search | FAISS `IndexFlatIP` (cosine similarity) with **NumPy exact-search fallback** (Python 3.15 has no `faiss-cpu` wheel) |
| Persistence | **SQLite** (`feedback.db`) with indexes; FAISS/NumPy index persisted to disk |
| Semantic search | Top-K retrieval with similarity scores |
| RAG Q&A | Grounded answers via Ollama `llama3.2:3b` with source IDs, excerpts, and similarity scores |
| Analytics | Overview, sentiment, topics, trends, summary — all computed from SQLite |
| Evaluation | Precision@K, Recall@K, MRR, faithfulness, context/answer relevance, evidence coverage against a manually labeled dataset |
| API | FastAPI with Pydantic validation, Swagger UI (`/docs`), consistent JSON errors |
| Frontend | Streamlit dashboard (5 tabs: Overview, Explorer, Insights, AI Assistant, Evaluation) |
| Testing | **45 Pytest tests pass** — Ollama/FAISS mocked for offline unit tests |
| Docker | Backend, frontend, and Ollama containers; persistent volumes; health checks; Compose orchestration |

---

## Technology Stack

- **Backend:** Python, FastAPI, Pydantic, Uvicorn
- **Data processing:** Pandas, NumPy, Regular Expressions, sqlite3
- **ML/NLP:** Scikit-learn (available), rule-based sentiment & topic baselines
- **LLM:** Ollama (`llama3.2:3b`)
- **Embeddings:** Ollama (`nomic-embed-text`)
- **Vector retrieval:** FAISS with NumPy cosine-similarity fallback
- **Frontend:** Streamlit, Plotly
- **Testing:** Pytest, FastAPI TestClient
- **Infrastructure:** Docker, Docker Compose, persistent volumes, health checks

---

## Project Structure

```
customer-feedback-intelligence/
├── backend/
│   ├── Dockerfile
│   ├── main.py                 # FastAPI app factory, lifespan, health
│   ├── config.py               # pydantic-settings config from .env
│   ├── core/
│   │   ├── database.py         # SQLite persistence layer
│   │   ├── logging.py          # Structured logging
│   │   └── exceptions.py       # Typed app errors
│   ├── api/
│   │   ├── feedback.py         # upload / list / get / delete / search
│   │   ├── analytics.py        # overview / sentiment / topics / trends / summary
│   │   ├── rag.py              # /ask and /chat (RAG Q&A)
│   │   └── evaluation.py       # /run (RAG evaluation suite)
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response contracts
│   ├── services/
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # CSV ingest, clean, validate, deduplicate
│   │   ├── sentiment.py        # Rule-based polarity scoring
│   │   ├── topic.py            # Keyword topic classification
│   │   ├── embeddings.py       # Ollama embedding client + ping
│   │   ├── vector_store.py     # FAISS + NumPy fallback index
│   │   ├── retriever.py        # Semantic search wrapper
│   │   ├── rag.py              # LLM provider abstraction + grounded answers
│   │   ├── analytics.py        # SQLite aggregations
│   │   └── evaluation.py       # Precision@K, Recall@K, MRR, answer metrics
│   └── index/                  # persisted vector index (runtime)
├── frontend/
│   ├── Dockerfile
│   └── app.py                  # Streamlit dashboard (5 tabs)
├── ollama/
│   └── entrypoint.sh           # Ollama model warm-up entrypoint
├── data/
│   ├── sample_feedback.csv     # 75-row realistic synthetic dataset
│   └── evaluation_dataset.jsonl # Manually labeled RAG evaluation set
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API_DOCUMENTATION.md
│   ├── DOCKER_DEPLOYMENT.md
│   └── INTERVIEW_PREPARATION.md   # 50+ technical Q&A
├── tests/
│   ├── test_preprocessing.py
│   ├── test_sentiment.py
│   ├── test_topic.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_rag.py
│   └── test_api.py             # FastAPI TestClient tests (SQLite + fakes)
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
├── docker-compose.yml
└── README.md
```

> **Synthetic data notice:** `data/sample_feedback.csv` contains realistic
> synthetic feedback for demonstration. It is clearly sample data, not real
> customer data.

---

## Quick Start (Local)

### Prerequisites
- Python 3.9+ (tested on 3.15)
- [Ollama](https://ollama.com) running locally

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

### Install & run backend
```bash
python -m venv venv
# Windows: venv\Scripts\Activate.ps1
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # adjust if needed
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

API: http://127.0.0.1:8000  |  Swagger UI: http://127.0.0.1:8000/docs

### Run frontend
```bash
streamlit run frontend/app.py
```
Dashboard: http://localhost:8501

### Run tests
```bash
python -m pytest tests/ -v   # 45 tests, all offline/mocked
```

---

## Docker Compose Deployment (Recommended)

Docker Desktop on Windows 11 (WSL2 backend) or Docker Engine + Compose.

```bash
# Build all images
docker compose build

# Start all services
docker compose up -d
```

| Service | URL |
|---------|-----|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Ollama API | http://localhost:11434 |

Then upload `data/sample_feedback.csv` and ask questions in the AI Assistant tab.

Key Compose commands:
```bash
docker compose logs -f        # view logs
docker compose ps             # container health
docker compose stop           # stop services
docker compose down           # stop & remove containers
docker compose down -v        # also remove volumes (data lost!)
```

See `docs/DOCKER_DEPLOYMENT.md` for full details.

**Important networking rule:** Inside Docker, `localhost` means the current
container. Backend reaches Ollama at `http://ollama:11434`; frontend reaches
backend at `http://backend:8000`. Never use `localhost` for inter-container
calls.

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/feedback/upload` | Upload & process CSV |
| `GET` | `/feedback` | List feedback (filters, pagination) |
| `GET` | `/feedback/{id}` | Get one record |
| `DELETE` | `/feedback/{id}` | Delete one record |
| `POST` | `/feedback/search` | Semantic search (top-K) |
| `GET` | `/analytics/summary` | Aggregate stats |
| `GET` | `/analytics/overview` | High-level overview |
| `GET` | `/analytics/sentiment` | Sentiment distribution |
| `GET` | `/analytics/topics` | Topic distribution |
| `GET` | `/analytics/trends` | Volume trend over time |
| `POST` | `/rag/ask` | RAG question → grounded answer |
| `POST` | `/rag/chat` | Alias for `/rag/ask` |
| `POST` | `/evaluation/run` | Run RAG evaluation suite |
| `GET` | `/health` | Health check |
| `GET` | `/health/ollama` | Ollama readiness |

Full details: `docs/API_DOCUMENTATION.md`.

---

## Evaluation Framework

Run against a manually labeled dataset (`data/evaluation_dataset.jsonl`):

```bash
curl -X POST "http://localhost:8000/evaluation/run?top_k=5"
```

Metrics computed from actual output — **nothing is fabricated**:
- Retrieval: Precision@K, Recall@K, MRR
- Answer: Faithfulness, Context Relevance, Answer Relevance, Evidence Coverage

---

## Architecture

See `docs/ARCHITECTURE.md` for the layered design, data-flow diagram, and
design decisions (provider-independent LLM, FAISS + NumPy fallback, rule-based
baselines, no fabricated metrics).

---

## Limitations

- Sentiment and topic classification are **rule-based baselines** (documented),
  not trained deep-learning models. Swap via the service interfaces when needed.
- In-memory record cache during a server session; durable store is SQLite.
- `faiss-cpu` has no wheel for Python 3.15 → NumPy fallback is used on 3.15.
- Deleting a record from SQLite does not remove its vector from FAISS (full
  rebuild needed to purge from index).
- No authentication/authorization (internal/demo use).
- Single Ollama instance handles both embeddings and chat; large CSV uploads
  may be slow (no embedding batching yet).

---

## Interview Preparation

See `docs/INTERVIEW_PREPARATION.md` for **50+ technical Q&A** covering Python,
OOP, ML, NLP, Transformers, Embeddings, Vector databases, FAISS, RAG, AI
agents, LLM evaluation, FastAPI, REST APIs, Docker, Docker Compose, Networking,
Volumes, System design, Scalability, and Debugging.

---

## Future Improvements

- Replace rule-based sentiment/topic with transformer / zero-shot models
- Approximate FAISS indexes (IVF, HNSW) for large datasets
- Embedding caching by content hash + background task queue (Celery/RQ)
- PostgreSQL + pgvector or a dedicated vector database
- Authentication/authorization
- Unit tests with `pytest-httpx` for HTTP-level assertions
- GPU acceleration for Ollama (optional, documented)

---

## License

Internal / demo project. Add a license as appropriate for your organization.