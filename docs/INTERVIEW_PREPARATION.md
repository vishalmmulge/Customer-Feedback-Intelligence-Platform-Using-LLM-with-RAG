# Interview Preparation

Customer Feedback Intelligence Platform — 50+ technical questions and
detailed answers covering Python, OOP, ML, NLP, Transformers, Embeddings,
Vector databases, FAISS, RAG, AI agents, LLM evaluation, FastAPI, REST APIs,
Docker, Docker Compose, Networking, Volumes, System design, Scalability, and
Debugging.

---

## Python

**Q1. What is the Global Interpreter Lock (GIL), and how does it affect this project?**

**A:** The GIL is a mutex in CPython that allows only one thread to execute
Python bytecode at a time. In this project, embedding generation and LLM calls
are I/O-bound (network requests to Ollama), so the GIL releases during `await`
/`httpx` calls and the app remains responsive. CPU-bound work (NumPy/FAISS
search) releases the GIL inside native code, so multi-core search is possible.
For true CPU parallelism in Python we would use `multiprocessing` or
`concurrent.futures.ProcessPoolExecutor`.

**Q2. Why use Pydantic models instead of plain dicts?**

**A:** Pydantic provides runtime data validation, type coercion, clear error
messages, and automatic OpenAPI/Swagger schema generation. In this platform
every API request and the `FeedbackRecord` domain model are Pydantic models,
so malformed CSV rows or bad API payloads are rejected early with 422/400
responses instead of causing obscure downstream failures.

**Q3. How does `pydantic-settings` help with configuration?**

**A:** It loads settings from environment variables and `.env` files into typed
Pydantic models. This keeps secrets (API keys, DB paths, model names) out of
source code, supports 12-factor app principles, and makes the same code run
locally and in Docker with different env values.

**Q4. What is the difference between `async def` and regular `def` in FastAPI?**

**A:** FastAPI runs `async def` endpoints on the event loop (non-blocking) and
regular `def` endpoints in a threadpool. We use `async def` for I/O-bound
upload (file read + network to Ollama) and regular `def` for CPU-bound sync
operations (SQLite, NumPy search) so we don't block the event loop.

---

## OOP

**Q5. How is the service layer organized, and why?**

**A:** Each feature is a separate class (`SentimentService`, `TopicService`,
`RetrieverService`, `RAGService`, etc.) with a single responsibility. Classes
are instantiated once and stored on `app.state`. This gives encapsulation,
easy unit testing (inject fakes), and the ability to swap implementations
(e.g., replace the rule-based sentiment service with a transformer) without
touching the API layer.

**Q6. What is the `LLMProvider` abstract base class for?**

**A:** It defines a `generate(prompt)` interface, making the RAG service
provider-independent. `OllamaProvider` implements it today; we could add
`OpenAIProvider` or `AnthropicProvider` later without changing `RAGService`.
This is the Dependency Inversion Principle: depend on abstractions, not
concrete providers.

**Q7. How does composition beat inheritance here?**

**A:** `RAGService` composes a `RetrieverService` and an `LLMProvider` rather
than inheriting from them. This lets us mix and match retrieval and generation
strategies at runtime and avoids fragile base-class hierarchies.

**Q8. Explain the `Database` class design.**

**A:** It encapsulates the `sqlite3` connection, table creation, indexes, and
CRUD/aggregation queries behind a clean interface. The FastAPI app holds a
single `Database` instance on `app.state`, connecting on startup and closing
on shutdown via the lifespan context manager.

---

## Machine Learning

**Q9. What is sentiment analysis, and how is it implemented here?**

**A:** Sentiment analysis classifies text as positive, negative, or neutral.
The MVP uses a transparent, rule-based lexicon: each word carries a weight,
scores are summed, clamped to [-1, 1], and thresholded into labels. This is
documented as a baseline — not a trained deep-learning model — so results are
explainable and auditable.

**Q10. Why not use a pretrained transformer for sentiment out of the box?**

**A:** Transformers require model downloads, a GPU for reasonable latency, and
more dependencies. For an internship/MVP, a lexicon baseline runs offline,
is fast, and is fully explainable. It can be swapped later via the
`SentimentService` interface without changing the pipeline.

**Q11. What is topic classification, and what baseline is used?**

**A:** Topic classification assigns each feedback record to a category (e.g.,
billing, bugs, performance). The MVP uses an explicit keyword bucket map with
case-insensitive substring matching and a `general` fallback. It is
extensible: new topics/keywords can be added without retraining.

**Q12. What is the difference between classification and clustering?**

**A:** Classification assigns labels from a predefined set (supervised). Clustering
groups similar items without predefined labels (unsupervised). This project uses
rule-based classification; semantic clustering could be added later via
embedding-based k-means or HDBSCAN.

**Q13. What is overfitting, and is it relevant here?**

**A:** Overfitting is when a model memorizes training data and fails to
generalize. Our rule-based baselines have no trainable parameters, so
overfitting in the classical sense doesn't apply — but the rules can be
over-tuned to the sample dataset, which is why they're documented as baselines
and evaluated on labeled data.

---

## NLP

**Q14. What preprocessing is applied to feedback text?**

**A:** Whitespace stripping, normalization of `nan`/`None`/blank strings to
missing, duplicate detection via SHA-256 hashing of normalized text, and
preservation of the original text. Empty/blank feedback rows are dropped.

**Q15. Why deduplicate feedback?**

**A:** Duplicate rows would bias analytics (e.g., sentiment distribution) and
waste embedding/index storage. We hash normalized text and keep only the first
occurrence.

**Q16. What is tokenization in this context?**

**A:** The sentiment service uses `re.findall(r"\w+", text.lower())` to split
text into word tokens for lexicon lookup. No subword tokenization (BPE/WordPiece)
is used because the baseline is word-level.

---

## Transformers

**Q17. Are transformers used in this project?**

**A:** Not in the MVP. The design allows swapping `SentimentService` and
`TopicService` for Hugging Face Transformers models later (e.g.,
`cardiffnlp/twitter-roberta-base-sentiment` for sentiment, a zero-shot
classifier for topics). The interface stays the same.

**Q18. What is self-attention, briefly?**

**A:** A mechanism in transformers that computes weighted representations of
each token based on all other tokens in the sequence, allowing the model to
capture long-range dependencies and context.

**Q19. What is a transformer embedding?**

**A:** A dense vector representation of text produced by a transformer model,
capturing semantic meaning. Similar texts have similar vectors (high cosine
similarity).

---

## Embeddings

**Q20. What embedding model is used, and why?**

**A:** `nomic-embed-text` via Ollama. It is open-weight, runs locally (no API
key), produces 768-dimensional vectors, and is well-suited for semantic
search. Running it locally keeps the stack offline and free.

**Q21. How are embeddings generated and stored?**

**A:** Each feedback text is sent to Ollama's `/api/embeddings` endpoint. The
resulting 768-dim vector is L2-normalized and stored in the vector index
mapped to the feedback ID. Vectors are persisted to disk (`backend/index/`).

**Q22. What is embedding normalization, and why does it matter?**

**A:** L2-normalizing vectors to unit length makes the dot product equal to
cosine similarity. FAISS `IndexFlatIP` (inner product) on normalized vectors
thus performs cosine-similarity search.

**Q23. What is the embedding dimension, and is it validated?**

**A:** `nomic-embed-text` produces 768-dim vectors. The vector store validates
dimension on creation and every query vector must match the index dimension.

**Q24. What is the NumPy fallback for FAISS?**

**A:** On Python 3.15 no `faiss-cpu` wheel exists. The app falls back to exact
NumPy cosine-similarity search with an identical API, so functionality is
preserved without FAISS.

---

## Vector Databases / FAISS

**Q25. What is FAISS, and how is it used here?**

**A:** FAISS (Facebook AI Similarity Search) is a library for efficient
similarity search on dense vectors. We use `IndexFlatIP` (exact inner-product
search) on L2-normalized vectors for cosine-similarity retrieval of feedback
records.

**Q26. What is IndexFlatIP?**

**A:** An exact, brute-force FAISS index that computes the inner product
between the query and every stored vector. It is accurate but O(N) — fine for
MVP-scale datasets; approximate indexes (IVF, HNSW) can replace it at scale.

**Q27. How is the FAISS index persisted?**

**A:** `faiss.write_index` / `faiss.read_index` for FAISS; `np.save` /
`np.load` for the NumPy fallback. Records are serialized to JSON alongside
the index so ID↔vector mapping survives restarts.

**Q28. What is cosine similarity?**

**A:** A measure of vector orientation similarity ranging from -1 (opposite)
to 1 (identical). For normalized vectors, dot product == cosine similarity.
We use it to rank feedback by semantic relevance to a query.

**Q29. What is top-K retrieval?**

**A:** Returning the K most similar vectors to a query. Here K is configurable
(default 5, max 50 for search, max 20 for RAG) to balance relevance and
prompt size.

**Q30. What is stale-record handling in the vector store?**

**A:** Deleting a record from SQLite does not automatically remove its vector
from FAISS (an index doesn't support efficient single-vector deletion). The
MVP notes this; a full rebuild or `IndexIDMap` with `remove_ids` can address
it at scale.

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

**A:** It prevents hallucination. The system prompt instructs the LLM to answer
using only the provided feedback context and to say so if evidence is
insufficient. Source IDs and excerpts are returned so answers are auditable.

**Q33. What happens if Ollama is down?**

**A:** `OllamaProvider.generate` raises `ProviderError`, which the RAG endpoint
catches and returns as a clear error answer instead of crashing or fabricating
a response.

**Q34. Is the LLM provider hardcoded to OpenAI?**

**A:** No. It is provider-independent via the `LLMProvider` interface. The
shipped implementation is `OllamaProvider`; others can be added. No API keys
are hardcoded — all config comes from environment variables.

**Q35. What is an AI agent, and is this platform one?**

**A:** An AI agent is an LLM-based system that can reason, use tools, and act
autonomously toward a goal. This platform is a RAG pipeline, not a fully
autonomous agent — it retrieves and generates but does not iterate with tools
or external actions. Agent behavior (e.g., query decomposition, tool use)
would be a future enhancement.

---

## LLM Evaluation

**Q36. Why evaluate rather than just demonstrate?**

**A:** Demos are subjective. Evaluation measures actual retrieval and
generation quality against a labeled ground-truth dataset, producing
reproducible numbers and revealing failure cases.

**Q37. What retrieval metrics are computed?**

**A:** Precision@K (fraction of top-K retrieved that are relevant), Recall@K
(fraction of relevant items retrieved), and Mean Reciprocal Rank (MRR, inverse
rank of the first relevant result).

**Q38. What answer-quality metrics are computed?**

**A:** Faithfulness (answer overlap with sources), Context Relevance (query
overlap with sources), Answer Relevance (query-answer overlap), and Evidence
Coverage (fraction of sources referenced by the answer). These are
deterministic heuristics, not fabricated scores.

**Q39. How is relevance judged without hardcoded IDs?**

**A:** The evaluation dataset stores expected text substrings per question. A
retrieved record is relevant if its feedback text contains any expected
substring (case-insensitive), avoiding dependence on UUIDs that change per
upload.

**Q40. What does "never fabricate metrics" mean?**

**A:** All numbers are computed from actual model output and labeled data. No
accuracy/precision values are invented or hardcoded for the demo.

---

## FastAPI / REST APIs

**Q41. What is FastAPI, and why was it chosen?**

**A:** A modern, high-performance Python web framework built on Starlette and
Pydantic. Chosen for automatic OpenAPI docs, type-safe request validation,
async support, and great developer experience.

**Q42. What are the key endpoints?**

**A:** `POST /feedback/upload`, `GET /feedback`, `GET /feedback/{id}`,
`DELETE /feedback/{id}`, `POST /feedback/search`, `GET /analytics/{summary,
overview,sentiment,topics,trends}`, `POST /rag/ask`, `POST /rag/chat`,
`POST /evaluation/run`, `GET /health`, `GET /health/ollama`.

**Q43. How are errors handled consistently?**

**A:** Custom `AppError` subclasses (`ValidationError`, `NotFoundError`,
`ProviderError`) carry an error code, HTTP status, and details. FastAPI
exception handlers convert them to a consistent JSON envelope. Pydantic
validation errors become 422 responses.

**Q44. What is the lifespan context manager used for?**

**A:** It replaces the deprecated `on_event("startup")`. It initializes the
SQLite database, Ollama client, services, and restores the vector index on
startup, and closes connections on shutdown.

**Q45. What is Swagger/OpenAPI, and why does it matter?**

**A:** FastAPI auto-generates OpenAPI schemas from Pydantic models and route
signatures, served at `/docs`. It provides interactive API exploration and
machine-readable contracts for clients.

**Q46. What HTTP status codes are used and why?**

**A:** 200 OK, 201 Created, 204 No Content (delete), 400 validation, 404 not
found, 422 unprocessable entity, 502 provider error (Ollama). Correct codes
let clients distinguish client errors from server/provider failures.

---

## Docker

**Q47. Why containerize this application?**

**A:** Reproducible builds, dependency isolation, consistent environments
(local/CI/production), and easy multi-service orchestration via Compose.

**Q48. What are the three containers and their roles?**

**A:** `ollama` (local LLM/embedding inference), `backend` (FastAPI API),
`frontend` (Streamlit dashboard).

**Q49. Why not bake model files into the images?**

**A:** Ollama models are multi-gigabyte. Baking them would make images huge
and slow to build. Instead, models are pulled once into a persistent volume
(`/root/.ollama`) on first container start.

**Q50. What is a multi-stage build, and is it used here?**

**A:** A build with multiple `FROM` stages to copy only build artifacts into a
small final image. Not strictly needed here (single-stage `python:3.15-slim`
is already minimal), but it's a best practice for larger apps.

**Q51. Why run as a non-root user?**

**A:** Security: if the container is compromised, the attacker has limited
permissions. We create `appuser` (uid 1000) and `USER appuser` in both
Dockerfiles.

**Q52. What is `.dockerignore` for?**

**A:** Excludes files from the build context (`.env`, `__pycache__`, `*.db`,
`.git`), making builds faster and preventing secrets/runtime data from leaking
into images.

---

## Docker Compose

**Q53. What is Docker Compose?**

**A:** A tool for defining and running multi-container Docker applications
with a single YAML file. One command (`docker compose up`) starts Ollama,
backend, and frontend with networking and volumes.

**Q54. Why a dedicated bridge network (`app-network`)?**

**A:** It isolates the application's containers and gives them DNS-resolvable
service names (`ollama`, `backend`, `frontend`) for inter-container
communication.

**Q55. What is the critical networking rule, and why?**

**A:** Inside Docker, `localhost` means the current container. Backend must
reach Ollama at `http://ollama:11434`; frontend must reach backend at
`http://backend:8000`. Using `localhost` for inter-container calls would fail.

**Q56. What are `depends_on` with `condition: service_healthy`?**

**A:** They enforce startup order: backend waits for Ollama to be healthy;
frontend waits for backend. This prevents race conditions where a service
starts before its dependency is ready.

**Q57. What are health checks, and what do they verify here?**

**A:** Periodic commands that report container health. Backend hits `/health`,
Ollama hits `/api/tags`, frontend hits Streamlit's health endpoint. Unhealthy
containers are not considered ready by dependents.

**Q58. What persistent volumes are used?**

**A:** `ollama_models` (models), `feedback_data` (SQLite DB), `vector_index`
(FAISS index), `uploads` (CSV uploads). Data survives container recreation.

**Q59. How do you rebuild and restart a single service?**

**A:** `docker compose build <service>` then `docker compose up -d <service>`.

---

## Networking / Volumes / System Design

**Q58. What is a Docker bridge network?**

**A:** A private network where containers can communicate by service name with
automatic DNS, isolated from the host network except for published ports.

**Q59. What is a named volume vs. a bind mount?**

**A:** Named volumes are managed by Docker (persist across `docker compose
down` unless `-v` is used). Bind mounts map a host path. We use named volumes
for portability and Docker-managed storage.

**Q60. How does the system scale vertically vs. horizontally?**

**A:** Vertically: more CPU/RAM for Ollama and embedding batching. Horizontally:
multiple backend replicas behind a load balancer sharing a central database
and object-store-based index — not yet implemented (this is a modular
monolith).

**Q61. What are the scalability limits of the current design?**

**A:** Single-process backend, in-memory record cache per session, exact
FAISS IndexFlatIP is O(N), SQLite is single-writer. For production we'd add
PostgreSQL, pgvector or a vector DB, approximate indexes (HNSW), and a task
queue (Celery/RQ) for embedding jobs.

**Q62. How would you make embedding generation faster?**

**A:** Batch texts into single Ollama calls where supported, run on GPU, add
a background task queue, and cache embeddings by content hash to avoid
re-embedding unchanged feedback.

**Q63. What is circuit-breaker / graceful degradation for Ollama?**

**A:** If Ollama is unavailable, the app returns clear errors (502) instead of
crashing, and search/RAG degrade gracefully. A production version would add
retries with exponential backoff and a circuit breaker.

---

## Debugging

**Q64. How do you debug a 502 from the /feedback/upload endpoint?**

**A:** Check `docker compose logs backend` for Ollama connection errors;
verify Ollama is healthy (`GET /health/ollama`); confirm `OLLAMA_BASE_URL`
points to `http://ollama:11434` (not `localhost`) inside Compose.

**Q65. How do you debug missing search results?**

**A:** Verify the vector store is not empty (`GET /health` shows
`vector_backend` and index size); confirm upload completed and embeddings
were generated; check the query embedding dimension matches the index.

**Q66. How do you debug a failing Docker build?**

**A:** Run `docker compose build --no-cache <service>` and read the Dockerfile
layer output; common causes are missing `requirements.txt`, wrong base image
for Python 3.15, or copied paths not matching the build context.

**Q67. How do you inspect container state?**

**A:** `docker compose ps` (status), `docker compose logs <service>` (logs),
`docker compose exec <service> sh` (shell inside container), `docker inspect`
(low-level config).

**Q68. How do you verify persistence works?**

**A:** Upload data, restart containers (`docker compose restart`), confirm
records and search results are still present because SQLite and FAISS volumes
persist.

**Q69. What is the most likely cause of "backend not reachable" from the
frontend?**

**A:** `BACKEND_URL` set to `localhost` inside the frontend container instead
of `http://backend:8000`. Fixed via the Compose environment variable.

**Q70. Summarize the project in 60 seconds.**

**A:** It's an Enterpret-inspired customer intelligence platform: upload CSV
feedback, clean and validate it, run rule-based sentiment and topic analysis,
generate local Ollama embeddings, store in FAISS (NumPy fallback), retrieve
via semantic search, and answer questions with a grounded RAG pipeline. It has
a FastAPI backend, Streamlit dashboard, SQLite persistence, 45 passing Pytest
tests, an evaluation framework with real metrics, and full Docker Compose
orchestration — no paid API keys and no fabricated results.
