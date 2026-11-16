# Docker Deployment

## Prerequisites

- Docker Desktop (Windows 11, WSL2 backend) or Docker Engine + Compose
- No paid API key required; all inference runs locally via Ollama

## 1. Clone / locate the project

All files are in this repository. From the project root:

```powershell
cd customer-feedback-intelligence
```

## 2. Environment

Copy the example env file (optional for Compose; values are set in
`docker-compose.yml`):

```powershell
copy .env.example .env
```

Edit `.env` if you need to change models, paths, or log level.

## 3. Build and start

```powershell
# Build all images
docker compose build

# Start all services
docker compose up -d
```

This starts:
- `cf-ollama` (port 11434)
- `cf-backend` (port 8000)
- `cf-frontend` (port 8501)

## 4. Access the application

| Service | URL |
|---------|-----|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Ollama API | http://localhost:11434 |

## 5. Verify everything works

1. Open http://localhost:8501.
2. Confirm sidebar shows "Backend status: ok" and "Ollama ready: true".
3. Upload `data/sample_feedback.csv` via the dashboard.
4. Wait for processing (embeddings via Ollama).
5. Try the AI Assistant tab with a question such as:
   - "What are customers complaining about most?"
6. Check the Evaluation tab and run the suite.

## 6. Lifecycle commands

```powershell
# View logs
docker compose logs -f

# View container health
docker compose ps

# Stop services
docker compose stop

# Stop and remove containers
docker compose down

# Restart a single service
docker compose restart backend

# Rebuild a single service
docker compose build backend
docker compose up -d backend
```

## 7. Persistent data

Data survives container recreation because of Docker volumes:

| Volume | Mount path | Purpose |
|--------|------------|---------|
| `ollama_models` | `/root/.ollama` | Ollama models (nomic-embed-text, llama3.2:3b) |
| `feedback_data` | `/data` | SQLite database (`feedback.db`) |
| `vector_index` | `/app/backend/index` | FAISS / NumPy vector index |
| `uploads` | `/app/data/uploads` | Uploaded CSV files |

To remove all data (including models, requiring re-download):

```powershell
docker compose down -v
```

## 8. Networking (important)

Inside Docker, `localhost` refers to the current container. Inter-container
communication uses Compose service names:

- Backend → Ollama: `http://ollama:11434`
- Frontend → Backend: `http://backend:8000`

Do not use `localhost` for inter-container calls.

## 9. GPU (optional)

By default the platform runs CPU-only. To enable NVIDIA GPU acceleration for
Ollama, uncomment the `deploy.resources.reservations.devices` block in
`docker-compose.yml` under the `ollama` service and ensure the NVIDIA Container
Toolkit is installed.

## 10. Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| Ollama not ready | Models still downloading | Wait; check `docker compose logs ollama` |
| Backend 502 on upload | Ollama unreachable | Confirm `OLLAMA_BASE_URL=http://ollama:11434` |
| Frontend can't reach backend | Wrong `BACKEND_URL` | Set `BACKEND_URL=http://backend:8000` |
| Data lost after restart | Volumes not used | Do not run `docker compose down -v` unless intended |
| `faiss-cpu` install fails | No wheel for Python 3.15 | NumPy fallback activates automatically |

## 11. Security

- Containers run as a non-root user (`appuser`, uid 1000).
- Minimal base images (`python:3.15-slim`).
- `.dockerignore` excludes secrets, caches, and runtime data.
- No privileged mode. No secrets baked into images.
- Restricted exposed ports (8000, 8501, 11434).
