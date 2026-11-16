"""FastAPI application factory.

Wires together config, logging, services, SQLite database, and API routers.
On startup it initializes the database and restores the FAISS / NumPy vector
index from persisted records so search survives restarts.
"""

from contextlib import asynccontextmanager
from typing import Dict, List

from fastapi import FastAPI
from fastapi.exception_handlers import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api import analytics, evaluation, feedback, rag
from .core.config import settings
from .core.database import Database
from .core.exceptions import AppError
from .core.logging import logger
from .models.schemas import FeedbackRecord, HealthResponse
from .services import (
    EmbeddingsService,
    OllamaClient,
    OllamaProvider,
    PreprocessingService,
    RAGService,
    RetrieverService,
    SentimentService,
    TopicService,
    VectorStore,
    EvaluationService,
)


def _build_vector_store_from_records(
    embeddings: EmbeddingsService,
    records: Dict[str, FeedbackRecord],
) -> Optional[VectorStore]:
    """Hydrate a VectorStore from persisted records on startup.

    Returns None if there are no records yet, or if embedding fails
    (e.g. Ollama models not yet available).
    """
    if not records:
        return None

    try:
        # Determine embedding dimension from the first record.
        first_text = next(iter(records.values())).feedback_text
        first_vec = embeddings.embed_text(first_text)
        vs = VectorStore(dimension=len(first_vec))

        vectors = []
        ids = []
        for rid, record in records.items():
            vectors.append(embeddings.embed_text(record.feedback_text))
            ids.append(rid)
        vs.add(vectors, ids)
        logger.info(f"Restored vector store with {len(records)} record(s)")
        return vs
    except Exception as exc:  # Ollama not ready yet; index rebuilds on first upload
        logger.warning(f"Could not restore vector store on startup: {exc}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Customer Feedback Intelligence API")

    # Core services (stateless, safe to create up front).
    app.state.preprocessor = PreprocessingService()
    app.state.sentiment_service = SentimentService()
    app.state.topic_service = TopicService()
    app.state.embeddings = EmbeddingsService()

    # SQLite database.
    db = Database(db_path=settings.DATABASE_PATH)
    db.connect()
    app.state.database = db

    # Load all records from SQLite into memory for the current session.
    db_records: List[FeedbackRecord] = []
    try:
        rows = db.list_records(page=1, page_size=100_000)
        db_records = rows
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not load records from DB: {exc}")
        rows = []

    records_dict: Dict[str, FeedbackRecord] = {r.id: r for r in db_records}

    # LLM provider for RAG. Degrades gracefully if Ollama is absent.
    ollama_client = OllamaClient()
    app.state.ollama_client = ollama_client
    app.state.llm_provider = OllamaProvider(client=ollama_client)

    # Vector store (populated on first upload or restored here).
    app.state.vector_store = None
    app.state.retriever = None
    app.state.rag_service = None
    app.state.evaluation_service = None

    if records_dict:
        vs = _build_vector_store_from_records(app.state.embeddings, records_dict)
        if vs is not None:
            app.state.vector_store = vs
            app.state.retriever = RetrieverService(
                app.state.embeddings, vs, records_dict
            )
            app.state.rag_service = RAGService(
                app.state.retriever, app.state.llm_provider
            )
            app.state.evaluation_service = EvaluationService(
                app.state.retriever, app.state.rag_service
            )
            logger.info(f"Restored index with {len(records_dict)} record(s)")
        else:
            logger.info("No records found; index will be created on first upload")
    else:
        logger.info("No persisted records found; index will be created on first upload")

    yield

    # Shutdown.
    try:
        ollama_client.close()
    except Exception:  # pragma: no cover - defensive
        pass
    db.close()
    logger.info("Shutting down Customer Feedback Intelligence API")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Customer Feedback Intelligence Platform: upload CSV feedback, "
            "run sentiment + topic analysis, embed with Ollama, search with "
            "FAISS, and answer questions with RAG."
        ),
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ #
    # Exception handlers -> consistent JSON errors
    # ------------------------------------------------------------------ #
    @app.exception_handler(AppError)
    async def app_error_handler(request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "http_error", "message": str(exc.detail)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Invalid request",
                "details": exc.errors(),
            },
        )

    # ------------------------------------------------------------------ #
    # Routers
    # ------------------------------------------------------------------ #
    app.include_router(feedback.router)
    app.include_router(analytics.router)
    app.include_router(rag.router)
    app.include_router(evaluation.router)

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health():
        vs = app.state.vector_store
        backend = vs.backend if vs else "none"
        return HealthResponse(
            status="ok",
            version=settings.APP_VERSION,
            ollama_configured=bool(settings.OLLAMA_BASE_URL),
            ollama_ready=_ollama_ping(app.state.ollama_client),
            vector_backend=backend,
            database_backend="sqlite",
        )

    @app.get("/health/ollama", tags=["health"])
    def health_ollama():
        ready = _ollama_ping(app.state.ollama_client)
        return {
            "ollama_ready": ready,
            "base_url": settings.OLLAMA_BASE_URL,
            "embed_model": settings.OLLAMA_EMBED_MODEL,
            "llm_model": settings.OLLAMA_LLM_MODEL,
        }

    return app


def _ollama_ping(client: OllamaClient) -> bool:
    return client.ping()


app = create_app()
