"""Feedback API routes: upload, list, get, delete, and semantic search.

The upload endpoint runs the full preprocessing + enrichment pipeline and
builds / updates the FAISS vector index so the feedback becomes searchable.
All records are durably persisted to SQLite.
"""

import io
import logging
from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse

from ..core.config import settings
from ..core.database import Database
from ..core.exceptions import AppError
from ..core.logging import logger
from ..models.schemas import (
    AnalyticsSummary,
    FeedbackListResponse,
    FeedbackRecord,
    FeedbackUploadResponse,
    RAGRequest,
    RAGResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResult,
)
from ..services.embeddings import EmbeddingsService
from ..services.evaluation import EvaluationService
from ..services.retriever import RetrieverService
from ..services.rag import RAGService

router = APIRouter(prefix="/feedback", tags=["feedback"])


def _get_state(request: Request):
    return request.app.state


# ---------------------------------------------------------------------- #
# Upload
# ---------------------------------------------------------------------- #
@router.post("/upload", response_model=FeedbackUploadResponse, status_code=201)
async def upload_feedback(
    request: Request,
    file: UploadFile = File(..., description="CSV file with a feedback_text column"),
    source: str = Query(default="csv", description="Source label for the batch"),
    column_mapping: str = Query(
        default="{}",
        description="JSON mapping of CSV columns to internal schema, e.g. {\"Text\": \"feedback_text\", \"User ID\": \"customer_id\"}",
    ),
):
    """Upload and process a CSV of customer feedback.

    Pipeline: validate columns -> clean -> deduplicate -> enrich
    (sentiment, topic) -> embed (Ollama) -> add to FAISS index ->
    persist to SQLite + vector store on disk.
    """
    import json as _json

    state = _get_state(request)
    preprocessor = state.preprocessor
    sentiment_service = state.sentiment_service
    topic_service = state.topic_service
    embeddings: EmbeddingsService = state.embeddings
    database: Database = state.database

    # Parse column mapping.
    try:
        mapping = _json.loads(column_mapping) if column_mapping else {}
        if not isinstance(mapping, dict):
            raise ValueError("column_mapping must be a JSON object")
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "error": "validation_error",
                "message": f"Invalid column_mapping: {exc}",
                "details": {},
            },
        )

    # 1. Read & clean.
    try:
        content = await file.read()
        raw_records, _ = preprocessor.process(
            io.BytesIO(content), source=source, column_mapping=mapping
        )
    except AppError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": exc.message, "details": exc.details},
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Upload preprocessing failed")
        return JSONResponse(
            status_code=500,
            content={
                "error": "processing_error",
                "message": f"Failed to process upload: {exc}",
            },
        )

    if not raw_records:
        return FeedbackUploadResponse(
            accepted=0,
            rejected=0,
            total=0,
            message="No valid feedback rows found in the CSV.",
        )

    # 2. Enrich with sentiment + topic.
    for record in raw_records:
        label, score = sentiment_service.analyze(record.feedback_text)
        record.sentiment = label
        record.sentiment_score = score
        record.topic = topic_service.classify(record.feedback_text)

    # 3. Lazily create the vector store on first upload (discover embedding dim).
    if state.vector_store is None:
        first_vec = embeddings.embed_text(raw_records[0].feedback_text)
        from ..services.vector_store import VectorStore

        state.vector_store = VectorStore(dimension=len(first_vec))

    vector_store: VectorStore = state.vector_store

    # 4. Embed all records and add to the index.
    texts = [r.feedback_text for r in raw_records]
    vectors = embeddings.embed_texts(texts)
    ids = [r.id for r in raw_records]
    vector_store.add(vectors, ids)

    # 5. Persist to SQLite + vector store on disk.
    database.insert_records(raw_records, source=source)
    # Load existing records and merge as FeedbackRecord objects (for retriever).
    all_records: Dict[str, FeedbackRecord] = {}
    try:
        existing = database.list_records(page=1, page_size=100_000)
        all_records = {r.id: r for r in existing}
    except Exception:
        pass
    for r in raw_records:
        all_records[r.id] = r
    # Separate dict serialization for the vector-store sidecar file.
    serializable_all = {rid: r.model_dump() for rid, r in all_records.items()}
    vector_store.save(settings.INDEX_PATH, settings.RECORDS_PATH, serializable_all)

    # 6. Rebuild retriever + RAG service so search/QA work immediately after upload.
    state.retriever = RetrieverService(
        state.embeddings, state.vector_store, all_records
    )
    state.rag_service = RAGService(state.retriever, state.llm_provider)
    state.evaluation_service = EvaluationService(
        state.retriever, state.rag_service
    )

    return FeedbackUploadResponse(
        accepted=len(raw_records),
        rejected=0,
        total=len(raw_records),
        message=f"Processed {len(raw_records)} feedback record(s).",
    )


# ---------------------------------------------------------------------- #
# List
# ---------------------------------------------------------------------- #
@router.get("", response_model=FeedbackListResponse)
def list_feedback(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sentiment: str = Query(default=None),
    topic: str = Query(default=None),
    channel: str = Query(default=None),
    source: str = Query(default=None),
    product: str = Query(default=None),
):
    """List processed feedback with optional filters and pagination."""
    state = _get_state(request)
    database: Database = state.database

    items = database.list_records(
        page=page,
        page_size=page_size,
        sentiment=sentiment or None,
        topic=topic or None,
        source=source or None,
        product=product or None,
    )
    total = database.count_records(
        sentiment=sentiment or None,
        topic=topic or None,
        source=source or None,
        product=product or None,
    )

    return FeedbackListResponse(
        records=items,
        total=total,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------- #
# Get one
# ---------------------------------------------------------------------- #
@router.get("/{feedback_id}", response_model=FeedbackRecord)
def get_feedback(feedback_id: str, request: Request):
    state = _get_state(request)
    database: Database = state.database
    record = database.get_record(feedback_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return record


# ---------------------------------------------------------------------- #
# Delete one
# ---------------------------------------------------------------------- #
@router.delete("/{feedback_id}", status_code=204)
def delete_feedback(feedback_id: str, request: Request):
    state = _get_state(request)
    database: Database = state.database
    if not database.get_record(feedback_id):
        raise HTTPException(status_code=404, detail="Feedback not found")
    database.delete_record(feedback_id)
    # Note: vector store is not rebuilt on delete in this MVP to keep it simple;
    # re-upload or restart to rebuild the index without this record.
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------- #
# Semantic search
# ---------------------------------------------------------------------- #
@router.post("/search", response_model=SemanticSearchResponse)
def semantic_search(
    request: Request,
    body: SemanticSearchRequest,
):
    state = _get_state(request)
    retriever: object = state.retriever

    if (
        retriever is None
        or state.vector_store is None
        or state.vector_store.size == 0
    ):
        return SemanticSearchResponse(
            query=body.query,
            top_k=body.top_k,
            results=[],
        )

    hits = retriever.search(body.query, body.top_k)
    results = [
        SemanticSearchResult(feedback=record, score=score) for record, score in hits
    ]
    return SemanticSearchResponse(
        query=body.query,
        top_k=body.top_k,
        results=results,
    )
