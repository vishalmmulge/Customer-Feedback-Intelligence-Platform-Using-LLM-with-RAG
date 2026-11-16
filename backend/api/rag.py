"""RAG question-answering route."""

from fastapi import APIRouter, Request

from ..core.exceptions import ProviderError
from ..core.logging import logger
from ..models.schemas import RAGRequest, RAGResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ask", response_model=RAGResponse)
def ask_question(request: Request, body: RAGRequest):
    """Answer a natural-language question using retrieved feedback context."""
    state = request.app.state
    rag_service: object = state.rag_service

    if rag_service is None:
        return RAGResponse(
            question=body.question,
            answer="RAG service is not initialized yet. Upload feedback first.",
            sources=[],
        )

    try:
        result = rag_service.answer(body.question, body.top_k)
    except ProviderError as exc:
        logger.warning(f"RAG provider error: {exc.message}")
        return RAGResponse(
            question=body.question,
            answer=f"I couldn't generate an answer: {exc.message}",
            sources=[],
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("RAG request failed")
        return RAGResponse(
            question=body.question,
            answer=f"An unexpected error occurred: {exc}",
            sources=[],
        )

    return RAGResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[
            {
                "id": s["id"],
                "feedback_text": s["feedback_text"],
                "sentiment": s["sentiment"],
                "score": s["score"],
            }
            for s in result["sources"]
        ],
    )


@router.post("/chat", response_model=RAGResponse)
def chat(request: Request, body: RAGRequest):
    """Alias for /rag/ask. Natural-language question -> grounded RAG answer."""
    return ask_question(request, body)
