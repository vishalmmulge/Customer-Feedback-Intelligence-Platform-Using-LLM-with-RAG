"""Evaluation API routes: run RAG evaluation against a labeled dataset."""

from fastapi import APIRouter, Request

from ..core.logging import logger
from ..services.evaluation import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _get_evaluation_service(request: Request) -> EvaluationService:
    svc = request.app.state.evaluation_service
    if svc is None:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="Evaluation service not initialized. Upload feedback first.",
        )
    return svc


@router.post("/run")
def run_evaluation(request: Request, top_k: int = 5):
    """Run the full evaluation suite against the labeled dataset.

    Returns aggregate Precision@K, Recall@K, MRR, and answer-quality metrics,
    plus per-case results.
    """
    svc = _get_evaluation_service(request)
    import os

    from ..core.config import settings

    dataset_path = os.path.join(settings.DATA_DIR, "evaluation_dataset.jsonl")
    result = svc.run_all(dataset_path, top_k=top_k)
    logger.info(f"Evaluation run complete: {result['num_cases']} cases")
    return result
