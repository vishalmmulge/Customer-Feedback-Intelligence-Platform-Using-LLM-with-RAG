"""Pydantic data models and API contracts.

These schemas define the shape of every request and response in the system.
They serve as both validation rules and self-documenting API contracts.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------- #
# Internal domain model
# ---------------------------------------------------------------------- #
class FeedbackRecord(BaseModel):
    """A single piece of customer feedback, enriched by the pipeline."""

    model_config = ConfigDict(from_attributes=True)

    # Primary key / unique identifier
    id: str = Field(default_factory=lambda: str(uuid4()))
    # Alias stored in SQLite as feedback_id
    feedback_id: Optional[str] = None

    customer_id: str
    feedback_text: str
    source: Optional[str] = None
    product: Optional[str] = None
    channel: Optional[str] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    timestamp: Optional[str] = None

    # Enriched by services
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    topic: Optional[str] = None

    # Audit
    processed_at: Optional[str] = None

    @property
    def feedback_id_value(self) -> str:
        """Return the stable unique id used for persistence."""
        return self.feedback_id or self.id


# ---------------------------------------------------------------------- #
# Request / response schemas
# ---------------------------------------------------------------------- #
class FeedbackUploadResponse(BaseModel):
    accepted: int
    rejected: int
    total: int
    message: str
    errors: List[Dict[str, Any]] = []


class FeedbackListResponse(BaseModel):
    records: List[FeedbackRecord]
    total: int
    page: int
    page_size: int


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language search query")
    top_k: int = Field(default=5, ge=1, le=50)


class SemanticSearchResult(BaseModel):
    feedback: FeedbackRecord
    score: float


class SemanticSearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SemanticSearchResult]


class RAGRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RAGSource(BaseModel):
    id: str
    feedback_text: str
    sentiment: Optional[str] = None
    score: float


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: List[RAGSource]


class AnalyticsSummary(BaseModel):
    total_feedback: int
    average_rating: Optional[float] = None
    sentiment_distribution: Dict[str, int]
    topic_distribution: Dict[str, int]
    source_distribution: Dict[str, int]
    product_distribution: Dict[str, int]
    rating_distribution: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_configured: bool
    ollama_ready: bool
    vector_backend: str
    database_backend: str
