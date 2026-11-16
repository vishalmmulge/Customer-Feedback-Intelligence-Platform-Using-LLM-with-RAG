"""Semantic retrieval over the vector store.

Encodes a natural-language query with Ollama, searches the FAISS / NumPy
index for the most similar feedback vectors, and returns the matching records
with their similarity scores.
"""

from typing import Dict, List, Tuple

from ..core.logging import logger
from ..models.schemas import FeedbackRecord
from .embeddings import EmbeddingsService
from .vector_store import VectorStore


class RetrieverService:
    def __init__(
        self,
        embeddings: EmbeddingsService,
        vector_store: VectorStore,
        records: Dict[str, FeedbackRecord],
    ):
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.records = records

    def search(self, query: str, top_k: int = 5) -> List[Tuple[FeedbackRecord, float]]:
        """Return the top-k most similar feedback records for ``query``."""
        if not query or not query.strip():
            return []
        query_vec = self.embeddings.embed_text(query)
        hits = self.vector_store.search(query_vec, top_k)

        results: List[Tuple[FeedbackRecord, float]] = []
        for record_id, score in hits:
            record = self.records.get(record_id)
            if record is not None:
                results.append((record, score))

        logger.info(f"Retrieved {len(results)} result(s) for query={query!r}")
        return results
