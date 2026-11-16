"""Tests for the semantic retriever with mocked embeddings + store."""

from backend.services.retriever import RetrieverService
from backend.models.schemas import FeedbackRecord


class FakeEmbeddings:
    def embed_text(self, text):
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self):
        self.backend = "numpy"
        self._ids = ["r1", "r2"]

    def search(self, query_vector, top_k):
        return [("r1", 0.95), ("r2", 0.80)]


def test_retriever_returns_records_with_scores():
    records = {
        "r1": FeedbackRecord(id="r1", customer_id="c1", feedback_text="first", sentiment="positive"),
        "r2": FeedbackRecord(id="r2", customer_id="c2", feedback_text="second", sentiment="negative"),
    }
    retriever = RetrieverService(FakeEmbeddings(), FakeVectorStore(), records)
    hits = retriever.search("query", top_k=5)
    assert len(hits) == 2
    assert hits[0][0].id == "r1"
    assert abs(hits[0][1] - 0.95) < 1e-6
    assert hits[1][0].id == "r2"
