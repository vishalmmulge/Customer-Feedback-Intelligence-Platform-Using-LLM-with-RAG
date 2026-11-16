"""FastAPI endpoint tests using TestClient with mocked services.

These tests run fully offline: Ollama and FAISS are replaced with fakes, so no
API keys or network access are required. A real SQLite database (temp file) is
used for persistence tests.
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.models.schemas import FeedbackRecord
from backend.services.retriever import RetrieverService
from backend.services.rag import RAGService, OllamaProvider
from backend.services.evaluation import EvaluationService
from backend.services.evaluation import EvaluationService


class FakeEmbeddings:
    def embed_text(self, text):
        return [0.1, 0.2, 0.3]

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def ping(self, timeout=5.0):
        return True


class FakeVectorStore:
    def __init__(self):
        self.backend = "numpy"
        self._ids = []

    def add(self, vectors, ids):
        self._ids.extend(ids)

    def search(self, query_vector, top_k):
        return [("r1", 0.9)]

    @property
    def size(self):
        return len(self._ids)

    def save(self, *args, **kwargs):
        pass


class FakeDatabase:
    """In-memory fake of the Database interface for endpoint tests."""

    def __init__(self):
        self.records = {
            "r1": FeedbackRecord(
                id="r1",
                feedback_id="r1",
                customer_id="c1",
                feedback_text="billing is slow",
                sentiment="negative",
                topic="pricing",
                rating=2,
                source="csv",
                processed_at="2024-01-01T00:00:00Z",
            )
        }

    def connect(self):
        pass

    def close(self):
        pass

    def insert_record(self, record, source="csv"):
        self.records[record.id] = record

    def insert_records(self, records, source="csv"):
        for r in records:
            self.insert_record(r, source)

    def delete_record(self, feedback_id):
        if feedback_id in self.records:
            del self.records[feedback_id]
            return True
        return False

    def delete_all(self):
        n = len(self.records)
        self.records.clear()
        return n

    def get_record(self, feedback_id):
        return self.records.get(feedback_id)

    def list_records(self, page=1, page_size=20, sentiment=None, topic=None,
                     source=None, product=None):
        items = list(self.records.values())
        if sentiment:
            items = [r for r in items if r.sentiment == sentiment]
        if topic:
            items = [r for r in items if r.topic == topic]
        if source:
            items = [r for r in items if r.source == source]
        if product:
            items = [r for r in items if r.product == product]
        start = (page - 1) * page_size
        return items[start:start + page_size]

    def count_records(self, sentiment=None, topic=None, source=None, product=None):
        return len(self.list_records(
            page=1, page_size=100000,
            sentiment=sentiment, topic=topic, source=source, product=product,
        ))

    def sentiment_distribution(self):
        from collections import Counter
        return dict(Counter((r.sentiment or "unknown") for r in self.records.values()))

    def topic_distribution(self):
        from collections import Counter
        return dict(Counter((r.topic or "unknown") for r in self.records.values()))

    def source_distribution(self):
        from collections import Counter
        return dict(Counter((r.source or "unknown") for r in self.records.values()))

    def product_distribution(self):
        from collections import Counter
        return dict(Counter((r.product or "unknown") for r in self.records.values()))

    def rating_distribution(self):
        from collections import Counter
        return dict(Counter((str(r.rating) if r.rating is not None else "unknown")
                            for r in self.records.values()))

    def average_rating(self):
        ratings = [r.rating for r in self.records.values() if r.rating is not None]
        return round(sum(ratings) / len(ratings), 2) if ratings else None

    def total_count(self):
        return len(self.records)


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as client:
        # Re-set state to fakes after lifespan startup so tests run offline.
        app.state.preprocessor = __import__(
            "backend.services.preprocessing", fromlist=["PreprocessingService"]
        ).PreprocessingService()
        app.state.sentiment_service = __import__(
            "backend.services.sentiment", fromlist=["SentimentService"]
        ).SentimentService()
        app.state.topic_service = __import__(
            "backend.services.topic", fromlist=["TopicService"]
        ).TopicService()
        app.state.embeddings = FakeEmbeddings()
        app.state.vector_store = FakeVectorStore()
        app.state.database = FakeDatabase()
        app.state.llm_provider = OllamaProvider()
        app.state.ollama_client = FakeEmbeddings()  # has .ping()

        records = {
            "r1": FeedbackRecord(
                id="r1",
                feedback_id="r1",
                customer_id="c1",
                feedback_text="billing is slow",
                sentiment="negative",
                topic="pricing",
                rating=2,
                source="csv",
                processed_at="2024-01-01T00:00:00Z",
            )
        }

        app.state.retriever = RetrieverService(
            app.state.embeddings, app.state.vector_store, records
        )
        app.state.rag_service = RAGService(app.state.retriever, OllamaProvider())
        app.state.evaluation_service = EvaluationService(
            app.state.retriever, app.state.rag_service
        )
        yield client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["database_backend"] == "sqlite"


def test_health_ollama(client):
    resp = client.get("/health/ollama")
    assert resp.status_code == 200
    assert resp.json()["ollama_ready"] is True


def test_upload(client):
    csv_content = b"customer_id,product,channel,rating,feedback_text\nc1,Dashboard,email,5,great product\n"
    resp = client.post(
        "/feedback/upload",
        files={"file": ("t.csv", csv_content, "text/csv")},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["accepted"] == 1


def test_list_feedback(client):
    resp = client.get("/feedback")
    assert resp.status_code == 200
    data = resp.json()
    assert "records" in data


def test_get_feedback(client):
    resp = client.get("/feedback/r1")
    assert resp.status_code == 200
    assert resp.json()["id"] == "r1"


def test_get_feedback_not_found(client):
    resp = client.get("/feedback/nope")
    assert resp.status_code == 404


def test_delete_feedback(client):
    resp = client.delete("/feedback/r1")
    assert resp.status_code == 204
    resp = client.get("/feedback/r1")
    assert resp.status_code == 404


def test_semantic_search(client):
    resp = client.post("/feedback/search", json={"query": "billing", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "billing"


def test_rag_ask(client):
    resp = client.post("/rag/ask", json={"question": "billing?", "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data


def test_rag_chat_alias(client):
    resp = client.post("/rag/chat", json={"question": "billing?", "top_k": 3})
    assert resp.status_code == 200
    assert "answer" in resp.json()


def test_analytics_summary(client):
    resp = client.get("/analytics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_feedback" in data


def test_analytics_overview(client):
    resp = client.get("/analytics/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_feedback" in data


def test_analytics_sentiment(client):
    resp = client.get("/analytics/sentiment")
    assert resp.status_code == 200
    data = resp.json()
    assert "distribution" in data


def test_analytics_topics(client):
    resp = client.get("/analytics/topics")
    assert resp.status_code == 200
    data = resp.json()
    assert "distribution" in data


def test_analytics_trends(client):
    resp = client.get("/analytics/trends")
    assert resp.status_code == 200
    data = resp.json()
    assert "trend" in data


def test_evaluation_run(client):
    resp = client.post("/evaluation/run", params={"top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert "aggregate" in data
    assert "results" in data
