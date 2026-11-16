"""Tests for the RAG service with mocked retriever + LLM provider."""

from backend.models.schemas import FeedbackRecord
from backend.services.rag import RAGService, OllamaProvider


class FakeRetriever:
    def search(self, question, top_k):
        r = FeedbackRecord(
            id="r1",
            customer_id="c1",
            feedback_text="The billing page is slow.",
            sentiment="negative",
            rating=2,
        )
        return [(r, 0.9)]


class FakeLLM:
    def generate(self, prompt):
        return "Customers are unhappy about slow billing."


def test_rag_returns_answer_and_sources():
    rag = RAGService(FakeRetriever(), FakeLLM())
    result = rag.answer("Why are customers unhappy?", top_k=3)
    assert "slow billing" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["id"] == "r1"
