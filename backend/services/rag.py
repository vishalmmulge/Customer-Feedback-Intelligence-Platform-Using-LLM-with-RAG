"""Retrieval-Augmented Generation (RAG) question answering.

Provider-independent by design:

* ``LLMProvider`` is an abstract interface with a single ``generate`` method.
* ``OllamaProvider`` implements it against a local Ollama instance.
* Selection happens through environment variables, never hardcoded keys.

The RAG service retrieves relevant feedback via the retriever, builds a
grounded prompt, and asks the LLM to answer using only that context.
"""

from abc import ABC, abstractmethod
from typing import List

from ..core.config import settings
from ..core.exceptions import ProviderError
from ..core.logging import logger
from ..models.schemas import FeedbackRecord
from .embeddings import OllamaClient
from .retriever import RetrieverService


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the LLM's text completion for ``prompt``."""


class OllamaProvider(LLMProvider):
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()
        self.model = settings.OLLAMA_LLM_MODEL
        self.temperature = settings.RAG_TEMPERATURE

    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful customer-insights analyst. Answer the "
                    "user's question using ONLY the provided customer feedback "
                    "context. Do not invent facts. If the context is "
                    "insufficient, say so. Be concise and cite the feedback "
                    "naturally."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self.client.chat(self.model, messages, self.temperature)


class RAGService:
    def __init__(self, retriever: RetrieverService, llm: LLMProvider):
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, top_k: int | None = None) -> dict:
        top_k = top_k or settings.RAG_TOP_K

        # 1. Retrieve relevant feedback.
        try:
            hits = self.retriever.search(question, top_k)
        except ProviderError as exc:
            logger.warning(f"Retrieval failed for RAG: {exc.message}")
            return {
                "question": question,
                "answer": f"I couldn't search the feedback index: {exc.message}",
                "sources": [],
            }

        if not hits:
            return {
                "question": question,
                "answer": "No relevant feedback found to answer your question.",
                "sources": [],
            }

        # 2. Build a grounded context block.
        context_lines = []
        for i, (record, score) in enumerate(hits, 1):
            rating = f"{record.rating}/5" if record.rating is not None else "n/a"
            context_lines.append(
                f"[{i}] (rating {rating}, sentiment {record.sentiment or 'n/a'}): "
                f"{record.feedback_text}"
            )
        context = "\n\n".join(context_lines)

        prompt = (
            "Answer the question using only the customer feedback below. "
            "Do not invent information. If the feedback is insufficient, say so.\n\n"
            f"Customer Feedback:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        # 3. Generate the answer.
        try:
            answer = self.llm.generate(prompt)
        except ProviderError as exc:
            logger.warning(f"LLM generation failed: {exc.message}")
            answer = f"I couldn't generate an answer: {exc.message}"

        # 4. Package sources.
        sources = [
            {
                "id": record.id,
                "feedback_text": record.feedback_text,
                "sentiment": record.sentiment,
                "score": round(score, 3),
            }
            for record, score in hits
        ]

        return {"question": question, "answer": answer, "sources": sources}
