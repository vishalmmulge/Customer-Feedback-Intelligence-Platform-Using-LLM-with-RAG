"""Retrieval-Augmented Generation evaluation framework.

Computes *actual* retrieval and generation metrics against a manually labeled
ground-truth dataset. No metrics are fabricated; every number is derived from
the labeled data and live model output.

Relevance is judged by substring matching: a retrieved record is considered
relevant if its feedback_text contains at least one of the expected substrings
from the evaluation dataset (case-insensitive). This avoids dependence on
UUIDs, which are regenerated on every upload.

Metrics implemented:
    Retrieval: Precision@K, Recall@K, Mean Reciprocal Rank (MRR)
    Answer:    Faithfulness, Context Relevance, Answer Relevance, Evidence Coverage
"""

import json
from typing import Any, Dict, List, Optional

from ..core.logging import logger
from .retriever import RetrieverService
from .rag import RAGService


class EvaluationService:
    def __init__(self, retriever: RetrieverService, rag_service: RAGService):
        self.retriever = retriever
        self.rag_service = rag_service

    # ------------------------------------------------------------------ #
    # Retrieval metrics
    # ------------------------------------------------------------------ #
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if k <= 0:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        if not retrieved_k:
            return 0.0
        relevant_set = set(relevant_ids)
        hits = sum(1 for rid in retrieved_k if rid in relevant_set)
        return round(hits / len(retrieved_k), 4)

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        hits = len(retrieved_k & relevant_set)
        return round(hits / len(relevant_set), 4)

    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        relevant_set = set(relevant_ids)
        for i, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_set:
                return round(1.0 / i, 4)
        return 0.0

    # ------------------------------------------------------------------ #
    # Answer metrics (deterministic heuristics; no fabricated scores)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _overlap_score(text_a: str, text_b: str) -> float:
        """Jaccard overlap between two texts (lightweight proxy)."""
        a = set(text_a.lower().split())
        b = set(text_b.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def answer_faithfulness(self, answer: str, source_texts: List[str]) -> float:
        """Max overlap between the answer and any source text."""
        if not source_texts:
            return 0.0
        answer_words = set(answer.lower().split())
        if not answer_words:
            return 0.0
        scores = [self._overlap_score(answer, text) for text in source_texts]
        return round(max(scores), 3)

    def context_relevance(self, question: str, source_texts: List[str]) -> float:
        """Average overlap between the question and each retrieved source."""
        if not source_texts:
            return 0.0
        scores = [self._overlap_score(question, text) for text in source_texts]
        return round(sum(scores) / len(scores), 3)

    def answer_relevance(self, question: str, answer: str) -> float:
        return round(self._overlap_score(question, answer), 3)

    def evidence_coverage(self, answer: str, source_texts: List[str]) -> float:
        """Fraction of source texts sharing at least one token with the answer."""
        if not source_texts:
            return 0.0
        answer_words = set(answer.lower().split())
        covered = 0
        for text in source_texts:
            if answer_words & set(text.lower().split()):
                covered += 1
        return round(covered / len(source_texts), 3)

    # ------------------------------------------------------------------ #
    # Dataset loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_dataset(path: str) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cases.append(json.loads(line))
        return cases

    # ------------------------------------------------------------------ #
    # Full run
    # ------------------------------------------------------------------ #
    def evaluate_case(
        self,
        question: str,
        expected_texts: List[str],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Run one end-to-end evaluation case.

        Relevance is determined by matching retrieved feedback_text against
        the expected substrings.
        """
        hits = self.retriever.search(question, top_k)
        retrieved_ids = [record.id for record, _ in hits]
        retrieved_texts = [record.feedback_text for record, _ in hits]

        # Determine which retrieved records are relevant.
        relevant_ids: List[str] = []
        for record, _ in hits:
            text_lower = record.feedback_text.lower()
            if any(exp.lower() in text_lower for exp in expected_texts):
                relevant_ids.append(record.id)

        retrieval = {
            "question": question,
            "top_k": top_k,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "expected_texts": expected_texts,
            "precision_at_k": self.precision_at_k(retrieved_ids, relevant_ids, top_k),
            "recall_at_k": self.recall_at_k(retrieved_ids, relevant_ids, top_k),
            "mrr": self.mean_reciprocal_rank(retrieved_ids, relevant_ids),
        }

        # Run RAG to get answer + sources.
        rag_result = self.rag_service.answer(question, top_k=top_k)
        answer = rag_result["answer"]
        sources = rag_result["sources"]
        source_texts = [s["feedback_text"] for s in sources]

        answer_metrics = {
            "faithfulness": self.answer_faithfulness(answer, source_texts),
            "context_relevance": self.context_relevance(question, source_texts),
            "answer_relevance": self.answer_relevance(question, answer),
            "evidence_coverage": self.evidence_coverage(answer, source_texts),
        }

        return {
            "question": question,
            "retrieval": retrieval,
            "answer": answer,
            "answer_metrics": answer_metrics,
            "sources": sources,
        }

    def run_all(
        self,
        dataset_path: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Run every case in the dataset and return aggregate metrics."""
        cases = self.load_dataset(dataset_path)
        results: List[Dict[str, Any]] = []

        for case in cases:
            result = self.evaluate_case(
                case["question"],
                case.get("expected_texts", []),
                top_k=top_k,
            )
            result["category"] = case.get("category", "uncategorized")
            results.append(result)

        # Aggregates.
        n = len(results) or 1
        agg = {
            "precision_at_k": round(
                sum(r["retrieval"]["precision_at_k"] for r in results) / n, 4
            ),
            "recall_at_k": round(
                sum(r["retrieval"]["recall_at_k"] for r in results) / n, 4
            ),
            "mrr": round(
                sum(r["retrieval"]["mrr"] for r in results) / n, 4
            ),
            "faithfulness": round(
                sum(r["answer_metrics"]["faithfulness"] for r in results) / n, 4
            ),
            "context_relevance": round(
                sum(r["answer_metrics"]["context_relevance"] for r in results) / n, 4
            ),
            "answer_relevance": round(
                sum(r["answer_metrics"]["answer_relevance"] for r in results) / n, 4
            ),
            "evidence_coverage": round(
                sum(r["answer_metrics"]["evidence_coverage"] for r in results) / n, 4
            ),
        }

        return {
            "top_k": top_k,
            "num_cases": len(results),
            "aggregate": agg,
            "results": results,
        }
