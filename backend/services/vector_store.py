"""Vector store for semantic search.

Uses FAISS (IndexFlatIP on L2-normalised vectors = cosine similarity) when
available. On Python versions / environments where the ``faiss-cpu`` wheel is
not published (for example Python 3.15), it transparently falls back to an
exact NumPy cosine-similarity search. The rest of the app does not care which
backend is active — the API is identical.

Vectors are persisted to disk so the index survives restarts.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np

try:  # FAISS is optional; NumPy fallback keeps the app runnable everywhere.
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover - depends on the installed wheel
    faiss = None  # type: ignore
    _HAS_FAISS = False

from ..core.config import settings
from ..core.logging import logger


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows so dot-product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._ids: List[str] = []
        if _HAS_FAISS:
            self.index = faiss.IndexFlatIP(dimension)
            self.backend = "faiss"
        else:
            self.vectors = np.zeros((0, dimension), dtype=np.float32)
            self.backend = "numpy"
        logger.info(
            f"VectorStore ready: backend={self.backend}, dimension={dimension}"
        )

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def add(self, vectors: List[List[float]], ids: List[str]) -> None:
        if not vectors:
            return
        arr = np.array(vectors, dtype=np.float32)
        arr = _normalise(arr)
        if self.backend == "faiss":
            self.index.add(arr)
        else:
            self.vectors = (
                np.vstack([self.vectors, arr]) if self.vectors.size else arr
            )
        self._ids.extend(ids)
        logger.debug(f"Added {len(vectors)} vector(s) ({self.backend})")

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def search(self, query_vector: List[float], top_k: int) -> List[Tuple[str, float]]:
        if not self._ids:
            return []
        top_k = min(top_k, len(self._ids))
        if top_k <= 0:
            return []

        q = np.array([query_vector], dtype=np.float32)
        q = _normalise(q)

        if self.backend == "faiss":
            scores, indices = self.index.search(q, top_k)
            results: List[Tuple[str, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                results.append((self._ids[int(idx)], float(score)))
            return results

        # NumPy exact cosine-similarity fallback.
        sims = (self.vectors @ q.T).flatten()
        order = np.argsort(-sims)[:top_k]
        results = [
            (self._ids[int(i)], float(sims[int(i)])) for i in order
        ]
        return results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, index_path: str, records_path: str, records: Dict[str, dict]) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if self.backend == "faiss":
            faiss.write_index(self.index, index_path)  # type: ignore[attr-defined]
        else:
            npy_path = os.path.splitext(index_path)[0] + ".npy"
            np.save(npy_path, self.vectors)
        payload = {"ids": self._ids, "records": records}
        with open(records_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        logger.info(
            f"Saved vector store ({len(self._ids)} vectors) via {self.backend}"
        )

    def load(self, index_path: str, records_path: str) -> bool:
        if self.backend == "faiss":
            if not os.path.exists(index_path) or not os.path.exists(records_path):
                return False
            self.index = faiss.read_index(index_path)  # type: ignore[attr-defined]
        else:
            npy_path = os.path.splitext(index_path)[0] + ".npy"
            if not os.path.exists(npy_path) or not os.path.exists(records_path):
                return False
            self.vectors = np.load(npy_path)
        with open(records_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._ids = data.get("ids", [])
        logger.info(f"Loaded vector store ({len(self._ids)} vectors) via {self.backend}")
        return True

    @property
    def size(self) -> int:
        return len(self._ids)
