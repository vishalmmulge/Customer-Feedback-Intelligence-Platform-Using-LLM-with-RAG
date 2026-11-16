"""Tests for the vector store (NumPy fallback backend).

faiss-cpu is unavailable on Python 3.15, so the store uses the NumPy
cosine-similarity fallback. The API is identical either way.
"""

import os
import tempfile

import numpy as np
import pytest

from backend.services.vector_store import VectorStore


@pytest.fixture
def store():
    return VectorStore(dimension=3)


def test_add_and_search(store):
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ids = ["a", "b", "c"]
    store.add(vecs, ids)
    assert store.size == 3

    hits = store.search([1.0, 0.0, 0.0], top_k=2)
    assert len(hits) == 2
    # Top hit should be "a" (identical vector -> cosine 1.0).
    assert hits[0][0] == "a"
    assert abs(hits[0][1] - 1.0) < 1e-5


def test_search_returns_sorted_by_score(store):
    store.add([[1.0, 0.0], [0.9, 0.1]], ["x", "y"])
    hits = store.search([1.0, 0.0], top_k=2)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_save_and_load(tmp_path):
    store = VectorStore(dimension=2)
    store.add([[1.0, 0.0], [0.0, 1.0]], ["p", "q"])

    index_path = tmp_path / "index.bin"
    records_path = tmp_path / "records.json"
    store.save(str(index_path), str(records_path), {"p": {}, "q": {}})

    loaded = VectorStore(dimension=2)
    assert loaded.load(str(index_path), str(records_path)) is True
    assert loaded.size == 2
    hits = loaded.search([1.0, 0.0], top_k=1)
    assert hits[0][0] == "p"


def test_load_missing_returns_false(tmp_path, store):
    assert store.load(str(tmp_path / "nope.bin"), str(tmp_path / "nope.json")) is False
