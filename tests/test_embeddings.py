"""Tests for the embeddings service with a mocked Ollama client."""

from backend.services.embeddings import EmbeddingsService, OllamaClient


class FakeOllamaClient(OllamaClient):
    def embed(self, model, input_text):
        # Deterministic fake embedding based on text length.
        return [float(len(input_text)), 0.1, 0.2]

    def chat(self, model, messages, temperature=0.2):
        return "fake answer"


def test_embed_text_returns_vector():
    svc = EmbeddingsService(client=FakeOllamaClient())
    vec = svc.embed_text("hello world")
    assert isinstance(vec, list)
    assert len(vec) == 3
    assert vec[0] == 11.0  # length of "hello world"


def test_embed_texts_returns_list_of_vectors():
    svc = EmbeddingsService(client=FakeOllamaClient())
    vecs = svc.embed_texts(["a", "bb", "ccc"])
    assert len(vecs) == 3
    assert vecs[0][0] == 1.0
    assert vecs[1][0] == 2.0
    assert vecs[2][0] == 3.0
