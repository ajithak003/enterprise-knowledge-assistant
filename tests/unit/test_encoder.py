
import numpy as np
import pytest

from knowledge_assistant.embeddings.encoder import EmbeddingEncoder


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array([3.0, 4.0, 0.0]) / 5.0

        return np.array([
            [3.0, 4.0, 0.0] for _ in texts
        ]) / 5.0


def test_embed_documents_returns_normalized_vectors(monkeypatch):
    monkeypatch.setattr(
        "knowledge_assistant.embeddings.encoder.SentenceTransformer",
        FakeSentenceTransformer,
    )
    encoder = EmbeddingEncoder(
        model_name="test-model",
        expected_dimension=3,
    )

    vectors = encoder.embed_documents(["Python documentation"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 3
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)


def test_embed_query_returns_normalized_vector(monkeypatch):
    monkeypatch.setattr(
        "knowledge_assistant.embeddings.encoder.SentenceTransformer",
        FakeSentenceTransformer,
    )
    encoder = EmbeddingEncoder(
        model_name="test-model",
        expected_dimension=3,
    )

    vector = encoder.embed_query("How do I use Python?")

    assert len(vector) == 3
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_empty_document_list_returns_empty_list(monkeypatch):
    monkeypatch.setattr(
        "knowledge_assistant.embeddings.encoder.SentenceTransformer",
        FakeSentenceTransformer,
    )
    encoder = EmbeddingEncoder(
        model_name="test-model",
        expected_dimension=3,
    )

    assert encoder.embed_documents([]) == []


def test_blank_document_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "knowledge_assistant.embeddings.encoder.SentenceTransformer",
        FakeSentenceTransformer,
    )
    encoder = EmbeddingEncoder(
        model_name="test-model",
        expected_dimension=3,
    )

    with pytest.raises(ValueError, match="empty text"):
        encoder.embed_documents(["   "])


def test_wrong_model_dimension_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "knowledge_assistant.embeddings.encoder.SentenceTransformer",
        FakeSentenceTransformer,
    )

    with pytest.raises(ValueError, match="dimensional embeddings"):
        EmbeddingEncoder(
            model_name="test-model",
            expected_dimension=384,
        )
