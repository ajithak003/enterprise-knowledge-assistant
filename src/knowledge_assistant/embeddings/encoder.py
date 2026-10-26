
"""Generate normalized text embeddings using a Sentence Transformers model."""

from collections.abc import Sequence

from sentence_transformers import SentenceTransformer


class EmbeddingEncoder:
    """Load an embedding model once and reuse it for all encoding requests."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        expected_dimension: int = 384,
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        actual_dimension = self.model.get_embedding_dimension()
        if actual_dimension != expected_dimension:
            raise ValueError(
                f"Model {model_name!r} produces {actual_dimension}-dimensional "
                f"embeddings; expected {expected_dimension}."
            )

        self.dimension = actual_dimension

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Encode document chunks as normalized vectors."""
        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input must not contain empty text.")

        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode a search query using the same embedding model."""
        if not query.strip():
            raise ValueError("Query must not be empty.")

        vector = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return vector.tolist()
