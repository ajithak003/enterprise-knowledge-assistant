
"""Orchestrate document fetching, chunking, embedding, and indexing."""

from qdrant_client import QdrantClient

from knowledge_assistant.chunking.chunker import chunk_document
from knowledge_assistant.config import settings
from knowledge_assistant.embeddings.encoder import EmbeddingEncoder
from knowledge_assistant.indexing.indexer import IncrementalIndexer
from knowledge_assistant.indexing.manifest import IndexManifest
from knowledge_assistant.ingestion.fetcher import fetch_document


class IngestionPipeline:
    """Ingest public documentation into the configured vector collection."""

    def __init__(self) -> None:
        self.encoder = EmbeddingEncoder(
            model_name=settings.embedding_model,
            expected_dimension=settings.embedding_dimension,
        )
        self.manifest = IndexManifest()
        self.qdrant = QdrantClient(url=settings.qdrant_url)

        self.indexer = IncrementalIndexer(
            manifest=self.manifest,
            embedder=self.encoder,
            qdrant_client=self.qdrant,
            collection_name=settings.qdrant_collection,
            embedding_model_name=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
            chunking_version=(
                f"char-v1-size{settings.chunk_size}"
                f"-overlap{settings.chunk_overlap}"
            ),
        )

    def ingest_url(self, url: str) -> dict[str, int | bool]:
        """Fetch and index one approved public documentation URL."""
        document = fetch_document(url)

        chunks = chunk_document(
            document,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        return self.indexer.index_document(document, chunks)

    def close(self) -> None:
        """Release the Qdrant client."""
        self.qdrant.close()
