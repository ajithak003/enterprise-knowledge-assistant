
from unittest.mock import MagicMock

from knowledge_assistant.chunking.chunker import chunk_document
from knowledge_assistant.indexing.indexer import IncrementalIndexer
from knowledge_assistant.ingestion.fetcher import SourceDocument


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed_documents(self, texts):
     self.calls.append(list(texts))
     return [
        [1.0, 0.0, 0.0]
        for _ in texts
    ]


def test_second_ingestion_reuses_unchanged_vectors(tmp_path):
    document = SourceDocument(
        url="https://docs.python.org/3/tutorial/index.html",
        title="Python Tutorial",
        text="Python makes programming accessible. " * 30,
    )
    chunks = chunk_document(
        document,
        chunk_size=100,
        chunk_overlap=20,
    )

    manifest = MagicMock()
    manifest.get_document.return_value = None
    manifest.get_chunks.return_value = []

    embedder = FakeEmbedder()
    qdrant = MagicMock()
    qdrant.collection_exists.return_value = True

    indexer = IncrementalIndexer(
        manifest=manifest,
        embedder=embedder,
        qdrant_client=qdrant,
        collection_name="test_collection",
        embedding_model_name="test-embedding-model",
        embedding_dimension=3,
        chunking_version="char-v1-size100-overlap20",
    )

    first = indexer.index_document(document, chunks)

    manifest.get_document.return_value = {
        "embedding_model": "test-embedding-model",
        "chunking_version": "char-v1-size100-overlap20",
        "document_content_hash": chunks[0].document_content_hash,
    }
    manifest.get_chunks.return_value = [
        {
            "chunk_id": chunk.chunk_id,
            "content_hash": chunk.content_hash,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]

    second = indexer.index_document(document, chunks)

    assert first["chunks_embedded"] == len(chunks)
    assert second["chunks_embedded"] == 0
    assert second["chunks_reused"] == len(chunks)
    assert len(embedder.calls) == 1
