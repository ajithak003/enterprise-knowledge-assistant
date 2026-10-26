
"""Incrementally embed documents and synchronize them with Qdrant."""

from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from knowledge_assistant.chunking.chunker import DocumentChunk
from knowledge_assistant.ingestion.fetcher import SourceDocument
from knowledge_assistant.indexing.manifest import IndexManifest


class IncrementalIndexer:
    """Synchronize document chunks with Qdrant and a SQLite manifest."""

    def __init__(
        self,
        manifest: IndexManifest,
        embedder: Any,
        qdrant_client: QdrantClient,
        collection_name: str,
        embedding_model_name: str,
        embedding_dimension: int,
        chunking_version: str,
    ) -> None:
        self.manifest = manifest
        self.embedder = embedder
        self.qdrant = qdrant_client
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_dimension = embedding_dimension
        self.chunking_version = chunking_version

    def _ensure_collection(self) -> None:
        if not self.qdrant.collection_exists(
            collection_name=self.collection_name
        ):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    
    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.embedder.embed_documents(texts)

        if len(vectors) != len(texts):
            raise ValueError(
                f"Expected {len(texts)} embeddings, received {len(vectors)}."
            )

        result = [
            vector.tolist() if hasattr(vector, "tolist") else list(vector)
            for vector in vectors
        ]

        for vector in result:
            if len(vector) != self.embedding_dimension:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"expected {self.embedding_dimension}, "
                    f"received {len(vector)}."
                )

        return result

    def index_document(
            self,
            document: SourceDocument,
            chunks: list[DocumentChunk],
        ) -> dict[str, int | bool]:
            """Index a document idempotently; update the manifest last."""

            self._ensure_collection()

            if any(chunk.source_url != document.url for chunk in chunks):
                raise ValueError("All chunks must belong to the supplied document.")

            document_id = chunks[0].document_id if chunks else None

            if document_id is None:
                # Empty documents can still have existing chunks to remove.
                import hashlib

                document_id = hashlib.sha256(
                    document.url.encode("utf-8")
                ).hexdigest()

            if any(chunk.document_id != document_id for chunk in chunks):
                raise ValueError("Chunks from multiple documents were supplied.")

            old_document = self.manifest.get_document(document_id)
            old_chunks = self.manifest.get_chunks(document_id)
            old_by_id = {item["chunk_id"]: item for item in old_chunks}
            new_by_id = {chunk.chunk_id: chunk for chunk in chunks}

            if len(new_by_id) != len(chunks):
                raise ValueError("Duplicate chunk IDs detected.")

            force_reembed = (
                old_document is not None
                and (
                    old_document["embedding_model"] != self.embedding_model_name
                    or old_document["chunking_version"] != self.chunking_version
                )
            )

            chunks_to_embed = [
                chunk
                for chunk in chunks
                if force_reembed
                or chunk.chunk_id not in old_by_id
                or old_by_id[chunk.chunk_id]["content_hash"] != chunk.content_hash
            ]

            vectors = self._embed([chunk.text for chunk in chunks_to_embed])

            points = []
            for chunk, vector in zip(chunks_to_embed, vectors, strict=True):
                points.append(
                    models.PointStruct(
                        id=chunk.chunk_id,
                        vector=vector,
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "document_id": chunk.document_id,
                            "document_content_hash": chunk.document_content_hash,
                            "content_hash": chunk.content_hash,
                            "source_url": chunk.source_url,
                            "title": chunk.title,
                            "chunk_index": chunk.chunk_index,
                            "text": chunk.text,
                            "embedding_model": self.embedding_model_name,
                            "chunking_version": self.chunking_version,
                        },
                    )
                )

            # Upsert first. If this fails, do not delete old points or update
            # the manifest.
            if points:
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )

            obsolete_ids = [
                chunk_id
                for chunk_id in old_by_id
                if chunk_id not in new_by_id
            ]

            if obsolete_ids:
                self.qdrant.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=obsolete_ids),
                    wait=True,
                )

            # Persist indexing state only after vector operations succeed.
            self.manifest.replace_document_records(
                document_id=document_id,
                source_url=document.url,
                title=document.title,
                document_content_hash=(
                    chunks[0].document_content_hash
                    if chunks
                    else (
                        old_document["document_content_hash"]
                        if old_document
                        else ""
                    )
                ),
                embedding_model=self.embedding_model_name,
                chunking_version=self.chunking_version,
                chunks=[
                    {
                        "chunk_id": chunk.chunk_id,
                        "content_hash": chunk.content_hash,
                        "chunk_index": chunk.chunk_index,
                    }
                    for chunk in chunks
                ],
            )

            return {
                "chunks_total": len(chunks),
                "chunks_embedded": len(chunks_to_embed),
                "chunks_reused": len(chunks) - len(chunks_to_embed),
                "chunks_deleted": len(obsolete_ids),
                "unchanged_document": (
                    old_document is not None
                    and old_document["document_content_hash"]
                    == (chunks[0].document_content_hash if chunks else "")
                    and not force_reembed
                ),
            }
