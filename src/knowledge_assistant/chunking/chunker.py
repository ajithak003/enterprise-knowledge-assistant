
"""Create deterministic chunks with stable, content-based identifiers."""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re
import uuid

from knowledge_assistant.ingestion.fetcher import SourceDocument


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk with source metadata and content fingerprints."""

    chunk_id: str
    document_id: str
    document_content_hash: str
    content_hash: str
    source_url: str
    title: str
    chunk_index: int
    text: str


def normalize_text(text: str) -> str:
    """Normalize whitespace consistently."""
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    """Return the SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document(
    document: SourceDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[DocumentChunk]:
    """Create deterministic, overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    normalized = normalize_text(document.text)
    if not normalized:
        return []

    document_id = sha256_text(document.url)
    document_content_hash = sha256_text(normalized)
    step = chunk_size - chunk_overlap

    chunks: list[DocumentChunk] = []
    occurrences: dict[str, int] = defaultdict(int)

    for index, start in enumerate(range(0, len(normalized), step)):
        chunk_text = normalized[start : start + chunk_size]
        if not chunk_text:
            continue

        chunk_hash = sha256_text(chunk_text)

        # Distinguish identical chunk text occurring multiple times.
        occurrence = occurrences[chunk_hash]
        occurrences[chunk_hash] += 1

        chunk_id = str(
          uuid.uuid5(
           uuid.NAMESPACE_URL,
        f"{document_id}:{chunk_hash}:{occurrence}",
    )
)
    
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                document_content_hash=document_content_hash,
                content_hash=chunk_hash,
                source_url=document.url,
                title=document.title,
                chunk_index=index,
                text=chunk_text,
            )
        )

        if start + chunk_size >= len(normalized):
            break

    return chunks