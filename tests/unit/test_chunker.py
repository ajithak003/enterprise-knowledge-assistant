
import pytest
import uuid


from knowledge_assistant.chunking.chunker import (
    chunk_document,
    normalize_text,
    sha256_text,
)
from knowledge_assistant.ingestion.fetcher import SourceDocument


def make_document(text: str) -> SourceDocument:
    return SourceDocument(
        url="https://docs.python.org/3/tutorial/index.html",
        title="Python Tutorial",
        text=text,
    )


def test_normalize_text():
    assert normalize_text("  Python \n   is\t great.  ") == "Python is great."


def test_content_hash_is_deterministic():
    assert sha256_text("Python") == sha256_text("Python")
    assert sha256_text("Python") != sha256_text("Java")



def test_chunking_preserves_metadata_and_overlap():
    document = make_document("abcdefghij" * 5)

    chunks = chunk_document(document, chunk_size=20, chunk_overlap=5)

    for chunk in chunks:
        uuid.UUID(chunk.chunk_id)  # Raises ValueError if the ID is invalid.
    
    assert len(chunks) > 1
    assert chunks[0].text[-5:] == chunks[1].text[:5]

    assert all(c.source_url == document.url for c in chunks)
    assert all(c.title == document.title for c in chunks)

    # Every chunk belongs to the same source document.
    assert all(
        c.document_content_hash == chunks[0].document_content_hash
        for c in chunks
    )

    # Each chunk hash must match its own text.
    assert all(
        c.content_hash == sha256_text(c.text)
        for c in chunks
    )

    # Each chunk ID must be unique.
    assert len({c.chunk_id for c in chunks}) == len(chunks)

def test_empty_document_returns_no_chunks():
    assert chunk_document(make_document("   \n ")) == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, 10), (10, 11), (10, -1)],
)
def test_invalid_chunk_settings(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_document(
            make_document("some content"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )