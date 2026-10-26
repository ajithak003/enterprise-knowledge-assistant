
from knowledge_assistant.indexing.manifest import IndexManifest


def test_new_document_is_not_registered(tmp_path):
    manifest = IndexManifest(str(tmp_path / "manifest.sqlite3"))

    assert manifest.get_document("doc-1") is None


def test_replace_document_records_persists_document_and_chunks(tmp_path):
    manifest = IndexManifest(str(tmp_path / "manifest.sqlite3"))

    manifest.replace_document_records(
        document_id="doc-1",
        source_url="https://docs.python.org/test",
        title="Python test",
        document_content_hash="document-hash",
        embedding_model="BAAI/bge-small-en-v1.5",
        chunking_version="char-v1",
        chunks=[
            {
                "chunk_id": "chunk-1",
                "content_hash": "hash-1",
                "chunk_index": 0,
            },
            {
                "chunk_id": "chunk-2",
                "content_hash": "hash-2",
                "chunk_index": 1,
            },
        ],
    )

    document = manifest.get_document("doc-1")
    chunks = manifest.get_chunks("doc-1")

    assert document is not None
    assert document["document_content_hash"] == "document-hash"
    assert document["status"] == "indexed"
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "chunk-1"


def test_replacing_document_records_removes_obsolete_chunks(tmp_path):
    manifest = IndexManifest(str(tmp_path / "manifest.sqlite3"))

    common = {
        "source_url": "https://docs.python.org/test",
        "title": "Python test",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "chunking_version": "char-v1",
    }

    manifest.replace_document_records(
        document_id="doc-1",
        document_content_hash="hash-v1",
        chunks=[
            {"chunk_id": "old-1", "content_hash": "a", "chunk_index": 0},
            {"chunk_id": "old-2", "content_hash": "b", "chunk_index": 1},
        ],
        **common,
    )

    manifest.replace_document_records(
        document_id="doc-1",
        document_content_hash="hash-v2",
        chunks=[
            {"chunk_id": "new-1", "content_hash": "c", "chunk_index": 0},
        ],
        **common,
    )

    chunks = manifest.get_chunks("doc-1")

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "new-1"
