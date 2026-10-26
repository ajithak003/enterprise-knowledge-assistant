
"""Persistent SQLite manifest for document and chunk indexing state."""

import sqlite3
from pathlib import Path


class IndexManifest:
    """Track indexed documents, chunks, and pipeline versions."""

    def __init__(self, db_path: str = "data/manifest.sqlite3") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    source_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    document_content_hash TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    chunking_version TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    FOREIGN KEY (document_id)
                        REFERENCES documents(document_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                    ON chunks(document_id);

                CREATE INDEX IF NOT EXISTS idx_documents_status
                    ON documents(status);
                """
            )

    def get_document(self, document_id: str) -> dict | None:
        """Return a document record, or None if it is not registered."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_chunks(self, document_id: str) -> list[dict]:
        """Return all manifest chunk records for a document."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index
                """,
                (document_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def replace_document_records(
        self,
        *,
        document_id: str,
        source_url: str,
        title: str,
        document_content_hash: str,
        embedding_model: str,
        chunking_version: str,
        chunks: list[dict],
    ) -> None:
        """Atomically replace manifest records for one document.

        Call this only after the corresponding Qdrant operations succeed.
        """
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, source_url, title, document_content_hash,
                    embedding_model, chunking_version, status, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'indexed', CURRENT_TIMESTAMP)
                ON CONFLICT(document_id) DO UPDATE SET
                    source_url = excluded.source_url,
                    title = excluded.title,
                    document_content_hash = excluded.document_content_hash,
                    embedding_model = excluded.embedding_model,
                    chunking_version = excluded.chunking_version,
                    status = 'indexed',
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    document_id,
                    source_url,
                    title,
                    document_content_hash,
                    embedding_model,
                    chunking_version,
                ),
            )

            connection.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (document_id,),
            )

            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, content_hash, chunk_index, status
                )
                VALUES (?, ?, ?, ?, 'indexed')
                """,
                [
                    (
                        chunk["chunk_id"],
                        document_id,
                        chunk["content_hash"],
                        chunk["chunk_index"],
                    )
                    for chunk in chunks
                ],
            )

    def delete_document_records(self, document_id: str) -> None:
        """Delete manifest records for a document."""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (document_id,),
            )
            connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
