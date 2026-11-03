"""Storage layer.

* **SQLite** (single file ``data/library.sqlite``) is the source of truth for
  documents and chunks, and provides **BM25 keyword search** through the
  built-in FTS5 extension (``unicode61`` tokenizer handles Cyrillic + Latin,
  ``porter`` stems English words, diacritics are folded).
* **LanceDB** (folder ``data/vectors.lance``) stores the dense embedding
  vectors and answers **nearest-neighbour** queries. Below
  ``ann_index_threshold`` chunks it does an exact (brute-force) cosine search,
  which is both fast and 100 % accurate at that size; above it, ``rag optimize``
  builds an HNSW index (IVF_HNSW_SQ = HNSW graph + 8-bit scalar quantization).

Both stores share the chunk id as join key and run embedded in-process: no
server, no Docker, everything stays on your disk.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa

from .config import Config

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1"
VECTOR_TABLE = "chunks"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT UNIQUE NOT NULL,
    sha256      TEXT NOT NULL,
    title       TEXT NOT NULL,
    author      TEXT,
    category    TEXT,
    tags        TEXT,
    language    TEXT,
    file_type   TEXT,
    n_pages     INTEGER,
    n_chunks    INTEGER DEFAULT 0,
    n_chars     INTEGER DEFAULT 0,
    size_bytes  INTEGER,
    mtime       REAL,
    warnings    TEXT,
    indexed_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    chapter     TEXT,
    page_start  INTEGER,
    page_end    INTEGER,
    n_tokens    INTEGER,
    section_idx INTEGER,
    char_start  INTEGER,
    char_end    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_ord ON chunks(doc_id, ord);

-- BM25 keyword index. rowid = chunks.id. Columns are weighted at query time.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    body, title, chapter,
    tokenize = "porter unicode61 remove_diacritics 2"
);
"""

_WORD = re.compile(r"\w+", re.UNICODE)


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query: 'how to sleep?' -> '"how" OR "to" OR "sleep"'."""
    words = [w for w in _WORD.findall(text.lower()) if len(w) > 1 or w.isdigit()]
    seen, uniq = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return " OR ".join(f'"{w}"' for w in uniq[:64])


class Store:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.ensure_dirs()
        self._local = threading.local()
        self._init_sqlite()
        self._lance = None

    # ------------------------------------------------------------------ SQLite
    @property
    def db(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.cfg.sqlite_path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_sqlite(self) -> None:
        self.db.executescript(_SCHEMA)
        self.set_meta("schema_version", SCHEMA_VERSION)
        self.db.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    # ---------------------------------------------------------------- LanceDB
    @property
    def lance(self):
        if self._lance is None:
            import lancedb

            # read_consistency_interval=0 -> always see writes made by other processes
            self._lance = lancedb.connect(self.cfg.lancedb_path, read_consistency_interval=timedelta(0))
        return self._lance

    def _vector_table(self, dim: int | None = None, create: bool = False):
        names = self.lance.table_names() if hasattr(self.lance, "table_names") else self.lance.list_tables()
        if VECTOR_TABLE in list(names):
            return self.lance.open_table(VECTOR_TABLE)
        if not create:
            return None
        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("doc_id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )
        return self.lance.create_table(VECTOR_TABLE, schema=schema)

    # ---------------------------------------------------------------- writes
    def check_embedding_model(self, model: str, dim: int) -> None:
        stored = self.get_meta("embedding_model")
        if stored and stored != model:
            raise RuntimeError(
                f"The index was built with '{stored}' but config uses '{model}'. "
                "Run `rag sync --rebuild` to re-embed the whole library with the new model."
            )
        if not stored:
            self.set_meta("embedding_model", model)
            self.set_meta("embedding_dim", str(dim))
            self.db.commit()

    def add_document(self, doc: dict[str, Any], chunks: list[dict[str, Any]], vectors: np.ndarray) -> int:
        """Insert a document, its chunks, FTS rows and vectors atomically (best effort across stores)."""
        db = self.db
        cur = db.cursor()
        try:
            cur.execute("BEGIN")
            cur.execute(
                """INSERT INTO documents(path, sha256, title, author, category, tags, language, file_type,
                                         n_pages, n_chunks, n_chars, size_bytes, mtime, warnings)
                   VALUES (:path, :sha256, :title, :author, :category, :tags, :language, :file_type,
                           :n_pages, :n_chunks, :n_chars, :size_bytes, :mtime, :warnings)""",
                doc,
            )
            doc_id = cur.lastrowid
            ids = []
            for ch in chunks:
                cur.execute(
                    """INSERT INTO chunks(doc_id, ord, text, chapter, page_start, page_end, n_tokens,
                                          section_idx, char_start, char_end)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (doc_id, ch["ord"], ch["text"], ch["chapter"], ch["page_start"], ch["page_end"],
                     ch["n_tokens"], ch["section_idx"], ch["char_start"], ch["char_end"]),
                )
                cid = cur.lastrowid
                ids.append(cid)
                author = f" {doc['author']}" if doc.get("author") else ""
                cur.execute(
                    "INSERT INTO chunks_fts(rowid, body, title, chapter) VALUES (?,?,?,?)",
                    (cid, ch["text"], doc["title"] + author, ch["chapter"] or ""),
                )
            if len(ids):
                table = self._vector_table(dim=vectors.shape[1], create=True)
                table.add(
                    pa.table(
                        {
                            "id": pa.array(ids, pa.int64()),
                            "doc_id": pa.array([doc_id] * len(ids), pa.int64()),
                            "vector": pa.FixedSizeListArray.from_arrays(
                                pa.array(vectors.reshape(-1), pa.float32()), vectors.shape[1]
                            ),
                        }
                    )
                )
            db.commit()
            return doc_id
        except Exception:
            db.rollback()
            if "doc_id" in locals():
                self._delete_vectors(doc_id)
            raise

    def _delete_vectors(self, doc_id: int) -> None:
        table = self._vector_table()
        if table is not None:
            table.delete(f"doc_id = {int(doc_id)}")

    def delete_document(self, doc_id: int) -> None:
        db = self.db
        ids = [r[0] for r in db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
        db.executemany("DELETE FROM chunks_fts WHERE rowid=?", [(i,) for i in ids])
        db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        db.commit()
        self._delete_vectors(doc_id)

    def update_document_location(self, doc_id: int, path: str, category: str | None, mtime: float) -> None:
        self.db.execute("UPDATE documents SET path=?, category=?, mtime=? WHERE id=?", (path, category, mtime, doc_id))
        self.db.commit()

    def reset(self) -> None:
        """Delete the whole index (books on disk are untouched)."""
        db = self.db
        db.executescript("DELETE FROM chunks_fts; DELETE FROM chunks; DELETE FROM documents; DELETE FROM meta;")
        db.commit()
        if self._vector_table() is not None:
            self.lance.drop_table(VECTOR_TABLE)
        self.set_meta("schema_version", SCHEMA_VERSION)
        db.commit()

    # ---------------------------------------------------------------- reads
    def documents(self) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM documents ORDER BY category, title").fetchall()
        return [dict(r) for r in rows]

    def document_by_path(self, path: str) -> dict[str, Any] | None:
        r = self.db.execute("SELECT * FROM documents WHERE path=?", (path,)).fetchone()
        return dict(r) if r else None

    def documents_by_hash(self, sha: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM documents WHERE sha256=?", (sha,))]

    def filter_doc_ids(
        self, book: str | None = None, author: str | None = None, category: str | None = None
    ) -> list[int] | None:
        """Metadata filtering: return matching document ids (None = no filter)."""
        if not (book or author or category):
            return None
        clauses, params = [], []
        if book:
            clauses.append("(title LIKE ? OR path LIKE ?)")
            params += [f"%{book}%", f"%{book}%"]
        if author:
            clauses.append("author LIKE ?")
            params.append(f"%{author}%")
        if category:
            clauses.append("(category LIKE ? OR tags LIKE ?)")
            params += [f"%{category}%", f"%{category}%"]
        sql = "SELECT id FROM documents WHERE " + " AND ".join(clauses)
        return [r[0] for r in self.db.execute(sql, params)]

    def get_chunks(self, ids: Iterable[int]) -> dict[int, dict[str, Any]]:
        ids = list(ids)
        if not ids:
            return {}
        out: dict[int, dict[str, Any]] = {}
        for i in range(0, len(ids), 500):
            part = ids[i : i + 500]
            q = f"""SELECT c.*, d.title, d.author, d.category, d.path, d.file_type
                    FROM chunks c JOIN documents d ON d.id = c.doc_id
                    WHERE c.id IN ({','.join('?' * len(part))})"""
            for r in self.db.execute(q, part):
                out[r["id"]] = dict(r)
        return out

    def neighbor_chunks(self, doc_id: int, ord_from: int, ord_to: int) -> list[dict[str, Any]]:
        q = """SELECT c.*, d.title, d.author, d.category, d.path, d.file_type
               FROM chunks c JOIN documents d ON d.id = c.doc_id
               WHERE c.doc_id=? AND c.ord BETWEEN ? AND ? ORDER BY c.ord"""
        return [dict(r) for r in self.db.execute(q, (doc_id, ord_from, ord_to))]

    def keyword_search(self, query: str, limit: int, doc_ids: list[int] | None = None) -> list[tuple[int, float]]:
        """BM25 via FTS5. Returns [(chunk_id, score)] best first (higher = better)."""
        match = fts_query(query)
        if not match:
            return []
        # bm25() weights per column: body, title, chapter. FTS5 returns lower = better.
        sql = "SELECT rowid, bm25(chunks_fts, 1.0, 0.5, 0.8) AS s FROM chunks_fts WHERE chunks_fts MATCH ?"
        params: list[Any] = [match]
        if doc_ids is not None:
            if not doc_ids:
                return []
            sql += f" AND rowid IN (SELECT id FROM chunks WHERE doc_id IN ({','.join('?' * len(doc_ids))}))"
            params += doc_ids
        sql += " ORDER BY s LIMIT ?"
        params.append(limit)
        try:
            return [(r[0], -r[1]) for r in self.db.execute(sql, params)]
        except sqlite3.OperationalError as exc:
            log.warning("FTS query failed (%s) for %r", exc, match)
            return []

    def vector_search(
        self, vector: np.ndarray, limit: int, doc_ids: list[int] | None = None
    ) -> list[tuple[int, float]]:
        """Cosine nearest neighbours. Returns [(chunk_id, similarity)] best first."""
        table = self._vector_table()
        if table is None:
            return []
        if doc_ids is not None and not doc_ids:
            return []
        q = table.search(vector.astype(np.float32), vector_column_name="vector").metric("cosine").limit(limit)
        if doc_ids is not None:
            q = q.where(f"doc_id IN ({','.join(str(int(d)) for d in doc_ids)})", prefilter=True)
        rows = q.select(["id", "_distance"]).to_list()
        return [(int(r["id"]), 1.0 - float(r["_distance"])) for r in rows]

    # ---------------------------------------------------------------- maintenance
    def stats(self) -> dict[str, Any]:
        db = self.db
        n_docs = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_tokens = db.execute("SELECT COALESCE(SUM(n_tokens),0) FROM chunks").fetchone()[0]
        cats = db.execute(
            "SELECT COALESCE(category,'(root)') c, COUNT(*) n FROM documents GROUP BY c ORDER BY n DESC"
        ).fetchall()
        table = self._vector_table()
        n_vectors = table.count_rows() if table is not None else 0
        indexed = False
        if table is not None:
            try:
                indexed = bool(table.list_indices())
            except Exception:
                pass

        def du(p: Path) -> int:
            if p.is_file():
                return p.stat().st_size
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0

        return {
            "documents": n_docs,
            "chunks": n_chunks,
            "tokens": n_tokens,
            "vectors": n_vectors,
            "ann_index": indexed,
            "embedding_model": self.get_meta("embedding_model"),
            "embedding_dim": self.get_meta("embedding_dim"),
            "categories": {r["c"]: r["n"] for r in cats},
            "sqlite_mb": round(du(self.cfg.sqlite_path) / 1e6, 1),
            "vectors_mb": round(du(self.cfg.lancedb_path) / 1e6, 1),
        }

    def optimize(self, force_index: bool = False) -> str:
        """Compact storage and (if large enough) build the ANN index."""
        msgs = []
        self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        self.db.commit()
        msgs.append("FTS5 index optimized.")
        table = self._vector_table()
        if table is None:
            return "Nothing to optimize yet."
        try:
            table.optimize()
            msgs.append("Vector table compacted.")
        except Exception as exc:  # pragma: no cover
            msgs.append(f"Compaction skipped ({exc}).")
        n = table.count_rows()
        if n >= self.cfg.retrieval.ann_index_threshold or (force_index and n >= 256):
            table.create_index(metric="cosine", index_type="IVF_HNSW_SQ", vector_column_name="vector", replace=True)
            msgs.append(f"Built IVF_HNSW_SQ index over {n} vectors.")
        else:
            msgs.append(
                f"{n} vectors < threshold {self.cfg.retrieval.ann_index_threshold}: exact search is used "
                "(fast and 100% recall at this size)."
            )
        return " ".join(msgs)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
