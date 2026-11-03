"""Incremental library synchronisation.

``sync()`` makes the index mirror the ``library/`` folder:

* new files        -> load, chunk, embed, index
* changed files    -> (content hash differs) old version removed, re-indexed
* moved / renamed  -> (same hash, new path) only the path/category is updated,
                      nothing is re-embedded
* deleted files    -> removed from the index
* unchanged files  -> skipped in milliseconds (size + mtime, then sha256)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .chunking import Chunk, chunk_document, context_header, make_token_counter
from .config import Config
from .loaders import load_document
from .models import Embedder
from .store import Store, dumps

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    moved: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)
    chunks_added: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        s = (
            f"added {len(self.added)}, updated {len(self.updated)}, moved {len(self.moved)}, "
            f"removed {len(self.removed)}, unchanged {self.unchanged}, failed {len(self.failed)} "
            f"- {self.chunks_added} new chunks in {self.seconds:.1f}s"
        )
        return s


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan_library(cfg: Config) -> list[Path]:
    files = []
    for p in cfg.library_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in cfg.extensions:
            continue
        rel_parts = p.relative_to(cfg.library_dir).parts
        if any(part.startswith((".", "~$", "_")) for part in rel_parts):
            continue  # hidden folders, Office lock files, "_ignored" folders
        if p.name.lower() == "readme.md" and p.parent == cfg.library_dir:
            continue
        files.append(p)
    return sorted(files)


def rel_key(cfg: Config, path: Path) -> str:
    return path.relative_to(cfg.library_dir).as_posix()


def category_of(cfg: Config, path: Path) -> str | None:
    parts = path.relative_to(cfg.library_dir).parts
    return "/".join(parts[:-1]) if len(parts) > 1 else None


class Indexer:
    def __init__(self, cfg: Config, store: Store | None = None, embedder: Embedder | None = None):
        self.cfg = cfg
        self.store = store or Store(cfg)
        self.embedder = embedder or Embedder(cfg)
        self._count_tokens = None

    @property
    def count_tokens(self):
        if self._count_tokens is None:
            self._count_tokens = make_token_counter(self.cfg.embedding.model, str(self.cfg.models_dir))
        return self._count_tokens

    def chunk_file(self, path: Path) -> tuple[object, list[Chunk]]:
        doc = load_document(path)
        embed = None
        if self.cfg.chunking.strategy == "semantic":
            embed = lambda texts: self.embedder.embed_documents(texts)  # noqa: E731
        chunks = chunk_document(doc, self.cfg.chunking, self.count_tokens, embed)
        return doc, chunks

    def index_file(self, path: Path, sha: str, progress: Progress | None = None) -> tuple[int, list[str]]:
        doc, chunks = self.chunk_file(path)
        if not chunks:
            raise ValueError("no text could be extracted" + (f" ({'; '.join(doc.warnings)})" if doc.warnings else ""))
        if progress:
            progress(f"  {len(chunks)} chunks - embedding ...")
        texts = []
        for ch in chunks:
            header = context_header(doc.title, doc.author, ch.chapter) if self.cfg.chunking.context_header else ""
            texts.append(header + ch.text)
        vectors = self.embedder.embed_documents(texts, show_progress=progress is not None and len(texts) > 64)
        self.store.check_embedding_model(self.cfg.embedding.index_id, vectors.shape[1])
        st = path.stat()
        self.store.add_document(
            {
                "path": rel_key(self.cfg, path),
                "sha256": sha,
                "title": doc.title,
                "author": doc.author,
                "category": category_of(self.cfg, path),
                "tags": ",".join(doc.tags) if doc.tags else None,
                "language": doc.language,
                "file_type": doc.file_type,
                "n_pages": doc.n_pages,
                "n_chunks": len(chunks),
                "n_chars": doc.char_count,
                "size_bytes": st.st_size,
                "mtime": st.st_mtime,
                "warnings": dumps(doc.warnings) if doc.warnings else None,
            },
            [ch.__dict__ for ch in chunks],
            vectors,
        )
        return len(chunks), doc.warnings

    def sync(self, rebuild: bool = False, only: list[Path] | None = None, progress: Progress | None = None,
             dry_run: bool = False) -> SyncReport:
        t0 = time.perf_counter()
        rep = SyncReport()
        say = progress or (lambda m: None)
        if rebuild and not dry_run:
            say("Rebuilding: deleting the existing index ...")
            self.store.reset()

        stored_model = self.store.get_meta("embedding_model")
        if stored_model and stored_model != self.cfg.embedding.index_id:
            raise RuntimeError(
                f"Index was built with '{stored_model}', config now uses '{self.cfg.embedding.index_id}'. "
                "Run `rag sync --rebuild`."
            )

        files = [p.resolve() for p in only] if only else scan_library(self.cfg)
        on_disk = {rel_key(self.cfg, p): p for p in files}
        indexed = {d["path"]: d for d in self.store.documents()}

        # Deleted files (only when scanning the whole library)
        if not only:
            missing = [d for k, d in indexed.items() if k not in on_disk]
        else:
            missing = []

        # Detect moves first: a "missing" doc whose hash appears at a new path.
        new_paths = [k for k in on_disk if k not in indexed]
        hashes: dict[str, str] = {}
        for k in new_paths:
            hashes[k] = sha256_file(on_disk[k])
        missing_by_hash = {d["sha256"]: d for d in missing}
        for k in list(new_paths):
            d = missing_by_hash.pop(hashes[k], None)
            if d is not None:
                say(f"moved   {d['path']} -> {k}")
                if not dry_run:
                    p = on_disk[k]
                    self.store.update_document_location(d["id"], k, category_of(self.cfg, p), p.stat().st_mtime)
                rep.moved.append(k)
                new_paths.remove(k)
                missing = [m for m in missing if m["id"] != d["id"]]

        for d in missing:
            say(f"removed {d['path']}")
            if not dry_run:
                self.store.delete_document(d["id"])
            rep.removed.append(d["path"])

        total = len(on_disk)
        for i, (k, p) in enumerate(on_disk.items(), start=1):
            if k in rep.moved:
                continue
            existing = indexed.get(k)
            try:
                if existing:
                    st = p.stat()
                    if existing["size_bytes"] == st.st_size and abs((existing["mtime"] or 0) - st.st_mtime) < 1e-3:
                        rep.unchanged += 1
                        continue
                    sha = sha256_file(p)
                    if sha == existing["sha256"]:
                        if not dry_run:
                            self.store.update_document_location(existing["id"], k, category_of(self.cfg, p), st.st_mtime)
                        rep.unchanged += 1
                        continue
                    say(f"[{i}/{total}] updated {k}")
                    if dry_run:
                        rep.updated.append(k)
                        continue
                    self.store.delete_document(existing["id"])
                    n, warns = self.index_file(p, sha, say)
                    rep.updated.append(k)
                else:
                    sha = hashes.get(k) or sha256_file(p)
                    say(f"[{i}/{total}] new     {k}")
                    if dry_run:
                        rep.added.append(k)
                        continue
                    n, warns = self.index_file(p, sha, say)
                    rep.added.append(k)
                rep.chunks_added += n
                for w in warns:
                    rep.warnings.append((k, w))
                    say(f"  warning: {w}")
            except Exception as exc:
                log.exception("failed to index %s", k)
                rep.failed.append((k, str(exc)))
                say(f"  FAILED: {exc}")

        if not dry_run and (rep.added or rep.updated or rep.removed):
            n_vec = self.store.stats()["vectors"]
            if n_vec >= self.cfg.retrieval.ann_index_threshold:
                say("Library is large - (re)building the ANN index ...")
                say(self.store.optimize())
        rep.seconds = time.perf_counter() - t0
        return rep
