"""Incremental library synchronisation.

``sync()`` makes the index mirror the ``library/`` folder:

* new files        -> load (OCR for scanned PDFs), chunk, embed, index
* changed files    -> (content or sidecar .yaml changed) old version removed, re-indexed
* moved / renamed  -> (same hash, new path) only the path/category is updated, nothing re-embedded
* deleted files    -> removed from the index
* unchanged files  -> skipped in milliseconds (size + mtime, then sha256)
* duplicates       -> the same book under another name / in another folder (detected by text
                      similarity, MinHash-style bottom-k sketch of 5-word shingles) is recorded
                      as a duplicate and not embedded twice
* empty files      -> recorded as empty (no text), not retried until the file changes
* ignored files    -> patterns in ``library/.ragignore`` (gitignore-like globs)
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .chunking import Chunk, chunk_document, context_header, make_token_counter
from .config import Config
from .loaders import GENERIC_STEM, load_document, sidecar_path
from .models import Embedder
from .store import Store, dumps

log = logging.getLogger(__name__)

Progress = Callable[[str], None]
IGNORE_FILE = ".ragignore"


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    moved: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)  # (path, duplicate of title)
    empty: list[str] = field(default_factory=list)
    unchanged: int = 0
    ignored: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)
    chunks_added: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"added {len(self.added)}, updated {len(self.updated)}, moved {len(self.moved)}, "
            f"removed {len(self.removed)}, duplicates skipped {len(self.duplicates)}, empty {len(self.empty)}, "
            f"unchanged {self.unchanged}, ignored {self.ignored}, failed {len(self.failed)} "
            f"- {self.chunks_added} new chunks in {_fmt_secs(self.seconds)}"
        )


def _fmt_secs(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def file_fingerprint(path: Path) -> str:
    """SHA-256 of the file plus its optional metadata sidecar (so editing the .yaml re-indexes)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    side = sidecar_path(path)
    if side.exists():
        h.update(b"\x00sidecar\x00" + side.read_bytes())
    return h.hexdigest()


sha256_file = file_fingerprint  # backwards compatible name


def stat_key(path: Path) -> tuple[int, float]:
    """(size, mtime) of the file and its sidecar - the fast 'unchanged?' check."""
    st = path.stat()
    size, mtime = st.st_size, st.st_mtime
    side = sidecar_path(path)
    if side.exists():
        sst = side.stat()
        size, mtime = size + sst.st_size, max(mtime, sst.st_mtime)
    return size, mtime


def load_ignore_patterns(cfg: Config) -> list[str]:
    f = cfg.library_dir / IGNORE_FILE
    if not f.exists():
        return []
    pats = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pats.append(line.replace("\\", "/"))
    return pats


def is_ignored(rel: str, patterns: list[str]) -> bool:
    rel_l, name_l = rel.lower(), rel.rsplit("/", 1)[-1].lower()
    for pat in patterns:
        p = pat.lower().rstrip("/")
        if fnmatch.fnmatch(rel_l, p) or fnmatch.fnmatch(name_l, p) or rel_l.startswith(p + "/"):
            return True
    return False


def _name_penalty(p: Path) -> int:
    """Lower = better file name. Well-named copies are indexed first, so they become the original."""
    stem = p.stem
    pen = 0
    if GENERIC_STEM.match(stem.strip()):
        pen += 2
    if re.search(r"\(\d\)$|[_ ]\d{6}[_ ]\d{6}$|copy", stem, re.I):
        pen += 1
    return pen


def scan_library(cfg: Config, report: SyncReport | None = None) -> list[Path]:
    patterns = load_ignore_patterns(cfg)
    files = []
    for p in cfg.library_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in cfg.extensions:
            continue
        rel = p.relative_to(cfg.library_dir)
        if any(part.startswith((".", "~$", "_")) for part in rel.parts):
            continue  # hidden folders, Office lock files, "_ignored" folders
        if p.name.lower() == "readme.md" and p.parent == cfg.library_dir:
            continue
        if patterns and is_ignored(rel.as_posix(), patterns):
            if report is not None:
                report.ignored += 1
            continue
        files.append(p)
    return sorted(files, key=lambda p: (_name_penalty(p), p.relative_to(cfg.library_dir).as_posix().lower()))


def rel_key(cfg: Config, path: Path) -> str:
    return path.relative_to(cfg.library_dir).as_posix()


def category_of(cfg: Config, path: Path) -> str | None:
    parts = path.relative_to(cfg.library_dir).parts
    return "/".join(parts[:-1]) if len(parts) > 1 else None


# ---------------------------------------------------------------------------
# Near-duplicate detection (bottom-k MinHash sketch over 5-word shingles)
# ---------------------------------------------------------------------------

SKETCH_K = 256


def text_sketch(text: str, k: int = SKETCH_K) -> np.ndarray:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 200:
        return np.zeros(0, dtype=np.uint64)
    hs = {
        int.from_bytes(hashlib.blake2b(" ".join(words[i : i + 5]).encode(), digest_size=8).digest(), "little")
        for i in range(len(words) - 4)
    }
    return np.sort(np.fromiter(hs, dtype=np.uint64))[:k]


def sketch_similarity(a: np.ndarray, b: np.ndarray, k: int = SKETCH_K) -> float:
    """Estimated Jaccard similarity of the two shingle sets."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    union_k = np.union1d(a, b)[:k]
    both = np.intersect1d(a, b, assume_unique=True)
    return len(np.intersect1d(both, union_k, assume_unique=True)) / len(union_k)


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class Indexer:
    def __init__(self, cfg: Config, store: Store | None = None, embedder: Embedder | None = None):
        self.cfg = cfg
        self.store = store or Store(cfg)
        self.embedder = embedder or Embedder(cfg)
        self._count_tokens = None
        self._sketches: list[tuple[int, str, np.ndarray]] | None = None

    @property
    def count_tokens(self):
        if self._count_tokens is None:
            self._count_tokens = make_token_counter(self.cfg.embedding.model, str(self.cfg.models_dir))
        return self._count_tokens

    @property
    def ocr_dir(self) -> Path | None:
        return self.cfg.data_dir / "ocr" if self.cfg.ocr else None

    def chunk_file(self, path: Path) -> tuple[object, list[Chunk]]:
        doc = load_document(path, self.ocr_dir)
        embed = None
        if self.cfg.chunking.strategy == "semantic":
            embed = lambda texts: self.embedder.embed_documents(texts)  # noqa: E731
        chunks = chunk_document(doc, self.cfg.chunking, self.count_tokens, embed)
        return doc, chunks

    def _find_duplicate(self, sketch: np.ndarray) -> tuple[int, str, float] | None:
        if self._sketches is None:
            self._sketches = self.store.document_sketches()
        best = None
        for doc_id, title, sk in self._sketches:
            sim = sketch_similarity(sketch, sk)
            if sim >= self.cfg.dedupe_threshold and (best is None or sim > best[2]):
                best = (doc_id, title, sim)
        return best

    def _record(self, path: Path, sha: str, status: str, doc=None, duplicate_of: int | None = None,
                warnings: list[str] | None = None) -> None:
        size, mtime = stat_key(path)
        self.store.add_document(
            {
                "path": rel_key(self.cfg, path), "sha256": sha,
                "title": getattr(doc, "title", None) or path.stem, "author": getattr(doc, "author", None),
                "category": category_of(self.cfg, path), "tags": None, "language": None,
                "file_type": path.suffix.lower().lstrip("."), "n_pages": getattr(doc, "n_pages", None),
                "n_chunks": 0, "n_chars": getattr(doc, "char_count", 0), "size_bytes": size, "mtime": mtime,
                "warnings": dumps(warnings) if warnings else None,
                "status": status, "duplicate_of": duplicate_of, "fingerprint": None,
            },
            [], None,
        )

    def index_file(self, path: Path, sha: str, progress: Progress | None = None) -> tuple[str, int, list[str], str]:
        """Index one file. Returns (status, n_chunks, warnings, detail)."""
        say = progress or (lambda m: None)
        if path.stat().st_size == 0:
            self._record(path, sha, "empty", warnings=["empty file (0 bytes)"])
            return "empty", 0, [], "0-byte file"
        doc, chunks = self.chunk_file(path)
        if not chunks:
            self._record(path, sha, "empty", doc, warnings=doc.warnings)
            return "empty", 0, doc.warnings, "no extractable text"

        sketch = text_sketch("\n".join(c.text for c in chunks))
        dup = self._find_duplicate(sketch) if self.cfg.dedupe_threshold < 1 else None
        if dup:
            self._record(path, sha, "duplicate", doc, duplicate_of=dup[0])
            return "duplicate", 0, [], f"same text as '{dup[1]}' (similarity {dup[2]:.2f})"

        say(f"  '{doc.title}'{' — ' + doc.author if doc.author else ''}: {len(chunks)} chunks, "
            f"{len({c.chapter for c in chunks if c.chapter})} chapters - embedding ...")
        texts = []
        for ch in chunks:
            header = context_header(doc.title, doc.author, ch.chapter) if self.cfg.chunking.context_header else ""
            texts.append(header + ch.text)
        vectors = self.embedder.embed_documents(texts, show_progress=progress is not None and len(texts) > 64)
        self.store.check_embedding_model(self.cfg.embedding.index_id, vectors.shape[1])
        size, mtime = stat_key(path)
        doc_id = self.store.add_document(
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
                "size_bytes": size,
                "mtime": mtime,
                "warnings": dumps(doc.warnings) if doc.warnings else None,
                "status": "indexed",
                "duplicate_of": None,
                "fingerprint": sketch.tobytes(),
            },
            [ch.__dict__ for ch in chunks],
            vectors,
        )
        if self._sketches is not None and len(sketch):
            self._sketches.append((doc_id, doc.title, sketch))
        return "indexed", len(chunks), doc.warnings, ""

    def sync(self, rebuild: bool = False, only: list[Path] | None = None, progress: Progress | None = None,
             dry_run: bool = False) -> SyncReport:
        t0 = time.perf_counter()
        rep = SyncReport()
        say = progress or (lambda m: None)
        if rebuild and not dry_run:
            say("Rebuilding: deleting the existing index (OCR results are kept) ...")
            self.store.reset()
            self._sketches = None

        stored_model = self.store.get_meta("embedding_model")
        if stored_model and stored_model != self.cfg.embedding.index_id:
            raise RuntimeError(
                f"Index was built with '{stored_model}', config now uses '{self.cfg.embedding.index_id}'. "
                "Run `rag sync --rebuild`."
            )

        files = [p.resolve() for p in only] if only else scan_library(self.cfg, rep)
        on_disk = {rel_key(self.cfg, p): p for p in files}
        records = {d["path"]: d for d in self.store.documents(status=None)}

        missing = [d for k, d in records.items() if k not in on_disk] if not only else []

        # Moves: a missing record whose content hash now appears at a new path.
        new_paths = [k for k in on_disk if k not in records]
        hashes = {k: file_fingerprint(on_disk[k]) for k in new_paths}
        missing_by_hash = {d["sha256"]: d for d in missing}
        for k in list(new_paths):
            d = missing_by_hash.pop(hashes[k], None)
            if d is not None:
                say(f"moved   {d['path']} -> {k}")
                if not dry_run:
                    p = on_disk[k]
                    self.store.update_document_location(d["id"], k, category_of(self.cfg, p), stat_key(p)[1])
                rep.moved.append(k)
                new_paths.remove(k)
                missing = [m for m in missing if m["id"] != d["id"]]

        for d in missing:
            say(f"removed {d['path']}")
            if not dry_run:
                self.store.delete_document(d["id"])
            rep.removed.append(d["path"])

        if not dry_run:
            # duplicates whose original disappeared must be re-evaluated (they may now be the only copy)
            for k in self.store.orphan_duplicates():
                records.pop(k, None)
            records = {d["path"]: d for d in self.store.documents(status=None)}

        todo = [(k, p) for k, p in on_disk.items() if k not in rep.moved]
        total, done_work, t_work = len(todo), 0, 0.0
        for i, (k, p) in enumerate(todo, start=1):
            existing = records.get(k)
            try:
                if existing:
                    size, mtime = stat_key(p)
                    if existing["size_bytes"] == size and abs((existing["mtime"] or 0) - mtime) < 1e-3:
                        rep.unchanged += 1
                        continue
                    sha = file_fingerprint(p)
                    if sha == existing["sha256"]:
                        if not dry_run:
                            self.store.update_document_location(existing["id"], k, category_of(self.cfg, p), mtime)
                        rep.unchanged += 1
                        continue
                    kind = "updated"
                else:
                    sha = hashes.get(k) or file_fingerprint(p)
                    kind = "new"
                eta = ""
                if done_work:
                    remaining = sum(1 for kk, _ in todo[i - 1:] if kk not in records)
                    eta = f" · elapsed {_fmt_secs(time.perf_counter() - t0)}, ~{_fmt_secs(t_work / done_work * remaining)} left"
                say(f"[{i}/{total}] {kind:7s} {k}{eta}")
                if dry_run:
                    (rep.updated if kind == "updated" else rep.added).append(k)
                    continue
                ts = time.perf_counter()
                if existing:
                    self.store.delete_document(existing["id"])
                    self._sketches = None
                status, n, warns, detail = self.index_file(p, sha, say)
                t_work += time.perf_counter() - ts
                done_work += 1
                if status == "indexed":
                    (rep.updated if kind == "updated" else rep.added).append(k)
                    rep.chunks_added += n
                    say(f"  done: {n} chunks in {_fmt_secs(time.perf_counter() - ts)}")
                elif status == "duplicate":
                    rep.duplicates.append((k, detail))
                    say(f"  duplicate - skipped: {detail}")
                else:
                    rep.empty.append(k)
                    say(f"  skipped: {detail}")
                for w in warns:
                    rep.warnings.append((k, w))
                    say(f"  note: {w}")
            except KeyboardInterrupt:
                say("Interrupted - finished books are saved; run `rag sync` again to continue.")
                raise
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
