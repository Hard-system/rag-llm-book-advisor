"""Configuration loading (config.yaml + environment variable overrides)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels above this file (src/book_rag/config.py -> repo).
PROJECT_ROOT = Path(os.environ.get("RAG_HOME", Path(__file__).resolve().parents[2]))


@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-m3"
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 16
    max_seq_length: int = 512
    device: str = "auto"
    backend: str = "onnx-int8"  # onnx-int8 (fast on CPU) | torch (full precision / GPU)

    @property
    def index_id(self) -> str:
        """Identity of the vector space - vectors from different ids must never be mixed."""
        return f"{self.model}|{self.backend}"


@dataclass
class RerankerConfig:
    enabled: bool = True
    model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    max_length: int = 512
    candidates: int = 15
    backend: str = "onnx-int8"


@dataclass
class ChunkingConfig:
    strategy: str = "recursive"
    chunk_size: int = 400
    chunk_overlap: int = 80
    min_chunk_size: int = 40
    semantic_breakpoint_percentile: float = 90.0
    context_header: bool = True


@dataclass
class RetrievalConfig:
    top_k: int = 6
    keyword_candidates: int = 50
    vector_candidates: int = 50
    rrf_k: int = 60
    keyword_weight: float = 1.0
    vector_weight: float = 1.0
    neighbor_window: int = 1
    min_relevance: float = 0.05
    rerank_blend: float = 0.5
    ann_index_threshold: int = 100_000


PROFILES: dict[str, dict[str, str]] = {
    # Bulgarian + English (and 100+ other languages). Default - fast on a laptop CPU.
    "multilingual": {
        "embedding": "BAAI/bge-m3",
        "query_prefix": "",
        "document_prefix": "",
        "reranker": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    },
    # Same embeddings, large multilingual reranker: best precision, ~10 s/query on a 4-core CPU.
    "quality": {
        "embedding": "BAAI/bge-m3",
        "query_prefix": "",
        "document_prefix": "",
        "reranker": "BAAI/bge-reranker-v2-m3",
    },
    # The models used in the course labs - English only, ~3x faster on CPU.
    "english": {
        "embedding": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
        "reranker": "BAAI/bge-reranker-base",
    },
    # Smallest multilingual pair for slow machines.
    "light": {
        "embedding": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "reranker": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    },
}


@dataclass
class Config:
    profile: str = "multilingual"
    threads: int = 0
    ocr: bool = True                 # OCR scanned PDFs (needs rapidocr_onnxruntime)
    dedupe_threshold: float = 0.5    # text similarity above which a file is treated as a duplicate book
    library_dir: Path = PROJECT_ROOT / "library"
    data_dir: Path = PROJECT_ROOT / "data"
    extensions: list[str] = field(
        default_factory=lambda: [".pdf", ".epub", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".ipynb"]
    )
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # Derived paths -------------------------------------------------------
    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "library.sqlite"

    @property
    def lancedb_path(self) -> Path:
        return self.data_dir / "vectors.lance"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def ensure_dirs(self) -> None:
        for p in (self.library_dir, self.data_dir, self.models_dir, self.logs_dir):
            p.mkdir(parents=True, exist_ok=True)


def _merge(dc_obj: Any, values: dict[str, Any]) -> None:
    valid = {f.name: f for f in fields(dc_obj)}
    for key, val in (values or {}).items():
        if key not in valid:
            continue
        current = getattr(dc_obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(val, dict):
            _merge(current, val)
        else:
            setattr(dc_obj, key, val)


def _resolve(p: str | Path, base: Path) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml (or RAG_CONFIG) and apply RAG_* environment overrides."""
    cfg = Config()
    cfg_path = Path(path or os.environ.get("RAG_CONFIG", PROJECT_ROOT / "config.yaml"))
    base = cfg_path.parent if cfg_path.exists() else PROJECT_ROOT
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    # 1) profile defaults, 2) explicit values from config.yaml on top
    profile = str(os.environ.get("RAG_PROFILE") or raw.get("profile") or cfg.profile).lower()
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Choose one of: {', '.join(PROFILES)}")
    p = PROFILES[profile]
    cfg.embedding.model = p["embedding"]
    cfg.embedding.query_prefix = p["query_prefix"]
    cfg.embedding.document_prefix = p["document_prefix"]
    cfg.reranker.model = p["reranker"]
    _merge(cfg, raw)
    cfg.profile = profile

    if os.environ.get("RAG_LIBRARY_DIR"):
        cfg.library_dir = os.environ["RAG_LIBRARY_DIR"]
    if os.environ.get("RAG_DATA_DIR"):
        cfg.data_dir = os.environ["RAG_DATA_DIR"]
    if os.environ.get("RAG_EMBEDDING_MODEL"):
        cfg.embedding.model = os.environ["RAG_EMBEDDING_MODEL"]
    if os.environ.get("RAG_RERANKER_MODEL"):
        cfg.reranker.model = os.environ["RAG_RERANKER_MODEL"]
    if os.environ.get("RAG_RERANK") is not None:
        cfg.reranker.enabled = os.environ["RAG_RERANK"].lower() in ("1", "true", "yes", "on")

    cfg.library_dir = _resolve(cfg.library_dir, base)
    cfg.data_dir = _resolve(cfg.data_dir, base)
    cfg.extensions = [e.lower() if e.startswith(".") else "." + e.lower() for e in cfg.extensions]
    return cfg
