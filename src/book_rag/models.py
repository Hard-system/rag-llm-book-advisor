"""Embedding model (bi-encoder) and reranker (cross-encoder) wrappers.

Models are loaded lazily on first use and cached under ``data/models`` so the
library works fully offline after the first download.

CPU speed-up - int8 quantization (course module 5)
----------------------------------------------------
With ``backend: onnx-int8`` (default) each model is exported once to ONNX and
its weights are dynamically quantized to 8-bit integers. Measured on a 4-core
i7-1165G7 laptop:

* bge-m3 embeddings:           3.3 s -> 0.77 s per 400-token chunk (4.3x), cosine to fp32 = 0.978
* bge-reranker-v2-m3 (20 pairs): 77 s -> 23 s, Spearman rank correlation to fp32 = 0.985

The one-time export takes a few minutes per model and the quantized files are
stored in ``data/models/onnx/<model>``. If export fails the code falls back to
plain PyTorch automatically.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import Config

log = logging.getLogger(__name__)
_lock = threading.Lock()


def _configure_torch(cfg: Config) -> str:
    import torch

    threads = cfg.threads or max(1, (os.cpu_count() or 2) // 2)  # physical cores on hyper-threaded CPUs
    torch.set_num_threads(threads)
    device = cfg.embedding.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return device


def _quant_arch() -> str:
    """Pick the ONNX Runtime int8 quantization config for this CPU."""
    import platform

    if platform.machine().lower() in ("arm64", "aarch64"):
        return "arm64"
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats  # numpy >= 2
    except Exception:  # pragma: no cover
        try:
            from numpy.core._multiarray_umath import __cpu_features__ as feats
        except Exception:
            feats = {}
    if feats.get("AVX512VNNI") or feats.get("AVX512_VNNI") or feats.get("AVX512_CLX"):
        return "avx512_vnni"
    if feats.get("AVX512F"):
        return "avx512"
    return "avx2"


def _onnx_dir(cfg: Config, model_name: str) -> Path:
    return cfg.models_dir / "onnx" / model_name.replace("/", "__")


def _load_onnx_int8(cls, cfg: Config, model_name: str, **kwargs):
    """Load an int8-quantized ONNX copy of ``model_name``, exporting it on first use."""
    from sentence_transformers import export_dynamic_quantized_onnx_model

    arch = _quant_arch()
    out = _onnx_dir(cfg, model_name)
    qfile = f"onnx/model_qint8_{arch}.onnx"
    if not (out / qfile).exists():
        log.warning("One-time export of %s to int8 ONNX (%s) - this can take a few minutes ...", model_name, arch)
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "onnx" / "model.onnx").exists():
            # export=True: always convert from the PyTorch weights. Some repos ship their own ONNX
            # files with non-standard outputs (e.g. BAAI/bge-m3) that sentence-transformers can't use.
            m = cls(model_name, backend="onnx", cache_folder=str(cfg.models_dir),
                    model_kwargs={"export": True}, **kwargs)
            m.save_pretrained(str(out))
            del m
        # Reload from the saved copy: the HF cache uses symlinks, which the ONNX quantizer refuses to read.
        m = cls(str(out), backend="onnx", **kwargs)
        export_dynamic_quantized_onnx_model(m, arch, str(out))
        # the fp32 ONNX copy is no longer needed (can be > 2 GB)
        for f in ("onnx/model.onnx", "onnx/model.onnx_data"):
            try:
                (out / f).unlink()
            except FileNotFoundError:
                pass
        del m
    return cls(str(out), backend="onnx", model_kwargs={"file_name": qfile}, **kwargs)


def _load(cls, cfg: Config, model_name: str, backend: str, **kwargs):
    device = _configure_torch(cfg)
    if backend == "onnx-int8" and device == "cpu":
        try:
            return _load_onnx_int8(cls, cfg, model_name, **kwargs)
        except Exception as exc:
            log.warning("int8 ONNX unavailable for %s (%s) - falling back to PyTorch.", model_name, exc)
            shutil.rmtree(_onnx_dir(cfg, model_name), ignore_errors=True)
    return cls(model_name, device=device, cache_folder=str(cfg.models_dir), **kwargs)


class Embedder:
    """Dense bi-encoder used for semantic search (and semantic chunking)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            with _lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("Loading embedding model %s (%s) ...", self.cfg.embedding.model, self.cfg.embedding.backend)
                    m = _load(SentenceTransformer, self.cfg, self.cfg.embedding.model, self.cfg.embedding.backend)
                    m.max_seq_length = self.cfg.embedding.max_seq_length
                    self._model = m
        return self._model

    @property
    def dim(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(getter())

    def embed_documents(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        prefix = self.cfg.embedding.document_prefix
        return self._encode([prefix + t for t in texts], show_progress)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prefix = self.cfg.embedding.query_prefix
        return self._encode([prefix + t for t in texts], False)

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=self.cfg.embedding.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)


class Reranker:
    """Cross-encoder that re-scores (query, passage) pairs for precision."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            with _lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    log.info("Loading reranker %s (%s) ...", self.cfg.reranker.model, self.cfg.reranker.backend)
                    self._model = _load(
                        CrossEncoder, self.cfg, self.cfg.reranker.model, self.cfg.reranker.backend,
                        max_length=self.cfg.reranker.max_length,
                    )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict(
            [(query, p) for p in passages], batch_size=8, convert_to_numpy=True, show_progress_bar=False
        )
        scores = np.asarray(scores, dtype=np.float32).reshape(len(passages), -1)[:, -1]
        # Some cross-encoders return raw logits; squash to 0..1 for readable scores.
        if scores.min() < 0 or scores.max() > 1:
            scores = 1.0 / (1.0 + np.exp(-scores))
        return scores
