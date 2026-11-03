"""OCR for scanned PDFs (pages that are images without a text layer).

Uses RapidOCR (ONNX Runtime, pip-installable, no Tesseract needed). OCR is slow
on CPU (~10 s per page), so results are cached per file (SHA-256) and page in
``data/ocr/<sha>.json``. An interrupted run resumes, and a later ``--rebuild``
never re-OCRs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

log = logging.getLogger(__name__)
_engine = None
_lock = threading.Lock()


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
    return _engine


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _lines_to_text(result) -> str:
    """Join OCR lines; a large vertical gap starts a new paragraph."""
    if not result:
        return ""
    items = []
    for box, text, _score in result:
        ys = [p[1] for p in box]
        items.append((min(ys), max(ys), min(p[0] for p in box), text.strip()))
    items.sort(key=lambda t: (round(t[0] / 8), t[2]))
    heights = [b - a for a, b, _, _ in items if b > a]
    line_h = float(np.median(heights)) if heights else 20.0
    out, prev_bottom = [], None
    for top, bottom, _x, text in items:
        if not text:
            continue
        if prev_bottom is not None:
            out.append("\n\n" if top - prev_bottom > 0.9 * line_h else " ")
        out.append(text)
        prev_bottom = bottom
    return "".join(out)


def ocr_pdf_pages(path: Path, pages: Iterable[int], cache_dir: Path, dpi: int = 150,
                  progress: Callable[[str], None] | None = None) -> dict[int, str]:
    """OCR the given 1-based pages of a PDF, using/updating the cache. Returns {page: text}."""
    import pymupdf

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{file_sha(path)}.json"
    cache: dict[str, str] = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    todo = [p for p in pages if str(p) not in cache]
    if todo:
        engine = _get_engine()
        doc = pymupdf.open(path)
        for i, pno in enumerate(todo, 1):
            pix = doc[pno - 1].get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
            result, _ = engine(np.ascontiguousarray(img))
            cache[str(pno)] = _lines_to_text(result)
            if i % 10 == 0 or i == len(todo):
                cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                if progress:
                    progress(f"  OCR {path.name}: {i}/{len(todo)} pages")
        doc.close()
    return {p: cache.get(str(p), "") for p in pages}
