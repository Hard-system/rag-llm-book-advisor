"""Chunking strategies.

Three strategies are implemented (see docs/RESEARCH.md for the rationale):

* ``recursive`` (default) – structure-aware recursive splitting. Text is split on
  paragraphs first, then lines, sentences, clauses and finally words, only going
  one level deeper for pieces that are still too large. The resulting units are
  greedily packed into chunks of ``chunk_size`` tokens, with ``chunk_overlap``
  tokens of whole units repeated between neighbouring chunks.
* ``fixed`` – classic fixed-size token windows with overlap (baseline).
* ``semantic`` – sentences are embedded and a new chunk starts where the cosine
  distance between consecutive sentence windows jumps above a percentile
  threshold (Kamradt "semantic chunking"). Large groups are re-packed.

Chunks never cross chapter boundaries, and every chunk knows its chapter and
page range so answers can cite their sources precisely. Sizes are measured in
tokens of the *embedding model's own tokenizer*, so chunks are guaranteed to fit
the model's context instead of being silently truncated.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .config import ChunkingConfig
from .loaders import LoadedDocument

log = logging.getLogger(__name__)

TokenCounter = Callable[[Sequence[str]], list[int]]


@dataclass
class Chunk:
    ord: int
    text: str
    chapter: str | None
    page_start: int | None
    page_end: int | None
    n_tokens: int
    section_idx: int
    char_start: int
    char_end: int


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def approx_token_counter(texts: Sequence[str]) -> list[int]:
    """Rough fallback: ~4 characters per token."""
    return [max(1, len(t) // 4) for t in texts]


def make_token_counter(model_name: str, cache_dir: str | None = None) -> TokenCounter:
    """Return a batched token counter that uses the embedding model's tokenizer."""
    try:
        import transformers
        from transformers import AutoTokenizer

        transformers.logging.set_verbosity_error()
        try:  # offline first: no network round-trips once the model is cached
            tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    except Exception as exc:  # pragma: no cover - network / model issues
        log.warning("Could not load tokenizer for %s (%s); using approximate token counts.", model_name, exc)
        return approx_token_counter

    def count(texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        out: list[int] = []
        for i in range(0, len(texts), 512):
            enc = tok(list(texts[i : i + 512]), add_special_tokens=False, truncation=False)["input_ids"]
            out.extend(len(ids) for ids in enc)
        return out

    return count


# ---------------------------------------------------------------------------
# Streams: consecutive sections of the same chapter, concatenated
# ---------------------------------------------------------------------------


@dataclass
class _Stream:
    text: str
    chapter: str | None
    page_marks: list[tuple[int, int | None]]  # (char_offset, page) sorted

    def page_at(self, offset: int) -> int | None:
        page = None
        for off, p in self.page_marks:
            if off > offset:
                break
            page = p
        return page


def _build_streams(doc: LoadedDocument) -> list[_Stream]:
    streams: list[_Stream] = []
    for sec in doc.sections:
        if not sec.text.strip():
            continue
        if streams and streams[-1].chapter == sec.chapter:
            st = streams[-1]
            offset = len(st.text) + 2
            st.text += "\n\n" + sec.text
            st.page_marks.append((offset, sec.page))
        else:
            streams.append(_Stream(sec.text, sec.chapter, [(0, sec.page)]))
    return streams


# ---------------------------------------------------------------------------
# Atomizing text into units that fit the chunk size
# ---------------------------------------------------------------------------

_SENTENCE_END = r"(?<=[.!?…。])[\"'»”)\]]*\s+"
SEPARATORS: list[str] = [
    r"\n\s*\n",        # paragraphs
    r"\n",             # lines
    _SENTENCE_END,     # sentences
    r"(?<=[;:,])\s+",  # clauses
    r"\s+",            # words
]


def _split_span(text: str, start: int, end: int, pattern: str) -> list[tuple[int, int]]:
    spans, cur = [], start
    for m in re.finditer(pattern, text[start:end]):
        cut = start + m.end()
        if cut > cur:
            spans.append((cur, cut))
            cur = cut
    if cur < end:
        spans.append((cur, end))
    return spans


def _atomize(
    text: str, size: int, count: TokenCounter, max_level: int = len(SEPARATORS)
) -> list[tuple[int, int, int]]:
    """Split ``text`` into (start, end, n_tokens) units, each <= size tokens if possible."""
    pending = [(0, len(text), 0)]  # (start, end, level)
    final: list[tuple[int, int, int]] = []
    while pending:
        counts = count([text[s:e] for s, e, _ in pending])
        nxt = []
        for (s, e, lvl), n in zip(pending, counts):
            if n <= size or lvl >= max_level:
                if n > size:  # giant unbreakable token run -> hard character split
                    step = max(1, (e - s) * size // n)
                    for hs in range(s, e, step):
                        he = min(e, hs + step)
                        final.append((hs, he, count([text[hs:he]])[0]))
                else:
                    final.append((s, e, n))
                continue
            parts = _split_span(text, s, e, SEPARATORS[lvl])
            if len(parts) == 1:
                nxt.append((s, e, lvl + 1))
            else:
                nxt.extend((ps, pe, lvl + 1) for ps, pe in parts)
        pending = nxt
    final.sort()
    return final


def _pack(
    units: list[tuple[int, int, int]], size: int, overlap: int, min_size: int,
    text: str | None = None, count: TokenCounter | None = None,
) -> list[tuple[int, int, int]]:
    """Greedily pack units into chunks (start, end, n_tokens).

    Overlap is made of whole trailing units (paragraphs / sentences). When the last unit is too big
    to be repeated, its trailing *sentences* are carried into the next chunk instead, so neighbouring
    chunks always share some context (needs ``text`` and ``count``).
    """
    chunks: list[tuple[int, int, int]] = []
    n, i = len(units), 0
    carry: tuple[int, int] | None = None  # (start offset, tokens) of carried-over sentences
    while i < n:
        if carry and carry[1] + units[i][2] > size:
            carry = None
        start, tok = (carry[0], carry[1]) if carry else (units[i][0], 0)
        j = i
        while j < n and (tok + units[j][2] <= size or j == i):
            tok += units[j][2]
            j += 1
        chunks.append((start, units[j - 1][1], tok))
        if j >= n:
            break
        k, ov = j, 0
        while k - 1 > i and ov + units[k - 1][2] <= overlap:
            k -= 1
            ov += units[k][2]
        carry = None
        if k == j and overlap > 0 and text is not None and count is not None:
            s, e, _ = units[j - 1]
            parts = _split_span(text, s, e, _SENTENCE_END)
            if len(parts) > 1:
                cnts = count([text[a:b] for a, b in parts])
                m, ov = len(parts), 0
                while m - 1 > 0 and ov + cnts[m - 1] <= overlap:
                    m -= 1
                    ov += cnts[m]
                if m < len(parts):
                    carry = (parts[m][0], ov)
        i = k
    # merge a tiny tail chunk into its predecessor
    if len(chunks) >= 2 and chunks[-1][2] < min_size:
        last = chunks.pop()
        prev = chunks.pop()
        chunks.append((prev[0], last[1], prev[2] + last[2]))
    return chunks


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------


def _semantic_groups(
    text: str,
    sentences: list[tuple[int, int, int]],
    embed: Callable[[list[str]], np.ndarray],
    percentile: float,
) -> list[list[tuple[int, int, int]]]:
    if len(sentences) < 3:
        return [sentences]
    # Embed each sentence together with its neighbours to smooth noise.
    windows = []
    for i in range(len(sentences)):
        lo, hi = max(0, i - 1), min(len(sentences), i + 2)
        windows.append(text[sentences[lo][0] : sentences[hi - 1][1]])
    vecs = embed(windows)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    dists = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)
    threshold = float(np.percentile(dists, percentile))
    groups, cur = [], [sentences[0]]
    for i, d in enumerate(dists, start=1):
        if d > threshold:
            groups.append(cur)
            cur = []
        cur.append(sentences[i])
    groups.append(cur)
    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_document(
    doc: LoadedDocument,
    cfg: ChunkingConfig,
    count_tokens: TokenCounter = approx_token_counter,
    embed: Callable[[list[str]], np.ndarray] | None = None,
) -> list[Chunk]:
    size, overlap, min_size = cfg.chunk_size, cfg.chunk_overlap, cfg.min_chunk_size
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    strategy = cfg.strategy.lower()
    if strategy == "semantic" and embed is None:
        log.warning("Semantic chunking needs an embedder; falling back to recursive.")
        strategy = "recursive"

    chunks: list[Chunk] = []
    for sidx, st in enumerate(_build_streams(doc)):
        if strategy == "fixed":
            units = _atomize(st.text, 1, count_tokens, max_level=len(SEPARATORS))  # word-level units
            spans = _pack(units, size, overlap, min_size, st.text, count_tokens)
        elif strategy == "semantic":
            sentences = _atomize(st.text, size, count_tokens, max_level=len(SEPARATORS))
            # force sentence granularity: re-split units that are whole paragraphs
            fine: list[tuple[int, int, int]] = []
            for s, e, n in sentences:
                parts = _split_span(st.text, s, e, _SENTENCE_END)
                if len(parts) > 1:
                    fine.extend((ps, pe, c) for (ps, pe), c in zip(parts, count_tokens([st.text[a:b] for a, b in parts])))
                else:
                    fine.append((s, e, n))
            spans = []
            for group in _semantic_groups(st.text, fine, embed, cfg.semantic_breakpoint_percentile):
                spans.extend(_pack(group, size, overlap, min_size, st.text, count_tokens))
            # merge undersized groups with their successor
            merged: list[tuple[int, int, int]] = []
            for sp in spans:
                if merged and merged[-1][2] < min_size and merged[-1][2] + sp[2] <= size:
                    prev = merged.pop()
                    sp = (prev[0], sp[1], prev[2] + sp[2])
                merged.append(sp)
            spans = merged
        else:  # recursive
            units = _atomize(st.text, size, count_tokens)
            spans = _pack(units, size, overlap, min_size, st.text, count_tokens)

        for s, e, n in spans:
            s, e = _trim(st.text, s, e)
            if e <= s:
                continue
            chunks.append(
                Chunk(
                    ord=len(chunks),
                    text=st.text[s:e],
                    chapter=st.chapter,
                    page_start=st.page_at(s),
                    page_end=st.page_at(max(s, e - 1)),
                    n_tokens=n,
                    section_idx=sidx,
                    char_start=s,
                    char_end=e,
                )
            )
    return chunks


def context_header(title: str, author: str | None, chapter: str | None) -> str:
    """Contextual chunk header prepended before embedding / keyword indexing."""
    head = title + (f" — {author}" if author else "")
    return head + (f"\n{chapter}" if chapter else "") + "\n\n"


def stitch(chunks: list[dict]) -> str:
    """Join consecutive chunks of the same document, removing duplicated overlap text."""
    if not chunks:
        return ""
    out = chunks[0]["text"]
    prev = chunks[0]
    for ch in chunks[1:]:
        same_stream = ch["section_idx"] == prev["section_idx"] and ch["ord"] == prev["ord"] + 1
        if same_stream and ch["char_start"] < prev["char_end"]:
            out += ch["text"][prev["char_end"] - ch["char_start"] :]
        else:
            out += "\n\n" + ch["text"]
        prev = ch
    return out
