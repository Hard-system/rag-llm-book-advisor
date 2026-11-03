"""Hybrid retrieval pipeline.

    queries (original + rewrites / HyDE passage written by Claude)
        │
        ├── BM25 keyword search (SQLite FTS5)      ─┐   per query, after
        └── dense semantic search (LanceDB, cosine) ─┤   metadata filtering
                                                      ▼
                         Reciprocal Rank Fusion (weighted)
                                                      ▼
                  Cross-encoder reranking of the top candidates
                                                      ▼
          small-to-big expansion: merge neighbouring chunks into passages
                                                      ▼
                        cited passages returned to Claude
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from .chunking import stitch
from .config import Config
from .models import Embedder, Reranker
from .store import Store

log = logging.getLogger(__name__)


@dataclass
class Passage:
    rank: int
    chunk_id: int
    doc_id: int
    title: str
    author: str | None
    category: str | None
    chapter: str | None
    page_start: int | None
    page_end: int | None
    score: float              # reranker probability (0..1) if reranking, otherwise fused score
    rrf_score: float
    keyword_rank: int | None  # best rank in BM25 lists (1 = top)
    vector_rank: int | None   # best rank in semantic lists
    text: str
    chunk_ids: list[int] = field(default_factory=list)

    def citation(self) -> str:
        parts = [self.title + (f" — {self.author}" if self.author else "")]
        if self.chapter:
            parts.append(self.chapter)
        if self.page_start:
            parts.append(
                f"p. {self.page_start}" if self.page_end in (None, self.page_start) else f"pp. {self.page_start}–{self.page_end}"
            )
        return " | ".join(parts)


@dataclass
class SearchResult:
    query: str
    queries: list[str]
    passages: list[Passage]
    timings_ms: dict[str, float]
    mode: str
    reranked: bool
    candidates: int
    pruned: int = 0

    @property
    def confidence(self) -> str:
        """Corrective-RAG style judgement of the retrieval, from the cross-encoder scores."""
        if not self.passages:
            return "none"
        if not self.reranked:
            return "unknown"
        best = max(p.score for p in self.passages)
        return "high" if best >= 0.5 else "medium" if best >= 0.15 else "low"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence"] = self.confidence
        for p, obj in zip(d["passages"], self.passages):
            p["citation"] = obj.citation()
        return d


def rrf_fuse(
    ranked_lists: list[tuple[list[int], float]], k: int = 60
) -> dict[int, float]:
    """Weighted Reciprocal Rank Fusion: score(d) = Σ w / (k + rank(d))."""
    scores: dict[int, float] = defaultdict(float)
    for ids, weight in ranked_lists:
        for rank, cid in enumerate(ids, start=1):
            scores[cid] += weight / (k + rank)
    return scores


class Retriever:
    def __init__(self, cfg: Config, store: Store | None = None, embedder: Embedder | None = None,
                 reranker: Reranker | None = None):
        self.cfg = cfg
        self.store = store or Store(cfg)
        self.embedder = embedder or Embedder(cfg)
        self.reranker = reranker or Reranker(cfg)
        self._qcache: dict[str, Any] = {}

    def _embed_queries(self, queries: list[str]):
        """Query embeddings with a small LRU-ish cache (iterative searches often repeat queries)."""
        missing = [q for q in queries if q not in self._qcache]
        if missing:
            for q, v in zip(missing, self.embedder.embed_queries(missing)):
                if len(self._qcache) >= 512:
                    self._qcache.pop(next(iter(self._qcache)))
                self._qcache[q] = v
        return [self._qcache[q] for q in queries]

    def warmup(self) -> None:
        """Load models and open the indexes ahead of time so the first real query is fast."""
        self.embedder.embed_queries(["warmup"])
        if self.cfg.reranker.enabled:
            self.reranker.score("warmup", ["warmup"])
        self.search("warmup", top_k=1, log_query=False)
        self._qcache.pop("warmup", None)

    def search(
        self,
        query: str,
        alternative_queries: list[str] | None = None,
        top_k: int | None = None,
        book: str | None = None,
        author: str | None = None,
        category: str | None = None,
        mode: str = "hybrid",
        rerank: bool | None = None,
        neighbor_window: int | None = None,
        log_query: bool = True,
    ) -> SearchResult:
        r = self.cfg.retrieval
        top_k = top_k or r.top_k
        rerank = self.cfg.reranker.enabled if rerank is None else rerank
        window = r.neighbor_window if neighbor_window is None else neighbor_window
        mode = mode.lower()
        if mode not in ("hybrid", "keyword", "semantic"):
            raise ValueError("mode must be hybrid, keyword or semantic")

        queries = [query] + [q for q in (alternative_queries or []) if q and q.strip() and q != query]
        t: dict[str, float] = {}
        t0 = time.perf_counter()

        doc_ids = self.store.filter_doc_ids(book, author, category)

        # 1) candidate generation -------------------------------------------------
        lists: list[tuple[list[int], float]] = []
        kw_rank: dict[int, int] = {}
        vec_rank: dict[int, int] = {}
        if mode in ("hybrid", "keyword"):
            ts = time.perf_counter()
            for q in queries:
                hits = self.store.keyword_search(q, r.keyword_candidates, doc_ids)
                ids = [h[0] for h in hits]
                lists.append((ids, r.keyword_weight))
                for i, cid in enumerate(ids, 1):
                    kw_rank[cid] = min(kw_rank.get(cid, 10**9), i)
            t["keyword"] = (time.perf_counter() - ts) * 1000
        if mode in ("hybrid", "semantic"):
            ts = time.perf_counter()
            qvecs = self._embed_queries(queries)
            t["embed_query"] = (time.perf_counter() - ts) * 1000
            ts = time.perf_counter()
            for qv in qvecs:
                hits = self.store.vector_search(qv, r.vector_candidates, doc_ids)
                ids = [h[0] for h in hits]
                lists.append((ids, r.vector_weight))
                for i, cid in enumerate(ids, 1):
                    vec_rank[cid] = min(vec_rank.get(cid, 10**9), i)
            t["vector"] = (time.perf_counter() - ts) * 1000

        # 2) fusion ---------------------------------------------------------------
        fused = rrf_fuse(lists, r.rrf_k)
        ordered = sorted(fused, key=fused.get, reverse=True)
        n_cand = max(top_k, self.cfg.reranker.candidates if rerank else top_k)
        candidates = ordered[:n_cand]
        chunks = self.store.get_chunks(candidates)
        candidates = [c for c in candidates if c in chunks]

        # 3) reranking ------------------------------------------------------------
        final_scores: dict[int, float] = {c: fused[c] for c in candidates}
        pruned = 0
        n_considered = len(candidates)
        if rerank and candidates:
            ts = time.perf_counter()
            texts = []
            for c in candidates:
                ch = chunks[c]
                head = ch["title"] + (f" / {ch['chapter']}" if ch["chapter"] else "")
                texts.append(f"{head}\n{ch['text']}")
            scores = self.reranker.score(query, texts)
            final_scores = {c: float(s) for c, s in zip(candidates, scores)}
            rerank_order = sorted(candidates, key=final_scores.get, reverse=True)
            # Blend the reranker's order with the retrieval order (rank fusion) instead of letting a
            # small cross-encoder override strong hybrid evidence (e.g. cross-language queries).
            b = min(max(r.rerank_blend, 0.0), 1.0)
            blended = rrf_fuse([(rerank_order, b), (candidates, 1.0 - b)], r.rrf_k)
            candidates = sorted(candidates, key=blended.get, reverse=True)
            t["rerank"] = (time.perf_counter() - ts) * 1000
            # context pruning: keep irrelevant chunks out of the LLM context (always keep the best 2)
            if r.min_relevance > 0:
                kept = [c for i, c in enumerate(candidates) if i < 2 or final_scores[c] >= r.min_relevance]
                pruned = len(candidates) - len(kept)
                candidates = kept

        # 4) small-to-big expansion & de-duplication -----------------------------
        ts = time.perf_counter()
        passages: list[Passage] = []
        covered: dict[int, list[tuple[int, int]]] = defaultdict(list)  # doc_id -> [(ord_from, ord_to)]
        for cid in candidates:
            if len(passages) >= top_k:
                break
            ch = chunks[cid]
            lo, hi = ch["ord"] - window, ch["ord"] + window
            # skip hits already contained in a passage we return
            if any(a <= ch["ord"] <= b for a, b in covered[ch["doc_id"]]):
                continue
            group = self.store.neighbor_chunks(ch["doc_id"], lo, hi) if window > 0 else [ch]
            # keep neighbours inside the same chapter stream only
            group = [g for g in group if g["section_idx"] == ch["section_idx"]] or [ch]
            # do not overlap with already returned passages
            group = [g for g in group if not any(a <= g["ord"] <= b for a, b in covered[ch["doc_id"]])] or [ch]
            covered[ch["doc_id"]].append((group[0]["ord"], group[-1]["ord"]))
            pages = [p for g in group for p in (g["page_start"], g["page_end"]) if p]
            passages.append(
                Passage(
                    rank=len(passages) + 1,
                    chunk_id=cid,
                    doc_id=ch["doc_id"],
                    title=ch["title"],
                    author=ch["author"],
                    category=ch["category"],
                    chapter=ch["chapter"],
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    score=round(final_scores[cid], 4),
                    rrf_score=round(fused[cid], 5),
                    keyword_rank=kw_rank.get(cid),
                    vector_rank=vec_rank.get(cid),
                    text=stitch(group),
                    chunk_ids=[g["id"] for g in group],
                )
            )
        t["expand"] = (time.perf_counter() - ts) * 1000
        t["total"] = (time.perf_counter() - t0) * 1000
        t = {k: round(v, 1) for k, v in t.items()}

        result = SearchResult(query, queries, passages, t, mode, bool(rerank), n_considered, pruned)
        if log_query:
            self._log(result, {"book": book, "author": author, "category": category})
        return result

    def read_around(self, chunk_id: int, window: int = 2) -> dict[str, Any] | None:
        """Return a wider passage around a chunk (for 'read more')."""
        ch = self.store.get_chunks([chunk_id]).get(chunk_id)
        if ch is None:
            return None
        group = self.store.neighbor_chunks(ch["doc_id"], ch["ord"] - window, ch["ord"] + window)
        pages = [p for g in group for p in (g["page_start"], g["page_end"]) if p]
        chapters = list(dict.fromkeys(g["chapter"] for g in group if g["chapter"]))
        return {
            "title": ch["title"],
            "author": ch["author"],
            "chapters": chapters,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "first_chunk_id": group[0]["id"],
            "last_chunk_id": group[-1]["id"],
            "text": stitch(group),
        }

    def _log(self, res: SearchResult, filters: dict[str, Any]) -> None:
        """Append every query to data/logs/queries.jsonl (observability / evaluation datasets)."""
        try:
            entry = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "query": res.query,
                "queries": res.queries,
                "filters": {k: v for k, v in filters.items() if v},
                "mode": res.mode,
                "reranked": res.reranked,
                "confidence": res.confidence,
                "timings_ms": res.timings_ms,
                "results": [
                    {"chunk_id": p.chunk_id, "title": p.title, "pages": [p.page_start, p.page_end], "score": p.score}
                    for p in res.passages
                ],
            }
            with open(self.cfg.logs_dir / "queries.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover
            log.debug("query log failed: %s", exc)


def format_results_markdown(res: SearchResult, max_chars: int | None = None) -> str:
    """Render results as compact, citeable markdown for Claude."""
    if not res.passages:
        return (
            f"No passages found for: {res.query!r}.\n"
            "Try different wording, synonyms, the language the book is written in, or remove filters."
        )
    advice = {
        "high": "Strong matches - answer from these passages.",
        "medium": "Partial matches - use what is relevant; consider one more reformulated search.",
        "low": "WEAK matches - the library may not cover this. Reformulate (synonyms, other language, "
               "keyword mode) before answering, and say so if still nothing relevant.",
        "unknown": "Not reranked - judge relevance yourself.",
    }[res.confidence]
    lines = [
        f"Retrieval confidence: {res.confidence.upper()} - {advice}",
        f"{len(res.passages)} passages (mode={res.mode}, reranked={res.reranked}, "
        f"{res.candidates} candidates, {res.pruned} pruned as irrelevant, {res.timings_ms.get('total', 0):.0f} ms). "
        "Use only these passages for book claims and cite them as [n].",
        "",
    ]
    for p in res.passages:
        text = p.text if not max_chars or len(p.text) <= max_chars else p.text[:max_chars] + " …"
        lines.append(f"[{p.rank}] {p.citation()}")
        lines.append(f"relevance={p.score:.3f} · chunk_id={p.chunk_id} · category={p.category or '-'}")
        lines.append("<<<")
        lines.append(text)
        lines.append(">>>")
        lines.append("")
    return "\n".join(lines)
