"""Retrieval evaluation (course module 2 & 5): hit-rate@k, MRR and precision@k.

Evaluation file = JSON Lines, one question per line::

    {"question": "How long does it take to form a habit?", "book": "Atomic Habits", "contains": "66 days"}
    {"question": "Какво е сложна лихва?", "book": "Богатият татко", "page": 87}

A retrieved passage counts as *relevant* when every given criterion matches:
``book`` (substring of the title), ``page`` (inside the passage page range)
and/or ``contains`` (case-insensitive substring of the passage text; can be a
list = any of them).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .retrieval import Passage, Retriever


@dataclass
class EvalRow:
    mode: str
    hit_rate: float
    mrr: float
    precision: float
    avg_ms: float
    n: int


def is_relevant(p: Passage, case: dict[str, Any]) -> bool:
    if case.get("book") and case["book"].lower() not in p.title.lower():
        return False
    if case.get("page") is not None:
        if p.page_start is None or not (p.page_start <= int(case["page"]) <= (p.page_end or p.page_start)):
            return False
    if case.get("contains"):
        needles = case["contains"] if isinstance(case["contains"], list) else [case["contains"]]
        if not any(n.lower() in p.text.lower() for n in needles):
            return False
    return True


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def evaluate(retriever: Retriever, cases: list[dict[str, Any]], k: int = 5,
             configs: list[tuple[str, str, bool]] | None = None) -> list[EvalRow]:
    configs = configs or [
        ("keyword (BM25)", "keyword", False),
        ("semantic", "semantic", False),
        ("hybrid (RRF)", "hybrid", False),
        ("hybrid + rerank", "hybrid", True),
    ]
    rows = []
    for label, mode, rerank in configs:
        hits = rr = prec = ms = 0.0
        for case in cases:
            res = retriever.search(case["question"], top_k=k, mode=mode, rerank=rerank,
                                   neighbor_window=0, log_query=False)
            rel = [is_relevant(p, case) for p in res.passages]
            if any(rel):
                hits += 1
                rr += 1.0 / (rel.index(True) + 1)
            prec += sum(rel) / k
            ms += res.timings_ms.get("total", 0)
        n = max(1, len(cases))
        rows.append(EvalRow(label, hits / n, rr / n, prec / n, ms / n, len(cases)))
    return rows
