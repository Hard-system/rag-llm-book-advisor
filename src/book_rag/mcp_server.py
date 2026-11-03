"""MCP server exposing the book library to Claude Code (stdio transport).

Register it once (scripts/install-claude.ps1 does this for you):

    claude mcp add book-rag --scope user -- <repo>\\.venv\\Scripts\\python.exe -m book_rag.mcp_server

Tools:
  search_library   hybrid search + rerank, returns cited passages
  read_passage     read more text around a hit (expand context)
  list_library     what books are indexed
  library_stats    index statistics
  sync_library     index new / changed books in the library folder
"""

from __future__ import annotations

import logging
import os
import sys
import threading

# Nothing may be written to stdout except MCP protocol messages.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="[book-rag] %(levelname)s %(message)s")

import anyio  # noqa: E402

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server  # noqa: E402
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # noqa: E402

from .config import load_config  # noqa: E402
from .retrieval import Retriever, format_results_markdown  # noqa: E402

log = logging.getLogger("book_rag.mcp")

cfg = load_config()
cfg.ensure_dirs()
retriever = Retriever(cfg)
_sync_lock = threading.Lock()

mcp = _Server(
    "book-rag",
    instructions=(
        "Local library of the user's books (Bulgarian and English). Use search_library to ground answers "
        "in the books, pass 2-4 alternative_queries (keyword rewrite, translation to the other language, "
        "a short hypothetical answer passage), and cite passages as [n] with book, chapter and page."
    ),
)


@mcp.tool()
async def search_library(
    query: str,
    alternative_queries: list[str] | None = None,
    top_k: int = 6,
    book: str | None = None,
    author: str | None = None,
    category: str | None = None,
    mode: str = "hybrid",
    rerank: bool = True,
) -> str:
    """Search the user's book library with hybrid search (BM25 keywords + semantic vectors), fused with
    Reciprocal Rank Fusion and re-ranked by a cross-encoder. Returns cited passages.

    Args:
        query: The user's question, as clear and specific as possible (this one is used for reranking).
        alternative_queries: 1-4 extra variants that improve recall: a keyword-rich rewrite with synonyms,
            a translation (Bulgarian <-> English), and/or a short hypothetical passage that would answer
            the question (HyDE).
        top_k: Number of passages to return (1-20).
        book: Optional filter - substring of the book title or file name.
        author: Optional filter - substring of the author name.
        category: Optional filter - library sub-folder (e.g. "psychology") or tag.
        mode: "hybrid" (default), "keyword" (exact terms, names, quotes) or "semantic" (meaning only).
        rerank: Use the cross-encoder reranker (more precise, ~2-4 s on CPU).
    """
    top_k = max(1, min(int(top_k or 6), 20))

    def run():
        return retriever.search(query, alternative_queries or [], top_k, book, author, category, mode, rerank)

    try:
        res = await anyio.to_thread.run_sync(run)
    except Exception as exc:
        log.exception("search failed")
        return f"Search failed: {exc}"
    return format_results_markdown(res)


@mcp.tool()
async def read_passage(chunk_id: int, window: int = 3) -> str:
    """Read a longer stretch of a book around a passage returned by search_library.

    Args:
        chunk_id: The chunk_id shown in a search result.
        window: How many chunks before and after to include (1-10). Each chunk is ~400 tokens.
    """
    window = max(1, min(int(window), 10))
    data = await anyio.to_thread.run_sync(lambda: retriever.read_around(chunk_id, window))
    if not data:
        return f"chunk_id {chunk_id} not found."
    pages = ""
    if data["page_start"]:
        pages = f" | pp. {data['page_start']}–{data['page_end']}"
    chapters = " / ".join(data["chapters"]) or "-"
    return (
        f"{data['title']}{' — ' + data['author'] if data['author'] else ''} | {chapters}{pages}\n"
        f"(chunks {data['first_chunk_id']}..{data['last_chunk_id']}; call again with those ids to keep reading)\n"
        f"<<<\n{data['text']}\n>>>"
    )


@mcp.tool()
async def list_library(category: str | None = None) -> str:
    """List the books in the library (title, author, category, pages, chunks).

    Args:
        category: Optional sub-folder / tag filter.
    """
    docs = await anyio.to_thread.run_sync(retriever.store.documents)
    if category:
        docs = [d for d in docs if category.lower() in (d["category"] or "").lower() or category.lower() in (d["tags"] or "").lower()]
    if not docs:
        return f"The library is empty. Put books into {cfg.library_dir} and run `rag sync` (or call sync_library)."
    lines = [f"{len(docs)} books:"]
    for d in docs:
        lines.append(
            f"- {d['title']}{' — ' + d['author'] if d['author'] else ''} "
            f"[{d['category'] or 'root'}] {d['file_type']}, {d['n_pages'] or '?'} pages, {d['n_chunks']} chunks"
        )
    return "\n".join(lines)


@mcp.tool()
async def library_stats() -> str:
    """Index statistics: number of books, chunks, models in use, disk size."""
    import json

    s = await anyio.to_thread.run_sync(retriever.store.stats)
    s["reranker"] = cfg.reranker.model if cfg.reranker.enabled else "disabled"
    s["library_dir"] = str(cfg.library_dir)
    return json.dumps(s, ensure_ascii=False, indent=2)


@mcp.tool()
async def sync_library() -> str:
    """Index new, changed, moved or deleted books in the library folder (incremental).
    Large new books can take several minutes on CPU; for big imports prefer `rag sync` in a terminal."""
    from .ingest import Indexer

    def run():
        if not _sync_lock.acquire(blocking=False):
            return "A sync is already running."
        try:
            idx = Indexer(cfg, store=retriever.store, embedder=retriever.embedder)
            return idx.sync(progress=lambda m: log.info(m))
        finally:
            _sync_lock.release()

    rep = await anyio.to_thread.run_sync(run)
    if isinstance(rep, str):
        return rep
    out = "Sync finished: " + rep.summary()
    if rep.added:
        out += "\nAdded: " + ", ".join(rep.added)
    if rep.failed:
        out += "\nFailed: " + "; ".join(f"{p}: {e}" for p, e in rep.failed)
    if rep.warnings:
        out += "\nWarnings: " + "; ".join(f"{p}: {w}" for p, w in rep.warnings)
    return out


def _warmup() -> None:
    try:
        retriever.warmup()
        log.info("models loaded - ready")
    except Exception as exc:  # pragma: no cover
        log.warning("warmup failed: %s", exc)


def main() -> None:
    threading.Thread(target=_warmup, daemon=True).start()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
