"""Command line interface:  rag <command> --help"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .config import load_config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="book-rag - local hybrid-search RAG over your book library (use it from Claude Code with /use-rag).",
)
console = Console()


def _setup(verbose: bool = False):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config()
    cfg.ensure_dirs()
    return cfg


@app.command()
def sync(
    rebuild: bool = typer.Option(False, "--rebuild", help="Delete the index and re-embed everything (needed after changing the embedding model or chunking settings)."),
    file: Optional[list[Path]] = typer.Option(None, "--file", "-f", help="Only (re)index these files (must be inside the library folder)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change without indexing."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation (for unattended runs)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Synchronise the vector database with the library folder (add / update / move / remove books)."""
    cfg = _setup(verbose)
    from .ingest import Indexer

    console.print(f"[bold]Library:[/] {cfg.library_dir}\n[bold]Index:[/]   {cfg.data_dir}\n"
                  f"[bold]Model:[/]   {cfg.embedding.model} · chunking={cfg.chunking.strategy} "
                  f"{cfg.chunking.chunk_size}/{cfg.chunking.chunk_overlap} tokens")
    if rebuild and not dry_run and not yes and not typer.confirm("This deletes the existing index and re-embeds every book. Continue?"):
        raise typer.Exit(1)
    for f in file or []:
        if cfg.library_dir.resolve() not in f.resolve().parents:
            console.print(f"[red]{f} is not inside {cfg.library_dir}[/]")
            raise typer.Exit(2)
    idx = Indexer(cfg)
    log_path = cfg.logs_dir / "sync.log"

    def progress(msg: str) -> None:
        console.print(escape(msg))
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    rep = idx.sync(rebuild=rebuild, only=file, progress=progress, dry_run=dry_run)
    progress("SYNC FINISHED: " + rep.summary())
    color = "red" if rep.failed else "green"
    console.print(Panel(rep.summary(), title="dry run" if dry_run else "sync finished", border_style=color))
    for path, why in rep.duplicates:
        console.print(f"[yellow]duplicate[/] {escape(path)}: {escape(why)}")
    for path in rep.empty:
        console.print(f"[yellow]no text[/] {escape(path)}")
    for path, err in rep.failed:
        console.print(f"[red]failed[/] {escape(path)}: {escape(err)}")
    if rep.failed:
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Your question or search terms."),
    alt: Optional[list[str]] = typer.Option(None, "--alt", "-a", help="Extra query variants (rewrites, translations, HyDE passage). Repeatable."),
    k: int = typer.Option(0, "--top-k", "-k", help="Passages to return (default from config)."),
    book: Optional[str] = typer.Option(None, "--book", "-b", help="Only books whose title/path contains this."),
    author: Optional[str] = typer.Option(None, "--author"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Library sub-folder or tag."),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="hybrid | keyword | semantic"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip the cross-encoder (faster)."),
    window: int = typer.Option(-1, "--window", "-w", help="Neighbour chunks to add around each hit (default from config)."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Search the library and print cited passages."""
    cfg = _setup(verbose)
    from .retrieval import Retriever, format_results_markdown

    r = Retriever(cfg)
    res = r.search(query, alt, k or None, book, author, category, mode,
                   rerank=False if no_rerank else None, neighbor_window=None if window < 0 else window)
    if as_json:
        sys.stdout.write(json.dumps(res.to_dict(), ensure_ascii=False, indent=2) + "\n")
        return
    if not res.passages:
        console.print(format_results_markdown(res))
        return
    for p in res.passages:
        meta = (f"relevance {p.score:.3f} · BM25 rank {p.keyword_rank or '-'} · vector rank {p.vector_rank or '-'}"
                f" · chunk {p.chunk_id}")
        console.print(Panel(escape(p.text), title=f"[{p.rank}] {escape(p.citation())}", subtitle=meta,
                            title_align="left", border_style="cyan"))
    console.print(f"[dim]timings (ms): {res.timings_ms}[/]")


@app.command("list")
def list_books(
    all_: bool = typer.Option(False, "--all", "-a", help="Also show skipped duplicates and empty files."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """List indexed books."""
    cfg = _setup(verbose)
    from .store import Store

    docs = Store(cfg).documents(status=None if all_ else "indexed")
    t = Table(title=f"{len(docs)} documents in {cfg.library_dir}")
    for col in ("id", "status", "category", "title", "author", "pages", "chunks", "file"):
        t.add_column(col, overflow="fold")
    for d in docs:
        status = d.get("status") or "indexed"
        if status == "duplicate":
            status = f"duplicate of #{d.get('duplicate_of')}"
        t.add_row(str(d["id"]), status, d["category"] or "-", d["title"], d["author"] or "-",
                  str(d["n_pages"] or "-"), str(d["n_chunks"]), d["path"])
    console.print(t)


@app.command()
def stats(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Show index statistics."""
    cfg = _setup(verbose)
    from .store import Store

    s = Store(cfg).stats()
    s.update({"profile": cfg.profile, "configured_model": cfg.embedding.model, "reranker": cfg.reranker.model if cfg.reranker.enabled else "disabled",
              "chunking": f"{cfg.chunking.strategy} {cfg.chunking.chunk_size}/{cfg.chunking.chunk_overlap}"})
    console.print_json(json.dumps(s, ensure_ascii=False))


@app.command()
def inspect(
    file: Path = typer.Argument(..., exists=True, help="Any supported file (does not need to be in the library)."),
    show: int = typer.Option(3, "--show", "-n", help="How many chunks to print."),
    strategy: Optional[str] = typer.Option(None, "--strategy", "-s", help="recursive | fixed | semantic"),
    size: Optional[int] = typer.Option(None, "--size"),
    overlap: Optional[int] = typer.Option(None, "--overlap"),
):
    """Preview how a file is parsed and chunked (nothing is indexed) - use it to tune chunking."""
    cfg = _setup(False)
    if strategy:
        cfg.chunking.strategy = strategy
    if size:
        cfg.chunking.chunk_size = size
    if overlap is not None:
        cfg.chunking.chunk_overlap = overlap
    from .ingest import Indexer

    idx = Indexer(cfg)
    doc, chunks = idx.chunk_file(file)
    toks = [c.n_tokens for c in chunks] or [0]
    chapters = list(dict.fromkeys(c.chapter for c in chunks if c.chapter))
    console.print(Panel(
        f"title: {doc.title}\nauthor: {doc.author}\ntype: {doc.file_type} · pages: {doc.n_pages} · chars: {doc.char_count:,}\n"
        f"chapters detected: {len(chapters)}\nchunks: {len(chunks)} · tokens min/avg/max: "
        f"{min(toks)}/{sum(toks)//len(toks)}/{max(toks)}\nstrategy: {cfg.chunking.strategy} "
        f"{cfg.chunking.chunk_size}/{cfg.chunking.chunk_overlap}"
        + ("\nwarnings: " + "; ".join(doc.warnings) if doc.warnings else ""),
        title=escape(file.name)))
    for c in chunks[:show]:
        console.print(Panel(escape(c.text), title=f"chunk {c.ord} · {escape(c.chapter or '-')} · p.{c.page_start}-{c.page_end} · {c.n_tokens} tok",
                            title_align="left", border_style="magenta"))


@app.command()
def optimize(force: bool = typer.Option(False, "--force", help="Build the ANN index even below the threshold.")):
    """Compact the index and build the HNSW vector index for large libraries."""
    cfg = _setup(False)
    from .store import Store

    console.print(Store(cfg).optimize(force_index=force))


@app.command("eval")
def eval_cmd(
    file: Path = typer.Argument(..., exists=True, help="JSONL test set (see evaluation/example_eval.jsonl)."),
    k: int = typer.Option(5, "--top-k", "-k"),
):
    """Measure retrieval quality (hit-rate@k, MRR, precision@k) for keyword / semantic / hybrid / hybrid+rerank."""
    cfg = _setup(False)
    from .evaluate import evaluate, load_cases
    from .retrieval import Retriever

    cases = load_cases(file)
    rows = evaluate(Retriever(cfg), cases, k)
    t = Table(title=f"Retrieval evaluation on {len(cases)} questions (k={k})")
    for col in ("pipeline", f"hit@{k}", "MRR", f"precision@{k}", "avg ms"):
        t.add_column(col)
    for r in rows:
        t.add_row(r.mode, f"{r.hit_rate:.2%}", f"{r.mrr:.3f}", f"{r.precision:.3f}", f"{r.avg_ms:.0f}")
    console.print(t)


@app.command()
def doctor():
    """Check the installation, download the models and run a self-test."""
    import time

    cfg = _setup(True)
    console.print(f"Python {sys.version.split()[0]} · config profile: [bold]{cfg.profile}[/]")
    console.print(f"library: {cfg.library_dir} (exists={cfg.library_dir.exists()})")
    console.print(f"data:    {cfg.data_dir}")
    import torch

    console.print(f"torch {torch.__version__} · cuda={torch.cuda.is_available()} · threads={torch.get_num_threads()}")
    from .models import Embedder, Reranker

    t = time.perf_counter()
    e = Embedder(cfg)
    v = e.embed_queries(["Какво е щастието?", "What is happiness?"])
    console.print(f"[green]✓[/] embedding model {cfg.embedding.model} dim={v.shape[1]} "
                  f"(BG↔EN similarity {float(v[0] @ v[1]):.2f}) loaded in {time.perf_counter() - t:.1f}s")
    if cfg.reranker.enabled:
        t = time.perf_counter()
        s = Reranker(cfg).score("What is the capital of Canada?", ["Ottawa is the capital of Canada.", "Canada is the maple syrup capital of the world."])
        console.print(f"[green]✓[/] reranker {cfg.reranker.model} scores={[round(float(x), 3) for x in s]} "
                      f"loaded in {time.perf_counter() - t:.1f}s")
    from .store import Store

    st = Store(cfg).stats()
    console.print(f"[green]✓[/] index: {st['documents']} books, {st['chunks']} chunks, {st['vectors']} vectors")


@app.command()
def mcp():
    """Run the MCP server on stdio (this is what Claude Code starts automatically)."""
    from .mcp_server import main

    main()


if __name__ == "__main__":
    app()
