# 📚 rag-llm-book-advisor

**A local, advanced RAG (Retrieval-Augmented Generation) system for your personal book library, used
directly from Claude Code.**

Drop your books (PDF, EPUB, DOCX, TXT, …) into a folder and run `rag sync`. Then, in any Claude Code
chat, type:

```
/use-rag How can I build better habits and stop procrastinating?
```

Claude searches your books with **hybrid search** (BM25 keywords + multilingual semantic vectors),
**reranks** the results with a cross-encoder, reads the best passages, and answers **with citations
(book, chapter, page)**.

Everything except Claude runs **on your computer**: no cloud vector database, no API keys, no Docker,
and your books never leave your disk. It works with **Bulgarian and English** books (and 100+ other
languages), including cross-language questions: ask in Bulgarian, find the answer in an English book.

---

## Contents

1. [Features](#1-features)
2. [How it works](#2-how-it-works)
3. [Requirements](#3-requirements)
4. [Installation (step by step)](#4-installation-step-by-step)
5. [Adding books](#5-adding-books)
6. [Syncing books into the vector database](#6-syncing-books-into-the-vector-database)
7. [Using it from Claude Code (`/use-rag`)](#7-using-it-from-claude-code-use-rag)
8. [Using it from the terminal](#8-using-it-from-the-terminal-cli-reference)
9. [Chunking: how books are split and how to tune it](#9-chunking-how-books-are-split-and-how-to-tune-it)
10. [Configuration reference](#10-configuration-reference-configyaml)
11. [Models & performance](#11-models--performance)
12. [Measuring quality (`rag eval`)](#12-measuring-quality-rag-eval)
13. [Troubleshooting](#13-troubleshooting)
14. [Project structure](#14-project-structure)
15. [FAQ](#15-faq)

Design rationale and the research behind every choice: **[docs/RESEARCH.md](docs/RESEARCH.md)**.

---

## 1. Features

| Area | What you get |
|---|---|
| **File formats** | PDF (outline/TOC → chapters, page numbers), EPUB (chapters from TOC), DOCX (heading styles), TXT (auto-detects `CHAPTER 1` / `Глава 1`, cp1251 encodings), Markdown, HTML, Jupyter notebooks |
| **Chunking** | structure-aware recursive splitting (chapter → paragraph → sentence → word), 400 tokens with 20 % overlap, measured in real model tokens, never crossing chapters; optional `semantic` and `fixed` strategies |
| **Context headers** | each chunk is indexed with its book title, author and chapter, so passages are findable by book even when the text never names it |
| **Keyword search** | BM25 via SQLite FTS5 (Cyrillic + Latin, English stemming, diacritics folded) |
| **Semantic search** | `BAAI/bge-m3` multilingual embeddings in LanceDB (exact search; HNSW index for big libraries) |
| **Hybrid fusion** | weighted Reciprocal Rank Fusion across several query variants |
| **Reranking** | multilingual cross-encoder, int8-quantized for CPU speed |
| **Context pruning + confidence** | irrelevant passages are dropped; each search reports HIGH / MEDIUM / LOW confidence so Claude knows when to search again |
| **Small-to-big** | each hit is expanded with neighbouring chunks into a readable passage, without duplicated overlap |
| **Filters** | by book, author, category (sub-folder) or tags |
| **Incremental sync** | only new or changed books are processed; moved or renamed books are detected by content hash and not re-embedded; deleted books are removed |
| **Claude integration** | MCP server (`book-rag`) + `/use-rag` skill: query rewriting, BG↔EN translation, HyDE, iterative search, parallel sub-questions, cited and faithfulness-checked answers |
| **Evaluation** | `rag eval` compares BM25 / semantic / hybrid / hybrid + rerank (hit@k, MRR, precision@k) |
| **Observability** | every query is logged with per-stage timings to `data/logs/queries.jsonl` |

## 2. How it works

```
INDEXING (rag sync)
  library/*.pdf|epub|…
     └─► loader (text + chapters + pages)
          └─► recursive chunker (400 tokens, 80 overlap, per chapter)
               └─► "Title — Author / Chapter" header + chunk
                    ├─► bge-m3 (int8 ONNX) ─► vectors ─► LanceDB
                    └─► text ─────────────────────────► SQLite FTS5 (BM25)

QUERYING (/use-rag in Claude Code)
  your question
     └─► Claude writes 2–4 queries (clean question, keyword rewrite,
         BG↔EN translation, hypothetical answer passage) + filters
          └─► MCP tool search_library
               ├─► BM25 top 50 per query  ┐
               └─► vector top 50 per query ┴─► weighted RRF fusion
                    └─► top 15 ─► cross-encoder rerank ─► prune irrelevant
                         └─► expand with neighbour chunks ─► 6 cited passages
          └─► Claude checks confidence, searches again if needed,
              answers with [1] [2] citations and a Sources list
```

Claude is the **generator**. The local Python service is the **retriever**. Details and reasoning:
[docs/RESEARCH.md](docs/RESEARCH.md).

## 3. Requirements

* Windows 10/11 (also works on macOS/Linux; use the `.venv/bin/` paths there)
* **Python 3.10–3.13** (`python --version`)
* **[uv](https://docs.astral.sh/uv/)** recommended (`winget install astral-sh.uv`); plain `pip` also works
* **Claude Code** (`claude --version`)
* ~5 GB free disk for models (downloaded once), plus about 1 MB of index per 100 book pages
* 8 GB RAM minimum, 16 GB recommended. No GPU needed.

## 4. Installation (step by step)

Open **PowerShell**:

```powershell
# 1. Get the code
git clone https://github.com/Hard-system/rag-llm-book-advisor.git C:\workspace\rag
cd C:\workspace\rag

# 2. Create the virtual environment, install everything, download the models and run a self-test
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

On the first run the models are downloaded (about 3 GB). On first use each model is also
**converted to a fast int8 ONNX version**, which takes about 5 minutes once. At the end you should see:

```
✓ embedding model BAAI/bge-m3 dim=1024 (BG↔EN similarity 0.92) …
✓ reranker cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 scores=[…] …
✓ index: 0 books, 0 chunks, 0 vectors
```

```powershell
# 3. Connect it to Claude Code (registers the MCP server + installs the /use-rag skill globally)
powershell -ExecutionPolicy Bypass -File scripts\install-claude.ps1 -AddToPath
```

This does three things:

1. `claude mcp add book-rag --scope user -- C:\workspace\rag\.venv\Scripts\python.exe -m book_rag.mcp_server`
   makes the search tools available in **every** Claude Code session, whatever folder you are in.
2. Copies the skill to `%USERPROFILE%\.claude\skills\use-rag\SKILL.md`, so `/use-rag` works everywhere.
3. With `-AddToPath`, adds `C:\workspace\rag` to your PATH so `rag …` works in any new terminal.
   Without it, run `.\rag …` from the repo folder.

**Restart Claude Code** and check:

```powershell
claude mcp list          # should show: book-rag ... ✓ Connected
```

<details>
<summary>Manual installation (without the scripts)</summary>

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.venv\Scripts\rag.exe doctor
claude mcp add book-rag --scope user -- C:\workspace\rag\.venv\Scripts\python.exe -m book_rag.mcp_server
mkdir $env:USERPROFILE\.claude\skills\use-rag
copy .claude\skills\use-rag\SKILL.md $env:USERPROFILE\.claude\skills\use-rag\
```
</details>

## 5. Adding books

Put files into **`C:\workspace\rag\library\`**. **Sub-folders become categories**, which you can
filter on:

```
library/
├── psychology/
│   ├── Daniel Kahneman - Thinking, Fast and Slow.pdf
│   └── James Clear - Atomic Habits.epub
├── finance/
│   └── Робърт Кийосаки - Богат татко, беден татко.pdf
├── philosophy/stoicism/
│   └── Marcus Aurelius - Meditations.txt
└── _inbox/                ← folders starting with _ or . are ignored
```

* **Supported:** `.pdf .epub .docx .txt .md .html .htm .ipynb`
* **Title and author** are read from the file's metadata. If that is missing or junk (e.g.
  "Microsoft Word - final.docx"), the file name pattern **`Author - Title.ext`** is used.
* **Override metadata** with a sidecar file `<book file name>.yaml` next to the book:
  ```yaml
  # library/finance/kiyosaki.pdf.yaml
  title: Богат татко, беден татко
  author: Робърт Кийосаки
  language: bg
  tags: [finance, personal-development]
  ```
* **Scanned PDFs** (images of pages) are **OCR'd automatically** with RapidOCR (`ocr: true` in
  `config.yaml`). This takes about 10 s per page on CPU, once: results are cached in `data/ocr/`.
* **Excluding files**: list glob patterns in **`library/.ragignore`** (one per line, e.g. prompt
  templates or drafts). Folders starting with `_` or `.` are always ignored.
* **Duplicates are handled for you.** The same book saved twice (other file name, other folder,
  another copy) is detected by text similarity and **not embedded twice**. `rag list --all` shows
  which file was skipped as a duplicate of which book. Well-named files win over `inv(3).pdf` or `book (1).pdf`.
* **Generic file names** such as `Summary.docx`, `резюме.docx` or `New Microsoft Word Document.docx`
  get their title from their folder, e.g. *"Deep work rules for focused success — Summary"*.
* **PDFs without a table of contents**: chapter headings are detected from the font size, so
  chunks still stay inside chapters and citations show the chapter.
* DRM-protected e-books cannot be read. Use DRM-free copies.
* Use a different folder: set `library_dir:` in `config.yaml` or the `RAG_LIBRARY_DIR` environment variable.

## 6. Syncing books into the vector database

```powershell
rag sync
```

`sync` makes the index mirror the folder:

| You did… | `rag sync` does… |
|---|---|
| added a book | loads → chunks → embeds → indexes it |
| replaced a book with a new version | removes the old chunks, indexes the new file |
| moved or renamed a book (or its folder) | detects it by SHA-256 and only updates path and category (nothing re-embedded) |
| deleted a book | removes it from the index |
| saved the same book twice | indexes it once; the copy is recorded as a duplicate |
| added an empty or image-only file | OCRs it (scanned PDF) or records it as empty |
| edited a `<book>.yaml` sidecar | re-indexes that book with the new title/author |
| nothing | finishes in about a second |

Useful variants:

```powershell
rag sync --dry-run                          # show what would change
rag sync -f "library\psychology\new.pdf"    # index just one file
rag sync --rebuild                          # wipe and re-index everything (needed after changing
                                            # the embedding model, profile, backend or chunking settings)
rag list                                    # what is indexed (--all: also duplicates/empty)
rag sync -y                                 # no confirmation prompts (unattended runs)
rag stats                                   # counts, models, disk size
```

**How long does it take?** On a 4-core laptop CPU, about **1 s per chunk**. A 300-page book is
roughly 400–600 chunks, so 7–10 minutes, **once**. Re-syncs skip unchanged books. Let a large first
import run in a terminal (e.g. overnight). You can keep searching in Claude while a sync runs, and you can **stop it any time** (Ctrl+C): every
finished book is saved and `rag sync` continues where it stopped. Progress with an ETA is written
to `data/logs/sync.log`.

You can also ask Claude to sync ("sync my library"), which calls the `sync_library` tool. For big
imports the terminal is better because you see progress.

## 7. Using it from Claude Code (`/use-rag`)

In any Claude Code chat:

```
/use-rag What do my books say about overcoming procrastination?
/use-rag Как да инвестирам първите си спестявания?
/use-rag Compare what Kahneman and Taleb say about prediction
/use-rag book:"Atomic Habits" what is the 2-minute rule?
/use-rag category:stoicism how to deal with anger
/use-rag summarize the main ideas of "Meditations"
```

What happens:

1. **Query parsing.** Claude spots book titles, authors and categories and turns them into filters.
2. **Query expansion.** Claude writes several search variants: a cleaned question, a keyword
   rewrite with synonyms, a Bulgarian↔English translation, and a short hypothetical answer passage
   (HyDE).
3. **Hybrid search + rerank** in the local MCP server returns about 6 cited passages and a
   **confidence** level.
4. **Corrective loop.** On LOW or MEDIUM confidence Claude reformulates and searches again (up to 3
   rounds), or reads more around a good passage with `read_passage`.
5. **Answer.** Answers come in your language, with inline citations [1], [2] and a **Sources**
   list (`Title — Author, chapter, p. X`). Book-based claims are kept separate from Claude's
   general knowledge, and every citation is self-checked against its passage.

You don't always need the slash command. "What do my books say about sleep?" or "check my books
for…" also triggers the skill, and Claude can call the tools (`search_library`, `read_passage`,
`list_library`, `library_stats`, `sync_library`) directly whenever it decides they help.

## 8. Using it from the terminal (CLI reference)

Run `rag --help` or `rag <command> --help` for all options.

| Command | Purpose |
|---|---|
| `rag sync [--rebuild] [--dry-run] [-f FILE]` | index new, changed, moved or deleted books |
| `rag search "question" [options]` | search and print cited passages |
| `rag list` | list indexed books |
| `rag stats` | index statistics |
| `rag inspect FILE [-s strategy] [--size N] [--overlap N] [-n 5]` | preview parsing and chunking of a file **without indexing** |
| `rag eval FILE.jsonl [-k 5]` | measure retrieval quality |
| `rag optimize [--force]` | compact the index; build the HNSW ANN index (automatic above 100k chunks) |
| `rag doctor` | check installation, models and index |
| `rag mcp` | run the MCP server manually (Claude Code starts it automatically) |

`rag search` options:

```powershell
rag search "how to form habits" `
    -a "habit formation cue routine reward" `
    -a "как да изградим навици" `
    -k 8 --category psychology --mode hybrid --window 1
#  -a / --alt       extra query variants (repeatable)
#  -k / --top-k     passages to return
#  -b / --book      title/file-name filter      --author   author filter
#  -c / --category  sub-folder or tag filter
#  -m / --mode      hybrid | keyword | semantic
#  --no-rerank      skip the cross-encoder (faster)
#  -w / --window    neighbouring chunks to include (0 = just the hit)
#  --json           machine-readable output
```

## 9. Chunking: how books are split and how to tune it

Default: **`recursive`, 400 tokens, 80 tokens overlap (20 %)**.

1. The loader splits the book into **chapters** (PDF outline, EPUB TOC, DOCX headings, Markdown
   `#`, `CHAPTER`/`Глава` lines) and keeps **page numbers**.
2. Inside each chapter the text is split on **paragraphs**. Paragraphs that are too long are split
   on lines, then **sentences**, then clauses, then words. Each piece goes only as deep as needed.
3. Pieces are packed into chunks of at most 400 tokens (counted with the embedding model's own
   tokenizer). The last 80 tokens' worth of whole sentences are repeated at the start of the next
   chunk, so ideas at the edges are never lost.
4. Each chunk is indexed as `Title — Author\nChapter\n\n<text>`.

**Preview before indexing:**

```powershell
rag inspect "library\psychology\Atomic Habits.epub" -n 3
rag inspect book.pdf --size 250 --overlap 50     # try a different size
rag inspect book.pdf -s semantic                 # semantic chunking
```

**Which settings?**

| Book type | Suggested `chunk_size` / `chunk_overlap` |
|---|---|
| non-fiction, self-help, business (default) | 400 / 80 |
| dense technical / textbooks / law | 300 / 60 |
| fiction, narrative, philosophy with long arguments | 500 / 100 |
| Q&A, aphorisms, short notes | 200 / 30 |

Strategies (`chunking.strategy`):

* `recursive` (default): structure-aware, best all-rounder.
* `semantic`: starts a new chunk where the topic changes (sentence-embedding distance above the
  90th percentile). Can be more precise on books with frequent topic shifts, but indexing is about
  2× slower.
* `fixed`: plain token windows. A baseline for comparison.

After changing chunking settings run **`rag sync --rebuild`**, then compare with `rag eval`.

## 10. Configuration reference (`config.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `library_dir` | `library` | where your books are |
| `data_dir` | `data` | index, models cache, logs |
| `extensions` | pdf, epub, docx, txt, md, html, ipynb | file types to index |
| `profile` | `multilingual` | model pair: `multilingual`, `quality`, `english`, `light` (see §11) |
| `threads` | `0` | CPU threads (0 = physical cores) |
| `embedding.model` | (from profile) | any sentence-transformers model |
| `embedding.backend` | `onnx-int8` | `onnx-int8` (about 4× faster on CPU) or `torch` (full precision / GPU) |
| `embedding.batch_size` | 16 | embedding batch size |
| `embedding.max_seq_length` | 512 | max tokens per chunk fed to the model |
| `reranker.enabled` | true | use the cross-encoder |
| `reranker.candidates` | 15 | fused candidates to rerank (more = better recall, slower) |
| `chunking.strategy` | `recursive` | `recursive`, `semantic` or `fixed` |
| `chunking.chunk_size` | 400 | tokens per chunk |
| `chunking.chunk_overlap` | 80 | overlap tokens |
| `chunking.min_chunk_size` | 40 | merge smaller tails |
| `chunking.context_header` | true | prepend title, author and chapter when indexing |
| `retrieval.top_k` | 6 | passages returned |
| `retrieval.keyword_candidates` / `vector_candidates` | 50 / 50 | hits per retriever per query |
| `retrieval.rrf_k` | 60 | RRF constant |
| `retrieval.keyword_weight` / `vector_weight` | 1.0 / 1.0 | fusion weights (English-only library: 0.7 / 0.3 like the course's `alpha=0.3`) |
| `retrieval.neighbor_window` | 1 | neighbouring chunks added to each hit |
| `retrieval.min_relevance` | 0.05 | prune reranked passages below this score |
| `retrieval.ann_index_threshold` | 100000 | build the HNSW index above this many chunks |

Environment overrides: `RAG_CONFIG`, `RAG_HOME`, `RAG_LIBRARY_DIR`, `RAG_DATA_DIR`, `RAG_PROFILE`,
`RAG_EMBEDDING_MODEL`, `RAG_RERANKER_MODEL`, `RAG_RERANK=0/1`.

## 11. Models & performance

| Profile | Embeddings | Reranker | Indexing (4-core CPU) | Query | Use when |
|---|---|---|---|---|---|
| **multilingual** (default) | BAAI/bge-m3 | mmarco-mMiniLMv2-L12 (multilingual) | ~1 s/chunk | ~3–5 s | BG + EN library on a laptop |
| quality | BAAI/bge-m3 | BAAI/bge-reranker-v2-m3 | ~1 s/chunk | ~10–15 s | best precision; ideal with an NVIDIA GPU |
| english | BAAI/bge-base-en-v1.5 | BAAI/bge-reranker-base | ~0.3 s/chunk | ~2–4 s | English-only library (the course's models) |
| light | multilingual-e5-small | mmarco-mMiniLMv2-L12 | ~0.15 s/chunk | ~2–3 s | slow or old machines |

* **int8 quantization** (course module 5) is on by default: bge-m3 went from 3.3 s to about
  0.8–1.2 s per chunk on an i7-1165G7, with 0.978 cosine fidelity to full precision.
* With an **NVIDIA GPU**: install CUDA-enabled PyTorch, set `embedding.backend: torch` and
  `reranker.backend: torch`, and choose `profile: quality`.
* Changing `profile`, `embedding.model` or `embedding.backend` requires `rag sync --rebuild`. The
  tool refuses to mix vectors from different models.
* Models are cached in `data/models/`. After the first download everything works **offline**.

## 12. Measuring quality (`rag eval`)

Create a JSONL file with questions whose answers you know are in your books (see
[`evaluation/example_eval.jsonl`](evaluation/example_eval.jsonl)):

```json
{"question": "How long does it take to form a habit?", "book": "Atomic Habits", "contains": ["66 days"]}
{"question": "Какво е сложна лихва?", "book": "Богат татко", "page": 87}
```

A passage counts as relevant when it matches every given criterion: `book` (title substring),
`page` (within the passage's page range) and/or `contains` (text substring, any of a list).

```powershell
rag eval evaluation\my_questions.jsonl -k 5
```

```
            Retrieval evaluation on 20 questions (k=5)
 pipeline          hit@5    MRR    precision@5  avg ms
 keyword (BM25)    70.00%   0.583  0.180        12
 semantic          80.00%   0.650  0.210        180
 hybrid (RRF)      90.00%   0.741  0.250        190
 hybrid + rerank   95.00%   0.883  0.290        3400
```

Use it to compare settings (chunk size, weights, profile) on your own books. Real questions you
asked are logged in `data/logs/queries.jsonl`, which makes a good source for test questions.

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| `claude mcp list` shows book-rag failed | run `C:\workspace\rag\.venv\Scripts\python.exe -m book_rag.mcp_server` manually to see the error; re-run `scripts\install-claude.ps1` |
| `/use-rag` not found | restart Claude Code; check `%USERPROFILE%\.claude\skills\use-rag\SKILL.md` exists |
| `SSL: CERTIFICATE_VERIFY_FAILED` while downloading models or packages | antivirus or corporate TLS inspection. book-rag already uses the Windows certificate store (`truststore`); for installs use `uv pip install --native-tls …`, for git `git config --global http.sslBackend schannel` |
| "Index was built with … but config uses …" | you changed the model, profile or backend: run `rag sync --rebuild` |
| A PDF gives 0 chunks or a "scanned" warning | it is an image PDF: OCR it with `ocrmypdf` |
| Wrong title or author | add a `<file>.yaml` sidecar (see §5) and run `rag sync -f <file>` |
| First query in a session is slow | models load once per session (~10–30 s); the MCP server preloads them at start |
| Searches miss obvious passages | try `rag search` with `--mode keyword` and `--mode semantic`; check chunks with `rag inspect`; tune with `rag eval` |
| Out of memory | use `profile: light`, lower `embedding.batch_size` to 4 |
| Long MCP sync times out | run `rag sync` in a terminal instead |
| Start completely fresh | delete the `data\` folder (models are re-downloaded) or run `rag sync --rebuild` (keeps models) |

## 14. Project structure

```
rag/
├── README.md                    ← this manual
├── config.yaml                  ← all settings
├── pyproject.toml               ← package & dependencies (installs the `rag` command)
├── rag.cmd                      ← `rag …` wrapper for Windows
├── docs/RESEARCH.md             ← design rationale (course modules → decisions)
├── .claude/skills/use-rag/SKILL.md   ← the /use-rag skill for Claude Code
├── scripts/
│   ├── setup.ps1                ← create venv, install, download models, self-test
│   └── install-claude.ps1       ← register MCP server + install skill (+ PATH)
├── src/book_rag/
│   ├── config.py                ← config loading, model profiles
│   ├── loaders.py               ← PDF/EPUB/DOCX/TXT/MD/HTML/IPYNB → text + chapters + pages
│   ├── chunking.py              ← recursive / semantic / fixed chunking, token counting
│   ├── models.py                ← embedding model + reranker, int8 ONNX export
│   ├── store.py                 ← SQLite (metadata + FTS5 BM25) + LanceDB (vectors, HNSW)
│   ├── ingest.py                ← incremental sync (hashing, move detection)
│   ├── retrieval.py             ← hybrid search, RRF, rerank, pruning, small-to-big
│   ├── evaluate.py              ← hit@k / MRR / precision@k
│   ├── mcp_server.py            ← MCP tools for Claude Code
│   └── cli.py                   ← `rag` command line
├── evaluation/example_eval.jsonl
├── tests/test_core.py           ← unit tests (pytest)
├── library/                     ← YOUR BOOKS (git-ignored)
└── data/                        ← index, models, logs (git-ignored)
```

Run the tests: `.venv\Scripts\python.exe -m pytest -q`

## 15. FAQ

**Does it send my books anywhere?** No. Indexing and search are 100 % local. Only the passages
retrieved for a question are sent to Claude, as part of your conversation, the same as if you
pasted them yourself.

**Why not Weaviate / Chroma / Qdrant?** Weaviate's embedded mode (used in the course labs) doesn't
run on Windows, and a server means running Docker all the time. LanceDB + SQLite FTS5 give the same
HNSW + BM25 + hybrid + filter feature set embedded in one Python process. See
[docs/RESEARCH.md §3](docs/RESEARCH.md#3-vector-database--ann-m3).

**How many books can it handle?** Thousands. Exact vector search is fast up to ~100k chunks
(~250 books). Above that an HNSW index is built automatically. SQLite FTS5 scales to millions of rows.

**Can I use it from Claude Desktop too?** Yes. Add the same server to
`%APPDATA%\Claude\claude_desktop_config.json`:

```json
{ "mcpServers": { "book-rag": {
    "command": "C:\\workspace\\rag\\.venv\\Scripts\\python.exe",
    "args": ["-m", "book_rag.mcp_server"],
    "env": { "RAG_HOME": "C:\\workspace\\rag", "PYTHONUTF8": "1" } } } }
```

There, ask "search my book library for …". Skills are a Claude Code feature, but the tools work
the same way.

**Can I add my own notes?** Yes. Markdown or TXT files in the library are indexed like books.

## License

MIT
