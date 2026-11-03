# Research & design decisions

This document explains **why** book-rag is built the way it is. Each decision traces back to the
DeepLearning.AI RAG course material (modules M1–M5 and the labs), adapted for one target: a
**local Windows laptop** (4-core i7-1165G7, 16 GB RAM, no NVIDIA GPU) holding a **mixed Bulgarian +
English** book library, used from **Claude Code**.

---

## 1. Architecture at a glance

```
                ┌────────────────────────── Claude Code ──────────────────────────┐
  you: /use-rag │  skill = query parsing · rewriting · HyDE · translation ·        │
                │          relevance judgement · grounded, cited answer            │
                └───────────────┬──────────────────────────────▲──────────────────┘
                                │ MCP (stdio)                  │ cited passages + confidence
                ┌───────────────▼──────────────────────────────┴──────────────────┐
                │ book-rag MCP server (Python, models stay loaded in memory)        │
                │                                                                    │
                │  queries ─┬─► BM25 (SQLite FTS5) ──┐                              │
                │           └─► dense (LanceDB)  ────┼─► weighted RRF ─► cross-     │
                │      metadata filter (book/author/category) applied to both       │
                │                                     encoder rerank (int8 ONNX) ─► │
                │                     context pruning ─► small-to-big expansion     │
                └────────────────────────────────────────────────────────────────────┘
                ┌────────────────────────────────────────────────────────────────────┐
  rag sync ───► │ loaders (PDF/EPUB/DOCX/TXT/MD/HTML/IPYNB) → chapter/page aware      │
                │ recursive chunking (400 tok, 20 % overlap) → contextual header →   │
                │ bge-m3 int8 embeddings → SQLite + LanceDB (incremental, hash based) │
                └────────────────────────────────────────────────────────────────────┘
```

**Key idea:** the course splits RAG into a *retriever* and a *generator* (M1). Here **Claude is the
generator** and also does the LLM-side tricks (query rewriting, HyDE, NER, evaluation). Those
normally need extra LLM calls and API keys. The local Python side does what small, fast models do
best: retrieval and reranking.

---

## 2. Retrieval: hybrid search (M2)

| Technique | Strength | Weakness | Used here |
|---|---|---|---|
| Keyword search: TF-IDF → **BM25** | exact terms, names, rare words, quotes | no synonyms, no cross-language matching | ✅ SQLite FTS5 `bm25()` |
| Semantic search (bi-encoder) | meaning, synonyms, paraphrases, **BG↔EN** | can miss exact rare terms | ✅ bge-m3 in LanceDB |
| Metadata filtering | strict scoping | not a search by itself | ✅ book / author / category |
| **Hybrid** = both lists fused | best of both | needs a fusion method | ✅ **weighted RRF** |

* **BM25 over TF-IDF** (M2): term-frequency saturation (k₁) and gentler length normalisation (b)
  give the same cost as TF-IDF with better rankings. FTS5 implements Okapi BM25 natively. Its
  `unicode61 remove_diacritics` tokenizer handles Cyrillic, and `porter` stems English words.
  Columns are weighted at query time: body 1.0, chapter 0.8, title 0.5.
* **Reciprocal Rank Fusion**: `score(d) = Σ wᵢ / (k + rankᵢ(d))`, with k = 60. RRF uses ranks only,
  so BM25 scores and cosine similarities never need to be put on the same scale. Weights play the
  role of Weaviate's `alpha` from the M3 lab:
  * **Mixed BG/EN library → 1.0 / 1.0.** BM25 cannot match a Bulgarian question to an English
    book, so leaning on keywords (the lab's `alpha = 0.3`) would hurt cross-language questions.
  * **English-only library →** `keyword_weight: 0.7, vector_weight: 0.3` reproduces the course setting.
* **Multi-query fusion.** Claude sends the original question plus 2–4 variants (rewrite,
  translation, HyDE passage). Each variant is searched with both retrievers, and all lists go into
  one RRF. Documents found by several queries and both retrievers rise to the top.

## 3. Vector database & ANN (M3)

The course uses **Weaviate**. It is excellent, but:

* `weaviate.connect_to_embedded()` (used in the labs) **does not run on Windows**. On Windows,
  Weaviate needs Docker running all the time, which is heavy on a laptop.
* For a personal library (10³–10⁶ chunks), an **embedded** database in the same process is
  simpler, faster to start and has nothing to administer.

**Choice: LanceDB (vectors) + SQLite FTS5 (BM25).** Both are embedded, file-based and free, and
together they give exactly the Weaviate feature set the course uses:

| Weaviate concept (M3) | book-rag equivalent |
|---|---|
| collection + properties | SQLite `documents` / `chunks` tables |
| `near_text` (HNSW ANN) | LanceDB cosine search; `IVF_HNSW_SQ` index for big libraries |
| `bm25` | FTS5 `bm25()` |
| `hybrid(alpha=…)` | weighted RRF of both lists |
| `filters=Filter.by_property(...)` | `book` / `author` / `category` pre-filter on both retrievers |
| `rerank=Rerank(...)` | cross-encoder reranker |

**KNN vs ANN.** Below ~100k chunks (about 250 books), exact brute-force KNN takes milliseconds and
has 100 % recall, so no index is built. Above that, `rag optimize` builds
**HNSW + 8-bit scalar quantization** (`IVF_HNSW_SQ`), the hierarchical proximity graph explained in
M3, with quantized vectors as recommended in M5. The threshold is `ann_index_threshold`.

## 4. Chunking (M3 + chunking lab)

Course findings:

* Whole books as single vectors give "averaged", useless embeddings. Chunk at paragraph scale.
* Overlap (lab: `overlap_fraction = 0.2`) reduces cut-off context.
* Fixed-size and **recursive character splitting are good defaults**. Semantic and LLM chunking
  can do better but cost more. **Context-aware chunking is a good first improvement.**
* The lab chunks a book by **chapter → section → paragraph** and keeps `chapter_title`,
  `filename` and `chunk_index` as metadata.

Implementation (`chunking.py`):

1. **Structure first.** Loaders produce sections tagged with chapter and page: the PDF outline/TOC,
   EPUB navigation, DOCX heading styles, Markdown `#`, or `CHAPTER`/`Глава` lines in TXT.
   **Chunks never cross a chapter boundary.**
2. **Recursive splitting.** Split on paragraphs, then lines, then sentences (including Cyrillic
   punctuation and quotes), then clauses, then words. A piece only goes one level deeper if it is
   still too large.
3. **Measured in real tokens** of the embedding model's own tokenizer, not characters, so no chunk
   is silently truncated by the model.
4. **Greedy packing to 400 tokens with 80 tokens (20 %) of overlap**, made of whole sentences or
   paragraphs. Trailing sentences are carried over when a paragraph is too big to repeat. Tiny
   tails are merged (the lab's `min_length` idea).
5. **Contextual chunk header.** `"Title — Author\nChapter"` is prepended to each chunk before
   embedding, and stored as extra BM25 columns. This is the cheap, LLM-free version of
   *context-aware chunking*: a passage that never names its book still matches
   "what does *Atomic Habits* say about…".
6. **Exact character offsets** are stored, so neighbouring chunks can be stitched back together
   without duplicating the overlap (used by small-to-big retrieval).

Why 400 tokens: big enough to hold a complete idea or argument from a non-fiction book (about
300 words), small enough for a sharp embedding, and about 1.6k tokens of context per passage after
neighbour expansion. `rag inspect book.pdf --size 300 --overlap 60` lets you try other sizes.
`strategy: semantic` (Kamradt-style breakpoints on sentence-embedding distance) and
`strategy: fixed` are available for experiments with `rag eval`.

## 5. Models — chosen for a 4-core laptop CPU and Bulgarian + English

The course labs use `BAAI/bge-base-en-v1.5` (embeddings) and `BAAI/bge-reranker-base` (reranker).
Both are **English-only**. For a mixed library:

| Role | Default | Why |
|---|---|---|
| Embeddings | **BAAI/bge-m3** | top-tier multilingual retrieval model (100+ languages incl. Bulgarian), 1024-dim, 8k context; BG↔EN query similarity ≈ 0.92 |
| Reranker | **cross-encoder/mmarco-mMiniLMv2-L12-H384-v1** | multilingual cross-encoder, 118M params; ranked the same top passages as the large `bge-reranker-v2-m3` in our tests (EN and BG queries) at about ⅓ of the CPU time |

Profiles in `config.yaml`: `multilingual` (default), `quality` (bge-reranker-v2-m3; best precision,
use it if you get a GPU), `english` (the course models, about 3× faster indexing), and `light`
(multilingual-e5-small).

### int8 quantization (M5) — the biggest speed-up on CPU

M5 shows that 8-bit integer quantization costs embedding models only a few points of recall. Each
model is exported once to ONNX and its weights are dynamically quantized to int8
(AVX-512 VNNI kernels on this CPU). Measured on this laptop:

| Model | PyTorch fp32 | ONNX int8 | Fidelity |
|---|---|---|---|
| bge-m3, per 400-token chunk | 3.3 s | **0.8–1.2 s** | cosine to fp32 vectors 0.978 |
| bge-reranker-v2-m3, 20 pairs | 77 s | 23 s | Spearman rank correlation 0.985, identical top 3 |
| mMiniLMv2 reranker, 15–20 pairs | 6.2 s | **3.4 s** | same top passages |

## 6. Query-side techniques (M3) — done by Claude, for free

| Course technique | Where it happens |
|---|---|
| **Query rewriting** (clarify, add synonyms, remove noise) | `/use-rag` skill → `alternative_queries` |
| **HyDE** (embed a hypothetical answer passage) | skill writes a 2–3 sentence "book-like" answer as one of the queries |
| **NER (GLiNER)** → filters (book, person, …) | Claude extracts titles and authors → `book` / `author` / `category` filters |
| Cross-language search | Claude adds a BG↔EN translation so BM25 also matches books in the other language |

Running GLiNER plus an LLM rewriter locally would add seconds of CPU time and another model.
Claude does all of this better, inside the same turn.

## 7. Reranking (M3)

Bi-encoders are fast but approximate. Cross-encoders read query and passage **together** and are
much more precise, but too slow to run on everything. Hence the course's two-stage design:
**hybrid retrieval → top 15 fused candidates → cross-encoder → best 6.** ColBERT-style late
interaction was left out: it needs a vector per token (≈ 400× the storage), and the cross-encoder
stage already captures most of the gain.

## 8. Generation-side techniques (M4) — applied to Claude as the LLM

| M4 technique | Applied as |
|---|---|
| RAG prompt template: "use only retrieved documents, judge relevance, cite [DOC n], admit missing info" | rules in the `/use-rag` skill; the MCP output is formatted as numbered, cited passages |
| **Context-window management / pruning** | passages with reranker relevance < `min_relevance` are dropped before reaching Claude, so fewer tokens and less noise |
| Lightweight model as **evaluator** in agentic RAG | the local cross-encoder produces a **retrieval confidence** (HIGH / MEDIUM / LOW). On LOW the skill makes Claude reformulate and search again (corrective RAG) instead of answering from weak context |
| **Iterative workflow** (writer ↔ evaluator) | up to 3 search rounds; `read_passage` to read further into a book |
| **Parallel workflow** (orchestrator → workers → synthesizer) | multi-part questions are split into sub-questions searched in parallel tool calls |
| Citation generation / ContextCite-style attribution, **faithfulness** (RAGAS) | inline [n] citations plus a mandatory self-check that every cited sentence is supported by its passage |
| Chain-of-thought / reasoning models | Claude reasons over the passages before writing |
| Self-consistency | **not used**: M4 calls it costly and unreliable |
| Fine-tuning | **not used**: M4 says "fine-tuning does not teach new facts well; RAG is best for knowledge injection" |

Query embeddings are cached in the server, so iterative rounds that repeat a query cost nothing.

## 9. Evaluation & observability (M2, M5)

* `rag eval tests.jsonl` computes **hit-rate@k, MRR and precision@k** for four pipelines side by
  side: BM25 only, vectors only, hybrid, and hybrid + rerank. Use it to check the effect of any
  change (chunk size, weights, models) on your own books, as the M5 "experiment, then keep what
  helps" loop recommends.
* Every query is appended to `data/logs/queries.jsonl` with the variants, filters, per-stage
  timings (keyword / embed / vector / rerank) and returned passages. This is a lightweight trace
  (M5 observability), and a source of real questions for your evaluation set.

## 10. Indexing & sync

* Incremental: files are fingerprinted by size, mtime and **SHA-256**. Unchanged books are skipped,
  changed books are re-indexed, **moved or renamed books are detected by hash and only their path
  is updated** (nothing re-embedded), and deleted books are removed.
* The index records the embedding model and backend. Switching models without
  `rag sync --rebuild` is refused, so vectors from different spaces are never mixed (M2: "only
  compare vectors from the same embedding model").
* SQLite runs in WAL mode and LanceDB with `read_consistency_interval=0`, so you can sync in a
  terminal while Claude searches through the MCP server.

## 11. Ideas for later

* **Semantic chunking / LLM context headers per chunk** (Anthropic's "contextual retrieval"):
  better recall, at the cost of an LLM call per chunk during indexing.
* **OCR** for scanned PDFs (`ocrmypdf`) built into `rag sync`.
* **GPU**: with an NVIDIA card, set `backend: torch` and `profile: quality`. Indexing becomes
  10–50× faster.
* A **file watcher** that runs `rag sync` automatically when a book is added.
