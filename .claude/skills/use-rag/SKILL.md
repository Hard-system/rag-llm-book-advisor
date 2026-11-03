---
name: use-rag
description: Answer a question using the user's local book library (RAG over their PDFs/EPUBs via the book-rag MCP server). Use when the user types /use-rag, says "use rag", "check my books", "what do my books say about…", or asks for advice that should be grounded in their library.
argument-hint: <your question> [book:"title"] [author:"name"] [category:folder]
allowed-tools: mcp__book-rag__search_library, mcp__book-rag__read_passage, mcp__book-rag__list_library, mcp__book-rag__library_stats, mcp__book-rag__sync_library
---

# /use-rag — answer from the user's book library

The user's question: **$ARGUMENTS**

You are the *generator* of a Retrieval-Augmented Generation system. The `book-rag` MCP
server is the *retriever* (hybrid BM25 + vector search → RRF fusion → cross-encoder
rerank). Your job: plan the retrieval, retrieve, read, and write a grounded answer.

## 1. Parse the question (query parsing / NER)
- If $ARGUMENTS is empty, ask the user what they want to know (or call `list_library` if they want to see what is available).
- Extract entities that should become **metadata filters**: a book title → `book`, an author → `author`,
  a topic folder → `category`. Only filter when the user clearly restricts the scope
  (`book:"Atomic Habits"`, "according to Stoic books", "in Kahneman"). If unsure which titles exist, call `list_library` first.
- Detect the language. The library holds Bulgarian **and** English books; the embedding model is
  cross-lingual, but BM25 keyword search is not.

## 2. Build 2–4 queries (query rewriting + HyDE)
Call `search_library` once with:
- `query`: the user's question, cleaned up and made specific (this one is used for reranking).
- `alternative_queries`:
  1. a **keyword-rich rewrite** with synonyms and domain terms;
  2. a **translation** into the other library language (BG ↔ EN) so BM25 can match books in that language;
  3. a **HyDE passage**: 2–3 sentences written the way a book would answer the question.
- `top_k`: 6 for normal questions, 10–12 for broad "compare / summarize / what do my books say" questions.

**Multi-part or comparative questions** ("compare X and Y", "what do my books say about A, B and C"):
decompose into sub-questions and issue one `search_library` call per sub-question **in parallel in a
single message** (orchestrator → parallel retrieval → you synthesize). Number sources continuously
across all calls.

## 3. Evaluate and iterate (corrective / agentic retrieval)
Every result starts with `Retrieval confidence: HIGH | MEDIUM | LOW` (computed by the local
cross-encoder, which acts as the relevance evaluator; irrelevant passages are already pruned).
- **HIGH** → answer.
- **MEDIUM** → use what is relevant; one more reformulated search is often worth it.
- **LOW** → do not answer from these passages. Reformulate (synonyms, the other language,
  `mode: "keyword"` for names/quotes/exact terms, drop filters) and search again — max 3 rounds total.
- Per-passage `relevance` is a 0–1 probability: ≥ 0.5 strong, 0.15–0.5 partial, < 0.15 weak — ignore
  passages that do not actually address the question even if they were returned.
- If a passage is relevant but cut off, call `read_passage(chunk_id, window=3)` to read more of that book.

## 4. Answer (RAG prompt rules)
- Answer in the user's language. Think through the passages first; then write.
- **Book claims only from retrieved passages**, each cited inline as **[1]**, **[2]** right after the
  sentence it supports.
- End with a **Sources** list: `[n] Title — Author, chapter, p. X` (only sources you actually cited).
- When books disagree, show both views and attribute each one.
- Clearly separate what comes from the books from your own general knowledge
  (e.g. a short "Beyond the library:" paragraph). Never invent quotes, page numbers or titles.
- **Faithfulness self-check before sending:** for every sentence with a citation, confirm the cited
  passage really says it; remove or re-attribute anything unsupported. Quotes must be verbatim.
- Keep the answer focused: practical advice → concise steps with citations; "explain" → structured
  explanation; "summarize book X" → use `book` filter, several searches across topics, then synthesize.
- If nothing relevant is found, say so plainly, show what you searched for, and suggest adding a book
  on the topic to the library folder and running `rag sync`.
- If the library is empty or the tools are missing, explain how to set it up (README: `scripts/setup.ps1`,
  `scripts/install-claude.ps1`, then put books in `library/` and run `rag sync`).
