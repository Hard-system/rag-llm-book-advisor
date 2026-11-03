"""Fast unit tests - no models are downloaded (approximate token counter, fake embeddings)."""

from pathlib import Path

import numpy as np
import pytest

from book_rag.chunking import approx_token_counter, chunk_document, context_header, stitch
from book_rag.config import ChunkingConfig, load_config
from book_rag.loaders import LoadedDocument, Section, _good_meta_title, _split_plain_text_by_chapters, clean_text
from book_rag.retrieval import rrf_fuse
from book_rag.store import fts_query


def make_doc(sections):
    return LoadedDocument(Path("x.txt"), "Title", "Author", "txt", sections)


def para(i, sentences=6):
    return " ".join(f"Sentence {k} of paragraph {i} has some extra words here." for k in range(sentences))


# ---------------------------------------------------------------- chunking
def test_chunks_respect_size_and_overlap():
    text = "\n\n".join(para(i) for i in range(30))
    cfg = ChunkingConfig(chunk_size=200, chunk_overlap=40, min_chunk_size=20)
    chunks = chunk_document(make_doc([Section(text, "Ch 1", 1)]), cfg, approx_token_counter)
    assert len(chunks) > 3
    assert all(c.n_tokens <= 200 + 20 for c in chunks[:-1])
    # consecutive chunks overlap: big paragraphs carry their trailing sentences forward
    for a, b in zip(chunks, chunks[1:]):
        assert b.char_start < a.char_end
    # every chunk is an exact slice -> stitching reconstructs the full text without duplicates
    rows = [c.__dict__ for c in chunks]
    assert stitch(rows) == text


def test_chunks_never_cross_chapters_and_track_pages():
    doc = make_doc([
        Section(para(1), "Chapter 1", 1),
        Section(para(2), "Chapter 1", 2),
        Section(para(3), "Chapter 2", 3),
    ])
    chunks = chunk_document(doc, ChunkingConfig(chunk_size=1000, chunk_overlap=50), approx_token_counter)
    assert [c.chapter for c in chunks] == ["Chapter 1", "Chapter 2"]
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)
    assert (chunks[1].page_start, chunks[1].page_end) == (3, 3)


def test_long_paragraph_is_split_on_sentences():
    sentences = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(200))
    chunks = chunk_document(make_doc([Section(sentences)]), ChunkingConfig(chunk_size=100, chunk_overlap=10),
                            approx_token_counter)
    assert len(chunks) > 5
    assert all(c.text.rstrip().endswith(".") for c in chunks)  # boundaries on sentence ends


def test_fixed_strategy():
    text = " ".join(f"w{i}" for i in range(2000))
    chunks = chunk_document(make_doc([Section(text)]), ChunkingConfig(strategy="fixed", chunk_size=100, chunk_overlap=20),
                            approx_token_counter)
    assert len(chunks) > 10


def test_semantic_strategy_with_fake_embedder():
    topic_a = " ".join(f"Cats purr softly number {i}." for i in range(20))
    topic_b = " ".join(f"Stocks fell sharply number {i}." for i in range(20))

    def embed(texts):
        return np.array([[1.0, 0.0] if "Cats" in t and "Stocks" not in t else
                         [0.0, 1.0] if "Stocks" in t and "Cats" not in t else [0.7, 0.7] for t in texts])

    cfg = ChunkingConfig(strategy="semantic", chunk_size=400, chunk_overlap=0, min_chunk_size=5,
                         semantic_breakpoint_percentile=95)
    chunks = chunk_document(make_doc([Section(topic_a + " " + topic_b)]), cfg, approx_token_counter, embed)
    assert any("Cats" in c.text and "Stocks" not in c.text for c in chunks)
    assert any("Stocks" in c.text and "Cats" not in c.text for c in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_document(make_doc([Section("a b c")]), ChunkingConfig(chunk_size=10, chunk_overlap=10))


def test_context_header():
    assert context_header("Book", "Me", "Ch 1") == "Book — Me\nCh 1\n\n"
    assert context_header("Book", None, None) == "Book\n\n"


# ---------------------------------------------------------------- loaders
def test_clean_text_fixes_hyphenation_and_spaces():
    assert clean_text("inter-\nnational   text\n\n\n\nnext") == "international text\n\nnext"


def test_plain_text_chapters_english_and_bulgarian():
    text = "Intro text\nCHAPTER 1\nFirst chapter text\nГлава 2\nВтора глава текст"
    secs = _split_plain_text_by_chapters(text)
    assert [s.chapter for s in secs] == [None, "CHAPTER 1", "Глава 2"]


def test_junk_pdf_titles_are_ignored():
    assert _good_meta_title("M3 - Edited") is None
    assert _good_meta_title("Microsoft Word - book.docx") is None
    assert _good_meta_title("Thinking, Fast and Slow") == "Thinking, Fast and Slow"


# ---------------------------------------------------------------- retrieval helpers
def test_rrf_rewards_agreement():
    scores = rrf_fuse([([1, 2, 3], 1.0), ([3, 1, 4], 1.0)], k=60)
    assert max(scores, key=scores.get) == 1
    assert scores[3] > scores[2] and scores[3] > scores[4]


def test_rrf_weights():
    scores = rrf_fuse([([1], 0.3), ([2], 0.7)], k=60)
    assert scores[2] > scores[1]


def test_fts_query_is_safe_and_unicode():
    q = fts_query('What is "RRF"? Какво е хибридно търсене (BM25)*')
    assert '"какво"' in q and '"bm25"' in q and "*" not in q
    assert fts_query("?!") == ""


def test_profiles(monkeypatch, tmp_path):
    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text("profile: english\n", encoding="utf-8")
    cfg = load_config(cfgfile)
    assert cfg.embedding.model == "BAAI/bge-base-en-v1.5"
    assert cfg.reranker.model == "BAAI/bge-reranker-base"
    cfgfile.write_text("profile: multilingual\nembedding:\n  model: custom/model\n", encoding="utf-8")
    cfg = load_config(cfgfile)
    assert cfg.embedding.model == "custom/model"  # explicit value overrides profile
    assert cfg.embedding.index_id == "custom/model|onnx-int8"


# ---------------------------------------------------------------- library hygiene
def test_title_resolution_from_messy_names():
    from book_rag.loaders import resolve_title_author

    assert resolve_title_author(Path("Eric Ries - The Lean Startup.pdf")) == ("The Lean Startup", "Eric Ries")
    t, a = resolve_title_author(Path("Deep Work (Cal Newport) (Z-Library).pdf"))
    assert (t, a) == ("Deep Work", "Cal Newport")
    # 'Title - Subtitle 2009' is not 'Author - Title'
    t, a = resolve_title_author(Path("Trading in the Zone - Master the Market 2000.pdf"), "Master the Market", "vpavan")
    assert t == "Trading in the Zone - Master the Market 2000" and a is None
    # generic file names take their title from the folder
    assert resolve_title_author(Path("lib/Atomic Habits/Summary.docx"))[0] == "Atomic Habits — Summary"
    assert resolve_title_author(Path("x/inv(3).pdf"), "file:///C:/Documents")[0] == "x — inv(3)"


def test_duplicate_sketch():
    from book_rag.ingest import sketch_similarity, text_sketch

    book = " ".join(f"word{i % 997} token{i % 991}" for i in range(20000))
    copy = book.replace("word5 ", "WORD5 ", 3)  # tiny differences (other edition / scan)
    other = " ".join(f"alpha{i % 983} beta{i % 977}" for i in range(20000))
    a, b, c = text_sketch(book), text_sketch(copy), text_sketch(other)
    assert sketch_similarity(a, b) > 0.9
    assert sketch_similarity(a, c) < 0.05
    assert len(text_sketch("too short")) == 0


def test_ragignore_patterns():
    from book_rag.ingest import is_ignored

    pats = ["Prompt*.txt", "drafts/", "*.pdf.yaml"]
    assert is_ignored("business/Prompt for Summarizing a Book.txt", pats)
    assert is_ignored("drafts/notes.md", pats)
    assert is_ignored("x/book.pdf.yaml", pats)
    assert not is_ignored("business/Atomic Habits.pdf", pats)
