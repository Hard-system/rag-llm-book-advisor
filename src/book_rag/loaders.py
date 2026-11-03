"""Document loaders.

Every loader turns a file into a :class:`LoadedDocument` = metadata + an ordered
list of :class:`Section` objects. A section is a piece of text that belongs to
one chapter/heading and (for paginated formats) one page. Keeping chapter and
page information this early lets every chunk carry precise citations
("Atomic Habits, Chapter 3, p. 45") later on.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass
class Section:
    text: str
    chapter: str | None = None
    page: int | None = None  # 1-based page number when the format has pages


@dataclass
class LoadedDocument:
    path: Path
    title: str
    author: str | None
    file_type: str
    sections: list[Section]
    n_pages: int | None = None
    language: str | None = None
    tags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s*)?\d{1,4}\s*$", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalize unicode, fix hyphenated line breaks and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    text = text.replace("­", "")  # soft hyphen
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Title / author resolution
# ---------------------------------------------------------------------------

# Download-site noise in file names and metadata.
_NAME_NOISE = [
    re.compile(r"\((?=[^()]*\b(z-library|z-lib|1lib|pdfdrive|libgen|ebook|epub)\b)[^()]*\)", re.I),
    re.compile(r"^pdfcoffee\.com[_-]", re.I),
    re.compile(r"[-_ ](pdf[-_ ]room[-_ ])?pdf[-_ ]free$", re.I),
    re.compile(r"\s*-\s*pdfdrive\.com\s*$", re.I),
    re.compile(r"\s*\(\d\)$"),              # "book (1).pdf" copy marker
    re.compile(r"[_ ]\d{6}[_ ]\d{6}$"),      # "_241229_105032" timestamps
    re.compile(r"\s*-\s*annotated$", re.I),
]
# File names that say nothing about the content -> title comes from the folder.
GENERIC_STEM = re.compile(
    r"^(summary|sumarry|summery|резюме|обобщение|notes?|бележки|new (microsoft )?(word|text) document( \(\d+\))?|"
    r"document\d*|untitled.*|inv\(\d+\)|book\d*|\d+)$",
    re.IGNORECASE,
)
_JUNK_TITLE = re.compile(
    r"(^untitled|^microsoft (word|powerpoint)|^document\d*$|^presentation\d*$|\.(docx?|pptx?|pdf|indd|qxd)$|"
    r"\b(edited|draft|copy|final|v\d+)\s*$|^\W*$|^file:|^inv\(\d+\)$|^cover$|^title$|^contents$|^[\w.-]{1,3}$)",
    re.IGNORECASE,
)
_NAME_CONNECTORS = {"de", "van", "von", "der", "da", "di", "la", "le", "bin", "al", "&", "and"}
_NOT_A_NAME_START = {"the", "a", "an", "how", "why", "what", "when", "where", "who", "my", "your", "our"}


def _strip_noise(s: str) -> str:
    s = s.replace("\\'", "'").strip()
    for rx in _NAME_NOISE:
        s = rx.sub("", s).strip()
    return re.sub(r"\s{2,}", " ", s).strip(" -_.")


def _looks_like_person(s: str) -> bool:
    """'Eric Ries', 'Seth A. Klarman', 'Max Finger, Oliver Samwer' -> True; 'Trading in the Zone' -> False."""
    s = re.sub(r"\(auth\.?\)|\(eds?\.?\)", "", s, flags=re.I).strip(" ,")
    if not s or len(s) > 60 or re.search(r"\d", s):
        return False
    words = re.findall(r"[\w.'’-]+", s)
    if not 1 < len(words) <= 7 or words[0].lower() in _NOT_A_NAME_START:
        return False
    return all(w[0].isupper() or w.lower() in _NAME_CONNECTORS for w in words)


def _clean_person(s: str | None) -> str | None:
    if not s:
        return None
    s = re.sub(r"\(auth\.?\)|\(eds?\.?\)", "", s, flags=re.I).strip(" ,;")
    s = re.sub(r"\s{2,}", " ", s)
    # uploader names / tools, not authors: 'vpavan', 'Julie', 'Administrator', 'Adobe ...'
    if (len(s) < 4 or " " not in s.replace(",", " ").strip()
            or re.search(r"admin|user|adobe|microsoft|calibre|pdf|www\.|@", s, re.I)):
        return None
    return s


def _parse_filename(path: Path) -> tuple[str, str | None]:
    """Best-effort (title, author) from a file name.

    Handles 'Author - Title', 'Title (Author) (Z-Library)', 'Title - Subtitle 2009', slugs."""
    raw = path.stem
    had_zlib = bool(re.search(r"\([^()]*\b(z-library|z-lib|1lib)\b[^()]*\)", raw, re.I))
    stem = _strip_noise(_strip_noise(raw).replace("_", " "))
    if " " not in stem and stem.count("-") >= 2:  # url slug
        stem = stem.replace("-", " ").strip().capitalize()
    author = None
    # 'Title (Author Name) (Z-Library)' -> trailing parenthesis is the author
    m = re.match(r"^(.*\S)\s*\(([^()]+)\)$", stem)
    if m and (had_zlib or _looks_like_person(m.group(2))):
        cand = m.group(2).split("[")[0].strip(" ,")
        if _looks_like_person(cand) or had_zlib:
            stem, author = m.group(1).strip(), cand
    # '[Author] Title'
    m = re.match(r"^\[([^\]]+)\]\s*(.+)$", stem)
    if m and not author:
        author, stem = m.group(1).strip(), m.group(2).strip()
    # 'Author - Title' (but not 'Title - Subtitle 2009')
    if " - " in stem and not author:
        left, right = stem.split(" - ", 1)
        if _looks_like_person(left) and not re.search(r"\b(19|20)\d\d\b|edition", right, re.I):
            stem, author = right.strip(), _clean_person(left) or left.strip()
    return stem or path.stem, author


def _good_meta_title(title: str | None) -> str | None:
    title = _strip_noise(title or "")
    if len(title) < 3 or _JUNK_TITLE.search(title):
        return None
    return title


def resolve_title_author(path: Path, meta_title: str | None = None,
                         meta_author: str | None = None) -> tuple[str, str | None]:
    """Combine embedded metadata and the file name into the best (title, author)."""
    fn_title, fn_author = _parse_filename(path)
    mt = _good_meta_title(meta_title)
    if GENERIC_STEM.match(path.stem.strip()):
        # 'Summary.docx' -> 'Deep work rules for focused success — Summary'
        title = mt or f"{_strip_noise(path.parent.name)} — {path.stem.strip()}"
    elif mt and mt.lower() in fn_title.lower() and len(fn_title) > len(mt) + 5:
        title = fn_title  # file name has the full title, metadata only the subtitle
    else:
        title = mt or fn_title
    author = _clean_person(meta_author) or fn_author
    return title.strip(), author


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1251", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# Plain-text chapter detection (English + Bulgarian/Russian headings).
_CHAPTER_LINE = re.compile(
    r"^\s*((chapter|part|book|section|глава|част|раздел|книга)\s+([0-9]+|[ivxlcdm]+|[a-zа-я]+)\b.*|"
    r"(prologue|epilogue|introduction|preface|conclusion|пролог|епилог|увод|предговор|заключение))\s*$",
    re.IGNORECASE,
)


def _split_plain_text_by_chapters(text: str) -> list[Section]:
    sections: list[Section] = []
    current_title: str | None = None
    buf: list[str] = []
    for line in text.split("\n"):
        if len(line) < 80 and _CHAPTER_LINE.match(line):
            if buf and "".join(buf).strip():
                sections.append(Section("\n".join(buf).strip(), current_title))
            current_title, buf = line.strip(), []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        sections.append(Section("\n".join(buf).strip(), current_title))
    return sections


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------


def _detect_pdf_headings(doc) -> dict[int, str]:
    """Chapter detection for PDFs without an outline: lines set in a clearly larger font than the
    body text, near the top of a page, are treated as chapter headings. Returns {page: heading}."""
    from collections import Counter

    sizes: Counter = Counter()
    candidates: dict[int, list[tuple[float, str]]] = {}
    for pno, page in enumerate(doc, start=1):
        height = page.rect.height or 1
        try:
            d = page.get_text("dict")
        except Exception:
            continue
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = " ".join(s["text"].strip() for s in spans)
                # font size that most characters of the line use (ignores decorative drop-caps)
                per_size: Counter = Counter()
                for s in spans:
                    per_size[round(s["size"] * 2) / 2] += len(s["text"].strip())
                size = per_size.most_common(1)[0][0]
                sizes[size] += len(text)
                if line["bbox"][1] < height * 0.45 and _plausible_heading(text):
                    candidates.setdefault(pno, []).append((size, text))
    if not sizes:
        return {}
    body = sizes.most_common(1)[0][0]
    for factor in (1.3, 1.6):
        found: dict[int, str] = {}
        for pno, cands in candidates.items():
            big = [t for s, t in cands if s >= body * factor]
            if big:
                found[pno] = re.sub(r"\s+", " ", " ".join(big[:3]))[:120]
        # plausible structure: at least 3 headings, not more than one every ~7 pages
        if 3 <= len(found) <= max(3, doc.page_count * 0.15):
            return found
    return {}  # too few (only a cover title) or too noisy (slides, charts) -> no chapter info


def _plausible_heading(text: str) -> bool:
    t = text.strip()
    if not 3 <= len(t) <= 100 or len(t.split()) > 14:
        return False
    if re.search(r"www\.|http|\bfig(ure)?\b|©|\.com\b", t, re.I):
        return False
    letters = sum(c.isalpha() for c in t)
    if letters < 4 or letters / len(t.replace(" ", "")) < 0.75 or not t.isprintable():
        return False
    words = re.findall(r"[^\W\d_]+", t)
    return bool(words) and max(len(w) for w in words) >= 4 and sum(len(w) for w in words) / len(words) >= 3


def load_pdf(path: Path, ocr_cache_dir: Path | None = None) -> LoadedDocument:
    import pymupdf

    doc = pymupdf.open(path)
    meta = doc.metadata or {}
    title, author = resolve_title_author(path, meta.get("title"), meta.get("author"))

    # Map every page to the deepest TOC entry (level <= 2) that starts on/before it.
    toc = [(lvl, t.strip(), p) for lvl, t, p in doc.get_toc(simple=True) if lvl <= 2 and t.strip()]
    page_chapter: dict[int, str] = {}
    if not toc:
        # no outline: detect headings by font size and carry them forward page by page
        current = None
        headings = _detect_pdf_headings(doc)
        for pno in range(1, doc.page_count + 1):
            current = headings.get(pno, current)
            if current:
                page_chapter[pno] = current
    if toc:
        toc.sort(key=lambda x: x[2])
        stack: dict[int, str] = {}
        idx = 0
        for pno in range(1, doc.page_count + 1):
            while idx < len(toc) and toc[idx][2] <= pno:
                lvl, t, _ = toc[idx]
                stack[lvl] = t
                for deeper in [k for k in stack if k > lvl]:
                    del stack[deeper]
                idx += 1
            if stack:
                page_chapter[pno] = " > ".join(stack[k] for k in sorted(stack))

    page_texts: dict[int, str] = {}
    empty: list[int] = []
    for i, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks", sort=True)
        paras = []
        for b in blocks:
            if b[6] != 0:  # image block
                continue
            txt = b[4].strip()
            if not txt or _PAGE_NUMBER_LINE.match(txt):
                continue
            # Inside a block, line breaks are just layout -> join with spaces.
            txt = _HYPHEN_BREAK.sub(r"\1\2", txt)
            paras.append(re.sub(r"\s*\n\s*", " ", txt))
        page_text = clean_text("\n\n".join(paras))
        if len(page_text) < 20:
            empty.append(i)
        else:
            page_texts[i] = page_text
    n_pages = doc.page_count
    doc.close()

    warnings = []
    scanned = n_pages and len(empty) / n_pages > 0.5
    if scanned:
        from . import ocr

        if ocr_cache_dir is not None and ocr.available():
            for pno, txt in ocr.ocr_pdf_pages(path, empty, ocr_cache_dir).items():
                txt = clean_text(txt)
                if len(txt) >= 20:
                    page_texts[pno] = txt
            warnings.append(f"scanned PDF - text recognised with OCR on {len(empty)} pages")
        else:
            warnings.append(
                f"{len(empty)}/{n_pages} pages have no extractable text - this PDF is probably scanned. "
                "Enable OCR (pip install rapidocr_onnxruntime, ocr: true) or run `ocrmypdf` and re-sync."
            )
    sections = [Section(page_texts[p], page_chapter.get(p), p) for p in sorted(page_texts)]
    return LoadedDocument(path, title, author, "pdf", sections, n_pages=n_pages, warnings=warnings)


def _html_to_sections(html: str | bytes, default_heading: str | None = None) -> list[Section]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    body = soup.body or soup
    sections: list[Section] = []
    heading = default_heading
    buf: list[str] = []
    for el in body.find_all(["h1", "h2", "h3", "p", "li", "blockquote", "pre", "td", "div"]):
        # Skip containers that have block-level children (their text is emitted by the children).
        if el.name == "div" and el.find(["p", "div", "h1", "h2", "h3", "li", "blockquote", "pre"]):
            continue
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ("h1", "h2", "h3"):
            if buf:
                sections.append(Section(clean_text("\n\n".join(buf)), heading))
                buf = []
            heading = text[:200]
        else:
            buf.append(text)
    if buf:
        sections.append(Section(clean_text("\n\n".join(buf)), heading))
    return [s for s in sections if s.text]


def load_epub(path: Path) -> LoadedDocument:
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path), options={"ignore_ncx": False})
    md_title = book.get_metadata("DC", "title")
    md_author = book.get_metadata("DC", "creator")
    md_lang = book.get_metadata("DC", "language")
    title, author = resolve_title_author(path, md_title[0][0] if md_title else None,
                                         md_author[0][0] if md_author else None)
    language = md_lang[0][0] if md_lang else None

    # Map document file names to TOC titles.
    toc_titles: dict[str, str] = {}

    def walk(items):
        for it in items:
            if isinstance(it, tuple):
                section, children = it
                if getattr(section, "href", None):
                    toc_titles.setdefault(section.href.split("#")[0], section.title)
                walk(children)
            elif getattr(it, "href", None):
                toc_titles.setdefault(it.href.split("#")[0], it.title)

    walk(book.toc)

    sections: list[Section] = []
    for item_id, _linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        default_heading = toc_titles.get(item.get_name())
        sections.extend(_html_to_sections(item.get_content(), default_heading))
    return LoadedDocument(path, title, author, "epub", sections, language=language)


def load_docx(path: Path) -> LoadedDocument:
    import docx

    d = docx.Document(str(path))
    cp = d.core_properties
    title, author = resolve_title_author(path, cp.title, cp.author)
    sections: list[Section] = []
    heading, buf = None, []
    for p in d.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style = (p.style.name or "").lower() if p.style is not None else ""
        if style.startswith("heading") or style == "title":
            if buf:
                sections.append(Section(clean_text("\n\n".join(buf)), heading))
                buf = []
            heading = text[:200]
        else:
            buf.append(text)
    for table in d.tables:
        rows = [" | ".join(c.text.strip() for c in row.cells) for row in table.rows]
        if rows:
            buf.append("\n".join(rows))
    if buf:
        sections.append(Section(clean_text("\n\n".join(buf)), heading))
    return LoadedDocument(path, title, author, "docx", sections)


def load_markdown(path: Path) -> LoadedDocument:
    text = _read_text_file(path)
    title, author = resolve_title_author(path)
    sections: list[Section] = []
    heading, buf = None, []
    in_code = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        m = re.match(r"^(#{1,3})\s+(.*)$", line) if not in_code else None
        if m:
            if "\n".join(buf).strip():
                sections.append(Section(clean_text("\n".join(buf)), heading))
            heading, buf = m.group(2).strip()[:200], []
        else:
            buf.append(line)
    if "\n".join(buf).strip():
        sections.append(Section(clean_text("\n".join(buf)), heading))
    return LoadedDocument(path, title, author, "md", sections)


def load_txt(path: Path) -> LoadedDocument:
    text = clean_text(_read_text_file(path))
    title, author = resolve_title_author(path)
    sections = _split_plain_text_by_chapters(text)
    if len(sections) <= 1:
        sections = [Section(text, None)]
    return LoadedDocument(path, title, author, "txt", sections)


def load_html(path: Path) -> LoadedDocument:
    from bs4 import BeautifulSoup

    raw = _read_text_file(path)
    soup = BeautifulSoup(raw, "lxml")
    title, author = resolve_title_author(path, soup.title.string if soup.title and soup.title.string else None)
    return LoadedDocument(path, title, author, "html", _html_to_sections(raw))


def load_ipynb(path: Path) -> LoadedDocument:
    nb = json.loads(_read_text_file(path))
    title, author = resolve_title_author(path)
    sections: list[Section] = []
    heading, buf = None, []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if cell.get("cell_type") == "markdown":
            m = re.match(r"^\s*#{1,3}\s+(.*)", src)
            if m:
                if buf:
                    sections.append(Section(clean_text("\n\n".join(buf)), heading))
                    buf = []
                heading = m.group(1).strip()[:200]
            buf.append(src)
        elif cell.get("cell_type") == "code":
            buf.append("```python\n" + src + "\n```")
    if buf:
        sections.append(Section(clean_text("\n\n".join(buf)), heading))
    return LoadedDocument(path, title, author, "ipynb", sections)


LOADERS = {
    ".pdf": load_pdf,
    ".epub": load_epub,
    ".docx": load_docx,
    ".txt": load_txt,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".ipynb": load_ipynb,
}


def sidecar_path(path: Path) -> Path:
    """Optional metadata override file: 'My Book.pdf' -> 'My Book.pdf.yaml'."""
    return path.with_name(path.name + ".yaml")


def load_document(path: Path, ocr_cache_dir: Path | None = None) -> LoadedDocument:
    """Load any supported file. ``ocr_cache_dir`` enables OCR for scanned PDFs."""
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    doc = loader(path, ocr_cache_dir) if loader is load_pdf else loader(path)

    side = sidecar_path(path)
    if side.exists():
        try:
            over = yaml.safe_load(side.read_text(encoding="utf-8")) or {}
            doc.title = over.get("title", doc.title)
            doc.author = over.get("author", doc.author)
            doc.language = over.get("language", doc.language)
            tags = over.get("tags") or []
            doc.tags = [tags] if isinstance(tags, str) else list(tags)
        except Exception as exc:  # pragma: no cover - user file
            doc.warnings.append(f"Could not read sidecar {side.name}: {exc}")

    if doc.char_count < 50:
        doc.warnings.append("Document contains almost no extractable text.")
    return doc
