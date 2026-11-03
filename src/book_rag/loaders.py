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


def _title_author_from_filename(path: Path) -> tuple[str, str | None]:
    """'Author Name - Book Title.pdf' -> ('Book Title', 'Author Name')."""
    stem = path.stem.replace("_", " ").strip()
    if " - " in stem:
        left, right = stem.split(" - ", 1)
        if 2 <= len(left) <= 60:
            return right.strip(), left.strip()
    return stem, None


_JUNK_TITLE = re.compile(
    r"(^untitled|^microsoft (word|powerpoint)|^document\d*$|^presentation\d*$|\.(docx?|pptx?|pdf|indd|qxd)$|"
    r"\b(edited|draft|copy|final|v\d+)\s*$|^\W*$)",
    re.IGNORECASE,
)


def _good_meta_title(title: str | None) -> str | None:
    title = (title or "").strip()
    if len(title) < 3 or _JUNK_TITLE.search(title):
        return None
    return title


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


def load_pdf(path: Path) -> LoadedDocument:
    import pymupdf

    doc = pymupdf.open(path)
    meta = doc.metadata or {}
    title, author = _title_author_from_filename(path)
    title = _good_meta_title(meta.get("title")) or title
    author = (meta.get("author") or "").strip() or author

    # Map every page to the deepest TOC entry (level <= 2) that starts on/before it.
    toc = [(lvl, t.strip(), p) for lvl, t, p in doc.get_toc(simple=True) if lvl <= 2 and t.strip()]
    page_chapter: dict[int, str] = {}
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

    sections: list[Section] = []
    empty_pages = 0
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
            empty_pages += 1
            continue
        sections.append(Section(page_text, page_chapter.get(i), i))

    warnings = []
    if doc.page_count and empty_pages / doc.page_count > 0.5:
        warnings.append(
            f"{empty_pages}/{doc.page_count} pages have no extractable text - this PDF is probably "
            "scanned. Run OCR first (e.g. `ocrmypdf in.pdf out.pdf`) and re-sync."
        )
    n_pages = doc.page_count
    doc.close()
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
    title, author = _title_author_from_filename(path)
    md_title = book.get_metadata("DC", "title")
    md_author = book.get_metadata("DC", "creator")
    md_lang = book.get_metadata("DC", "language")
    title = md_title[0][0].strip() if md_title else title
    author = md_author[0][0].strip() if md_author else author
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
    title, author = _title_author_from_filename(path)
    cp = d.core_properties
    title = _good_meta_title(cp.title) or title
    author = (cp.author or "").strip() or author
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
    title, author = _title_author_from_filename(path)
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
    title, author = _title_author_from_filename(path)
    sections = _split_plain_text_by_chapters(text)
    if len(sections) <= 1:
        sections = [Section(text, None)]
    return LoadedDocument(path, title, author, "txt", sections)


def load_html(path: Path) -> LoadedDocument:
    from bs4 import BeautifulSoup

    raw = _read_text_file(path)
    title, author = _title_author_from_filename(path)
    soup = BeautifulSoup(raw, "lxml")
    if soup.title and soup.title.string:
        title = soup.title.string.strip() or title
    return LoadedDocument(path, title, author, "html", _html_to_sections(raw))


def load_ipynb(path: Path) -> LoadedDocument:
    nb = json.loads(_read_text_file(path))
    title, author = _title_author_from_filename(path)
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


def load_document(path: Path) -> LoadedDocument:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    doc = loader(path)

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
