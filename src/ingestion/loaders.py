from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from config import Settings, get_settings
from ingestion.metadata import load_source_metadata, metadata_for_file


MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _load_text_documents(path: Path) -> list[Document]:
    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()


def _load_html_documents(path: Path) -> list[Document]:
    from langchain_community.document_loaders import BSHTMLLoader

    return BSHTMLLoader(
        str(path),
        open_encoding="utf-8",
        bs_kwargs={"features": "html.parser"},
    ).load()


def _load_pdf_documents(path: Path) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(str(path)).load()


def iter_source_files(settings: Settings) -> Iterable[Path]:
    patterns = {
        "markdown": ("*.md", "*.markdown"),
        "html": ("*.html", "*.htm"),
        "pdf": ("*.pdf",),
        "txt": ("*.txt",),
    }
    for document_type, globs in patterns.items():
        directory = settings.raw_dir / document_type
        for pattern in globs:
            yield from sorted(directory.glob(pattern))


def extract_section_title(path: Path, text: str, document_type: str) -> str:
    if not text:
        return path.stem

    if document_type == "markdown":
        match = MARKDOWN_HEADING_RE.search(text)
        if match:
            return _clean_title(match.group(1))

    if document_type == "html":
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
            title = soup.find(["h1", "title", "h2"])
            if title and title.get_text(strip=True):
                return _clean_title(title.get_text(" ", strip=True))
        except Exception:
            text = re.sub(r"<[^>]+>", " ", text)

    for line in text.splitlines():
        candidate = line.strip().strip("=-`#* ")
        if 3 <= len(candidate) <= 140:
            return _clean_title(candidate)

    return path.stem


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _read_text_for_title(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _load_file(path: Path, document_type: str) -> list[Document]:
    if document_type in {"markdown", "txt"}:
        return _load_text_documents(path)
    if document_type == "html":
        return _load_html_documents(path)
    if document_type == "pdf":
        return _load_pdf_documents(path)
    return []


def load_documents(settings: Settings | None = None) -> list[Document]:
    settings = settings or get_settings()
    catalog = load_source_metadata(settings)
    documents: list[Document] = []

    for path in iter_source_files(settings):
        base_metadata = metadata_for_file(path, catalog)
        document_type = str(base_metadata.get("document_type", "unknown"))
        title_text = _read_text_for_title(path) if document_type != "pdf" else path.stem
        section_title = extract_section_title(path, title_text, document_type)

        try:
            loaded_documents = _load_file(path, document_type)
        except Exception as exc:
            print(f"SKIP unreadable {path}: {exc}")
            continue

        non_empty_documents = [
            document
            for document in loaded_documents
            if document.page_content and document.page_content.strip()
        ]
        if not non_empty_documents:
            print(f"SKIP empty {path}")
            continue

        for document in non_empty_documents:
            metadata = dict(document.metadata)
            metadata.update(base_metadata)
            metadata["section_title"] = section_title

            if document_type == "pdf":
                raw_page = metadata.get("page")
                if isinstance(raw_page, int):
                    metadata["page_number"] = raw_page + 1

            document.metadata = metadata
            documents.append(document)

    return documents
