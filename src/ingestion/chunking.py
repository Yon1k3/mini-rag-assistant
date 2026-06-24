from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings, get_settings


CHUNK_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def create_text_splitter(settings: Settings | None = None) -> RecursiveCharacterTextSplitter:
    settings = settings or get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(
    documents: list[Document], settings: Settings | None = None
) -> list[Document]:
    splitter = create_text_splitter(settings)
    chunks = splitter.split_documents(documents)
    ingestion_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for index, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.metadata)
        metadata["chunk_index"] = index
        metadata["chunk_id"] = build_chunk_id(chunk, index)
        metadata["ingestion_timestamp"] = ingestion_timestamp

        chunk_title = extract_chunk_heading(chunk.page_content)
        if chunk_title:
            metadata["section_title"] = chunk_title

        chunk.metadata = metadata

    return chunks


def build_chunk_id(chunk: Document, index: int) -> str:
    source_file = str(chunk.metadata.get("source_file", "unknown"))
    page_number = str(chunk.metadata.get("page_number", ""))
    digest_input = f"{source_file}|{page_number}|{index}|{chunk.page_content[:200]}"
    digest = hashlib.sha1(digest_input.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"chunk-{index:06d}-{digest}"


def extract_chunk_heading(text: str) -> str:
    match = CHUNK_HEADING_RE.search(text)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()
