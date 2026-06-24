from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Settings, get_settings
from ingestion.chunking import chunk_documents
from ingestion.loaders import load_documents
from ingestion.metadata import sanitize_metadata
from retrieval.retriever import get_embeddings


def clean_chroma_directory(settings: Settings) -> None:
    for child in settings.chroma_dir.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def save_chunk_records(chunks: list[object], settings: Settings) -> None:
    with settings.chunk_records_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            record = {
                "text": chunk.page_content,
                "metadata": sanitize_metadata(dict(chunk.metadata)),
            }
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_index(settings: Settings | None = None) -> int:
    settings = settings or get_settings()

    documents = load_documents(settings)
    if not documents:
        raise RuntimeError(
            "No documents found in data/raw. Add prepared documents before indexing."
        )

    chunks = chunk_documents(documents, settings)
    if not chunks:
        raise RuntimeError("No chunks were produced from the loaded documents.")

    save_chunk_records(chunks, settings)
    clean_chroma_directory(settings)

    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    chroma_documents = [
        Document(
            page_content=chunk.page_content,
            metadata=sanitize_metadata(dict(chunk.metadata)),
        )
        for chunk in chunks
    ]

    vectorstore = Chroma.from_documents(
        documents=chroma_documents,
        embedding=get_embeddings(settings),
        collection_name="mini_rag",
        persist_directory=str(settings.chroma_dir),
    )

    if hasattr(vectorstore, "persist"):
        vectorstore.persist()

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Saved chunk records: {settings.chunk_records_path}")
    print(f"Persisted Chroma index: {settings.chroma_dir}")
    return len(chunks)


def main() -> int:
    build_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
