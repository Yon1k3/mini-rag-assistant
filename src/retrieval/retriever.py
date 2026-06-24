from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from config import Settings, get_settings
from ingestion.metadata import metadata_to_source


RetrievedChunk = dict[str, Any]


@lru_cache(maxsize=1)
def get_embeddings(settings: Settings | None = None) -> Any:
    settings = settings or get_settings()

    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        return HuggingFaceEmbeddings(model_name=settings.local_embedding_model)
    except Exception as exc:
        if not looks_like_network_error(exc):
            raise
        print("Embedding model network check failed; retrying with cached files only.")
        return HuggingFaceEmbeddings(
            model_name=settings.local_embedding_model,
            model_kwargs={"local_files_only": True},
        )


def looks_like_network_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "winerror 10013",
            "network",
            "connection",
            "client has been closed",
            "timeout",
            "forbidden",
        )
    )


@lru_cache(maxsize=1)
def load_vectorstore(settings: Settings | None = None) -> Any:
    # Vectorstore теж кешується в межах процесу: це прискорює eval і Streamlit.
    settings = settings or get_settings()
    if settings.vector_db != "chroma":
        raise ValueError("Only Chroma is supported by this mini project.")

    from langchain_chroma import Chroma

    return Chroma(
        collection_name="mini_rag",
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embeddings(settings),
    )


def clean_metadata_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata_filter:
        return {}

    clean: dict[str, Any] = {}
    for key, value in metadata_filter.items():
        if value is None or value == "":
            continue
        if key == "page_number" and isinstance(value, str) and value.isdigit():
            clean[key] = int(value)
        else:
            clean[key] = value
    return clean


def build_chroma_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
    clean = clean_metadata_filter(metadata_filter)
    if not clean:
        return None
    if len(clean) == 1:
        return clean
    return {"$and": [{key: value} for key, value in clean.items()]}


def metadata_matches(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
    clean = clean_metadata_filter(metadata_filter)
    for key, expected in clean.items():
        actual = metadata.get(key)
        if key == "source_file":
            if str(actual).replace("\\", "/") != str(expected).replace("\\", "/"):
                return False
        elif str(actual) != str(expected):
            return False
    return True


def load_chunk_documents(
    settings: Settings | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[Document]:
    settings = settings or get_settings()
    if not settings.chunk_records_path.exists():
        return []

    documents: list[Document] = []
    with settings.chunk_records_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            if not metadata_matches(metadata, metadata_filter):
                continue
            documents.append(
                Document(page_content=record.get("text", ""), metadata=metadata)
            )
    return documents


def similarity_search(
    query: str,
    settings: Settings | None = None,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    settings = settings or get_settings()
    vectorstore = load_vectorstore(settings)
    k = top_k or settings.top_k
    chroma_filter = build_chroma_filter(metadata_filter)

    raw_results = vectorstore.similarity_search_with_score(
        query,
        k=k,
        filter=chroma_filter,
    )

    results: list[RetrievedChunk] = []
    for rank, item in enumerate(raw_results, start=1):
        document, distance = item
        score = distance_to_score(distance)
        results.append(result_from_document(document, score=score, rank=rank))
    return results


def distance_to_score(distance: float | None) -> float | None:
    if distance is None:
        return None
    return 1.0 / (1.0 + max(float(distance), 0.0))


def result_from_document(
    document: Document,
    score: float | None = None,
    rank: int | None = None,
) -> RetrievedChunk:
    metadata = dict(document.metadata)
    source = metadata_to_source(metadata)
    return {
        "text": document.page_content,
        "metadata": metadata,
        "relevance_score": score,
        "rank": rank,
        "source_file": source["source_file"],
        "source_url": source["source_url"],
        "document_source": source["document_source"],
        "document_type": source["document_type"],
        "page_number": source["page_number"],
        "section_title": source["section_title"],
        "chunk_id": source["chunk_id"],
    }


def retrieve_chunks(
    question: str,
    retrieval_mode: str = "similarity",
    settings: Settings | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    settings = settings or get_settings()
    mode = retrieval_mode.lower()

    if mode == "hybrid":
        from retrieval.hybrid_retriever import hybrid_search

        return hybrid_search(question, settings=settings, metadata_filter=metadata_filter)

    if mode == "metadata_filter":
        return similarity_search(
            question,
            settings=settings,
            top_k=settings.top_k,
            metadata_filter=metadata_filter,
        )

    if mode == "query_rewrite":
        from retrieval.query_rewriter import rewrite_query

        rewritten_query = rewrite_query(question, settings=settings)
        results = similarity_search(
            rewritten_query,
            settings=settings,
            top_k=settings.top_k,
            metadata_filter=metadata_filter,
        )
        for result in results:
            result["query_used"] = rewritten_query
        return results

    if mode == "rerank":
        from retrieval.reranker import rerank_chunks

        candidate_count = max(settings.top_k * 4, settings.rerank_top_n * 4, 10)
        candidates = similarity_search(
            question,
            settings=settings,
            top_k=candidate_count,
            metadata_filter=metadata_filter,
        )
        return rerank_chunks(question, candidates, settings=settings)

    return similarity_search(
        question,
        settings=settings,
        top_k=settings.top_k,
        metadata_filter=metadata_filter,
    )


def timed_retrieve(
    question: str,
    retrieval_mode: str = "similarity",
    settings: Settings | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> tuple[list[RetrievedChunk], float]:
    start = time.perf_counter()
    chunks = retrieve_chunks(
        question,
        retrieval_mode=retrieval_mode,
        settings=settings,
        metadata_filter=metadata_filter,
    )
    return chunks, time.perf_counter() - start


def available_filter_options(settings: Settings | None = None) -> dict[str, list[Any]]:
    documents = load_chunk_documents(settings)
    options: dict[str, set[Any]] = {
        "document_source": set(),
        "document_type": set(),
        "source_file": set(),
        "page_number": set(),
    }

    for document in documents:
        metadata = document.metadata
        for key in options:
            value = metadata.get(key)
            if value not in (None, ""):
                options[key].add(value)

    return {key: sorted(values) for key, values in options.items()}
