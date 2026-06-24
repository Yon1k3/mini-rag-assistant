from __future__ import annotations

from typing import Any

from config import Settings, get_settings
from retrieval.retriever import (
    RetrievedChunk,
    load_chunk_documents,
    result_from_document,
    similarity_search,
)


def hybrid_search(
    query: str,
    settings: Settings | None = None,
    metadata_filter: dict[str, Any] | None = None,
    top_k: int | None = None,
    dense_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> list[RetrievedChunk]:
    settings = settings or get_settings()
    final_k = top_k or settings.top_k
    candidate_k = max(final_k * 4, 10)

    # Hybrid retrieval бере і semantic dense search, і keyword BM25 search.
    dense_results = similarity_search(
        query,
        settings=settings,
        top_k=candidate_k,
        metadata_filter=metadata_filter,
    )
    keyword_results = bm25_search(
        query,
        settings=settings,
        metadata_filter=metadata_filter,
        top_k=candidate_k,
    )

    fused = reciprocal_rank_fusion(
        dense_results,
        keyword_results,
        dense_weight=dense_weight,
        keyword_weight=keyword_weight,
    )
    return fused[:final_k]


def bm25_search(
    query: str,
    settings: Settings,
    metadata_filter: dict[str, Any] | None = None,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    from langchain_community.retrievers import BM25Retriever

    documents = load_chunk_documents(settings, metadata_filter=metadata_filter)
    if not documents:
        return []

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = top_k

    try:
        raw_documents = retriever.invoke(query)
    except AttributeError:
        raw_documents = retriever.get_relevant_documents(query)

    return [
        result_from_document(document, score=None, rank=index)
        for index, document in enumerate(raw_documents, start=1)
    ]


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    keyword_results: list[RetrievedChunk],
    dense_weight: float,
    keyword_weight: float,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}

    def add_results(results: list[RetrievedChunk], weight: float) -> None:
        for rank, result in enumerate(results, start=1):
            key = str(result.get("chunk_id") or result.get("source_file") or rank)
            if key not in merged:
                merged[key] = dict(result)
                merged[key]["fused_score"] = 0.0
            merged[key]["fused_score"] += weight / (rrf_k + rank)

    add_results(dense_results, dense_weight)
    add_results(keyword_results, keyword_weight)

    fused = sorted(
        merged.values(),
        key=lambda item: float(item.get("fused_score", 0.0)),
        reverse=True,
    )
    for rank, result in enumerate(fused, start=1):
        result["rank"] = rank
        result["relevance_score"] = result.get("fused_score")
    return fused
