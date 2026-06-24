from __future__ import annotations

import math
import re
from typing import Any

from config import Settings, get_settings
from retrieval.retriever import RetrievedChunk


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def rerank_chunks(
    question: str,
    chunks: list[RetrievedChunk],
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    settings = settings or get_settings()
    if not chunks:
        return []

    provider = settings.reranker_provider.lower()
    if provider in {"auto", "cross_encoder"}:
        try:
            return cross_encoder_rerank(question, chunks, settings)
        except Exception as exc:
            print(f"Cross-encoder rerank unavailable, using lexical fallback: {exc}")

    return lexical_rerank(question, chunks, top_n=settings.rerank_top_n)


def cross_encoder_rerank(
    question: str,
    chunks: list[RetrievedChunk],
    settings: Settings,
) -> list[RetrievedChunk]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(settings.cross_encoder_model)
    pairs = [(question, str(chunk.get("text", ""))) for chunk in chunks]
    scores = model.predict(pairs)

    scored: list[RetrievedChunk] = []
    for chunk, score in zip(chunks, scores):
        updated = dict(chunk)
        updated["rerank_score"] = float(score)
        updated["relevance_score"] = float(score)
        scored.append(updated)

    scored.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
    return _with_ranks(scored[: settings.rerank_top_n])


def lexical_rerank(
    question: str,
    chunks: list[RetrievedChunk],
    top_n: int,
) -> list[RetrievedChunk]:
    query_tokens = tokenize(question)
    scored: list[RetrievedChunk] = []
    for chunk in chunks:
        updated = dict(chunk)
        score = lexical_score(query_tokens, tokenize(str(chunk.get("text", ""))))
        updated["rerank_score"] = score
        updated["relevance_score"] = score
        scored.append(updated)
    scored.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
    return _with_ranks(scored[:top_n])


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def lexical_score(query_tokens: set[str], text_tokens: set[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(text_tokens))


def _with_ranks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    for rank, chunk in enumerate(chunks, start=1):
        chunk["rank"] = rank
    return chunks
