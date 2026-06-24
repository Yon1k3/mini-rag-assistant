from __future__ import annotations

import re
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DONT_KNOW = "I don't know based on the provided context."


def source_recall_at_k(
    retrieved_chunks: list[dict[str, Any]],
    expected_sources: list[dict[str, Any]],
    k: int,
) -> bool:
    if not expected_sources:
        return False

    top_chunks = retrieved_chunks[:k]
    for expected in expected_sources:
        for chunk in top_chunks:
            if source_matches(chunk, expected):
                return True
    return False


def source_matches(chunk: dict[str, Any], expected: dict[str, Any]) -> bool:
    expected_file = str(expected.get("source_file") or "").replace("\\", "/")
    actual_file = str(chunk.get("source_file") or "").replace("\\", "/")

    file_matches = not expected_file or (
        actual_file == expected_file
        or Path(actual_file).name == Path(expected_file).name
    )
    source_matches_value = not expected.get("document_source") or (
        chunk.get("document_source") == expected.get("document_source")
    )
    expected_page = expected.get("page_number")
    page_matches = expected_page in (None, "") or (
        chunk.get("page_number") == expected_page
    )
    return file_matches and source_matches_value and page_matches


def answer_keyword_score(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
    return hits / len(expected_keywords)


def groundedness_score(answer: str, retrieved_chunks: list[dict[str, Any]]) -> float:
    if DONT_KNOW.lower() in answer.lower():
        return 0.0
    context = " ".join(str(chunk.get("text", "")) for chunk in retrieved_chunks)
    answer_tokens = meaningful_tokens(answer)
    context_tokens = meaningful_tokens(context)
    if not answer_tokens or not context_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens)
    return min(1.0, overlap / max(1, len(answer_tokens)))


def meaningful_tokens(text: str) -> set[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "with",
        "is",
        "are",
        "it",
        "this",
        "that",
        "on",
        "by",
        "as",
    }
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 2 and token.lower() not in stopwords
    }

