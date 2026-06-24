from __future__ import annotations

import time
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from answering.llm import get_chat_model
from answering.prompts import RAG_HUMAN_PROMPT, SYSTEM_PROMPT
from config import Settings, get_settings
from retrieval.retriever import RetrievedChunk, timed_retrieve


DONT_KNOW = "I don't know based on the provided context."


def answer_question(
    question: str,
    retrieval_mode: str = "similarity",
    metadata_filter: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    start = time.perf_counter()
    failure_reason = ""

    try:
        retrieved_chunks, retrieval_latency = timed_retrieve(
            question,
            retrieval_mode=retrieval_mode,
            settings=settings,
            metadata_filter=metadata_filter,
        )
    except Exception as exc:
        retrieved_chunks = []
        retrieval_latency = 0.0
        failure_reason = f"retrieval_error: {exc}"

    if not retrieved_chunks:
        latency = time.perf_counter() - start
        return {
            "answer": format_final_answer(DONT_KNOW, []),
            "answer_text": DONT_KNOW,
            "sources": [],
            "retrieved_context": [],
            "latency_seconds": latency,
            "retrieval_latency_seconds": retrieval_latency,
            "retrieval_mode": retrieval_mode,
            "failure_reason": failure_reason or "no_context",
        }

    context = build_context(retrieved_chunks)
    sources = collect_sources(retrieved_chunks)

    try:
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", RAG_HUMAN_PROMPT)]
        )
        chain = prompt | get_chat_model(settings) | StrOutputParser()
        raw_answer = chain.invoke({"question": question, "context": context})
        answer_text = extract_answer_text(raw_answer)
    except Exception as exc:
        answer_text = DONT_KNOW
        failure_reason = failure_reason or f"llm_error: {exc}"

    if not answer_text or DONT_KNOW.lower() in answer_text.lower():
        answer_text = DONT_KNOW

    latency = time.perf_counter() - start
    return {
        "answer": format_final_answer(answer_text, sources),
        "answer_text": answer_text,
        "sources": sources,
        "retrieved_context": retrieved_chunks,
        "latency_seconds": latency,
        "retrieval_latency_seconds": retrieval_latency,
        "retrieval_mode": retrieval_mode,
        "failure_reason": failure_reason,
    }


def build_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.get("page_number")
        page_display = "null" if page in (None, "") else str(page)
        blocks.append(
            "\n".join(
                [
                    f"[chunk {index}]",
                    f"source_file: {chunk.get('source_file', '')}",
                    f"document_source: {chunk.get('document_source', '')}",
                    f"document_type: {chunk.get('document_type', '')}",
                    f"page: {page_display}",
                    f"chunk_id: {chunk.get('chunk_id', '')}",
                    f"url: {chunk.get('source_url', '')}",
                    "text:",
                    str(chunk.get("text", "")),
                ]
            )
        )
    return "\n\n".join(blocks)


def collect_sources(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        key = str(chunk.get("chunk_id") or chunk.get("source_file"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source_file": chunk.get("source_file", ""),
                "source_url": chunk.get("source_url", ""),
                "document_source": chunk.get("document_source", ""),
                "document_type": chunk.get("document_type", ""),
                "page_number": chunk.get("page_number"),
                "section_title": chunk.get("section_title", ""),
                "chunk_id": chunk.get("chunk_id", ""),
            }
        )
    return sources


def extract_answer_text(raw_answer: str) -> str:
    answer = raw_answer.strip()
    if "Sources:" in answer:
        answer = answer.split("Sources:", 1)[0].strip()
    if answer.lower().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    return answer


def format_final_answer(answer_text: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return f"Answer:\n{DONT_KNOW}\n\nSources:\n"

    source_lines = []
    for index, source in enumerate(sources, start=1):
        page = source.get("page_number")
        page_display = "null" if page in (None, "") else str(page)
        source_lines.append(
            f"{index}. source_file: {source.get('source_file', '')}, "
            f"document_source: {source.get('document_source', '')}, "
            f"page: {page_display}, "
            f"chunk_id: {source.get('chunk_id', '')}, "
            f"url: {source.get('source_url', '')}"
        )

    return f"Answer:\n{answer_text}\n\nSources:\n\n" + "\n".join(source_lines)
