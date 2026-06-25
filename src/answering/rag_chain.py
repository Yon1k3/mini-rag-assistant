from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Literal, Optional

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from answering.llm import get_chat_model
from config import Settings, get_settings
from retrieval.retriever import PIPELINE_NAME, PIPELINE_STEPS, RetrievedChunk, timed_retrieve


DONT_KNOW = "I don't know based on the provided context."
DEFAULT_THREAD_ID = "mini-rag-default"

DocumentSource = Literal["FastAPI", "Pydantic", "LangChain", "Python"]
DocumentType = Literal["markdown", "html", "pdf", "txt"]
FILTER_KEYS = {
    "document_source", "document_type", "document_year",
    "document_date", "source_file", "page_number",
}
SOURCE_KEYS = (
    "source_file", "source_url", "document_source", "document_type",
    "document_year", "document_date", "page_number", "section_title", "chunk_id",
)
CHUNK_KEYS = (
    "text", "metadata", *SOURCE_KEYS,
    "relevance_score", "rank", "query_used", "metadata_filter_used",
)


class DocumentMetadata(BaseModel):
    """Metadata schema for RAG documents. Used to filter the vector database."""

    document_source: Optional[DocumentSource] = Field(
        None,
        description="Use ONLY if the user explicitly asks about FastAPI, Pydantic, LangChain, or Python.",
    )
    document_type: Optional[DocumentType] = Field(
        None,
        description="Use ONLY if the user explicitly asks for markdown, html, pdf, or txt documents.",
    )
    document_year: Optional[int] = Field(
        None,
        description="Use ONLY if the user explicitly asks for a year, for example 2022.",
    )
    document_date: Optional[str] = Field(
        None,
        description="Use ONLY if the user explicitly asks for an exact date in YYYY-MM-DD format.",
    )
    source_file: Optional[str] = Field(
        None,
        description="Use ONLY if the user explicitly mentions a source file path.",
    )
    page_number: Optional[int] = Field(
        None,
        description="Use ONLY if the user explicitly asks for a PDF page number.",
    )


AGENT_SYSTEM_PROMPT = f"""You are a helpful documentation RAG assistant.
You must use the provided tool to find information. NEVER make up information.

Workflow:
1. Read the user question.
2. Generate arguments for the `search_rag_database` tool.
3. If the user asks for a year, date, source, document type, file, or page, put it into the metadata argument.
4. Read the tool output and write the final answer.

Critical requirements:
- Always call `search_rag_database` before answering.
- Answer only from the tool output.
- If the tool output does not contain the answer, say exactly: {DONT_KNOW}
- Always cite source_file, document_source, year, page, chunk_id, and url when available.

Current retrieval pipeline: {PIPELINE_NAME}
Pipeline steps: {PIPELINE_STEPS}
"""


class RAGAgent:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.memory = MemorySaver()
        self.ui_filter: dict[str, Any] = {}
        self.last_tool_payload = make_payload()
        self.agent = create_agent(
            model=get_chat_model(self.settings),
            tools=[self._build_search_tool()],
            checkpointer=self.memory,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

    def invoke(
        self,
        query: str,
        config: dict[str, Any],
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._prepare_run(metadata_filter)
        return self.agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)

    def stream(
        self,
        query: str,
        config: dict[str, Any],
        metadata_filter: dict[str, Any] | None = None,
    ) -> None:
        self._prepare_run(metadata_filter)
        for step in self.agent.stream({"messages": [HumanMessage(content=query)]}, config=config):
            for node_name, state_update in step.items():
                print(f"\n--- [Node: {node_name}] ---")
                messages = state_update.get("messages", [])
                if messages:
                    print_stream_message(messages[-1])

    def _prepare_run(self, metadata_filter: dict[str, Any] | None) -> None:
        self.ui_filter = metadata_to_filter(metadata_filter)
        self.last_tool_payload = make_payload(failure_reason="tool_not_called")

    def _build_search_tool(self) -> Any:
        @tool
        def search_rag_database(
            query: str,
            metadata: Optional[DocumentMetadata] = None,
        ) -> str:
            """
            Search the local RAG database. For metadata, always follow the provided schema.
            """
            metadata_filter = {**metadata_to_filter(metadata), **self.ui_filter}
            print(f"\n[Tool Execution] RAG query: '{query}'")
            if metadata_filter:
                print(f"[Tool Execution] Metadata filter: {metadata_filter}")

            try:
                chunks, retrieval_latency = timed_retrieve(
                    query,
                    settings=self.settings,
                    metadata_filter=metadata_filter or None,
                )
                failure_reason = "" if chunks else "no_context"
            except Exception as exc:
                chunks, retrieval_latency = [], 0.0
                failure_reason = f"retrieval_error: {exc}"

            self.last_tool_payload = make_payload(
                chunks=chunks,
                metadata_filter=metadata_filter,
                retrieval_latency=retrieval_latency,
                failure_reason=failure_reason,
            )
            return format_tool_results(chunks, metadata_filter, failure_reason)

        return search_rag_database


def answer_question(
    question: str,
    metadata_filter: dict[str, Any] | None = None,
    settings: Settings | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    agent = get_rag_agent(settings or get_settings())
    safe_thread_id = thread_id or DEFAULT_THREAD_ID
    start = time.perf_counter()

    try:
        state = agent.invoke(
            query=question,
            metadata_filter=metadata_filter,
            config={"configurable": {"thread_id": safe_thread_id}},
        )
        answer_text = extract_answer_text(latest_message_content(state))
        payload = agent.last_tool_payload
    except Exception as exc:
        answer_text = DONT_KNOW
        payload = make_payload(failure_reason=f"agent_error: {exc}")

    return build_response(
        answer_text=answer_text,
        payload=payload,
        latency_seconds=time.perf_counter() - start,
        thread_id=safe_thread_id,
    )


@lru_cache(maxsize=1)
def get_rag_agent(settings: Settings) -> RAGAgent:
    return RAGAgent(settings=settings)


def metadata_to_filter(metadata: DocumentMetadata | dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    values = metadata.model_dump(exclude_none=True) if isinstance(metadata, BaseModel) else metadata
    return {
        key: value
        for key, value in values.items()
        if key in FILTER_KEYS and value not in (None, "")
    }


def make_payload(
    chunks: list[RetrievedChunk] | None = None,
    metadata_filter: dict[str, Any] | None = None,
    retrieval_latency: float = 0.0,
    failure_reason: str = "",
) -> dict[str, Any]:
    return {
        "retrieval_pipeline": PIPELINE_NAME,
        "pipeline_steps": PIPELINE_STEPS,
        "metadata_filter_used": metadata_filter or {},
        "retrieval_latency_seconds": retrieval_latency,
        "chunks": [compact_chunk(chunk) for chunk in chunks or []],
        "failure_reason": failure_reason,
    }


def compact_chunk(chunk: RetrievedChunk) -> RetrievedChunk:
    return {key: chunk.get(key) for key in CHUNK_KEYS}


def format_tool_results(
    chunks: list[RetrievedChunk],
    metadata_filter: dict[str, Any],
    failure_reason: str,
) -> str:
    if not chunks:
        return (
            "No relevant context found.\n"
            f"Failure reason: {failure_reason or 'no_context'}\n"
            f"Metadata filter used: {metadata_filter or {}}\n"
        )
    return "\n\n---\n\n".join(format_chunk_for_tool(chunk) for chunk in chunks)


def format_chunk_for_tool(chunk: RetrievedChunk) -> str:
    page = chunk.get("page_number")
    return "\n".join(
        [
            f"Source file: {chunk.get('source_file', '')}",
            f"Document source: {chunk.get('document_source', '')}",
            f"Document type: {chunk.get('document_type', '')}",
            f"Year: {chunk.get('document_year', '')}",
            f"Date: {chunk.get('document_date', '')}",
            f"Page: {'null' if page in (None, '') else page}",
            f"Chunk id: {chunk.get('chunk_id', '')}",
            f"URL: {chunk.get('source_url', '')}",
            "Content:",
            str(chunk.get("text", "")),
        ]
    )


def build_response(
    answer_text: str,
    payload: dict[str, Any],
    latency_seconds: float,
    thread_id: str,
) -> dict[str, Any]:
    chunks = payload.get("chunks", [])
    failure_reason = str(payload.get("failure_reason", ""))
    answer_text = normalize_answer_text(answer_text)
    if not chunks:
        answer_text = DONT_KNOW
        failure_reason = failure_reason or "no_context"

    sources = collect_sources(chunks)
    return {
        "answer": format_final_answer(answer_text, sources),
        "answer_text": answer_text,
        "sources": sources,
        "retrieved_context": chunks,
        "latency_seconds": latency_seconds,
        "retrieval_latency_seconds": float(payload.get("retrieval_latency_seconds", 0.0)),
        "retrieval_pipeline": payload.get("retrieval_pipeline", PIPELINE_NAME),
        "failure_reason": failure_reason,
        "thread_id": thread_id,
    }


def latest_message_content(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages", [])):
        content = getattr(message, "content", "")
        if content:
            return message_content_to_text(content)
    return ""


def print_stream_message(message: Any) -> None:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        print(f"Agent called: {tool_calls[0]['name']}")
    elif getattr(message, "content", None):
        print(f"Message: {message_content_to_text(message.content)}")


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("content") or item)
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    return str(content)


def normalize_answer_text(answer_text: str) -> str:
    if not answer_text or DONT_KNOW.lower() in answer_text.lower():
        return DONT_KNOW
    return answer_text


def extract_answer_text(raw_answer: str) -> str:
    answer = raw_answer.strip()
    for marker in ("\nSources:", "\nSource:", "\nCitations:", "\nCitation:"):
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()
    if answer.lower().startswith("answer:"):
        answer = answer.split(":", 1)[1].strip()
    return answer


def collect_sources(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        key = str(chunk.get("chunk_id") or chunk.get("source_file"))
        if key and key not in seen:
            seen.add(key)
            sources.append({key: chunk.get(key) for key in SOURCE_KEYS})
    return sources


def format_final_answer(answer_text: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return f"Answer:\n{DONT_KNOW}\n\nSources:\n"

    source_lines = []
    for index, source in enumerate(sources, start=1):
        page = source.get("page_number")
        source_lines.append(
            f"{index}. source_file: {source.get('source_file', '')}, "
            f"document_source: {source.get('document_source', '')}, "
            f"year: {source.get('document_year', '')}, "
            f"page: {'null' if page in (None, '') else page}, "
            f"chunk_id: {source.get('chunk_id', '')}, "
            f"url: {source.get('source_url', '')}"
        )
    return f"Answer:\n{answer_text}\n\nSources:\n\n" + "\n".join(source_lines)
