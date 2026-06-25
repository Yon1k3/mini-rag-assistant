from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from answering.llm import get_chat_model
from answering.prompts import RAG_HUMAN_PROMPT, SYSTEM_PROMPT
from config import Settings, get_settings
from retrieval.retriever import PIPELINE_NAME, RetrievedChunk, timed_retrieve


DONT_KNOW = "I don't know based on the provided context."
DEFAULT_THREAD_ID = "mini-rag-default"


class RagState(TypedDict, total=False):
    question: str
    metadata_filter: dict[str, Any] | None
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_context: list[RetrievedChunk]
    retrieval_latency_seconds: float
    context: str
    sources: list[dict[str, Any]]
    answer_text: str
    answer: str
    start_time: float
    latency_seconds: float
    retrieval_pipeline: str
    failure_reason: str


def answer_question(
    question: str,
    metadata_filter: dict[str, Any] | None = None,
    settings: Settings | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    graph = get_rag_graph(settings)
    safe_thread_id = thread_id or DEFAULT_THREAD_ID

    state = graph.invoke(
        {
            "question": question,
            "metadata_filter": metadata_filter,
            "messages": [HumanMessage(content=question)],
        },
        config={"configurable": {"thread_id": safe_thread_id}},
    )

    return {
        "answer": state.get("answer", format_final_answer(DONT_KNOW, [])),
        "answer_text": state.get("answer_text", DONT_KNOW),
        "sources": state.get("sources", []),
        "retrieved_context": state.get("retrieved_context", []),
        "latency_seconds": state.get("latency_seconds", 0.0),
        "retrieval_latency_seconds": state.get("retrieval_latency_seconds", 0.0),
        "retrieval_pipeline": state.get("retrieval_pipeline", PIPELINE_NAME),
        "failure_reason": state.get("failure_reason", ""),
        "thread_id": safe_thread_id,
    }


@lru_cache(maxsize=4)
def get_rag_graph(settings: Settings) -> Any:
    workflow = StateGraph(RagState)

    def initialize_state(state: RagState) -> dict[str, Any]:
        return {
            "start_time": time.perf_counter(),
            "failure_reason": "",
            "retrieval_pipeline": PIPELINE_NAME,
        }

    def retrieve_context(state: RagState) -> dict[str, Any]:
        try:
            chunks, retrieval_latency = timed_retrieve(
                state["question"],
                settings=settings,
                metadata_filter=state.get("metadata_filter"),
            )
            return {
                "retrieved_context": chunks,
                "retrieval_latency_seconds": retrieval_latency,
            }
        except Exception as exc:
            return {
                "retrieved_context": [],
                "retrieval_latency_seconds": 0.0,
                "failure_reason": f"retrieval_error: {exc}",
            }

    def route_after_retrieval(state: RagState) -> str:
        return "prepare_context" if state.get("retrieved_context") else "finalize_response"

    def prepare_context(state: RagState) -> dict[str, Any]:
        chunks = state.get("retrieved_context", [])
        return {
            "context": build_context(chunks),
            "sources": collect_sources(chunks),
        }

    def generate_answer(state: RagState) -> dict[str, Any]:
        try:
            prompt = ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("human", RAG_HUMAN_PROMPT)]
            )
            chain = prompt | get_chat_model(settings) | StrOutputParser()
            raw_answer = chain.invoke(
                {
                    "question": state["question"],
                    "context": state.get("context", ""),
                }
            )
            answer_text = normalize_answer_text(extract_answer_text(raw_answer))
            return {
                "answer_text": answer_text,
                "messages": [AIMessage(content=answer_text)],
            }
        except Exception as exc:
            return {
                "answer_text": DONT_KNOW,
                "failure_reason": state.get("failure_reason") or f"llm_error: {exc}",
                "messages": [AIMessage(content=DONT_KNOW)],
            }

    def finalize_response(state: RagState) -> dict[str, Any]:
        sources = state.get("sources", [])
        answer_text = normalize_answer_text(state.get("answer_text", ""))
        failure_reason = state.get("failure_reason", "")

        if not state.get("retrieved_context"):
            answer_text = DONT_KNOW
            sources = []
            failure_reason = failure_reason or "no_context"

        latency = time.perf_counter() - state.get("start_time", time.perf_counter())
        return {
            "answer": format_final_answer(answer_text, sources),
            "answer_text": answer_text,
            "sources": sources,
            "latency_seconds": latency,
            "retrieval_pipeline": PIPELINE_NAME,
            "failure_reason": failure_reason,
        }

    workflow.add_node("initialize_state", initialize_state)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("prepare_context", prepare_context)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("finalize_response", finalize_response)

    workflow.add_edge(START, "initialize_state")
    workflow.add_edge("initialize_state", "retrieve_context")
    workflow.add_conditional_edges(
        "retrieve_context",
        route_after_retrieval,
        {
            "prepare_context": "prepare_context",
            "finalize_response": "finalize_response",
        },
    )
    workflow.add_edge("prepare_context", "generate_answer")
    workflow.add_edge("generate_answer", "finalize_response")
    workflow.add_edge("finalize_response", END)

    return workflow.compile(checkpointer=MemorySaver())


def normalize_answer_text(answer_text: str) -> str:
    if not answer_text or DONT_KNOW.lower() in answer_text.lower():
        return DONT_KNOW
    return answer_text


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
                    f"document_year: {chunk.get('document_year', '')}",
                    f"document_date: {chunk.get('document_date', '')}",
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
                "document_year": chunk.get("document_year"),
                "document_date": chunk.get("document_date", ""),
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
            f"year: {source.get('document_year', '')}, "
            f"page: {page_display}, "
            f"chunk_id: {source.get('chunk_id', '')}, "
            f"url: {source.get('source_url', '')}"
        )

    return f"Answer:\n{answer_text}\n\nSources:\n\n" + "\n".join(source_lines)
