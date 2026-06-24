from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from answering.rag_chain import answer_question
from config import get_settings
from retrieval.retriever import available_filter_options


def main() -> None:
    st.set_page_config(page_title="Mini RAG Assistant", layout="wide")
    st.title("Mini RAG Assistant")

    settings = get_settings()
    options = available_filter_options(settings)

    with st.sidebar:
        retrieval_mode = st.selectbox(
            "Режим пошуку",
            ["similarity", "hybrid", "metadata_filter", "query_rewrite", "rerank"],
            index=1,
        )
        metadata_filter = render_metadata_filters(options)

    question = st.text_input("Питання", placeholder="How do I declare a path parameter?")
    ask = st.button("Запитати", type="primary")

    if ask:
        if not question.strip():
            st.warning("Введи питання.")
            return

        with st.spinner("Шукаю контекст і генерую відповідь..."):
            result = answer_question(
                question.strip(),
                retrieval_mode=retrieval_mode,
                metadata_filter=metadata_filter,
                settings=settings,
            )

        if not result["retrieved_context"]:
            st.warning("Контекст не знайдено.")

        st.markdown(result["answer"])
        st.caption(f"Latency: {result['latency_seconds']:.3f}s")

        if result.get("failure_reason"):
            st.warning(result["failure_reason"])

        if result["sources"]:
            st.subheader("Джерела")
            st.dataframe(result["sources"], use_container_width=True)

        if result["retrieved_context"]:
            st.subheader("Знайдені chunks")
            for index, chunk in enumerate(result["retrieved_context"], start=1):
                label = (
                    f"{index}. {chunk.get('source_file', '')} "
                    f"#{chunk.get('chunk_id', '')}"
                )
                with st.expander(label):
                    st.json(chunk.get("metadata", {}))
                    st.write(chunk.get("text", ""))


def render_metadata_filters(options: dict[str, list[Any]]) -> dict[str, Any]:
    metadata_filter: dict[str, Any] = {}

    document_source = st.selectbox(
        "Джерело документа",
        [""] + [str(value) for value in options.get("document_source", [])],
    )
    document_type = st.selectbox(
        "Тип документа",
        [""] + [str(value) for value in options.get("document_type", [])],
    )
    source_file = st.selectbox(
        "Файл джерела",
        [""] + [str(value) for value in options.get("source_file", [])],
    )
    page_number = st.text_input("Номер сторінки PDF")

    if document_source:
        metadata_filter["document_source"] = document_source
    if document_type:
        metadata_filter["document_type"] = document_type
    if source_file:
        metadata_filter["source_file"] = source_file
    if page_number.strip():
        try:
            metadata_filter["page_number"] = int(page_number.strip())
        except ValueError:
            st.warning("Номер сторінки PDF має бути цілим числом.")

    return metadata_filter


if __name__ == "__main__":
    main()
