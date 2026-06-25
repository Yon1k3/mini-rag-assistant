from __future__ import annotations

import sys
import uuid
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
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex}"

    with st.sidebar:
        st.subheader("Metadata filters")
        metadata_filter = render_metadata_filters(options)

    question = st.text_input(
        "Question",
        placeholder="Give me a FastAPI answer from 2022",
    )
    ask = st.button("Ask", type="primary")

    if ask:
        if not question.strip():
            st.warning("Enter a question.")
            return

        with st.spinner("Searching context and generating answer..."):
            result = answer_question(
                question.strip(),
                metadata_filter=metadata_filter,
                settings=settings,
                thread_id=st.session_state.thread_id,
            )

        if not result["retrieved_context"]:
            st.warning("Context was not found.")

        st.markdown(result["answer"])
        st.caption(f"Latency: {result['latency_seconds']:.3f}s")
        st.caption(f"Pipeline: {result['retrieval_pipeline']}")

        if result.get("failure_reason"):
            st.warning(result["failure_reason"])

        if result["sources"]:
            st.subheader("Sources")
            st.dataframe(result["sources"], use_container_width=True)

        if result["retrieved_context"]:
            st.subheader("Retrieved chunks")
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
        "Document source",
        [""] + [str(value) for value in options.get("document_source", [])],
    )
    document_type = st.selectbox(
        "Document type",
        [""] + [str(value) for value in options.get("document_type", [])],
    )
    document_year = st.selectbox(
        "Document year",
        [""] + [str(value) for value in options.get("document_year", [])],
    )
    document_date = st.selectbox(
        "Document date",
        [""] + [str(value) for value in options.get("document_date", [])],
    )
    source_file = st.selectbox(
        "Source file",
        [""] + [str(value) for value in options.get("source_file", [])],
    )
    page_number = st.text_input("PDF page number")

    if document_source:
        metadata_filter["document_source"] = document_source
    if document_type:
        metadata_filter["document_type"] = document_type
    if document_year:
        metadata_filter["document_year"] = int(document_year)
    if document_date:
        metadata_filter["document_date"] = document_date
    if source_file:
        metadata_filter["source_file"] = source_file
    if page_number.strip():
        try:
            metadata_filter["page_number"] = int(page_number.strip())
        except ValueError:
            st.warning("PDF page number must be an integer.")

    return metadata_filter


if __name__ == "__main__":
    main()
