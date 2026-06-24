from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from answering.llm import get_chat_model
from config import Settings, get_settings


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question as a concise documentation search query. "
            "Keep product names, API names, and technical terms. Return only the query.",
        ),
        ("human", "{question}"),
    ]
)


def rewrite_query(question: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    try:
        chain = REWRITE_PROMPT | get_chat_model(settings) | StrOutputParser()
        rewritten = chain.invoke({"question": question}).strip()
    except Exception as exc:
        print(f"Query rewrite failed, using original query: {exc}")
        return question

    return rewritten or question

