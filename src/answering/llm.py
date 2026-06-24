from __future__ import annotations

from typing import Any

from config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> Any:
    settings = settings or get_settings()

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        temperature=0,
    )
