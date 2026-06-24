from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from answering.rag_chain import answer_question
from config import get_settings


RetrievalMode = Literal[
    "similarity",
    "hybrid",
    "metadata_filter",
    "query_rewrite",
    "rerank",
]

app = FastAPI(title="Mini RAG Assistant API", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    retrieval_mode: RetrievalMode = "hybrid"
    metadata_filter: dict[str, Any] | None = None


class AskResponse(BaseModel):
    answer: str
    answer_text: str
    sources: list[dict[str, Any]]
    retrieved_context: list[dict[str, Any]]
    latency_seconds: float
    retrieval_latency_seconds: float
    retrieval_mode: str
    failure_reason: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    settings = get_settings()
    result = answer_question(
        request.question.strip(),
        retrieval_mode=request.retrieval_mode,
        metadata_filter=request.metadata_filter,
        settings=settings,
    )
    return AskResponse(**result)
