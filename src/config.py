from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  
    def load_dotenv(*_: object, **__: object) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    chroma_dir: Path
    eval_dir: Path
    source_metadata_path: Path
    chunk_records_path: Path
    eval_questions_path: Path
    eval_results_path: Path
    ollama_model: str
    local_embedding_model: str
    vector_db: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    rerank_top_n: int
    reranker_provider: str
    cross_encoder_model: str


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    data_dir = PROJECT_ROOT / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    eval_dir = data_dir / "eval"

    return Settings(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        chroma_dir=data_dir / "chroma",
        eval_dir=eval_dir,
        source_metadata_path=processed_dir / "source_metadata.json",
        chunk_records_path=processed_dir / "chunks.jsonl",
        eval_questions_path=eval_dir / "eval_questions.json",
        eval_results_path=eval_dir / "eval_results.json",
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip(),
        local_embedding_model=os.getenv(
            "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        vector_db=os.getenv("VECTOR_DB", "chroma").strip().lower(),
        chunk_size=_env_int("CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 150),
        top_k=_env_int("TOP_K", 5),
        rerank_top_n=_env_int("RERANK_TOP_N", 3),
        reranker_provider=os.getenv("RERANKER_PROVIDER", "auto").strip().lower(),
        cross_encoder_model=os.getenv(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ).strip(),
    )


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()
