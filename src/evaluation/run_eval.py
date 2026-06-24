from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from answering.rag_chain import answer_question
from config import Settings, get_settings
from evaluation.metrics import (
    answer_keyword_score,
    groundedness_score,
    source_recall_at_k,
)
from retrieval.retriever import timed_retrieve


RETRIEVAL_MODES = [
    "similarity",
    "hybrid",
    "metadata_filter",
    "query_rewrite",
    "rerank",
]


def env_int(name: str, default: int = 0) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return int(value)


def env_float(name: str, default: float = 0.0) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return float(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "y", "on"}


def selected_retrieval_modes() -> list[str]:
    raw_modes = os.getenv("EVAL_RETRIEVAL_MODES", "").strip()
    if not raw_modes:
        return RETRIEVAL_MODES

    modes = [mode.strip() for mode in raw_modes.split(",") if mode.strip()]
    unsupported = sorted(set(modes) - set(RETRIEVAL_MODES))
    if unsupported:
        raise ValueError(f"Unsupported EVAL_RETRIEVAL_MODES: {unsupported}")
    return modes


def load_eval_questions(settings: Settings) -> list[dict[str, Any]]:
    return json.loads(settings.eval_questions_path.read_text(encoding="utf-8"))


def metadata_filter_for_question(question: dict[str, Any]) -> dict[str, Any]:
    expected_sources = question.get("expected_sources", [])
    if not expected_sources:
        return {}
    expected = expected_sources[0]
    metadata_filter: dict[str, Any] = {}
    for key in ("document_source", "source_file", "page_number"):
        value = expected.get(key)
        if value not in (None, ""):
            metadata_filter[key] = value
    return metadata_filter


def evaluate_one(
    question: dict[str, Any],
    retrieval_mode: str,
    settings: Settings,
) -> dict[str, Any]:
    if env_bool("EVAL_RETRIEVAL_ONLY", default=False):
        return evaluate_retrieval_only(question, retrieval_mode, settings)

    metadata_filter = (
        metadata_filter_for_question(question)
        if retrieval_mode == "metadata_filter"
        else None
    )
    result = answer_question(
        question["question"],
        retrieval_mode=retrieval_mode,
        metadata_filter=metadata_filter,
        settings=settings,
    )
    retrieved_context = result.get("retrieved_context", [])
    source_recall = source_recall_at_k(
        retrieved_context,
        question.get("expected_sources", []),
        k=settings.top_k,
    )
    keyword_score = answer_keyword_score(
        result.get("answer_text", ""),
        question.get("expected_keywords", []),
    )
    groundedness = groundedness_score(
        result.get("answer_text", ""),
        retrieved_context,
    )

    return {
        "id": question["id"],
        "question": question["question"],
        "difficulty": question.get("difficulty", ""),
        "retrieval_mode": retrieval_mode,
        "metadata_filter": metadata_filter or {},
        "answer": result.get("answer", ""),
        "answer_text": result.get("answer_text", ""),
        "sources": result.get("sources", []),
        "source_recall_at_k": source_recall,
        "groundedness": groundedness,
        "answer_contains_expected_keywords": keyword_score == 1.0,
        "answer_keyword_match_score": keyword_score,
        "latency_seconds": result.get("latency_seconds", 0.0),
        "number_of_retrieved_chunks": len(retrieved_context),
        "failure_reason": result.get("failure_reason", ""),
    }


def evaluate_retrieval_only(
    question: dict[str, Any],
    retrieval_mode: str,
    settings: Settings,
) -> dict[str, Any]:
    metadata_filter = (
        metadata_filter_for_question(question)
        if retrieval_mode == "metadata_filter"
        else None
    )

    if retrieval_mode == "query_rewrite":
        retrieved_context: list[dict[str, Any]] = []
        latency = 0.0
        failure_reason = "skipped_query_rewrite_in_retrieval_only_mode"
    else:
        retrieved_context, latency = timed_retrieve(
            question["question"],
            retrieval_mode=retrieval_mode,
            settings=settings,
            metadata_filter=metadata_filter,
        )
        failure_reason = ""

    source_recall = source_recall_at_k(
        retrieved_context,
        question.get("expected_sources", []),
        k=settings.top_k,
    )
    return {
        "id": question["id"],
        "question": question["question"],
        "difficulty": question.get("difficulty", ""),
        "retrieval_mode": retrieval_mode,
        "metadata_filter": metadata_filter or {},
        "answer": "to be generated after LLM eval",
        "answer_text": "to be generated after LLM eval",
        "sources": [],
        "source_recall_at_k": source_recall,
        "groundedness": 0.0,
        "answer_contains_expected_keywords": False,
        "answer_keyword_match_score": 0.0,
        "latency_seconds": latency,
        "number_of_retrieved_chunks": len(retrieved_context),
        "failure_reason": failure_reason,
    }


def run_evaluation(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    questions = load_eval_questions(settings)
    max_questions = env_int("EVAL_MAX_QUESTIONS", default=0)
    if max_questions > 0:
        questions = questions[:max_questions]

    retrieval_modes = selected_retrieval_modes()
    sleep_seconds = env_float("EVAL_SLEEP_SECONDS", default=0.0)
    records = load_resume_records(
        settings,
        total_questions=len(questions),
        retrieval_modes=retrieval_modes,
    )
    completed_keys = {
        (record.get("id"), record.get("retrieval_mode")) for record in records
    }

    print(
        f"Running {len(questions)} questions x {len(retrieval_modes)} modes "
        f"with EVAL_SLEEP_SECONDS={sleep_seconds:g}"
    )
    if records:
        print(f"Resuming from {len(records)} saved runs.")

    try:
        for question in questions:
            for retrieval_mode in retrieval_modes:
                record_key = (question["id"], retrieval_mode)
                if record_key in completed_keys:
                    continue

                try:
                    records.append(evaluate_one(question, retrieval_mode, settings))
                except Exception as exc:
                    records.append(
                        {
                            "id": question["id"],
                            "question": question["question"],
                            "difficulty": question.get("difficulty", ""),
                            "retrieval_mode": retrieval_mode,
                            "metadata_filter": {},
                            "answer": "",
                            "answer_text": "",
                            "sources": [],
                            "source_recall_at_k": False,
                            "groundedness": 0.0,
                            "answer_contains_expected_keywords": False,
                            "answer_keyword_match_score": 0.0,
                            "latency_seconds": 0.0,
                            "number_of_retrieved_chunks": 0,
                            "failure_reason": f"eval_error: {exc}",
                        }
                    )

                completed_keys.add(record_key)
                save_results(
                    records,
                    settings,
                    total_questions=len(questions),
                    status="running",
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        payload = save_results(
            records,
            settings,
            total_questions=len(questions),
            status="interrupted",
        )
        print("\nEvaluation interrupted. Partial results were saved.")
        print_summary(payload["summary"], records)
        return payload

    payload = save_results(records, settings, total_questions=len(questions))
    print_summary(payload["summary"], records)
    return payload


def load_resume_records(
    settings: Settings,
    total_questions: int,
    retrieval_modes: list[str],
) -> list[dict[str, Any]]:
    if not env_bool("EVAL_RESUME", default=True):
        return []
    if not settings.eval_results_path.exists():
        return []

    try:
        payload = json.loads(settings.eval_results_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read previous eval results, starting fresh: {exc}")
        return []

    summary = payload.get("summary", {})
    if int(summary.get("total_questions", 0)) != total_questions:
        return []

    allowed_modes = set(retrieval_modes)
    records = [
        record
        for record in payload.get("results", [])
        if record.get("id") and record.get("retrieval_mode") in allowed_modes
    ]
    return records


def save_results(
    records: list[dict[str, Any]],
    settings: Settings,
    total_questions: int,
    status: str = "complete",
) -> dict[str, Any]:
    summary = summarize(records, total_questions=total_questions)
    payload = {"status": status, "summary": summary, "results": records}
    write_json_safely(settings.eval_results_path, payload)
    return payload


def write_json_safely(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    temp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp")

    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
        return
    except OSError as exc:
        print(f"Atomic write failed for {path}: {exc}. Trying direct write.")

    try:
        path.write_text(text, encoding="utf-8")
        return
    except OSError as exc:
        fallback_path = path.with_name(
            f"{path.stem}.partial.{int(time.time())}{path.suffix}"
        )
        fallback_path.write_text(text, encoding="utf-8")
        print(f"Could not write {path}. Saved partial results to {fallback_path}.")


def summarize(records: list[dict[str, Any]], total_questions: int) -> dict[str, Any]:
    if not records:
        return {
            "total_questions": total_questions,
            "total_runs": 0,
            "average_latency": 0.0,
            "source_recall_at_k": 0.0,
            "groundedness_score": 0.0,
            "answer_keyword_match_score": 0.0,
            "best_retrieval_mode": "",
        }

    mode_scores: dict[str, list[float]] = defaultdict(list)
    for record in records:
        composite = (
            float(bool(record["source_recall_at_k"]))
            + float(record["groundedness"])
            + float(record["answer_keyword_match_score"])
        ) / 3
        mode_scores[record["retrieval_mode"]].append(composite)

    best_mode = max(
        mode_scores,
        key=lambda mode: mean(mode_scores[mode]) if mode_scores[mode] else 0.0,
    )

    return {
        "total_questions": total_questions,
        "total_runs": len(records),
        "average_latency": mean(float(item["latency_seconds"]) for item in records),
        "source_recall_at_k": mean(
            float(bool(item["source_recall_at_k"])) for item in records
        ),
        "groundedness_score": mean(float(item["groundedness"]) for item in records),
        "answer_keyword_match_score": mean(
            float(item["answer_keyword_match_score"]) for item in records
        ),
        "best_retrieval_mode": best_mode,
    }


def print_summary(summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    print(f"Total questions: {summary['total_questions']}")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Average latency: {summary['average_latency']:.3f}s")
    print(f"Source recall@k: {summary['source_recall_at_k']:.3f}")
    print(f"Groundedness score: {summary['groundedness_score']:.3f}")
    print(f"Answer keyword match score: {summary['answer_keyword_match_score']:.3f}")
    print(f"Best retrieval mode: {summary['best_retrieval_mode']}")

    ranked = sorted(
        records,
        key=lambda item: (
            bool(item["source_recall_at_k"]),
            float(item["groundedness"]),
            float(item["answer_keyword_match_score"]),
        ),
        reverse=True,
    )
    print("\n5 good examples:")
    for item in ranked[:5]:
        print(f"- {item['id']} [{item['retrieval_mode']}]: {item['question']}")

    print("\n5 failed or weak examples:")
    for item in ranked[-5:]:
        reason = item.get("failure_reason") or "low metric score"
        print(f"- {item['id']} [{item['retrieval_mode']}]: {reason}")


def main() -> int:
    run_evaluation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
