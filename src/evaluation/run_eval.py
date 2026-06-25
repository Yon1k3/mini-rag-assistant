from __future__ import annotations

import json
import os
import sys
import time
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
from retrieval.retriever import PIPELINE_NAME, PIPELINE_STEPS


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


def load_eval_questions(settings: Settings) -> list[dict[str, Any]]:
    return json.loads(settings.eval_questions_path.read_text(encoding="utf-8"))


def metadata_filter_for_question(question: dict[str, Any]) -> dict[str, Any]:
    metadata_filter = question.get("metadata_filter", {})
    return metadata_filter if isinstance(metadata_filter, dict) else {}


def evaluate_one(question: dict[str, Any], settings: Settings) -> dict[str, Any]:
    metadata_filter = metadata_filter_for_question(question)
    result = answer_question(
        question["question"],
        metadata_filter=metadata_filter or None,
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
        "retrieval_pipeline": PIPELINE_NAME,
        "metadata_filter": metadata_filter,
        "answer": result.get("answer", ""),
        "answer_text": result.get("answer_text", ""),
        "sources": result.get("sources", []),
        "source_recall_at_k": source_recall,
        "groundedness": groundedness,
        "answer_contains_expected_keywords": keyword_score == 1.0,
        "answer_keyword_match_score": keyword_score,
        "latency_seconds": result.get("latency_seconds", 0.0),
        "retrieval_latency_seconds": result.get("retrieval_latency_seconds", 0.0),
        "number_of_retrieved_chunks": len(retrieved_context),
        "failure_reason": result.get("failure_reason", ""),
    }


def run_evaluation(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    questions = load_eval_questions(settings)
    max_questions = env_int("EVAL_MAX_QUESTIONS", default=0)
    if max_questions > 0:
        questions = questions[:max_questions]

    sleep_seconds = env_float("EVAL_SLEEP_SECONDS", default=0.0)
    records = load_resume_records(settings, total_questions=len(questions))
    completed_ids = {record.get("id") for record in records}

    print(
        f"Running {len(questions)} questions with {PIPELINE_NAME} "
        f"and EVAL_SLEEP_SECONDS={sleep_seconds:g}"
    )
    if records:
        print(f"Resuming from {len(records)} saved runs.")

    try:
        for question in questions:
            if question["id"] in completed_ids:
                continue

            try:
                records.append(evaluate_one(question, settings))
            except Exception as exc:
                records.append(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        "difficulty": question.get("difficulty", ""),
                        "retrieval_pipeline": PIPELINE_NAME,
                        "metadata_filter": {},
                        "answer": "",
                        "answer_text": "",
                        "sources": [],
                        "source_recall_at_k": False,
                        "groundedness": 0.0,
                        "answer_contains_expected_keywords": False,
                        "answer_keyword_match_score": 0.0,
                        "latency_seconds": 0.0,
                        "retrieval_latency_seconds": 0.0,
                        "number_of_retrieved_chunks": 0,
                        "failure_reason": f"eval_error: {exc}",
                    }
                )

            completed_ids.add(question["id"])
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

    return [
        record
        for record in payload.get("results", [])
        if record.get("id") and record.get("retrieval_pipeline") == PIPELINE_NAME
    ]


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
    except OSError:
        fallback_path = path.with_name(
            f"{path.stem}.partial.{int(time.time())}{path.suffix}"
        )
        fallback_path.write_text(text, encoding="utf-8")
        print(f"Could not write {path}. Saved partial results to {fallback_path}.")


def summarize(records: list[dict[str, Any]], total_questions: int) -> dict[str, Any]:
    if not records:
        return {
            "retrieval_pipeline": PIPELINE_NAME,
            "pipeline_steps": PIPELINE_STEPS,
            "total_questions": total_questions,
            "total_runs": 0,
            "average_latency": 0.0,
            "source_recall_at_k": 0.0,
            "groundedness_score": 0.0,
            "answer_keyword_match_score": 0.0,
        }

    return {
        "retrieval_pipeline": PIPELINE_NAME,
        "pipeline_steps": PIPELINE_STEPS,
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
    }


def print_summary(summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    print(f"Retrieval pipeline: {summary['retrieval_pipeline']}")
    print(f"Pipeline steps: {summary['pipeline_steps']}")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Average latency: {summary['average_latency']:.3f}s")
    print(f"Source recall@k: {summary['source_recall_at_k']:.3f}")
    print(f"Groundedness score: {summary['groundedness_score']:.3f}")
    print(f"Answer keyword match score: {summary['answer_keyword_match_score']:.3f}")

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
        print(f"- {item['id']}: {item['question']}")

    print("\n5 failed or weak examples:")
    for item in ranked[-5:]:
        reason = item.get("failure_reason") or "low metric score"
        print(f"- {item['id']}: {reason}")


def main() -> int:
    run_evaluation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
