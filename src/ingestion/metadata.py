from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import Settings, get_settings, project_relative


def load_source_metadata(settings: Settings | None = None) -> dict[str, dict[str, Any]]:
    settings = settings or get_settings()
    if not settings.source_metadata_path.exists():
        return {}

    raw = json.loads(settings.source_metadata_path.read_text(encoding="utf-8-sig"))
    records = raw if isinstance(raw, list) else raw.get("documents", [])

    catalog: dict[str, dict[str, Any]] = {}
    for record in records:
        local_path = record.get("local_path")
        if not local_path:
            continue
        absolute_path = (settings.project_root / str(local_path)).resolve()
        catalog[str(absolute_path)] = dict(record)
    return catalog


def metadata_for_file(path: Path, catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    record = catalog.get(str(path.resolve()), {})
    document_type = record.get("document_type") or infer_document_type(path)
    metadata = {
        "source_file": record.get("local_path") or project_relative(path),
        "source_url": record.get("original_url", ""),
        "document_source": record.get("document_source", "Manual"),
        "document_type": document_type,
    }
    for key in ("document_date", "download_timestamp"):
        if record.get(key):
            metadata[key] = record[key]

    document_year = record.get("document_year") or record.get("year")
    if document_year is None:
        document_year = year_from_date(record.get("document_date", ""))
    if document_year is not None:
        metadata["document_year"] = int(document_year)

    return metadata


def year_from_date(value: str) -> int | None:
    if not value or len(value) < 4:
        return None
    year = value[:4]
    return int(year) if year.isdigit() else None


def infer_document_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".txt":
        return "txt"
    return "unknown"


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, Path):
            sanitized[key] = value.as_posix()
        else:
            sanitized[key] = json.dumps(value, ensure_ascii=True)
    return sanitized


def metadata_to_source(metadata: dict[str, Any]) -> dict[str, Any]:
    page_number = metadata.get("page_number")
    if page_number == "":
        page_number = None
    return {
        "source_file": metadata.get("source_file", ""),
        "source_url": metadata.get("source_url", ""),
        "document_source": metadata.get("document_source", ""),
        "document_type": metadata.get("document_type", ""),
        "document_year": metadata.get("document_year"),
        "document_date": metadata.get("document_date", ""),
        "page_number": page_number,
        "section_title": metadata.get("section_title", ""),
        "chunk_id": metadata.get("chunk_id", ""),
    }
