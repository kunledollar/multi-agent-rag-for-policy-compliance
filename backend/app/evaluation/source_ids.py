from __future__ import annotations

import re
from typing import Any, Dict


_RAW_PREFIX = re.compile(r"^(?:app/)?data/raw/|^raw/", re.IGNORECASE)


def normalize_source_id(value: Any) -> str:
    """Return the corpus-relative, case-insensitive form of a source identifier."""
    if value is None:
        return ""
    normalized = str(value).strip().replace("\\", "/")
    normalized = re.sub(r"/+", "/", normalized)
    normalized = normalized.removeprefix("./").lstrip("/")
    normalized = _RAW_PREFIX.sub("", normalized)
    return normalized.strip("/").casefold()


def normalize_retrieved_chunk(value: Any) -> Dict[str, Any]:
    """Coerce dispatcher results and expose common document/chunk ID fields."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    chunk = dict(value) if isinstance(value, dict) else {"text": str(value)}
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    if not chunk.get("source"):
        chunk["source"] = next(
            (metadata.get(key) or chunk.get(key) for key in ("source", "document_id", "file_path", "path") if metadata.get(key) or chunk.get(key)),
            None,
        )
    def identifier(candidate: Any) -> Any:
        if candidate is None or not str(candidate).strip():
            return None
        if str(candidate).strip().casefold() in {"none", "null"}:
            return None
        return candidate

    # document_id is a source/path identifier in this repository, not a chunk
    # identifier.  Only the explicit chunk identifier fields are promoted.
    chunk_id = next(
        (
            candidate
            for candidate in (
                identifier(chunk.get("id")),
                identifier(chunk.get("chunk_id")),
                identifier(metadata.get("id")),
                identifier(metadata.get("chunk_id")),
            )
            if candidate is not None
        ),
        None,
    )
    if chunk_id is not None:
        chunk["id"] = chunk_id
        chunk["chunk_id"] = chunk_id
    return chunk


def retrieved_document_id(chunk: Dict[str, Any]) -> str:
    return normalize_source_id(chunk.get("source") or chunk.get("document_id") or chunk.get("file_path") or chunk.get("path"))


def retrieved_chunk_id(chunk: Dict[str, Any]) -> str:
    identifier = chunk.get("id") or chunk.get("chunk_id")
    if str(identifier or "").strip().casefold() in {"none", "null"}:
        return ""
    return normalize_source_id(identifier)
