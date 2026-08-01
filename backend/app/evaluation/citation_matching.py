from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .source_ids import normalize_source_id, retrieved_chunk_id, retrieved_document_id


def citation_matches_retrieved(citation: Any, retrieved_chunks: Iterable[Dict[str, Any]] | None) -> bool:
    """Deterministically match one structured citation to retrieved evidence.

    A supplied chunk ID is authoritative. Citations without a chunk ID fall
    back to their source/document path. Both identifier types use the same
    corpus-relative normalization, and empty or non-object citations never
    count as evidence.
    """
    if not isinstance(citation, dict):
        return False

    chunks = retrieved_chunks or []
    chunk_id = normalize_source_id(citation.get("chunk_id"))
    if chunk_id:
        return chunk_id in {
            retrieved_chunk_id(chunk)
            for chunk in chunks
            if isinstance(chunk, dict) and retrieved_chunk_id(chunk)
        }

    source = normalize_source_id(citation.get("source"))
    if not source:
        return False
    return source in {
        retrieved_document_id(chunk)
        for chunk in chunks
        if isinstance(chunk, dict) and retrieved_document_id(chunk)
    }


def valid_citations(citations: Iterable[Any] | None, retrieved_chunks: Iterable[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """Return only citations grounded by :func:`citation_matches_retrieved`."""
    return [citation for citation in (citations or []) if citation_matches_retrieved(citation, retrieved_chunks)]
