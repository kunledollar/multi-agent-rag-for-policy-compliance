"""Adapter for legalbench records; no external dataset package is required."""
from typing import Any, Iterable
from app.evaluation.models import BenchmarkCase


def adapt(records: Iterable[dict[str, Any]]) -> list[BenchmarkCase]:
    """Normalize already-loaded benchmark rows into Sentinel's contract."""
    cases=[]
    for index,row in enumerate(records):
        question=row.get("question") or row.get("prompt") or row.get("query")
        if not question: continue
        cases.append(BenchmarkCase(question_id=str(row.get("id") or row.get("question_id") or index),
            category=str(row.get("category") or "legalbench"), question=str(question),
            reference_answer=row.get("answer") or row.get("reference_answer"), source_fields=dict(row)))
    return cases
