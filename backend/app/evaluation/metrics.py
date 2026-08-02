"""Null-preserving ablation metric aggregation."""
from __future__ import annotations

from statistics import mean

METRICS = ("faithfulness", "answer_relevancy", "uncertainty_correct", "citations_complete",
           "refusal_correct", "enforcement_correct", "verification_correct", "trace_completeness",
           "escalation_correct", "handoff_success", "precision_at_5", "recall_at_5",
           "reciprocal_rank", "ndcg_at_5", "latency_ms")


def summarize(rows):
    result = {}
    for metric in METRICS:
        values = [row.get(metric) for row in rows if row.get(metric) is not None]
        result[metric] = {"mean": mean(values), "n": len(values)} if values else {"mean": None, "n": 0}
    return result
