from __future__ import annotations

import math
from statistics import mean, median

from .models import DetailedResult, ExecutionMode, SummaryRow

METRICS = [
    ("Uncertainty Handling", "Correct Uncertainty Signaling Rate", "uncertainty_correct"),
    ("Citation Grounding", "Citation Completeness", "citations_complete"),
    ("Policy Compliance", "Policy-Adherence Accuracy", "policy_adherence_correct"),
    ("Risk Control", "Correct Risk-Aware Refusal Rate", "refusal_correct"),
    ("Governance Decisions", "Compliance Enforcement Accuracy", "enforcement_correct"),
    ("Verification", "Fact Verification Accuracy", "verification_correct"),
    ("Explainability", "Decision Trace Completeness", "trace_completeness"),
    ("Auditability", "Audit Log Completeness", "audit_completeness"),
    ("Human Oversight", "Escalation Accuracy", "escalation_correct"),
    ("Multi-Agent", "Agent Handoff Success Rate", "handoff_success"),
    ("Operations", "Governance Latency", "latency_ms"),
    ("Retrieval", "Precision@5", "precision_at_5"), ("Retrieval", "Recall@5", "recall_at_5"),
    ("Retrieval", "MRR", "reciprocal_rank"), ("Retrieval", "NDCG@5", "ndcg_at_5"),
]


def aggregate(results: list[DetailedResult]) -> list[SummaryRow]:
    rows = []
    for category, label, field in METRICS:
        values = {}
        for mode in ExecutionMode:
            applicable = [getattr(r, field) for r in results if r.execution_mode == mode and r.processing_status == "success" and getattr(r, field) is not None]
            values[mode.value] = (mean(applicable) if applicable else None, len(applicable))
        valid = {mode: pair[0] for mode, pair in values.items() if pair[0] is not None}
        if not valid: best = "N/A"
        else:
            optimum = min(valid.values()) if field == "latency_ms" else max(valid.values())
            winners = [mode for mode, value in valid.items() if math.isclose(value, optimum)]
            best = winners[0] if len(winners) == 1 else "Tie: " + ", ".join(winners)
        rows.append(SummaryRow(category=category, primary_metric=label,
            full_sentinel=values["full_sentinel"][0], rag_only=values["rag_only"][0], llm_only=values["llm_only"][0],
            best_performing_mode=best, sample_size_full_sentinel=values["full_sentinel"][1],
            sample_size_rag_only=values["rag_only"][1], sample_size_llm_only=values["llm_only"][1],
            notes="N/A values are excluded; latency selects the lowest value."))
    return rows


def percentile(values, p):
    ordered = sorted(values)
    if not ordered: return None
    position = (len(ordered) - 1) * p; low = int(position); high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def latency_summary(results):
    output = []
    for mode in ExecutionMode:
        values = [r.latency_ms for r in results if r.execution_mode == mode and r.processing_status == "success"]
        if values: output.append({"execution_mode": mode.value, "count": len(values), "mean": mean(values), "median": median(values), "minimum": min(values), "maximum": max(values), "p90": percentile(values,.9), "p95": percentile(values,.95), "p99": percentile(values,.99)})
    return output
