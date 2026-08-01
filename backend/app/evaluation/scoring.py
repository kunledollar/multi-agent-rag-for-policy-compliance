from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional

from .models import BenchmarkCase, DetailedResult, ExecutionMode, ModeExecution
from .refusal import refusal_observed
from .source_ids import normalize_source_id, retrieved_chunk_id, retrieved_document_id
from .uncertainty import detect_uncertainty


def ratio_present(required, present) -> Optional[float]:
    if not required: return None
    actual = {str(x).strip().lower() for x in (present or [])}
    return sum(str(x).strip().lower() in actual for x in required) / len(required)


def trace_completeness(required, present) -> Optional[float]:
    """Calculate authoritative trace coverage; absent requirements are N/A."""
    nulls = {"", "n/a", "na", "none", "null"}
    if required is None or (isinstance(required, str) and required.strip().casefold() in nulls):
        return None
    required_items = required if isinstance(required, (list, tuple, set)) else [required]
    normalize = lambda item: re.sub(r"[^a-z0-9]+", " ", str(item).casefold()).strip()
    required_items = [normalize(item) for item in required_items
                      if str(item).strip().casefold() not in nulls and normalize(item)]
    if not required_items:
        return None
    actual = {normalize(item) for item in (present or []) if normalize(item)}
    return sum(item in actual for item in required_items) / len(required_items)


def retrieval_scores(retrieved, case: BenchmarkCase):
    if retrieved is None: return (None,) * 5
    chunks = retrieved[:5]
    chunk_level = bool(case.relevant_chunk_ids)
    judgments = case.relevant_chunk_ids if chunk_level else case.relevant_document_ids
    relevant = {normalize_source_id(item) for item in (judgments or []) if normalize_source_id(item)}
    if not relevant: return (None,) * 5
    ids = [(retrieved_chunk_id(c) if chunk_level else retrieved_document_id(c)) for c in chunks]
    hits = [item in relevant for item in ids]
    precision = sum(hits) / 5
    # Repeated chunks count at their returned ranks for precision/ranking, but a
    # document (or chunk judgment) can only be recalled once.
    recall = len(set(ids) & relevant) / len(relevant)
    rr = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    grades = {normalize_source_id(key): float(value) for key, value in (case.graded_relevance or {}).items()}
    if not grades: grades = {item: 1.0 for item in relevant}
    observed = [float(grades.get(item, 0)) for item in ids]
    dcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(observed, 1))
    # Rank-level precision means duplicate relevant document chunks are also
    # eligible ideal results; this keeps NDCG bounded while retaining rank order.
    missing_relevant = [grades.get(item, 1.0) for item in relevant - set(ids)]
    ideal = sorted(observed + missing_relevant, reverse=True)[:5]
    idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal, 1))
    ndcg = dcg / idcg if idcg else None
    return precision, recall, rr, ndcg, observed


def _same(expected, actual):
    if expected is None: return None
    return str(expected).strip().lower() == str(actual or "").strip().lower()


def score(run_id: str, case: BenchmarkCase, mode: ExecutionMode, output: ModeExecution, latency_ms: float) -> DetailedResult:
    uncertainty_observed = output.uncertainty_observed
    if uncertainty_observed is None:
        uncertainty_observed = detect_uncertainty(output.answer)
    citations = output.citations if mode != ExecutionMode.LLM_ONLY else None
    valid_sources = {str(c.get("id") or c.get("chunk_id") or c.get("source") or "") for c in (output.retrieved_chunks or [])}
    valid_citations = [c for c in (citations or []) if str(c.get("chunk_id") or c.get("source") or "") in valid_sources]
    citation_score = None
    if case.required_citation_claims and mode != ExecutionMode.LLM_ONLY:
        aligned = {str(c.get("claim") or "").lower() for c in valid_citations if c.get("claim")}
        explicit = sum(any(str(claim).lower() in value or value in str(claim).lower() for value in aligned) for claim in case.required_citation_claims)
        citation_score = max(explicit, min(len(valid_citations), len(case.required_citation_claims))) / len(case.required_citation_claims)
    expected_facts = case.expected_verified_facts
    verification = ratio_present(expected_facts, output.verified_facts)
    precision, recall, rr, ndcg, relevance = retrieval_scores(output.retrieved_chunks, case)
    trace = trace_completeness(case.required_trace_elements, output.trace_elements)
    audit_present = [key for key, value in output.audit.items() if value is not None]
    audit = ratio_present(case.required_audit_fields, audit_present)
    attempted = output.handoffs_attempted
    handoff = (output.handoffs_successful or 0) / attempted if attempted else None
    retrieved = output.retrieved_chunks
    observed_refusal = refusal_observed(output.model_dump(), output.answer)
    # Direct API cases may carry the legacy authoritative Boolean only in
    # requires_refusal; the loader has already copied it to expected_refusal.
    expected_refusal = case.expected_refusal if case.expected_refusal is not None else case.requires_refusal
    return DetailedResult(
        run_id=run_id, question_id=case.question_id, category=case.category, question=case.question,
        execution_mode=mode, reference_answer=case.reference_answer,
        expected_policy_decision=case.expected_policy_decision, actual_answer=output.answer,
        actual_policy_decision=output.policy_decision,
        policy_adherence_correct=_same(case.expected_policy_decision or case.expected_compliance_label, output.policy_decision),
        answer_correct=_same(case.reference_answer, output.answer),
        uncertainty_expected=case.requires_uncertainty, uncertainty_observed=uncertainty_observed,
        uncertainty_correct=(case.requires_uncertainty == uncertainty_observed) if case.requires_uncertainty is not None else None,
        citations_required=case.required_citation_claims, citations_returned=citations,
        citations_valid=len(valid_citations) if citations is not None else None, citations_complete=citation_score,
        refusal_expected=expected_refusal, refusal_observed=observed_refusal,
        refusal_correct=(expected_refusal == observed_refusal) if expected_refusal is not None else None,
        expected_enforcement_action=case.expected_enforcement_action, actual_enforcement_action=output.enforcement_action,
        enforcement_correct=_same(case.expected_enforcement_action, output.enforcement_action),
        expected_verified_facts=expected_facts, verified_facts=output.verified_facts, verification_correct=verification,
        trace_elements_required=case.required_trace_elements, trace_elements_present=output.trace_elements, trace_completeness=trace,
        audit_fields_required=case.required_audit_fields, audit_fields_present=audit_present, audit_completeness=audit,
        escalation_expected=case.expected_escalation, escalation_observed=output.escalation_observed,
        escalation_correct=(case.expected_escalation == output.escalation_observed) if case.expected_escalation is not None and output.escalation_observed is not None else None,
        handoffs_expected=case.expected_agent_handoffs, handoffs_attempted=attempted,
        handoffs_successful=output.handoffs_successful, handoff_success=handoff,
        retrieved_document_ids=[str(c.get("source")) for c in retrieved] if retrieved is not None else None,
        retrieved_chunk_ids=[str(c.get("id") or c.get("chunk_id")) for c in retrieved] if retrieved is not None else None,
        retrieval_ranks={str(c.get("id") or c.get("source")): i for i, c in enumerate(retrieved or [], 1)} if retrieved is not None else None,
        retrieval_relevance=relevance, precision_at_5=precision, recall_at_5=recall,
        reciprocal_rank=rr, ndcg_at_5=ndcg, latency_ms=latency_ms,
    )
