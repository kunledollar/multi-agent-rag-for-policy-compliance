from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ExecutionMode(str, Enum):
    FULL_SENTINEL = "full_sentinel"
    RAG_ONLY = "rag_only"
    LLM_ONLY = "llm_only"


ALL_MODES = list(ExecutionMode)


class BenchmarkCase(BaseModel):
    question_id: str = Field(min_length=1)
    category: str = "Uncategorized"
    question: str = Field(min_length=1)
    reference_answer: Optional[str] = None
    expected_policy_decision: Optional[str] = None
    expected_compliance_label: Optional[str] = None
    requires_uncertainty: Optional[bool] = None
    expected_uncertainty_behavior: Optional[str] = None
    requires_refusal: Optional[bool] = None
    expected_refusal: Optional[bool] = None
    expected_enforcement_action: Optional[str] = None
    expected_verified_facts: Optional[List[str]] = None
    expected_escalation: Optional[bool] = None
    relevant_document_ids: Optional[List[str]] = None
    relevant_chunk_ids: Optional[List[str]] = None
    graded_relevance: Optional[Dict[str, float]] = None
    required_citation_claims: Optional[List[str]] = None
    required_trace_elements: Optional[List[str]] = None
    required_audit_fields: Optional[List[str]] = None
    expected_agent_handoffs: Optional[List[str]] = None
    notes: Optional[str] = None
    source_fields: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("question_id", "question", mode="before")
    @classmethod
    def strip_required(cls, value: Any) -> str:
        return str(value or "").strip()


class BenchmarkDataset(BaseModel):
    filename: str
    version: str
    cases: List[BenchmarkCase]

    @model_validator(mode="after")
    def validate_cases(self):
        if not self.cases:
            raise ValueError("Governance Evaluation Dataset contains no benchmark questions")
        ids = [case.question_id for case in self.cases]
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        if duplicates:
            raise ValueError(f"Duplicate question IDs: {', '.join(duplicates)}")
        self.cases.sort(key=lambda case: case.question_id)
        return self


class ModeExecution(BaseModel):
    answer: str = ""
    policy_decision: Optional[str] = None
    enforcement_action: Optional[str] = None
    uncertainty_observed: Optional[bool] = None
    refusal_observed: Optional[bool] = None
    verified_facts: Optional[List[str]] = None
    escalation_observed: Optional[bool] = None
    citations: Optional[List[Dict[str, Any]]] = None
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    trace_elements: Optional[List[str]] = None
    audit: Dict[str, Any] = Field(default_factory=dict)
    handoffs_attempted: Optional[int] = None
    handoffs_successful: Optional[int] = None


class DetailedResult(BaseModel):
    run_id: str; question_id: str; category: str; question: str
    execution_mode: ExecutionMode
    reference_answer: Optional[str] = None
    expected_policy_decision: Optional[str] = None
    actual_answer: str = ""
    actual_policy_decision: Optional[str] = None
    policy_adherence_correct: Optional[bool] = None
    answer_correct: Optional[bool] = None
    uncertainty_expected: Optional[bool] = None
    uncertainty_observed: Optional[bool] = None
    uncertainty_correct: Optional[bool] = None
    citations_required: Optional[List[str]] = None
    citations_returned: Optional[List[Dict[str, Any]]] = None
    citations_valid: Optional[int] = None
    citations_complete: Optional[float] = None
    refusal_expected: Optional[bool] = None
    refusal_observed: Optional[bool] = None
    refusal_correct: Optional[bool] = None
    expected_enforcement_action: Optional[str] = None
    actual_enforcement_action: Optional[str] = None
    enforcement_correct: Optional[bool] = None
    expected_verified_facts: Optional[List[str]] = None
    verified_facts: Optional[List[str]] = None
    verification_correct: Optional[float] = None
    trace_elements_required: Optional[List[str]] = None
    trace_elements_present: Optional[List[str]] = None
    trace_completeness: Optional[float] = None
    audit_fields_required: Optional[List[str]] = None
    audit_fields_present: Optional[List[str]] = None
    audit_completeness: Optional[float] = None
    escalation_expected: Optional[bool] = None
    escalation_observed: Optional[bool] = None
    escalation_correct: Optional[bool] = None
    handoffs_expected: Optional[List[str]] = None
    handoffs_attempted: Optional[int] = None
    handoffs_successful: Optional[int] = None
    handoff_success: Optional[float] = None
    retrieved_document_ids: Optional[List[str]] = None
    retrieved_chunk_ids: Optional[List[str]] = None
    retrieval_ranks: Optional[Dict[str, int]] = None
    retrieval_relevance: Optional[List[float]] = None
    precision_at_5: Optional[float] = None
    recall_at_5: Optional[float] = None
    reciprocal_rank: Optional[float] = None
    ndcg_at_5: Optional[float] = None
    latency_ms: float
    processing_status: str = "success"
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SummaryRow(BaseModel):
    category: str
    primary_metric: str
    full_sentinel: Optional[float] = None
    rag_only: Optional[float] = None
    llm_only: Optional[float] = None
    best_performing_mode: str
    sample_size_full_sentinel: int = 0
    sample_size_rag_only: int = 0
    sample_size_llm_only: int = 0
    notes: str = ""


class GovernanceEvaluationRequest(BaseModel):
    benchmark_source: Optional[str] = None
    benchmark_cases: Optional[List[BenchmarkCase]] = None
    benchmark_version: Optional[str] = None
    selected_modes: List[ExecutionMode] = Field(default_factory=lambda: ALL_MODES.copy())
    run_name: Optional[str] = None
    force_rebuild: bool = False
    output_formats: List[str] = Field(default_factory=lambda: ["xlsx", "json"])

    @model_validator(mode="after")
    def require_authoritative_dataset(self):
        if not self.benchmark_source and self.benchmark_cases is None:
            raise ValueError("A Governance Evaluation Dataset source or cases are required")
        if not self.selected_modes:
            raise ValueError("At least one execution mode is required")
        return self
