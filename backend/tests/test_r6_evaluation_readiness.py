"""Small-sample evaluation and external adapter readiness checks."""
import importlib

from app.agents.contracts import AgentDecision, AgentMessage, RevisionRecord
from app.evaluation.models import BenchmarkCase, ExecutionMode, ModeExecution
from app.evaluation.scoring import score


ADAPTERS = ("mtrag", "legalbench", "xstest", "harmbench", "ragtruth")


def test_all_dataset_adapters_return_benchmark_cases():
    row = {"id": "one", "question": "Is this permitted?", "answer": "Yes"}
    for name in ADAPTERS:
        adapter = importlib.import_module(f"app.evaluation.datasets.{name}_adapter")
        cases = adapter.adapt([row])
        assert len(cases) == 1
        assert isinstance(cases[0], BenchmarkCase)


def test_three_case_sample_exposes_legacy_and_r6_metrics():
    results = []
    for index in range(3):
        case = BenchmarkCase(
            question_id=f"q{index}", question="Question", relevant_chunk_ids=["c1"],
            requires_uncertainty=False, expected_refusal=False,
        )
        output = ModeExecution(
            answer="Grounded answer", uncertainty_observed=False, refusal_observed=False,
            retrieved_chunks=[{"id": "c1", "chunk_id": "c1", "source": "policy"}],
            citations=[], agent_decisions=[AgentDecision("policy", "APPROVE", .9, "Supported")],
            agent_messages=[AgentMessage("policy", "answer", "handoff", "Proceed")],
            revision_history=[RevisionRecord(1, "critic", ["answer"], "Improve", "APPROVE", True)],
            conflict_count=1, revision_success=True, handoffs_attempted=1, handoffs_successful=1,
        )
        results.append(score("sample", case, ExecutionMode.FULL_SENTINEL, output, 5.0).model_dump())

    required = {
        "precision_at_5", "recall_at_5", "reciprocal_rank", "ndcg_at_5",
        "citations_complete", "uncertainty_correct", "refusal_correct", "latency_ms",
        "agent_decision_accuracy", "conflict_resolution_accuracy", "self_correction_rate",
        "collaboration_score",
    }
    assert len(results) == 3
    assert all(required <= result.keys() for result in results)
