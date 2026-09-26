"""Execution validation for both experiment architectures."""
from unittest.mock import patch

from app.agents.compliance_agent import ComplianceResult
from app.rag.sequential_graph import run_sentinel_graph


def test_r5_sequential_baseline_executes_expected_trace():
    chunks = [{"id": "c1", "chunk_id": "c1", "source": "policy", "page": 1,
               "text": "Policy evidence", "score": .9}]
    compliance = ComplianceResult(
        "ok", "compliant", .9, "Supported", [], [], {},
        violation_risk="Low", policy_alignment_score=.9,
    )
    with patch("app.agents.retriever_agent.RetrieverAgent.__init__", return_value=None), \
         patch("app.agents.retriever_agent.RetrieverAgent.retrieve", return_value=chunks), \
         patch("app.agents.compliance_agent.ComplianceAgent.run", return_value=compliance), \
         patch("app.agents.r5_reasoning_agent.ReasoningAgent.run",
               return_value={"summary_reasoning": "Supported"}), \
         patch("app.agents.r5_answer_generation_agent.AnswerGenerationAgent.run",
               return_value={"answer": "Answer", "citations": [], "action_items": []}):
        result = run_sentinel_graph("Question")
    assert [entry["agent_name"] for entry in result["agent_trace"]] == [
        "RetrieverAgent", "ComplianceAgent", "ReasoningAgent", "AnswerGenerationAgent"
    ]
