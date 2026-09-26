"""Regression coverage for autonomous R6 agent contracts and retained intelligence."""
from unittest.mock import Mock

from app.agents.answer_agent import AnswerAgent
from app.agents.contracts import AgentDecision, SentinelState
from app.agents.critic_agent import CriticAgent
from app.agents.final_decision_agent import FinalDecisionAgent
from app.agents.policy_agent import PolicyAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.verification_agent import VerificationAgent
from app.agents.compliance_agent import ComplianceResult


CHUNKS = [
    {"id": "c1", "chunk_id": "c1", "source": "policy.pdf", "page": 1,
     "text": "Employees may work remotely with manager approval.", "score": 0.91},
    {"id": "c2", "chunk_id": "c2", "source": "policy.pdf", "page": 2,
     "text": "Requests must use the approved workflow.", "score": 0.84},
]


def state() -> SentinelState:
    return SentinelState(task_id="test", question="May I work remotely?",
                         retrieved_chunks=[dict(chunk) for chunk in CHUNKS])


def compliance_result() -> ComplianceResult:
    return ComplianceResult(
        status="ok", verdict="compliant", confidence=0.91,
        rationale="The cited policy permits remote work with approval.",
        policy_citations=[{"source": "policy.pdf", "page": 1,
                           "quote_hint": "may work remotely"}],
        safety_flags=[], timings_ms={}, violation_risk="Low",
        policy_alignment_score=0.91,
    )


def test_all_autonomous_agents_return_agent_decision():
    current = state()
    retriever = object.__new__(RetrieverAgent)
    retriever.top_k = 2
    retriever.retrieve = Mock(return_value=[dict(chunk) for chunk in CHUNKS])
    policy = PolicyAgent(compliance_engine=Mock(run=Mock(return_value=compliance_result())))
    generator = Mock(run=Mock(return_value={
        "answer": "Remote work is permitted with manager approval.",
        "citations": [{"source": "policy.pdf", "page": 1}],
        "action_items": ["Request manager approval."],
    }))
    agents = [retriever, VerificationAgent(), policy, ReasoningAgent(),
              AnswerAgent(generation_engine=generator), CriticAgent(), FinalDecisionAgent()]
    for agent in agents:
        result = agent.run(current)
        assert isinstance(result, AgentDecision), agent
        current.record(result)


def test_policy_agent_preserves_compliance_fields():
    current = state()
    current.verification_result = {"decision": "APPROVE_EVIDENCE"}
    engine = Mock(run=Mock(return_value=compliance_result()))
    result = PolicyAgent(compliance_engine=engine).run(current)
    assert result.decision == "APPROVE"
    assert current.policy_result["policy_citations"]
    assert current.policy_result["violation_risk"] == "Low"
    assert current.policy_result["policy_alignment_score"] == 0.91
    engine.run.assert_called_once_with(query=current.question, retrieved_chunks=current.retrieved_chunks)


def test_answer_agent_rejects_unverified_evidence():
    current = state()
    current.verification_result = {"decision": "REJECT_EVIDENCE"}
    result = AnswerAgent(generation_engine=Mock()).run(current)
    assert result.decision == "REJECT_UPSTREAM"
    assert current.draft_answer["citations"] == []


def test_answer_agent_generates_grounded_answer_and_citations():
    current = state()
    current.verification_result = {"decision": "APPROVE_EVIDENCE"}
    current.policy_result = {"decision": "APPROVE", "verdict": "compliant"}
    current.reasoning_result = {"decision": "SYNTHESIZE", "summary_reasoning": "Supported."}
    generator = Mock(run=Mock(return_value={
        "answer": "Remote work is permitted with manager approval.",
        "citations": [{"source": "policy.pdf", "page": 1, "quote_hint": "manager approval"}],
        "action_items": ["Request manager approval."],
    }))
    result = AnswerAgent(generation_engine=generator).run(current)
    assert result.decision == "DRAFT"
    assert current.draft_answer["answer"]
    assert current.draft_answer["citations"][0]["chunk_id"] == "c1"
    assert current.draft_answer["action_items"]
