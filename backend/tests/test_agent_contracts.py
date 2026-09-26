"""Contract validation for every independently callable R6 agent."""
from unittest.mock import Mock

from app.agents.answer_agent import AnswerAgent
from app.agents.compliance_agent import ComplianceResult
from app.agents.contracts import AgentDecision, SentinelState
from app.agents.critic_agent import CriticAgent
from app.agents.final_decision_agent import FinalDecisionAgent
from app.agents.policy_agent import PolicyAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.retriever_agent import RetrievalAgent
from app.agents.risk_agent import RiskAgent
from app.agents.verification_agent import VerificationAgent


def test_all_r6_agents_implement_agent_decision_contract():
    chunk = {"id": "c1", "source": "policy", "page": 1, "text": "Policy permits it", "score": .9}
    state = SentinelState(task_id="contract", question="Is it permitted?", retrieved_chunks=[chunk])
    retrieval = object.__new__(RetrievalAgent)
    retrieval.top_k = 1
    retrieval.retrieve = Mock(return_value=[chunk])
    compliance = ComplianceResult("ok", "compliant", .9, "Supported", [], [], {},
                                  violation_risk="Low", policy_alignment_score=.9)
    policy = PolicyAgent(Mock(run=Mock(return_value=compliance)))
    answer = AnswerAgent(Mock(run=Mock(return_value={"answer": "Yes", "citations": [
        {"source": "policy", "page": 1}], "action_items": []})))
    for agent in (retrieval, VerificationAgent(), policy, ReasoningAgent(), answer,
                  CriticAgent(), RiskAgent(), FinalDecisionAgent()):
        result = agent.run(state)
        assert isinstance(result, AgentDecision)
        state.record(result)
