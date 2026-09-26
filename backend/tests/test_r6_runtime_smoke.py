"""Integration-level smoke tests for the R6 graph and its bounded revision loop."""
from app.agents.contracts import AgentDecision, SentinelState
from app.rag.multi_agent_graph import run_multi_agent_graph


class StubAgent:
    def __init__(self, name, decision, update=None):
        self.name, self.decision, self.update = name, decision, update
        self.calls = 0

    def run(self, state):
        self.calls += 1
        if self.update:
            self.update(state)
        return AgentDecision(self.name, self.decision, 0.8, f"{self.name} rationale")


def agents(critic_decision="APPROVE"):
    chunks = [{"id": "c1", "chunk_id": "c1", "source": "policy.txt", "page": 1,
               "text": "Approved policy evidence", "score": 0.9}]
    return {
        "retrieval": StubAgent("retrieval", "EVIDENCE_SUFFICIENT",
                               lambda s: setattr(s, "retrieved_chunks", chunks)),
        "verification": StubAgent("verification", "APPROVE_EVIDENCE",
                                  lambda s: setattr(s, "verification_result", {"decision": "APPROVE_EVIDENCE"})),
        "policy": StubAgent("policy", "APPROVE",
                            lambda s: setattr(s, "policy_result", {"decision": "APPROVE", "evidence_ids": ["c1"]})),
        "reasoning": StubAgent("reasoning", "SYNTHESIZE",
                               lambda s: setattr(s, "reasoning_result", {"decision": "SYNTHESIZE"})),
        "risk": StubAgent("risk", "PROCEED",
                          lambda s: setattr(s, "risk_result", {"decision": "PROCEED"})),
        "answer": StubAgent("answer", "DRAFT", lambda s: setattr(s, "draft_answer", {
            "answer": "Grounded answer", "citations": [{"chunk_id": "c1"}], "action_items": []})),
        "critic": StubAgent("critic", critic_decision,
                            lambda s: setattr(s, "critic_feedback", {"decision": critic_decision,
                                "target_agent": "answer", "rationale": "Review draft"})),
        "final_decision": StubAgent("final_decision", "ANSWER",
                                    lambda s: setattr(s, "final_action", "ANSWER")),
    }


def test_r6_runtime_exposes_complete_governance_trace():
    state = SentinelState(task_id="test-001", question="Test governance question")
    result = run_multi_agent_graph(state, agents=agents())
    for field in ("agent_decisions", "agent_messages", "agent_trace", "critic_feedback",
                  "revision_history", "final_action"):
        assert field in result
    assert result["final_action"] == "ANSWER"
    assert {"agent_name", "decision", "confidence", "rationale", "iteration"} <= result["agent_trace"][0].keys()
    assert [entry["agent_name"] for entry in result["agent_trace"]] == [
        "retrieval", "verification", "policy", "reasoning", "risk", "answer",
        "critic", "final_decision",
    ]


def test_revision_loop_stops_at_two_iterations():
    state = SentinelState(task_id="test-002", question="Revise", max_iterations=2)
    configured = agents("REVISE")
    result = run_multi_agent_graph(state, agents=configured)
    assert result["iteration"] == 2
    assert len(result["revision_history"]) == 2
    assert configured["answer"].calls == 3
    assert configured["critic"].calls == 3
