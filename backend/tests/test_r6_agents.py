import unittest

from app.agents.answer_agent import AnswerAgent
from app.agents.contracts import SentinelState
from app.agents.critic_agent import CriticAgent
from app.agents.final_decision_agent import FinalDecisionAgent
from app.agents.policy_agent import PolicyAgent
from app.agents.risk_agent import RiskAgent
from app.agents.verification_agent import VerificationAgent


class R6AgentTests(unittest.TestCase):
    def state(self):
        return SentinelState(task_id="t", question="May I do this?", retrieved_chunks=[
            {"id": "c1", "source": "policy.pdf", "page": 1, "text": "The action is permitted.", "score": .9}
        ])

    def test_agents_make_traceable_independent_decisions(self):
        state = self.state()
        for agent in (VerificationAgent(), PolicyAgent(), RiskAgent(), AnswerAgent(), CriticAgent()):
            state.record(agent.run(state))
        self.assertEqual(state.verification_result["decision"], "APPROVE_EVIDENCE")
        self.assertEqual(state.critic_feedback["decision"], "APPROVE")
        self.assertTrue(all(0 <= d.confidence <= 1 and d.rationale for d in state.agent_decisions))

    def test_weak_evidence_triggers_retrieval_instead_of_answer(self):
        state = SentinelState(task_id="t", question="unknown")
        for agent in (VerificationAgent(), PolicyAgent(), RiskAgent(), AnswerAgent(), CriticAgent()):
            state.record(agent.run(state))
        decision = FinalDecisionAgent().run(state)
        self.assertEqual(decision.decision, "RETRIEVE_MORE")
        self.assertTrue(state.agent_messages)


if __name__ == "__main__":
    unittest.main()
