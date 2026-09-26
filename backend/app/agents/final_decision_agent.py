from .base_agent import BaseAgent
from .contracts import AgentDecision, SentinelState


class FinalDecisionAgent(BaseAgent):
    name = "final_decision"

    def run(self, state: SentinelState) -> AgentDecision:
        verification = state.verification_result.get("decision")
        risk = state.risk_result.get("decision")
        critic = state.critic_feedback.get("decision")
        if verification in {"REQUEST_MORE_EVIDENCE", "REJECT_EVIDENCE"}:
            action, rationale = "RETRIEVE_MORE", "Verified evidence is insufficient for a governed answer."
        elif risk == "ESCALATE":
            action, rationale = "ESCALATE", "Policy conflict exceeds autonomous decision authority."
        elif critic == "REJECT":
            action, rationale = "REFUSE", "The final draft remained unsupported after review."
        elif critic == "REVISE":
            action, rationale = "CLARIFY", "The answer remains incomplete after the permitted revision loop."
        else:
            action, rationale = "ANSWER", "All governance agents approved a grounded response."
        result = AgentDecision(self.name, action, 0.9, rationale, evidence_ids=state.policy_result.get("policy_citations", []))
        state.final_action = action
        return result
