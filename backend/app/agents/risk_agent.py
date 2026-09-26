from .base_agent import BaseAgent
from .contracts import AgentDecision, SentinelState


class RiskAgent(BaseAgent):
    name = "risk"

    def run(self, state: SentinelState) -> AgentDecision:
        verified = state.verification_result.get("decision") == "APPROVE_EVIDENCE"
        policy = state.policy_result.get("decision")
        if policy == "ESCALATE":
            decision, confidence, action = "ESCALATE", 0.92, "Route to an authorized policy owner."
        elif not verified:
            decision, confidence, action = "WITHHOLD", 0.9, "Do not make unsupported claims; retrieve more evidence."
        else:
            decision, confidence, action = "PROCEED", 0.78, None
        result = AgentDecision(self.name, decision, confidence,
            "Assessed uncertainty, refusal need, and escalation risk from verification and policy outcomes.",
            action, "final_decision" if action else None,
            concerns=[] if decision == "PROCEED" else ["governance_risk"])
        state.risk_result = result.to_dict()
        return result

