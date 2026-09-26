from __future__ import annotations

from .base_agent import BaseAgent
from .contracts import AgentDecision, SentinelState


class PolicyAgent(BaseAgent):
    name = "policy"

    def run(self, state: SentinelState) -> AgentDecision:
        verification = state.verification_result.get("decision")
        citations = [str(c.get("chunk_id") or c.get("id")) for c in state.retrieved_chunks if c.get("chunk_id") or c.get("id")]
        conflict = any(c.get("conflict_detected") or c.get("contradicts") for c in state.retrieved_chunks)
        if verification != "APPROVE_EVIDENCE":
            verdict, confidence, action, target = "DEFER", 0.9, "Resolve evidence quality before policy judgment.", "verification"
        elif conflict:
            verdict, confidence, action, target = "ESCALATE", 0.85, "Resolve policy precedence with a policy owner.", "final_decision"
        else:
            verdict, confidence, action, target = "APPROVE", 0.72, None, None
        rationale = ("Policy judgment withheld because evidence is not verified." if verdict == "DEFER" else
                     "Conflicting policy evidence requires human escalation." if verdict == "ESCALATE" else
                     "Verified policy evidence supports a grounded response; no explicit conflict was found.")
        result = AgentDecision(self.name, verdict, confidence, rationale, action, target, citations,
                               ["policy_conflict"] if conflict else [])
        state.policy_result = {**result.to_dict(), "policy_citations": citations,
                               "conflict_detected": conflict,
                               "risk": "high" if verdict in {"DEFER", "ESCALATE"} else "low"}
        return result

