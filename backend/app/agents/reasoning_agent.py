"""Independent R6 reasoning agent."""
from __future__ import annotations

from typing import List

from .base_agent import BaseAgent
from .contracts import AgentDecision, SentinelState


class ReasoningAgent(BaseAgent):
    """Reconcile evidence and policy decisions, challenging either when warranted."""

    name = "reasoning"

    def run(self, state: SentinelState) -> AgentDecision:
        verification = state.verification_result.get("decision")
        policy_decision = state.policy_result.get("decision")
        policy_verdict = state.policy_result.get("verdict")
        conflict = bool(
            state.policy_result.get("conflict_detected")
            or state.policy_result.get("potential_conflict")
        )
        evidence_ids = [
            str(chunk.get("chunk_id") or chunk.get("id"))
            for chunk in state.retrieved_chunks
            if chunk.get("chunk_id") or chunk.get("id")
        ]

        concerns: List[str] = []
        if verification != "APPROVE_EVIDENCE":
            concerns.append("unverified_evidence")
        if conflict:
            concerns.append("conflicting_policy_evidence")
        if policy_decision in {"DEFER", "ESCALATE", "REJECT"} or policy_verdict == "unknown":
            concerns.append("policy_not_resolved")

        if concerns:
            target = "verification" if any("evidence" in concern for concern in concerns) else "policy"
            result = AgentDecision(
                agent_name=self.name,
                decision="CHALLENGE",
                confidence=0.86,
                rationale="Policy reasoning depends on unverified, conflicting, or unresolved evidence.",
                requested_action="Verify the evidence again and resolve policy precedence.",
                target_agent=target,
                evidence_ids=evidence_ids,
                concerns=list(dict.fromkeys(concerns)),
            )
        else:
            result = AgentDecision(
                agent_name=self.name,
                decision="SYNTHESIZE",
                confidence=max(0.0, min(1.0, float(state.policy_result.get("confidence", 0.0)))),
                rationale=(
                    f"Verified evidence supports the '{policy_verdict}' policy assessment; "
                    "no unresolved conflict prevents answer synthesis."
                ),
                evidence_ids=evidence_ids,
            )

        state.reasoning_result = result.to_dict()
        state.reasoning_result["summary_reasoning"] = result.rationale
        return result
