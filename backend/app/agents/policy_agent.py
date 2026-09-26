from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from .base_agent import BaseAgent
from .contracts import AgentDecision, SentinelState
from .compliance_agent import ComplianceAgent


class PolicyAgent(BaseAgent):
    name = "policy"

    def __init__(self, compliance_engine: Optional[ComplianceAgent] = None) -> None:
        self.compliance_engine = compliance_engine or ComplianceAgent()

    def run(self, state: SentinelState) -> AgentDecision:
        compliance = self.compliance_engine.run(
            query=state.question, retrieved_chunks=state.retrieved_chunks
        )
        policy = asdict(compliance)
        verification_approved = state.verification_result.get("decision") == "APPROVE_EVIDENCE"

        if not verification_approved or compliance.status != "ok":
            decision, action, target = "DEFER", "Verify or retrieve stronger policy evidence.", "verification"
        elif compliance.conflict_detected or compliance.potential_conflict:
            decision, action, target = "ESCALATE", "Resolve policy precedence with an authorized owner.", "final_decision"
        elif compliance.verdict == "non_compliant":
            decision, action, target = "REJECT", "Apply the policy restriction or approval workflow.", "final_decision"
        elif compliance.verdict == "unknown":
            decision, action, target = "DEFER", "Clarify which policy rule governs this request.", "verification"
        else:
            decision, action, target = "APPROVE", None, None

        evidence_ids = [
            str(chunk.get("chunk_id") or chunk.get("id"))
            for chunk in state.retrieved_chunks
            if chunk.get("chunk_id") or chunk.get("id")
        ]
        concerns = list(compliance.safety_flags)
        if compliance.conflict_detected or compliance.potential_conflict:
            concerns.append("policy_conflict")
        result = AgentDecision(
            agent_name=self.name, decision=decision,
            confidence=compliance.confidence, rationale=compliance.rationale,
            requested_action=action, target_agent=target,
            evidence_ids=evidence_ids, concerns=concerns,
        )
        # Preserve the complete, research-relevant ComplianceAgent output while
        # also exposing the autonomous policy decision used by downstream agents.
        state.policy_result = {**policy, "decision": decision, "evidence_ids": evidence_ids}
        return result
