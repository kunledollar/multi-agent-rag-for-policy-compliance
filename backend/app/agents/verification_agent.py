from __future__ import annotations

from collections import defaultdict

from .base_agent import BaseAgent
from .contracts import AgentDecision, SentinelState


class VerificationAgent(BaseAgent):
    name = "verification"

    def run(self, state: SentinelState) -> AgentDecision:
        usable = [c for c in state.retrieved_chunks if c.get("text") and c.get("source")]
        scores = [float(c.get("score", 0.0)) for c in usable]
        by_source = defaultdict(list)
        for chunk in usable:
            by_source[str(chunk.get("source"))].append(chunk)
        weak = not usable or max(scores, default=0.0) < 0.35
        # Explicit metadata is preferred to unreliable keyword-level contradiction inference.
        conflicts = [c for c in usable if c.get("contradicts") or c.get("conflict_detected")]
        if weak:
            decision, confidence = "REQUEST_MORE_EVIDENCE", 0.9
            rationale = "Retrieved material is missing or below the minimum evidence-quality threshold."
            action, target = "Broaden the query and retrieve additional authoritative sources.", "retrieval"
        elif conflicts:
            decision, confidence = "REJECT_EVIDENCE", 0.85
            rationale = "The evidence contains an explicit unresolved contradiction."
            action, target = "Retrieve the governing policy version and resolve source precedence.", "retrieval"
        else:
            decision = "APPROVE_EVIDENCE"
            confidence = min(0.98, 0.55 + max(scores, default=0.0) * 0.4 + min(len(by_source), 2) * 0.05)
            rationale = f"Validated {len(usable)} attributable chunks across {len(by_source)} source(s)."
            action = target = None
        result = AgentDecision(self.name, decision, confidence, rationale, action, target,
            [str(c.get("chunk_id") or c.get("id")) for c in usable if c.get("chunk_id") or c.get("id")],
            ["contradictory_evidence"] if conflicts else (["weak_evidence"] if weak else []))
        state.verification_result = result.to_dict()
        return result

