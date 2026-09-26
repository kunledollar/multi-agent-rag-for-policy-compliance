"""Autonomous answer agent with authority to reject unsupported upstream claims."""
from __future__ import annotations

from .base_agent import BaseAgent
from .contracts import AgentDecision, SentinelState


class AnswerAgent(BaseAgent):
    name = "answer"

    def run(self, state: SentinelState) -> AgentDecision:
        verified = state.verification_result.get("decision") == "APPROVE_EVIDENCE"
        policy = state.policy_result.get("decision")
        reasoning = state.reasoning_result.get("decision")
        if not verified or policy == "DEFER" or reasoning == "CHALLENGE":
            result = AgentDecision(self.name, "REJECT_UPSTREAM", 0.91,
                "Evidence does not support a final policy claim.",
                "Obtain stronger evidence and revise the reasoning.", "retrieval",
                concerns=["unsupported_claim"])
            state.draft_answer = {
                "answer": "I cannot provide a policy conclusion from the available evidence.",
                "action": "RETRIEVE_MORE", "citations": []}
            return result

        citations = [{"chunk_id": c.get("chunk_id") or c.get("id"),
                      "source": c.get("source"), "page": c.get("page")}
                     for c in state.retrieved_chunks[:3]]
        excerpts = [(c.get("text") or "").strip() for c in state.retrieved_chunks[:2]]
        answer = "Based on the retrieved policy evidence: " + " ".join(excerpts)
        state.draft_answer = {"answer": answer.strip(), "action": "ANSWER", "citations": citations,
                              "action_items": []}
        return AgentDecision(self.name, "DRAFT", 0.78,
            "Produced a conservative answer using only verified evidence.",
            evidence_ids=[str(c["chunk_id"]) for c in citations if c.get("chunk_id")])


# Transitional class name for external clients; R6 code uses AnswerAgent.
AnswerGenerationAgent = AnswerAgent
