"""Autonomous answer agent with authority to reject unsupported upstream claims."""
from __future__ import annotations

from .base_agent import BaseAgent
from .contracts import AgentDecision, SentinelState
from .r5_answer_generation_agent import AnswerGenerationAgent as GroundedAnswerEngine


class AnswerAgent(BaseAgent):
    name = "answer"

    def __init__(self, generation_engine=None) -> None:
        self.generation_engine = generation_engine or GroundedAnswerEngine()

    def run(self, state: SentinelState) -> AgentDecision:
        verified = state.verification_result.get("decision") == "APPROVE_EVIDENCE"
        policy = state.policy_result.get("decision")
        reasoning = state.reasoning_result.get("decision")
        if not verified or policy in {"DEFER", "ESCALATE", "REJECT"} or reasoning == "CHALLENGE":
            result = AgentDecision(self.name, "REJECT_UPSTREAM", 0.91,
                "Evidence does not support a final policy claim.",
                "Obtain stronger evidence and revise the reasoning.", "retrieval",
                concerns=["unsupported_claim"])
            state.draft_answer = {
                "answer": "I cannot provide a policy conclusion from the available evidence.",
                "action": "RETRIEVE_MORE", "citations": [],
                "action_items": ["Retrieve authoritative policy evidence before retrying."]}
            return result

        generated = self.generation_engine.run(
            question=state.question,
            compliance_result={
                **state.policy_result,
                "citations": state.policy_result.get("policy_citations", []),
                "flags": state.policy_result.get("safety_flags", []),
            },
            reasoning_result=state.reasoning_result,
            retrieved_chunks=state.retrieved_chunks,
        )
        # The generation engine grounds source/page citations. Add canonical chunk
        # IDs so the critic can prove each citation came from retrieved evidence.
        by_location = {
            (chunk.get("source"), chunk.get("page")): chunk.get("chunk_id") or chunk.get("id")
            for chunk in state.retrieved_chunks
        }
        citations = []
        for citation in generated.get("citations", []):
            citation = dict(citation)
            citation["chunk_id"] = by_location.get((citation.get("source"), citation.get("page")))
            if citation["chunk_id"] is not None:
                citations.append(citation)
        state.draft_answer = {
            "answer": str(generated.get("answer", "")).strip(),
            "citations": citations,
            "action_items": list(generated.get("action_items") or []),
            "action": "ANSWER",
        }
        return AgentDecision(self.name, "DRAFT", 0.78,
            "Produced a conservative answer using only verified evidence.",
            evidence_ids=[str(c["chunk_id"]) for c in citations if c.get("chunk_id")])
