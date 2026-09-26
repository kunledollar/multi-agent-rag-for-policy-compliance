from .base_agent import BaseAgent
from .contracts import AgentDecision, SentinelState


class CriticAgent(BaseAgent):
    name = "critic"

    def run(self, state: SentinelState) -> AgentDecision:
        draft = state.draft_answer
        cited = draft.get("citations") or []
        valid_ids = {str(c.get("chunk_id") or c.get("id")) for c in state.retrieved_chunks}
        citation_ids = {str(c.get("chunk_id") or c.get("id")) for c in cited if isinstance(c, dict)}
        unsupported = bool(citation_ids - valid_ids)
        upstream_block = state.risk_result.get("decision") in {"WITHHOLD", "ESCALATE"}
        if unsupported:
            decision, target, rationale = "REJECT", "answer", "Draft cites evidence that was not retrieved."
        elif not draft.get("answer") or (not cited and state.retrieved_chunks):
            decision, target, rationale = "REVISE", "answer", "Draft is empty or does not preserve evidence citations."
        elif upstream_block and draft.get("action") == "ANSWER":
            decision, target, rationale = "REVISE", "answer", "Draft conflicts with the risk agent's withholding decision."
        else:
            decision, target, rationale = "APPROVE", None, "Evidence, policy outcome, reasoning, and answer are mutually consistent."
        result = AgentDecision(self.name, decision, 0.9 if decision != "APPROVE" else 0.82, rationale,
            "Regenerate a conservative, evidence-grounded draft." if target else None, target,
            concerns=[] if decision == "APPROVE" else ["draft_quality"])
        state.critic_feedback = result.to_dict()
        return result

