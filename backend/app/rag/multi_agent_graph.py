"""Sentinel R6 autonomous, governance-oriented multi-agent execution graph."""
from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional

from app.agents.answer_agent import AnswerAgent
from app.agents.contracts import RevisionRecord, SentinelState
from app.agents.critic_agent import CriticAgent
from app.agents.final_decision_agent import FinalDecisionAgent
from app.agents.policy_agent import PolicyAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.risk_agent import RiskAgent
from app.agents.verification_agent import VerificationAgent
from app.evaluation.refusal import refusal_observed


def _run(state: SentinelState, agents: Iterable[Any]) -> None:
    for agent in agents:
        state.record(agent.run(state))


def run_multi_agent_graph(
    state: SentinelState,
    *,
    top_k: int = 5,
    agents: Optional[Dict[str, Any]] = None,
    disabled_agents: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Execute an initialized R6 state and return its complete governance trace."""
    state.max_iterations = min(2, max(0, state.max_iterations))
    supplied = agents or {}
    disabled = disabled_agents or set()
    retrieval = supplied.get("retrieval") or RetrieverAgent(top_k=top_k)
    verification = supplied.get("verification") or VerificationAgent()
    policy = supplied.get("policy") or PolicyAgent()
    reasoning = supplied.get("reasoning") or ReasoningAgent()
    risk = supplied.get("risk") or RiskAgent()
    answer = supplied.get("answer") or AnswerAgent()
    critic = supplied.get("critic") or CriticAgent()

    state.record(retrieval.run(state))
    if "verification" in disabled:
        state.verification_result = {"decision": "APPROVE_EVIDENCE", "bypassed": True}
    else:
        state.record(verification.run(state))
    if "policy" in disabled:
        state.policy_result = {"decision": "APPROVE", "verdict": "compliant",
                               "confidence": 0.0, "bypassed": True}
    else:
        state.record(policy.run(state))
    _run(state, (reasoning, risk, answer))
    if "critic" in disabled:
        state.critic_feedback = {"decision": "APPROVE", "bypassed": True}
    else:
        state.record(critic.run(state))

    while state.critic_feedback.get("decision") in {"REVISE", "REJECT"} and state.iteration < state.max_iterations:
        feedback = state.critic_feedback
        target = feedback.get("target_agent") or "answer"
        record = RevisionRecord(state.iteration + 1, "critic", [target], feedback.get("rationale", "Revision requested"))
        state.revision_history.append(record)
        state.iteration += 1
        # A rejection caused by evidence returns control to retrieval and all dependent agents.
        if target == "retrieval":
            state.record(retrieval.run(state))
            if "verification" not in disabled:
                state.record(verification.run(state))
            if "policy" not in disabled:
                state.record(policy.run(state))
            _run(state, (reasoning, risk))
        state.record(answer.run(state))
        state.record(critic.run(state))
        record.outcome = state.critic_feedback.get("decision")
        record.successful = record.outcome == "APPROVE"

    final = supplied.get("final_decision") or FinalDecisionAgent()
    state.record(final.run(state))
    if state.final_action == "ANSWER":
        state.final_answer = state.draft_answer.get("answer", "")
    elif state.final_action == "ESCALATE":
        state.final_answer = "This request requires review by an authorized policy owner."
    elif state.final_action == "REFUSE":
        state.final_answer = "I cannot provide an unsupported policy answer."
    else:
        state.final_answer = "More authoritative policy evidence is required before I can answer."

    citations = state.draft_answer.get("citations", []) if state.final_action == "ANSWER" else []
    return {
        "answer": state.final_answer, "action_items": state.draft_answer.get("action_items", []),
        "citations": citations, "confidence": state.agent_decisions[-1].confidence,
        "trace_id": state.task_id, "policy_decision": state.policy_result.get("decision"),
        "enforcement_action": state.final_action, "final_action": state.final_action,
        "iteration": state.iteration,
        "uncertainty_observed": state.final_action != "ANSWER",
        "refusal_observed": refusal_observed({"answer": state.final_answer}),
        "escalation_observed": state.final_action == "ESCALATE",
        "retrieved_chunks": state.retrieved_chunks,
        "agent_decisions": [asdict(d) for d in state.agent_decisions],
        "agent_messages": [asdict(m) for m in state.agent_messages],
        "critic_feedback": state.critic_feedback,
        "revision_history": [asdict(r) for r in state.revision_history],
        "conflict_count": sum(bool(d.concerns) for d in state.agent_decisions),
        "revision_success": any(r.successful for r in state.revision_history),
        "agent_trace": [asdict(d) for d in state.agent_decisions], "ragas_metrics": {},
    }


def run_sentinel_graph(question: str, *, top_k: int = 5,
                       trace_id: Optional[str] = None, max_iterations: int = 2,
                       agents: Optional[Dict[str, Any]] = None,
                       disabled_agents: Optional[set[str]] = None) -> Dict[str, Any]:
    """Compatibility entry point used by the API and evaluation dispatcher."""
    state = SentinelState(
        task_id=trace_id or str(uuid.uuid4()), question=question,
        max_iterations=max_iterations,
    )
    return run_multi_agent_graph(
        state, top_k=top_k, agents=agents, disabled_agents=disabled_agents
    )
