"""Typed contracts shared by every Sentinel R6 agent."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentDecision:
    agent_name: str
    decision: str
    confidence: float
    rationale: str
    requested_action: Optional[str] = None
    target_agent: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    iteration: int = 0

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    message_type: str
    content: str
    priority: str = "normal"
    iteration: int = 0


@dataclass
class RevisionRecord:
    iteration: int
    requested_by: str
    target_agents: List[str]
    reason: str
    outcome: Optional[str] = None
    successful: Optional[bool] = None


@dataclass
class SentinelState:
    task_id: str
    question: str
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    agent_decisions: List[AgentDecision] = field(default_factory=list)
    agent_messages: List[AgentMessage] = field(default_factory=list)
    verification_result: Dict[str, Any] = field(default_factory=dict)
    policy_result: Dict[str, Any] = field(default_factory=dict)
    reasoning_result: Dict[str, Any] = field(default_factory=dict)
    risk_result: Dict[str, Any] = field(default_factory=dict)
    draft_answer: Dict[str, Any] = field(default_factory=dict)
    critic_feedback: Dict[str, Any] = field(default_factory=dict)
    revision_history: List[RevisionRecord] = field(default_factory=list)
    final_action: Optional[str] = None
    final_answer: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 2

    def record(self, decision: AgentDecision) -> AgentDecision:
        decision.iteration = self.iteration
        self.agent_decisions.append(decision)
        if decision.requested_action and decision.target_agent:
            self.agent_messages.append(AgentMessage(
                sender=decision.agent_name, recipient=decision.target_agent,
                message_type="action_request", content=decision.requested_action,
                priority="high" if decision.concerns else "normal",
                iteration=self.iteration,
            ))
        return decision

