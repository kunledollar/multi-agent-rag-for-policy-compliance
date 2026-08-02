"""Typed, isolated execution configurations for the Sentinel ablation study."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Mapping

from app.agents.answer_generation_agent import AnswerGenerationAgent
from app.agents.compliance_agent import ComplianceAgent
from app.agents.fact_check_agent import FactCheckAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.retriever_agent import RetrieverAgent

from .dispatcher import ExecutionDispatcher
from .models import ExecutionMode, ModeExecution
from .refusal import refusal_observed
from .source_ids import normalize_retrieved_chunk
from .uncertainty import uncertainty_observed


class AblationId(str, Enum):
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"
    A6 = "A6"


@dataclass(frozen=True)
class AblationConfiguration:
    configuration_id: AblationId
    configuration_name: str
    execution_mode: ExecutionMode
    enable_reranker: bool
    enable_verification: bool
    enable_compliance_enforcement: bool
    enable_answer_generation: bool
    description: str

    @property
    def enabled_components(self) -> tuple[str, ...]:
        return tuple(name for name, enabled in self._components().items() if enabled)

    @property
    def disabled_components(self) -> tuple[str, ...]:
        return tuple(name for name, enabled in self._components().items() if not enabled)

    def _components(self) -> Mapping[str, bool]:
        return {"reranker": self.enable_reranker, "verification": self.enable_verification,
                "compliance_enforcement": self.enable_compliance_enforcement,
                "answer_generation": self.enable_answer_generation}


def _config(identifier: AblationId, name: str, mode: ExecutionMode, flags: tuple[bool, bool, bool, bool], description: str):
    return AblationConfiguration(identifier, name, mode, *flags, description)


CONFIGURATIONS: Dict[AblationId, AblationConfiguration] = {
    AblationId.A0: _config(AblationId.A0, "full_sentinel", ExecutionMode.FULL_SENTINEL, (True, True, True, True), "Existing Full Sentinel behavior."),
    AblationId.A1: _config(AblationId.A1, "no_reranker", ExecutionMode.FULL_SENTINEL, (False, True, True, True), "Bypass stable reranking and retain retrieval order."),
    AblationId.A2: _config(AblationId.A2, "no_verification", ExecutionMode.FULL_SENTINEL, (True, False, True, True), "Bypass verification only."),
    AblationId.A3: _config(AblationId.A3, "no_compliance", ExecutionMode.FULL_SENTINEL, (True, True, False, True), "Bypass compliance enforcement only."),
    AblationId.A4: _config(AblationId.A4, "no_answer_generation", ExecutionMode.FULL_SENTINEL, (True, True, True, False), "Return the upstream reasoning summary without synthesis."),
    AblationId.A5: _config(AblationId.A5, "rag_only", ExecutionMode.RAG_ONLY, (False, False, False, True), "Existing single-stage RAG mode."),
    AblationId.A6: _config(AblationId.A6, "llm_only", ExecutionMode.LLM_ONLY, (False, False, False, False), "Existing monolithic LLM mode without retrieval or specialized generation."),
}


def get_configuration(value: str | AblationId) -> AblationConfiguration:
    try:
        identifier = value if isinstance(value, AblationId) else AblationId(str(value).upper())
        return CONFIGURATIONS[identifier]
    except ValueError as error:
        raise ValueError(f"Unknown configuration {value!r}; expected A0 through A6") from error


class AblationDispatcher:
    """Execute ablations without changing the production dispatcher defaults."""
    def __init__(self, production=None, retriever_factory=RetrieverAgent,
                 compliance_factory=ComplianceAgent, reasoning_factory=ReasoningAgent,
                 generation_factory=AnswerGenerationAgent, verification_factory=FactCheckAgent,
                 reranker: Callable[[list[dict]], list[dict]] | None = None):
        self.production = production or ExecutionDispatcher()
        self.retriever_factory = retriever_factory; self.compliance_factory = compliance_factory
        self.reasoning_factory = reasoning_factory; self.generation_factory = generation_factory
        self.verification_factory = verification_factory
        # The current explicit reranker is identity-preserving; A1 bypasses this call.
        self.reranker = reranker or (lambda chunks: list(chunks))

    def execute(self, question: str, configuration: AblationConfiguration, *, top_k=5, run_id="", question_id="") -> ModeExecution:
        if configuration.configuration_id == AblationId.A0:
            output = self.production.execute(question, ExecutionMode.FULL_SENTINEL, top_k=top_k, run_id=run_id, question_id=question_id)
            return self._instrument(output, configuration, True, True, True, True)
        if configuration.execution_mode in {ExecutionMode.RAG_ONLY, ExecutionMode.LLM_ONLY}:
            output = self.production.execute(question, configuration.execution_mode, top_k=top_k, run_id=run_id, question_id=question_id)
            return self._instrument(output, configuration, False, False, False, configuration.enable_answer_generation)

        chunks = [normalize_retrieved_chunk(c) for c in self.retriever_factory().retrieve(question, top_k=top_k)]
        reranked = configuration.enable_reranker
        if reranked: chunks = self.reranker(chunks)
        verified = configuration.enable_verification
        if verified: self.verification_factory().run({"question": question, "retrieved_chunks": chunks})
        compliance_invoked = configuration.enable_compliance_enforcement
        if compliance_invoked:
            compliance = self.compliance_factory().run(query=question, retrieved_chunks=chunks).__dict__
        else:
            compliance = {"verdict": "unknown", "status": "not_applicable", "rationale": "Compliance enforcement disabled by A3.", "citations": [], "flags": [], "confidence": 0.0}
        reasoning = self.reasoning_factory().run(question=question, compliance_result=compliance, retrieved_chunks=chunks)
        generated = configuration.enable_answer_generation
        if generated:
            answer = self.generation_factory().run(question=question, compliance_result=compliance, reasoning_result=reasoning, retrieved_chunks=chunks)
            text = answer.get("answer", ""); citations = answer.get("citations", [])
        else:
            # Nearest valid upstream response: an existing reasoning summary, never newly fabricated prose.
            text = reasoning.get("summary_reasoning", ""); citations = None
        output = ModeExecution(answer=text, policy_decision=compliance.get("verdict") if compliance_invoked else None,
            enforcement_action=compliance.get("status") if compliance_invoked else None,
            uncertainty_observed=uncertainty_observed({}, text), refusal_observed=refusal_observed({}, text),
            verified_facts=[] if verified else None, citations=citations, retrieved_chunks=chunks,
            trace_elements=["retrieval step", "source selection", "policy interpretation", "risk assessment"] + (["compliance decision"] if compliance_invoked else []) + (["final answer rationale"] if generated else []),
            audit={"run_id": run_id, "question_id": question_id, "mode": configuration.execution_mode.value},
            handoffs_attempted=sum((verified, compliance_invoked, True, generated)), handoffs_successful=sum((verified, compliance_invoked, True, generated)))
        return self._instrument(output, configuration, reranked, verified, compliance_invoked, generated)

    @staticmethod
    def _instrument(output: ModeExecution, config: AblationConfiguration, reranker: bool, verification: bool, compliance: bool, generation: bool):
        output.audit.update({"configuration_id": config.configuration_id.value, "configuration_name": config.configuration_name,
            "enabled_components": list(config.enabled_components), "disabled_components": list(config.disabled_components),
            "reranker_invoked": reranker, "verification_invoked": verification,
            "compliance_invoked": compliance, "answer_generation_invoked": generation})
        return output
