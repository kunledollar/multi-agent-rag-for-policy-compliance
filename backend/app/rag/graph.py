import uuid
import time
from typing import Dict, Any, List, Optional

from app.telemetry.metrics import (
    agent_execution_total,
    agent_execution_duration_seconds,
)

# ----------------------------
# RAGAS imports (ADD ONLY)
# ----------------------------
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset


def run_sentinel_graph(
    question: str,
    *,
    top_k: int = 5,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:

    trace_id = trace_id or str(uuid.uuid4())

    # --------------------------------------------------
    # Agent Trace Container
    # --------------------------------------------------
    agent_trace: List[Dict[str, Any]] = []

    from app.agents.retriever_agent import RetrieverAgent
    from app.agents.compliance_agent import ComplianceAgent
    from app.agents.reasoning_agent import ReasoningAgent
    from app.agents.answer_generation_agent import AnswerGenerationAgent

    # --------------------------------------------------
    # 1) Retriever Agent
    # --------------------------------------------------
    retriever_start = time.time()

    retriever = RetrieverAgent()
    retrieved_chunks: List[Dict[str, Any]] = retriever.retrieve(
        query=question,
        top_k=top_k,
    )

    retriever_duration = time.time() - retriever_start

    agent_execution_total.labels(agent_name="retriever").inc()
    agent_execution_duration_seconds.labels(
        agent_name="retriever"
    ).observe(retriever_duration)

    agent_trace.append({
        "agent_name": "RetrieverAgent",
        "execution_order": 1,
        "status": "success",
        "latency_ms": int(retriever_duration * 1000),
        "input_summary": f"User question: {question}",
        "decision_rationale": f"Retrieved top {len(retrieved_chunks)} chunks using vector similarity search",
        "confidence_score": 1.0 if retrieved_chunks else 0.0,
        "risk_flag": "Low" if retrieved_chunks else "Medium",
        "chunks_retrieved": len(retrieved_chunks),
        "source_diversity": len({c.get("source") for c in retrieved_chunks if c.get("source")}),
    })

    # --------------------------------------------------
    # 2) Compliance Agent
    # --------------------------------------------------
    compliance_start = time.time()

    compliance = ComplianceAgent().run(
        query=question,
        retrieved_chunks=retrieved_chunks,
    )

    compliance_duration = time.time() - compliance_start

    agent_trace.append({
        "agent_name": "ComplianceAgent",
        "execution_order": 2,
        "status": compliance.status,
        "latency_ms": int(compliance_duration * 1000),
        "input_summary": "Retrieved evidence passed for policy evaluation",
        "decision_rationale": compliance.rationale or "Evaluated policy alignment and compliance risk",
        "confidence_score": compliance.confidence,
        "policy_alignment_score": compliance.policy_alignment_score,
        "violation_risk": compliance.violation_risk,
        "conflict_detected": compliance.conflict_detected,
        "potential_conflict": compliance.potential_conflict,
        "conflict_reason": compliance.conflict_reason,
        "risk_flag": (
            "High" if compliance.violation_risk == "High"
            else "Medium" if compliance.violation_risk == "Medium"
            else "Low"
        ),
        "restriction_triggered": compliance.status != "ok",
        "legal_guardrail_hit": compliance.status != "ok",
    })

    # --------------------------------------------------
    # 3) Reasoning Agent
    # --------------------------------------------------
    reasoning_start = time.time()

    reasoning = ReasoningAgent().run(
        question=question,
        compliance_result=compliance.__dict__,
        retrieved_chunks=retrieved_chunks,
    )

    reasoning_duration = time.time() - reasoning_start

    agent_trace.append({
        "agent_name": "ReasoningAgent",
        "execution_order": 3,
        "status": "success",
        "latency_ms": int(reasoning_duration * 1000),
        "input_summary": "Compliance assessment and retrieved evidence",
        "decision_rationale": "Synthesized reasoning over policy evidence",
        "confidence_score": compliance.confidence,
        "risk_flag": "Low" if compliance.confidence >= 0.75 else "Medium",
    })

    # --------------------------------------------------
    # 4) Answer Generation Agent
    # --------------------------------------------------
    answer_start = time.time()

    answer = AnswerGenerationAgent().run(
        question=question,
        compliance_result=compliance.__dict__,
        reasoning_result=reasoning,
        retrieved_chunks=retrieved_chunks,
    )

    answer_duration = time.time() - answer_start

    agent_trace.append({
        "agent_name": "AnswerGenerationAgent",
        "execution_order": 4,
        "status": "success",
        "latency_ms": int(answer_duration * 1000),
        "input_summary": "Reasoned policy interpretation with evidence",
        "decision_rationale": "Generated final grounded response with citations",
        "confidence_score": compliance.confidence,
        "citations_attached": len(answer.get("citations", [])),
        "risk_flag": "Low" if compliance.confidence >= 0.75 else "Medium",
    })

    # --------------------------------------------------
    # RAGAS EVALUATION (REAL, ISOLATED, NON-BREAKING)
    # --------------------------------------------------
    ragas_metrics: Dict[str, float] = {}

    try:
        contexts = [c.get("text", "") for c in retrieved_chunks if c.get("text")]

        if contexts and answer.get("answer"):
            ragas_dataset = Dataset.from_dict({
                "question": [question],
                "answer": [answer.get("answer")],
                "contexts": [contexts],
                "ground_truth": [""],  # optional, left blank intentionally
            })

            ragas_result = evaluate(
                ragas_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )

            ragas_metrics = {
                "faithfulness": round(float(ragas_result["faithfulness"]), 3),
                "answer_relevancy": round(float(ragas_result["answer_relevancy"]), 3),
                "context_precision": round(float(ragas_result["context_precision"]), 3),
                "context_recall": round(float(ragas_result["context_recall"]), 3),
            }

    except Exception:
        # RAGAS failure must NEVER break Sentinel
        ragas_metrics = {}

    # --------------------------------------------------
    # Final Response (User + Admin Metadata)
    # --------------------------------------------------
    return {
        # ----------------------------
        # User-facing output
        # ----------------------------
        "answer": answer.get("answer"),
        "action_items": (
            ["Consult the authoritative policy owner to resolve conflicting or ambiguous requirements."]
            if compliance.conflict_detected or compliance.potential_conflict
            else answer.get("action_items", [])
        ),
        "citations": answer.get("citations", []),
        "confidence": compliance.confidence,
        "trace_id": trace_id,

        # ----------------------------
        # Admin / Auditor Introspection
        # ----------------------------
        "retrieved_chunks": [
            {
                "text": chunk.get("text"),
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "score": round(float(chunk.get("score", 0.0)), 3),
            }
            for chunk in retrieved_chunks
        ],

        "agent_trace": agent_trace,

        # ----------------------------
        # RAGAS Metrics (NEW, SAFE)
        # ----------------------------
        "ragas_metrics": ragas_metrics,
    }
