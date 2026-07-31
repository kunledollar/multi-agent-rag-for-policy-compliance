from __future__ import annotations

import os
from typing import Callable

from openai import OpenAI

from app.agents.answer_generation_agent import AnswerGenerationAgent
from app.agents.retriever_agent import RetrieverAgent
from app.rag.graph import run_sentinel_graph

from .models import ExecutionMode, ModeExecution
from .refusal import refusal_observed
from .source_ids import normalize_retrieved_chunk
from .uncertainty import uncertainty_observed


class ExecutionDispatcher:
    def __init__(self, full=None, retriever_factory=RetrieverAgent, llm=None):
        self.full = full or run_sentinel_graph; self.retriever_factory = retriever_factory; self.llm = llm or self._direct_llm

    def execute(self, question: str, mode: ExecutionMode, *, top_k=5, run_id="", question_id="") -> ModeExecution:
        if mode == ExecutionMode.FULL_SENTINEL:
            raw = self.full(question=question, top_k=top_k)
            trace = raw.get("agent_trace", []); chunks = self._normalize_chunks(raw.get("retrieved_chunks", []))
            return ModeExecution(answer=raw.get("answer", ""), policy_decision=raw.get("policy_decision"), enforcement_action=raw.get("enforcement_action"),
                uncertainty_observed=uncertainty_observed(raw), refusal_observed=refusal_observed(raw), escalation_observed=raw.get("escalation_observed"), citations=raw.get("citations", []), retrieved_chunks=chunks,
                trace_elements=["retrieval step", "source selection", "policy interpretation", "risk assessment", "compliance decision", "final answer rationale"],
                audit={"run_id":run_id,"question_id":question_id,"mode":mode.value,"timestamps":True,"selected_sources":[c.get("source") for c in chunks],"agent_names":[x.get("agent_name") for x in trace],"model_identifier":os.getenv("CHAT_MODEL","gpt-4.1-mini"),"latency":True,"error_status":"success"},
                handoffs_attempted=max(len(trace)-1,0), handoffs_successful=sum(x.get("status") in {"success","ok"} for x in trace[1:]))
        if mode == ExecutionMode.RAG_ONLY:
            chunks = self._normalize_chunks(self.retriever_factory().retrieve(question, top_k=top_k))
            answer = AnswerGenerationAgent().run(question, {"verdict":"unknown"}, {}, chunks)
            return ModeExecution(answer=answer.get("answer", ""), uncertainty_observed=uncertainty_observed(answer), refusal_observed=refusal_observed(answer), citations=answer.get("citations", []), retrieved_chunks=chunks,
                trace_elements=["retrieval step", "source selection", "final answer rationale"], audit={"run_id":run_id,"question_id":question_id,"mode":mode.value,"timestamps":True,"selected_sources":[c.get("source") for c in chunks],"model_identifier":os.getenv("CHAT_MODEL","gpt-4.1-mini"),"latency":True,"error_status":"success"})
        answer = self.llm(question)
        return ModeExecution(answer=answer, uncertainty_observed=uncertainty_observed({}, answer), refusal_observed=refusal_observed({}, answer), citations=None, retrieved_chunks=None,
            trace_elements=["final answer rationale"], audit={"run_id":run_id,"question_id":question_id,"mode":mode.value,"timestamps":True,"model_identifier":os.getenv("CHAT_MODEL","gpt-4.1-mini"),"latency":True,"error_status":"success"})

    @staticmethod
    def _normalize_chunks(chunks):
        return [normalize_retrieved_chunk(chunk) for chunk in (chunks or [])]

    @staticmethod
    def _direct_llm(question):
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key: raise RuntimeError("OPENAI_API_KEY is not configured")
        response = OpenAI(api_key=key).chat.completions.create(model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"), messages=[{"role":"system","content":"Answer the question directly. Do not create citations or claim retrieval."},{"role":"user","content":question}], temperature=0)
        return response.choices[0].message.content or ""
