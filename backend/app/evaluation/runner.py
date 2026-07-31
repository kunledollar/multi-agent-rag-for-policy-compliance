from __future__ import annotations

import logging
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .aggregation import aggregate, latency_summary
from .benchmark_loader import load_benchmark
from .dispatcher import ExecutionDispatcher
from .exporter import export_run
from .models import BenchmarkDataset, DetailedResult, GovernanceEvaluationRequest, ModeExecution
from .scoring import score

logger = logging.getLogger("sentinel.evaluation")


class EvaluationRunner:
    def __init__(self, dispatcher=None, output_dir=None):
        self.dispatcher = dispatcher or ExecutionDispatcher()
        self.output_dir = Path(output_dir or os.getenv("EVALUATION_RESULTS_DIR", "/app/artifacts/governance_evaluations"))

    def run(self, request: GovernanceEvaluationRequest):
        started = datetime.now(timezone.utc); run_id = str(uuid.uuid4())
        if request.benchmark_cases is not None:
            dataset = BenchmarkDataset(filename="api-benchmark.json", version=request.benchmark_version or "unversioned", cases=request.benchmark_cases)
        else: dataset = load_benchmark(Path(request.benchmark_source))
        results=[]; errors=[]
        for case in dataset.cases:
            for mode in request.selected_modes:
                began=time.perf_counter()
                logger.info("governance evaluation started", extra={"run_id":run_id,"question_id":case.question_id,"execution_mode":mode.value})
                try:
                    execution=self.dispatcher.execute(case.question, mode, top_k=5, run_id=run_id, question_id=case.question_id)
                    row=score(run_id, case, mode, execution, (time.perf_counter()-began)*1000)
                except Exception as error:
                    message=f"{type(error).__name__}: {error}"; now=datetime.now(timezone.utc)
                    row=DetailedResult(run_id=run_id,question_id=case.question_id,category=case.category,question=case.question,execution_mode=mode,reference_answer=case.reference_answer,expected_policy_decision=case.expected_policy_decision,latency_ms=(time.perf_counter()-began)*1000,processing_status="failed",error_message=message)
                    errors.append({"run_id":run_id,"question_id":case.question_id,"mode":mode.value,"error_type":type(error).__name__,"error_message":str(error),"timestamp":now.isoformat()})
                    logger.error("governance evaluation case failed", extra={"run_id":run_id,"question_id":case.question_id,"execution_mode":mode.value})
                results.append(row)
        summary=aggregate(results); latency=latency_summary(results); completed=datetime.now(timezone.utc)
        try: commit=subprocess.check_output(["git","rev-parse","HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception: commit="unknown"
        metadata={"run_id":run_id,"run_name":request.run_name or "governance_evaluation","start_time":started.isoformat(),"completion_time":completed.isoformat(),"governance_dataset":dataset.filename,"benchmark_version":dataset.version,"benchmark_questions":len(dataset.cases),"evaluation_date":started.isoformat(),"corpus_version":os.getenv("CORPUS_VERSION","unknown"),"faiss_index_version":os.getenv("FAISS_INDEX_VERSION","index.faiss"),"model":os.getenv("CHAT_MODEL","gpt-4.1-mini"),"embedding_model":os.getenv("EMBEDDING_MODEL","text-embedding-3-large"),"temperature":0,"retrieval_top_k":5,"code_commit":commit,"selected_modes":[m.value for m in request.selected_modes],"total_cases":len(results),"successful_cases":sum(r.processing_status=="success" for r in results),"failed_cases":len(errors)}
        retrieval=[r.model_dump() for r in results if r.precision_at_5 is not None]
        payload={"run_id":run_id,"status":"completed_with_errors" if errors else "completed","benchmark_metadata":metadata,"summary_rows":[r.model_dump() for r in summary],"detailed_results":[r.model_dump() for r in results],"retrieval_metrics":retrieval,"latency_summary":latency,"errors":errors}
        payload["_sheets"]={"Questions":[c.source_fields or c.model_dump() for c in dataset.cases],"Detailed Results":results,"Governance Summary":summary,"Retrieval Metrics":retrieval,"Latency Summary":latency,"Errors":errors,"Run Metadata":[metadata]}
        artifacts=export_run(self.output_dir, f"{request.run_name or 'governance_evaluation'}_{run_id}", payload, request.output_formats)
        payload["generated_artifacts"]=artifacts
        return payload
