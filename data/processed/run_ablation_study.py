#!/usr/bin/env python3
"""Safe, resumable A0--A6 Sentinel ablation runner."""
from __future__ import annotations

import argparse, json, os, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "backend"))
from app.evaluation.ablation import AblationDispatcher, CONFIGURATIONS, get_configuration
from app.evaluation.benchmark_loader import load_benchmark
from app.evaluation.metrics import summarize
from app.evaluation.scoring import score

DEFAULT_OUTPUT = ROOT / "artifacts/governance_evaluations/ablation"
DEFAULT_MERGED = "sentinel-ablation-600-seven-configurations-v1.0.json"


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8"); temporary.replace(path)


def merge_rows(raw_payloads):
    rows = []; seen = set()
    for payload in raw_payloads:
        for row in payload.get("rows", []):
            key = (row["question_id"], row["configuration_id"])
            if key in seen: raise ValueError(f"Duplicate question/configuration pair: {key}")
            seen.add(key); rows.append(row)
    return sorted(rows, key=lambda row: (row["question_id"], row["configuration_id"]))


def discover_benchmark():
    candidates = [ROOT/"artifacts/governance_evaluations/governance-evaluation-dataset-600-v1.0.json", ROOT/"data/processed/governance-evaluation-dataset-600-v1.0.json"]
    return next((path for path in candidates if path.exists()), None)


def run(args, dispatcher=None):
    benchmark = Path(args.benchmark) if args.benchmark else discover_benchmark()
    if benchmark is None: raise FileNotFoundError("Finalized 600-query benchmark not found; pass --benchmark PATH")
    dataset = load_benchmark(benchmark)
    if args.full and len(dataset.cases) != 600: raise ValueError("--full requires exactly 600 unique benchmark questions")
    limit = len(dataset.cases) if args.full else (args.limit if args.limit is not None else 2)
    selected = list(CONFIGURATIONS.values()) if args.configuration.lower() == "all" else [get_configuration(args.configuration)]
    output_dir=Path(args.output_dir); dispatcher=dispatcher or AblationDispatcher(); raw_payloads=[]; run_id=str(uuid.uuid4())
    for config in selected:
        path=output_dir/f"sentinel-ablation-{config.configuration_id.value.lower()}-raw-v1.0.json"
        existing=json.loads(path.read_text()) if args.resume and path.exists() else {"rows":[]}
        done={(r["question_id"],r["configuration_id"]) for r in existing["rows"]}; rows=list(existing["rows"])
        for case in dataset.cases[:limit]:
            key=(case.question_id,config.configuration_id.value)
            if key in done: continue
            began=time.perf_counter(); status="success"; error=None
            try:
                execution=dispatcher.execute(case.question,config,top_k=5,run_id=run_id,question_id=case.question_id)
                result=score(run_id,case,config.execution_mode,execution,(time.perf_counter()-began)*1000).model_dump(mode="json")
                # Disabled agents make their own outcome metrics inapplicable.
                if not config.enable_verification: result["verification_correct"]=None
                if not config.enable_compliance_enforcement: result["enforcement_correct"]=None
                if not config.enable_answer_generation: result["faithfulness"]=None; result["answer_relevancy"]=None
                audit=execution.audit
            except Exception as exc:
                status="failed"; error=f"{type(exc).__name__}: {exc}"; audit={}
                result={"run_id":run_id,"question_id":case.question_id,"category":case.category,"question":case.question,"execution_mode":config.execution_mode.value,"actual_answer":"","latency_ms":(time.perf_counter()-began)*1000}
            result.update({"configuration_id":config.configuration_id.value,"configuration_name":config.configuration_name,
                "enabled_components":list(config.enabled_components),"disabled_components":list(config.disabled_components),
                "retrieved_document_ids":result.get("retrieved_document_ids"),"retrieved_chunk_ids":result.get("retrieved_chunk_ids"),
                "reranker_invoked":audit.get("reranker_invoked",False),"verification_invoked":audit.get("verification_invoked",False),
                "compliance_invoked":audit.get("compliance_invoked",False),"answer_generation_invoked":audit.get("answer_generation_invoked",False),
                "processing_status":status,"error_message":error,"timestamp":datetime.now(timezone.utc).isoformat()})
            rows.append(result); atomic_json(path,{"configuration":config.configuration_id.value,"benchmark":dataset.filename,"rows":rows})
        raw_payload={"configuration":config.configuration_id.value,"benchmark":dataset.filename,"rows":rows}; raw_payloads.append(raw_payload)
    merged_rows=merge_rows(raw_payloads)
    merged={"status":"completed_with_errors" if any(r["processing_status"]!="success" for r in merged_rows) else "completed",
        "benchmark":dataset.filename,"query_count":len({r["question_id"] for r in merged_rows}),
        "configurations":[c.configuration_id.value for c in selected],"rows":merged_rows}
    atomic_json(output_dir/DEFAULT_MERGED,merged)
    groups={c.configuration_id.value:summarize([r for r in merged_rows if r["configuration_id"]==c.configuration_id.value]) for c in selected}
    for name, subset in (("rqem",("faithfulness","answer_relevancy")),("sgem",("uncertainty_correct","citations_complete","refusal_correct","enforcement_correct","verification_correct","trace_completeness","escalation_correct","handoff_success")),("retrieval",("precision_at_5","recall_at_5","reciprocal_rank","ndcg_at_5")),("latency",("latency_ms",))):
        atomic_json(output_dir/f"ablation-{name}-summary-v1.0.json",{key:{metric:value[metric] for metric in subset} for key,value in groups.items()})
    return merged


def parser():
    p=argparse.ArgumentParser(); p.add_argument("--configuration",default="A0"); p.add_argument("--limit",type=int); p.add_argument("--resume",action="store_true"); p.add_argument("--output-dir",default=str(DEFAULT_OUTPUT)); p.add_argument("--full",action="store_true"); p.add_argument("--benchmark"); return p

if __name__ == "__main__": run(parser().parse_args())
