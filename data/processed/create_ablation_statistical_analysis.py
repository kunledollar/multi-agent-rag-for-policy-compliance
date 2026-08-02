#!/usr/bin/env python3
"""Create descriptive and question-ID-paired ablation publication tables."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"backend"))
from app.evaluation.metrics import METRICS, summarize
from app.evaluation.statistics import binary, continuous, holm_adjust

BINARY={"uncertainty_correct","refusal_correct","enforcement_correct","escalation_correct"}
COMPARISONS=[("A0",f"A{i}") for i in range(1,7)]

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({key for row in rows for key in row}) if rows else []
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def create(input_path,output_dir):
    payload=json.loads(Path(input_path).read_text()); rows=payload["rows"]
    descriptive=[]
    for config in sorted({r["configuration_id"] for r in rows}):
        summary=summarize([r for r in rows if r["configuration_id"]==config])
        descriptive.extend({"configuration":config,"metric":m,"value":summary[m]["mean"],"n":summary[m]["n"]} for m in METRICS)
    tests=[]
    for left,right in COMPARISONS:
        for metric in METRICS:
            result=(binary if metric in BINARY else continuous)(rows,left,right,metric)
            if result: result.update({"comparison":f"{left} vs {right}","test_type":"binary" if metric in BINARY else "continuous"}); tests.append(result)
    p_fields=[("mcnemar_p" if r["test_type"]=="binary" else "paired_t_p") for r in tests]
    adjusted=holm_adjust([r[field] for r,field in zip(tests,p_fields)]) if tests else []
    for result,value in zip(tests,adjusted): result["holm_adjusted_p"]=value
    compact=[]
    names={"faithfulness":"Faithfulness","answer_relevancy":"Answer Relevance","uncertainty_correct":"Uncertainty Signaling","citations_complete":"Citation Completeness","refusal_correct":"Risk-Aware Refusal","trace_completeness":"Decision Trace Completeness","escalation_correct":"Escalation Accuracy","handoff_success":"Agent Handoff Success","latency_ms":"Mean Latency"}
    for config in sorted({r["configuration_id"] for r in rows}):
        summary=summarize([r for r in rows if r["configuration_id"]==config]); compact.append({"Configuration":config,**{label:(summary[m]["mean"] if summary[m]["mean"] is not None else "N/A") for m,label in names.items()}})
    out=Path(output_dir); write_csv(out/"table-ablation-descriptive-v1.0.csv",compact)
    flattened=[{k:v for k,v in row.items() if k!="question_ids"} for row in tests]
    write_csv(out/"table-ablation-statistical-significance-v1.0.csv",flattened); write_csv(out/"table-ablation-supplementary-statistics-v1.0.csv",descriptive)
    (out/"table-ablation-statistical-significance-v1.0.json").write_text(json.dumps(tests,indent=2,allow_nan=False))
    return tests
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output-dir",default=str(ROOT/"artifacts/governance_evaluations/publication_tables")); a=p.parse_args(); create(a.input,a.output_dir)
