"""Offline regression and smoke tests for the ablation-only RAGAS workflow."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


runner = load("run_ablation_ragas", "data/processed/run_ablation_ragas.py")
analysis = load("create_ablation_rqem", "data/processed/create_ablation_rqem_statistical_analysis.py")


def sample_rows(count=2):
    return [{"question_id": f"q{q}", "configuration_id": f"A{c}", "question": f"Question {q}?",
             "actual_answer": f"Answer {q}", "reference_answer": f"Reference {q}", "retrieved_contexts": [f"Evidence {q}"],
             "latency_ms": c * 10, "governance_score": .75} for c in range(7) for q in range(count)]


def args(source, output, **changes):
    values = {"input": str(source), "configuration": "all", "limit": None, "resume": False, "output_dir": str(output), "full": False, "enable_context_metrics": False}
    values.update(changes); return argparse.Namespace(**values)


def fake(sample, names):
    return {name: {"faithfulness": .8, "answer_relevancy": .7, "context_precision": .6, "context_recall": .5}[name] for name in names}


def test_two_query_smoke_all_configurations_and_null_rules(tmp_path):
    rows = sample_rows(); rows[0]["retrieved_contexts"] = []
    source = tmp_path / "input.json"; source.write_text(json.dumps({"rows": rows}))
    result = runner.run(args(source, tmp_path / "out"), evaluator=fake, sleep=lambda _: None)
    assert len(result["rows"]) == 14
    assert {(r["question_id"], r["configuration_id"]) for r in result["rows"]} == {(r["question_id"], r["configuration_id"]) for r in rows}
    assert all(r["faithfulness"] is None for r in result["rows"] if r["configuration_id"] == "A6")
    assert next(r for r in result["rows"] if r["configuration_id"] == "A0" and r["question_id"] == "q0")["faithfulness"] is None
    assert all(r["context_precision"] is None and r["context_recall"] is None for r in result["rows"])
    assert all(r["latency_ms"] == next(x["latency_ms"] for x in rows if x["question_id"] == r["question_id"] and x["configuration_id"] == r["configuration_id"]) for r in result["rows"])


def test_resume_does_not_rescore_and_duplicates_rejected(tmp_path):
    rows = sample_rows(); source = tmp_path / "input.json"; source.write_text(json.dumps({"rows": rows}))
    output = tmp_path / "out"; runner.run(args(source, output), evaluator=fake, sleep=lambda _: None)
    calls = []
    runner.run(args(source, output, resume=True), evaluator=lambda sample, names: calls.append(1), sleep=lambda _: None)
    assert calls == []
    rows.append(dict(rows[0])); source.write_text(json.dumps({"rows": rows}))
    with pytest.raises(ValueError, match="Duplicate"): runner.run(args(source, output), evaluator=fake)


def test_context_metrics_require_passing_alignment_audit(tmp_path):
    rows = sample_rows(); rows[0].pop("reference_answer")
    source = tmp_path / "input.json"; source.write_text(json.dumps({"rows": rows}))
    with pytest.raises(ValueError, match="audit failed"):
        runner.run(args(source, tmp_path / "out", enable_context_metrics=True), evaluator=fake)
    rows[0]["reference_answer"] = "restored"; source.write_text(json.dumps({"rows": rows}))
    result = runner.run(args(source, tmp_path / "enabled", enable_context_metrics=True), evaluator=fake)
    assert all(r["context_precision"] == .6 for r in result["rows"] if r["configuration_id"] != "A6")


def test_deterministic_paired_statistics_and_a6_exclusion(tmp_path):
    rows = []
    for q, a0, a1 in (("q1", .9, .6), ("q2", .7, .5), ("q3", .8, .4)):
        rows += [{"question_id": q, "configuration_id": "A0", "faithfulness": a0, "answer_relevancy": a0},
                 {"question_id": q, "configuration_id": "A1", "faithfulness": a1, "answer_relevancy": a1}]
    for c in range(2, 7):
        for q in ("q1", "q2", "q3"): rows.append({"question_id": q, "configuration_id": f"A{c}", "faithfulness": None if c == 6 else .5, "answer_relevancy": .5})
    source = tmp_path / "scores.json"; source.write_text(json.dumps({"context_metrics_enabled": False, "rows": rows}))
    result = analysis.create(source, tmp_path / "tables")
    comparison = next(r for r in result["tests"] if r["comparison"] == "A0 vs A1" and r["metric"] == "faithfulness")
    assert comparison["n"] == 3 and comparison["mean_difference"] == pytest.approx(.3)
    assert not any(r["comparison"] == "A0 vs A6" and r["metric"] == "faithfulness" for r in result["tests"])
    assert not any(r["metric"].startswith("context_") for r in result["tests"])
    assert (tmp_path / "tables/table-ablation-rqem-descriptive-v1.0.csv").exists()
