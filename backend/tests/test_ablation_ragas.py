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
    values = {"input": str(source), "configuration": "all", "limit": None, "resume": False, "output_dir": str(output), "full": False, "enable_context_metrics": False,
              "batch_size": 10, "checkpoint_every": 10, "attempts": 3, "quiet": True}
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


def test_quote_hint_scalar_vector_and_progress_helpers():
    assert runner.contexts({"citations_returned": [{"quote_hint": " cited text "}]}) == ["cited text"]
    assert runner.normalize_score([.4]) == .4
    assert runner.normalize_score([.4, .5]) is None
    stats = runner.progress_stats([{"ragas_processing_status": "completed", "faithfulness": .5, "answer_relevancy": .6},
                                   {"ragas_processing_status": "failed", "faithfulness": None}], 4, 60, 2, 10)
    assert stats["completed"] == 2 and stats["rate"] == 2 and stats["eta_seconds"] == 60


def test_batch_retry_split_and_failed_row_isolated(tmp_path):
    rows = sample_rows(3)[:3]; source = tmp_path / "input.json"; source.write_text(json.dumps({"rows": rows}))
    class Evaluator:
        def __init__(self): self.sizes = []
        def score_batch(self, samples, names):
            self.sizes.append(len(samples))
            if len(samples) > 1: raise RuntimeError("batch fails")
            if samples[0]["question"].startswith("Question 1"): raise RuntimeError("row fails")
            return [{name: .75 for name in names}]
    evaluator = Evaluator()
    result = runner.run(args(source, tmp_path / "out", configuration="A0", limit=3, batch_size=3, attempts=1), evaluator=evaluator, sleep=lambda _: None)
    assert 3 in evaluator.sizes and 1 in evaluator.sizes
    assert sum(r["ragas_processing_status"] == "failed" for r in result["rows"]) == 1
    assert next(r for r in result["rows"] if r["question_id"] == "q1")["faithfulness"] is None


def test_resume_only_missing_metric_and_fingerprint_rejection(tmp_path):
    rows = sample_rows(1); source = tmp_path / "input.json"; source.write_text(json.dumps({"rows": rows}))
    output = tmp_path / "out"; runner.run(args(source, output, configuration="A0"), evaluator=fake, sleep=lambda _: None)
    checkpoint = output / "ablation-ragas-a0-scored-v1.0.json"
    payload = json.loads(checkpoint.read_text()); payload["rows"][0]["faithfulness"] = None; checkpoint.write_text(json.dumps(payload))
    calls = []
    def recording(sample, names): calls.append(names); return {n: .9 for n in names}
    runner.run(args(source, output, configuration="A0", resume=True), evaluator=recording, sleep=lambda _: None)
    assert calls == [["faithfulness"]]
    payload = json.loads(checkpoint.read_text()); payload["source_fingerprint"] = "wrong"; checkpoint.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        runner.run(args(source, output, configuration="A0", resume=True), evaluator=fake, sleep=lambda _: None)


def test_atomic_write_retries_permission_error_and_removes_temp(tmp_path, monkeypatch):
    real_replace, calls = runner.os.replace, []
    def flaky(source, destination):
        calls.append(1)
        if len(calls) < 3: raise PermissionError("locked")
        return real_replace(source, destination)
    monkeypatch.setattr(runner.os, "replace", flaky)
    target = tmp_path / "checkpoint.json"; runner.atomic_json(target, {"valid": True}, sleep=lambda _: None)
    assert json.loads(target.read_text()) == {"valid": True} and len(calls) == 3
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*.tmp"))


def test_keyboard_interrupt_leaves_checkpoint_and_interrupted_manifest(tmp_path):
    rows = sample_rows(1)[:1]; source = tmp_path / "input.json"; source.write_text(json.dumps({"rows": rows}))
    class Interrupting:
        def score_batch(self, samples, names): raise KeyboardInterrupt
    output = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        runner.run(args(source, output, configuration="A0"), evaluator=Interrupting(), sleep=lambda _: None)
    assert json.loads((output / "ablation-ragas-a0-scored-v1.0.json").read_text())["rows"][0]["question_id"] == "q0"
    assert json.loads((output / "ablation-ragas-run-manifest-v1.0.json").read_text())["status"] == "interrupted"
    assert not list(output.glob("*.tmp")) and not list(output.glob(".*.tmp"))


def test_ten_query_mock_pilot_batches_and_merged_uniqueness(tmp_path):
    rows = sample_rows(10); source = tmp_path / "input.json"; source.write_text(json.dumps({"rows": rows}))
    class BatchMock:
        def __init__(self): self.calls = 0
        def score_batch(self, samples, names):
            self.calls += 1
            return [{name: .5 for name in names} for _ in samples]
    mock = BatchMock()
    result = runner.run(args(source, tmp_path / "out", limit=10, batch_size=4), evaluator=mock, sleep=lambda _: None)
    keys = [(r["question_id"], r["configuration_id"]) for r in result["rows"]]
    assert len(keys) == len(set(keys)) == 70
    assert mock.calls == 21  # three batches for each of seven metric/configuration groups


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
