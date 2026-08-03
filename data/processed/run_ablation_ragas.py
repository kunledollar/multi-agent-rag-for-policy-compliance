#!/usr/bin/env python3
"""Production-safe, resumable RAGAS 0.1.21 scorer for Sentinel ablations.

Only RAGAS fields and RAGAS processing metadata are added to immutable source
rows.  The Sentinel configurations and all existing study metrics are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "artifacts/governance_evaluations/ablation-final-v1.0/sentinel-ablation-600-seven-configurations-v1.0.json"
DEFAULT_OUTPUT = ROOT / "artifacts/governance_evaluations/ablation-ragas"
CONFIGURATIONS = tuple(f"A{i}" for i in range(7))
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
RAGAS_METADATA = ("ragas_processing_status", "ragas_error_message", "ragas_attempt_count")
CONTEXT_FIELDS = ("retrieved_contexts", "contexts", "retrieved_chunks", "evidence", "retrieved_evidence")
REFERENCE_FIELDS = ("reference_answer", "ground_truth", "expected_answer")
RAGAS_VERSION, DATASETS_VERSION = "0.1.21", "2.19.2"
JUDGE_MODEL, EMBEDDING_MODEL = "gpt-4.1-mini", "text-embedding-3-large"


def atomic_json(path: Path, payload: Any, sleep: Callable[[float], None] = time.sleep) -> None:
    """Durably replace *path*, retrying transient Windows file locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary, path); return
            except PermissionError:
                if attempt == 7: raise
                sleep(0.05 * (2 ** attempt))
    finally:
        try: temporary.unlink(missing_ok=True)
        except OSError: pass


def _text(value: Any) -> str:
    if isinstance(value, str): return value.strip()
    if isinstance(value, dict):
        for key in ("text", "content", "page_content", "chunk_text", "evidence_text"):
            if value.get(key): return str(value[key]).strip()
    return ""


def contexts(row: dict[str, Any]) -> list[str]:
    for field in CONTEXT_FIELDS:
        value = row.get(field)
        if value is None: continue
        if isinstance(value, (str, dict)): value = [value]
        if isinstance(value, list) and (result := [text for item in value if (text := _text(item))]): return result
    citations = row.get("citations_returned")
    if isinstance(citations, list) and (result := [_text(c.get("quote_hint")) for c in citations if isinstance(c, dict) and _text(c.get("quote_hint"))]):
        return result
    return []


def reference(row: dict[str, Any]) -> str:
    return next((_text(row.get(field)) for field in REFERENCE_FIELDS if _text(row.get(field))), "")


def normalize_score(value: Any) -> float | None:
    """Extract scalar or one-element vector values without converting null to 0."""
    if hasattr(value, "tolist"): value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1: value = value[0]
    try: score = float(value)
    except (TypeError, ValueError): return None
    return score if math.isfinite(score) and 0 <= score <= 1 else None


def validate_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    for number, row in enumerate(rows, 1):
        try: key = (str(row["question_id"]), str(row["configuration_id"]).upper())
        except KeyError as exc: raise ValueError(f"Row {number} lacks required key {exc.args[0]}") from exc
        if key[1] not in CONFIGURATIONS: raise ValueError(f"Invalid configuration_id at row {number}: {key[1]}")
        if key in seen: raise ValueError(f"Duplicate question_id/configuration_id pair: {key}")
        seen.add(key)


def alignment_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [r for r in rows if str(r.get("configuration_id", "")).upper() != "A6"]
    missing_reference = [r["question_id"] for r in eligible if not reference(r)]
    missing_context = [r["question_id"] for r in eligible if not contexts(r)]
    return {"audit_version": "1.0", "criterion": "every non-A6 selected row has a textual reference and textual retrieved context",
            "eligible_rows": len(eligible), "aligned_rows": sum(bool(reference(r)) and bool(contexts(r)) for r in eligible),
            "missing_reference_question_ids": missing_reference, "missing_context_question_ids": missing_context,
            "passed": bool(eligible) and not missing_reference and not missing_context}


class RagasEvaluator:
    """One process-wide lifecycle for all OpenAI and metric resources."""
    instances_created = 0

    def __init__(self) -> None:
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        except ImportError as exc: raise RuntimeError("Install pinned RAGAS scoring dependencies") from exc
        if not os.getenv("OPENAI_API_KEY"): raise RuntimeError("OPENAI_API_KEY is required for RAGAS scoring")
        type(self).instances_created += 1
        self.chat = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
        self.embedding_client = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.llm = LangchainLLMWrapper(self.chat)
        self.embeddings = LangchainEmbeddingsWrapper(self.embedding_client)
        self.metrics = {"faithfulness": faithfulness, "answer_relevancy": answer_relevancy,
                        "context_precision": context_precision, "context_recall": context_recall}
        for name, metric in self.metrics.items():
            metric.llm = self.llm
            if name == "answer_relevancy": metric.embeddings = self.embeddings

    def score_batch(self, samples: list[dict[str, Any]], metric_names: list[str]) -> list[dict[str, Any]]:
        from datasets import Dataset, Features, Sequence, Value
        from ragas import evaluate
        features = Features({"question": Value("string"), "answer": Value("string"),
                             "contexts": Sequence(Value("string")), "ground_truth": Value("string")})
        dataset = Dataset.from_dict({k: [s[k] for s in samples] for k in features}, features=features)
        result = evaluate(dataset, metrics=[self.metrics[n] for n in metric_names], raise_exceptions=False)
        output = []
        for index in range(len(samples)):
            output.append({name: normalize_score(result[name][index] if hasattr(result[name], "__getitem__") else result[name]) for name in metric_names})
        return output

    def close(self) -> None:
        for resource in (getattr(self.chat, "client", None), getattr(self.embedding_client, "client", None), self.chat, self.embedding_client):
            close = getattr(resource, "close", None)
            if callable(close):
                try: close()
                except Exception: pass


def _payload_rows(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows): raise ValueError("Input must be a JSON row list or an object containing a rows list")
    return rows


def _metric_names(row: dict[str, Any], context_enabled: bool) -> list[str]:
    config = str(row["configuration_id"]).upper()
    question, answer, evidence, truth = _text(row.get("question")), _text(row.get("actual_answer")), contexts(row), reference(row)
    names = []
    if config != "A6" and question and answer and evidence: names.append("faithfulness")
    if question and answer: names.append("answer_relevancy")
    if context_enabled and config != "A6" and question and answer and evidence and truth: names += ["context_precision", "context_recall"]
    return names


def _sample(row: dict[str, Any]) -> dict[str, Any]:
    return {"question": _text(row.get("question")), "answer": _text(row.get("actual_answer")),
            "contexts": contexts(row), "ground_truth": reference(row)}


def _call(evaluator: Any, samples: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    if hasattr(evaluator, "score_batch"): return evaluator.score_batch(samples, names)
    return [evaluator(sample, names) for sample in samples]  # legacy/mock compatibility


def _score_resilient(items: list[tuple[dict[str, Any], dict[str, Any]]], names: list[str], evaluator: Any,
                     attempts: int, sleep: Callable[[float], None], rng: random.Random) -> list[tuple[dict[str, Any], dict[str, Any] | None, int, str | None]]:
    """Retry a batch, recursively split it, and isolate permanently failed rows."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            values = _call(evaluator, [sample for _, sample in items], names)
            if len(values) != len(items): raise ValueError("Evaluator result count does not match batch")
            return [(row, value, attempt, None) for (row, _), value in zip(items, values)]
        except Exception as exc:
            last = exc
            if attempt < attempts: sleep((2 ** (attempt - 1)) + rng.random())
    if len(items) > 1:
        middle = len(items) // 2
        return (_score_resilient(items[:middle], names, evaluator, attempts, sleep, rng) +
                _score_resilient(items[middle:], names, evaluator, attempts, sleep, rng))
    return [(items[0][0], None, attempts, f"{type(last).__name__}: {last}")]


def _fingerprint(path: Path, digest: str, configs: Iterable[str], context_enabled: bool) -> str:
    policy = {"absolute_input": str(path.resolve()), "input_sha256": digest, "configurations": list(configs),
              "ragas_version": RAGAS_VERSION, "datasets_version": DATASETS_VERSION, "judge_model": JUDGE_MODEL,
              "embedding_model": EMBEDDING_MODEL, "context_metrics_enabled": context_enabled,
              "faithfulness_policy": "non-A6 with usable question, answer, and evidence"}
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


def progress_stats(rows: list[dict[str, Any]], total: int, elapsed: float, overall_done: int, overall_total: int) -> dict[str, Any]:
    completed = sum(r.get("ragas_processing_status") in {"completed", "failed", "not_applicable"} for r in rows)
    rate = completed / elapsed * 60 if elapsed > 0 else 0.0
    return {"completed": completed, "total": total, "faithfulness_scored": sum(r.get("faithfulness") is not None for r in rows),
            "answer_relevancy_scored": sum(r.get("answer_relevancy") is not None for r in rows),
            "null_faithfulness": sum(r.get("faithfulness") is None for r in rows),
            "failures": sum(r.get("ragas_processing_status") == "failed" for r in rows), "rate": rate,
            "eta_seconds": ((total - completed) / (rate / 60)) if rate else None, "overall_done": overall_done, "overall_total": overall_total}


def _duration(seconds: float | None) -> str:
    if seconds is None: return "--:--:--"
    return time.strftime("%H:%M:%S", time.gmtime(max(0, seconds)))


def run(args: argparse.Namespace, evaluator: Any = None, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    started_wall, started = time.monotonic(), datetime.now(timezone.utc).isoformat()
    input_path = Path(args.input); raw = input_path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    source_rows = _payload_rows(json.loads(raw)); validate_rows(source_rows)
    configs = CONFIGURATIONS if args.configuration.lower() == "all" else (args.configuration.upper(),)
    per_config = {c: sorted((r for r in source_rows if str(r["configuration_id"]).upper() == c), key=lambda r: str(r["question_id"])) for c in configs}
    if args.full and (bad := {c: len(v) for c, v in per_config.items() if len(v) != 600}): raise ValueError(f"--full requires exactly 600 input rows per selected configuration: {bad}")
    limit = 600 if args.full else (args.limit if args.limit is not None else 2)
    if limit < 0: raise ValueError("--limit must be non-negative")
    batch_size = getattr(args, "batch_size", 10); checkpoint_every = getattr(args, "checkpoint_every", 10); attempts = getattr(args, "attempts", 3)
    if not 1 <= batch_size <= 100: raise ValueError("--batch-size must be between 1 and 100")
    if checkpoint_every < 1 or attempts < 1: raise ValueError("checkpoint interval and attempts must be positive")
    output_dir = Path(args.output_dir); selected = [r for c in configs for r in per_config[c][:limit]]
    audit = alignment_audit(selected); atomic_json(output_dir / "ablation-ragas-alignment-audit-v1.0.json", audit, sleep)
    if args.enable_context_metrics and not audit["passed"]: raise ValueError("Context metrics requested, but reference-evidence alignment audit failed")
    fingerprint = _fingerprint(input_path, digest, configs, args.enable_context_metrics)
    manifest_path = output_dir / "ablation-ragas-run-manifest-v1.0.json"
    resume_count = 0
    if args.resume and manifest_path.exists(): resume_count = int(json.loads(manifest_path.read_text()).get("resume_count", 0)) + 1
    manifest = {"source_input": str(input_path.resolve()), "input_sha256": digest, "selected_configurations": list(configs),
                "evaluator_versions": {"ragas": RAGAS_VERSION, "datasets": DATASETS_VERSION}, "judge_model": JUDGE_MODEL,
                "embedding_model": EMBEDDING_MODEL, "batch_size": batch_size, "checkpoint_interval": checkpoint_every,
                "start_time": started, "end_time": None, "elapsed_seconds": None, "completed_rows": 0,
                "scored_metric_counts": {}, "null_metric_counts": {}, "failed_rows": 0, "resume_count": resume_count,
                "status": "running", "source_fingerprint": fingerprint}
    atomic_json(manifest_path, manifest, sleep)
    owned = evaluator is None
    if evaluator is None: evaluator = RagasEvaluator()
    all_output: list[dict[str, Any]] = []; interrupted = False
    try:
        for config in configs:
            source = per_config[config][:limit]; path = output_dir / f"ablation-ragas-{config.lower()}-scored-v1.0.json"
            has_checkpoint = args.resume and path.exists()
            checkpoint = json.loads(path.read_text(encoding="utf-8")) if has_checkpoint else {"rows": []}
            if has_checkpoint and checkpoint.get("source_fingerprint") != fingerprint: raise ValueError("Resume checkpoint source fingerprint mismatch")
            existing = _payload_rows(checkpoint); validate_rows(existing)
            source_index = {(str(r["question_id"]), config): r for r in source}
            dynamic = set(METRICS + RAGAS_METADATA)
            for prior in existing:
                key = (str(prior["question_id"]), str(prior["configuration_id"]).upper())
                if key not in source_index: raise ValueError(f"Resume checkpoint row is not in selected input: {key}")
                if any(prior.get(k) != v for k, v in source_index[key].items() if k not in dynamic): raise ValueError(f"Resume checkpoint does not align with input row: {key}")
            by_key = {(str(r["question_id"]), config): dict(r) for r in existing}
            rows = []
            for original in source:
                key = (str(original["question_id"]), config); out = by_key.get(key, dict(original))
                for metric in METRICS: out.setdefault(metric, None)
                rows.append(out)
            newly = 0
            def save() -> None:
                atomic_json(path, {"configuration": config, "ragas_version": RAGAS_VERSION, "source_fingerprint": fingerprint, "rows": rows}, sleep)
            groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
            for row in rows:
                applicable = _metric_names(row, args.enable_context_metrics)
                missing = tuple(n for n in applicable if row.get(n) is None)
                if missing: groups.setdefault(missing, []).append(row)
                elif not applicable: row.update(ragas_processing_status="not_applicable", ragas_error_message=None, ragas_attempt_count=0)
            try:
                for names, pending in groups.items():
                    for offset in range(0, len(pending), batch_size):
                        batch = pending[offset:offset + batch_size]
                        results = _score_resilient([(r, _sample(r)) for r in batch], list(names), evaluator, attempts, sleep, random.Random(0))
                        for row, scores, count, error in results:
                            row["ragas_attempt_count"] = int(row.get("ragas_attempt_count") or 0) + count
                            row["ragas_error_message"] = error
                            row["ragas_processing_status"] = "failed" if error else "completed"
                            if scores is not None:
                                for name in names: row[name] = normalize_score(scores.get(name))
                            newly += 1
                            if newly % checkpoint_every == 0: save()
                        save()  # every completed batch, including failed isolated rows
                        if not getattr(args, "quiet", False):
                            current_done = sum(r.get("ragas_processing_status") in {"completed", "failed", "not_applicable"} for r in rows)
                            s = progress_stats(rows, len(source), time.monotonic() - started_wall, len(all_output) + current_done, len(selected))
                            print(f"Configuration: {config}\nCompleted: {s['completed']}/{s['total']}\nFaithfulness scored: {s['faithfulness_scored']}\nAnswer Relevancy scored: {s['answer_relevancy_scored']}\nNull Faithfulness: {s['null_faithfulness']}\nFailures: {s['failures']}\nElapsed: {_duration(time.monotonic()-started_wall)}\nRate: {s['rate']:.2f} rows/min\nETA: {_duration(s['eta_seconds'])}\nOverall: {s['overall_done']}/{s['overall_total']}")
            except KeyboardInterrupt:
                save(); interrupted = True; raise
            all_output.extend(rows)
        validate_rows(all_output)
        merged = {"ragas_version": RAGAS_VERSION, "judge_model": JUDGE_MODEL, "embedding_model": EMBEDDING_MODEL,
                  "context_metrics_enabled": bool(args.enable_context_metrics), "source_fingerprint": fingerprint,
                  "rows": sorted(all_output, key=lambda r: (str(r["question_id"]), str(r["configuration_id"]))) }
        atomic_json(output_dir / "ablation-ragas-4200-merged-v1.0.json", merged, sleep)
        summary = {c: {m: {"mean": (sum(v) / len(v) if (v := [r.get(m) for r in all_output if str(r["configuration_id"]).upper() == c and r.get(m) is not None]) else None), "n": len(v)} for m in METRICS} for c in configs}
        atomic_json(output_dir / "ablation-rqem-summary-v1.0.json", summary, sleep)
        manifest["status"] = "completed"; return merged
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        print(f"Interrupted. Resume with:\n{sys.executable} {Path(__file__)} --input {args.input!s} --configuration {args.configuration} --output-dir {args.output_dir!s} --resume", file=sys.stderr)
        raise
    except Exception:
        manifest["status"] = "failed"; raise
    finally:
        if owned:
            try: evaluator.close()
            except Exception: pass
        manifest["end_time"] = datetime.now(timezone.utc).isoformat(); manifest["elapsed_seconds"] = round(time.monotonic() - started_wall, 3)
        manifest["completed_rows"] = len(all_output); manifest["failed_rows"] = sum(r.get("ragas_processing_status") == "failed" for r in all_output)
        manifest["scored_metric_counts"] = {m: sum(r.get(m) is not None for r in all_output) for m in METRICS}
        manifest["null_metric_counts"] = {m: sum(r.get(m) is None for r in all_output) for m in METRICS}
        atomic_json(manifest_path, manifest, sleep)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(DEFAULT_INPUT)); p.add_argument("--configuration", choices=(*CONFIGURATIONS, "all"), default="A0")
    p.add_argument("--limit", type=int); p.add_argument("--resume", action="store_true"); p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--full", action="store_true"); p.add_argument("--enable-context-metrics", action="store_true")
    p.add_argument("--batch-size", type=int, default=10); p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--attempts", type=int, default=3); p.add_argument("--quiet", action="store_true")
    return p


def main() -> int:
    try: run(parser().parse_args()); return 0
    except KeyboardInterrupt: return 130


if __name__ == "__main__": raise SystemExit(main())
