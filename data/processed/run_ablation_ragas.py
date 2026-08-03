#!/usr/bin/env python3
"""Score an existing Sentinel ablation export with RAGAS (and nothing else).

The runner deliberately treats the input rows as immutable: output rows are copies
with four RAGAS fields added/replaced.  It never invokes the Sentinel application.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "artifacts/governance_evaluations/ablation-final-v1.0/sentinel-ablation-600-seven-configurations-v1.0.json"
DEFAULT_OUTPUT = ROOT / "artifacts/governance_evaluations/ablation-ragas"
CONFIGURATIONS = tuple(f"A{i}" for i in range(7))
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
CONTEXT_FIELDS = ("retrieved_contexts", "contexts", "retrieved_chunks", "evidence", "retrieved_evidence")
REFERENCE_FIELDS = ("reference_answer", "ground_truth", "expected_answer")
MAX_ATTEMPTS = 3


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _text(value: Any) -> str:
    if isinstance(value, str): return value.strip()
    if isinstance(value, dict):
        for key in ("text", "content", "page_content", "chunk_text", "evidence_text"):
            if value.get(key): return str(value[key]).strip()
    return ""


def contexts(row: dict[str, Any]) -> list[str]:
    """Return non-empty context strings from known ablation evidence shapes."""
    for field in CONTEXT_FIELDS:
        value = row.get(field)
        if value is None: continue
        if isinstance(value, (str, dict)): value = [value]
        if isinstance(value, list):
            result = [text for item in value if (text := _text(item))]
            if result: return result
    return []


def reference(row: dict[str, Any]) -> str:
    return next((_text(row.get(field)) for field in REFERENCE_FIELDS if _text(row.get(field))), "")


def normalize_score(value: Any) -> float | None:
    """Use finalized-study null handling: finite numeric [0, 1], otherwise null."""
    try: score = float(value)
    except (TypeError, ValueError): return None
    return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None


def validate_rows(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    for number, row in enumerate(rows, 1):
        try: key = (str(row["question_id"]), str(row["configuration_id"]).upper())
        except KeyError as exc: raise ValueError(f"Row {number} lacks required key {exc.args[0]}") from exc
        if key[1] not in CONFIGURATIONS: raise ValueError(f"Invalid configuration_id at row {number}: {key[1]}")
        if key in seen: raise ValueError(f"Duplicate question_id/configuration_id pair: {key}")
        seen.add(key)


def alignment_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Conservatively audit whether references can be aligned to retrieved evidence.

    RAGAS context metrics are reference-based.  Consequently every row selected for
    those metrics must have both a reference and textual evidence.  The audit does
    not pretend that lexical overlap proves correctness; it records structural
    alignment only and fails closed if any selected row is not alignable.
    """
    eligible = [r for r in rows if str(r.get("configuration_id", "")).upper() != "A6"]
    missing_reference = [r["question_id"] for r in eligible if not reference(r)]
    missing_context = [r["question_id"] for r in eligible if not contexts(r)]
    passed = bool(eligible) and not missing_reference and not missing_context
    return {"audit_version": "1.0", "criterion": "every non-A6 selected row has a textual reference and textual retrieved context",
            "eligible_rows": len(eligible), "aligned_rows": sum(bool(reference(r)) and bool(contexts(r)) for r in eligible),
            "missing_reference_question_ids": missing_reference, "missing_context_question_ids": missing_context,
            "passed": passed}


def ragas_evaluator(sample: dict[str, Any], metric_names: list[str]) -> dict[str, Any]:
    """Evaluate one checkpointable sample using the pinned legacy RAGAS API."""
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise RuntimeError("RAGAS scoring dependencies are unavailable; install backend/requirements.txt and langchain-openai/datasets") from exc
    if not os.getenv("OPENAI_API_KEY"): raise RuntimeError("OPENAI_API_KEY is required for RAGAS scoring")
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-large"))
    available = {"faithfulness": faithfulness, "answer_relevancy": answer_relevancy,
                 "context_precision": context_precision, "context_recall": context_recall}
    chosen = []
    for name in metric_names:
        metric = available[name]
        metric.llm = llm
        if name == "answer_relevancy": metric.embeddings = embeddings
        chosen.append(metric)
    dataset = Dataset.from_dict({key: [value] for key, value in sample.items()})
    result = evaluate(dataset, metrics=chosen, raise_exceptions=False)
    return {name: result[name][0] for name in metric_names}


def _score_with_retry(sample: dict[str, Any], names: list[str], evaluator: Callable, sleep: Callable = time.sleep) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try: return evaluator(sample, names)
        except Exception as exc:
            error = exc
            if attempt + 1 < MAX_ATTEMPTS: sleep(2 ** attempt)
    raise RuntimeError(f"RAGAS failed after {MAX_ATTEMPTS} attempts: {error}") from error


def _payload_rows(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows): raise ValueError("Input must be a JSON row list or an object containing a rows list")
    return rows


def run(args: argparse.Namespace, evaluator: Callable = ragas_evaluator, sleep: Callable = time.sleep) -> dict[str, Any]:
    source_payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    source_rows = _payload_rows(source_payload); validate_rows(source_rows)
    selected_configs = CONFIGURATIONS if args.configuration.lower() == "all" else (args.configuration.upper(),)
    selected = [r for r in source_rows if str(r["configuration_id"]).upper() in selected_configs]
    per_config = {c: [r for r in selected if str(r["configuration_id"]).upper() == c] for c in selected_configs}
    if args.full:
        bad = {c: len(rows) for c, rows in per_config.items() if len(rows) != 600}
        if bad: raise ValueError(f"--full requires exactly 600 input rows per selected configuration: {bad}")
        limit = 600
    else: limit = args.limit if args.limit is not None else 2
    if limit < 0: raise ValueError("--limit must be non-negative")
    output_dir = Path(args.output_dir)
    audit = alignment_audit([r for rows in per_config.values() for r in rows[:limit]])
    atomic_json(output_dir / "ablation-ragas-alignment-audit-v1.0.json", audit)
    if args.enable_context_metrics and not audit["passed"]:
        raise ValueError("Context metrics requested, but reference-evidence alignment audit failed")

    all_output: list[dict[str, Any]] = []
    for config, input_rows in per_config.items():
        path = output_dir / f"ablation-ragas-{config.lower()}-scored-v1.0.json"
        existing_rows = _payload_rows(json.loads(path.read_text(encoding="utf-8"))) if args.resume and path.exists() else []
        validate_rows(existing_rows)
        source_index = {(str(r["question_id"]), config): r for r in input_rows[:limit]}
        metric_keys = set(METRICS)
        for prior in existing_rows:
            key = (str(prior["question_id"]), str(prior["configuration_id"]).upper())
            if key not in source_index:
                raise ValueError(f"Resume checkpoint row is not in the selected input: {key}")
            # A checkpoint is safe to resume only when every non-RAGAS source field
            # still aligns exactly. This prevents silently mixing study versions.
            expected = source_index[key]
            if any(prior.get(field) != value for field, value in expected.items() if field not in metric_keys):
                raise ValueError(f"Resume checkpoint does not align with input row: {key}")
        done = {(str(r["question_id"]), str(r["configuration_id"]).upper()) for r in existing_rows}
        rows = list(existing_rows)
        for original in input_rows[:limit]:
            key = (str(original["question_id"]), config)
            if key in done: continue
            output = dict(original)
            output.update({metric: None for metric in METRICS})
            question, answer, evidence, truth = _text(original.get("question")), _text(original.get("actual_answer")), contexts(original), reference(original)
            names: list[str] = []
            if config != "A6" and question and answer and evidence: names.append("faithfulness")
            if question and answer: names.append("answer_relevancy")
            if args.enable_context_metrics and config != "A6" and question and answer and evidence and truth:
                names.extend(("context_precision", "context_recall"))
            if names:
                sample = {"question": question, "answer": answer, "contexts": evidence, "ground_truth": truth}
                scores = _score_with_retry(sample, names, evaluator, sleep)
                for name in names: output[name] = normalize_score(scores.get(name))
            rows.append(output); done.add(key)
            atomic_json(path, {"configuration": config, "ragas_version": "0.1.21", "rows": rows})
        all_output.extend(rows)
    validate_rows(all_output)
    merged = {"ragas_version": "0.1.21", "judge_model": "gpt-4.1-mini", "embedding_model": "text-embedding-3-large",
              "context_metrics_enabled": bool(args.enable_context_metrics), "rows": sorted(all_output, key=lambda r: (str(r["question_id"]), str(r["configuration_id"]))) }
    atomic_json(output_dir / "ablation-ragas-4200-merged-v1.0.json", merged)
    summary = {}
    for config in selected_configs:
        group = [r for r in all_output if str(r["configuration_id"]).upper() == config]
        summary[config] = {m: {"mean": (sum(v) / len(v) if (v := [r.get(m) for r in group if r.get(m) is not None]) else None), "n": len(v)} for m in METRICS}
    atomic_json(output_dir / "ablation-rqem-summary-v1.0.json", summary)
    return merged


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=str(DEFAULT_INPUT)); p.add_argument("--configuration", choices=(*CONFIGURATIONS, "all"), default="A0")
    p.add_argument("--limit", type=int); p.add_argument("--resume", action="store_true"); p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--full", action="store_true"); p.add_argument("--enable-context-metrics", action="store_true")
    return p


if __name__ == "__main__": run(parser().parse_args())
