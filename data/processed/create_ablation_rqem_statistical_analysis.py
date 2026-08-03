#!/usr/bin/env python3
"""Create publication tables for question-ID-paired ablation RAGAS scores."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
LABELS = {"faithfulness": "Faithfulness", "answer_relevancy": "Answer Relevancy",
          "context_precision": "Context Precision", "context_recall": "Context Recall"}
COMPARISONS = tuple(("A0", f"A{i}") for i in range(1, 7))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def validate(rows: list[dict[str, Any]]) -> None:
    seen = set()
    for row in rows:
        key = (str(row["question_id"]), str(row["configuration_id"]))
        if key in seen: raise ValueError(f"Duplicate question/configuration pair: {key}")
        seen.add(key)


def describe(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if not n: return {"n": 0, "mean": None, "standard_deviation": None, "standard_error": None, "median": None, "minimum": None, "maximum": None, "ci95_low": None, "ci95_high": None}
    avg = mean(values); sd = stdev(values) if n > 1 else 0.0; se = sd / math.sqrt(n)
    critical = float(stats.t.ppf(.975, n - 1)) if stats is not None and n > 1 else 1.96
    return {"n": n, "mean": avg, "standard_deviation": sd, "standard_error": se, "median": median(values),
            "minimum": min(values), "maximum": max(values), "ci95_low": avg - critical * se, "ci95_high": avg + critical * se}


def holm(values: list[float]) -> list[float]:
    result = [0.0] * len(values); running = 0.0
    for rank, (position, value) in enumerate(sorted(enumerate(values), key=lambda item: item[1])):
        running = max(running, min(1.0, (len(values) - rank) * value)); result[position] = running
    return result


def paired(rows: list[dict[str, Any]], left: str, right: str, metric: str) -> dict[str, Any] | None:
    indexes = {}
    for config in (left, right):
        indexes[config] = {str(r["question_id"]): float(r[metric]) for r in rows if r["configuration_id"] == config and r.get(metric) is not None}
    ids = sorted(set(indexes[left]) & set(indexes[right])); a = [indexes[left][i] for i in ids]; b = [indexes[right][i] for i in ids]
    if not ids: return None
    differences = [x - y for x, y in zip(a, b)]; summary = describe(differences); n = len(ids)
    if stats is not None and n > 1:
        raw_t_p = float(stats.ttest_rel(a, b).pvalue)
        t_p = raw_t_p if math.isfinite(raw_t_p) else (1.0 if not any(differences) else 0.0)
    else: t_p = 1.0
    nonzero = [d for d in differences if d]
    if stats is not None and nonzero:
        wilcoxon_p = float(stats.wilcoxon(a, b).pvalue)
        ranks = stats.rankdata([abs(d) for d in nonzero]); positive = sum(rank for rank, d in zip(ranks, nonzero) if d > 0); negative = sum(rank for rank, d in zip(ranks, nonzero) if d < 0)
        rank_biserial = float((positive - negative) / (positive + negative))
    else: wilcoxon_p, rank_biserial = 1.0, 0.0
    return {"comparison": f"{left} vs {right}", "metric": metric, "n": n, "left_mean": mean(a), "right_mean": mean(b),
            "mean_difference": summary["mean"], "mean_difference_ci95_low": summary["ci95_low"], "mean_difference_ci95_high": summary["ci95_high"],
            "difference_standard_deviation": summary["standard_deviation"], "difference_standard_error": summary["standard_error"],
            "difference_median": summary["median"], "difference_minimum": summary["minimum"], "difference_maximum": summary["maximum"],
            "paired_t_p": t_p, "wilcoxon_p": wilcoxon_p,
            "cohens_d_paired": summary["mean"] / summary["standard_deviation"] if summary["standard_deviation"] else (0.0 if not summary["mean"] else None),
            "rank_biserial_correlation": rank_biserial, "question_ids": ids}


def create(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8")); rows = payload["rows"] if isinstance(payload, dict) else payload
    validate(rows); context_enabled = bool(payload.get("context_metrics_enabled")) if isinstance(payload, dict) else False
    enabled_metrics = ["faithfulness", "answer_relevancy"] + (["context_precision", "context_recall"] if context_enabled else [])
    descriptive = []
    for config in (f"A{i}" for i in range(7)):
        for metric in METRICS:
            values = [float(r[metric]) for r in rows if r["configuration_id"] == config and r.get(metric) is not None]
            descriptive.append({"configuration": config, "metric": metric, **describe(values)})
    tests = []
    for left, right in COMPARISONS:
        for metric in enabled_metrics:
            if metric == "faithfulness" and right == "A6": continue
            result = paired(rows, left, right, metric)
            if result: tests.append(result)
    for field, adjusted_field in (("paired_t_p", "paired_t_holm_p"), ("wilcoxon_p", "wilcoxon_holm_p")):
        for result, adjusted in zip(tests, holm([r[field] for r in tests])): result[adjusted_field] = adjusted
    compact = []
    for config in (f"A{i}" for i in range(7)):
        item = {"Configuration": config}
        for metric in METRICS:
            record = next(r for r in descriptive if r["configuration"] == config and r["metric"] == metric)
            item[LABELS[metric]] = record["mean"] if record["mean"] is not None else "N/A"; item[f"Applicable n - {LABELS[metric]}"] = record["n"]
        compact.append(item)
    out = Path(output_dir)
    write_csv(out / "table-ablation-rqem-descriptive-v1.0.csv", compact)
    write_csv(out / "table-ablation-rqem-statistical-significance-v1.0.csv", [{k: v for k, v in r.items() if k != "question_ids"} for r in tests])
    write_csv(out / "table-ablation-rqem-supplementary-statistics-v1.0.csv", descriptive)
    write_csv(out / "table-ablation-rqem-all-configurations-v1.0.csv", descriptive)
    (out / "table-ablation-rqem-statistical-significance-v1.0.json").write_text(json.dumps(tests, indent=2, allow_nan=False), encoding="utf-8")
    return {"descriptive": descriptive, "tests": tests, "compact": compact}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--input", required=True)
    p.add_argument("--output-dir", default=str(ROOT / "artifacts/governance_evaluations/publication_tables")); return p


if __name__ == "__main__":
    args = parser().parse_args(); create(args.input, args.output_dir)
