# Sentinel ablation study

This framework estimates the contribution of individual Sentinel stages while holding the benchmark, corpus, models, retrieval depth, and evaluators fixed. It does not alter production defaults or existing API modes.

## Configurations

| ID | Definition |
|---|---|
| A0 | Existing Full Sentinel, all stages enabled |
| A1 | Reranker bypassed; original retrieval ordering retained |
| A2 | Verification bypassed |
| A3 | Compliance enforcement bypassed; downstream receives an explicit neutral `unknown` representation |
| A4 | Specialized generation bypassed; returns the existing upstream `summary_reasoning` verbatim, without creating an answer |
| A5 | Existing `rag_only` dispatcher path |
| A6 | Existing `llm_only` dispatcher path, without retrieval |

The pass-through verification agent and behavior-preserving stable reranker reflect the current repository implementations. Conclusions about those stages should acknowledge this limitation.

## Running

The runner intentionally defaults to two questions. Supply the finalized benchmark if it is not in a discovered standard location:

```bash
python data/processed/run_ablation_study.py --configuration all --limit 2 --benchmark PATH.json
python data/processed/run_ablation_study.py --configuration A2 --limit 20 --resume --benchmark PATH.json
```

The full study is opt-in and validates that the input has exactly 600 unique questions:

```bash
python data/processed/run_ablation_study.py --configuration all --full --resume --benchmark PATH.json
```

This can make up to 4,200 model-backed executions, plus evaluator calls; estimate cost and rate limits from the configured chat/embedding/RAGAS providers before starting. Temperature is fixed by the existing execution implementations, but remote model nondeterminism can remain.

Checkpoints are atomic per-configuration JSON files. `--resume` skips existing `(question_id, configuration_id)` pairs; without it, that configuration starts a new artifact. The merged artifact and RQEM, SGEM, retrieval, and latency summaries are written under `artifacts/governance_evaluations/ablation/`.

Create publication tables with:

```bash
python data/processed/create_ablation_statistical_analysis.py --input artifacts/governance_evaluations/ablation/sentinel-ablation-600-seven-configurations-v1.0.json
```

Continuous outcomes use paired t-tests, Wilcoxon signed-rank tests, paired Cohen's *d*, and mean-difference confidence intervals. Binary outcomes use Wilson intervals, exact McNemar tests, matched-pair odds ratios, and absolute rate differences. Holm correction is applied across reported primary comparisons. Every pair is an inner join on `question_id`; null observations are excluded, never replaced with zero. No comparison is emitted without applicable paired observations.

Interpret results as within-benchmark associations, not universal causal effects. A4's upstream text is not a generated final answer, A6 has no retrieval metrics, and disabled-stage outcome metrics are N/A. Inspect failures, missingness, multiplicity, API version metadata, and practical effect sizes alongside p-values.
