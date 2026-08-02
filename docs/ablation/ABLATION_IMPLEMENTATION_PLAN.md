# Sentinel Ablation Implementation Plan

## Repository analysis and exact files

The production Full Sentinel entry point is `backend/app/rag/graph.py::run_sentinel_graph`, reached by `ExecutionDispatcher.execute`. Retrieval is implemented by `RetrieverAgent`; the checked-in system has no separate reranker, so the ablation layer will add an explicit stable score reranker whose enabled result preserves the existing retrieval order. Verification is represented by the currently pass-through `FactCheckAgent`. Compliance is `ComplianceAgent.run`, reasoning is `ReasoningAgent.run`, and specialized generation is `AnswerGenerationAgent.run`. Existing `full_sentinel`, `rag_only`, and `llm_only` routes are in `backend/app/evaluation/dispatcher.py`. Benchmark loading, execution, governance scoring, aggregation/export, and optional per-request RAGAS are respectively in `benchmark_loader.py`, `runner.py`, `scoring.py`, `aggregation.py`/`exporter.py`, and `rag/graph.py`.

Files to create or modify:

- `backend/app/evaluation/ablation.py`: immutable typed A0–A6 definitions and ablation execution.
- `backend/app/evaluation/metrics.py`: null-preserving per-configuration summaries.
- `backend/app/evaluation/statistics.py`: question-ID-aligned paired inference.
- `data/processed/run_ablation_study.py`: safe, resumable CLI and artifacts.
- `data/processed/create_ablation_statistical_analysis.py`: publication outputs.
- `backend/tests/test_ablation.py`: behavioral, instrumentation, resume, uniqueness, null, and statistics tests.
- `docs/ablation/README.md`: reproducibility guide.

Production dispatcher, routes, modes, graph, and API defaults will not be changed.

## Existing call graph

`EvaluationRunner -> ExecutionDispatcher -> {run_sentinel_graph, existing RAG_ONLY path, existing LLM_ONLY path}`. Full Sentinel calls `RetrieverAgent -> ComplianceAgent -> ReasoningAgent -> AnswerGenerationAgent`; RAG-only calls `RetrieverAgent -> AnswerGenerationAgent`; LLM-only calls the direct chat model. `score -> governance/retrieval metrics -> aggregate/export`.

## Configuration model and behavior

A frozen `AblationConfiguration` selected by an `AblationId` enum carries all required enable flags and execution mode. A0 invokes the unmodified existing Full Sentinel dispatcher, ensuring behavioral equivalence. A1–A4 run the same staged pipeline with one named stage bypassed: A1 retains retrieval order; A2 skips pass-through verification; A3 supplies a documented neutral `unknown` compliance representation; A4 returns the reasoning agent's upstream response text without synthesizing an answer. A5 and A6 delegate directly to the existing RAG-only and LLM-only paths.

The new layer records explicit invocation booleans and enabled/disabled lists. A disabled stage is `not_applicable`, never failed. Verification and enforcement metrics are forced to null for configurations that disable their corresponding stage. Retrieval metrics remain null for A6. Generation-dependent RQEM values remain null for A4 unless an external evaluator explicitly supplies an applicable score.

## Outputs

The runner writes checkpointed per-configuration raw JSON plus a merged schema with benchmark metadata, selected configurations, and unique `(question_id, configuration_id)` rows. Evaluation summaries preserve JSON null. The analysis command emits the four requested CSV/JSON publication tables using question-ID inner alignment.

## Risks and compatibility

- The repository does not contain the finalized 600-query artifact; the runner discovers standard artifact locations or requires `--benchmark`, and refuses `--full` unless exactly 600 unique questions are loaded.
- The current `FactCheckAgent` is pass-through and the added stable reranker is behavior-preserving; their contribution can only become substantive when production implementations replace them.
- Live smoke execution may use OpenAI APIs. A deterministic `--offline` smoke mode is provided for reproducible CI without fabricated benchmark metrics; its rows are explicitly marked `smoke_stub` and must not be used as study results.
- Existing frozen artifacts are never modified, and output replacement is atomic. Default production APIs and environment behavior remain unchanged.
