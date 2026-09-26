# Sentinel R6 Experiment Readiness Report

## Architecture

- **Autonomous agents implemented:** retrieval, verification, policy, reasoning, risk,
  answer, critic, and final-decision agents return the shared `AgentDecision` contract.
- **Shared state implemented:** `SentinelState` carries evidence, decisions, messages,
  intermediate results, revisions, and the final action.
- **Communication implemented:** requested actions are converted into addressed
  `AgentMessage` records and retained in the response audit trail.
- **Critic implemented:** answer evidence, upstream risk, and citation identity are
  reviewed before final arbitration.
- **Revision loop implemented:** critic rejections create `RevisionRecord` entries and
  re-run the requested work, with a hard maximum of two iterations.

## Evaluation

- **External datasets connected:** MTRAG, LegalBench, XSTest, HarmBench, and RAGTruth
  adapters normalize source rows to `BenchmarkCase`.
- **R5 baseline preserved:** `sequential_graph.py` uses the R5 answer and reasoning
  implementations and remains selectable as ablation A5.
- **R6 metrics available:** agent-decision accuracy, conflict-resolution accuracy,
  self-correction rate, and collaboration score supplement the existing retrieval,
  citation, uncertainty, refusal, and latency metrics.

## Testing

- **R6 smoke and contract tests:** 10 focused integration tests pass, validating the
  complete trace, required trace fields, both R5/R6 execution paths, all agent return
  contracts, and two-iteration termination.
- **Dataset adapter test:** imports all five adapters and validates `BenchmarkCase`
  output.
- **Evaluation sample:** scores three cases and asserts both legacy and R6 metric
  fields are emitted.
- **Local full-suite status:** the repository declares all runtime dependencies in
  `backend/requirements.txt`. In the validation container, dependency installation is
  blocked by the package proxy (HTTP 403), so dependency-free validation uses import
  stubs; CI must run plain `pytest` with the declared requirements installed.
