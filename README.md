# Multi-Agent RAG for Policy Compliance

An applied AI system implementing a **multi-agent Retrieval-Augmented Generation (RAG)** architecture for **policy and regulatory compliance analysis**, with automated CI/CD deployment to AWS EC2.

---

##  Project Overview

This project demonstrates how **Large Language Models (LLMs)** can be combined with **retrieval systems and autonomous agents** to analyze, interpret, and respond to organizational policies and compliance documents.

The system ingests structured and unstructured policy texts, retrieves relevant context, and coordinates multiple agents to deliver accurate, explainable compliance insights.

It is designed as an **end-to-end applied AI project**, covering:

* Model orchestration
* Data ingestion
* Monitoring & observability
* Production deployment with CI/CD

---

##  Architecture Highlights

* **Multi-Agent RAG Pipeline**

  * Retriever agent for policy context
  * Reasoning agent for compliance analysis
  * Response agent for structured outputs

* **Retrieval-Augmented Generation (RAG)**

  * Embeddings-based document retrieval
  * Context-aware LLM responses

* **Containerized Microservices**

  * API service (FastAPI)
  * Dashboard (Streamlit)
  * Monitoring stack (Prometheus, Grafana, Tempo, OpenTelemetry)

* **Production-Grade CI/CD**

  * GitHub Actions
  * SSH-based deployment to AWS EC2
  * Zero-downtime Docker Compose rebuilds

---

##  Repository Structure

```text
.
├── backend/                # API & agent logic
├── dashboard/              # Streamlit dashboard
├── data/
│   └── raw/                # Policy and compliance documents
├── monitoring/             # Prometheus, Grafana, Tempo configs
├── nginx/                  # Reverse proxy configuration
├── logs/                   # Runtime logs
├── .github/workflows/      # GitHub Actions CI/CD pipeline
├── docker-compose.yml      # Multi-service orchestration
├── deploy.sh               # Manual deployment script
├── main.py                 # Application entry point
├── README.md               # Project documentation
└── .env.example            # Environment variable template
```

---

## Run Locally

### Prerequisites

* Docker Engine with Docker Compose v2
* An OpenAI API key
* Internet access while building the images and when asking questions

### Start Sentinel

1. Create the local environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and replace the `OPENAI_API_KEY` placeholder. Never commit
   the resulting `.env` file.

3. Start the application:

   ```bash
   docker compose up --build -d
   ```

4. Check the API, then open the dashboard:

   ```bash
   curl http://localhost:8000/health
   ```

   * Dashboard: <http://localhost:8501>
   * API documentation: <http://localhost:8000/docs>
   * Grafana: <http://localhost:3000>
   * Prometheus: <http://localhost:9090>

The repository includes a FAISS index in `data/processed/faiss_index`, so it
can answer questions without re-embedding all included policies. To rebuild
the index after adding documents to `data/raw`, run:

```bash
docker compose --profile tools run --rm ingest
```

Rebuilding the index uses the configured OpenAI embedding model and may incur
API charges. Stop all local services with:

```bash
docker compose down
```

---

##  Deployment & CI/CD

### Continuous Deployment Flow

1. Code pushed to `main`
2. GitHub Actions workflow triggers
3. Secure SSH connection to EC2
4. Repository sync on server
5. Docker services rebuilt and restarted

✔ Fully automated
✔ No manual server access required

---

##  Security & Configuration

* SSH keys used for GitHub Actions → EC2 authentication
* Secrets managed via **GitHub Actions Secrets**
* No credentials committed to source control
* `.env.example` provided for local configuration

---

##  Monitoring & Observability

The system includes a full observability stack:

* **Prometheus** – metrics collection
* **Grafana** – dashboards & visualization
* **Tempo** – distributed tracing
* **OpenTelemetry Collector** – telemetry pipeline

---

##  Use Cases

* Policy compliance checks
* Regulatory document analysis
* Internal governance automation
* AI-assisted auditing workflows

## Governance Evaluation

Sentinel's governance evaluation suite compares `full_sentinel`, `rag_only`,
and `llm_only` against one authoritative **Governance Evaluation Dataset**. It
does not accept an arbitrary question as an evaluation and never invents a
ground truth. The complete Sentinel graph is used by `full_sentinel`;
`rag_only` runs the existing retriever and grounded answer generator without
governance agents; and `llm_only` calls the configured chat model directly,
without retrieval or citations. Cases and modes run sequentially in sorted
question-ID order, and an individual failure is retained as a failed row while
the rest of the benchmark continues.

### Benchmark schema

Upload an `.xlsx` workbook with a `Questions` sheet (the legacy
`All_Queries` sheet is also supported), or a JSON array. Required columns are
`question_id` and `question`; legacy `query_id` and `query` map safely to those
fields. `reference_answer` (`ground_truth_answer`), `category`,
`expected_policy_decision`, `expected_compliance_label`,
`requires_uncertainty`, `expected_uncertainty_behavior`, `requires_refusal`,
`expected_refusal`, `expected_enforcement_action`, `expected_verified_facts`,
`expected_escalation`, `relevant_document_ids`, `relevant_chunk_ids`,
`graded_relevance`, `required_citation_claims`, `required_trace_elements`,
`required_audit_fields`, `expected_agent_handoffs`, `notes`, and
`benchmark_version`/`dataset_version` are supported. On legacy `All_Queries`
sheets, `source_document` is preserved in `source_fields` and mapped to
`relevant_document_ids`. Lists may be JSON arrays or comma-, semicolon-, pipe-,
newline-, or JSON-delimited text. Duplicate IDs, empty datasets, missing
required columns, malformed workbooks, and unreadable files terminate with a
clear 422 response. The input workbook is only read and is never overwritten.

### Metrics and N/A

Rates and accuracies are arithmetic means across applicable benchmark labels:
uncertainty/refusal/escalation compare the observed Boolean decision with the
dataset expectation (therefore correct negative decisions count); policy and
enforcement compare normalized exact labels; fact verification, required
trace elements, and audit fields use required-item coverage. Citation
completeness counts required claims aligned to citations whose source exists in
retrieved evidence—unknown sources never count. Handoff success is successful
divided by attempted handoffs. Latency is end-to-end elapsed milliseconds and
is summarized with count, mean, median, min, max, p90, p95, and p99.

Retrieval document IDs are compared case-insensitively after trimming whitespace,
converting Windows separators, collapsing duplicate slashes, and removing the
runtime corpus prefixes `/app/data/raw/`, `data/raw/`, `./data/raw/`, or `raw/`.
Meaningful subdirectories are retained. When document-level judgments are used,
every returned chunk from a relevant document is relevant at its rank: duplicates
therefore count in Precision@5, MRR, and NDCG@5. Recall@5 deduplicates IDs, so a
relevant document is recalled at most once. Missing or N/A judgments leave all
retrieval metrics N/A rather than fabricating negative labels.

Uncertainty observation first uses structured execution metadata when present:
`uncertainty_observed`, `uncertainty`, `evidence_status`, `needs_more_context`,
`unknown`, `insufficient_evidence`, and `confidence`. If none of those fields
contains a usable value, a deterministic text fallback normalizes case,
whitespace, punctuation, and common contractions, then looks for an explicit
evidence boundary (for example, that a fact is not specified in the context,
cannot be determined, or cannot be verified from available evidence). Generic
hedges such as “may,” “might,” and “possibly,” and unrelated refusal language,
do not count. This rule-based detector intentionally recognizes evidence
limitations rather than every possible expression of caution; novel or highly
implicit wording can therefore be missed. Benchmark fields remain authoritative,
and a missing `requires_uncertainty` expectation keeps the metric N/A.

Retrieval scoring uses chunk IDs when supplied, otherwise document/source IDs.
Precision@5 divides top-five relevant hits by five; Recall@5 divides unique
top-five relevant hits by all known relevant IDs; MRR is the reciprocal first
relevant rank (zero for no hit); NDCG@5 uses `(2^rel-1)/log2(rank+1)` and binary
grades when explicit grades are absent. A case with no relevance judgments, or
an IDCG of zero, is N/A rather than zero. Retrieval and citation metrics are
N/A for `llm_only`; handoffs are N/A for `rag_only` and `llm_only` unless a
real pipeline handoff occurs. All other missing expectations are also N/A and
are excluded from denominators and best-mode selection. Automated exact-label
and item-coverage scoring cannot replace subject-matter review of semantic
answer quality.

### API, dashboard, and artifacts

Enable Admin/Auditor Mode in the dashboard and open **Governance Evaluation**.
Upload the dataset, select modes (all three by default), run it, then inspect
the tabular summary, detailed rows, retrieval results, latency, and errors.
The response and dashboard show dataset filename, benchmark version/count,
evaluation timestamp, corpus/index versions, and model identifier.

```bash
curl -X POST http://localhost:8000/v1/evaluations/governance \
  -H 'Content-Type: application/json' \
  -d '{"benchmark_source":"/app/data/processed/governance_evaluation_dataset.xlsx","selected_modes":["full_sentinel","rag_only","llm_only"],"run_name":"release-check"}'
```

Outputs are placed under `artifacts/governance_evaluations` with sanitized,
run-specific names. Excel exports contain **Questions**, **Detailed Results**,
**Governance Summary**, **Retrieval Metrics**, **Latency Summary**, **Errors**,
and **Run Metadata** sheets; JSON is also supported. Metadata includes model
and embedding configuration, temperature, top-k, corpus/index/benchmark
versions, commit, timestamps, modes, and counts for reproducibility.

Rebuild and validate locally:

```bash
docker compose down
docker compose build api dashboard
docker compose up -d
docker compose exec api python -m unittest discover -s tests -v
```

---

##  Tech Stack

* **Python**
* **FastAPI**
* **Streamlit**
* **Docker & Docker Compose**
* **GitHub Actions**
* **AWS EC2**
* **Prometheus / Grafana / Tempo**
* **LLMs & Embeddings**

---

## Learning Outcomes

This project demonstrates practical experience in:

* Applied NLP & LLM systems
* Multi-agent architectures
* Retrieval-Augmented Generation
* Cloud deployment & DevOps
* Production AI system design

---

## License

This project is provided for educational and demonstration purposes.

---

## Author

**Akeem Asiru**
