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
