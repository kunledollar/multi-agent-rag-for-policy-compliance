from prometheus_client import Counter, Histogram

# --------------------------------------------------
# Agent Execution Metrics
# --------------------------------------------------

agent_execution_total = Counter(
    name="sentinel_agent_execution_total",
    documentation="Total number of agent executions",
    labelnames=["agent_name"],
)

agent_execution_duration_seconds = Histogram(
    name="sentinel_agent_execution_duration_seconds",
    documentation="Execution time per agent",
    labelnames=["agent_name"],
)

# --------------------------------------------------
# RAG Quality Metrics (placeholders)
# --------------------------------------------------

rag_queries_total = Counter(
    name="sentinel_rag_queries_total",
    documentation="Total RAG queries processed",
)

rag_low_confidence_total = Counter(
    name="sentinel_rag_low_confidence_total",
    documentation="Number of low-confidence RAG responses",
)
