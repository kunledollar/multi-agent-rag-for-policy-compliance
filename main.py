import os
import re
import math
import requests
import streamlit as st
from datetime import datetime

# ==========================================================
# RAGAS (SAFE, OPTIONAL — DOES NOT BREAK APP)
# ==========================================================
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except Exception:
    RAGAS_AVAILABLE = False

# ==========================================================
# CONFIG
# ==========================================================
API_BASE = os.getenv("SENTINEL_API_BASE", "http://sentinel_api:8000")
ADMIN_ENV_ENABLED = os.getenv("SENTINEL_ADMIN_MODE", "false").lower() == "true"

DEFAULT_QUESTION = "What are compliance requirements for employees' daily operational hours?"

st.set_page_config(page_title="Sentinel", layout="wide")

# ==========================================================
# SESSION STATE
# ==========================================================
if "is_admin" not in st.session_state:
    st.session_state.is_admin = True

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "question" not in st.session_state:
    st.session_state.question = DEFAULT_QUESTION

if "auto_run_default" not in st.session_state:
    st.session_state.auto_run_default = True

# ==========================================================
# RAGAS — RUN ONCE IF MISSING (NO UI SIDE EFFECT)
# ==========================================================
def run_ragas_if_missing(data: dict) -> dict:
    if not RAGAS_AVAILABLE:
        return data
    if not st.session_state.is_admin:
        return data
    if data.get("ragas_metrics"):
        return data

    answer = data.get("answer")
    chunks = data.get("chunks", [])

    if not answer or not chunks:
        return data

    contexts = [c.get("text") for c in chunks if c.get("text")]
    if not contexts:
        return data

    try:
        dataset = Dataset.from_dict({
            "question": [st.session_state.question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truths": [""],
        })

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        df = result.to_pandas()
        data["ragas_metrics"] = {
            row["metric"]: float(row["score"])
            for _, row in df.iterrows()
            if isinstance(row.get("score"), (int, float))
        }
    except Exception:
        pass

    return data

# ==========================================================
# HELPERS — SAFE, UI-ONLY (UNCHANGED)
# ==========================================================
def compliance_status(confidence: float) -> str:
    if confidence >= 0.85:
        return "🟢 CONFIRMED"
    if confidence >= 0.50:
        return "🟡 PARTIAL"
    return "🔴 RESTRICTED"

def groundedness(answer: str, citations: list) -> int:
    if not answer:
        return 0
    sentences = [s for s in re.split(r"[.!?]", answer) if s.strip()]
    return min(100, int((len(citations) / max(len(sentences), 1)) * 100))

def safety_flags(confidence: float, citations: list) -> list:
    flags = []
    if confidence < 0.5:
        flags.append("Low confidence response")
    if not citations:
        flags.append("No citations returned")
    return flags

def location_label(page):
    return "Section-based" if page in (None, "None", "") else f"Page {page}"

def pretty_source_name(source: str) -> str:
    if not source:
        return "Unknown Policy"
    name = source.replace("_", " ").replace("-", " ").replace(".txt", "")
    return name.strip().title() + " Policy"

# ---------- Agent Trace helpers (UNCHANGED) ----------
def status_badge(status: str) -> str:
    s = (status or "success").lower()
    if s == "failed":
        return "🔴 Failed"
    if s == "warning":
        return "🟡 Warning"
    return "🟢 Success"

def risk_badge(risk: str) -> str:
    r = (risk or "Low").lower()
    if r == "high":
        return "🔴 High Risk"
    if r == "medium":
        return "🟡 Medium Risk"
    return "🟢 Low Risk"

def render_agent_card(step: dict, idx: int):
    name = step.get("agent_name") or step.get("name") or f"Agent {idx}"
    latency = step.get("latency_ms", "—")
    status = step.get("status", "success")
    risk = step.get("risk_flag", "Low")

    header = f"{name} • {latency} ms • {status_badge(status)} • {risk_badge(risk)}"

    with st.expander(header, expanded=(idx == 1)):
        if step.get("input_summary"):
            st.markdown("**Input Summary**")
            st.write(step["input_summary"])

        if step.get("decision_rationale"):
            st.markdown("**Decision Rationale**")
            st.write(step["decision_rationale"])

        metric_keys = [
            "confidence_score",
            "chunks_retrieved",
            "chunks_used",
            "avg_similarity",
            "min_similarity",
            "source_diversity",
            "conflict_detected",
            "policy_alignment_score",
            "violation_risk",
            "restriction_triggered",
            "legal_guardrail_hit",
            "citations_attached",
            "groundedness_ratio",
        ]

        rows = [(k, step.get(k)) for k in metric_keys if k in step]
        if rows:
            st.markdown("**Key Metrics**")
            for k, v in rows:
                st.write(f"- **{k}**: {v}")

        if step.get("output") and not step.get("decision_rationale"):
            st.markdown("**Raw Output**")
            st.write(step["output"])
# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="font-size:4rem; font-weight:800;">Sentinel</h1>
        <p style="font-size:1.4rem;">Enterprise Policy & Compliance Intelligence Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# ACCESS CONTROL — ALWAYS VISIBLE
# ==========================================================
lock_icon = "🔓" if st.session_state.is_admin else "🔒"
st.markdown(f"## {lock_icon} Access Control")
st.checkbox("Enable Admin / Auditor Mode (Demo)", key="is_admin")
st.divider()

# ==========================================================
# TABS — 7 TOTAL
# ==========================================================
tabs = ["Policy Question"]
if st.session_state.is_admin:
    tabs += [
        "LLM Output",
        "Retreived Chunks",
        "Execution Flow",
        "System Health",
        "RAGAS",
        "Observability",
    ]

tab_objs = st.tabs(tabs)

# ==========================================================
# 1️⃣ POLICY QUESTION
# ==========================================================
with tab_objs[0]:
    st.subheader("Ask a Policy or Compliance Question")

    # Prepopulate question (per your instruction)
    question = st.text_area("Question", height=140, value=st.session_state.question)
    st.session_state.question = question  # keep in session

    run_clicked = st.button("Run Sentinel")

    # Auto-run the default question once (per your instruction)
    should_run = (run_clicked and question.strip()) or (st.session_state.auto_run_default and question.strip())

    if should_run:
        with st.spinner("Running Sentinel…"):
            try:
                r = requests.post(
                    f"{API_BASE}/v1/ask",
                    json={"question": question},
                    timeout=60
                )
                if r.ok:
                    st.session_state.last_result = r.json()
                else:
                    st.error("API error")
            except Exception as e:
                st.error(f"API error: {e}")

        # ensure auto-run happens only once
        st.session_state.auto_run_default = False

    data = st.session_state.last_result
    if data:
        st.success("Answer")
        st.write(data.get("answer"))

        if st.session_state.is_admin:
            citations = data.get("citations", [])
            if citations:
                st.markdown("### 📎 Citations")
                for i, c in enumerate(citations, start=1):
                    st.markdown(
                        f"{i}. **{pretty_source_name(c.get('source'))}** — "
                        f"{location_label(c.get('page'))}"
                    )

# ==========================================================
# 2️⃣ LLM OUTPUT
# ==========================================================
if st.session_state.is_admin:
    with tab_objs[1]:
        st.subheader("LLM Output Metrics")

        data = st.session_state.last_result
        if not data:
            st.info("Run a query to view LLM metrics.")
        else:
            answer = data.get("answer", "")
            confidence = float(data.get("confidence", 0.0))
            citations = data.get("citations", [])

            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{confidence:.2f}")
            col2.metric("Compliance Status", compliance_status(confidence))
            col3.metric("Groundedness", f"{groundedness(answer, citations)}%")

            flags = safety_flags(confidence, citations)
            st.markdown("### ⚠️ Safety Signals")
            if flags:
                for f in flags:
                    st.warning(f)
            else:
                st.success("No safety issues detected")

            st.markdown("### 🧠 Raw LLM Output")
            st.markdown(
                f"""
                <pre style="
                    background-color:#ffffff;
                    color:#000000;
                    padding:16px;
                    border-radius:6px;
                    border:1px solid #e6e6e6;
                    white-space:pre-wrap;
                    font-size:14px;
                    line-height:1.5;
                ">
{answer}
                </pre>
                """,
                unsafe_allow_html=True
            )

# ==========================================================
# 3️⃣ EVIDENCE
# ==========================================================
if st.session_state.is_admin:
    with tab_objs[2]:
        st.subheader("Top Retrieved Evidence Chunks")
        st.caption("Deduplicated, ranked, color-coded evidence used to ground the answer.")

        data = st.session_state.last_result
        if not data:
            st.info("Run a query to view evidence.")
        else:
            raw_chunks = data.get("chunks", [])

            best_by_source = {}
            for ch in raw_chunks:
                src = ch.get("source", "Unknown")
                score = float(ch.get("score", 0.0))
                if src not in best_by_source or score > best_by_source[src]["score"]:
                    best_by_source[src] = {
                        "source": src,
                        "text": ch.get("text", ""),
                        "page": ch.get("page"),
                        "score": score,
                    }

            deduped = sorted(best_by_source.values(), key=lambda x: x["score"], reverse=True)

            primary = [c for c in deduped if c["score"] >= 0.60]
            secondary = [c for c in deduped if 0.40 <= c["score"] < 0.60]
            supporting = [c for c in deduped if c["score"] < 0.40]

            def similarity_badge(score):
                if score >= 0.80:
                    return "🟢 Very High"
                if score >= 0.60:
                    return "🟡 Medium"
                if score >= 0.40:
                    return "🟠 Low"
                return "🔴 Very Low"

            chunk_counter = {"index": 1}

            def render_group(title, chunks):
                if not chunks:
                    return
                st.markdown(f"### {title}")
                for ch in chunks:
                    idx = chunk_counter["index"]
                    header = (
                        f"Chunk {idx} — {pretty_source_name(ch['source'])} • "
                        f"{similarity_badge(ch['score'])} ({ch['score']:.2f}) • "
                        f"{location_label(ch['page'])}"
                    )
                    with st.expander(header, expanded=(idx == 1)):
                        st.write(ch["text"])
                    chunk_counter["index"] += 1

            render_group("Primary Evidence", primary)
            render_group("Secondary Evidence", secondary)
            render_group("Supporting Evidence (Low Confidence)", supporting)

# ==========================================================
# 4️⃣ AGENT TRACE — UPGRADED (ONLY SECTION MODIFIED)
# ==========================================================
if st.session_state.is_admin:
    with tab_objs[3]:
        st.subheader("Agent Execution Trace")

        data = st.session_state.last_result
        if not data or not data.get("agent_trace"):
            st.info("No agent trace available for this request.")
        else:
            for i, step in enumerate(data.get("agent_trace", []), start=1):
                render_agent_card(step, i)

# ==========================================================
# 5️⃣ SYSTEM HEALTH — PROMETHEUS-AWARE MVP
# ==========================================================
if st.session_state.is_admin:
    with tab_objs[4]:
        st.subheader("System Health")
        st.caption("Live reliability, performance, and governance signals")

        PROM_URL = "http://sentinel-prometheus:9090"

        def promql(query):
            try:
                r = requests.get(
                    f"{PROM_URL}/api/v1/query",
                    params={"query": query},
                    timeout=3,
                )
                if r.ok:
                    result = r.json()["data"]["result"]
                    if result:
                        return float(result[0]["value"][1])
            except Exception:
                return None
            return None

        # --------------------------
        # Core Metrics (Prometheus)
        # --------------------------
        up = promql('up{job="sentinel_api"}')
        avg_latency = promql(
            'rate(http_request_duration_seconds_sum[1m]) '
            '/ rate(http_request_duration_seconds_count[1m])'
        )
        success_rate = promql(
            'sum(rate(http_requests_total{status_code=~"2.."}[1m])) '
            '/ sum(rate(http_requests_total[1m]))'
        )

        # --------------------------
        # Fallback health check
        # --------------------------
        if up is None:
            try:
                r = requests.get(f"{API_BASE}/health", timeout=5)
                up = 1 if r.ok else 0
            except Exception:
                up = 0

        # --------------------------
        # Derived UI values
        # --------------------------
        system_status = "🟢 Healthy" if up == 1 else "🔴 Down"

        latency_ms = int(avg_latency * 1000) if avg_latency else None
        success_pct = round(success_rate * 100, 1) if success_rate else None

        data = st.session_state.last_result or {}
        restricted = float(data.get("confidence", 1.0)) == 0.0

        agent_trace = data.get("agent_trace", [])
        confirmed_conflicts = sum(
            1 for a in agent_trace if a.get("conflict_detected") is True
        )
        potential_conflicts = sum(
            1 for a in agent_trace if a.get("potential_conflict") is True
        )

        # --------------------------
        # Badge helpers
        # --------------------------
        def latency_badge(ms):
            if ms is None:
                return "—"
            if ms < 2000:
                return f"🟢 {ms} ms"
            if ms < 5000:
                return f"🟡 {ms} ms"
            return f"🔴 {ms} ms"

        def success_badge(pct):
            if pct is None:
                return "—"
            if pct >= 99:
                return f"🟢 {pct}%"
            if pct >= 95:
                return f"🟡 {pct}%"
            return f"🔴 {pct}%"

        # --------------------------
        # EXECUTIVE SNAPSHOT
        # --------------------------
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("System Status", system_status)
        col2.metric("Avg API Latency", latency_badge(latency_ms))
        col3.metric("Success Rate", success_badge(success_pct))
        col4.metric(
            "Guarded Responses",
            "🔒 Active" if restricted else "🟢 Clear",
        )
        col5.metric(
            "Conflicts Detected",
            f"{confirmed_conflicts} | {potential_conflicts}",
        )

        st.divider()
        st.caption("Powered by Prometheus • Fallback-safe")

# ==========================================================
# 6️⃣ RAGAS  (ONLY THIS TAB WAS CHANGED)
# ==========================================================
if st.session_state.is_admin:
    with tab_objs[5]:
        st.subheader("RAGAS Evaluation")

        data = st.session_state.last_result
        metrics = (data or {}).get("ragas_metrics") or (data or {}).get("ragas") or {}

        def ragas_badge(score: float) -> str:
            if score >= 0.80:
                return "🟢"
            if score >= 0.60:
                return "🟡"
            return "🔴"

        if not metrics:
            st.info("RAGAS metrics pipeline connected. Awaiting evaluation run.")
        else:
            table_rows = []
            for k, v in metrics.items():
                try:
                    score = float(v)
                except Exception:
                    continue
                table_rows.append(
                    {
                        "Metric": k.replace("_", " ").title(),
                        "Score": round(score, 3),
                        "Status": ragas_badge(score),
                    }
                )

            st.table(table_rows)
# ==========================================================
# 7️⃣ OBSERVABILITY (ONLY SECTION UPDATED — SAFE & STABLE)
# ==========================================================
if st.session_state.is_admin:
    with tab_objs[6]:
        import math

        st.subheader("Observability")
        st.caption("Operational performance, reliability, and telemetry")

        # ✅ FIXED env var name + container-safe default
        PROM_URL = os.getenv("PROMETHEUS_URL", "http://sentinel_prometheus:9090")
        GRAFANA_BASE = os.getenv("GRAFANA_BASE_URL", "http://grafana:3000")
        GRAFANA_DASHBOARD_UID = os.getenv(
            "GRAFANA_DASHBOARD_UID",
            "sentinel-observability"
        )

        # --------------------------
        # Prometheus helper (NaN-safe)
        # --------------------------
        def promql(query: str):
            try:
                r = requests.get(
                    f"{PROM_URL}/api/v1/query",
                    params={"query": query},
                    timeout=3,
                )
                if r.ok:
                    result = r.json().get("data", {}).get("result", [])
                    if result:
                        val = float(result[0]["value"][1])
                        if math.isnan(val):
                            return None
                        return val
            except Exception:
                return None
            return None

        # --------------------------
        # Safe formatters
        # --------------------------
        def safe_ms(value):
            if value is None:
                return "—"
            return f"{int(value * 1000)} ms"

        def safe_pct(value):
            if value is None:
                return "—"
            return f"{round(value * 100, 2)}%"

        def safe_rps(value):
            if value is None:
                return "—"
            return f"{round(value, 2)} rps"

        # ==================================================
        # 🧠 SENTINEL SUMMARY (HIGH-SIGNAL KPIs)
        # ==================================================
        st.markdown("### 🧠 Sentinel Summary")

        col1, col2, col3, col4 = st.columns(4)

        # ✅ API availability
        up = promql('up{job="sentinel-api"}')
        col1.metric(
            "API Status",
            "🟢 Up" if up == 1 else "🔴 Down"
        )
        # ✅ P95 latency
        p95 = promql(
            'histogram_quantile(0.95, '
            'sum(rate(http_request_duration_highr_seconds_bucket{job="sentinel-api"}[5m])) by (le))'
        )
        col2.metric("P95 Latency", safe_ms(p95))

        # ✅ Error rate
        err = promql(
            'sum(rate(http_requests_total{job="sentinel-api",status_code=~"5.."}[5m])) '
            '/ sum(rate(http_requests_total{job="sentinel-api"}[5m]))'
        )
        col3.metric("Error Rate", safe_pct(err))

        # ✅ Throughput
        thr = promql(
            'sum(rate(http_requests_total{job="sentinel-api"}[5m]))'
        )
        col4.metric("Throughput", safe_rps(thr))

        st.caption(
            "Metrics display '—' when Prometheus has no samples. "
            "Values are live, job-scoped, and never fabricated."
        )

        st.divider()

        # ==================================================
        # 📊 GRAFANA (DEEP OBSERVABILITY)
        # ==================================================
        st.markdown("### 📊 Live Grafana Dashboard")
        st.caption(
            "Latency percentiles, error trends, throughput, "
            "resource usage, and trace-level observability"
        )

        grafana_url = (
            f"{GRAFANA_BASE}/d/{GRAFANA_DASHBOARD_UID}"
            f"?orgId=1&refresh=30s&kiosk=tv"
        )

        st.components.v1.iframe(
            grafana_url,
            height=900,
            scrolling=True,
        )
