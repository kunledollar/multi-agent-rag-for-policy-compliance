import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# -------------------------
# LOCKED container root
# -------------------------
ROOT = Path("/app")
load_dotenv(ROOT / ".env")

# -------------------------
# Conservative defaults
# -------------------------
DEFAULT_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
MIN_CONTEXT_CHUNKS = int(os.getenv("MIN_CONTEXT_CHUNKS", "2"))
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.55"))


@dataclass
class RetrievedChunk:
    score: float
    source: str
    page: Optional[int]
    text: str
    chunk_id: Optional[str] = None


@dataclass
class OrchestratorResult:
    status: str
    query: str
    top_k: int
    retrieved_chunks: List[RetrievedChunk]
    policy_evidence_ok: bool
    gating_reason: Optional[str]
    timings_ms: Dict[str, float]


# -------------------------
# SAFE IMPORT (package + CLI)
# -------------------------
try:
    # When imported as part of the app package (FastAPI)
    from .retriever_agent import RetrieverAgent
except ImportError:
    # When run as a script: fix sys.path and import
    sys.path.insert(0, str(ROOT / "app"))
    from agents.retriever_agent import RetrieverAgent


class OrchestratorAgent:
    """
    Sentinel Orchestrator Agent (Conservative / Enterprise-safe)
    """

    def __init__(
        self,
        retriever_agent: Optional[Any] = None,
        top_k: int = DEFAULT_TOP_K,
        min_context_chunks: int = MIN_CONTEXT_CHUNKS,
        min_retrieval_score: float = MIN_RETRIEVAL_SCORE,
    ):
        self.top_k = top_k
        self.min_context_chunks = min_context_chunks
        self.min_retrieval_score = min_retrieval_score

        self.retriever = retriever_agent or RetrieverAgent()

    def _gate_evidence(self, chunks: List[RetrievedChunk]) -> Tuple[bool, Optional[str]]:
        if not chunks or len(chunks) < self.min_context_chunks:
            return False, f"Not enough policy context (need ≥ {self.min_context_chunks} chunks)."

        best = max(c.score for c in chunks)
        if best < self.min_retrieval_score:
            return False, (
                f"Policy evidence too weak "
                f"(best_score={best:.4f} < {self.min_retrieval_score})."
            )

        return True, None

    def run(self, query: str, top_k: Optional[int] = None) -> OrchestratorResult:
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}

        try:
            # Retrieve
            t_retrieve0 = time.perf_counter()
            k = int(top_k or self.top_k)
            raw_results = self.retriever.retrieve(query=query, top_k=k)

            chunks: List[RetrievedChunk] = [
                RetrievedChunk(
                    score=float(r.get("score", 0.0)),
                    source=str(r.get("source", "")),
                    page=r.get("page", None),
                    text=str(r.get("text", "")),
                    chunk_id=r.get("id", None),
                )
                for r in raw_results
            ]

            timings["retrieve_ms"] = (time.perf_counter() - t_retrieve0) * 1000

            # Gate
            t_gate0 = time.perf_counter()
            ok, reason = self._gate_evidence(chunks)
            timings["gate_ms"] = (time.perf_counter() - t_gate0) * 1000
            timings["total_ms"] = (time.perf_counter() - t0) * 1000

            if not ok:
                return OrchestratorResult(
                    status="insufficient_evidence",
                    query=query,
                    top_k=k,
                    retrieved_chunks=chunks,
                    policy_evidence_ok=False,
                    gating_reason=reason,
                    timings_ms=timings,
                )

            return OrchestratorResult(
                status="ok",
                query=query,
                top_k=k,
                retrieved_chunks=chunks,
                policy_evidence_ok=True,
                gating_reason=None,
                timings_ms=timings,
            )

        except Exception as e:
            timings["total_ms"] = (time.perf_counter() - t0) * 1000
            return OrchestratorResult(
                status="error",
                query=query,
                top_k=int(top_k or self.top_k),
                retrieved_chunks=[],
                policy_evidence_ok=False,
                gating_reason=f"{type(e).__name__}: {e}",
                timings_ms=timings,
            )


# -------------------------
# CLI test
# -------------------------
if __name__ == "__main__":
    print("🚀 Sentinel Orchestrator Agent starting")

    agent = OrchestratorAgent()
    q = os.getenv("TEST_QUERY", "What is the policy on remote work?")
    result = agent.run(q)

    print(f"🔎 Query: {result.query}")
    print(f"📌 Status: {result.status}")
    if result.gating_reason:
        print(f"🧱 Gating: {result.gating_reason}")
    print(f"⏱️ Timings(ms): {result.timings_ms}")

    if result.retrieved_chunks:
        print(f"\n✅ Retrieved {len(result.retrieved_chunks)} chunks:")
        for i, c in enumerate(result.retrieved_chunks, start=1):
            preview = c.text[:160].replace("\n", " ")
            print(
                f"{i}. score={c.score:.4f} "
                f"source={c.source} page={c.page} "
                f"text={preview}..."
            )
