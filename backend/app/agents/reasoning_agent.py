import os
import sys
import json
import time
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -------------------------------------------------------------------
# Make runnable both as package import and direct execution
# -------------------------------------------------------------------
THIS_FILE = os.path.abspath(__file__)
APP_DIR = os.path.dirname(os.path.dirname(THIS_FILE))   # /app/app
PKG_ROOT = os.path.dirname(APP_DIR)                     # /app
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

# -------------------------
# Env Config
# -------------------------
CHAT_MODEL = os.getenv("CHAT_MODEL", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MAX_TOKENS = int(os.getenv("REASONING_MAX_TOKENS", "350"))

client = OpenAI(api_key=OPENAI_API_KEY)


class ReasoningAgent:
    """
    Hybrid Reasoning Agent (Enterprise Safe):
    - Deterministic decision path
    - LLM used only for explanation polish
    - NEVER crashes if LLM misbehaves
    """

    def __init__(self, chat_model: str = CHAT_MODEL):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing from .env")
        if not chat_model:
            raise RuntimeError("CHAT_MODEL missing from .env")
        self.chat_model = chat_model

    # -------------------------
    # Deterministic logic
    # -------------------------
    def _decision_steps(
        self,
        compliance_result: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[str]:
        steps = []
        steps.append(f"Compliance verdict was '{compliance_result.get('verdict')}'")

        if len(retrieved_chunks) > 1:
            steps.append("Multiple policy documents were retrieved")

        sources = {c.get("source") for c in retrieved_chunks}
        if len(sources) > 1:
            steps.append("Policies originated from different source files")

        if compliance_result.get("verdict") == "unknown":
            steps.append("No authoritative policy precedence could be determined")

        steps.append("System avoided assumptions beyond available evidence")
        return steps

    # -------------------------
    # Prompt
    # -------------------------
    def _prompt(self, question: str, verdict: str, steps: List[str]) -> str:
        return f"""
You are Sentinel's Reasoning Agent.

Explain the compliance outcome clearly and conservatively.

Question:
{question}

Compliance verdict:
{verdict}

Decision path:
{json.dumps(steps, indent=2)}

Return ONLY valid JSON with:
summary_reasoning (string)
confidence_note (string)
"""

    # -------------------------
    # Safe JSON parsing
    # -------------------------
    def _safe_parse(self, raw: str) -> Dict[str, str]:
        if not raw or not raw.strip():
            raise ValueError("Empty LLM response")

        raw = raw.strip()

        # Try direct parse
        try:
            return json.loads(raw)
        except Exception:
            pass

        # Try extracting JSON block
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                pass

        raise ValueError("Invalid JSON from LLM")

    # -------------------------
    # Run
    # -------------------------
    def run(
        self,
        question: str,
        compliance_result: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        steps = self._decision_steps(compliance_result, retrieved_chunks)
        verdict = compliance_result.get("verdict", "unknown")
        prompt = self._prompt(question, verdict, steps)

        try:
            resp = client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=MAX_TOKENS,
            )

            raw = resp.choices[0].message.content
            parsed = self._safe_parse(raw)

            summary = parsed.get("summary_reasoning", "").strip()
            confidence = parsed.get("confidence_note", "").strip()

        except Exception:
            # 🔒 HARD FAIL SAFE (enterprise requirement)
            summary = (
                "The system identified ambiguity across multiple policy documents "
                "and therefore avoided issuing a definitive interpretation."
            )
            confidence = "High confidence in ambiguity due to conflicting evidence."

        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "summary_reasoning": summary,
            "decision_path": steps,
            "confidence_note": confidence,
            "timings_ms": {"total_ms": total_ms},
        }


# -------------------------
# CLI Test
# -------------------------
def _demo():
    print("🧠 Sentinel Reasoning Agent starting")

    from app.agents.retriever_agent import RetrieverAgent
    from app.agents.compliance_agent import ComplianceAgent

    query = "What is the policy on remote work?"

    retriever = RetrieverAgent()
    chunks = retriever.retrieve(query=query, top_k=5)

    compliance = ComplianceAgent().run(query=query, retrieved_chunks=chunks)

    result = ReasoningAgent().run(
        question=query,
        compliance_result=compliance.__dict__,
        retrieved_chunks=chunks,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _demo()
