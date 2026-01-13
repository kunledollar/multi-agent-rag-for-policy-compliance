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

CHAT_MODEL = os.getenv("CHAT_MODEL", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "700"))

client = OpenAI(api_key=OPENAI_API_KEY)


class AnswerGenerationAgent:
    """
    Enterprise-grade Answer Generation:
    - Grounded in retrieved chunks (no free-styling)
    - Produces JSON output (safe for API + UI)
    - If compliance verdict is 'unknown', returns a cautious response + next steps
    """

    def __init__(self, chat_model: str = CHAT_MODEL):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing from .env")
        if not chat_model:
            raise RuntimeError("CHAT_MODEL missing from .env")
        self.chat_model = chat_model

    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        # Keep compact and deterministic; include citations
        lines = []
        for i, c in enumerate(retrieved_chunks, start=1):
            src = c.get("source")
            page = c.get("page")
            text = (c.get("text") or "").replace("\n", " ").strip()
            if len(text) > 900:
                text = text[:900] + "..."
            lines.append(f"[{i}] source={src} page={page} text={text}")
        return "\n".join(lines)

    def _safe_parse(self, raw: str) -> Dict[str, Any]:
        if not raw or not raw.strip():
            raise ValueError("Empty LLM response")

        raw = raw.strip()

        try:
            return json.loads(raw)
        except Exception:
            pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])

        raise ValueError("Invalid JSON from LLM")

    def _build_prompt(
        self,
        question: str,
        compliance_result: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> str:
        verdict = compliance_result.get("verdict", "unknown")
        rationale = compliance_result.get("rationale", "")
        citations = compliance_result.get("citations", [])
        flags = compliance_result.get("flags", [])

        context = self._format_context(retrieved_chunks)

        return f"""
You are Sentinel. Generate a final answer grounded ONLY in the provided context.

User question:
{question}

Compliance verdict:
{verdict}

Compliance rationale:
{rationale}

Reasoning summary:
{reasoning_result.get("summary_reasoning", "")}

Flags:
{json.dumps(flags, indent=2)}

Citations (hints):
{json.dumps(citations, indent=2)}

Retrieved Context (the ONLY allowed evidence):
{context}

Return ONLY valid JSON with:
answer (string, concise, enterprise tone)
action_items (array of strings)
citations (array of objects: source, page, quote_hint)
safety_note (string, optional)

Rules:
- If verdict is "unknown": be conservative, say policy versions conflict/are ambiguous, suggest how to confirm authoritative version.
- Do NOT invent policy details not explicitly present in context.
- Keep answer <= 10 lines.
"""

    def run(
        self,
        question: str,
        compliance_result: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        verdict = compliance_result.get("verdict", "unknown")

        # Deterministic fallback (never fail)
        fallback = {
            "answer": (
                "I found multiple Remote Work Policy documents, but they appear to represent different versions. "
                "Because Sentinel cannot confirm which version is authoritative, I can’t provide a single definitive interpretation yet."
                if verdict == "unknown"
                else "Based on the retrieved policy context, here is the most supported answer."
            ),
            "action_items": (
                [
                    "Confirm the authoritative policy version (e.g., latest effective date / official policy owner).",
                    "If available, provide the approved policy repository link or policy hierarchy rule.",
                    "Re-run the query after confirming which version applies to your business unit/region.",
                ]
                if verdict == "unknown"
                else ["Confirm any exceptions or approval workflow mentioned in the policy source."]
            ),
            "citations": compliance_result.get("citations", []) or [],
            "safety_note": "This response is grounded only in the retrieved policy text.",
        }

        prompt = self._build_prompt(question, compliance_result, reasoning_result, retrieved_chunks)

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

            # Basic shape validation
            if not isinstance(parsed, dict) or "answer" not in parsed:
                return fallback

            if "action_items" not in parsed or not isinstance(parsed["action_items"], list):
                parsed["action_items"] = fallback["action_items"]

            if "citations" not in parsed or not isinstance(parsed["citations"], list):
                parsed["citations"] = fallback["citations"]

        except Exception:
            parsed = fallback

        total_ms = (time.perf_counter() - t0) * 1000.0
        parsed["timings_ms"] = {"total_ms": total_ms}
        return parsed


# -------------------------
# CLI Test
# -------------------------
def _demo():
    print("🧾 Sentinel Answer Generation Agent starting")

    from app.agents.retriever_agent import RetrieverAgent
    from app.agents.compliance_agent import ComplianceAgent
    from app.agents.reasoning_agent import ReasoningAgent

    query = "What is the policy on remote work?"

    chunks = RetrieverAgent().retrieve(query=query, top_k=5)
    compliance = ComplianceAgent().run(query=query, retrieved_chunks=chunks)
    reasoning = ReasoningAgent().run(
        question=query,
        compliance_result=compliance.__dict__,
        retrieved_chunks=chunks,
    )

    result = AnswerGenerationAgent().run(
        question=query,
        compliance_result=compliance.__dict__,
        reasoning_result=reasoning,
        retrieved_chunks=chunks,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _demo()
