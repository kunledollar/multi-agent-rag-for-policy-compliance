import os
import sys
import json
import time
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# -------------------------------------------------------------------
# Make script runnable both as a package import and as a direct script
# -------------------------------------------------------------------
THIS_FILE = os.path.abspath(__file__)
APP_DIR = os.path.dirname(os.path.dirname(THIS_FILE))  # /app/app
PKG_ROOT = os.path.dirname(APP_DIR)                    # /app
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

# -------------------------
# Env Config
# -------------------------
CHAT_MODEL = os.getenv("CHAT_MODEL", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

MIN_EVIDENCE_CHUNKS = int(os.getenv("MIN_EVIDENCE_CHUNKS", "2"))
MIN_EVIDENCE_SCORE = float(os.getenv("MIN_EVIDENCE_SCORE", "0.55"))
COMPLIANCE_MAX_TOKENS = int(os.getenv("COMPLIANCE_MAX_TOKENS", "450"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# -------------------------
# Data Model
# -------------------------
@dataclass
class ComplianceResult:
    status: str
    verdict: str
    confidence: float
    rationale: str
    policy_citations: List[Dict[str, Any]]
    safety_flags: List[str]
    timings_ms: Dict[str, float]

    conflict_detected: bool = False
    potential_conflict: bool = False
    conflict_reason: str = ""
    violation_risk: str = "Low"
    policy_alignment_score: float = 0.0


# -------------------------
# Compliance Agent
# -------------------------
class ComplianceAgent:
    def __init__(
        self,
        min_chunks: int = MIN_EVIDENCE_CHUNKS,
        min_score: float = MIN_EVIDENCE_SCORE,
        chat_model: str = CHAT_MODEL,
    ) -> None:
        self.min_chunks = min_chunks
        self.min_score = min_score
        self.chat_model = chat_model

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing from environment")
        if not self.chat_model:
            raise RuntimeError("CHAT_MODEL missing from environment")

    # -------------------------
    # Evidence Gate
    # -------------------------
    def _evidence_gate(self, chunks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        flags: List[str] = []

        if not chunks:
            flags.append("no_retrieved_chunks")
            return False, flags

        if len(chunks) < self.min_chunks:
            flags.append(f"insufficient_chunks<{self.min_chunks}")
            return False, flags

        top_score = float(chunks[0].get("score", 0.0) or 0.0)
        if top_score < self.min_score:
            flags.append(f"low_top_score<{self.min_score}")
            return False, flags

        return True, flags

    # -------------------------
    # Conflict Detection
    # -------------------------
    def _detect_conflicts(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[bool, bool, str]:

        if not chunks:
            return False, False, ""

        numbers_by_source: Dict[str, set] = {}
        all_numbers = set()

        for c in chunks:
            text = (c.get("text") or "").lower()
            nums = set(re.findall(r"\b\d+\b", text))
            if nums:
                numbers_by_source.setdefault(
                    c.get("source", "unknown"), set()
                ).update(nums)
                all_numbers.update(nums)

        if len(all_numbers) >= 2:
            return True, False, (
                "Different numeric requirements found across policy sources"
            )

        if len(numbers_by_source) >= 2:
            return False, True, (
                "Multiple policy sources reference requirements without clear "
                "authoritative precedence"
            )

        return False, False, ""

    # -------------------------
    # Prompt Builder
    # -------------------------
    def _build_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        evidence_lines = []
        for i, c in enumerate(chunks, start=1):
            evidence_lines.append(
                f"[{i}] score={c.get('score')} source={c.get('source')} "
                f"page={c.get('page')} evidence={c.get('text', '').replace(chr(10), ' ')}"
            )

        return (
            "You are Sentinel's Compliance Agent.\n"
            "Classify the user's question using ONLY the provided policy evidence.\n\n"
            "Return STRICT JSON with:\n"
            "- verdict: compliant | partially_compliant | non_compliant | unknown\n"
            "- confidence: 0.0–1.0\n"
            "- rationale: concise, business-safe explanation\n"
            "- policy_citations: [{source, page, quote_hint}]\n"
            "- safety_flags: []\n\n"
            f"User question:\n{query}\n\n"
            "Policy evidence:\n"
            + "\n".join(evidence_lines)
            + "\n\nRules:\n"
            "- Never invent policy\n"
            "- If evidence is insufficient or conflicting, verdict MUST be 'unknown'\n"
            "- quote_hint must be copied verbatim (≤12 words)\n"
        )

    # -------------------------
    # LLM Call
    # -------------------------
    def _llm_classify(self, prompt: str) -> Dict[str, Any]:
        resp = client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=COMPLIANCE_MAX_TOKENS,
        )

        raw = (resp.choices[0].message.content or "").strip()

        try:
            return json.loads(raw)
        except Exception:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start:end + 1])
            raise

    # -------------------------
    # Public API
    # -------------------------
    def run(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> ComplianceResult:
        t0 = time.perf_counter()

        gate_ok, gate_flags = self._evidence_gate(retrieved_chunks)
        confirmed_conflict, potential_conflict, conflict_reason = (
            self._detect_conflicts(retrieved_chunks)
        )

        if not gate_ok:
            confirmed_conflict = False

        gate_ms = (time.perf_counter() - t0) * 1000

        if not gate_ok:
            return ComplianceResult(
                status="needs_more_context",
                verdict="unknown",
                confidence=0.0,
                rationale=(
                    "Policy confirmation restricted by legal/safety limits due to "
                    "insufficient authoritative evidence."
                ),
                policy_citations=[],
                safety_flags=gate_flags,
                timings_ms={
                    "gate_ms": gate_ms,
                    "llm_ms": 0.0,
                    "total_ms": gate_ms,
                },
                conflict_detected=confirmed_conflict,
                potential_conflict=potential_conflict,
                conflict_reason=conflict_reason,
                violation_risk="High",
                policy_alignment_score=0.0,
            )

        prompt = self._build_prompt(query, retrieved_chunks)

        t1 = time.perf_counter()
        result = self._llm_classify(prompt)
        llm_ms = (time.perf_counter() - t1) * 1000
        total_ms = (time.perf_counter() - t0) * 1000

        verdict = result.get("verdict", "unknown")
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
        rationale = result.get("rationale", "").strip()

        allowed = {"compliant", "partially_compliant", "non_compliant", "unknown"}
        if verdict not in allowed:
            verdict = "unknown"
            gate_flags.append("invalid_verdict_from_llm")

        if confirmed_conflict or potential_conflict or verdict == "non_compliant":
            violation_risk = "High"
        elif verdict == "partially_compliant":
            violation_risk = "Medium"
        else:
            violation_risk = "Low"

        return ComplianceResult(
            status="ok",
            verdict=verdict,
            confidence=confidence,
            rationale=rationale,
            policy_citations=result.get("policy_citations", []),
            safety_flags=list(set(result.get("safety_flags", []) + gate_flags)),
            timings_ms={
                "gate_ms": gate_ms,
                "llm_ms": llm_ms,
                "total_ms": total_ms,
            },
            conflict_detected=confirmed_conflict,
            potential_conflict=potential_conflict,
            conflict_reason=conflict_reason,
            violation_risk=violation_risk,
            policy_alignment_score=confidence,
        )

