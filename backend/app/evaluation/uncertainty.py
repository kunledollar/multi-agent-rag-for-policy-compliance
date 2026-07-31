from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Optional, Tuple


_DIRECT_FIELDS = ("uncertainty_observed", "uncertainty", "unknown", "insufficient_evidence", "needs_more_context")


def _boolean(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "unknown", "insufficient", "uncertain", "needs_more_context"}:
            return True
        if normalized in {"false", "no", "n", "0", "known", "sufficient", "certain"}:
            return False
    return None


def _structured_uncertainty(result: Mapping[str, Any]) -> Tuple[bool, Optional[bool]]:
    """Return whether usable structured metadata exists and its decision."""
    for field in _DIRECT_FIELDS:
        if field in result and result[field] is not None:
            parsed = _boolean(result[field])
            if parsed is not None:
                return True, parsed
    if result.get("evidence_status") is not None:
        status = re.sub(r"[^a-z0-9]+", "_", str(result["evidence_status"]).strip().lower()).strip("_")
        if status in {"insufficient", "insufficient_evidence", "missing", "not_found", "unverified", "unknown"}:
            return True, True
        if status in {"sufficient", "established", "found", "verified", "known"}:
            return True, False
    if result.get("confidence") is not None:
        confidence = result["confidence"]
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and 0 <= confidence <= 1:
            return True, confidence < 0.5
        label = str(confidence).strip().lower()
        if label in {"low", "very low", "none", "unknown", "uncertain"}:
            return True, True
        if label in {"medium", "moderate", "high", "very high", "certain"}:
            return True, False
    return False, None


def normalize_uncertainty_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    contractions = {
        r"\bcan't\b": "cannot", r"\bcannot've\b": "cannot have", r"\bwon't\b": "will not",
        r"\bdoesn't\b": "does not", r"\bdon't\b": "do not", r"\bisn't\b": "is not",
        r"\baren't\b": "are not", r"\bwasn't\b": "was not", r"\bweren't\b": "were not",
    }
    for pattern, replacement in contractions.items():
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


_EVIDENCE_BOUNDARIES = tuple(re.compile(pattern) for pattern in (
    r"\b(?:is|are|was|were|has|have)?\s*not (?:specified|provided|established|stated|documented|available)\b",
    r"\bcannot (?:be )?(?:determined|confirmed|established|verified|ascertained)\b",
    r"\b(?:insufficient|inadequate) (?:information|evidence|context)\b",
    r"\bno (?:relevant )?(?:information|evidence|documentation|details?) (?:was |were |is |are )?(?:found|available|provided)\b",
    r"\b(?:the )?(?:available |provided )?(?:documents?|sources?|context|evidence|materials?) (?:do|does) not (?:state|specify|mention|provide|establish|confirm|show|contain|include)\b",
    r"\bunable to (?:determine|confirm|establish|verify|ascertain)\b",
    r"\bunclear from (?:the )?(?:provided|available|cited|source) (?:context|documents?|sources?|evidence|materials?|information)\b",
    r"\bunknown based on (?:the )?(?:provided|available|cited|source) (?:context|documents?|sources?|evidence|materials?|information)\b",
))


def detect_uncertainty(text: Any) -> bool:
    """Detect an explicit inability to establish a fact from available evidence."""
    normalized = normalize_uncertainty_text(text)
    return bool(normalized) and any(pattern.search(normalized) for pattern in _EVIDENCE_BOUNDARIES)


def uncertainty_observed(result: Mapping[str, Any], answer: Any = None) -> bool:
    """Prefer pipeline metadata; use deterministic answer-text rules as fallback."""
    present, decision = _structured_uncertainty(result)
    if present:
        return bool(decision)
    return detect_uncertainty(result.get("answer", answer))
