from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List
from xml.etree import ElementTree as ET

from .models import BenchmarkCase, BenchmarkDataset

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}

ALIASES = {
    "question_id": ("question_id", "query_id"), "question": ("question", "query"),
    "reference_answer": ("reference_answer", "ground_truth_answer"),
    "expected_policy_decision": ("expected_policy_decision", "expected_compliance_label"),
    "expected_enforcement_action": ("expected_enforcement_action", "expected_action"),
    "expected_uncertainty_behavior": ("expected_uncertainty_behavior", "expected_behavior"),
    "required_trace_elements": ("required_trace_elements", "decision_trace_requirements"),
    "relevant_document_ids": ("relevant_document_ids", "source_document"),
    "required_citation_claims": ("required_citation_claims", "required_answer_elements"),
    "version": ("benchmark_version", "dataset_version"),
}


def _bool(value: Any):
    if value is None or str(value).strip().lower() in {"", "n/a", "conditional"}: return None
    return str(value).strip().lower() in {"1", "true", "yes", "y", "required"}


def _list(value: Any):
    if value is None or str(value).strip().lower() in {"", "n/a", "na", "null", "none"}: return None
    if isinstance(value, (list, tuple, set)): return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if text.startswith("[") or text.startswith('"'):
        try:
            parsed = json.loads(text)
            values = parsed if isinstance(parsed, list) else [parsed]
            return [str(item).strip() for item in values if item is not None and str(item).strip()] or None
        except (json.JSONDecodeError, TypeError):
            pass
    return [item.strip() for item in re.split(r"[,;|\r\n]+", text) if item.strip()] or None


def _mapped(row: Dict[str, Any], key: str, default=None):
    for alias in ALIASES.get(key, (key,)):
        if row.get(alias) not in (None, ""): return row[alias]
    return default


def map_row(row: Dict[str, Any]) -> BenchmarkCase:
    action = str(_mapped(row, "expected_enforcement_action", "") or "").lower()
    dataset_type = str(row.get("dataset_type", "")).lower()
    escalation = _bool(row.get("expected_escalation", row.get("requires_escalation")))
    fields = {
        "question_id": _mapped(row, "question_id"), "category": row.get("category", "Uncategorized"),
        "question": _mapped(row, "question"), "reference_answer": _mapped(row, "reference_answer"),
        "expected_policy_decision": _mapped(row, "expected_policy_decision"),
        "expected_compliance_label": row.get("expected_compliance_label"),
        "requires_uncertainty": _bool(row.get("requires_uncertainty")) if "requires_uncertainty" in row else (True if dataset_type == "uncertainty" else None),
        "expected_uncertainty_behavior": _mapped(row, "expected_uncertainty_behavior"),
        "requires_refusal": _bool(row.get("requires_refusal")) if "requires_refusal" in row else (True if dataset_type == "risk_control" else None),
        "expected_refusal": _bool(row.get("expected_refusal")) if "expected_refusal" in row else (True if "refuse" in action else None),
        "expected_enforcement_action": _mapped(row, "expected_enforcement_action"),
        "expected_verified_facts": _list(row.get("expected_verified_facts")), "expected_escalation": escalation,
        "relevant_document_ids": _list(_mapped(row, "relevant_document_ids")),
        "relevant_chunk_ids": _list(row.get("relevant_chunk_ids")),
        "graded_relevance": row.get("graded_relevance") if isinstance(row.get("graded_relevance"), dict) else (json.loads(row["graded_relevance"]) if str(row.get("graded_relevance", "")).startswith("{") else None),
        "required_citation_claims": _list(_mapped(row, "required_citation_claims")),
        "required_trace_elements": _list(_mapped(row, "required_trace_elements")),
        "required_audit_fields": _list(row.get("required_audit_fields")),
        "expected_agent_handoffs": _list(row.get("expected_agent_handoffs")), "notes": row.get("notes"),
        "source_fields": row,
    }
    return BenchmarkCase.model_validate(fields)


def _xlsx_rows(path: Path) -> List[Dict[str, str]]:
    with zipfile.ZipFile(path) as book:
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        sheet = next((s for s in workbook.findall(".//m:sheet", NS) if s.attrib.get("name") in {"Questions", "All_Queries"}), None)
        if sheet is None: raise ValueError("Workbook must contain a Questions or All_Queries sheet")
        rel_id = sheet.attrib[f"{{{NS['r']}}}id"]
        rels = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        target = next(r.attrib["Target"] for r in rels if r.attrib.get("Id") == rel_id)
        target = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
        shared = []
        if "xl/sharedStrings.xml" in book.namelist():
            shared = ["".join(node.itertext()) for node in ET.fromstring(book.read("xl/sharedStrings.xml")).findall("m:si", NS)]
        xml = ET.fromstring(book.read(target))
        table = []
        for row in xml.findall(".//m:row", NS):
            values = {}
            for cell in row.findall("m:c", NS):
                col = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                inline = cell.find("m:is", NS); value = cell.find("m:v", NS)
                text = "".join(inline.itertext()) if inline is not None else (value.text if value is not None else "")
                if cell.attrib.get("t") == "s" and text: text = shared[int(text)]
                values[col] = text
            table.append(values)
    if not table: return []
    columns = sorted(table[0], key=lambda x: (len(x), x)); headers = [table[0].get(c, "").strip() for c in columns]
    return [{headers[i]: row.get(c, "") for i, c in enumerate(columns) if headers[i]} for row in table[1:] if any(row.values())]


def load_benchmark(path: Path) -> BenchmarkDataset:
    if not path.is_file(): raise ValueError(f"Governance Evaluation Dataset cannot be loaded: {path}")
    try:
        if path.suffix.lower() == ".xlsx": rows = _xlsx_rows(path)
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8")); rows = payload.get("cases", payload) if isinstance(payload, dict) else payload
        else: raise ValueError("Governance Evaluation Dataset must be .xlsx or .json")
        if not rows: raise ValueError("Governance Evaluation Dataset contains no benchmark questions")
        required = {_mapped(rows[0], "question_id"), _mapped(rows[0], "question")}
        if None in required or "" in required: raise ValueError("Missing required benchmark columns: question_id/query_id and question/query")
        cases = [map_row(row) for row in rows]
        version = str(_mapped(rows[0], "version", "unversioned"))
        return BenchmarkDataset(filename=path.name, version=version, cases=cases)
    except ValueError: raise
    except Exception as error: raise ValueError(f"Malformed Governance Evaluation Dataset: {error}") from error
