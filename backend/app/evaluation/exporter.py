from __future__ import annotations

import html
import json
import re
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any


SHEETS = ["Questions", "Detailed Results", "Governance Summary", "Retrieval Metrics", "Latency Summary", "Errors", "Run Metadata"]


def sanitize_filename(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (clean or "governance_evaluation")[:100]


def _serial(value):
    if value is None: return ""
    if isinstance(value, (datetime, date)): return value.isoformat()
    if isinstance(value, (list, dict)): return json.dumps(value, ensure_ascii=False, default=str)
    return value.value if hasattr(value, "value") else str(value)


def _col(number):
    result = ""
    while number: number, rem = divmod(number - 1, 26); result = chr(65 + rem) + result
    return result


def _table(rows):
    if not rows: return [["No records"]]
    normalized = [r.model_dump() if hasattr(r, "model_dump") else r for r in rows]
    headers = list(normalized[0]); return [headers] + [[_serial(row.get(h)) for h in headers] for row in normalized]


def write_workbook(path: Path, sheets: dict[str, list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as book:
        overrides = "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,8))
        book.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'+overrides+'</Types>')
        book.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        names = "".join(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i,name in enumerate(SHEETS,1))
        book.writestr("xl/workbook.xml", '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+names+'</sheets></workbook>')
        rels = "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,8))
        book.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+rels+'</Relationships>')
        for index, name in enumerate(SHEETS, 1):
            table = _table(sheets.get(name, [])); xml_rows=[]
            for r,row in enumerate(table,1):
                cells="".join(f'<c r="{_col(c)}{r}" t="inlineStr"><is><t xml:space="preserve">{html.escape(str(value), quote=False)}</t></is></c>' for c,value in enumerate(row,1))
                xml_rows.append(f'<row r="{r}">{cells}</row>')
            book.writestr(f"xl/worksheets/sheet{index}.xml", '<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+''.join(xml_rows)+'</sheetData></worksheet>')


def export_run(output_dir: Path, run_name: str, payload: dict, formats: list[str]):
    output_dir.mkdir(parents=True, exist_ok=True); stem = sanitize_filename(run_name); artifacts=[]
    sheets = payload.pop("_sheets")
    if "xlsx" in formats:
        path = output_dir / f"{stem}.xlsx"; write_workbook(path, sheets); artifacts.append(str(path))
    if "json" in formats:
        path = output_dir / f"{stem}.json"; path.write_text(json.dumps(payload, default=_serial, ensure_ascii=False, indent=2), encoding="utf-8"); artifacts.append(str(path))
    return artifacts
