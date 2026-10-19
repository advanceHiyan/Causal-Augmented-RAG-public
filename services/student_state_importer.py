from __future__ import annotations

from io import BytesIO, StringIO
import json
import re
from pathlib import Path

import pandas as pd


def normalize_student_state_key(raw_key: str, slot_alias_map: dict[str, str] | None = None) -> str:
    key = (raw_key or "").strip()
    if not key:
        return ""
    alias = slot_alias_map or {}
    if key in alias:
        return alias[key]
    lowered = key.lower()
    if lowered in alias:
        return alias[lowered]
    lowered = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", lowered).strip("_")
    return lowered[:64]


def _normalize_row_keys(row: dict, slot_alias_map: dict[str, str] | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (row or {}).items():
        nk = normalize_student_state_key(str(k), slot_alias_map=slot_alias_map)
        if not nk or v is None:
            continue
        sv = str(v).strip()
        if not sv or sv.lower() in {"nan", "none", "null"}:
            continue
        out[nk] = sv
    return out


def _row_record(
    row: dict,
    *,
    source_file: str,
    slot_alias_map: dict[str, str] | None = None,
    sheet: str = "",
    orientation: str = "horizontal",
) -> dict:
    fields: dict[str, str] = {}
    for k, v in (row or {}).items():
        key = str(k).strip()
        if not key or key.lower().startswith("unnamed:") or v is None:
            continue
        sv = str(v).strip()
        if not sv or sv.lower() in {"nan", "none", "null"}:
            continue
        fields[key] = sv
    normalized = _normalize_row_keys(fields, slot_alias_map=slot_alias_map)
    return {
        "source_file": source_file,
        "sheet": sheet,
        "orientation": orientation,
        "headers": list(fields.keys()),
        "fields": fields,
        "normalized_fields": normalized,
    }


def _vertical_records_from_dataframe(
    df: pd.DataFrame,
    *,
    source_file: str,
    slot_alias_map: dict[str, str] | None = None,
    sheet: str = "",
) -> list[dict]:
    if df.shape[1] < 2:
        return []
    filled = df.fillna("")
    label_col = filled.columns[0]
    labels = [str(v).strip() for v in filled[label_col].tolist()]
    normalized_labels = {
        normalize_student_state_key(v, slot_alias_map=slot_alias_map) for v in labels if v
    }
    if not ({"student_id", "name"} & normalized_labels):
        return []

    records: list[dict] = []
    for col in filled.columns[1:]:
        row: dict[str, str] = {}
        for label, value in zip(labels, filled[col].tolist()):
            if label and str(value).strip():
                row[label] = str(value).strip()
        rec = _row_record(
            row,
            source_file=source_file,
            slot_alias_map=slot_alias_map,
            sheet=sheet,
            orientation="vertical",
        )
        if rec.get("normalized_fields"):
            records.append(rec)
    return records


def _parse_markdown_tables(text: str) -> list[dict]:
    rows: list[dict] = []
    lines = (text or "").splitlines()
    i = 0
    n = len(lines)
    while i + 2 < n:
        l1 = lines[i].strip()
        l2 = lines[i + 1].strip()
        if "|" not in l1 or "|" not in l2 or "-" not in l2:
            i += 1
            continue
        header = [c.strip() for c in l1.strip("|").split("|")]
        if len(header) < 2:
            i += 1
            continue
        j = i + 2
        while j < n:
            body = lines[j].strip()
            if "|" not in body:
                break
            cells = [c.strip() for c in body.strip("|").split("|")]
            if len(cells) != len(header):
                break
            rows.append({header[idx]: cells[idx] for idx in range(len(header))})
            j += 1
        i = j
    return rows


def _parse_kv_lines(text: str, slot_alias_map: dict[str, str] | None = None) -> dict[str, str]:
    kv: dict[str, str] = {}
    pat = re.compile(r"^\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,64})\s*[:：]\s*(.+?)\s*$")
    for ln in (text or "").splitlines():
        m = pat.match(ln)
        if not m:
            continue
        nk = normalize_student_state_key(m.group(1), slot_alias_map=slot_alias_map)
        vv = m.group(2).strip()
        if nk and vv:
            kv[nk] = vv
    return kv


def parse_student_state_document(
    uploaded_file,
    *,
    slot_alias_map: dict[str, str] | None = None,
) -> tuple[dict, str]:
    """
    Parse uploaded student-state file into a non-chunked structured payload.
    Supported: markdown/json/jsonl/csv/tsv/xlsx/xls/txt.
    """
    name = getattr(uploaded_file, "name", "") or "uploaded"
    suffix = Path(name).suffix.lower()
    data = uploaded_file.getvalue() or b""
    text = data.decode("utf-8", errors="ignore")
    parsed: dict = {
        "kind": "mixed",
        "rows": [],
        "kv": {},
    }

    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows = []
            for ln in text.splitlines():
                s = ln.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(
                        _row_record(
                            obj,
                            source_file=name,
                            slot_alias_map=slot_alias_map,
                            orientation="jsonl",
                        )
                    )
            parsed["rows"] = [r for r in rows if r]
            return parsed, f"JSONL rows={len(parsed['rows'])}"

        obj = json.loads(text)
        if isinstance(obj, list):
            rows = [
                _row_record(r, source_file=name, slot_alias_map=slot_alias_map, orientation="json")
                for r in obj
                if isinstance(r, dict)
            ]
            parsed["rows"] = [r for r in rows if r]
            return parsed, f"JSON array rows={len(parsed['rows'])}"
        if isinstance(obj, dict):
            kv_source = {k: v for k, v in obj.items() if not isinstance(v, (list, dict))}
            parsed["kv"] = _normalize_row_keys(kv_source, slot_alias_map=slot_alias_map)
            rows: list[dict] = []
            for v in obj.values():
                if isinstance(v, list):
                    rows.extend(
                        [
                            _row_record(
                                x,
                                source_file=name,
                                slot_alias_map=slot_alias_map,
                                orientation="json",
                            )
                            for x in v
                            if isinstance(x, dict)
                        ]
                    )
            parsed["rows"] = [r for r in rows if r]
            return parsed, f"JSON object rows={len(parsed['rows'])} kv={len(parsed['kv'])}"
        return parsed, "JSON unsupported structure"

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(StringIO(text), sep=sep)
        rows = [
            _row_record(r, source_file=name, slot_alias_map=slot_alias_map, orientation="horizontal")
            for r in df.fillna("").to_dict(orient="records")
        ]
        rows.extend(
            _vertical_records_from_dataframe(df, source_file=name, slot_alias_map=slot_alias_map, sheet="")
        )
        parsed["rows"] = [r for r in rows if r.get("normalized_fields")]
        return parsed, f"table rows={len(parsed['rows'])} cols={len(df.columns)}"

    if suffix in {".xlsx", ".xls"}:
        rows: list[dict] = []
        xls = pd.ExcelFile(BytesIO(data))
        sheet_notes: list[str] = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str).dropna(how="all")
            if df.empty:
                sheet_notes.append(f"{sheet_name}:0")
                continue
            df = df.fillna("")
            sheet_rows = []
            for raw_row in df.to_dict(orient="records"):
                record = _row_record(
                    raw_row,
                    source_file=name,
                    slot_alias_map=slot_alias_map,
                    sheet=str(sheet_name),
                    orientation="horizontal",
                )
                if record.get("normalized_fields"):
                    sheet_rows.append(record)
            sheet_rows.extend(
                _vertical_records_from_dataframe(
                    df,
                    source_file=name,
                    slot_alias_map=slot_alias_map,
                    sheet=str(sheet_name),
                )
            )
            rows.extend(sheet_rows)
            sheet_notes.append(f"{sheet_name}:{len(sheet_rows)}")
        parsed["rows"] = rows
        parsed["sheets"] = xls.sheet_names
        return parsed, f"excel rows={len(rows)} sheets={'; '.join(sheet_notes)}"

    if suffix == ".md":
        md_rows = [
            _row_record(r, source_file=name, slot_alias_map=slot_alias_map, orientation="markdown_table")
            for r in _parse_markdown_tables(text)
        ]
        parsed["rows"] = [r for r in md_rows if r.get("normalized_fields")]
        parsed["kv"] = _parse_kv_lines(text, slot_alias_map=slot_alias_map)
        return parsed, f"markdown rows={len(parsed['rows'])} kv={len(parsed['kv'])}"

    if suffix == ".txt":
        parsed["kv"] = _parse_kv_lines(text, slot_alias_map=slot_alias_map)
        return parsed, f"text kv={len(parsed['kv'])}"

    return parsed, f"unsupported suffix={suffix or 'none'}"
