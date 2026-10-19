from __future__ import annotations

from io import BytesIO, StringIO
import json
import re
from pathlib import Path
from typing import Any
import time

import pandas as pd

from retrieval.profile_service import PROFILE_SLOT_ALIAS
from services.student_state_importer import parse_student_state_document


def normalize_student_fact_key(raw_key: str) -> str:
    key = (raw_key or "").strip()
    if not key:
        return ""
    if key in PROFILE_SLOT_ALIAS:
        return PROFILE_SLOT_ALIAS[key]
    k = key.lower()
    if k in PROFILE_SLOT_ALIAS:
        return PROFILE_SLOT_ALIAS[k]
    k = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", k).strip("_")
    return k[:64]


def _normalize_row_keys(row: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (row or {}).items():
        nk = normalize_student_fact_key(str(k))
        if not nk:
            continue
        if v is None:
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
    sheet: str = "",
    orientation: str = "horizontal",
) -> dict:
    fields: dict[str, str] = {}
    for k, v in (row or {}).items():
        key = str(k).strip()
        if not key or key.lower().startswith("unnamed:"):
            continue
        if v is None:
            continue
        sv = str(v).strip()
        if not sv or sv.lower() in {"nan", "none", "null"}:
            continue
        fields[key] = sv
    normalized = _normalize_row_keys(fields)
    return {
        "source_file": source_file,
        "sheet": sheet,
        "orientation": orientation,
        "headers": list(fields.keys()),
        "fields": fields,
        "normalized_fields": normalized,
    }


def _vertical_records_from_dataframe(df: pd.DataFrame, *, source_file: str, sheet: str = "") -> list[dict]:
    if df.shape[1] < 2:
        return []
    filled = df.fillna("")
    label_col = filled.columns[0]
    labels = [str(v).strip() for v in filled[label_col].tolist()]
    normalized_labels = {normalize_student_fact_key(v) for v in labels if v}
    if not ({"student_id", "name"} & normalized_labels):
        return []

    records: list[dict] = []
    for col in filled.columns[1:]:
        row: dict[str, str] = {}
        for label, value in zip(labels, filled[col].tolist()):
            if label and str(value).strip():
                row[label] = str(value).strip()
        rec = _row_record(row, source_file=source_file, sheet=sheet, orientation="vertical")
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
        if "|" not in l1 or "|" not in l2:
            i += 1
            continue
        if "-" not in l2:
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
            row = {header[idx]: cells[idx] for idx in range(len(header))}
            rows.append(row)
            j += 1
        i = j
    return rows


def parse_student_kb_document(uploaded_file) -> tuple[dict, str]:
    """
    Parse uploaded student KB file into a non-chunked structured payload.
    Supported: markdown/json/jsonl/csv/tsv/xlsx/xls/txt.
    """
    return parse_student_state_document(uploaded_file, slot_alias_map=PROFILE_SLOT_ALIAS)

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
                    rows.append(_row_record(obj, source_file=name, orientation="jsonl"))
            parsed["rows"] = [r for r in rows if r]
            return parsed, f"JSONL rows={len(parsed['rows'])}"

        obj = json.loads(text)
        if isinstance(obj, list):
            rows = [_row_record(r, source_file=name, orientation="json") for r in obj if isinstance(r, dict)]
            parsed["rows"] = [r for r in rows if r]
            return parsed, f"JSON array rows={len(parsed['rows'])}"
        if isinstance(obj, dict):
            kv_source = {
                k: v for k, v in obj.items()
                if not isinstance(v, (list, dict))
            }
            parsed["kv"] = _normalize_row_keys(kv_source)
            rows: list[dict] = []
            for v in obj.values():
                if isinstance(v, list):
                    rows.extend([_row_record(x, source_file=name, orientation="json") for x in v if isinstance(x, dict)])
            parsed["rows"] = [r for r in rows if r]
            return parsed, f"JSON object rows={len(parsed['rows'])} kv={len(parsed['kv'])}"
        return parsed, "JSON unsupported structure"

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(StringIO(text), sep=sep)
        rows = [
            _row_record(r, source_file=name, orientation="horizontal")
            for r in df.fillna("").to_dict(orient="records")
        ]
        rows.extend(_vertical_records_from_dataframe(df, source_file=name, sheet=""))
        parsed["rows"] = [r for r in rows if r.get("normalized_fields")]
        return parsed, f"table rows={len(parsed['rows'])} cols={len(df.columns)}"

    if suffix in {".xlsx", ".xls"}:
        rows: list[dict] = []
        xls = pd.ExcelFile(BytesIO(data))
        sheet_notes: list[str] = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
            df = df.dropna(how="all")
            if df.empty:
                sheet_notes.append(f"{sheet_name}:0")
                continue
            df = df.fillna("")
            sheet_rows = []
            for raw_row in df.to_dict(orient="records"):
                record = _row_record(
                    raw_row,
                    source_file=name,
                    sheet=str(sheet_name),
                    orientation="horizontal",
                )
                if record.get("normalized_fields"):
                    sheet_rows.append(record)
            sheet_rows.extend(_vertical_records_from_dataframe(df, source_file=name, sheet=str(sheet_name)))
            rows.extend(sheet_rows)
            sheet_notes.append(f"{sheet_name}:{len(sheet_rows)}")
        parsed["rows"] = rows
        parsed["sheets"] = xls.sheet_names
        return parsed, f"excel rows={len(rows)} sheets={'; '.join(sheet_notes)}"

    if suffix == ".md":
        md_rows = [
            _row_record(r, source_file=name, orientation="markdown_table")
            for r in _parse_markdown_tables(text)
        ]
        parsed["rows"] = [r for r in md_rows if r.get("normalized_fields")]
        kv: dict[str, str] = {}
        for ln in text.splitlines():
            m = re.match(r"^\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,64})\s*[:：]\s*(.+?)\s*$", ln)
            if not m:
                continue
            nk = normalize_student_fact_key(m.group(1))
            vv = m.group(2).strip()
            if nk and vv:
                kv[nk] = vv
        parsed["kv"] = kv
        return parsed, f"markdown rows={len(parsed['rows'])} kv={len(parsed['kv'])}"

    if suffix == ".txt":
        kv: dict[str, str] = {}
        for ln in text.splitlines():
            m = re.match(r"^\s*([A-Za-z0-9_\-\u4e00-\u9fff]{1,64})\s*[:：]\s*(.+?)\s*$", ln)
            if not m:
                continue
            nk = normalize_student_fact_key(m.group(1))
            vv = m.group(2).strip()
            if nk and vv:
                kv[nk] = vv
        parsed["kv"] = kv
        return parsed, f"text kv={len(parsed['kv'])}"

    return parsed, f"unsupported suffix={suffix or 'none'}"


def render_student_profile_panel(
    *,
    st: Any,
    DB: Any,
    current_kb_id: str | None,
    current_conv_id: str | None,
    current_user_id: str,
    enable_profile_llm_review: bool,
    current_state_lib_id: str = "",
    title: str = "🎓 学生状态库（独立检索）",
    expanded: bool = False,
    key_prefix: str = "student_status",
    show_identity_controls: bool = True,
    show_file_management: bool = True,
    force_refresh_next_query: bool = False,
) -> dict[str, Any]:
    """
    Render student-status library panel.
    Returns updated state values.
    """
    conv_row = DB.conv_get(current_conv_id) if current_conv_id else None
    conv_user_id = str((conv_row or {}).get("user_id") or "").strip()
    if conv_user_id != str(current_user_id or "").strip():
        current_user_id = conv_user_id

    with st.expander(title, expanded=expanded):
        bound_user = DB.user_get(current_user_id) if current_user_id else None
        if show_identity_controls:
            st.markdown("**当前会话学生身份**")
            st.caption("这里绑定的是本轮问答要查询的学生身份，不是把学生状态库绑定到某个学生。")
            if bound_user:
                st.caption(
                    "当前学生："
                    f"{bound_user.get('name') or '未命名'}"
                    f" / 学号 {bound_user.get('student_id') or '-'}"
                )
            else:
                st.caption("当前会话未指定学生")

            col_u1, col_u2 = st.columns(2)
            input_student_id = col_u1.text_input(
                "学号",
                value=(bound_user or {}).get("student_id", "") or "",
                key=f"{key_prefix}_student_id::{current_conv_id or 'none'}",
            )
            input_name = col_u2.text_input(
                "姓名",
                value=(bound_user or {}).get("name", "") or "",
                key=f"{key_prefix}_name::{current_conv_id or 'none'}",
            )
            col_u3, col_u4 = st.columns(2)
            input_college = col_u3.text_input(
                "学院",
                value=(bound_user or {}).get("college", "") or "",
                key=f"{key_prefix}_college::{current_conv_id or 'none'}",
            )
            input_major = col_u4.text_input(
                "专业",
                value=(bound_user or {}).get("major", "") or "",
                key=f"{key_prefix}_major::{current_conv_id or 'none'}",
            )

            action_col1, action_col2 = st.columns(2)
            if action_col1.button("设为当前会话学生", key=f"{key_prefix}_bind::{current_conv_id or 'none'}", width="stretch"):
                if not current_conv_id:
                    st.warning("请先创建并选择一个对话。")
                elif not (input_student_id.strip() or input_name.strip()):
                    st.warning("至少填写学号或姓名。")
                else:
                    user = DB.user_upsert(
                        student_id=input_student_id.strip(),
                        name=input_name.strip(),
                        college=input_college.strip(),
                        major=input_major.strip(),
                    )
                    current_user_id = user.get("id", "")
                    DB.conv_bind_user(current_conv_id, current_user_id or None)
                    st.success("已设置当前会话学生。")
                    st.rerun()

            if action_col2.button("清除会话学生", key=f"{key_prefix}_unbind::{current_conv_id or 'none'}", width="stretch"):
                if current_conv_id:
                    DB.conv_bind_user(current_conv_id, None)
                current_user_id = ""
                st.success("已清除当前会话学生。")
                st.rerun()

            enable_profile_llm_review = st.toggle(
                "启用状态检索LLM复核",
                value=bool(enable_profile_llm_review),
                help="仅对检索到的结构化学生状态做简短相关性复核。",
                key=f"{key_prefix}_llm_review::{current_conv_id or 'none'}",
            )

            if bound_user:
                st.markdown("**已入库并绑定到该学生的状态事实**")
                facts = DB.profile_fact_list(bound_user["id"])
                if facts:
                    st.caption("这些事实存储在数据库 `profile_facts` 中，并通过 `user_id` 强绑定到当前学生。")
                    df = pd.DataFrame(
                        [
                            {
                                "字段": f.get("fact_key", ""),
                                "值": f.get("fact_value", ""),
                                "来源": f.get("source", ""),
                                "置信度": round(float(f.get("confidence", 0.0)), 2),
                            }
                            for f in facts
                        ]
                    )
                    st.dataframe(df, hide_index=True, width="stretch")
                else:
                    st.caption("暂无已缓存状态字段。")

                with st.form(f"{key_prefix}_fact_form::{bound_user['id']}"):
                    st.caption("手动新增/更新当前学生状态字段")
                    f1, f2 = st.columns([2, 3])
                    fact_key = f1.text_input("字段key", placeholder="例如: gpa / score / class_name")
                    fact_value = f2.text_input("字段值", placeholder="例如: 3.72")
                    submitted = st.form_submit_button("保存字段", width="stretch")
                    if submitted:
                        if fact_key.strip() and fact_value.strip():
                            DB.profile_fact_upsert(
                                user_id=bound_user["id"],
                                fact_key=fact_key.strip(),
                                fact_value=fact_value.strip(),
                                source="manual_ui",
                                confidence=1.0,
                            )
                            st.success("已保存字段。")
                            st.rerun()
                        else:
                            st.warning("字段key和值都不能为空。")

        st.markdown("**学生状态库选择**" if show_identity_controls else "**学生状态库**")
        if not show_identity_controls:
            st.caption(
                "学生状态库是批量多学生数据源，通常包含成百上千名学生的表格记录；"
                "这里不绑定单个学生。"
            )
        else:
            st.caption("问答时会用当前会话学生身份，从选中的学生状态库里定位对应记录。")

        kb_id_for_doc = current_kb_id
        state_lib_id = str(current_state_lib_id or "").strip()
        if kb_id_for_doc:
            state_libs = DB.student_state_lib_list(kb_id_for_doc)
            if state_libs:
                state_lib_ids = [x["id"] for x in state_libs]
                if state_lib_id not in state_lib_ids:
                    state_lib_id = state_lib_ids[0]
                state_lib_names = [
                    f"{x.get('name', '')}（{int(x.get('doc_count', 0) or 0)} 文件）"
                    for x in state_libs
                ]
                selected_idx = state_lib_ids.index(state_lib_id)
                selected_label = st.selectbox(
                    "当前学生状态库",
                    state_lib_names,
                    index=selected_idx,
                    key=f"{key_prefix}_state_lib_select::{kb_id_for_doc}",
                )
                state_lib_id = state_lib_ids[state_lib_names.index(selected_label)]
                selected_lib = next((x for x in state_libs if x["id"] == state_lib_id), {})
                if selected_lib.get("description"):
                    st.caption(selected_lib["description"])
            else:
                st.caption("当前通用三级知识库下暂无学生状态库。")

            if show_file_management:
                with st.form(f"{key_prefix}_state_lib_form::{kb_id_for_doc}"):
                    st.caption("新建学生状态库")
                    lib_name = st.text_input("状态库名称", placeholder="例：2024秋学业状态")
                    lib_desc = st.text_input("说明", placeholder="例：成绩、绩点、预警、考勤等状态数据")
                    lib_submitted = st.form_submit_button("创建学生状态库", width="stretch")
                    if lib_submitted:
                        if not lib_name.strip():
                            st.warning("状态库名称不能为空。")
                        else:
                            try:
                                lib = DB.student_state_lib_create(
                                    kb_id_for_doc,
                                    lib_name.strip(),
                                    lib_desc.strip(),
                                )
                                state_lib_id = str(lib.get("id", "") or "")
                                st.success("学生状态库已创建。")
                                st.rerun()
                            except Exception as e:
                                msg = str(e)
                                if "UNIQUE constraint failed: student_state_libraries.kb_id, student_state_libraries.name" in msg:
                                    st.warning("同一通用知识库下已存在同名学生状态库，请更换名称。")
                                else:
                                    st.error(f"创建学生状态库失败: {e}")
        else:
            state_libs = []
            st.caption("请先选择通用三级知识库。")

        if show_file_management:
            st.markdown("**学生状态库文件（批量多学生，不切分）**")
            doc_rows = DB.student_doc_list(kb_id_for_doc, state_lib_id) if kb_id_for_doc and state_lib_id else []
            if doc_rows:
                for d in doc_rows[:20]:
                    c_doc, c_del = st.columns([6, 1])
                    c_doc.caption(
                        f"{d.get('name', '')} | ext={d.get('file_ext', '')} | "
                        f"updated={d.get('updated_at', '')}"
                    )
                    if c_del.button("删除", key=f"{key_prefix}_doc_del::{d.get('id','')}", width="stretch"):
                        DB.student_doc_delete(d.get("id", ""))
                        st.rerun()
            else:
                st.caption("当前学生状态库暂无文件。")

            uploaded_student_docs = st.file_uploader(
                "上传学生状态库文件（可多选）",
                type=["json", "jsonl", "csv", "tsv", "xlsx", "xls", "txt", "md"],
                key=f"{key_prefix}_doc_upload::{current_kb_id or 'none'}",
                accept_multiple_files=True,
                help="按文件管理，不切分；可一次选择多个文件批量导入。",
            )
            if uploaded_student_docs:
                st.caption(
                    "已选择文件: "
                    + ", ".join(f.name for f in uploaded_student_docs[:8])
                    + ("" if len(uploaded_student_docs) <= 8 else f" ...（共 {len(uploaded_student_docs)} 个文件）")
                )
                if st.button(
                    "批量导入到学生状态库",
                    key=f"{key_prefix}_doc_import::{current_kb_id or 'none'}",
                    width="stretch",
                ):
                    try:
                        if not kb_id_for_doc:
                            st.warning("请先选择通用三级知识库。")
                        elif not state_lib_id:
                            st.warning("请先创建或选择学生状态库。")
                        else:
                            success_details: list[str] = []
                            failed_details: list[str] = []
                            for uploaded_student_doc in uploaded_student_docs:
                                try:
                                    parsed_doc, parse_note = parse_student_kb_document(uploaded_student_doc)
                                    rows_n = len(parsed_doc.get("rows", []) or [])
                                    kv_n = len(parsed_doc.get("kv", {}) or {})
                                    if rows_n <= 0 and kv_n <= 0:
                                        failed_details.append(
                                            f"{uploaded_student_doc.name}: 未解析到可用结构化内容（{parse_note}）"
                                        )
                                        continue
                                    last_err: Exception | None = None
                                    for attempt in range(3):
                                        try:
                                            DB.student_doc_create(
                                                kb_id=kb_id_for_doc,
                                                state_lib_id=state_lib_id,
                                                name=uploaded_student_doc.name,
                                                file_ext=Path(uploaded_student_doc.name).suffix.lower(),
                                                parsed=parsed_doc,
                                                raw_text=(uploaded_student_doc.getvalue() or b"").decode("utf-8", errors="ignore")[:200000],
                                            )
                                            last_err = None
                                            break
                                        except Exception as db_e:
                                            last_err = db_e
                                            if "database is locked" in str(db_e).lower() and attempt < 2:
                                                time.sleep(0.2 * (attempt + 1))
                                                continue
                                            raise
                                    if last_err is not None:
                                        raise last_err
                                    success_details.append(
                                        f"{uploaded_student_doc.name}: rows={rows_n}, kv={kv_n}（{parse_note}）"
                                    )
                                except Exception as file_e:
                                    failed_details.append(f"{uploaded_student_doc.name}: {file_e}")

                            if success_details:
                                st.success(f"批量导入完成，成功 {len(success_details)} 个文件。")
                                with st.expander("成功明细", expanded=False):
                                    for line in success_details:
                                        st.caption(line)
                            if failed_details:
                                st.warning(f"有失败 {len(failed_details)} 个文件。")
                                with st.expander("失败明细", expanded=True):
                                    for line in failed_details:
                                        st.caption(line)
                            if success_details and not failed_details:
                                st.info("导入结果已完成；如需刷新文件列表可点击页面其它操作或手动切页返回。")
                    except Exception as e:
                        st.error(f"批量导入失败: {e}")

        elif state_lib_id:
            st.caption("文件导入和库管理请到「学生状态库」页面。")

        if show_identity_controls:
            st.markdown("**状态检索控制**")
            st.caption("首次命中后默认仅复用用户画像；点击下方按钮可在下一轮强制重扫学生状态库。")
            r1, r2 = st.columns([1, 1])
            if r1.button(
                "下一轮强制重检索",
                key=f"{key_prefix}_force_refresh::{current_conv_id or 'none'}",
                width="stretch",
                disabled=not bool(current_conv_id),
            ):
                force_refresh_next_query = True
                st.success("已开启：下一轮提问将强制重检索学生状态库。")
            if r2.button(
                "取消强制重检索",
                key=f"{key_prefix}_cancel_refresh::{current_conv_id or 'none'}",
                width="stretch",
                disabled=not bool(current_conv_id),
            ):
                force_refresh_next_query = False
                st.info("已取消强制重检索。")

    return {
        "current_user_id": current_user_id,
        "current_student_state_lib_id": state_lib_id,
        "enable_profile_llm_review": enable_profile_llm_review,
        "force_refresh_next_query": bool(force_refresh_next_query),
    }
