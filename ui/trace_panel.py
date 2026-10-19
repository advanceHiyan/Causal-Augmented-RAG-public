from __future__ import annotations

import json
from typing import Any


STEP_NAME_ZH = {
    "Step1_history_assembly": "Step1 对话历史组装",
    "Step2_rag_query_rewrite": "Step2 RAG 查询改写",
    "Step3_student_state_retrieval": "Step3 学生状态库检索",
    "Step3_query_preanalysis": "Step3 Query 预分析",
    "Step4_compare_branches": "Step4 三路并行对比",
    "Step4_retrieval_hybrid": "Step4 三级混合检索",
    "Step4_evidence_selector": "Step4 证据选择器",
    "Step4_generation_stage2": "Step4 生成阶段",
}


def _zh_step_name(name: str) -> str:
    return STEP_NAME_ZH.get(name, name)


def render_trace_panel(*, st: Any, meta: dict[str, Any], title: str = "Trace 面板（统一协议）") -> None:
    trace = meta.get("trace", []) if isinstance(meta, dict) else []
    if not isinstance(trace, list) or not trace:
        return

    with st.expander(title, expanded=False):
        st.caption(f"步骤数={len(trace)}")
        for idx, step in enumerate(trace, start=1):
            if not isinstance(step, dict):
                continue
            step_name = str(step.get("step_name", f"step_{idx}"))
            module = str(step.get("module", ""))
            elapsed = float(step.get("elapsed_s", 0.0) or 0.0)
            header = f"{idx}. {_zh_step_name(step_name)}"
            if module:
                header += f" | {module}"
            header += f" | {elapsed:.3f}s"
            with st.expander(header, expanded=False):
                error = str(step.get("error", "") or "").strip()
                if error:
                    st.error(error)
                st.code(
                    json.dumps(
                        {
                            "输入": step.get("input", {}),
                            "输出": step.get("output", {}),
                            "入选": step.get("selected", []),
                            "丢弃": step.get("dropped", []),
                            "提示词信息": step.get("prompt", {}),
                            "原始模型输出": step.get("raw_llm_output", {}),
                            "回退信息": step.get("fallback", {}),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    language="json",
                )

