from __future__ import annotations

from typing import Any

from ui.student_profile_panel import render_student_profile_panel


def render_student_state_panel(
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
    Stable student-state panel entry for app orchestration.
    Kept as a wrapper for backward compatibility while internals evolve.
    """
    return render_student_profile_panel(
        st=st,
        DB=DB,
        current_kb_id=current_kb_id,
        current_conv_id=current_conv_id,
        current_user_id=current_user_id,
        enable_profile_llm_review=enable_profile_llm_review,
        current_state_lib_id=current_state_lib_id,
        title=title,
        expanded=expanded,
        key_prefix=key_prefix,
        show_identity_controls=show_identity_controls,
        show_file_management=show_file_management,
        force_refresh_next_query=force_refresh_next_query,
    )
