from __future__ import annotations

from time import perf_counter
from typing import Any


TRACE_FIELDS = (
    "step_name",
    "module",
    "input",
    "output",
    "selected",
    "dropped",
    "prompt",
    "raw_llm_output",
    "fallback",
    "elapsed_s",
    "error",
)


def start_timer() -> float:
    return perf_counter()


def finish_timer(start_at: float) -> float:
    return round(max(0.0, perf_counter() - float(start_at)), 6)


def make_trace_step(
    *,
    step_name: str,
    module: str,
    input_data: Any = None,
    output_data: Any = None,
    selected: Any = None,
    dropped: Any = None,
    prompt: Any = None,
    raw_llm_output: Any = None,
    fallback: Any = None,
    elapsed_s: float | None = None,
    error: str = "",
) -> dict[str, Any]:
    return {
        "step_name": step_name,
        "module": module,
        "input": input_data if input_data is not None else {},
        "output": output_data if output_data is not None else {},
        "selected": selected if selected is not None else [],
        "dropped": dropped if dropped is not None else [],
        "prompt": prompt if prompt is not None else {},
        "raw_llm_output": raw_llm_output if raw_llm_output is not None else {},
        "fallback": fallback if fallback is not None else {},
        "elapsed_s": float(elapsed_s or 0.0),
        "error": str(error or ""),
    }


def append_trace(trace: list[dict[str, Any]], step: dict[str, Any]) -> None:
    if not isinstance(trace, list):
        return
    if not isinstance(step, dict):
        return
    trace.append({k: step.get(k) for k in TRACE_FIELDS})
