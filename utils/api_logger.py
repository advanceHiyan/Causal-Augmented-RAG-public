"""
Debug API call logger.

When debug mode is enabled, every LLM / embedding / rerank call logs
its input and output to data/api_logs/YYYY-MM-DD.jsonl (one file per day).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config.settings import DATA_DIR


class APILogger:
    """Module-level singleton — class variables shared across the process."""

    _enabled: bool = False

    @classmethod
    def enable(cls) -> None:
        cls._enabled = True

    @classmethod
    def disable(cls) -> None:
        cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def log(
        cls,
        call_type: str,          # "llm" | "embed" | "rerank"
        model: str,
        input_data: Any,
        output_data: Any,
        duration_ms: float,
        extra: dict | None = None,
    ) -> None:
        if not cls._enabled:
            return
        log_dir = DATA_DIR / "api_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{today}.jsonl"

        entry = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": call_type,
            "model": model,
            "duration_ms": round(duration_ms, 1),
            "input": _truncate(input_data, 2000),
            "output": _truncate(output_data, 2000),
        }
        if extra:
            entry["extra"] = extra

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _truncate(obj: Any, max_chars: int) -> Any:
    """Recursively truncate long strings for readability."""
    if isinstance(obj, str):
        return obj[:max_chars] + (f"…[{len(obj)-max_chars} more chars]" if len(obj) > max_chars else "")
    if isinstance(obj, list):
        if obj and isinstance(obj[0], float):
            return f"[vector dim={len(obj)}, first5={[round(x,4) for x in obj[:5]]}]"
        return [_truncate(x, max_chars) for x in obj[:10]]
    if isinstance(obj, dict):
        return {k: _truncate(v, max_chars) for k, v in obj.items()}
    return obj
