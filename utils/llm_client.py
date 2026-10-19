"""LLM client wrapping DeepSeek / OpenRouter / Gemini."""

from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
import traceback
import uuid
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

from openai import OpenAI
from loguru import logger

from config.settings import settings, LLMConfig, BASE_DIR


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self._cfg = config or settings.llm
        self._clients: dict[str, OpenAI] = {}
        self._rate_lock = threading.Lock()
        self._provider_semaphores: dict[str, threading.Semaphore] = {}
        self._provider_semaphore_limits: dict[str, int] = {}
        self._provider_concurrency_overrides: dict[str, int] = {}
        self._provider_429_wait_s: dict[str, float] = {}
        self._provider_429_total_wait_s: dict[str, float] = {}
        self._console_progress_enabled = os.environ.get("LLM_CONSOLE_PROGRESS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _model_for(self, provider: str) -> str:
        if provider == "deepseek":
            return self._cfg.deepseek_model
        if provider == "openrouter":
            return self._cfg.openrouter_model
        if provider in {"gemini", "gemini_paid"}:
            return self._cfg.gemini_model
        raise ValueError(f"Unknown provider: {provider}")

    def _get_client(self, provider: str) -> OpenAI:
        if provider not in self._clients:
            if provider == "deepseek":
                if not self._cfg.deepseek_api_key:
                    raise ValueError("Missing DEEPSEEK_API_KEY")
                self._clients[provider] = OpenAI(
                    api_key=self._cfg.deepseek_api_key,
                    base_url=self._cfg.deepseek_base_url,
                )
            elif provider == "openrouter":
                if not self._cfg.openrouter_api_key:
                    raise ValueError("Missing OPENROUTER_API_KEY")
                self._clients[provider] = OpenAI(
                    api_key=self._cfg.openrouter_api_key,
                    base_url=self._cfg.openrouter_base_url,
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")
        return self._clients[provider]

    def _get_gemini_api_key(self, provider: str) -> str:
        if provider == "gemini_paid":
            return self._cfg.gemini_paid_api_key or self._cfg.gemini_api_key
        return self._cfg.gemini_api_key

    def _deepseek_extra_body_for_model(self, model_name: str) -> dict[str, Any]:
        model = (model_name or "").strip().lower()
        if model == self._cfg.deepseek_v4_pro_model.lower():
            return {
                "thinking": {"type": self._cfg.deepseek_v4_pro_thinking_type},
                "reasoning_effort": self._cfg.deepseek_v4_pro_reasoning_effort,
            }
        if model == self._cfg.deepseek_v4_flash_model.lower():
            return {
                "thinking": {"type": self._cfg.deepseek_v4_flash_thinking_type},
                "reasoning_effort": self._cfg.deepseek_v4_flash_reasoning_effort,
            }
        return {}

    @staticmethod
    def _console_error(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def set_console_progress(self, enabled: bool = True) -> None:
        self._console_progress_enabled = bool(enabled)
        self._console_error(f"[LLM PROGRESS] enabled={self._console_progress_enabled}")

    def _console_progress(self, message: str) -> None:
        if self._console_progress_enabled:
            print(message, file=sys.stderr, flush=True)

    @staticmethod
    def _exception_status_code(err: Exception) -> int | None:
        for attr in ("status_code", "status"):
            value = getattr(err, attr, None)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
        response = getattr(err, "response", None)
        value = getattr(response, "status_code", None) if response is not None else None
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
        text = str(err)
        for code in (429, 500, 502, 503, 504):
            if str(code) in text:
                return code
        return None

    @classmethod
    def _is_retryable_exception(cls, err: Exception) -> bool:
        status = cls._exception_status_code(err)
        if status in {429, 500, 502, 503, 504}:
            return True
        text = str(err).lower()
        return any(token in text for token in ["rate", "quota", "timeout", "temporarily", "try again"])

    def _base_429_wait_s(self) -> float:
        return max(1.0, float(getattr(self._cfg, "llm_rate_limit_cooldown_s", 30.0)))

    def _max_429_wait_s(self) -> float:
        return max(self._base_429_wait_s(), float(getattr(self._cfg, "llm_rate_limit_max_wait_s", 300.0)))

    def _provider_429_wait(self, provider: str) -> float:
        provider = (provider or "default").strip().lower()
        with self._rate_lock:
            return float(self._provider_429_wait_s.get(provider, self._base_429_wait_s()))

    def _record_429_wait(self, provider: str, wait_s: float) -> float:
        provider = (provider or "default").strip().lower()
        with self._rate_lock:
            total = float(self._provider_429_total_wait_s.get(provider, 0.0)) + max(0.0, float(wait_s))
            self._provider_429_total_wait_s[provider] = total
            return total

    def _increase_429_wait(self, provider: str) -> float:
        provider = (provider or "default").strip().lower()
        with self._rate_lock:
            current = float(self._provider_429_wait_s.get(provider, self._base_429_wait_s()))
            new_wait = min(self._max_429_wait_s(), max(self._base_429_wait_s(), current * 2.0))
            self._provider_429_wait_s[provider] = new_wait
            return new_wait

    def _reset_429_wait(self, provider: str) -> None:
        provider = (provider or "default").strip().lower()
        with self._rate_lock:
            self._provider_429_wait_s[provider] = self._base_429_wait_s()

    def _provider_semaphore(self, provider: str) -> threading.Semaphore:
        provider = (provider or "default").strip().lower()
        limit = max(
            1,
            int(
                self._provider_concurrency_overrides.get(
                    provider,
                    int(getattr(self._cfg, "llm_max_concurrent_calls", 4)),
                )
            ),
        )
        with self._rate_lock:
            if (
                provider not in self._provider_semaphores
                or self._provider_semaphore_limits.get(provider) != limit
            ):
                self._provider_semaphores[provider] = threading.Semaphore(limit)
                self._provider_semaphore_limits[provider] = limit
            return self._provider_semaphores[provider]

    def get_provider_concurrency(self, provider: str) -> int:
        provider = (provider or "default").strip().lower()
        return max(
            1,
            int(
                self._provider_concurrency_overrides.get(
                    provider,
                    int(getattr(self._cfg, "llm_max_concurrent_calls", 4)),
                )
            ),
        )

    def set_provider_concurrency(self, provider: str, max_calls: int) -> int:
        provider = (provider or "default").strip().lower()
        limit = max(1, int(max_calls))
        with self._rate_lock:
            self._provider_concurrency_overrides[provider] = limit
            self._provider_semaphores[provider] = threading.Semaphore(limit)
            self._provider_semaphore_limits[provider] = limit
        self._console_error(f"[LLM CONCURRENCY] provider={provider} max_concurrent_calls={limit}")
        return limit

    @staticmethod
    def _extract_codeblock_json(raw: str) -> str | None:
        import re

        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw or "")
        if not match:
            return None
        text = (match.group(1) or "").strip()
        return text or None

    @staticmethod
    def _extract_balanced_json(raw: str) -> str | None:
        text = raw or ""
        start = -1
        for idx, ch in enumerate(text):
            if ch in "{[":
                start = idx
                break
        if start < 0:
            return None

        stack: list[str] = []
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch in "{[":
                stack.append(ch)
                continue
            if ch in "}]":
                if not stack:
                    continue
                opener = stack[-1]
                if (opener == "{" and ch == "}") or (opener == "[" and ch == "]"):
                    stack.pop()
                    if not stack:
                        return text[start : i + 1]
        return None

    @staticmethod
    def _auto_close_json(raw: str) -> str | None:
        text = (raw or "").strip()
        start = -1
        for idx, ch in enumerate(text):
            if ch in "{[":
                start = idx
                break
        if start < 0:
            return None

        text = text[start:]
        stack: list[str] = []
        in_string = False
        escaped = False
        for ch in text:
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in "{[":
                stack.append(ch)
                continue
            if ch in "}]":
                if not stack:
                    continue
                opener = stack[-1]
                if (opener == "{" and ch == "}") or (opener == "[" and ch == "]"):
                    stack.pop()
        if in_string:
            text += '"'
        while stack:
            opener = stack.pop()
            text += "}" if opener == "{" else "]"
        return text

    @classmethod
    def _parse_json_robust(cls, raw: str) -> Any:
        def _truncate_to_last_comma(text: str) -> str | None:
            if not text:
                return None
            in_string = False
            escaped = False
            depth = 0
            comma_positions: list[int] = []
            for i, ch in enumerate(text):
                if in_string:
                    if escaped:
                        escaped = False
                        continue
                    if ch == "\\":
                        escaped = True
                        continue
                    if ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                    continue
                if ch in "{[":
                    depth += 1
                    continue
                if ch in "}]":
                    depth = max(0, depth - 1)
                    continue
                if ch == "," and depth >= 1:
                    comma_positions.append(i)
            if not comma_positions:
                return None
            return text[: comma_positions[-1]]

        candidates: list[str] = []
        base = (raw or "").strip()
        if base:
            candidates.append(base)

        codeblock = cls._extract_codeblock_json(base)
        if codeblock:
            candidates.append(codeblock)

        balanced = cls._extract_balanced_json(base)
        if balanced:
            candidates.append(balanced)

        closed = cls._auto_close_json(base)
        if closed:
            candidates.append(closed)
            # For truncated objects like {"a":1,"b":{"x":2},"vector_
            # keep backing off to previous comma and auto-close again.
            current = closed
            for _ in range(6):
                cut = _truncate_to_last_comma(current)
                if not cut:
                    break
                current = cut
                repaired = cls._auto_close_json(current)
                if repaired:
                    candidates.append(repaired)

        seen: set[str] = set()
        for text in candidates:
            t = text.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                continue
        raise json.JSONDecodeError("Unable to parse JSON", base[:200], 0)

    @staticmethod
    def _log_json_parse_failure(
        *,
        provider: str,
        model: str,
        raw: str,
        err: Exception,
        attempt_idx: int,
        attempts: int,
        messages: list[dict[str, str]],
    ) -> str:
        log_dir = BASE_DIR / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "llm_json_parse_errors.jsonl"
        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": provider,
            "model": model,
            "attempt": f"{attempt_idx}/{attempts}",
            "error_type": type(err).__name__,
            "error_message": str(err),
            "traceback": traceback.format_exc(),
            "raw_len": len(raw or ""),
            "raw_full": raw or "",
            "request_messages": messages,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return str(log_path)

    @staticmethod
    def _messages_to_gemini_payload(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, Any]]]:
        system_texts: list[str] = []
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                system_texts.append(content)
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]
        return "\n\n".join(system_texts), contents

    @staticmethod
    def _extract_gemini_text(payload: dict) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini empty candidates: {payload}")
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        texts = [str(p.get("text", "")) for p in parts if isinstance(p, dict) and p.get("text")]
        if texts:
            return "\n".join(texts).strip()
        finish = str(candidates[0].get("finishReason", "")).strip()
        raise RuntimeError(f"Gemini response has no text (finishReason={finish})")

    @staticmethod
    def _gemini_error_message(payload: dict, fallback: str = "") -> str:
        err = payload.get("error", {}) if isinstance(payload, dict) else {}
        msg = str(err.get("message", "")).strip()
        status = str(err.get("status", "")).strip()
        if msg and status:
            return f"{status}: {msg}"
        return msg or fallback

    def _chat_gemini(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY")

        system_text, contents = self._messages_to_gemini_payload(messages)
        attempts = max(1, int(self._cfg.gemini_retry_max_attempts))
        base_delay = max(0.2, float(self._cfg.gemini_retry_base_delay_s))
        url = (
            f"{self._cfg.gemini_base_url}/models/{model_name}:generateContent"
            f"?key={api_key}"
        )
        last_error_msg = "unknown error"

        for attempt in range(1, attempts + 1):
            body: dict[str, Any] = {
                "contents": contents,
                "generationConfig": {
                    "temperature": float(temperature),
                    "maxOutputTokens": int(max_tokens),
                },
            }
            if system_text:
                body["systemInstruction"] = {"parts": [{"text": system_text}]}
            if json_mode:
                body["generationConfig"]["responseMimeType"] = "application/json"

            req = url_request.Request(
                url=url,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            t0 = time.time()
            status_code = 0
            raw_text = ""
            output_text = ""
            try:
                with self._provider_semaphore("gemini"):
                    with url_request.urlopen(req, timeout=90) as resp:
                        status_code = int(getattr(resp, "status", 200))
                        raw_text = resp.read().decode("utf-8", errors="replace")
                payload = json.loads(raw_text) if raw_text else {}
                output_text = self._extract_gemini_text(payload)
                duration_ms = (time.time() - t0) * 1000
                try:
                    from utils.api_logger import APILogger

                    APILogger.log(
                        call_type="llm",
                        model=model_name,
                        input_data=messages,
                        output_data=output_text,
                        duration_ms=duration_ms,
                        extra={
                            "provider": "gemini",
                            "json_mode": json_mode,
                            "attempt": attempt,
                            "status_code": status_code,
                        },
                    )
                except Exception:
                    pass
                self._reset_429_wait("gemini")
                return output_text
            except url_error.HTTPError as e:
                status_code = int(getattr(e, "code", 0) or 0)
                raw_text = e.read().decode("utf-8", errors="replace")
                try:
                    payload = json.loads(raw_text) if raw_text else {}
                except Exception:
                    payload = {}
                last_error_msg = self._gemini_error_message(payload, fallback=raw_text[:240])
                retryable = status_code in {429, 500, 502, 503, 504}
            except Exception as e:
                last_error_msg = str(e)
                retryable = any(
                    token in last_error_msg.lower()
                    for token in ["rate", "quota", "resource_exhausted", "429", "503", "timeout"]
                )

            if retryable and attempt < attempts:
                wait_s = min(10.0, base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 0.3))
                if status_code == 429:
                    wait_s = self._provider_429_wait("gemini")
                    total_wait = self._record_429_wait("gemini", wait_s)
                else:
                    total_wait = 0.0
                logger.warning(
                    "Gemini retry {}/{} | model={} | status={} | wait={:.2f}s | err={}",
                    attempt,
                    attempts,
                    model_name,
                    status_code,
                    wait_s,
                    last_error_msg,
                )
                self._console_error(
                    f"[LLM API RETRY] provider=gemini model={model_name} "
                    f"attempt={attempt}/{attempts} status={status_code} wait={wait_s:.2f}s "
                    f"process_wait_total={total_wait:.2f}s error={last_error_msg}"
                )
                time.sleep(wait_s)
                continue
            if status_code == 429:
                next_wait = self._increase_429_wait("gemini")
                self._console_error(
                    f"[LLM API BACKOFF INCREASE] provider=gemini next_wait={next_wait:.2f}s max={self._max_429_wait_s():.2f}s"
                )

            logger.error(
                "Gemini failed | model={} | attempt={}/{} | status={} | err={}",
                model_name,
                attempt,
                attempts,
                status_code,
                last_error_msg,
            )
            self._console_error(
                f"[LLM API ERROR] provider=gemini model={model_name} "
                f"attempt={attempt}/{attempts} status={status_code} error={last_error_msg}"
            )
            raise RuntimeError(
                f"Gemini call failed after {attempt}/{attempts} attempts: {last_error_msg}"
            )

        raise RuntimeError(f"Gemini call failed: {last_error_msg}")

    def chat(
        self,
        messages: list[dict[str, str]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        provider = (provider or self._cfg.default_provider).strip().lower()
        model_name = model or self._model_for(provider)
        temp = temperature if temperature is not None else self._cfg.temperature
        out_tokens = max_tokens or self._cfg.max_tokens
        # Structured JSON tasks are fragile under low token caps.
        if json_mode:
            out_tokens = max(4096, int(out_tokens))
        # Legacy deepseek-reasoner had a strict 8K output cap.
        # Keep this guard only when callers explicitly pass the legacy alias.
        if provider == "deepseek" and "reasoner" in model_name.lower():
            out_tokens = min(int(out_tokens), 8192)

        if provider in {"gemini", "gemini_paid"}:
            gemini_key = self._get_gemini_api_key(provider)
            return self._chat_gemini(
                messages=messages,
                model_name=model_name,
                api_key=gemini_key,
                temperature=float(temp),
                max_tokens=int(out_tokens),
                json_mode=bool(json_mode),
            )

        client = self._get_client(provider)
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": int(out_tokens),
        }
        # Legacy deepseek-reasoner does not support temperature/top_p style args.
        if not (provider == "deepseek" and "reasoner" in model_name.lower()):
            kwargs["temperature"] = float(temp)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if provider == "deepseek":
            extra_body = self._deepseek_extra_body_for_model(model_name)
            if extra_body:
                kwargs["extra_body"] = extra_body

        t0 = time.time()
        output = ""
        attempts = max(1, int(getattr(self._cfg, "llm_retry_max_attempts", 1)))
        base_delay = max(0.2, float(getattr(self._cfg, "llm_retry_base_delay_s", 1.0)))
        for attempt in range(1, attempts + 1):
            try:
                self._console_progress(
                    f"[LLM START] provider={provider} model={model_name} "
                    f"json={json_mode} attempt={attempt}/{attempts} max_tokens={int(out_tokens)}"
                )
                with self._provider_semaphore(provider):
                    resp = client.chat.completions.create(**kwargs)
                output = resp.choices[0].message.content or ""
                self._reset_429_wait(provider)
                elapsed_s = time.time() - t0
                first = resp.choices[0] if getattr(resp, "choices", None) else None
                msg = getattr(first, "message", None)
                self._console_progress(
                    f"[LLM DONE] provider={provider} model={model_name} "
                    f"json={json_mode} elapsed={elapsed_s:.1f}s "
                    f"finish={getattr(first, 'finish_reason', None) if first else None} "
                    f"content_len={len(output)} reasoning_len={len(getattr(msg, 'reasoning_content', '') or '') if msg else 0}"
                )
                break
            except Exception as e:
                retryable = self._is_retryable_exception(e)
                status_code = self._exception_status_code(e)
                if retryable and attempt < attempts:
                    wait_s = min(20.0, base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5))
                    if status_code == 429:
                        wait_s = self._provider_429_wait(provider)
                        total_wait = self._record_429_wait(provider, wait_s)
                    else:
                        total_wait = 0.0
                    logger.warning(
                        "LLM retry {}/{} | provider={} model={} | status={} | wait={:.2f}s | err={}",
                        attempt,
                        attempts,
                        provider,
                        model_name,
                        status_code,
                        wait_s,
                        e,
                    )
                    retry_label = "LLM 429 WAIT" if status_code == 429 else "LLM API RETRY"
                    self._console_error(
                        f"[{retry_label}] provider={provider} model={model_name} "
                        f"attempt={attempt}/{attempts} status={status_code} wait={wait_s:.2f}s "
                        f"process_wait_total={total_wait:.2f}s error={e}"
                    )
                    time.sleep(wait_s)
                    continue
                if status_code == 429:
                    next_wait = self._increase_429_wait(provider)
                    self._console_error(
                        f"[LLM API BACKOFF INCREASE] provider={provider} "
                        f"next_wait={next_wait:.2f}s max={self._max_429_wait_s():.2f}s"
                    )
                logger.error(f"LLM call failed (provider={provider}, model={model_name}): {e}")
                self._console_error(
                    f"[LLM API ERROR] provider={provider} model={model_name} "
                    f"attempt={attempt}/{attempts} status={status_code} error={e}"
                )
                raise
        try:
            if json_mode and not output.strip():
                try:
                    log_dir = BASE_DIR / "data" / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_path = log_dir / "llm_empty_content.jsonl"
                    first = resp.choices[0] if getattr(resp, "choices", None) else None
                    msg = getattr(first, "message", None)
                    record = {
                        "id": str(uuid.uuid4()),
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "provider": provider,
                        "model": model_name,
                        "json_mode": True,
                        "finish_reason": getattr(first, "finish_reason", None) if first else None,
                        "has_reasoning_content": bool(getattr(msg, "reasoning_content", "")) if msg else False,
                        "reasoning_content_len": len(getattr(msg, "reasoning_content", "") or "") if msg else 0,
                        "content_len": len(getattr(msg, "content", "") or "") if msg else 0,
                        "response_dump": resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
                    }
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    logger.warning(
                        "LLM returned empty content in json_mode | provider={} model={} finish_reason={} debug={}",
                        provider,
                        model_name,
                        record["finish_reason"],
                        str(log_path),
                    )
                    self._console_error(
                        f"[LLM EMPTY JSON] provider={provider} model={model_name} "
                        f"finish={record['finish_reason']} reasoning_len={record['reasoning_content_len']} "
                        f"content_len=0 debug={log_path}"
                    )
                except Exception:
                    pass
        finally:
            duration_ms = (time.time() - t0) * 1000

        try:
            from utils.api_logger import APILogger

            APILogger.log(
                call_type="llm",
                model=model_name,
                input_data=messages,
                output_data=output,
                duration_ms=duration_ms,
                extra={"provider": provider, "json_mode": json_mode},
            )
        except Exception:
            pass

        return output

    def chat_json(
        self,
        messages: list[dict[str, str]],
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        parse_retry: int = 1,
    ) -> Any:
        attempts = max(1, int(parse_retry) + 1)
        last_raw = ""
        last_err: Exception | None = None
        p = (provider or self._cfg.default_provider).strip().lower()
        m = model or self._model_for(p)
        for idx in range(attempts):
            raw = self.chat(
                messages,
                provider=p,
                model=m,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
            )
            last_raw = raw or ""
            try:
                return self._parse_json_robust(last_raw)
            except Exception as e:
                last_err = e
                debug_path = self._log_json_parse_failure(
                    provider=p,
                    model=m,
                    raw=last_raw,
                    err=e,
                    attempt_idx=idx + 1,
                    attempts=attempts,
                    messages=messages,
                )
                if idx < attempts - 1:
                    logger.warning(
                        "JSON parse failed, retrying LLM call ({}/{}) | provider={} model={} | err={} | debug={}",
                        idx + 1,
                        attempts,
                        p,
                        m,
                        str(e)[:180],
                        debug_path,
                    )
                    self._console_error(
                        f"[LLM JSON RETRY] provider={p} model={m} "
                        f"attempt={idx + 1}/{attempts} error={e} debug={debug_path}"
                    )
                    continue
                logger.error(
                    "Failed to parse LLM JSON response | provider={} model={} | err={} | raw_len={} | debug={}",
                    p,
                    m,
                    str(e),
                    len(last_raw),
                    debug_path,
                )
                self._console_error(
                    f"[LLM JSON ERROR] provider={p} model={m} "
                    f"attempt={idx + 1}/{attempts} error={e} debug={debug_path}"
                )
        if last_err:
            raise last_err
        raise RuntimeError("chat_json failed without explicit exception")


llm_client = LLMClient()
