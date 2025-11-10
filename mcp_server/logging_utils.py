#!/usr/bin/env python3
"""Per-request logging utilities.

Creates one JSON log file per request under `logs/responses_traces/` named with a UTC timestamp.
Each file captures:
  - request start/end timestamps & total duration
  - an array of function call traces with:
        function name
        args sample (string truncated)
        start/end timestamps
        time taken (seconds)
        CPU user/system deltas
        memory before/after/delta (MB)
        function response (raw object)
  - final response object

Lightweight and dependency-free except for `psutil` for resource metrics.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - will raise clearly if missing
    psutil = None  # fallback; we will record placeholders


LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
TRACE_DIR = os.path.join(LOG_ROOT, "responses_traces")


def _ensure_dirs() -> None:
    os.makedirs(TRACE_DIR, exist_ok=True)


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


class RequestLogger:
    """Accumulates function call traces and writes them to a timestamped JSON file."""

    def __init__(self) -> None:
        _ensure_dirs()
        self.request_start_ts = time.time()
        self.request_start_iso = _utc_now()
        self.traces: list[Dict[str, Any]] = []
        # Precise unique filename using microseconds
        ts_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        self.file_path = os.path.join(TRACE_DIR, f"{ts_name}.json")

    def trace_call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute and record a function call trace with resource usage."""
        start = time.time()
        start_iso = _utc_now()

        proc = psutil.Process() if psutil else None
        cpu_before = proc.cpu_times() if proc else None
        mem_before = proc.memory_info().rss if proc else 0

        result = func(*args, **kwargs)

        end = time.time()
        end_iso = _utc_now()
        cpu_after = proc.cpu_times() if proc else None
        mem_after = proc.memory_info().rss if proc else 0

        cpu_user_delta = (cpu_after.user - cpu_before.user) if (cpu_before and cpu_after) else None
        cpu_system_delta = (cpu_after.system - cpu_before.system) if (cpu_before and cpu_after) else None
        mem_before_mb = mem_before / 1_000_000 if mem_before else None
        mem_after_mb = mem_after / 1_000_000 if mem_after else None
        mem_delta_mb = ((mem_after - mem_before) / 1_000_000) if mem_before and mem_after else None

        # Sample / truncate args for readability
        def _sample(x: Any) -> str:
            s = repr(x)
            return s if len(s) <= 240 else s[:237] + "..."

        args_sample = [_sample(a) for a in args]
        kwargs_sample = {k: _sample(v) for k, v in kwargs.items()}

        trace_entry: Dict[str, Any] = {
            "function": getattr(func, "__name__", "<callable>"),
            "args_sample": args_sample,
            "kwargs_sample": kwargs_sample,
            "start_timestamp": start_iso,
            "end_timestamp": end_iso,
            "time_taken_sec": round(end - start, 6),
            "cpu_user_delta": cpu_user_delta,
            "cpu_system_delta": cpu_system_delta,
            "memory_before_mb": mem_before_mb,
            "memory_after_mb": mem_after_mb,
            "memory_delta_mb": mem_delta_mb,
            "response": result,
        }
        self.traces.append(trace_entry)
        return result

    def finalize(self, final_response: Any) -> None:
        """Write accumulated traces and final response to log file."""
        request_end_ts = time.time()
        request_end_iso = _utc_now()
        total_duration = request_end_ts - self.request_start_ts

        payload = {
            "request_start": self.request_start_iso,
            "request_end": request_end_iso,
            "request_duration_sec": round(total_duration, 6),
            "traces": self.traces,
            "final_response": final_response,
        }

        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:  # pragma: no cover - logging should not break main flow
            # Fallback: write minimal error stub
            fallback = {
                "error": f"Failed to write log: {e}",
                "partial_payload": payload,
            }
            try:
                with open(self.file_path + ".err", "w", encoding="utf-8") as f:
                    json.dump(fallback, f, ensure_ascii=False, indent=2)
            except Exception:
                pass


__all__ = ["RequestLogger"]
