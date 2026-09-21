from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{12,}\b"), "[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED]"),
    (re.compile(r"(?i)((?:api[_-]?key|token|secret|password|passwd)\s*[=:]\s*)[^\s,;]+"), r"\1[REDACTED]"),
]


def redact_text(value: str) -> str:
    out = value
    for pattern, repl in _RULES:
        out = pattern.sub(repl, out)
    return out


def sanitize(value: Any, *, max_depth: int = 8, max_items: int = 256, max_string: int = 32_768, _depth: int = 0) -> Any:
    if _depth >= max_depth:
        return "[TRUNCATED_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = redact_text(value)
        return text if len(text) <= max_string else text[:max_string] + "...[TRUNCATED]"
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= max_items:
                out["__truncated__"] = True
                break
            key = redact_text(str(k))
            if str(k).lower() in {"authorization", "api_key", "apikey", "token", "secret", "password", "passwd"}:
                out[key] = "[REDACTED]"
            else:
                out[key] = sanitize(
                    v, max_depth=max_depth, max_items=max_items, max_string=max_string, _depth=_depth + 1
                )
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = list(value)[:max_items]
        items = [
            sanitize(x, max_depth=max_depth, max_items=max_items, max_string=max_string, _depth=_depth + 1) for x in seq
        ]
        if len(value) > max_items:
            items.append("[TRUNCATED_ITEMS]")
        return items
    return sanitize(repr(value), max_depth=max_depth, max_items=max_items, max_string=max_string, _depth=_depth + 1)
