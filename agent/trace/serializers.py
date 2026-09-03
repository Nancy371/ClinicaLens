"""Serialization helpers that never mutate business objects."""

from __future__ import annotations

import dataclasses
import enum
import json
from pathlib import Path
from typing import Any

from .redaction import redact_sensitive


def _to_jsonable(value: Any, depth: int, max_depth: int) -> Any:
    if depth > max_depth:
        return "<max_depth>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value), depth + 1, max_depth)
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item, depth + 1, max_depth)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item, depth + 1, max_depth) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _to_jsonable(to_dict(), depth + 1, max_depth)
        except Exception:
            return repr(value)
    return repr(value)


def _truncate_strings(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "...<truncated>"
    if isinstance(value, dict):
        return {key: _truncate_strings(item, max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_strings(item, max_chars) for item in value]
    return value


def safe_serialize(
    value: Any,
    *,
    redact: bool = True,
    max_string_chars: int = 4000,
    max_depth: int = 8,
) -> Any:
    """Return a detached JSON-compatible copy of ``value``."""
    jsonable = _to_jsonable(value, 0, max_depth)
    if redact:
        jsonable = redact_sensitive(jsonable)
    jsonable = _truncate_strings(jsonable, max_string_chars)
    try:
        return json.loads(json.dumps(jsonable, ensure_ascii=False))
    except Exception:
        return repr(jsonable)


def compact_summary(value: Any, *, max_string_chars: int = 500) -> Any:
    return safe_serialize(
        value,
        redact=True,
        max_string_chars=max_string_chars,
        max_depth=4,
    )
