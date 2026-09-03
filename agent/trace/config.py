"""Trace configuration."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    output_dir: str = "outputs/traces"
    fail_open: bool = True
    capture_artifacts: bool = True
    capture_raw_tool_result: bool = True
    capture_raw_model_prompt: bool = False
    max_inline_payload_chars: int = 4000
    max_artifact_bytes: int = 5_242_880
    flush_each_event: bool = True
    redact_sensitive_fields: bool = True

    @classmethod
    def from_mapping(cls, config: Dict[str, Any] | None) -> "TraceConfig":
        root = config or {}
        data = root.get("trace", root) if isinstance(root, dict) else {}
        if not isinstance(data, dict):
            data = {}
        defaults = cls()
        values = {
            "enabled": bool(data.get("enabled", defaults.enabled)),
            "output_dir": str(data.get("output_dir", defaults.output_dir)),
            "fail_open": bool(data.get("fail_open", defaults.fail_open)),
            "capture_artifacts": bool(
                data.get("capture_artifacts", defaults.capture_artifacts)
            ),
            "capture_raw_tool_result": bool(
                data.get("capture_raw_tool_result", defaults.capture_raw_tool_result)
            ),
            "capture_raw_model_prompt": bool(
                data.get("capture_raw_model_prompt", defaults.capture_raw_model_prompt)
            ),
            "max_inline_payload_chars": int(
                data.get("max_inline_payload_chars", defaults.max_inline_payload_chars)
            ),
            "max_artifact_bytes": int(
                data.get("max_artifact_bytes", defaults.max_artifact_bytes)
            ),
            "flush_each_event": bool(
                data.get("flush_each_event", defaults.flush_each_event)
            ),
            "redact_sensitive_fields": bool(
                data.get("redact_sensitive_fields", defaults.redact_sensitive_fields)
            ),
        }
        return cls(**values)
