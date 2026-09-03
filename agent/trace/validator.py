"""Trace structure and reference validator."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .enums import EVENT_TYPE_VALUES
from .exporter import sha256_bytes


REQUIRED_EVENT_FIELDS = (
    "event_id",
    "trace_id",
    "run_id",
    "case_id",
    "sequence",
    "event_type",
    "payload_schema_version",
    "stage",
    "component",
    "action",
    "round_id",
    "span_id",
    "parent_span_id",
    "status",
    "started_at",
    "ended_at",
    "duration_ms",
    "input_refs",
    "output_refs",
    "payload",
)


def _parse_iso(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _iter_refs(event: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for field in ("input_refs", "output_refs"):
        refs = event.get(field) or []
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if isinstance(ref, dict):
                yield ref


class TraceValidator:
    def validate(self, trace_dir: str | Path, *, write_report: bool = True) -> Dict[str, Any]:
        root = Path(trace_dir)
        events_path = root / "events.jsonl"
        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []
        if not events_path.exists():
            errors.append({"code": "MISSING_EVENTS_JSONL", "path": str(events_path)})
        else:
            with events_path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        events.append(json.loads(text))
                    except json.JSONDecodeError as exc:
                        errors.append(
                            {
                                "code": "INVALID_JSONL_LINE",
                                "line": line_no,
                                "message": str(exc),
                            }
                        )
        sequences: List[int] = []
        artifact_by_id: Dict[str, Dict[str, Any]] = {}
        event_ids = set()
        tool_calls: Dict[str, Dict[str, Any]] = {}
        tool_terminal_events: Dict[str, List[Dict[str, Any]]] = {}
        for index, event in enumerate(events):
            for field in REQUIRED_EVENT_FIELDS:
                if field not in event:
                    errors.append(
                        {
                            "code": "MISSING_EVENT_FIELD",
                            "sequence": event.get("sequence"),
                            "field": field,
                        }
                    )
            event_id = event.get("event_id")
            if event_id in event_ids:
                errors.append({"code": "DUPLICATE_EVENT_ID", "event_id": event_id})
            event_ids.add(event_id)
            sequence = event.get("sequence")
            if not isinstance(sequence, int):
                errors.append({"code": "INVALID_SEQUENCE", "sequence": sequence})
            else:
                sequences.append(sequence)
            if event.get("event_type") not in EVENT_TYPE_VALUES:
                errors.append(
                    {
                        "code": "INVALID_EVENT_TYPE",
                        "sequence": sequence,
                        "event_type": event.get("event_type"),
                    }
                )
            if not _parse_iso(event.get("started_at")) or not _parse_iso(event.get("ended_at")):
                errors.append({"code": "INVALID_TIMESTAMP", "sequence": sequence})
            if not isinstance(event.get("duration_ms"), int) or event.get("duration_ms") < 0:
                errors.append({"code": "INVALID_DURATION", "sequence": sequence})
            if event.get("event_type") == "artifact.created":
                artifact = (event.get("payload") or {}).get("artifact")
                if isinstance(artifact, dict) and artifact.get("artifact_id"):
                    artifact_by_id[str(artifact["artifact_id"])] = artifact
                else:
                    errors.append({"code": "INVALID_ARTIFACT_EVENT", "sequence": sequence})
            if event.get("event_type") == "tool.called":
                call_id = (event.get("payload") or {}).get("call_id")
                if not call_id:
                    errors.append({"code": "TOOL_CALL_MISSING_CALL_ID", "sequence": sequence})
                elif call_id in tool_calls:
                    errors.append({"code": "DUPLICATE_TOOL_CALL_ID", "call_id": call_id})
                else:
                    tool_calls[str(call_id)] = event
            if event.get("event_type") in {"tool.returned", "tool.failed"}:
                call_id = (event.get("payload") or {}).get("call_id")
                if not call_id:
                    errors.append({"code": "TOOL_TERMINAL_MISSING_CALL_ID", "sequence": sequence})
                else:
                    tool_terminal_events.setdefault(str(call_id), []).append(event)
            if index == 0 and event.get("event_type") != "trace.started":
                warnings.append({"code": "FIRST_EVENT_NOT_TRACE_STARTED"})
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            errors.append({"code": "SEQUENCE_NOT_STRICTLY_INCREASING"})
        for artifact in artifact_by_id.values():
            rel_path = artifact.get("path")
            content_hash = artifact.get("content_hash")
            if not rel_path:
                errors.append(
                    {
                        "code": "ARTIFACT_MISSING_PATH",
                        "artifact_id": artifact.get("artifact_id"),
                    }
                )
                continue
            artifact_path = root / rel_path
            if not artifact_path.exists():
                errors.append(
                    {
                        "code": "ARTIFACT_FILE_MISSING",
                        "artifact_id": artifact.get("artifact_id"),
                        "path": rel_path,
                    }
                )
                continue
            actual_hash = sha256_bytes(artifact_path.read_bytes())
            if content_hash and actual_hash != content_hash:
                errors.append(
                    {
                        "code": "ARTIFACT_HASH_MISMATCH",
                        "artifact_id": artifact.get("artifact_id"),
                    }
                )
        for event in events:
            for ref in _iter_refs(event):
                artifact_id = ref.get("artifact_id")
                if artifact_id and artifact_id not in artifact_by_id:
                    errors.append(
                        {
                            "code": "UNKNOWN_ARTIFACT_REF",
                            "sequence": event.get("sequence"),
                            "artifact_id": artifact_id,
                        }
                    )
        paired_calls = 0
        for call_id in tool_calls:
            terminal_events = tool_terminal_events.get(call_id, [])
            if len(terminal_events) != 1:
                errors.append(
                    {
                        "code": "TOOL_CALL_TERMINAL_EVENT_COUNT_INVALID",
                        "call_id": call_id,
                        "terminal_event_count": len(terminal_events),
                    }
                )
            else:
                paired_calls += 1
        for call_id in tool_terminal_events:
            if call_id not in tool_calls:
                errors.append({"code": "TOOL_TERMINAL_WITHOUT_CALL", "call_id": call_id})
        completed = any(event.get("event_type") == "trace.completed" for event in events)
        failed = any(event.get("event_type") == "trace.failed" for event in events)
        tool_pairing_rate = (
            round(paired_calls / len(tool_calls), 4)
            if tool_calls
            else 1.0
        )
        report = {
            "schema_valid": not errors,
            "semantic_valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "completeness": {
                "timestamp_rate": 1.0 if events and not any(e.get("code") == "INVALID_TIMESTAMP" for e in errors) else 0.0,
                "artifact_reference_rate": 1.0
                if not any(e.get("code") == "UNKNOWN_ARTIFACT_REF" for e in errors)
                else 0.0,
                "tool_pairing_rate": tool_pairing_rate,
                "submission_traceability_rate": 1.0
                if any(event.get("event_type") == "submission.created" for event in events)
                else 0.0,
                "trace_terminal_event": 1.0 if completed or failed else 0.0,
            },
        }
        if write_report:
            root.mkdir(parents=True, exist_ok=True)
            with (root / "validation_report.json").open("w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
        return report
