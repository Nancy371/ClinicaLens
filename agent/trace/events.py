"""Trace event envelope."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    trace_id: str
    run_id: str
    case_id: str
    sequence: int
    event_type: str
    payload_schema_version: str
    stage: Optional[str]
    component: Optional[str]
    action: Optional[str]
    round_id: str
    span_id: Optional[str]
    parent_span_id: Optional[str]
    status: str
    started_at: str
    ended_at: str
    duration_ms: int
    input_refs: List[Any] = field(default_factory=list)
    output_refs: List[Any] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "payload_schema_version": self.payload_schema_version,
            "stage": self.stage,
            "component": self.component,
            "action": self.action,
            "round_id": self.round_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "input_refs": list(self.input_refs or []),
            "output_refs": list(self.output_refs or []),
            "payload": dict(self.payload or {}),
        }
