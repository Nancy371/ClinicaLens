"""Fail-open trace collector."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .aggregator import TraceAggregator
from .artifacts import TraceArtifact
from .config import TraceConfig
from .context import TraceContext, get_trace_context, set_trace_context
from .enums import ArtifactType, TraceEventType, TraceStatus
from .events import TraceEvent
from .exporter import JsonlTraceExporter
from .serializers import compact_summary, safe_serialize
from .validator import TraceValidator

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


class TraceCollector:
    """Append-only trace writer.

    The collector is deliberately a side channel. When ``fail_open`` is true,
    trace failures are logged and swallowed.
    """

    def __init__(self, config: TraceConfig | None = None, *, run_id: str | None = None):
        self.config = config or TraceConfig()
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self.trace_dir: Optional[Path] = None
        self.exporter: Optional[JsonlTraceExporter] = None
        self.context: Optional[TraceContext] = None
        self._lock = threading.Lock()
        self._sequence = 0
        self._artifact_sequence = 0
        self._span_sequence = 0
        self._spans: Dict[str, Dict[str, Any]] = {}
        self._context_token = None

    @classmethod
    def from_config(cls, config: Dict[str, Any] | TraceConfig | None) -> "TraceCollector":
        trace_config = config if isinstance(config, TraceConfig) else TraceConfig.from_mapping(config)
        return cls(trace_config)

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def _guard(self, func, default=None):
        if not self.enabled:
            return default
        try:
            return func()
        except Exception as exc:
            logger.warning("[Trace] collector failure ignored: %s", exc)
            if not self.config.fail_open:
                raise
            return default

    def _next_sequence(self) -> int:
        with self._lock:
            self._sequence += 1
            if self.context is not None:
                self.context.sequence = self._sequence
            return self._sequence

    def _next_event_id(self, sequence: int) -> str:
        return f"evt_{sequence:06d}"

    def _next_artifact_id(self, artifact_type: str) -> str:
        with self._lock:
            self._artifact_sequence += 1
            suffix = self._artifact_sequence
        clean = str(artifact_type).replace(".", "_").replace(":", "_")
        return f"artifact_{clean}_{suffix:06d}"

    def _next_span_id(self, component: str, action: str) -> str:
        with self._lock:
            self._span_sequence += 1
            suffix = self._span_sequence
        clean_component = "".join(ch for ch in str(component or "span") if ch.isalnum() or ch == "_")
        clean_action = "".join(ch for ch in str(action or "action") if ch.isalnum() or ch == "_")
        return f"span_{clean_component}_{clean_action}_{suffix:06d}"

    def start_trace(self, case_id: str, metadata: Dict[str, Any] | None = None) -> Optional[str]:
        def _start() -> str:
            trace_id = f"tr_{uuid.uuid4().hex[:16]}"
            self.trace_dir = Path(self.config.output_dir) / str(case_id) / trace_id
            self.exporter = JsonlTraceExporter(
                self.trace_dir,
                flush_each_event=self.config.flush_each_event,
            )
            self.exporter.prepare()
            self._sequence = 0
            self._artifact_sequence = 0
            self._span_sequence = 0
            self._spans = {}
            self.context = TraceContext(
                trace_id=trace_id,
                run_id=self.run_id,
                case_id=str(case_id),
            )
            self._context_token = set_trace_context(self.context)
            payload = {
                "metadata": safe_serialize(
                    metadata or {},
                    redact=self.config.redact_sensitive_fields,
                    max_string_chars=self.config.max_inline_payload_chars,
                ),
                "training_eligibility": "pending_review",
            }
            self.emit_event(
                TraceEventType.TRACE_STARTED,
                payload=payload,
                stage="case",
                component="TraceCollector",
                action="start_trace",
            )
            return trace_id

        return self._guard(_start)

    def enter_round(self, round_id: str) -> None:
        def _enter() -> None:
            if self.context is not None:
                self.context.enter_round(round_id)
        self._guard(_enter)

    def start_span(self, stage: str, component: str, action: str) -> Optional[str]:
        def _start() -> str:
            context = self.context or get_trace_context()
            if context is None:
                raise RuntimeError("trace has not started")
            span_id = self._next_span_id(component, action)
            parent_span_id = context.span_id
            self._spans[span_id] = {
                "stage": stage,
                "component": component,
                "action": action,
                "parent_span_id": parent_span_id,
                "started_monotonic": time.monotonic(),
                "started_at": utc_now_iso(),
            }
            context.parent_span_id = parent_span_id
            context.span_id = span_id
            self.emit_event(
                TraceEventType.MODULE_STARTED,
                payload={},
                stage=stage,
                component=component,
                action=action,
                span_id=span_id,
                parent_span_id=parent_span_id,
            )
            return span_id

        return self._guard(_start)

    def end_span(self, span_id: str | None, status: str = "success", payload: Dict[str, Any] | None = None) -> None:
        def _end() -> None:
            if not span_id:
                return
            record = self._spans.get(span_id, {})
            event_type = (
                TraceEventType.MODULE_COMPLETED
                if status == TraceStatus.SUCCESS.value or status == "success"
                else TraceEventType.MODULE_FAILED
            )
            duration = 0
            if record.get("started_monotonic") is not None:
                duration = int(max(0.0, time.monotonic() - record["started_monotonic"]) * 1000)
            self.emit_event(
                event_type,
                payload=payload or {},
                stage=record.get("stage"),
                component=record.get("component"),
                action=record.get("action"),
                span_id=span_id,
                parent_span_id=record.get("parent_span_id"),
                status=status,
                started_at=record.get("started_at"),
                duration_ms=duration,
            )
            if self.context is not None and self.context.span_id == span_id:
                self.context.span_id = record.get("parent_span_id")
                self.context.parent_span_id = None

        self._guard(_end)

    def emit_event(
        self,
        event_type: TraceEventType | str,
        payload: Dict[str, Any] | None = None,
        input_refs: List[Any] | None = None,
        output_refs: List[Any] | None = None,
        *,
        stage: str | None = None,
        component: str | None = None,
        action: str | None = None,
        status: str = "success",
        span_id: str | None = None,
        parent_span_id: str | None = None,
        started_at: str | None = None,
        duration_ms: int = 0,
    ) -> Optional[str]:
        def _emit() -> str:
            if self.exporter is None or self.context is None:
                raise RuntimeError("trace has not started")
            sequence = self._next_sequence()
            now = utc_now_iso()
            serialized_payload = safe_serialize(
                payload or {},
                redact=self.config.redact_sensitive_fields,
                max_string_chars=self.config.max_inline_payload_chars,
            )
            context = self.context
            event = TraceEvent(
                event_id=self._next_event_id(sequence),
                trace_id=context.trace_id,
                run_id=context.run_id,
                case_id=context.case_id,
                sequence=sequence,
                event_type=event_type.value if isinstance(event_type, TraceEventType) else str(event_type),
                payload_schema_version="1.0",
                stage=stage,
                component=component,
                action=action,
                round_id=context.round_id,
                span_id=span_id if span_id is not None else context.span_id,
                parent_span_id=(
                    parent_span_id
                    if parent_span_id is not None
                    else context.parent_span_id
                ),
                status=status,
                started_at=started_at or now,
                ended_at=now,
                duration_ms=int(max(0, duration_ms or 0)),
                input_refs=safe_serialize(
                    input_refs or [],
                    redact=self.config.redact_sensitive_fields,
                    max_string_chars=self.config.max_inline_payload_chars,
                ),
                output_refs=safe_serialize(
                    output_refs or [],
                    redact=self.config.redact_sensitive_fields,
                    max_string_chars=self.config.max_inline_payload_chars,
                ),
                payload=serialized_payload if isinstance(serialized_payload, dict) else {"value": serialized_payload},
            )
            self.exporter.append_event(event.to_dict())
            return event.event_id

        return self._guard(_emit)

    def create_artifact(
        self,
        artifact_type: ArtifactType | str,
        content: Any,
        schema_version: str = "1.0",
    ) -> Optional[Dict[str, Any]]:
        def _create() -> Optional[Dict[str, Any]]:
            if self.exporter is None or self.context is None:
                raise RuntimeError("trace has not started")
            if not self.config.capture_artifacts:
                return None
            artifact_type_value = (
                artifact_type.value if isinstance(artifact_type, ArtifactType) else str(artifact_type)
            )
            artifact_id = self._next_artifact_id(artifact_type_value)
            serialized = safe_serialize(
                content,
                redact=self.config.redact_sensitive_fields,
                max_string_chars=self.config.max_inline_payload_chars,
            )
            relative_path, content_hash, size_bytes = self.exporter.write_artifact(
                artifact_id,
                serialized,
            )
            with self._lock:
                created_by_event_id = self._next_event_id(self._sequence + 1)
            metadata = TraceArtifact(
                artifact_id=artifact_id,
                artifact_type=artifact_type_value,
                schema_version=schema_version,
                path=relative_path,
                created_at=utc_now_iso(),
                created_by_event_id=created_by_event_id,
                content_hash=content_hash,
                size_bytes=size_bytes,
                redaction_status="redacted" if self.config.redact_sensitive_fields else "raw",
            ).to_dict()
            event_id = self.emit_event(
                TraceEventType.ARTIFACT_CREATED,
                payload={"artifact": metadata},
                stage="trace",
                component="TraceCollector",
                action="create_artifact",
            )
            if event_id and event_id != metadata["created_by_event_id"]:
                metadata["created_by_event_id"] = event_id
            return metadata

        return self._guard(_create)

    def emit_decision(
        self,
        decision_type: str,
        payload: Dict[str, Any],
        *,
        refs: List[Any] | None = None,
        stage: str = "judge",
        component: str = "DiagnosisJudge",
        action: str = "decision",
    ) -> Optional[str]:
        data = dict(payload or {})
        data.setdefault("decision_type", decision_type)
        return self.emit_event(
            TraceEventType.DECISION_MADE,
            payload=data,
            output_refs=refs or [],
            stage=stage,
            component=component,
            action=action,
        )

    def emit_submission(self, payload: Dict[str, Any], refs: List[Any] | None = None) -> Optional[str]:
        return self.emit_event(
            TraceEventType.SUBMISSION_CREATED,
            payload=payload,
            output_refs=refs or [],
            stage="submitter",
            component="Submitter",
            action="submit_final_result",
        )

    def complete_trace(self, final_result: Dict[str, Any] | None = None) -> None:
        def _complete() -> None:
            output_refs: List[Any] = []
            if final_result is not None:
                artifact = self.create_artifact(
                    ArtifactType.SUBMISSION_RESULT,
                    final_result,
                )
                if artifact:
                    output_refs.append(artifact)
            self.emit_event(
                TraceEventType.TRACE_COMPLETED,
                payload={
                    "final_result_summary": compact_summary(final_result or {}),
                    "training_eligibility": "pending_review",
                },
                output_refs=output_refs,
                stage="case",
                component="TraceCollector",
                action="complete_trace",
            )
            if self.trace_dir is not None:
                TraceAggregator().aggregate(self.trace_dir)
                TraceValidator().validate(self.trace_dir, write_report=True)

        self._guard(_complete)

    def fail_trace(self, error: BaseException | str) -> None:
        def _fail() -> None:
            self.emit_event(
                TraceEventType.TRACE_FAILED,
                payload={
                    "error_type": type(error).__name__ if isinstance(error, BaseException) else "Error",
                    "message": str(error),
                },
                status="failed",
                stage="case",
                component="TraceCollector",
                action="fail_trace",
            )
            if self.trace_dir is not None:
                TraceAggregator().aggregate(self.trace_dir)
                TraceValidator().validate(self.trace_dir, write_report=True)

        self._guard(_fail)
