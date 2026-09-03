"""Small helpers for tracing spans."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .collector import TraceCollector


@contextmanager
def trace_span(
    collector: TraceCollector | None,
    *,
    stage: str,
    component: str,
    action: str,
) -> Iterator[None]:
    span_id = collector.start_span(stage, component, action) if collector else None
    try:
        yield
    except Exception as exc:
        if collector:
            collector.end_span(span_id, status="failed", payload={"error": str(exc)})
        raise
    else:
        if collector:
            collector.end_span(span_id, status="success")
