"""Context variables for the active trace."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Optional


@dataclass
class TraceContext:
    trace_id: str
    run_id: str
    case_id: str
    round_id: str = "round_00"
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    sequence: int = 0

    def enter_round(self, round_id: str) -> None:
        self.round_id = str(round_id or self.round_id)


_current_context: ContextVar[Optional[TraceContext]] = ContextVar(
    "diagnostic_trace_context",
    default=None,
)


def get_trace_context() -> Optional[TraceContext]:
    return _current_context.get()


def set_trace_context(context: Optional[TraceContext]) -> Token:
    return _current_context.set(context)


def reset_trace_context(token: Token) -> None:
    _current_context.reset(token)
