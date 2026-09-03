"""Structured trace support for hospital-agent diagnostic runs."""

from .collector import TraceCollector
from .config import TraceConfig
from .context import TraceContext
from .enums import ArtifactType, TraceEventType, TraceStatus
from .validator import TraceValidator

__all__ = [
    "ArtifactType",
    "TraceCollector",
    "TraceConfig",
    "TraceContext",
    "TraceEventType",
    "TraceStatus",
    "TraceValidator",
]
