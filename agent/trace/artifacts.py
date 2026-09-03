"""Artifact metadata for trace files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class TraceArtifact:
    artifact_id: str
    artifact_type: str
    schema_version: str
    path: str
    created_at: str
    created_by_event_id: str | None
    content_hash: str
    size_bytes: int
    redaction_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "path": self.path,
            "created_at": self.created_at,
            "created_by_event_id": self.created_by_event_id,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "redaction_status": self.redaction_status,
        }
