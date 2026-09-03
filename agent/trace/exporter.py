"""Filesystem exporter for append-only trace events and immutable artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class JsonlTraceExporter:
    def __init__(self, trace_dir: Path, *, flush_each_event: bool = True):
        self.trace_dir = Path(trace_dir)
        self.artifacts_dir = self.trace_dir / "artifacts"
        self.events_path = self.trace_dir / "events.jsonl"
        self.flush_each_event = bool(flush_each_event)

    def prepare(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def append_event(self, event: Dict[str, Any]) -> None:
        self.prepare()
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.write("\n")
            if self.flush_each_event:
                handle.flush()
                os.fsync(handle.fileno())

    def write_artifact(self, artifact_id: str, content: Any) -> tuple[str, str, int]:
        self.prepare()
        relative_path = f"artifacts/{artifact_id}.json"
        final_path = self.trace_dir / relative_path
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        data = json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        with tmp_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
        return relative_path, sha256_bytes(data), len(data)
