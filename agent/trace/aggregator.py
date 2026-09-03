"""Build trace.json from events.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class TraceAggregator:
    def aggregate(self, trace_dir: str | Path) -> Dict[str, Any]:
        root = Path(trace_dir)
        events_path = root / "events.jsonl"
        events: List[Dict[str, Any]] = []
        if events_path.exists():
            with events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    events.append(json.loads(text))
        artifacts = [
            (event.get("payload") or {}).get("artifact")
            for event in events
            if event.get("event_type") == "artifact.created"
        ]
        artifacts = [item for item in artifacts if isinstance(item, dict)]
        status = "incomplete"
        if any(event.get("event_type") == "trace.completed" for event in events):
            status = "completed"
        elif any(event.get("event_type") == "trace.failed" for event in events):
            status = "failed"
        trace = {
            "trace_id": events[0].get("trace_id") if events else None,
            "run_id": events[0].get("run_id") if events else None,
            "case_id": events[0].get("case_id") if events else None,
            "status": status,
            "event_count": len(events),
            "artifact_count": len(artifacts),
            "first_event_at": events[0].get("started_at") if events else None,
            "last_event_at": events[-1].get("ended_at") if events else None,
            "events": events,
            "artifacts": artifacts,
        }
        output_path = root / "trace.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(trace, handle, ensure_ascii=False, indent=2)
        return trace
