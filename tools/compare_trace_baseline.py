"""Compare generated traces with golden baseline summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _case_key(path: Path) -> str:
    return path.stem.lower().replace("patient_", "patient_")


def _latest_trace_for_case(trace_root: Path, case_id: str) -> Path | None:
    case_dir = trace_root / case_id
    if not case_dir.exists():
        candidates = list(trace_root.rglob("trace.json"))
    else:
        candidates = list(case_dir.rglob("trace.json"))
    matching: List[Path] = []
    for candidate in candidates:
        try:
            trace = _load_json(candidate)
        except Exception:
            continue
        if str(trace.get("case_id", "")).lower() == case_id.lower():
            matching.append(candidate)
    if not matching:
        return None
    return max(matching, key=lambda item: item.stat().st_mtime)


def _final_submission(trace: Dict[str, Any]) -> List[str]:
    events = trace.get("events") or []
    for event in reversed(events):
        if event.get("event_type") == "submission.created":
            payload = event.get("payload") or {}
            diagnoses = payload.get("submitted_diagnoses")
            if isinstance(diagnoses, list):
                return [str(item) for item in diagnoses]
    for event in reversed(events):
        if event.get("event_type") == "trace.completed":
            summary = ((event.get("payload") or {}).get("final_result_summary") or {})
            diagnoses = summary.get("diagnosis")
            if isinstance(diagnoses, list):
                return [str(item) for item in diagnoses]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir")
    parser.add_argument("trace_dir")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    trace_dir = Path(args.trace_dir)
    comparisons = []
    ok = True
    for baseline_path in sorted(baseline_dir.glob("patient_*.json")):
        baseline = _load_json(baseline_path)
        case_id = str(baseline.get("case_id") or baseline.get("patient_id") or baseline_path.stem)
        trace_path = _latest_trace_for_case(trace_dir, case_id)
        if trace_path is None:
            comparisons.append({"case_id": case_id, "status": "missing_trace"})
            ok = False
            continue
        trace = _load_json(trace_path)
        expected = baseline.get("final_submission")
        actual = _final_submission(trace)
        status = "matched"
        if isinstance(expected, list) and [str(item) for item in expected] != actual:
            status = "submission_mismatch"
            ok = False
        comparisons.append(
            {
                "case_id": case_id,
                "status": status,
                "trace": str(trace_path),
                "expected_submission": expected,
                "actual_submission": actual,
            }
        )
    print(json.dumps({"ok": ok, "comparisons": comparisons}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
