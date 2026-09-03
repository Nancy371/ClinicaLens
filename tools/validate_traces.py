"""Validate generated structured traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.trace import TraceValidator  # noqa: E402


def _trace_dirs(path: Path):
    if (path / "events.jsonl").exists():
        yield path
        return
    for events_path in path.rglob("events.jsonl"):
        yield events_path.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Trace directory or parent directory")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warnings")
    args = parser.parse_args()

    validator = TraceValidator()
    roots = list(_trace_dirs(Path(args.path)))
    if not roots:
        print(json.dumps({"error": "no traces found", "path": args.path}, ensure_ascii=False))
        return 1
    reports = []
    ok = True
    for root in roots:
        report = validator.validate(root, write_report=True)
        reports.append({"trace_dir": str(root), **report})
        if not report.get("schema_valid") or not report.get("semantic_valid"):
            ok = False
        if args.strict and report.get("warnings"):
            ok = False
    print(json.dumps({"trace_count": len(reports), "reports": reports}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
