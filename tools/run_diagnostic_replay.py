"""Run deterministic diagnosis replay metrics from a JSONL trace file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.clinical_evidence import ClinicalEvidenceNormalizer
from agent.diagnosis_engine import DiagnosisDecisionEngine
from agent.replay import DiagnosticReplay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="outputs/runtime_state/diagnostic_replay.jsonl",
        help="JSONL replay trace path",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--strict", action="store_true", help="fail when target metrics are missed")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    ref_dir = str((ROOT / config.get("ref_data_dir", "data/ref_data")).resolve())
    engine = DiagnosisDecisionEngine(config, ref_dir=ref_dir)
    normalizer = ClinicalEvidenceNormalizer(ref_dir=ref_dir)
    report = DiagnosticReplay(engine, normalizer).evaluate(str((ROOT / args.path).resolve()))

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output = (ROOT / args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")

    if args.strict:
        targets = report.get("targets") or {}
        for metric, target in targets.items():
            if float(report.get(metric, 0.0)) < float(target):
                return 1
        for metric, target in (report.get("maximum_targets") or {}).items():
            if float(report.get(metric, 0.0)) > float(target):
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
