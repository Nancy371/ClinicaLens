"""Run frozen random-seed training batches and write comparable metrics."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from agent import MyDoctorAgent
from hospital_agent.base import summarize_training_results
from train import load_config, setup_logging


LOGGER = logging.getLogger("frozen_training")
REQUIRED_ENV = (
    "SERVICE_BASE_URL",
    "SERVICE_TRAIN_TOKEN",
    "MODEL_API_KEY",
    "TEAM_ID",
)
SHADOW_REF_FILES = {
    "pending_diagnostic_rules.json",
    "exam_aliases_pending.json",
}


def _runtime_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for directory in (root / "agent", root / "hospital_agent"):
        files.extend(sorted(directory.glob("*.py")))
    files.extend([root / "config.yaml", root / "train.py"])
    ref_dir = root / "data" / "ref_data"
    files.extend(
        path
        for path in sorted(ref_dir.glob("*.json"))
        if path.name not in SHADOW_REF_FILES
    )
    return [path for path in files if path.is_file()]


def _snapshot(root: Path) -> Dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _runtime_files(root)
    }


def _snapshot_id(snapshot: Dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(snapshot.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _changed_files(before: Dict[str, str], after: Dict[str, str]) -> List[str]:
    names = set(before) | set(after)
    return sorted(name for name in names if before.get(name) != after.get(name))


def _validate_environment() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "missing required environment variables: " + ", ".join(missing)
        )
    base_url = os.environ["SERVICE_BASE_URL"].strip()
    if not base_url.startswith(("http://", "https://")):
        raise RuntimeError("SERVICE_BASE_URL must start with http:// or https://")


def _aggregate(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return summarize_training_results(list(rows))


async def _run(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_environment()
    base_config = load_config(str((ROOT / args.config).resolve()))
    initial_snapshot = _snapshot(ROOT)
    batch_reports: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    for seed in args.seeds:
        before = _snapshot(ROOT)
        if before != initial_snapshot:
            changed = _changed_files(initial_snapshot, before)
            raise RuntimeError(
                "active runtime changed before seed "
                f"{seed}: {', '.join(changed)}"
            )

        config = copy.deepcopy(base_config)
        train_config = config.setdefault("train", {})
        train_config.update(
            {
                "selection": "random",
                "patient_count": args.patient_count,
                "random_seed": seed,
                "patient_ids": [],
            }
        )
        learning = config.setdefault("learning", {})
        learning["freeze_active_knowledge"] = True
        learning["auto_promote_exam_aliases"] = False

        LOGGER.info("starting frozen batch: seed=%s cases=%s", seed, args.patient_count)
        agent = MyDoctorAgent(config)
        try:
            run_result = await agent.run_train()
        finally:
            await agent._cleanup()

        after = _snapshot(ROOT)
        changed = _changed_files(initial_snapshot, after)
        if changed:
            raise RuntimeError(
                "active runtime changed during seed "
                f"{seed}: {', '.join(changed)}"
            )

        rows = list(run_result.get("results") or [])
        all_rows.extend(rows)
        report = {
            "seed": seed,
            "patient_count": args.patient_count,
            "run_dir": run_result.get("run_dir", ""),
            "summary_file": run_result.get("summary_file", ""),
            **(run_result.get("summary") or _aggregate(rows)),
        }
        batch_reports.append(report)
        LOGGER.info("finished frozen batch seed=%s: %s", seed, json.dumps(report, ensure_ascii=False))

    return {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "active_snapshot_sha256": _snapshot_id(initial_snapshot),
        "active_runtime_frozen": True,
        "seeds": list(args.seeds),
        "patient_count_per_seed": args.patient_count,
        "batches": batch_reports,
        "overall": _aggregate(all_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[46, 47, 48])
    parser.add_argument("--patient-count", type=int, default=5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.patient_count < 1:
        parser.error("--patient-count must be at least 1")

    setup_logging()
    try:
        report = asyncio.run(_run(args))
    except RuntimeError as exc:
        LOGGER.error("frozen training aborted: %s", exc)
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        output = (ROOT / args.output).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = ROOT / "outputs" / "frozen_training" / stamp / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
    LOGGER.info("frozen training report: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
