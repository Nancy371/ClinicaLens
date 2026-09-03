"""Compare Pattern Recall audit between two backend/train runs.

The tool accepts either run directories or ``training_results.jsonl`` files.
It is intentionally read-only and works with partial rows, so it can diagnose
where the Pattern Recall chain broke without requiring a successful final
submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="baseline run dir or training_results.jsonl")
    parser.add_argument("--experiment", required=True, help="experiment run dir or training_results.jsonl")
    parser.add_argument("--output", default="", help="optional JSON output path")
    args = parser.parse_args()

    baseline_rows = _load_rows(Path(args.baseline))
    experiment_rows = _load_rows(Path(args.experiment))
    report = compare_runs(baseline_rows, experiment_rows)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    return 0


def compare_runs(
    baseline_rows: Iterable[Dict[str, Any]],
    experiment_rows: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline = {_patient_id(row): row for row in baseline_rows if _patient_id(row)}
    experiment = {_patient_id(row): row for row in experiment_rows if _patient_id(row)}
    patient_ids = sorted(set(baseline) | set(experiment))
    cases: List[Dict[str, Any]] = []
    rescued = 0
    true_harmed = 0
    harm_breakdown: Dict[str, int] = {}
    for patient_id in patient_ids:
        base = _case_summary(baseline.get(patient_id) or {})
        exp = _case_summary(experiment.get(patient_id) or {})
        expected = exp.get("expected") or base.get("expected")
        base_hit20 = _contains_expected(base.get("candidate_top20"), expected)
        exp_hit20 = _contains_expected(exp.get("candidate_top20"), expected)
        base_submitted_hit = _contains_expected(base.get("submitted"), expected)
        exp_submitted_hit = _contains_expected(exp.get("submitted"), expected)
        rescue = bool(expected and not base_hit20 and exp_hit20)
        raw_harm = bool(expected and (base_hit20 or base_submitted_hit) and not exp_hit20 and not exp_submitted_hit)
        harm_kind = _harm_kind(base, exp, raw_harm)
        true_pattern_harm = harm_kind == "true_pattern_harm"
        if rescue:
            rescued += 1
        if true_pattern_harm:
            true_harmed += 1
        _count_key(harm_breakdown, harm_kind)
        cases.append(
            {
                "patient_id": patient_id,
                "expected": expected,
                "baseline": base,
                "experiment": exp,
                "candidate_rescue": rescue,
                "pattern_harm": true_pattern_harm,
                "raw_outcome_harm": raw_harm,
                "harm_attribution": harm_kind,
                "stage_delta": _stage_delta(base.get("pattern_stages"), exp.get("pattern_stages")),
            }
        )
    total = len(cases)
    return {
        "cases": cases,
        "summary": {
            "case_count": total,
            "candidate_rescue_count": rescued,
            "pattern_harm_count": true_harmed,
            "harm_attribution_counts": harm_breakdown,
            "candidate_rescue_rate": round(rescued / total, 4) if total else 0.0,
            "pattern_harm_rate": round(true_harmed / total, 4) if total else 0.0,
        },
    }


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        path = path / "training_results.jsonl"
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _patient_id(row: Dict[str, Any]) -> str:
    return str(
        row.get("patient_id")
        or row.get("patientId")
        or row.get("case_id")
        or row.get("caseId")
        or ""
    ).strip()


def _case_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    result = _first_dict(row.get("final_result"), row.get("result"), row.get("diagnosis_result"), row)
    report = _first_dict(row.get("evaluation_report"), row.get("report"), row.get("evaluation"))
    metrics = _first_dict(row.get("metrics"), result.get("_metrics"))
    decision = _first_dict(
        result.get("_diagnosis_decision"),
        row.get("diagnosis_decision"),
        result.get("diagnosis_decision"),
    )
    candidates = _candidate_names(
        decision.get("candidates")
        or result.get("candidates")
        or row.get("top_candidates")
        or []
    )
    expected = _expected_diagnoses(report, row)
    return {
        "submitted": _submitted(result, report, row),
        "expected": expected,
        "candidate_top20": candidates[:20],
        "candidate_top5": candidates[:5],
        "candidate_recall_at_20": metrics.get("candidate_recall_at_20"),
        "candidate_recall_at_5": metrics.get("candidate_recall_at_5"),
        "pattern_stages": _pattern_stages(decision),
        "llm_state": _llm_state(row, result, decision),
        "pipeline_state": {
            "fallback_used": bool(row.get("fallback_used") or result.get("fallback_used") or decision.get("fallback_used")),
            "status": str(row.get("status") or result.get("status") or ""),
        },
    }


def _pattern_stages(decision: Dict[str, Any]) -> Dict[str, Any]:
    audit = _first_dict(
        decision.get("pattern_recall_audit"),
        decision.get("pattern_pipeline_audit"),
    )
    admissions = list(decision.get("pattern_candidate_admissions") or [])
    if not audit:
        audit = {
            "proposal_count": len(decision.get("llm_pattern_hypotheses") or []),
            "verification_count": len(decision.get("verified_pattern_hypotheses") or [])
            + len(decision.get("rejected_pattern_hypotheses") or []),
            "signal_count": len(decision.get("pattern_recall_signals") or []),
            "linked_entity_ids": _linked_entities(decision),
            "signal_entity_ids": [
                str(item.get("entity_id") or "")
                for item in decision.get("pattern_recall_signals") or []
                if isinstance(item, dict) and item.get("entity_id")
            ],
        }
    compiler_audit = _first_dict(audit.get("compiler_audit"))
    pipeline_audit = _first_dict(audit.get("pattern_pipeline_audit"))
    return {
        "proposal_count": int(audit.get("proposal_count") or 0),
        "verification_statuses": dict(audit.get("verification_statuses") or {}),
        "rejection_reasons": dict(audit.get("rejection_reasons") or {}),
        "entity_link_count": int(audit.get("entity_link_count") or 0),
        "linked_entity_ids": list(audit.get("linked_entity_ids") or []),
        "signal_count": int(audit.get("signal_count") or 0),
        "signal_modes": dict(audit.get("signal_modes") or {}),
        "signal_entity_ids": list(audit.get("signal_entity_ids") or []),
        "controlled_admission_count": sum(
            1 for item in admissions if isinstance(item, dict) and item.get("admitted_to_controlled_pool")
        ),
        "open_world_admission_count": sum(
            1 for item in admissions if isinstance(item, dict) and item.get("admitted_to_open_world")
        ),
        "admission_entity_ids": [
            str(item.get("entity_id") or "")
            for item in admissions
            if isinstance(item, dict) and item.get("entity_id")
        ],
        "compiler_audit": compiler_audit,
        "pipeline_audit": pipeline_audit,
        "source_generated": {
            key: int((value or {}).get("generated") or 0)
            for key, value in (_first_dict(compiler_audit.get("sources")).items() if compiler_audit else [])
            if isinstance(value, dict)
        },
    }


def _llm_state(row: Dict[str, Any], result: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
    calls = list(row.get("llm_call_audit") or result.get("llm_call_audit") or decision.get("llm_call_audit") or [])
    summary = {
        "thinking_success": bool(row.get("thinking_success") or result.get("thinking_success") or decision.get("thinking_success")),
        "thinking_timeout": bool(row.get("thinking_timeout") or result.get("thinking_timeout") or decision.get("thinking_timeout")),
        "diagnosis_llm_success": bool(
            row.get("diagnosis_llm_success")
            or result.get("diagnosis_llm_success")
            or decision.get("diagnosis_llm_success")
        ),
        "diagnosis_llm_timeout": bool(
            row.get("diagnosis_llm_timeout")
            or result.get("diagnosis_llm_timeout")
            or decision.get("diagnosis_llm_timeout")
        ),
        "fallback_used": bool(row.get("fallback_used") or result.get("fallback_used") or decision.get("fallback_used")),
        "llm_call_audit": calls,
    }
    for call in calls:
        if not isinstance(call, dict):
            continue
        purpose = str(call.get("purpose") or "")
        status = str(call.get("status") or "")
        if "thinking" in purpose and status == "timeout":
            summary["thinking_timeout"] = True
        if "diagnosis" in purpose and status == "timeout":
            summary["diagnosis_llm_timeout"] = True
        if call.get("fallback_used"):
            summary["fallback_used"] = True
    return summary


def _harm_kind(base: Dict[str, Any], exp: Dict[str, Any], raw_harm: bool) -> str:
    if not raw_harm:
        return "no_harm"
    if _timeout_or_fallback(exp) and not _timeout_or_fallback(base):
        return "pipeline_timeout_harm"
    if not _pattern_inputs_comparable(base, exp):
        return "non_comparable_run"
    base_stage = _first_dict(base.get("pattern_stages"))
    exp_stage = _first_dict(exp.get("pattern_stages"))
    signal_changed = int(exp_stage.get("signal_count") or 0) > int(base_stage.get("signal_count") or 0)
    admission_changed = set(exp_stage.get("admission_entity_ids") or []) != set(base_stage.get("admission_entity_ids") or [])
    if signal_changed and admission_changed:
        return "true_pattern_harm"
    if signal_changed:
        return "candidate_ranking_harm"
    return "unattributed_harm"


def _timeout_or_fallback(case: Dict[str, Any]) -> bool:
    llm = _first_dict(case.get("llm_state"))
    pipe = _first_dict(case.get("pipeline_state"))
    return bool(
        llm.get("thinking_timeout")
        or llm.get("diagnosis_llm_timeout")
        or llm.get("fallback_used")
        or pipe.get("fallback_used")
    )


def _pattern_inputs_comparable(base: Dict[str, Any], exp: Dict[str, Any]) -> bool:
    if _timeout_or_fallback(base) != _timeout_or_fallback(exp):
        return False
    base_llm = _first_dict(base.get("llm_state"))
    exp_llm = _first_dict(exp.get("llm_state"))
    for key in ("thinking_success", "thinking_timeout", "diagnosis_llm_success", "diagnosis_llm_timeout", "fallback_used"):
        if bool(base_llm.get(key)) != bool(exp_llm.get(key)):
            return False
    return True


def _count_key(target: Dict[str, int], key: str) -> None:
    text = str(key or "unknown")
    target[text] = int(target.get(text, 0)) + 1


def _stage_delta(base: Any, exp: Any) -> Dict[str, Any]:
    base = _first_dict(base)
    exp = _first_dict(exp)
    keys = [
        "proposal_count",
        "entity_link_count",
        "signal_count",
        "controlled_admission_count",
        "open_world_admission_count",
    ]
    return {
        key: int(exp.get(key) or 0) - int(base.get(key) or 0)
        for key in keys
    }


def _linked_entities(decision: Dict[str, Any]) -> List[str]:
    result: List[str] = []
    for item in (
        list(decision.get("verified_pattern_hypotheses") or [])
        + list(decision.get("rejected_pattern_hypotheses") or [])
    ):
        if not isinstance(item, dict):
            continue
        for link in item.get("entity_links") or []:
            if isinstance(link, dict) and link.get("entity_id"):
                result.append(str(link.get("entity_id")))
    return list(dict.fromkeys(result))


def _candidate_names(items: Any) -> List[str]:
    names: List[str] = []
    for item in items or []:
        if isinstance(item, dict):
            name = item.get("diagnosis") or item.get("canonical_name") or item.get("name")
        else:
            name = str(item)
        if name:
            names.append(str(name))
    return names


def _submitted(result: Dict[str, Any], report: Dict[str, Any], row: Dict[str, Any]) -> List[str]:
    values = (
        result.get("diagnosis")
        or result.get("_submitter_final")
        or row.get("submitted_diagnoses")
        or row.get("submitted")
        or ((report.get("diagnosisDetail") or {}).get("submitted") if isinstance(report, dict) else None)
        or []
    )
    return [str(item) for item in _as_list(values) if str(item)]


def _expected_diagnoses(report: Dict[str, Any], row: Dict[str, Any]) -> List[str]:
    values = (
        row.get("expected")
        or row.get("expected_diagnoses")
        or ((report.get("diagnosisDetail") or {}).get("expected") if isinstance(report, dict) else None)
        or ((report.get("ground_truth") or {}).get("final_diagnosis") if isinstance(report, dict) else None)
        or []
    )
    return [str(item) for item in _as_list(values) if str(item)]


def _contains_expected(values: Any, expected: Any) -> bool:
    candidates = {str(item).strip() for item in _as_list(values) if str(item).strip()}
    targets = {str(item).strip() for item in _as_list(expected) if str(item).strip()}
    return bool(candidates & targets)


def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


if __name__ == "__main__":
    raise SystemExit(main())
