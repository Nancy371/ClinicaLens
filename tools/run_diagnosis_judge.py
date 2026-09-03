"""Replay DiagnosisJudge decisions from training or diagnostic JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.clinical_evidence import ClinicalEvidenceNormalizer, EvidenceBundle
from agent.diagnosis_engine import CandidateScore, DiagnosisDecision, DiagnosisDecisionEngine


def _load_rows(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
            if limit and len(rows) >= limit:
                break
    return rows


def _names(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    return list(dict.fromkeys(str(item).strip() for item in (value or []) if str(item).strip()))


def _candidate_from_dict(payload: Dict[str, Any]) -> CandidateScore:
    data: Dict[str, Any] = {}
    for item in fields(CandidateScore):
        if item.name in payload:
            data[item.name] = payload[item.name]
    defaults = {
        "diagnosis": str(payload.get("diagnosis") or ""),
        "score": float(payload.get("score", 0.0) or 0.0),
        "support_score": float(payload.get("support_score", 0.0) or 0.0),
        "source_prior": float(payload.get("source_prior", 0.0) or 0.0),
        "explanation_score": float(payload.get("explanation_score", 0.0) or 0.0),
        "coverage_score": float(payload.get("coverage_score", 0.0) or 0.0),
        "residual_score": float(payload.get("residual_score", 1.0) or 1.0),
        "contradiction_penalty": float(payload.get("contradiction_penalty", 0.0) or 0.0),
        "required_met": bool(payload.get("required_met", False)),
        "hard_contradiction": bool(payload.get("hard_contradiction", False)),
    }
    defaults.update(data)
    return CandidateScore(**defaults)


def _decision_from_candidate_payload(
    engine: DiagnosisDecisionEngine,
    candidates: Iterable[Dict[str, Any]],
) -> DiagnosisDecision:
    scores = [_candidate_from_dict(item) for item in candidates if item.get("diagnosis")]
    decision = DiagnosisDecision(
        final_diagnoses=[],
        trusted_diagnoses=[],
        candidates=scores,
        unexplained_evidence=[],
        confidence=float(scores[0].score if scores else 0.0),
        margin=0.0,
        low_confidence=False,
    )
    engine.judge_and_submit(decision)
    return decision


def _decision_from_replay_payload(
    engine: DiagnosisDecisionEngine,
    normalizer: ClinicalEvidenceNormalizer,
    row: Dict[str, Any],
) -> Optional[DiagnosisDecision]:
    evidence_payload = row.get("evidence") or (row.get("audit") or {}).get("evidence") or {}
    evidence = EvidenceBundle.from_dict(evidence_payload) if evidence_payload else None
    if evidence is None or not evidence.observations:
        if not (row.get("collected_info") or row.get("exam_results")):
            return None
        evidence = normalizer.normalize(
            row.get("collected_info") or {},
            row.get("exam_results") or {},
        )
    prior_names = row.get("llm_candidates") or row.get("top_candidates") or []
    rag_chunks = row.get("rag_chunks") or []
    return engine.decide({"diagnosis": prior_names}, rag_chunks, evidence)


def _priority_expected(engine: DiagnosisDecisionEngine, expected: List[str]) -> List[str]:
    priority: List[str] = []
    for name in expected:
        entry = engine.knowledge.get(name)
        dtype = str(entry.get("diagnosis_type") or "").lower()
        try:
            specificity = float(entry.get("specificity", 0.0) or 0.0)
        except (TypeError, ValueError):
            specificity = 0.0
        if dtype in {"etiology", "metabolic", "structural", "systemic"} or specificity >= 0.85:
            priority.append(name)
    return priority


def _audit_row(
    engine: DiagnosisDecisionEngine,
    normalizer: ClinicalEvidenceNormalizer,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    patient_id = row.get("patient_id") or row.get("case_id") or ""
    expected = _names(
        row.get("expected")
        or row.get("expected_diagnoses")
        or ((row.get("diagnosisDetail") or {}).get("expected"))
    )
    submitted = _names(row.get("submitted_diagnoses") or ((row.get("final_result") or {}).get("diagnosis")))

    embedded_decision = (
        (row.get("audit") or {}).get("diagnosis_decision")
        or (row.get("final_result") or {}).get("_diagnosis_decision")
        or row.get("diagnosis_decision")
        or {}
    )
    candidate_payload = embedded_decision.get("candidates") if isinstance(embedded_decision, dict) else None
    decision: Optional[DiagnosisDecision] = None
    replayable = False
    if candidate_payload:
        decision = _decision_from_candidate_payload(engine, candidate_payload)
        replayable = True
    else:
        decision = _decision_from_replay_payload(engine, normalizer, row)
        replayable = decision is not None

    if decision is None:
        top = _names(row.get("top_candidates"))
        retriever_top1 = str(row.get("retriever_top1") or (top[0] if top else "") or "")
        judge_primary = str(row.get("judge_primary") or "")
        final = _names(row.get("submitter_final") or submitted)
        gap_authorized = _names(row.get("required_gap_authorized_diagnoses"))
        judge_payload = (
            (row.get("audit") or {}).get("diagnosis_decision", {}).get("judge_decision")
            or {}
        )
        if not gap_authorized and (row.get("metrics") or {}).get("judge_gap_authorization_rate"):
            gap_authorized = ["<present>"]
        override = bool(retriever_top1 and judge_primary and retriever_top1 != judge_primary)
    else:
        retriever_top1 = decision.retriever_top1 or (decision.candidates[0].diagnosis if decision.candidates else "")
        judge_primary = decision.judge_primary or (decision.final_diagnoses[0] if decision.final_diagnoses else "")
        final = list(decision.final_diagnoses)
        gap_authorized = list(decision.required_gap_authorized_diagnoses)
        judge_payload = dict(decision.judge_decision or {})
        override = bool(decision.decision_override)

    priority = _priority_expected(engine, expected)
    return {
        "patient_id": patient_id,
        "expected": expected,
        "retriever_top1": retriever_top1,
        "judge_primary": judge_primary,
        "submitted": final,
        "original_submitted": submitted,
        "decision_override": override,
        "etiology_preference": (
            any(name in set(final) for name in priority) if priority else None
        ),
        "judge_primary_accuracy": (
            bool(expected and judge_primary in set(expected)) if expected else None
        ),
        "gap_authorized": bool(gap_authorized),
        "required_gap_authorized_diagnoses": gap_authorized,
        "differential_candidates": list(judge_payload.get("differential_candidates") or []),
        "pairwise_comparisons": list(judge_payload.get("pairwise_comparisons") or []),
        "discriminating_findings": list(judge_payload.get("discriminating_findings") or []),
        "discriminating_exams": list(judge_payload.get("discriminating_exams") or []),
        "replayable": replayable,
    }


def run(path: Path, config_path: Path, limit: Optional[int] = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    ref_dir = str((ROOT / config.get("ref_data_dir", "data/ref_data")).resolve())
    engine = DiagnosisDecisionEngine(config, ref_dir=ref_dir)
    normalizer = ClinicalEvidenceNormalizer(ref_dir=ref_dir)
    rows = [_audit_row(engine, normalizer, row) for row in _load_rows(path, limit=limit)]

    def mean_bool(name: str) -> Optional[float]:
        values = [item.get(name) for item in rows if item.get(name) is not None]
        if not values:
            return None
        return round(sum(1 for item in values if item) / len(values), 4)

    return {
        "cases": len(rows),
        "decision_override_rate": mean_bool("decision_override"),
        "etiology_preference": mean_bool("etiology_preference"),
        "judge_primary_accuracy": mean_bool("judge_primary_accuracy"),
        "judge_gap_authorization_rate": mean_bool("gap_authorized"),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="training_results.jsonl or diagnostic replay JSONL")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run(
        (ROOT / args.path).resolve(),
        (ROOT / args.config).resolve(),
        limit=args.limit or None,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    for row in report["rows"]:
        print(
            f"{row['patient_id']}: retriever={row['retriever_top1']} "
            f"judge={row['judge_primary']} submitted={row['submitted']} "
            f"override={int(bool(row['decision_override']))} "
            f"gap={int(bool(row['gap_authorized']))}"
        )
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
