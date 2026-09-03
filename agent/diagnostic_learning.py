"""Shadow diagnostic-rule learning backed by deterministic replay evidence."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .candidate_policy_store import promotion_decision


class DiagnosticLearningStore:
    """Store evaluation-derived rule candidates without changing active knowledge."""

    def __init__(self, path: str = "outputs/runtime_state/pending_diagnostic_rules.json"):
        self.path = path

    def record_feedback(
        self,
        patient_id: str,
        report: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None,
        diagnosis_decision: Optional[Dict[str, Any]] = None,
        error_types: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        detail = report.get("diagnosisDetail") or report.get("diagnosis_detail") or {}
        if not isinstance(detail, dict):
            detail = {}
        expected = _text_list(detail.get("expected") or report.get("finalDiagnosis"))
        matched = _text_list(detail.get("matched"))
        submitted = _text_list(detail.get("submitted") or report.get("diagnosis"))
        missing = [name for name in expected if name not in matched]
        if not missing:
            return {"pending": 0, "updated": 0}

        data = self._load()
        candidates = data.setdefault("candidates", [])
        added = 0
        updated = 0
        now = datetime.now(timezone.utc).isoformat()
        evidence_payload = evidence or {}
        decision_payload = diagnosis_decision or {}
        errors = list(dict.fromkeys(error_types or []))

        for diagnosis in missing:
            candidate = next(
                (
                    item for item in candidates
                    if isinstance(item, dict)
                    and item.get("diagnosis") == diagnosis
                    and item.get("status", "shadow") in {"pending", "shadow"}
                ),
                None,
            )
            case_record = {
                "patient_id": patient_id,
                "submitted": submitted,
                "expected": expected,
                "evidence": evidence_payload,
                "diagnosis_decision": decision_payload,
                "error_types": errors,
                "created_at": now,
            }
            if candidate is None:
                candidates.append(
                    {
                        "id": "shadow_diag_" + uuid.uuid4().hex[:10],
                        "status": "shadow",
                        "diagnosis": diagnosis,
                        "support_cases": [case_record],
                        "replay": {
                            "independent_cases": 0,
                            "success_ratio": 0.0,
                            "avg_diagnosis_gain": 0.0,
                        },
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                added += 1
                continue

            support_cases = candidate.setdefault("support_cases", [])
            if not any(
                isinstance(item, dict) and item.get("patient_id") == patient_id
                for item in support_cases
            ):
                support_cases.append(case_record)
                candidate["updated_at"] = now
                updated += 1

        self._save(data)
        return {"pending": added, "updated": updated}

    def record_replay(
        self,
        candidate_id: str,
        gains_by_case: Dict[str, float],
        promotion_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = self._load()
        candidate = next(
            (
                item for item in data.get("candidates", [])
                if isinstance(item, dict) and item.get("id") == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise ValueError(f"diagnostic rule candidate {candidate_id!r} does not exist")

        unique = {
            str(case_id): float(gain)
            for case_id, gain in (gains_by_case or {}).items()
            if str(case_id).strip()
        }
        count = len(unique)
        successes = sum(1 for gain in unique.values() if gain > 0)
        ratio = successes / count if count else 0.0
        average = sum(unique.values()) / count if count else 0.0
        decision = promotion_decision(promotion_metrics or {})
        replay = {
            "independent_cases": count,
            "success_ratio": round(ratio, 4),
            "avg_diagnosis_gain": round(average, 4),
            "case_gains": unique,
            "promotion_metrics": dict(promotion_metrics or {}),
            "promotion_decision": decision.to_dict(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        candidate["replay"] = replay
        if decision.promote_allowed:
            candidate["status"] = "active"
        elif candidate.get("status") == "active":
            candidate["status"] = "shadow"
        self._save(data)
        return {**replay, "status": candidate.get("status", "shadow")}

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                data.setdefault("candidates", [])
                return data
        except (OSError, TypeError, ValueError):
            pass
        return {"schema_version": 2, "candidates": []}

    def _save(self, data: Dict[str, Any]) -> None:
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        temp_path = f"{self.path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temp_path, self.path)
        except OSError:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _text_list(value: Any) -> List[str]:
    if isinstance(value, str):
        value = [value]
    return list(dict.fromkeys(str(item).strip() for item in (value or []) if str(item).strip()))
