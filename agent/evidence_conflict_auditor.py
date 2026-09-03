"""Audit reasoning/structured mismatches at the evidence-claim level."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from .targeted_evidence_verifier import (
    INVALID_CLAIM,
    UNRESOLVED,
    UNSUPPORTED,
    VerificationResult,
)


class EvidenceConflictAuditor:
    """Find claim-level conflicts that should trigger targeted re-verification.

    This module does not diagnose or change candidate status. It emits audit
    events that Judge/ExamStrategy can consume as opinions.
    """

    def audit(
        self,
        hypotheses: Sequence[Any],
        verification_results: Sequence[Any],
        candidate_pool: Any = None,
    ) -> List[Dict[str, Any]]:
        results = [
            item
            for item in (VerificationResult.from_any(value) for value in verification_results or [])
            if item is not None
        ]
        conflicts: List[Dict[str, Any]] = []
        unresolved_critical = [
            item
            for item in results
            if item.verification_status in {UNSUPPORTED, UNRESOLVED, INVALID_CLAIM}
            and str(item.reason or "")
            and _critical(item)
        ]
        for item in unresolved_critical:
            conflicts.append(
                {
                    "conflict_type": "critical_reasoning_claim_unverified",
                    "candidate": item.candidate,
                    "target_evidence": item.target_evidence_id,
                    "hypothesis_id": item.hypothesis_id,
                    "verification_status": item.verification_status,
                    "reason": item.reason,
                    "action": "targeted_reverification_or_exam_followup",
                }
            )
        return conflicts


def _critical(result: VerificationResult) -> bool:
    # VerificationResult does not carry importance in every legacy path, so use
    # critical effects and known anchor-like ids as a stable fallback.
    text = " ".join(
        [
            result.target_evidence_id,
            result.claim_type,
            result.recommended_exam,
            result.importance,
            result.expected_effect,
            result.reason,
        ]
    ).lower()
    return any(
        token in text
        for token in (
            "blast",
            "leukemia",
            "pulmonary_cta",
            "right_to_left_shunt",
            "prostate_tenderness",
            "confirmation",
            "critical",
        )
    )
