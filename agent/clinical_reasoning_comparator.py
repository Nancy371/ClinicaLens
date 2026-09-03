"""Clinical explanation comparator for primary arbitration.

The comparator is intentionally narrow: it compares two already-admitted
candidates on verified evidence, diagnostic/bridge patterns, anchor validity,
and actionable gaps. It never mutates candidates and never authorizes
submission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set

from .clinical_pattern_bridge import CROSS_SYSTEM_SCOPE, has_active_bridge_protection
from .diagnosis_eligibility import DEFERRED, EXCLUDED, PRIMARY_ELIGIBLE


ANCHOR_SATISFIED = "AnchorSatisfied"
PATTERN_SUPPORTED_BUT_UNCONFIRMED = "PatternSupportedButUnconfirmed"
NO_VALID_ANCHOR = "NoValidAnchor"
HARD_BLOCKED = "HardBlocked"

KEEP_CURRENT_PRIMARY = "KEEP_CURRENT_PRIMARY"
SWITCH_PRIMARY = "SWITCH_PRIMARY"
UNLOCK_AND_DEFER = "UNLOCK_AND_DEFER"
KEEP_CURRENT_AND_DEFER_CONTENDER = "KEEP_CURRENT_AND_DEFER_CONTENDER"
REJECT_CONTENDER = "REJECT_CONTENDER"
NO_MATERIAL_DIFFERENCE = "NO_MATERIAL_DIFFERENCE"

ESTABLISHED = "ESTABLISHED"
PROVISIONAL = "PROVISIONAL"
NOT_ESTABLISHED = "NOT_ESTABLISHED"
CONTRADICTED = "CONTRADICTED"
STALE = "STALE"

INCUMBENT_VALID = "VALID"
INCUMBENT_CHALLENGED = "CHALLENGED"
INCUMBENT_INVALIDATED = "INVALIDATED"

MATERIAL_NONE = "NONE"
MATERIAL_CONTENDER_GAIN = "CONTENDER_MATERIAL_GAIN"
MATERIAL_INCUMBENT_FAILURE = "INCUMBENT_EXPLANATORY_FAILURE"
MATERIAL_BOTH = "BOTH"

_ANCHOR_RANK = {
    HARD_BLOCKED: -2,
    NO_VALID_ANCHOR: 0,
    PATTERN_SUPPORTED_BUT_UNCONFIRMED: 1,
    ANCHOR_SATISFIED: 2,
}

_BROAD_EVIDENCE = {
    "acute_course",
    "chronic_course",
    "cough",
    "dyspnea",
    "fatigue",
    "fever",
    "pain",
    "rash",
    "symptom:signal",
    "weakness",
}

_BACKGROUND_EVIDENCE = {
    "diabetes",
    "diabetes_mellitus",
    "hypertension",
    "hyperlipidemia",
    "smoking_history",
}

_CRITICAL_EVIDENCE_TOKENS = (
    "target_claim",
    "radiation_field",
    "within_prior_radiation_field",
    "lesion_within",
    "filling_defect",
    "regurgitant_jet",
    "shunt",
)

_HIGH_EVIDENCE_TOKENS = (
    "ground_glass",
    "consolidation",
    "infiltrate",
    "opacity",
    "hypoxemia",
    "thoracic_radiotherapy",
    "radiotherapy",
    "pulmonary",
    "ct_",
    "imaging",
)

_MODERATE_EVIDENCE_TOKENS = (
    "dyspnea",
    "cough",
    "wheeze",
    "orthopnea",
    "murmur",
    "edema",
    "pain",
)

_SYSTEM_HINTS = {
    "respiratory": (
        "pulmonary",
        "lung",
        "dyspnea",
        "cough",
        "wheeze",
        "ground_glass",
        "consolidation",
        "atelectasis",
        "infiltrate",
        "opacity",
        "pleural",
        "hypox",
        "radiation_field",
    ),
    "treatment_exposure": (
        "radiotherapy",
        "chemotherapy",
        "immunotherapy",
        "drug_exposure",
        "medication",
        "exposure",
    ),
    "musculoskeletal": (
        "arthralgia",
        "arthritis",
        "joint",
        "bone",
        "fracture",
        "osteophyte",
        "trauma",
        "musculoskeletal",
    ),
    "cardiovascular": (
        "heart",
        "cardiac",
        "mitral",
        "aortic",
        "valve",
        "murmur",
        "left_atrium",
        "pulmonary_edema",
    ),
    "renal": ("renal", "kidney", "creatinine", "proteinuria", "hematuria"),
    "genitourinary": ("dysuria", "urethral", "urinary", "genitourinary"),
}


@dataclass
class ClinicalExplanatoryProfile:
    core_case_coverage: float = 0.0
    chief_complaint_alignment: str = "UNKNOWN"
    material_objective_alignment: str = "UNKNOWN"
    high_value_residuals: List[Dict[str, Any]] = field(default_factory=list)
    high_value_residual_burden: float = 0.0
    high_value_residual_burden_band: str = "LOW"
    new_material_evidence_alignment: str = "NOT_APPLICABLE"
    encounter_systems: List[str] = field(default_factory=list)
    candidate_systems: List[str] = field(default_factory=list)
    primary_explanatory_mismatch: bool = False
    primary_protection_status: str = "PROTECTED"
    profile_reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_case_coverage": round(float(self.core_case_coverage or 0.0), 4),
            "chief_complaint_alignment": self.chief_complaint_alignment,
            "material_objective_alignment": self.material_objective_alignment,
            "high_value_residuals": [dict(item) for item in self.high_value_residuals],
            "high_value_residual_burden": round(
                float(self.high_value_residual_burden or 0.0),
                4,
            ),
            "high_value_residual_burden_band": self.high_value_residual_burden_band,
            "new_material_evidence_alignment": self.new_material_evidence_alignment,
            "encounter_systems": list(self.encounter_systems),
            "candidate_systems": list(self.candidate_systems),
            "primary_explanatory_mismatch": bool(self.primary_explanatory_mismatch),
            "primary_protection_status": self.primary_protection_status,
            "profile_reason_codes": list(dict.fromkeys(self.profile_reason_codes)),
        }


@dataclass
class CandidateClinicalAnalysis:
    diagnosis: str
    entity_id: str = ""
    anchor_status: str = NO_VALID_ANCHOR
    eligibility_status: str = ""
    explained_core_evidence: List[str] = field(default_factory=list)
    explained_high_value_evidence: List[str] = field(default_factory=list)
    unexplained_high_value_evidence: List[str] = field(default_factory=list)
    contradicted_evidence: List[str] = field(default_factory=list)
    matched_diagnostic_patterns: List[str] = field(default_factory=list)
    matched_bridge_patterns: List[str] = field(default_factory=list)
    actionable_gaps: List[str] = field(default_factory=list)
    judge_score: float = 0.0
    candidate_score: float = 0.0
    explanatory_profile: ClinicalExplanatoryProfile = field(
        default_factory=ClinicalExplanatoryProfile
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "entity_id": self.entity_id,
            "anchor_status": self.anchor_status,
            "eligibility_status": self.eligibility_status,
            "explained_core_evidence": list(self.explained_core_evidence),
            "explained_high_value_evidence": list(self.explained_high_value_evidence),
            "unexplained_high_value_evidence": list(self.unexplained_high_value_evidence),
            "contradicted_evidence": list(self.contradicted_evidence),
            "matched_diagnostic_patterns": list(self.matched_diagnostic_patterns),
            "matched_bridge_patterns": list(self.matched_bridge_patterns),
            "actionable_gaps": list(self.actionable_gaps),
            "judge_score": round(float(self.judge_score or 0.0), 4),
            "candidate_score": round(float(self.candidate_score or 0.0), 4),
            "explanatory_profile": self.explanatory_profile.to_dict(),
        }


class ClinicalReasoningComparator:
    """Compare clinical explanatory adequacy before locking a primary."""

    def compare(
        self,
        current_primary: Any,
        contender: Any,
        *,
        judge_score_current: float = 0.0,
        judge_score_contender: float = 0.0,
        high_value_evidence: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        primary_analysis = self.analyze(
            current_primary,
            peer=contender,
            high_value_evidence=high_value_evidence,
            judge_score=judge_score_current,
        )
        contender_analysis = self.analyze(
            contender,
            peer=current_primary,
            high_value_evidence=high_value_evidence,
            judge_score=judge_score_contender,
        )
        reason_codes: List[str] = []

        if primary_analysis.anchor_status == HARD_BLOCKED:
            reason_codes.append("CURRENT_PRIMARY_HARD_BLOCKED")
        if contender_analysis.anchor_status == HARD_BLOCKED:
            return self._record(
                current_primary,
                contender,
                primary_analysis,
                contender_analysis,
                preferred=current_primary,
                action=REJECT_CONTENDER,
                reason_codes=["CONTENDER_HARD_BLOCKED"],
            )

        primary_explained = len(primary_analysis.explained_high_value_evidence)
        contender_explained = len(contender_analysis.explained_high_value_evidence)
        primary_residual = len(primary_analysis.unexplained_high_value_evidence)
        contender_residual = len(contender_analysis.unexplained_high_value_evidence)
        comparison = self._explanatory_comparison(
            primary_analysis,
            contender_analysis,
        )
        establishment = self.evaluate_contender_establishment(
            contender_analysis,
            contender,
        )
        incumbent_validity = self.evaluate_incumbent_validity(primary_analysis)
        material_difference = self.evaluate_material_difference(
            comparison,
            incumbent_validity,
        )
        comparison.update(
            {
                "contender_establishment_status": establishment["status"],
                "contender_clinically_established": bool(
                    establishment["status"] == ESTABLISHED
                ),
                "contender_establishment_reasons": list(establishment["reasons"]),
                "incumbent_validity_status": incumbent_validity["status"],
                "incumbent_protection_status": primary_analysis.explanatory_profile.primary_protection_status,
                "incumbent_validity_reasons": list(incumbent_validity["reasons"]),
                "material_difference_status": material_difference["status"],
                "material_difference_reasons": list(material_difference["reasons"]),
                "establishment_gate_status": establishment["status"],
                "eligibility_gate_status": contender_analysis.eligibility_status,
            }
        )
        clinically_established = establishment["status"] == ESTABLISHED
        provisional = establishment["status"] == PROVISIONAL
        contender_primary_eligible = contender_analysis.eligibility_status == PRIMARY_ELIGIBLE
        material_gain = comparison.get("contender_explanatory_gain") == "HIGH"
        incumbent_invalid = incumbent_validity["status"] == INCUMBENT_INVALIDATED
        incumbent_challenged = incumbent_validity["status"] == INCUMBENT_CHALLENGED
        valid_incumbent_switch_allowed = self._valid_incumbent_switch_allowed(
            comparison
        )

        base_reasons = list(reason_codes)
        base_reasons.extend(comparison.get("incumbent_challenge_reasons") or [])
        base_reasons.extend(incumbent_validity["reasons"])
        base_reasons.extend(material_difference["reasons"])
        base_reasons.extend(establishment["reasons"])

        if primary_analysis.anchor_status == NO_VALID_ANCHOR:
            base_reasons.append("CURRENT_PRIMARY_HAS_NO_VALID_ANCHOR")
            if contender_explained > primary_explained or contender_residual < primary_residual:
                base_reasons.append("CONTENDER_EXPLAINS_HIGH_VALUE_EVIDENCE_BETTER")
                if clinically_established and contender_primary_eligible:
                    return self._record(
                        current_primary,
                        contender,
                        primary_analysis,
                        contender_analysis,
                        preferred=contender,
                        action=SWITCH_PRIMARY,
                        reason_codes=base_reasons + ["SWITCH_PRIMARY_AUTHORIZED"],
                        explanatory_comparison=comparison,
                    )
                return self._record(
                    current_primary,
                    contender,
                    primary_analysis,
                    contender_analysis,
                    preferred=contender,
                    action=UNLOCK_AND_DEFER,
                    reason_codes=base_reasons
                    + ["CONTENDER_REQUIRES_CONFIRMATORY_GAP"],
                    explanatory_comparison=comparison,
                )

        if primary_analysis.anchor_status == ANCHOR_SATISFIED:
            base_reasons.append("INCUMBENT_ANCHOR_SATISFIED")

        if incumbent_invalid:
            if material_gain and clinically_established and contender_primary_eligible:
                return self._record(
                    current_primary,
                    contender,
                    primary_analysis,
                    contender_analysis,
                    preferred=contender,
                    action=SWITCH_PRIMARY,
                    reason_codes=base_reasons
                    + [
                        "CONTENDER_CLINICALLY_ESTABLISHED",
                        "CONTENDER_EXPLANATORY_GAIN_HIGH",
                        "SWITCH_PRIMARY_AUTHORIZED",
                    ],
                    explanatory_comparison=comparison,
                )
            if material_gain and provisional:
                return self._record(
                    current_primary,
                    contender,
                    primary_analysis,
                    contender_analysis,
                    preferred=contender,
                    action=UNLOCK_AND_DEFER,
                    reason_codes=base_reasons
                    + [
                        "MATERIAL_GAIN_BUT_CONTENDER_UNCONFIRMED",
                        "PRIMARY_UNLOCKED_CONTENDER_DEFERRED",
                    ],
                    explanatory_comparison=comparison,
                )
            gate_reason = (
                "CONTENDER_CLINICAL_ESTABLISHMENT_GATE_NOT_MET"
                if material_gain
                else "INCUMBENT_FAILED_BUT_NO_REPLACEMENT"
            )
            return self._record(
                current_primary,
                contender,
                primary_analysis,
                contender_analysis,
                preferred=contender,
                action=UNLOCK_AND_DEFER,
                reason_codes=base_reasons
                + [
                    gate_reason,
                    "INCUMBENT_FAILED_BUT_NO_REPLACEMENT",
                ],
                explanatory_comparison=comparison,
            )

        if material_gain:
            if (
                clinically_established
                and contender_primary_eligible
                and valid_incumbent_switch_allowed
            ):
                return self._record(
                    current_primary,
                    contender,
                    primary_analysis,
                    contender_analysis,
                    preferred=contender,
                    action=SWITCH_PRIMARY,
                    reason_codes=base_reasons
                    + [
                        "CONTENDER_CLINICALLY_ESTABLISHED",
                        "CONTENDER_EXPLANATORY_GAIN_HIGH",
                        "SWITCH_PRIMARY_AUTHORIZED",
                    ],
                    explanatory_comparison=comparison,
                )
            if clinically_established and contender_primary_eligible:
                return self._record(
                    current_primary,
                    contender,
                    primary_analysis,
                    contender_analysis,
                    preferred=current_primary,
                    action=KEEP_CURRENT_AND_DEFER_CONTENDER,
                    reason_codes=base_reasons
                    + [
                        "CONTENDER_CLINICALLY_ESTABLISHED",
                        "CONTENDER_EXPLANATORY_GAIN_HIGH",
                        "VALID_INCUMBENT_SWITCH_GATE_NOT_MET",
                    ],
                    explanatory_comparison=comparison,
                )
            if provisional:
                action = (
                    UNLOCK_AND_DEFER
                    if incumbent_challenged
                    else KEEP_CURRENT_AND_DEFER_CONTENDER
                )
                return self._record(
                    current_primary,
                    contender,
                    primary_analysis,
                    contender_analysis,
                    preferred=contender,
                    action=action,
                    reason_codes=base_reasons
                    + [
                        "MATERIAL_GAIN_BUT_CONTENDER_UNCONFIRMED",
                        "CONTENDER_REQUIRES_CONFIRMATORY_GAP",
                    ],
                    explanatory_comparison=comparison,
                )
            return self._record(
                current_primary,
                contender,
                primary_analysis,
                contender_analysis,
                preferred=current_primary,
                action=KEEP_CURRENT_AND_DEFER_CONTENDER,
                reason_codes=base_reasons
                + ["CONTENDER_CLINICAL_ESTABLISHMENT_GATE_NOT_MET"],
                explanatory_comparison=comparison,
            )

        if provisional and primary_analysis.anchor_status == ANCHOR_SATISFIED:
            return self._record(
                current_primary,
                contender,
                primary_analysis,
                contender_analysis,
                preferred=current_primary,
                action=KEEP_CURRENT_AND_DEFER_CONTENDER,
                reason_codes=base_reasons + ["CURRENT_PRIMARY_HAS_VALID_ANCHOR"],
                explanatory_comparison=comparison,
            )

        return self._record(
            current_primary,
            contender,
            primary_analysis,
            contender_analysis,
            preferred=current_primary,
            action=NO_MATERIAL_DIFFERENCE,
            reason_codes=base_reasons + ["NO_MATERIAL_DIFFERENCE"],
            explanatory_comparison=comparison,
        )

    def analyze(
        self,
        candidate: Any,
        *,
        peer: Any = None,
        high_value_evidence: Sequence[str] | None = None,
        judge_score: float = 0.0,
    ) -> CandidateClinicalAnalysis:
        if not candidate:
            return CandidateClinicalAnalysis(diagnosis="")
        matched_diagnostic = self._matched_diagnostic_patterns(candidate)
        matched_bridge = self._matched_bridge_patterns(candidate)
        high_value = set(high_value_evidence or self._high_value_universe(candidate, peer))
        explained = self._explained_findings(candidate)
        explained_high = sorted(explained & high_value)
        residual = self._residual_findings(candidate)
        residual_high = sorted(residual & high_value)
        profile = self._explanatory_profile(
            candidate,
            explained=explained,
            residual=residual,
            high_value=high_value,
        )
        if matched_bridge:
            explained_high.extend(item for item in matched_bridge if item not in explained_high)
        if matched_diagnostic:
            explained_high.extend(item for item in matched_diagnostic if item not in explained_high)
        return CandidateClinicalAnalysis(
            diagnosis=str(getattr(candidate, "diagnosis", "") or ""),
            entity_id=str(getattr(candidate, "entity_id", "") or ""),
            anchor_status=self.anchor_status(candidate),
            eligibility_status=str(getattr(candidate, "eligibility_status", "") or ""),
            explained_core_evidence=sorted(explained - _BROAD_EVIDENCE)[:12],
            explained_high_value_evidence=list(dict.fromkeys(explained_high))[:12],
            unexplained_high_value_evidence=residual_high[:12],
            contradicted_evidence=self._contradicted_findings(candidate)[:12],
            matched_diagnostic_patterns=matched_diagnostic,
            matched_bridge_patterns=matched_bridge,
            actionable_gaps=self._actionable_gaps(candidate)[:12],
            judge_score=judge_score,
            candidate_score=float(getattr(candidate, "score", 0.0) or 0.0),
            explanatory_profile=profile,
        )

    def anchor_status(self, candidate: Any) -> str:
        if not candidate:
            return NO_VALID_ANCHOR
        if bool(getattr(candidate, "hard_contradiction", False)):
            return HARD_BLOCKED
        explicit = str(getattr(candidate, "eligibility_anchor_status", "") or "")
        if explicit:
            return explicit
        status = str(getattr(candidate, "eligibility_status", "") or "")
        if status == EXCLUDED:
            return HARD_BLOCKED
        if self._matched_diagnostic_patterns(candidate) and status == PRIMARY_ELIGIBLE:
            return ANCHOR_SATISFIED
        if status == PRIMARY_ELIGIBLE and bool(getattr(candidate, "required_met", False)):
            return ANCHOR_SATISFIED
        if status == DEFERRED and (
            self._matched_bridge_patterns(candidate)
            or getattr(candidate, "required_gaps", None)
        ):
            return PATTERN_SUPPORTED_BUT_UNCONFIRMED
        if self._matched_bridge_patterns(candidate):
            return PATTERN_SUPPORTED_BUT_UNCONFIRMED
        return NO_VALID_ANCHOR

    def material_contender(self, candidate: Any, current_primary: Any = None) -> bool:
        if not candidate or bool(getattr(candidate, "hard_contradiction", False)):
            return False
        if str(getattr(candidate, "eligibility_status", "") or "") == EXCLUDED:
            return False
        if self._primary_eligible_contender(candidate):
            return True
        if has_active_bridge_protection(candidate, CROSS_SYSTEM_SCOPE):
            return True
        if self._protected_pattern_recall(candidate):
            return True
        if self._matched_bridge_patterns(candidate):
            return True
        if self._matched_diagnostic_patterns(candidate):
            return True
        if self.anchor_status(current_primary) != NO_VALID_ANCHOR:
            return False
        high_value = self._high_value_universe(candidate, current_primary)
        return len(self._explained_findings(candidate) & high_value) >= 2

    def evaluate_contender_establishment(
        self,
        analysis: CandidateClinicalAnalysis,
        candidate: Any = None,
    ) -> Dict[str, Any]:
        reasons: List[str] = []
        if analysis.anchor_status == HARD_BLOCKED or bool(
            getattr(candidate, "hard_contradiction", False)
        ):
            return {
                "status": CONTRADICTED,
                "reasons": [
                    "CONTENDER_ESTABLISHMENT_CONTRADICTED",
                    "ANCHOR_CONTRADICTION_STATE_INCONSISTENCY",
                ],
            }
        if self._establishment_state_stale(candidate):
            return {
                "status": STALE,
                "reasons": ["CONTENDER_ESTABLISHMENT_STALE"],
            }
        if analysis.anchor_status == ANCHOR_SATISFIED:
            reasons.append("CLINICALLY_ESTABLISHED_BY_CLAIM_ANCHOR")
            return {
                "status": ESTABLISHED,
                "reasons": reasons + ["CLAIM_ANCHOR_ESTABLISHED"],
            }
        if analysis.matched_diagnostic_patterns:
            reasons.append("PROVISIONAL_BY_DIAGNOSTIC_PATTERN")
        if analysis.matched_bridge_patterns:
            reasons.append("PROVISIONAL_BY_BRIDGE_PATTERN")
        if analysis.anchor_status == PATTERN_SUPPORTED_BUT_UNCONFIRMED:
            reasons.append("PROVISIONAL_BY_PATTERN_SUPPORTED_ANCHOR")
        if reasons:
            return {"status": PROVISIONAL, "reasons": reasons}
        if analysis.eligibility_status == PRIMARY_ELIGIBLE:
            reasons.append("ELIGIBILITY_WITHOUT_CLINICAL_ESTABLISHMENT")
        return {
            "status": NOT_ESTABLISHED,
            "reasons": reasons + ["CONTENDER_NOT_CLINICALLY_ESTABLISHED"],
        }

    def evaluate_incumbent_validity(
        self,
        analysis: CandidateClinicalAnalysis,
    ) -> Dict[str, Any]:
        profile = analysis.explanatory_profile
        reasons: List[str] = []
        if analysis.anchor_status in {HARD_BLOCKED, NO_VALID_ANCHOR}:
            reasons.append("CURRENT_PRIMARY_HAS_NO_VALID_ANCHOR")
            return {"status": INCUMBENT_INVALIDATED, "reasons": reasons}
        if str(profile.primary_protection_status or "").upper() == "LOST":
            reasons.append("INCUMBENT_PRIMARY_PROTECTION_LOST")
        if bool(profile.primary_explanatory_mismatch):
            reasons.append("INCUMBENT_PRIMARY_EXPLANATORY_FAILURE")
        if float(profile.core_case_coverage or 0.0) <= 0.05:
            reasons.append("INCUMBENT_CORE_CASE_COVERAGE_FAILURE")
        if str(profile.high_value_residual_burden_band or "") == "VERY_HIGH":
            reasons.append("INCUMBENT_HIGH_VALUE_RESIDUAL_FAILURE")
        if str(profile.chief_complaint_alignment or "") == "POOR":
            reasons.append("INCUMBENT_CHIEF_COMPLAINT_MISMATCH")
        if str(profile.material_objective_alignment or "") == "POOR":
            reasons.append("INCUMBENT_MATERIAL_OBJECTIVE_MISMATCH")
        if "INCUMBENT_PRIMARY_PROTECTION_LOST" in reasons:
            return {
                "status": INCUMBENT_INVALIDATED,
                "reasons": list(dict.fromkeys(reasons)),
            }
        if str(profile.primary_protection_status or "").upper() == "CHALLENGED":
            reasons.append("INCUMBENT_PRIMARY_PROTECTION_CHALLENGED")
            return {
                "status": INCUMBENT_CHALLENGED,
                "reasons": list(dict.fromkeys(reasons)),
            }
        return {"status": INCUMBENT_VALID, "reasons": []}

    def evaluate_material_difference(
        self,
        comparison: Dict[str, Any],
        incumbent_validity: Dict[str, Any],
    ) -> Dict[str, Any]:
        reasons: List[str] = []
        contender_gain = comparison.get("contender_explanatory_gain") == "HIGH"
        incumbent_failed = incumbent_validity.get("status") == INCUMBENT_INVALIDATED
        if contender_gain:
            reasons.extend(
                [
                    "MATERIAL_DIFFERENCE_EXISTS",
                    "CONTENDER_EXPLANATORY_GAIN_HIGH",
                ]
            )
            if float(comparison.get("residual_burden_delta") or 0.0) > 0:
                reasons.append("CONTENDER_REDUCES_HIGH_VALUE_RESIDUAL")
        if incumbent_failed:
            reasons.extend(
                [
                    "MATERIAL_DIFFERENCE_EXISTS",
                    "INCUMBENT_PRIMARY_EXPLANATORY_FAILURE",
                ]
            )
        if contender_gain and incumbent_failed:
            status = MATERIAL_BOTH
        elif contender_gain:
            status = MATERIAL_CONTENDER_GAIN
        elif incumbent_failed:
            status = MATERIAL_INCUMBENT_FAILURE
        else:
            status = MATERIAL_NONE
        return {"status": status, "reasons": list(dict.fromkeys(reasons))}

    @staticmethod
    def _valid_incumbent_switch_allowed(comparison: Dict[str, Any]) -> bool:
        incumbent_profile = dict(comparison.get("incumbent_profile") or {})
        residual_delta = float(comparison.get("residual_burden_delta") or 0.0)
        incumbent_burden_band = str(
            incumbent_profile.get("high_value_residual_burden_band") or ""
        ).upper()
        return bool(residual_delta >= 1.2 or incumbent_burden_band in {"HIGH", "VERY_HIGH"})

    @staticmethod
    def _establishment_state_stale(candidate: Any) -> bool:
        if not candidate:
            return False
        current = int(getattr(candidate, "diagnostic_state_version", 0) or 0)
        if current <= 0:
            return False
        for attr in (
            "anchor_state_version",
            "eligibility_state_version",
            "consumed_diagnostic_state_version",
        ):
            value = int(getattr(candidate, attr, 0) or 0)
            if value > 0 and value != current:
                return True
        return False

    def _primary_eligible_contender(self, candidate: Any) -> bool:
        anchor = self.anchor_status(candidate)
        status = str(getattr(candidate, "eligibility_status", "") or "")
        if anchor != ANCHOR_SATISFIED and status != PRIMARY_ELIGIBLE:
            return False
        return bool(
            getattr(candidate, "required_met", False)
            or getattr(candidate, "core_matched_evidence", None)
            or getattr(candidate, "diagnostic_matched_evidence", None)
            or self._matched_diagnostic_patterns(candidate)
            or self._component_score(candidate, "core_evidence_score") >= 0.20
            or self._component_score(candidate, "diagnostic_evidence_score") > 0.0
        )

    def pair_high_value_evidence(self, left: Any, right: Any) -> List[str]:
        return sorted(self._high_value_universe(left, right))

    @staticmethod
    def _record(
        current_primary: Any,
        contender: Any,
        primary_analysis: CandidateClinicalAnalysis,
        contender_analysis: CandidateClinicalAnalysis,
        *,
        preferred: Any,
        action: str,
        reason_codes: Sequence[str],
        explanatory_comparison: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return {
            "comparison_id": (
                "CRC-"
                + str(getattr(current_primary, "entity_id", "") or getattr(current_primary, "diagnosis", "primary"))
                + "-"
                + str(getattr(contender, "entity_id", "") or getattr(contender, "diagnosis", "contender"))
            ),
            "candidate_a": str(getattr(current_primary, "diagnosis", "") or ""),
            "candidate_b": str(getattr(contender, "diagnosis", "") or ""),
            "candidate_a_analysis": primary_analysis.to_dict(),
            "candidate_b_analysis": contender_analysis.to_dict(),
            "preferred_candidate": str(getattr(preferred, "diagnosis", "") or ""),
            "recommended_action": action,
            "decision_reason_codes": list(dict.fromkeys(reason_codes)),
            "clinical_explanatory_comparison": dict(explanatory_comparison or {}),
            "contender_establishment_status": str(
                (explanatory_comparison or {}).get("contender_establishment_status") or ""
            ),
            "contender_clinically_established": bool(
                (explanatory_comparison or {}).get("contender_clinically_established")
            ),
            "contender_establishment_reasons": list(
                (explanatory_comparison or {}).get("contender_establishment_reasons")
                or []
            ),
            "incumbent_validity_status": str(
                (explanatory_comparison or {}).get("incumbent_validity_status") or ""
            ),
            "incumbent_protection_status": str(
                (explanatory_comparison or {}).get("incumbent_protection_status") or ""
            ),
            "incumbent_challenge_reasons": list(
                (explanatory_comparison or {}).get("incumbent_challenge_reasons")
                or []
            ),
            "material_difference_status": str(
                (explanatory_comparison or {}).get("material_difference_status") or ""
            ),
            "material_difference_reasons": list(
                (explanatory_comparison or {}).get("material_difference_reasons")
                or []
            ),
            "establishment_gate_status": str(
                (explanatory_comparison or {}).get("establishment_gate_status") or ""
            ),
            "eligibility_gate_status": str(
                (explanatory_comparison or {}).get("eligibility_gate_status") or ""
            ),
        }

    def _explanatory_comparison(
        self,
        incumbent: CandidateClinicalAnalysis,
        contender: CandidateClinicalAnalysis,
    ) -> Dict[str, Any]:
        incumbent_profile = incumbent.explanatory_profile
        contender_profile = contender.explanatory_profile
        coverage_delta = (
            contender_profile.core_case_coverage
            - incumbent_profile.core_case_coverage
        )
        residual_delta = (
            incumbent_profile.high_value_residual_burden
            - contender_profile.high_value_residual_burden
        )
        material_delta = self._alignment_value(
            contender_profile.material_objective_alignment
        ) - self._alignment_value(incumbent_profile.material_objective_alignment)
        challenge_reasons = [
            code
            for code in incumbent_profile.profile_reason_codes
            if code
            in {
                "INCUMBENT_CORE_COVERAGE_LOW",
                "INCUMBENT_HIGH_VALUE_RESIDUAL_HIGH",
                "CHIEF_COMPLAINT_SYSTEM_MISMATCH",
                "MATERIAL_EVIDENCE_SYSTEM_MISMATCH",
                "NEW_MATERIAL_EVIDENCE_UNEXPLAINED",
                "PRIMARY_EXPLANATORY_MISMATCH",
            }
        ]
        gain_reasons: List[str] = []
        if coverage_delta >= 0.30:
            gain_reasons.append("CONTENDER_CORE_COVERAGE_HIGHER")
        if residual_delta >= 0.80:
            gain_reasons.append("CONTENDER_REDUCES_HIGH_VALUE_RESIDUAL")
        if material_delta >= 0.35:
            gain_reasons.append("CONTENDER_MATERIAL_ALIGNMENT_HIGHER")
        if contender_profile.core_case_coverage >= 0.65:
            gain_reasons.append("CONTENDER_CORE_COVERAGE_HIGH")
        gain = "HIGH" if challenge_reasons and len(gain_reasons) >= 2 else "LOW"
        if gain == "LOW" and challenge_reasons and coverage_delta >= 0.45:
            gain = "HIGH"
        return {
            "incumbent_profile": incumbent_profile.to_dict(),
            "contender_profile": contender_profile.to_dict(),
            "core_case_coverage_delta": round(float(coverage_delta), 4),
            "residual_burden_delta": round(float(residual_delta), 4),
            "material_alignment_delta": round(float(material_delta), 4),
            "contender_explanatory_gain": gain,
            "incumbent_challenge_reasons": list(dict.fromkeys(challenge_reasons)),
            "pairwise_findings": list(dict.fromkeys(gain_reasons)),
            "contender_readiness": contender.anchor_status,
        }

    def _explanatory_profile(
        self,
        candidate: Any,
        *,
        explained: Set[str],
        residual: Set[str],
        high_value: Set[str],
    ) -> ClinicalExplanatoryProfile:
        core_terms = {
            item
            for item in high_value
            if self._is_core_case_evidence(item)
            and not self._is_pattern_token(item)
            and item not in _BACKGROUND_EVIDENCE
        }
        if not core_terms:
            core_terms = {
                item
                for item in explained | residual
                if self._is_core_case_evidence(item)
                and not self._is_pattern_token(item)
                and item not in _BACKGROUND_EVIDENCE
            }
        weighted_total = sum(self._evidence_weight(item) for item in core_terms)
        weighted_explained = sum(
            self._evidence_weight(item) for item in core_terms if item in explained
        )
        coverage = weighted_explained / weighted_total if weighted_total > 0 else 0.0
        residual_terms = [item for item in core_terms if item in residual]
        residual_entries = [
            {
                "evidence": item,
                "value_band": self._evidence_value_band(item),
                "weight": self._evidence_weight(item),
                "systems": sorted(self._systems_for_text(item)),
            }
            for item in sorted(residual_terms)
        ]
        burden = sum(float(item["weight"]) for item in residual_entries)
        burden_band = self._burden_band(burden)
        system_weights: Dict[str, float] = {}
        for item in core_terms:
            for system in self._systems_for_text(item):
                system_weights[system] = system_weights.get(system, 0.0) + self._evidence_weight(item)
        max_system_weight = max(system_weights.values(), default=0.0)
        dominant_cutoff = max(0.8, max_system_weight * 0.50)
        encounter_systems = sorted(
            system
            for system, weight in system_weights.items()
            if weight >= dominant_cutoff
        )
        candidate_systems = sorted(self._candidate_systems(candidate, explained))
        system_intersection = set(encounter_systems) & set(candidate_systems)
        if not encounter_systems or not candidate_systems:
            chief_alignment = "UNKNOWN"
        elif system_intersection or coverage >= 0.65:
            chief_alignment = "STRONG"
        elif coverage >= 0.35:
            chief_alignment = "PARTIAL"
        else:
            chief_alignment = "POOR"
        objective_terms = [
            item
            for item in core_terms
            if self._evidence_value_band(item) in {"HIGH", "CRITICAL"}
        ]
        objective_explained = [item for item in objective_terms if item in explained]
        if not objective_terms:
            material_alignment = "UNKNOWN"
        elif len(objective_explained) == len(objective_terms):
            material_alignment = "STRONG"
        elif objective_explained:
            material_alignment = "PARTIAL"
        else:
            material_alignment = "POOR"
        material_terms = [
            item
            for item in core_terms
            if self._is_new_material_evidence(item)
        ]
        material_explained = [item for item in material_terms if item in explained]
        if not material_terms:
            new_material_alignment = "NOT_APPLICABLE"
        elif len(material_explained) == len(material_terms):
            new_material_alignment = "STRONG"
        elif material_explained:
            new_material_alignment = "PARTIAL"
        else:
            new_material_alignment = "POOR"
        reason_codes: List[str] = []
        if coverage < 0.35 and core_terms:
            reason_codes.append("INCUMBENT_CORE_COVERAGE_LOW")
        if burden_band in {"HIGH", "VERY_HIGH"}:
            reason_codes.append("INCUMBENT_HIGH_VALUE_RESIDUAL_HIGH")
        if chief_alignment == "POOR":
            reason_codes.append("CHIEF_COMPLAINT_SYSTEM_MISMATCH")
        if material_alignment == "POOR":
            reason_codes.append("MATERIAL_EVIDENCE_SYSTEM_MISMATCH")
        if new_material_alignment == "POOR":
            reason_codes.append("NEW_MATERIAL_EVIDENCE_UNEXPLAINED")
        mismatch = bool(
            chief_alignment == "POOR"
            and coverage < 0.35
            and burden_band in {"HIGH", "VERY_HIGH"}
        )
        protection = "PROTECTED"
        if mismatch:
            protection = "LOST"
            reason_codes.append("PRIMARY_EXPLANATORY_MISMATCH")
        elif (
            coverage < 0.45
            and burden_band in {"HIGH", "VERY_HIGH"}
            and material_alignment == "POOR"
        ):
            protection = "CHALLENGED"
        return ClinicalExplanatoryProfile(
            core_case_coverage=coverage,
            chief_complaint_alignment=chief_alignment,
            material_objective_alignment=material_alignment,
            high_value_residuals=residual_entries,
            high_value_residual_burden=burden,
            high_value_residual_burden_band=burden_band,
            new_material_evidence_alignment=new_material_alignment,
            encounter_systems=encounter_systems,
            candidate_systems=candidate_systems,
            primary_explanatory_mismatch=mismatch,
            primary_protection_status=protection,
            profile_reason_codes=reason_codes,
        )

    @staticmethod
    def _alignment_value(value: str) -> float:
        return {
            "STRONG": 1.0,
            "PARTIAL": 0.5,
            "UNKNOWN": 0.25,
            "NOT_APPLICABLE": 0.25,
            "POOR": 0.0,
        }.get(str(value or ""), 0.0)

    @staticmethod
    def _primary_protection_decision_reason(
        analysis: CandidateClinicalAnalysis,
    ) -> str:
        status = str(
            analysis.explanatory_profile.primary_protection_status or ""
        ).upper()
        if status == "LOST":
            return "INCUMBENT_PRIMARY_PROTECTION_LOST"
        return "INCUMBENT_PRIMARY_PROTECTION_CHALLENGED"

    @staticmethod
    def _burden_band(value: float) -> str:
        if value >= 2.0:
            return "VERY_HIGH"
        if value >= 1.2:
            return "HIGH"
        if value >= 0.5:
            return "MEDIUM"
        return "LOW"

    def _evidence_value_band(self, value: str) -> str:
        text = str(value or "").lower()
        if any(token in text for token in _CRITICAL_EVIDENCE_TOKENS):
            return "CRITICAL"
        if any(token in text for token in _HIGH_EVIDENCE_TOKENS):
            return "HIGH"
        if any(token in text for token in _MODERATE_EVIDENCE_TOKENS):
            return "MEDIUM"
        return "LOW"

    def _evidence_weight(self, value: str) -> float:
        return {
            "CRITICAL": 1.0,
            "HIGH": 0.8,
            "MEDIUM": 0.4,
            "LOW": 0.1,
        }.get(self._evidence_value_band(value), 0.1)

    def _is_core_case_evidence(self, value: str) -> bool:
        text = str(value or "").strip().lower()
        if not text or text in _BROAD_EVIDENCE or text in _BACKGROUND_EVIDENCE:
            return False
        if text.startswith(("reasoning:", "candidate:", "diagnosis:", "field:")):
            return False
        if self._evidence_value_band(text) in {"HIGH", "CRITICAL"}:
            return True
        if any(token in text for token in _MODERATE_EVIDENCE_TOKENS):
            return True
        return False

    @staticmethod
    def _is_pattern_token(value: str) -> bool:
        text = str(value or "").lower()
        return (
            text.startswith(("ph_", "cpm-", "dpa-", "pattern:"))
            or "pattern" in text
            or "protected_recall" in text
        )

    @staticmethod
    def _is_new_material_evidence(value: str) -> bool:
        text = str(value or "").lower()
        return any(token in text for token in _CRITICAL_EVIDENCE_TOKENS) or (
            "ground_glass" in text
            or "consolidation" in text
            or "target_claim" in text
        )

    def _candidate_systems(self, candidate: Any, explained: Set[str]) -> Set[str]:
        systems: Set[str] = set()
        for key in ("body_system", "candidate_body_system"):
            value = str(getattr(candidate, key, "") or "").strip()
            if value:
                systems.add(value)
        for source in getattr(candidate, "candidate_sources", []) or []:
            if not isinstance(source, dict):
                continue
            metadata = dict(source.get("metadata") or {})
            for key in ("body_system", "family_id", "suggested_family_id"):
                systems.update(self._systems_for_text(str(metadata.get(key) or "")))
        for item in explained:
            systems.update(self._systems_for_text(item))
        return systems

    @staticmethod
    def _systems_for_text(value: str) -> Set[str]:
        text = str(value or "").lower()
        systems: Set[str] = set()
        for system, hints in _SYSTEM_HINTS.items():
            if any(token in text for token in hints):
                systems.add(system)
        return systems

    def _high_value_universe(self, left: Any, right: Any = None) -> Set[str]:
        values: Set[str] = set()
        for candidate in (left, right):
            if not candidate:
                continue
            values.update(self._texts(getattr(candidate, "core_matched_evidence", []) or []))
            values.update(self._texts(getattr(candidate, "diagnostic_matched_evidence", []) or []))
            values.update(self._texts(getattr(candidate, "unexplained_core_evidence", []) or []))
            for pattern in getattr(candidate, "clinical_pattern_matches", []) or []:
                if isinstance(pattern, dict) and str(pattern.get("verification_status") or "") == "verified":
                    values.update(self._texts(pattern.get("supporting_findings") or []))
                    pattern_id = str(pattern.get("pattern_id") or "")
                    if pattern_id:
                        values.add(pattern_id)
            for assertion in getattr(candidate, "derived_pattern_assertions", []) or []:
                if isinstance(assertion, dict):
                    pattern_id = str(assertion.get("canonical_pattern") or "")
                    if pattern_id:
                        values.add(pattern_id)
        return {item for item in values if item and item not in _BROAD_EVIDENCE}

    def _explained_findings(self, candidate: Any) -> Set[str]:
        values: Set[str] = set()
        values.update(self._texts(getattr(candidate, "core_matched_evidence", []) or []))
        values.update(self._texts(getattr(candidate, "diagnostic_matched_evidence", []) or []))
        values.update(
            item
            for item in self._texts(getattr(candidate, "matched_evidence", []) or [])
            if item not in _BROAD_EVIDENCE and not item.startswith("field:")
        )
        for pattern in getattr(candidate, "clinical_pattern_matches", []) or []:
            if isinstance(pattern, dict) and str(pattern.get("verification_status") or "") == "verified":
                values.update(self._texts(pattern.get("supporting_findings") or []))
                pattern_id = str(pattern.get("pattern_id") or "")
                if pattern_id:
                    values.add(pattern_id)
        for assertion in getattr(candidate, "derived_pattern_assertions", []) or []:
            if isinstance(assertion, dict):
                pattern_id = str(assertion.get("canonical_pattern") or "")
                if pattern_id:
                    values.add(pattern_id)
        return values

    def _residual_findings(self, candidate: Any) -> Set[str]:
        values = set(self._texts(getattr(candidate, "residual_evidence", []) or []))
        values.update(self._texts(getattr(candidate, "unexplained_core_evidence", []) or []))
        return values

    def _contradicted_findings(self, candidate: Any) -> List[str]:
        values: List[str] = []
        for key in (
            "hard_contradicted_evidence",
            "soft_contradicted_evidence",
            "contradicted_evidence",
            "eligibility_blockers",
        ):
            values.extend(self._texts(getattr(candidate, key, []) or []))
        return list(dict.fromkeys(values))

    def _matched_diagnostic_patterns(self, candidate: Any) -> List[str]:
        result: List[str] = []
        for pattern in getattr(candidate, "evidence_pattern_matches", []) or []:
            if not isinstance(pattern, dict) or not bool(pattern.get("matched", False)):
                continue
            effect = dict(pattern.get("effect") or {})
            if str(effect.get("eligibility") or pattern.get("eligibility") or "") == PRIMARY_ELIGIBLE:
                pattern_id = str(pattern.get("pattern_id") or "")
                if pattern_id:
                    result.append(pattern_id)
        return list(dict.fromkeys(result))

    def _matched_bridge_patterns(self, candidate: Any) -> List[str]:
        result: List[str] = []
        result.extend(self._protected_pattern_recall(candidate))
        for assertion in getattr(candidate, "derived_pattern_assertions", []) or []:
            if isinstance(assertion, dict):
                value = str(assertion.get("canonical_pattern") or "")
                if value:
                    result.append(value)
        for pattern in getattr(candidate, "evidence_pattern_matches", []) or []:
            if not isinstance(pattern, dict):
                continue
            pattern_type = str(pattern.get("pattern_type") or pattern.get("role") or "")
            if "bridge" in pattern_type and bool(pattern.get("matched", False)):
                value = str(pattern.get("pattern_id") or "")
                if value:
                    result.append(value)
        return list(dict.fromkeys(result))

    @staticmethod
    def _protected_pattern_recall(candidate: Any) -> List[str]:
        result: List[str] = []
        for source in getattr(candidate, "candidate_sources", []) or []:
            if not isinstance(source, dict):
                continue
            metadata = dict(source.get("metadata") or {})
            if (
                str(source.get("source") or "") == "llm_pattern_hypothesis"
                and bool(metadata.get("protected_pool_slot"))
                and str(metadata.get("recall_mode") or "") == "protected_recall"
            ):
                pattern_id = str(metadata.get("pattern_hypothesis_id") or "")
                result.append(pattern_id or "verified_pattern_recall")
        return list(dict.fromkeys(item for item in result if item))

    def _actionable_gaps(self, candidate: Any) -> List[str]:
        values = list(self._texts(getattr(candidate, "required_gaps", []) or []))
        for gap in getattr(candidate, "evidence_gaps", []) or []:
            if isinstance(gap, dict):
                gap_id = str(gap.get("gap_id") or gap.get("target_evidence") or "")
                if gap_id:
                    values.append(gap_id)
        return list(dict.fromkeys(values))

    @staticmethod
    def _texts(values: Iterable[Any]) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()
        for value in values or []:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result
