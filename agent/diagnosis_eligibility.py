"""Primary diagnosis eligibility and evidence-contribution scoring."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .diagnostic_patterns import DiagnosticPatternEvaluator


PRIMARY_ELIGIBLE = "PrimaryEligible"
DEFERRED = "Deferred"
DIFFERENTIAL_ONLY = "DifferentialOnly"
EXCLUDED = "Excluded"

NEEDS_ANCHOR = "NeedsAnchor"
DEFERRED_NEEDS_OBSERVED_EVIDENCE = "DeferredNeedsObservedEvidence"
DEFERRED_NEEDS_DERIVED_PATTERN = "DeferredNeedsDerivedPattern"
DEFERRED_NEEDS_CONFIRMATORY_EXAM = "DeferredNeedsConfirmatoryExam"
DEFERRED_UNRESOLVED_NAMING = "DeferredUnresolvedNaming"
DEFERRED_LOW_PRIORITY = "DeferredLowPriority"
CONFLICT_NEEDS_ADJUDICATION = "ConflictNeedsAdjudication"
HARD_CONTRADICTION = "HardContradiction"
NO_SUPPORTING_EVIDENCE = "NoSupportingEvidence"
WEAK_DIFFERENTIAL_SIGNAL = "WeakDifferentialSignal"
INSUFFICIENT_EXPLANATION = "InsufficientExplanation"
ANCHORS_SATISFIED = "AnchorsSatisfied"
PATTERN_CONTRADICTED = "PatternContradicted"
ANCHOR_SATISFIED = "AnchorSatisfied"
PATTERN_SUPPORTED_BUT_UNCONFIRMED = "PatternSupportedButUnconfirmed"
NO_VALID_ANCHOR = "NoValidAnchor"
HARD_BLOCKED = "HardBlocked"


_PULMONARY_CRYPTOCOCCOSIS_ANCHORS = {
    "diagnosis:肺隐球菌病",
    "cryptococcal_antigen_positive",
    "crag_positive",
    "fungal_pneumonia",
    "fungal_culture_positive",
    "immunocompromised",
    "pulmonary_nodule",
    "cavitary_lesion",
}

_MYCOPLASMA_PNEUMONIA_ANCHORS = {
    "diagnosis:支原体肺炎",
    "mycoplasma_naat_positive",
    "mycoplasma_antibody_positive",
    "interstitial_infiltrate",
}


@dataclass
class EligibilityResult:
    diagnosis: str
    status: str
    reason: str
    missing_required_anchors: List[str] = field(default_factory=list)
    satisfied_required_anchors: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    evidence_contributions: List[Dict[str, Any]] = field(default_factory=list)
    evidence_pattern_matches: List[Dict[str, Any]] = field(default_factory=list)
    positive_evidence_score: float = 0.0
    evidence_specificity_score: float = 0.0
    anchor_status: str = ""
    anchor_policy: Dict[str, Any] = field(default_factory=dict)
    anchor_policy_audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceSpecificityCalculator:
    """Approximate evidence specificity from global knowledge plus the local pool."""

    def __init__(self, knowledge: Optional[Any] = None):
        self.knowledge = knowledge
        self.total_diseases = 0
        self.global_support: Dict[str, Set[str]] = {}
        self.match_strengths: Dict[str, Dict[str, float]] = {}
        self._build_global_index()

    def apply(self, candidates: Sequence[Any], evidence: Any = None) -> None:
        observations = list(getattr(evidence, "observations", []) or [])
        obs_by_finding = self._best_observation_by_finding(observations)
        local_counts = self._local_support_counts(candidates)
        pool_size = max(1, len([item for item in candidates if item]))
        for candidate in candidates or []:
            if not candidate:
                continue
            contributions = self._candidate_contributions(
                candidate,
                obs_by_finding,
                local_counts,
                pool_size,
            )
            positive = round(min(1.0, sum(item["contribution"] for item in contributions)), 4)
            specificity = round(
                max((item["combined_es"] for item in contributions), default=0.0),
                4,
            )
            setattr(candidate, "evidence_contributions", contributions)
            setattr(candidate, "positive_evidence_score", positive)
            setattr(candidate, "evidence_specificity_score", specificity)
            components = getattr(candidate, "component_scores", None)
            if isinstance(components, dict):
                components["positive_evidence_score"] = positive
                components["evidence_specificity_score"] = specificity

    def _candidate_contributions(
        self,
        candidate: Any,
        obs_by_finding: Dict[str, Any],
        local_counts: Dict[str, int],
        pool_size: int,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        diagnosis = str(getattr(candidate, "diagnosis", "") or "")
        matched = list(dict.fromkeys(getattr(candidate, "matched_evidence", []) or []))
        diagnostic = set(getattr(candidate, "diagnostic_matched_evidence", []) or [])
        core = set(getattr(candidate, "core_matched_evidence", []) or [])
        generic = set(getattr(candidate, "generic_matched_evidence", []) or [])
        temporal = self._temporal_fit(candidate)
        for finding in matched:
            text = str(finding or "")
            if not text:
                continue
            observation = obs_by_finding.get(text)
            confidence = self._confidence(observation)
            reliability = self._source_reliability(observation)
            match = self._match_strength(diagnosis, text, diagnostic, core, generic)
            global_es = self._global_specificity(text, observation)
            local_es = self._local_specificity(text, local_counts, pool_size)
            combined = max(0.0, min(1.0, 0.7 * global_es + 0.3 * local_es))
            contribution = combined * match * confidence * reliability * temporal
            result.append(
                {
                    "finding": text,
                    "global_es": round(global_es, 4),
                    "local_es": round(local_es, 4),
                    "combined_es": round(combined, 4),
                    "match_strength": round(match, 4),
                    "confidence": round(confidence, 4),
                    "source_reliability": round(reliability, 4),
                    "temporal_fit": round(temporal, 4),
                    "contribution": round(contribution, 4),
                    "source": str(getattr(observation, "source", "") or ""),
                    "evidence_level": str(getattr(observation, "evidence_level", "") or ""),
                }
            )
        result.sort(
            key=lambda item: (
                item["contribution"],
                item["combined_es"],
                item["match_strength"],
            ),
            reverse=True,
        )
        return result[:12]

    def _build_global_index(self) -> None:
        entries = getattr(self.knowledge, "entries", {}) or {}
        self.total_diseases = max(1, len(entries))
        for name, entry in entries.items():
            diagnosis = str(name or entry.get("name") or "")
            for spec in list(entry.get("supporting_evidence", []) or []):
                for finding in self._spec_findings(spec):
                    self.global_support.setdefault(finding, set()).add(diagnosis)
                    self._record_match_strength(diagnosis, finding, spec)
            for group in entry.get("required_groups", []) or []:
                for spec in group or []:
                    for finding in self._spec_findings(spec):
                        self.global_support.setdefault(finding, set()).add(diagnosis)
                        self._record_match_strength(diagnosis, finding, spec, floor=0.75)

    def _record_match_strength(
        self,
        diagnosis: str,
        finding: str,
        spec: Any,
        *,
        floor: float = 0.15,
    ) -> None:
        if not diagnosis or not finding:
            return
        weight = floor
        if isinstance(spec, dict):
            try:
                weight = float(spec.get("weight", weight) or weight)
            except (TypeError, ValueError):
                weight = floor
        strength = max(floor, min(1.0, weight * 2.5))
        by_finding = self.match_strengths.setdefault(diagnosis, {})
        by_finding[finding] = max(by_finding.get(finding, 0.0), strength)

    @staticmethod
    def _spec_findings(spec: Any) -> List[str]:
        if isinstance(spec, str):
            return [spec] if spec else []
        if not isinstance(spec, dict):
            return []
        values = []
        for key in ("finding", "concept", "evidence"):
            value = str(spec.get(key) or "").strip()
            if value:
                values.append(value)
        return values

    @staticmethod
    def _best_observation_by_finding(observations: Sequence[Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for item in observations or []:
            finding = str(getattr(item, "finding", "") or "")
            if not finding:
                continue
            current = result.get(finding)
            if current is None:
                result[finding] = item
                continue
            if EvidenceSpecificityCalculator._observation_quality(item) > EvidenceSpecificityCalculator._observation_quality(current):
                result[finding] = item
        return result

    @staticmethod
    def _observation_quality(item: Any) -> float:
        try:
            confidence = float(getattr(item, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            info = float(getattr(item, "information_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            info = 0.0
        return confidence * 0.4 + info * 0.6

    @staticmethod
    def _local_support_counts(candidates: Sequence[Any]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for candidate in candidates or []:
            for finding in set(getattr(candidate, "matched_evidence", []) or []):
                text = str(finding or "")
                if text:
                    counts[text] = counts.get(text, 0) + 1
        return counts

    def _global_specificity(self, finding: str, observation: Any) -> float:
        try:
            info = float(getattr(observation, "information_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            info = 0.0
        supporters = self.global_support.get(finding) or set()
        if not supporters:
            return max(0.05, min(1.0, info or 0.08))
        rarity = 1.0 - (
            math.log(1.0 + len(supporters)) / max(1.0, math.log(1.0 + self.total_diseases))
        )
        return max(0.05, min(1.0, 0.65 * rarity + 0.35 * (info or 0.08)))

    @staticmethod
    def _local_specificity(
        finding: str,
        local_counts: Dict[str, int],
        pool_size: int,
    ) -> float:
        count = max(1, int(local_counts.get(finding, 1) or 1))
        if pool_size <= 1:
            return 0.5
        return max(0.0, min(1.0, 1.0 - ((count - 1) / max(1, pool_size - 1))))

    def _match_strength(
        self,
        diagnosis: str,
        finding: str,
        diagnostic: Set[str],
        core: Set[str],
        generic: Set[str],
    ) -> float:
        if finding in diagnostic:
            return 1.0
        if finding in core:
            return max(0.75, self.match_strengths.get(diagnosis, {}).get(finding, 0.0))
        if finding in generic:
            return max(0.15, min(0.35, self.match_strengths.get(diagnosis, {}).get(finding, 0.25)))
        return self.match_strengths.get(diagnosis, {}).get(finding, 0.45)

    @staticmethod
    def _confidence(observation: Any) -> float:
        try:
            return max(0.0, min(1.0, float(getattr(observation, "confidence", 0.75) or 0.75)))
        except (TypeError, ValueError):
            return 0.75

    @staticmethod
    def _source_reliability(observation: Any) -> float:
        source = str(getattr(observation, "source", "") or "").lower()
        if source == "reasoning_inference":
            return 0.22
        if source == "raw_case_finding":
            return 0.55
        if any(token in source for token in ("病理", "病原", "培养", "naat", "afb", "pcr")):
            return 0.98
        if any(token in source for token in ("ct", "mri", "x线", "影像", "超声", "心电图", "ecg")):
            return 0.9
        if any(token in source for token in ("血", "尿", "实验", "检测", "镁负荷", "维生素", "pth")):
            return 0.86
        if any(token in source for token in ("体格", "查体", "医生观察")):
            return 0.74
        if any(token in source for token in ("问诊", "主诉", "patient")):
            return 0.6
        return 0.68

    @staticmethod
    def _temporal_fit(candidate: Any) -> float:
        components = getattr(candidate, "component_scores", {}) or {}
        try:
            temporal = float(components.get("temporal", 0.0) or 0.0)
        except (TypeError, ValueError):
            temporal = 0.0
        if temporal <= 0.0:
            return 1.0
        return max(0.7, min(1.0, 0.7 + 0.3 * temporal))


class DiagnosisEligibilityGate:
    """The single authority for whether a candidate may become final primary."""

    def __init__(self, knowledge: Optional[Any] = None):
        self.knowledge = knowledge
        self.specificity_calculator = EvidenceSpecificityCalculator(knowledge)
        self.pattern_evaluator = DiagnosticPatternEvaluator(knowledge)

    @staticmethod
    def _texts(values: Any) -> List[str]:
        if values is None:
            return []
        if isinstance(values, (str, bytes)):
            text = str(values).strip()
            return [text] if text else []
        try:
            iterator = iter(values)
        except TypeError:
            text = str(values).strip()
            return [text] if text else []
        result: List[str] = []
        for value in iterator:
            text = str(value or "").strip()
            if text:
                result.append(text)
        return result

    def evaluate_all(
        self,
        candidates: Sequence[Any],
        evidence: Any = None,
    ) -> Dict[str, Any]:
        self.specificity_calculator.apply(candidates, evidence)
        results = [self.evaluate(candidate, evidence=evidence) for candidate in candidates or [] if candidate]
        for candidate, result in zip([item for item in candidates or [] if item], results):
            self.apply_result(candidate, result)
        return self.summary(results)

    def evaluate(self, candidate: Any, evidence: Any = None) -> EligibilityResult:
        diagnosis = str(getattr(candidate, "diagnosis", "") or "")
        missing = list(dict.fromkeys(getattr(candidate, "required_gaps", []) or []))
        satisfied = self._satisfied_anchors(candidate)
        blockers: List[str] = []
        if getattr(candidate, "hard_contradiction", False):
            blockers.extend(getattr(candidate, "hard_contradicted_evidence", []) or [])
            return self._result(candidate, EXCLUDED, HARD_CONTRADICTION, missing, satisfied, blockers)
        if not getattr(candidate, "matched_evidence", None):
            return self._result(candidate, DIFFERENTIAL_ONLY, NO_SUPPORTING_EVIDENCE, missing, satisfied, blockers)
        if getattr(candidate, "unresolved_evidence_conflict", False):
            blockers.append("unresolved_reasoning_structured_evidence_conflict")
            return self._result(candidate, DEFERRED, CONFLICT_NEEDS_ADJUDICATION, missing, satisfied, blockers)
        pattern_summary: Dict[str, Any] = self.pattern_evaluator.evaluate(candidate, evidence=evidence)
        if pattern_summary.get("has_patterns"):
            pattern_audit = list(pattern_summary.get("matches", []) or [])
            pattern_audit.extend(pattern_summary.get("missing_primary_patterns", []) or [])
            self._append_evidence_patterns(candidate, pattern_audit)
            blockers.extend(pattern_summary.get("blockers", []) or [])
            missing_from_patterns = self._pattern_missing_anchors(pattern_summary)
            if pattern_summary.get("excluded_matches"):
                return self._result(
                    candidate,
                    EXCLUDED,
                    PATTERN_CONTRADICTED,
                    missing,
                    satisfied,
                    blockers or pattern_summary.get("negative_hits", []),
                )
            if pattern_summary.get("primary_eligible_matches"):
                anchor_decision = self._anchor_policy_decision(
                    candidate,
                    evidence=evidence,
                    pattern_summary=pattern_summary,
                )
                if anchor_decision.get("override_status"):
                    self._apply_anchor_policy_audit(candidate, anchor_decision)
                    missing = list(
                        dict.fromkeys(
                            list(missing)
                            + list(anchor_decision.get("missing_required_anchors") or [])
                        )
                    )
                    blockers = list(
                        dict.fromkeys(
                            list(blockers)
                            + list(anchor_decision.get("blockers") or [])
                        )
                    )
                    return self._result(
                        candidate,
                        str(anchor_decision.get("override_status") or DIFFERENTIAL_ONLY),
                        str(anchor_decision.get("reason") or NO_VALID_ANCHOR),
                        missing,
                        satisfied,
                        blockers,
                    )
                self._apply_anchor_policy_audit(candidate, anchor_decision)
                missing = []
                return self._result(candidate, PRIMARY_ELIGIBLE, ANCHORS_SATISFIED, missing, satisfied, blockers)
            if self._has_claim_anchor_contract(candidate):
                anchor_decision = self._anchor_policy_decision(
                    candidate,
                    evidence=evidence,
                    pattern_summary=pattern_summary,
                )
                self._apply_anchor_policy_audit(candidate, anchor_decision)
                missing = list(
                    dict.fromkeys(
                        list(missing)
                        + list(anchor_decision.get("missing_required_anchors") or [])
                    )
                )
                blockers = list(
                    dict.fromkeys(
                        list(blockers)
                        + list(anchor_decision.get("blockers") or [])
                    )
                )
                if str(anchor_decision.get("anchor_status") or "") == ANCHOR_SATISFIED:
                    return self._result(
                        candidate,
                        PRIMARY_ELIGIBLE,
                        ANCHORS_SATISFIED,
                        [],
                        satisfied,
                        blockers,
                    )
                if str(anchor_decision.get("anchor_status") or "") == PATTERN_SUPPORTED_BUT_UNCONFIRMED:
                    return self._result(candidate, DEFERRED, NEEDS_ANCHOR, missing, satisfied, blockers)
            if pattern_summary.get("differential_matches"):
                return self._result(
                    candidate,
                    DIFFERENTIAL_ONLY,
                    PATTERN_CONTRADICTED,
                    list(dict.fromkeys(list(missing) + missing_from_patterns)),
                    satisfied,
                    blockers or pattern_summary.get("negative_hits", []),
                )
            if pattern_summary.get("deferred_matches") or pattern_summary.get("required_primary_patterns"):
                missing = list(dict.fromkeys(list(missing) + missing_from_patterns))
                if self._deferred_worth_followup(candidate):
                    return self._result(candidate, DEFERRED, NEEDS_ANCHOR, missing, satisfied, blockers)
                return self._result(candidate, DIFFERENTIAL_ONLY, WEAK_DIFFERENTIAL_SIGNAL, missing, satisfied, blockers)
        if self._has_claim_anchor_contract(candidate):
            anchor_decision = self._anchor_policy_decision(
                candidate,
                evidence=evidence,
                pattern_summary=pattern_summary,
            )
            self._apply_anchor_policy_audit(candidate, anchor_decision)
            missing = list(
                dict.fromkeys(
                    list(missing)
                    + list(anchor_decision.get("missing_required_anchors") or [])
                )
            )
            blockers = list(
                dict.fromkeys(
                    list(blockers)
                    + list(anchor_decision.get("blockers") or [])
                )
            )
            if str(anchor_decision.get("anchor_status") or "") == ANCHOR_SATISFIED:
                return self._result(
                    candidate,
                    PRIMARY_ELIGIBLE,
                    ANCHORS_SATISFIED,
                    [],
                    satisfied,
                    blockers,
                )
            if str(anchor_decision.get("anchor_status") or "") == PATTERN_SUPPORTED_BUT_UNCONFIRMED:
                return self._result(candidate, DEFERRED, NEEDS_ANCHOR, missing, satisfied, blockers)
        claim_missing = self._claim_missing_anchors(candidate)
        if claim_missing:
            missing = list(dict.fromkeys(list(missing) + claim_missing))
            return self._result(candidate, DEFERRED, NEEDS_ANCHOR, missing, satisfied, blockers)
        if bool(getattr(candidate, "differential_only", False)):
            reason = str(getattr(candidate, "differential_only_reason", "") or WEAK_DIFFERENTIAL_SIGNAL)
            return self._result(candidate, DIFFERENTIAL_ONLY, reason, missing, satisfied, blockers)
        if missing:
            if self._deferred_worth_followup(candidate):
                return self._result(candidate, DEFERRED, NEEDS_ANCHOR, missing, satisfied, blockers)
            return self._result(candidate, DIFFERENTIAL_ONLY, WEAK_DIFFERENTIAL_SIGNAL, missing, satisfied, blockers)
        sanity_gap = self._diagnosis_anchor_sanity_gap(candidate)
        if sanity_gap:
            missing = list(dict.fromkeys(list(missing) + [sanity_gap]))
            return self._result(candidate, DEFERRED, NEEDS_ANCHOR, missing, satisfied, blockers)
        anchor_decision = self._anchor_policy_decision(
            candidate,
            evidence=evidence,
            pattern_summary=pattern_summary if isinstance(pattern_summary, dict) else None,
        )
        if anchor_decision.get("override_status"):
            self._apply_anchor_policy_audit(candidate, anchor_decision)
            missing = list(
                dict.fromkeys(
                    list(missing)
                    + list(anchor_decision.get("missing_required_anchors") or [])
                )
            )
            blockers = list(
                dict.fromkeys(
                    list(blockers)
                    + list(anchor_decision.get("blockers") or [])
                )
            )
            return self._result(
                candidate,
                str(anchor_decision.get("override_status") or DIFFERENTIAL_ONLY),
                str(anchor_decision.get("reason") or NO_VALID_ANCHOR),
                missing,
                satisfied,
                blockers,
            )
        self._apply_anchor_policy_audit(candidate, anchor_decision)
        if self._insufficient_explanation(candidate):
            return self._result(candidate, DIFFERENTIAL_ONLY, INSUFFICIENT_EXPLANATION, missing, satisfied, blockers)
        return self._result(candidate, PRIMARY_ELIGIBLE, ANCHORS_SATISFIED, missing, satisfied, blockers)

    def apply_result(self, candidate: Any, result: EligibilityResult) -> None:
        setattr(candidate, "eligibility_status", result.status)
        setattr(candidate, "eligibility_reason", result.reason)
        setattr(candidate, "missing_required_anchors", list(result.missing_required_anchors))
        setattr(candidate, "satisfied_required_anchors", list(result.satisfied_required_anchors))
        setattr(candidate, "eligibility_blockers", list(result.blockers))
        setattr(candidate, "evidence_pattern_matches", list(result.evidence_pattern_matches))
        setattr(candidate, "positive_evidence_score", result.positive_evidence_score)
        setattr(candidate, "evidence_specificity_score", result.evidence_specificity_score)
        setattr(candidate, "eligibility_anchor_status", result.anchor_status)
        setattr(candidate, "eligibility_anchor_policy", dict(result.anchor_policy))
        setattr(candidate, "eligibility_anchor_policy_audit", dict(result.anchor_policy_audit))
        setattr(candidate, "eligibility_substatus", self._deferred_substatus(candidate, result))
        if result.status == PRIMARY_ELIGIBLE:
            setattr(candidate, "required_met", True)
            setattr(candidate, "required_gaps", [])
        elif result.status in {DEFERRED, DIFFERENTIAL_ONLY, EXCLUDED} and result.missing_required_anchors:
            setattr(candidate, "required_met", False)
            setattr(candidate, "required_gaps", list(result.missing_required_anchors))

    @staticmethod
    def summary(results: Sequence[EligibilityResult]) -> Dict[str, Any]:
        distribution: Dict[str, int] = {}
        deferred: List[str] = []
        excluded: List[str] = []
        primary: List[str] = []
        differential: List[str] = []
        for result in results:
            distribution[result.status] = distribution.get(result.status, 0) + 1
            if result.status == PRIMARY_ELIGIBLE:
                primary.append(result.diagnosis)
            elif result.status == DEFERRED:
                deferred.append(result.diagnosis)
            elif result.status == EXCLUDED:
                excluded.append(result.diagnosis)
            elif result.status == DIFFERENTIAL_ONLY:
                differential.append(result.diagnosis)
        return {
            "eligibility_distribution": distribution,
            "primary_eligible_candidates": primary,
            "deferred_anchor_candidates": deferred,
            "excluded_candidates": excluded,
            "differential_only_candidates": differential,
        }

    def _result(
        self,
        candidate: Any,
        status: str,
        reason: str,
        missing: Sequence[str],
        satisfied: Sequence[str],
        blockers: Sequence[str],
    ) -> EligibilityResult:
        return EligibilityResult(
            diagnosis=str(getattr(candidate, "diagnosis", "") or ""),
            status=status,
            reason=reason,
            missing_required_anchors=list(missing),
            satisfied_required_anchors=list(satisfied),
            blockers=list(blockers),
            evidence_contributions=list(getattr(candidate, "evidence_contributions", []) or []),
            evidence_pattern_matches=list(getattr(candidate, "evidence_pattern_matches", []) or []),
            positive_evidence_score=float(getattr(candidate, "positive_evidence_score", 0.0) or 0.0),
            evidence_specificity_score=float(getattr(candidate, "evidence_specificity_score", 0.0) or 0.0),
            anchor_status=str(getattr(candidate, "eligibility_anchor_status", "") or ""),
            anchor_policy=dict(getattr(candidate, "eligibility_anchor_policy", {}) or {}),
            anchor_policy_audit=dict(getattr(candidate, "eligibility_anchor_policy_audit", {}) or {}),
        )

    def _anchor_policy_decision(
        self,
        candidate: Any,
        *,
        evidence: Any = None,
        pattern_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        entry = self._entry(candidate)
        if not entry:
            return self._anchor_audit(ANCHOR_SATISFIED, policy={})
        if getattr(candidate, "hard_contradiction", False):
            return self._anchor_audit(
                HARD_BLOCKED,
                policy=dict(entry.get("eligibility_anchor_policy") or {}),
                blockers=list(getattr(candidate, "hard_contradicted_evidence", []) or []),
                override_status=EXCLUDED,
                reason=HARD_CONTRADICTION,
            )
        policy = dict(entry.get("eligibility_anchor_policy") or {})
        required_groups = list(entry.get("required_groups", []) or [])
        diagnostic_patterns = list(entry.get("diagnostic_patterns", []) or [])
        if not policy:
            if required_groups or diagnostic_patterns:
                return self._anchor_audit(ANCHOR_SATISFIED, policy={})
            if self._legacy_empty_group_has_anchor(candidate):
                return self._anchor_audit(
                    ANCHOR_SATISFIED,
                    policy={"mode": "legacy_empty_group_specific_signal"},
                )
            return self._anchor_audit(
                NO_VALID_ANCHOR,
                policy={"mode": "missing_anchor_policy_inventory_only"},
                missing=["valid_diagnostic_anchor"],
                blockers=["empty_required_groups_without_valid_anchor"],
            )

        accepted = set(self._texts(policy.get("accepted_anchors") or []))
        matched_anchor_types: List[str] = []
        missing_anchor_groups: List[str] = []
        partial_signal = False

        if "required_evidence_group" in accepted and required_groups:
            if self._satisfied_anchors(candidate):
                matched_anchor_types.append("required_evidence_group")
            else:
                missing_anchor_groups.append("required_evidence_group")
        if "diagnostic_pattern" in accepted:
            primary_matches = list((pattern_summary or {}).get("primary_eligible_matches") or [])
            if primary_matches:
                matched_anchor_types.append("diagnostic_pattern")
            elif diagnostic_patterns:
                missing_anchor_groups.append("diagnostic_pattern")
        if "claim_resolution_contract" in accepted:
            claim_anchor = dict(getattr(candidate, "claim_anchor_evaluation", {}) or {})
            claim_anchor_status = str(claim_anchor.get("anchor_status_after") or "")
            if claim_anchor_status == ANCHOR_SATISFIED:
                matched_anchor_types.append("claim_resolution_contract")
            elif claim_anchor:
                if claim_anchor_status == PATTERN_SUPPORTED_BUT_UNCONFIRMED:
                    partial_signal = True
                missing_anchor_groups.extend(
                    str(item)
                    for item in claim_anchor.get("unresolved_claims", []) or []
                    if str(item)
                )
                missing_anchor_groups.extend(
                    str(item)
                    for item in claim_anchor.get("contradicted_claims", []) or []
                    if str(item)
                )
                missing_anchor_groups.extend(
                    str(item)
                    for item in claim_anchor.get("conflicted_claims", []) or []
                    if str(item)
                )
            elif entry.get("claim_anchor_contract"):
                missing_anchor_groups.extend(
                    str(item)
                    for item in (
                        entry.get("claim_anchor_contract", {}).get("required_claims", [])
                        or []
                    )
                    if str(item)
                )
        if "disease_specific_anchor" in accepted:
            pattern_result = self._match_anchor_patterns(
                candidate,
                list(policy.get("disease_specific_anchor_patterns") or policy.get("anchor_patterns") or []),
            )
            if pattern_result["matched"]:
                matched_anchor_types.append("disease_specific_anchor")
            else:
                partial_signal = partial_signal or bool(pattern_result["partial"])
                missing_anchor_groups.extend(pattern_result["missing"])
        if "direct_diagnostic_evidence" in accepted:
            direct_result = self._match_direct_diagnostic_anchor(candidate, policy)
            if direct_result["matched"]:
                matched_anchor_types.append("direct_diagnostic_evidence")
            else:
                partial_signal = partial_signal or bool(direct_result["partial"])
                missing_anchor_groups.extend(direct_result["missing"])

        if matched_anchor_types:
            return self._anchor_audit(
                ANCHOR_SATISFIED,
                policy=policy,
                matched_anchor_types=matched_anchor_types,
                missing=missing_anchor_groups,
            )

        if partial_signal:
            disposition = str(
                policy.get("atypical_anchor_disposition")
                or policy.get("no_anchor_disposition")
                or DEFERRED
            )
            return self._anchor_audit(
                PATTERN_SUPPORTED_BUT_UNCONFIRMED,
                policy=policy,
                missing=missing_anchor_groups or ["confirmatory_anchor"],
                blockers=["pattern_supported_but_anchor_unconfirmed"],
                override_status=self._eligibility_status_from_disposition(disposition),
                reason=NEEDS_ANCHOR,
            )

        disposition = str(policy.get("no_anchor_disposition") or DIFFERENTIAL_ONLY)
        return self._anchor_audit(
            NO_VALID_ANCHOR,
            policy=policy,
            missing=missing_anchor_groups or ["valid_diagnostic_anchor"],
            blockers=["no_valid_diagnostic_anchor"],
            override_status=self._eligibility_status_from_disposition(disposition),
            reason=NO_VALID_ANCHOR,
        )

    def _apply_anchor_policy_audit(self, candidate: Any, audit: Dict[str, Any]) -> None:
        if not audit:
            return
        setattr(candidate, "eligibility_anchor_status", str(audit.get("anchor_status") or ""))
        setattr(candidate, "eligibility_anchor_policy", dict(audit.get("policy") or {}))
        setattr(candidate, "eligibility_anchor_policy_audit", dict(audit))

    def _anchor_audit(
        self,
        anchor_status: str,
        *,
        policy: Dict[str, Any],
        matched_anchor_types: Optional[Sequence[str]] = None,
        missing: Optional[Sequence[str]] = None,
        blockers: Optional[Sequence[str]] = None,
        override_status: str = "",
        reason: str = "",
    ) -> Dict[str, Any]:
        return {
            "anchor_status": anchor_status,
            "policy": dict(policy or {}),
            "matched_anchor_types": list(matched_anchor_types or []),
            "missing_required_anchors": list(dict.fromkeys(missing or [])),
            "blockers": list(dict.fromkeys(blockers or [])),
            "override_status": override_status,
            "reason": reason,
        }

    @staticmethod
    def _eligibility_status_from_disposition(disposition: str) -> str:
        if disposition in {PRIMARY_ELIGIBLE, DEFERRED, DIFFERENTIAL_ONLY, EXCLUDED}:
            return disposition
        lowered = str(disposition or "").lower()
        if "deferred" in lowered:
            return DEFERRED
        if "exclude" in lowered:
            return EXCLUDED
        if "primary" in lowered:
            return PRIMARY_ELIGIBLE
        return DIFFERENTIAL_ONLY

    def _match_anchor_patterns(
        self,
        candidate: Any,
        patterns: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        any_partial = False
        missing: List[str] = []
        for pattern in patterns or []:
            if not isinstance(pattern, dict):
                continue
            required = pattern.get("required") or pattern.get("all_of") or []
            result = self._condition_result({"all_of": required}, matched)
            if result["matched"]:
                return {"matched": True, "partial": False, "missing": []}
            any_partial = any_partial or bool(result["matched_findings"])
            pattern_id = str(pattern.get("pattern_id") or "anchor_pattern")
            for item in result["missing_findings"]:
                missing.append(f"{pattern_id}:{item}")
        return {
            "matched": False,
            "partial": any_partial,
            "missing": list(dict.fromkeys(missing)),
        }

    def _match_direct_diagnostic_anchor(
        self,
        candidate: Any,
        policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        config = dict(policy.get("direct_diagnostic_evidence") or {})
        diagnosis = str(getattr(candidate, "diagnosis", "") or "")
        direct = config.get("any_of") or [f"diagnosis:{diagnosis}"]
        compatible = config.get("requires_any") or []
        direct_result = self._condition_result({"any_of": direct}, matched)
        compatible_result = (
            self._condition_result({"any_of": compatible}, matched)
            if compatible
            else {"matched": True, "matched_findings": [], "missing_findings": []}
        )
        return {
            "matched": bool(direct_result["matched"] and compatible_result["matched"]),
            "partial": bool(direct_result["matched"] or compatible_result["matched"]),
            "missing": list(
                dict.fromkeys(
                    [f"direct_diagnostic_evidence:{item}" for item in direct_result["missing_findings"]]
                    + [
                        f"direct_diagnostic_compatible_manifestation:{item}"
                        for item in compatible_result["missing_findings"]
                    ]
                )
            ),
        }

    def _condition_result(self, condition: Any, matched: Set[str]) -> Dict[str, Any]:
        if isinstance(condition, str):
            text = str(condition or "").strip()
            hit = text in matched
            return {
                "matched": hit,
                "matched_findings": [text] if hit else [],
                "missing_findings": [] if hit else [text],
            }
        if not isinstance(condition, dict):
            text = str(condition or "")
            return {"matched": False, "matched_findings": [], "missing_findings": [text]}
        if "finding" in condition:
            return self._condition_result(str(condition.get("finding") or ""), matched)
        if "any_of" in condition:
            matched_findings: List[str] = []
            missing_findings: List[str] = []
            for item in condition.get("any_of") or []:
                result = self._condition_result(item, matched)
                matched_findings.extend(result["matched_findings"])
                missing_findings.extend(result["missing_findings"])
            return {
                "matched": bool(matched_findings),
                "matched_findings": list(dict.fromkeys(matched_findings)),
                "missing_findings": list(dict.fromkeys(missing_findings)),
            }
        if "all_of" in condition:
            matched_findings = []
            missing_findings = []
            all_matched = True
            for item in condition.get("all_of") or []:
                result = self._condition_result(item, matched)
                matched_findings.extend(result["matched_findings"])
                missing_findings.extend(result["missing_findings"])
                all_matched = all_matched and bool(result["matched"])
            return {
                "matched": all_matched,
                "matched_findings": list(dict.fromkeys(matched_findings)),
                "missing_findings": list(dict.fromkeys(missing_findings)),
            }
        if "min_count" in condition:
            try:
                minimum = int(condition.get("min_count") or 0)
            except (TypeError, ValueError):
                minimum = 0
            matched_findings = []
            missing_findings = []
            for item in condition.get("of") or []:
                result = self._condition_result(item, matched)
                matched_findings.extend(result["matched_findings"])
                missing_findings.extend(result["missing_findings"])
            return {
                "matched": len(set(matched_findings)) >= max(0, minimum),
                "matched_findings": list(dict.fromkeys(matched_findings)),
                "missing_findings": list(dict.fromkeys(missing_findings)),
            }
        if "not_any_of" in condition:
            result = self._condition_result({"any_of": condition.get("not_any_of") or []}, matched)
            return {
                "matched": not bool(result["matched"]),
                "matched_findings": [],
                "missing_findings": result["matched_findings"],
            }
        return {"matched": False, "matched_findings": [], "missing_findings": [str(condition)]}

    @staticmethod
    def _legacy_empty_group_has_anchor(candidate: Any) -> bool:
        matched = {str(item) for item in getattr(candidate, "matched_evidence", []) or []}
        diagnosis = str(getattr(candidate, "diagnosis", "") or "")
        if f"diagnosis:{diagnosis}" in matched:
            return True
        if getattr(candidate, "diagnostic_matched_evidence", None):
            return True
        if float(getattr(candidate, "diagnostic_evidence_score", 0.0) or 0.0) >= 0.25:
            return True
        if (
            float(getattr(candidate, "core_evidence_score", 0.0) or 0.0) >= 0.50
            and float(getattr(candidate, "evidence_specificity_score", 0.0) or 0.0) >= 0.55
        ):
            return True
        components = getattr(candidate, "component_scores", {}) or {}
        return float(components.get("objective_evidence", 0.0) or 0.0) >= 1.0

    def _deferred_substatus(self, candidate: Any, result: EligibilityResult) -> str:
        if result.status != DEFERRED:
            return ""
        if not bool(getattr(candidate, "submittable", True)):
            return DEFERRED_UNRESOLVED_NAMING
        if result.reason == CONFLICT_NEEDS_ADJUDICATION:
            return DEFERRED_NEEDS_OBSERVED_EVIDENCE
        claims = [
            item
            for item in getattr(candidate, "unresolved_critical_evidence_claims", []) or []
            if isinstance(item, dict)
        ]
        if claims:
            if any(
                str(item.get("claim_type") or "") == "derived_pattern"
                or item.get("required_inputs")
                for item in claims
            ):
                return DEFERRED_NEEDS_DERIVED_PATTERN
            return DEFERRED_NEEDS_OBSERVED_EVIDENCE
        for pattern in getattr(candidate, "evidence_pattern_matches", []) or []:
            if isinstance(pattern, dict) and pattern.get("missing_required_groups"):
                return DEFERRED_NEEDS_DERIVED_PATTERN
        entry = self._entry(candidate)
        if (
            result.missing_required_anchors
            and (
                entry.get("discriminating_exams")
                or entry.get("strong_verification_exams")
                or entry.get("required_exams")
            )
        ):
            return DEFERRED_NEEDS_CONFIRMATORY_EXAM
        if result.missing_required_anchors:
            return DEFERRED_NEEDS_CONFIRMATORY_EXAM
        return DEFERRED_LOW_PRIORITY

    @staticmethod
    def _satisfied_anchors(candidate: Any) -> List[str]:
        values = []
        for finding in list(getattr(candidate, "diagnostic_matched_evidence", []) or []):
            values.append(str(finding))
        for finding in list(getattr(candidate, "core_matched_evidence", []) or []):
            values.append(str(finding))
        return list(dict.fromkeys(item for item in values if item))

    def _deferred_worth_followup(self, candidate: Any) -> bool:
        entry = self._entry(candidate)
        if entry.get("discriminating_exams") or entry.get("strong_verification_exams") or entry.get("required_exams"):
            return True
        if float(getattr(candidate, "source_prior", 0.0) or 0.0) >= 0.45:
            return True
        if float(getattr(candidate, "core_explanatory_coverage", 0.0) or 0.0) >= 0.35:
            return True
        if float(getattr(candidate, "coverage_score", 0.0) or 0.0) >= 0.45:
            return True
        if float(getattr(candidate, "evidence_specificity_score", 0.0) or 0.0) >= 0.65:
            return True
        return bool(getattr(candidate, "core_matched_evidence", None) or getattr(candidate, "diagnostic_matched_evidence", None))

    def _has_claim_anchor_contract(self, candidate: Any) -> bool:
        entry = self._entry(candidate)
        policy = dict(entry.get("eligibility_anchor_policy") or {})
        accepted = set(self._texts(policy.get("accepted_anchors") or []))
        return bool(
            "claim_resolution_contract" in accepted
            and entry.get("claim_anchor_contract")
            and getattr(candidate, "claim_anchor_evaluation", None)
        )

    def _diagnosis_anchor_sanity_gap(self, candidate: Any) -> str:
        diagnosis = str(getattr(candidate, "diagnosis", "") or "")
        entry = self._entry(candidate)
        if entry.get("diagnostic_patterns"):
            return ""
        if diagnosis == "肺隐球菌病":
            matched = {str(item) for item in getattr(candidate, "matched_evidence", []) or []}
            if matched & _PULMONARY_CRYPTOCOCCOSIS_ANCHORS:
                return ""
            return "pulmonary_cryptococcosis_requires_fungal_or_cryptococcal_anchor"
        if diagnosis == "支原体肺炎":
            matched = {str(item) for item in getattr(candidate, "matched_evidence", []) or []}
            if matched & _MYCOPLASMA_PNEUMONIA_ANCHORS:
                return ""
            return "mycoplasma_pneumonia_requires_pathogen_or_interstitial_anchor"
        return ""

    @staticmethod
    def _claim_missing_anchors(candidate: Any) -> List[str]:
        missing: List[str] = []
        for claim in getattr(candidate, "unresolved_critical_evidence_claims", []) or []:
            if not isinstance(claim, dict):
                continue
            target = str(claim.get("target_evidence") or "").strip()
            claim_id = str(claim.get("claim_id") or "").strip()
            if target and claim_id:
                missing.append(f"{claim_id}:{target}")
            elif target:
                missing.append(target)
        return list(dict.fromkeys(item for item in missing if item))

    @staticmethod
    def _append_evidence_patterns(candidate: Any, patterns: Sequence[Dict[str, Any]]) -> None:
        if not patterns:
            return
        existing = list(getattr(candidate, "evidence_pattern_matches", []) or [])
        seen = {
            str(item.get("pattern_id") or item.get("pattern") or "")
            for item in existing
            if isinstance(item, dict)
        }
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            pattern_id = str(pattern.get("pattern_id") or pattern.get("pattern") or "")
            if pattern_id and pattern_id in seen:
                continue
            existing.append(dict(pattern))
            if pattern_id:
                seen.add(pattern_id)
        setattr(candidate, "evidence_pattern_matches", existing)

    @staticmethod
    def _pattern_missing_anchors(pattern_summary: Dict[str, Any]) -> List[str]:
        missing: List[str] = []
        source_patterns = list(pattern_summary.get("missing_primary_patterns", []) or [])
        source_patterns.extend(pattern_summary.get("deferred_matches", []) or [])
        for pattern in source_patterns:
            if not isinstance(pattern, dict):
                continue
            pattern_id = str(pattern.get("pattern_id") or "")
            for group in pattern.get("missing_required_groups", []) or []:
                if not isinstance(group, dict):
                    continue
                condition = group.get("condition")
                label = _render_pattern_condition(condition)
                if pattern_id and label:
                    missing.append(f"{pattern_id}:{label}")
                elif label:
                    missing.append(label)
        return list(dict.fromkeys(item for item in missing if item))

    @staticmethod
    def _insufficient_explanation(candidate: Any) -> bool:
        if not bool(getattr(candidate, "required_met", False)):
            return False
        if float(getattr(candidate, "diagnostic_evidence_score", 0.0) or 0.0) >= 0.25:
            return False
        if float(getattr(candidate, "core_evidence_score", 0.0) or 0.0) >= 0.35:
            return False
        if int(getattr(candidate, "residual_core_evidence_count", 0) or 0) >= 3:
            return float(getattr(candidate, "coverage_score", 0.0) or 0.0) < 0.45
        return False

    def _entry(self, candidate: Any) -> Dict[str, Any]:
        if not self.knowledge:
            return {}
        try:
            return dict(self.knowledge.get(str(getattr(candidate, "diagnosis", "") or "")) or {})
        except Exception:
            return {}


def _render_pattern_condition(condition: Any) -> str:
    if isinstance(condition, str):
        return condition
    if isinstance(condition, dict):
        if "finding" in condition:
            return str(condition.get("finding") or "")
        if "any_of" in condition:
            return "|".join(
                item
                for item in (_render_pattern_condition(value) for value in condition.get("any_of") or [])
                if item
            )
        if "all_of" in condition:
            return "+".join(
                item
                for item in (_render_pattern_condition(value) for value in condition.get("all_of") or [])
                if item
            )
        if "min_count" in condition:
            nested = "|".join(
                item
                for item in (_render_pattern_condition(value) for value in condition.get("of") or [])
                if item
            )
            return f"min_count_{condition.get('min_count')}:{nested}" if nested else ""
        if "not_any_of" in condition:
            nested = "|".join(
                item
                for item in (_render_pattern_condition(value) for value in condition.get("not_any_of") or [])
                if item
            )
            return f"not_any_of:{nested}" if nested else ""
    return str(condition or "")


def eligibility_status(candidate: Any) -> str:
    return str(getattr(candidate, "eligibility_status", "") or "")


def is_primary_eligible(candidate: Any) -> bool:
    return eligibility_status(candidate) == PRIMARY_ELIGIBLE


def is_deferred(candidate: Any) -> bool:
    return eligibility_status(candidate) == DEFERRED


def is_excluded(candidate: Any) -> bool:
    return eligibility_status(candidate) == EXCLUDED
