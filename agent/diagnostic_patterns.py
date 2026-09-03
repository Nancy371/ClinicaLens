"""Evaluate declarative diagnostic evidence patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


PRIMARY_ELIGIBLE = "PrimaryEligible"
DEFERRED = "Deferred"
DIFFERENTIAL_ONLY = "DifferentialOnly"
EXCLUDED = "Excluded"


@dataclass
class ConditionResult:
    matched: bool
    matched_findings: List[str] = field(default_factory=list)
    missing_findings: List[str] = field(default_factory=list)
    objective_source_satisfied: bool = True


@dataclass
class PatternEvaluation:
    pattern_id: str
    pattern_type: str
    matched: bool
    matched_required_groups: List[Dict[str, Any]] = field(default_factory=list)
    missing_required_groups: List[Dict[str, Any]] = field(default_factory=list)
    negative_hits: List[str] = field(default_factory=list)
    effect: Dict[str, Any] = field(default_factory=dict)
    objective_source_satisfied: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "matched_required_groups": list(self.matched_required_groups),
            "missing_required_groups": list(self.missing_required_groups),
            "negative_hits": list(self.negative_hits),
            "effect": dict(self.effect),
            "objective_source_satisfied": bool(self.objective_source_satisfied),
            "matched": bool(self.matched),
        }
        role = self.pattern_type
        if role:
            data["role"] = role
        eligibility = str(self.effect.get("eligibility") or "")
        if eligibility:
            data["eligibility"] = eligibility
        return data


class DiagnosticPatternEvaluator:
    """Run all_of/any_of/min_count diagnostic patterns against candidate evidence."""

    def __init__(self, knowledge: Optional[Any] = None):
        self.knowledge = knowledge

    def evaluate(self, candidate: Any, evidence: Any = None) -> Dict[str, Any]:
        patterns = self._patterns(candidate)
        if not patterns:
            return {
                "has_patterns": False,
                "matches": [],
                "primary_eligible_matches": [],
                "deferred_matches": [],
                "differential_matches": [],
                "excluded_matches": [],
                "missing_primary_patterns": [],
                "required_primary_patterns": [],
                "negative_hits": [],
                "blockers": [],
            }

        context = _EvidenceContext(candidate, evidence)
        evaluations: List[PatternEvaluation] = []
        for pattern in patterns:
            evaluation = self.evaluate_pattern(pattern, context)
            if evaluation is not None:
                evaluations.append(evaluation)

        matches = [item.to_dict() for item in evaluations if item.matched]
        primary = [
            item.to_dict()
            for item in evaluations
            if item.matched and self._effect_status(item) == PRIMARY_ELIGIBLE
        ]
        deferred = [
            item.to_dict()
            for item in evaluations
            if item.matched and self._effect_status(item) == DEFERRED
        ]
        differential = [
            item.to_dict()
            for item in evaluations
            if item.matched and self._effect_status(item) == DIFFERENTIAL_ONLY
        ]
        excluded = [
            item.to_dict()
            for item in evaluations
            if item.matched and self._effect_status(item) == EXCLUDED
        ]
        missing_primary = [
            item.to_dict()
            for item in evaluations
            if (not item.matched) and self._effect_status(item) == PRIMARY_ELIGIBLE
        ]
        required_primary = [
            item.to_dict()
            for item in evaluations
            if self._effect_status(item) == PRIMARY_ELIGIBLE
        ]
        blockers = sorted(
            {
                hit
                for item in evaluations
                if item.matched
                and self._effect_status(item) in {DIFFERENTIAL_ONLY, EXCLUDED}
                for hit in item.negative_hits
            }
        )
        negative_hits = sorted({hit for item in evaluations for hit in item.negative_hits})
        return {
            "has_patterns": bool(evaluations),
            "matches": matches,
            "primary_eligible_matches": primary,
            "deferred_matches": deferred,
            "differential_matches": differential,
            "excluded_matches": excluded,
            "missing_primary_patterns": missing_primary,
            "required_primary_patterns": required_primary,
            "negative_hits": negative_hits,
            "blockers": blockers,
        }

    def evaluate_pattern(
        self,
        pattern: Dict[str, Any],
        context: "_EvidenceContext",
    ) -> Optional[PatternEvaluation]:
        if not isinstance(pattern, dict):
            return None
        effect = dict(pattern.get("effect") or {})
        if not effect.get("eligibility"):
            return None
        pattern_id = str(pattern.get("pattern_id") or pattern.get("id") or "").strip()
        if not pattern_id:
            return None
        pattern_type = str(pattern.get("pattern_type") or pattern.get("role") or "")
        required = list(pattern.get("required") or [])
        negative_if = list(pattern.get("negative_if") or [])
        not_any_of = list(pattern.get("not_any_of") or [])
        if not required and not negative_if:
            return None

        matched_groups: List[Dict[str, Any]] = []
        missing_groups: List[Dict[str, Any]] = []
        negative_hits: List[str] = []
        objective_ok = True

        not_hits = self._matched_findings_for_any(not_any_of, context)
        if not_hits:
            negative_hits.extend(not_hits)

        required_positive_only = pattern_type != "negative_pattern"
        for index, condition in enumerate(required):
            result = self._evaluate_condition(
                condition,
                context,
                positive_only=required_positive_only,
            )
            group_record = self._group_record(index, condition, result)
            if result.matched:
                matched_groups.append(group_record)
            else:
                missing_groups.append(group_record)
            objective_ok = objective_ok and result.objective_source_satisfied

        negative_condition_matched = False
        for condition in negative_if:
            result = self._evaluate_condition(condition, context, positive_only=False)
            if result.matched:
                negative_condition_matched = True
                negative_hits.extend(result.matched_findings)

        logic = str(pattern.get("logic") or "all_of")
        required_matched = self._top_level_required_matched(
            logic,
            required,
            matched_groups,
            missing_groups,
            pattern,
        )
        if not required:
            required_matched = True

        requires_objective = bool(pattern.get("requires_objective_source", False))
        if requires_objective and required and not objective_ok:
            required_matched = False

        matched = required_matched and not bool(not_hits)
        if negative_if:
            matched = matched and negative_condition_matched

        if matched and pattern_type == "negative_pattern":
            for group in matched_groups:
                negative_hits.extend(group.get("matched_findings", []) or [])

        return PatternEvaluation(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            matched=matched,
            matched_required_groups=matched_groups,
            missing_required_groups=missing_groups,
            negative_hits=sorted(set(negative_hits)),
            effect=effect,
            objective_source_satisfied=bool(objective_ok),
        )

    def _patterns(self, candidate: Any) -> List[Dict[str, Any]]:
        entry = self._entry(candidate)
        return [dict(item) for item in entry.get("diagnostic_patterns", []) or [] if isinstance(item, dict)]

    def _entry(self, candidate: Any) -> Dict[str, Any]:
        if not self.knowledge:
            return {}
        diagnosis = str(getattr(candidate, "diagnosis", "") or "")
        try:
            return dict(self.knowledge.get(diagnosis) or {})
        except Exception:
            return {}

    @staticmethod
    def _effect_status(item: PatternEvaluation) -> str:
        return str(item.effect.get("eligibility") or "")

    def _evaluate_condition(
        self,
        condition: Any,
        context: "_EvidenceContext",
        *,
        positive_only: bool,
    ) -> ConditionResult:
        if isinstance(condition, str):
            return self._finding_result(condition, context, positive_only=positive_only)
        if not isinstance(condition, dict):
            return ConditionResult(False, missing_findings=[str(condition)])
        if "finding" in condition:
            return self._finding_result(
                str(condition.get("finding") or ""),
                context,
                positive_only=positive_only,
            )
        if "any_of" in condition:
            return self._any_of(condition.get("any_of") or [], context, positive_only=positive_only)
        if "all_of" in condition:
            return self._all_of(condition.get("all_of") or [], context, positive_only=positive_only)
        if "min_count" in condition:
            try:
                minimum = int(condition.get("min_count") or 0)
            except (TypeError, ValueError):
                minimum = 0
            return self._min_count(
                minimum,
                condition.get("of") or [],
                context,
                positive_only=positive_only,
            )
        if "not_any_of" in condition:
            hits = self._matched_findings_for_any(condition.get("not_any_of") or [], context)
            return ConditionResult(
                matched=not bool(hits),
                matched_findings=[],
                missing_findings=hits,
                objective_source_satisfied=True,
            )
        return ConditionResult(False, missing_findings=[str(condition)])

    def _finding_result(
        self,
        finding: str,
        context: "_EvidenceContext",
        *,
        positive_only: bool,
    ) -> ConditionResult:
        text = str(finding or "").strip()
        if not text:
            return ConditionResult(False)
        matched = context.has_positive(text) if positive_only else context.has_any(text)
        if not matched:
            return ConditionResult(False, missing_findings=[text])
        return ConditionResult(
            True,
            matched_findings=[text],
            objective_source_satisfied=context.has_objective_positive(text),
        )

    def _any_of(
        self,
        items: Sequence[Any],
        context: "_EvidenceContext",
        *,
        positive_only: bool,
    ) -> ConditionResult:
        matched: List[str] = []
        missing: List[str] = []
        objective = False
        for item in items or []:
            result = self._evaluate_condition(item, context, positive_only=positive_only)
            if result.matched:
                matched.extend(result.matched_findings)
                objective = objective or result.objective_source_satisfied
            else:
                missing.extend(result.missing_findings)
        return ConditionResult(
            bool(matched),
            matched_findings=sorted(set(matched)),
            missing_findings=sorted(set(missing)),
            objective_source_satisfied=objective if matched else True,
        )

    def _all_of(
        self,
        items: Sequence[Any],
        context: "_EvidenceContext",
        *,
        positive_only: bool,
    ) -> ConditionResult:
        matched: List[str] = []
        missing: List[str] = []
        objective = True
        all_matched = True
        for item in items or []:
            result = self._evaluate_condition(item, context, positive_only=positive_only)
            if result.matched:
                matched.extend(result.matched_findings)
                objective = objective and result.objective_source_satisfied
            else:
                all_matched = False
                missing.extend(result.missing_findings)
        return ConditionResult(
            all_matched,
            matched_findings=sorted(set(matched)),
            missing_findings=sorted(set(missing)),
            objective_source_satisfied=objective,
        )

    def _min_count(
        self,
        minimum: int,
        items: Sequence[Any],
        context: "_EvidenceContext",
        *,
        positive_only: bool,
    ) -> ConditionResult:
        matched: List[str] = []
        missing: List[str] = []
        objective_flags: List[bool] = []
        for item in items or []:
            result = self._evaluate_condition(item, context, positive_only=positive_only)
            if result.matched:
                matched.extend(result.matched_findings)
                objective_flags.append(result.objective_source_satisfied)
            else:
                missing.extend(result.missing_findings)
        return ConditionResult(
            len(objective_flags) >= max(0, minimum),
            matched_findings=sorted(set(matched)),
            missing_findings=sorted(set(missing)),
            objective_source_satisfied=all(objective_flags) if objective_flags else True,
        )

    def _matched_findings_for_any(
        self,
        items: Sequence[Any],
        context: "_EvidenceContext",
    ) -> List[str]:
        hits: List[str] = []
        for item in items or []:
            result = self._evaluate_condition(item, context, positive_only=True)
            if result.matched:
                hits.extend(result.matched_findings)
        return sorted(set(hits))

    @staticmethod
    def _top_level_required_matched(
        logic: str,
        required: Sequence[Any],
        matched_groups: Sequence[Dict[str, Any]],
        missing_groups: Sequence[Dict[str, Any]],
        pattern: Dict[str, Any],
    ) -> bool:
        if not required:
            return True
        if logic == "any_of":
            return bool(matched_groups)
        if logic == "min_count":
            try:
                minimum = int(pattern.get("min_count") or 0)
            except (TypeError, ValueError):
                minimum = 0
            return len(matched_groups) >= max(0, minimum)
        return not bool(missing_groups)

    @staticmethod
    def _group_record(index: int, condition: Any, result: ConditionResult) -> Dict[str, Any]:
        return {
            "index": index,
            "condition": _condition_label(condition),
            "matched_findings": list(result.matched_findings),
            "missing_findings": list(result.missing_findings),
            "objective_source_satisfied": bool(result.objective_source_satisfied),
        }


class _EvidenceContext:
    def __init__(self, candidate: Any, evidence: Any = None):
        self.positive: Set[str] = {
            str(item)
            for item in getattr(candidate, "matched_evidence", []) or []
            if str(item or "").strip()
        }
        self.contradicted: Set[str] = {
            str(item)
            for item in (
                list(getattr(candidate, "soft_contradicted_evidence", []) or [])
                + list(getattr(candidate, "hard_contradicted_evidence", []) or [])
                + list(getattr(candidate, "contradicted_evidence", []) or [])
            )
            if str(item or "").strip()
        }
        self.positive_sources: Dict[str, Set[str]] = {}
        self._load_sources(candidate, evidence)

    def has_positive(self, finding: str) -> bool:
        return finding in self.positive

    def has_any(self, finding: str) -> bool:
        return finding in self.positive or finding in self.contradicted

    def has_objective_positive(self, finding: str) -> bool:
        if finding not in self.positive:
            return False
        sources = self.positive_sources.get(finding)
        if sources:
            return any(source != "reasoning_inference" for source in sources)
        return True

    def _load_sources(self, candidate: Any, evidence: Any = None) -> None:
        for item in getattr(candidate, "evidence_contributions", []) or []:
            if not isinstance(item, dict):
                continue
            finding = str(item.get("finding") or "")
            if not finding:
                continue
            self.positive_sources.setdefault(finding, set()).add(str(item.get("source") or ""))
        for item in getattr(evidence, "observations", []) or []:
            finding = str(getattr(item, "finding", "") or "")
            if not finding or str(getattr(item, "polarity", "positive") or "positive") != "positive":
                continue
            self.positive_sources.setdefault(finding, set()).add(str(getattr(item, "source", "") or ""))


def _condition_label(condition: Any) -> Any:
    if isinstance(condition, str):
        return condition
    if isinstance(condition, dict):
        if "finding" in condition:
            return str(condition.get("finding") or "")
        if "any_of" in condition:
            return {"any_of": [_condition_label(item) for item in condition.get("any_of") or []]}
        if "all_of" in condition:
            return {"all_of": [_condition_label(item) for item in condition.get("all_of") or []]}
        if "min_count" in condition:
            return {
                "min_count": condition.get("min_count"),
                "of": [_condition_label(item) for item in condition.get("of") or []],
            }
        if "not_any_of" in condition:
            return {
                "not_any_of": [_condition_label(item) for item in condition.get("not_any_of") or []]
            }
    return str(condition)
