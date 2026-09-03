"""Plan deterministic searches for canonical evidence hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .evidence_hypothesis import (
    COUNTEREVIDENCE,
    DERIVED_PATTERN,
    MISSING_CONFIRMATION,
    OBSERVED_FINDING,
    EvidenceHypothesis,
)
from .evidence_registry import EvidenceDefinitionRegistry


@dataclass
class EvidenceQueryTask:
    query_task_id: str
    hypothesis_id: str
    candidate: str
    target_evidence_id: str
    claim_type: str
    strategy: str
    aliases: List[str] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    positive_terms: List[str] = field(default_factory=list)
    negative_terms: List[str] = field(default_factory=list)
    ambiguous_terms: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    polarity_required: bool = True
    value_required: bool = False
    priority: int = 0
    importance: str = "medium"
    expected_effect: str = ""
    recommended_exam: str = ""
    case_version: int = 0
    confidence: float = 0.0
    entity_id: str = ""
    invalid_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Compatibility with existing evidence claim consumers.
        payload["claim_id"] = self.hypothesis_id
        payload["target_evidence"] = self.target_evidence_id
        payload["diagnosis_hypothesis"] = self.candidate
        payload["search_terms"] = list(self.aliases)
        return payload

    @classmethod
    def from_any(cls, value: Any) -> Optional["EvidenceQueryTask"]:
        if isinstance(value, EvidenceQueryTask):
            return value
        if not isinstance(value, dict):
            return None
        target = str(
            value.get("target_evidence_id")
            or value.get("target_evidence")
            or value.get("evidence_id")
            or ""
        ).strip()
        task_id = str(value.get("query_task_id") or value.get("task_id") or "").strip()
        hypothesis_id = str(
            value.get("hypothesis_id") or value.get("claim_id") or value.get("id") or ""
        ).strip()
        if not target or not hypothesis_id:
            return None
        return cls(
            query_task_id=task_id or f"QT-{hypothesis_id}",
            hypothesis_id=hypothesis_id,
            candidate=str(value.get("candidate") or value.get("diagnosis_hypothesis") or ""),
            target_evidence_id=target,
            claim_type=str(value.get("claim_type") or OBSERVED_FINDING),
            strategy=str(value.get("strategy") or "targeted_span_search"),
            aliases=_text_list(value.get("aliases") or value.get("search_terms") or []),
            sections=_text_list(value.get("sections") or []),
            positive_terms=_text_list(value.get("positive_terms") or []),
            negative_terms=_text_list(value.get("negative_terms") or []),
            ambiguous_terms=_text_list(value.get("ambiguous_terms") or []),
            required_inputs=_text_list(value.get("required_inputs") or []),
            polarity_required=bool(value.get("polarity_required", True)),
            value_required=bool(value.get("value_required", False)),
            priority=_int(value.get("priority"), 0),
            importance=str(value.get("importance") or "medium"),
            expected_effect=str(value.get("expected_effect") or ""),
            recommended_exam=str(value.get("recommended_exam") or ""),
            case_version=_int(value.get("case_version"), 0),
            confidence=_float(value.get("confidence"), 0.0),
            entity_id=str(value.get("entity_id") or ""),
            invalid_reason=str(value.get("invalid_reason") or ""),
        )


class EvidenceQueryPlanner:
    """Expand canonical hypotheses into executable deterministic query tasks."""

    def __init__(
        self,
        registry: Optional[EvidenceDefinitionRegistry] = None,
        *,
        max_candidates: int = 3,
        max_claims_per_candidate: int = 5,
    ):
        self.registry = registry or EvidenceDefinitionRegistry()
        self.max_candidates = max(1, int(max_candidates or 3))
        self.max_claims_per_candidate = max(1, int(max_claims_per_candidate or 5))

    def plan_all(self, hypotheses: Sequence[Any]) -> List[EvidenceQueryTask]:
        parsed = [
            item
            for item in (EvidenceHypothesis.from_any(value) for value in hypotheses or [])
            if item is not None
        ]
        parsed.sort(key=self._hypothesis_priority, reverse=True)
        limited = self._limit_by_candidate(parsed)
        tasks: List[EvidenceQueryTask] = []
        seen: set[tuple[str, str, str]] = set()
        for index, hypothesis in enumerate(limited, start=1):
            task = self.plan(hypothesis, index=index)
            key = (task.candidate, task.target_evidence_id, task.claim_type)
            if key in seen:
                continue
            seen.add(key)
            tasks.append(task)
        tasks.sort(key=lambda item: item.priority, reverse=True)
        return tasks

    def plan(self, hypothesis: EvidenceHypothesis, *, index: int = 1) -> EvidenceQueryTask:
        definition = self.registry.get(hypothesis.target_evidence_id)
        canonical_target = self.registry.normalize_evidence_id(hypothesis.target_evidence_id)
        if definition is None and hypothesis.claim_type not in {DERIVED_PATTERN, COUNTEREVIDENCE}:
            return EvidenceQueryTask(
                query_task_id=f"QT-{index:03d}",
                hypothesis_id=hypothesis.hypothesis_id,
                candidate=hypothesis.candidate,
                target_evidence_id=canonical_target or hypothesis.target_evidence_id,
                claim_type=hypothesis.claim_type,
                strategy="invalid_registry_target",
                priority=0,
                importance=hypothesis.importance,
                expected_effect=hypothesis.expected_effect,
                recommended_exam=hypothesis.recommended_exam,
                case_version=hypothesis.case_version,
                confidence=hypothesis.confidence,
                entity_id=hypothesis.entity_id,
                invalid_reason="target evidence is not defined in EvidenceDefinitionRegistry",
            )
        definition = definition or self.registry.require(canonical_target)
        strategy = "targeted_span_search"
        if hypothesis.claim_type == DERIVED_PATTERN:
            strategy = "derived_pattern_check"
        elif hypothesis.claim_type == COUNTEREVIDENCE:
            strategy = "counterevidence_search"
        elif hypothesis.claim_type == MISSING_CONFIRMATION:
            strategy = "confirmation_gap_search"
        return EvidenceQueryTask(
            query_task_id=f"QT-{index:03d}",
            hypothesis_id=hypothesis.hypothesis_id,
            candidate=hypothesis.candidate,
            target_evidence_id=definition.evidence_id,
            claim_type=hypothesis.claim_type,
            strategy=strategy,
            aliases=self.registry.aliases_for(definition.evidence_id),
            sections=list(definition.preferred_sections),
            positive_terms=list(definition.positive_terms),
            negative_terms=list(definition.negative_terms),
            ambiguous_terms=list(definition.ambiguous_terms),
            required_inputs=list(hypothesis.required_inputs),
            polarity_required=True,
            value_required=bool(definition.requires_value),
            priority=self._hypothesis_priority(hypothesis),
            importance=hypothesis.importance,
            expected_effect=hypothesis.expected_effect,
            recommended_exam=hypothesis.recommended_exam or definition.followup_exam,
            case_version=hypothesis.case_version,
            confidence=hypothesis.confidence,
            entity_id=hypothesis.entity_id,
        )

    def _limit_by_candidate(self, hypotheses: Sequence[EvidenceHypothesis]) -> List[EvidenceHypothesis]:
        result: List[EvidenceHypothesis] = []
        candidate_order: List[str] = []
        counts: Dict[str, int] = {}
        for hypothesis in hypotheses or []:
            candidate = hypothesis.candidate or "_global"
            if candidate not in candidate_order:
                if len(candidate_order) >= self.max_candidates:
                    continue
                candidate_order.append(candidate)
            if counts.get(candidate, 0) >= self.max_claims_per_candidate:
                continue
            counts[candidate] = counts.get(candidate, 0) + 1
            result.append(hypothesis)
        return result

    @staticmethod
    def _hypothesis_priority(hypothesis: EvidenceHypothesis) -> int:
        importance = {
            "critical": 60,
            "high": 45,
            "medium": 25,
            "low": 10,
        }.get(str(hypothesis.importance or "").lower(), 20)
        effect = str(hypothesis.expected_effect or "").lower()
        if "eligibility_blocker" in effect:
            importance += 35
        elif "eligibility" in effect:
            importance += 30
        elif "judge_pool" in effect:
            importance += 20
        if hypothesis.claim_type == COUNTEREVIDENCE:
            importance += 15
        if hypothesis.claim_type == MISSING_CONFIRMATION:
            importance += 10
        return min(100, importance)


def _text_list(values: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(str(item).strip() for item in values or [] if str(item).strip()))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
