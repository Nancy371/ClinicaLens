"""Deterministic targeted verification for evidence hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .clinical_evidence import EvidenceBundle, Observation
from .evidence_query_planner import EvidenceQueryTask
from .evidence_registry import EvidenceDefinitionRegistry


VERIFIED_POSITIVE = "verified_positive"
VERIFIED_NEGATIVE = "verified_negative"
UNSUPPORTED = "unsupported"
UNRESOLVED = "unresolved"
INVALID_CLAIM = "invalid_claim"
DERIVED = "derived"

_ADMITTED_STATUSES = {VERIFIED_POSITIVE, VERIFIED_NEGATIVE, DERIVED}


@dataclass
class VerificationResult:
    hypothesis_id: str
    query_task_id: str
    target_evidence_id: str
    candidate: str = ""
    claim_type: str = "observed_finding"
    verification_status: str = UNSUPPORTED
    polarity: str = "positive"
    source_section: str = ""
    source_field: str = ""
    source_span: str = ""
    value: Optional[float] = None
    unit: str = ""
    reason: str = ""
    recommended_exam: str = ""
    importance: str = "medium"
    expected_effect: str = ""
    required_inputs: List[str] = field(default_factory=list)
    derived_from: List[str] = field(default_factory=list)
    case_version: int = 0
    confidence: float = 0.0
    entity_id: str = ""

    @property
    def status(self) -> str:
        return _legacy_status(self.verification_status)

    @property
    def evidence_id(self) -> str:
        return self.target_evidence_id if self.verification_status in _ADMITTED_STATUSES else ""

    @property
    def target_evidence(self) -> str:
        return self.target_evidence_id

    @property
    def diagnosis_hypothesis(self) -> str:
        return self.candidate

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["claim_id"] = self.hypothesis_id
        payload["evidence_id"] = self.evidence_id
        payload["target_evidence"] = self.target_evidence_id
        payload["diagnosis_hypothesis"] = self.candidate
        payload["status"] = self.status
        payload["source"] = {
            "section": self.source_section,
            "field": self.source_field,
            "text": self.source_span,
        }
        return payload

    @classmethod
    def from_any(cls, value: Any) -> Optional["VerificationResult"]:
        if isinstance(value, VerificationResult):
            return value
        if not isinstance(value, dict):
            return None
        target = str(
            value.get("target_evidence_id")
            or value.get("target_evidence")
            or value.get("evidence_id")
            or ""
        ).strip()
        hypothesis_id = str(
            value.get("hypothesis_id")
            or value.get("claim_id")
            or value.get("id")
            or ""
        ).strip()
        if not target or not hypothesis_id:
            return None
        return cls(
            hypothesis_id=hypothesis_id,
            query_task_id=str(value.get("query_task_id") or ""),
            target_evidence_id=target,
            candidate=str(value.get("candidate") or value.get("diagnosis_hypothesis") or ""),
            claim_type=str(value.get("claim_type") or "observed_finding"),
            verification_status=_normalize_status(
                value.get("verification_status") or value.get("status")
            ),
            polarity=str(value.get("polarity") or "positive"),
            source_section=str(value.get("source_section") or ""),
            source_field=str(value.get("source_field") or ""),
            source_span=str(value.get("source_span") or ""),
            value=_optional_float(value.get("value")),
            unit=str(value.get("unit") or ""),
            reason=str(value.get("reason") or ""),
            recommended_exam=str(value.get("recommended_exam") or ""),
            importance=str(value.get("importance") or "medium"),
            expected_effect=str(value.get("expected_effect") or ""),
            required_inputs=_text_list(value.get("required_inputs") or []),
            derived_from=_text_list(value.get("derived_from") or []),
            case_version=_int(value.get("case_version"), 0),
            confidence=_float(value.get("confidence"), 0.0),
            entity_id=str(value.get("entity_id") or ""),
        )


class DeterministicEvidenceVerifier:
    """Verify query tasks against structured evidence and source spans.

    It never promotes LLM text directly. A result can enter observed evidence
    only with canonical evidence_id, polarity, source section/field/span, case
    version, and deterministic verification method.
    """

    def __init__(self, registry: Optional[EvidenceDefinitionRegistry] = None):
        self.registry = registry or EvidenceDefinitionRegistry()

    def verify_all(
        self,
        query_tasks: Sequence[Any],
        evidence: EvidenceBundle,
    ) -> List[VerificationResult]:
        results: List[VerificationResult] = []
        verified_positive = self._verified_positive_finding_set(evidence)
        for raw_task in query_tasks or []:
            task = EvidenceQueryTask.from_any(raw_task)
            if task is None:
                continue
            result = self.verify(task, evidence, verified_positive)
            if result.verification_status in {VERIFIED_POSITIVE, DERIVED}:
                verified_positive.add(result.target_evidence_id)
            results.append(result)
        return results

    def verify(
        self,
        task: EvidenceQueryTask,
        evidence: EvidenceBundle,
        verified_positive: Optional[set[str]] = None,
    ) -> VerificationResult:
        verified_positive = set(verified_positive or self._verified_positive_finding_set(evidence))
        if task.invalid_reason:
            return self._base_result(task, INVALID_CLAIM, reason=task.invalid_reason)
        if task.strategy == "derived_pattern_check":
            missing = [
                item for item in task.required_inputs
                if item not in verified_positive and not self._alias_present(item, verified_positive)
            ]
            if missing:
                return self._base_result(
                    task,
                    UNSUPPORTED,
                    reason="missing verified inputs: " + ", ".join(missing),
                )
            return self._base_result(
                task,
                DERIVED,
                source_section="pattern_compiler",
                source_field="derived_pattern_check",
                source_span="derived from " + ", ".join(task.required_inputs),
                derived_from=list(task.required_inputs),
                reason="all required inputs verified",
                confidence=max(task.confidence, 0.86),
            )

        direct = self._direct_observation_match(task, evidence)
        if direct is not None:
            return direct

        span = self._span_match(task, evidence)
        if span is not None:
            return span

        return self._base_result(task, UNSUPPORTED, reason="source span not found")

    def observations_from_results(
        self,
        results: Sequence[Any],
        evidence: EvidenceBundle,
    ) -> List[Observation]:
        existing = self._finding_set(evidence)
        observations: List[Observation] = []
        for value in results or []:
            result = VerificationResult.from_any(value)
            if result is None:
                continue
            if result.verification_status not in {VERIFIED_POSITIVE, VERIFIED_NEGATIVE}:
                continue
            if result.target_evidence_id in existing:
                continue
            if not (result.source_span and (result.source_section or result.source_field)):
                continue
            definition = self.registry.require(result.target_evidence_id)
            observations.append(
                Observation(
                    finding=result.target_evidence_id,
                    source="targeted_evidence_verifier",
                    value=result.value,
                    unit=result.unit,
                    polarity="negative"
                    if result.verification_status == VERIFIED_NEGATIVE
                    else "positive",
                    confidence=max(0.7, min(0.98, result.confidence or 0.78)),
                    raw_text=result.source_span,
                    source_text=result.source_span,
                    field_path=result.source_field or result.source_section,
                    evidence_level=definition.evidence_level,
                    information_value=definition.information_value,
                )
            )
            existing.add(result.target_evidence_id)
        return observations

    def _direct_observation_match(
        self,
        task: EvidenceQueryTask,
        evidence: EvidenceBundle,
    ) -> Optional[VerificationResult]:
        target_ids = {task.target_evidence_id}
        target_ids.update(self._finding_aliases(task.target_evidence_id))
        reasoning_only: Optional[Observation] = None
        for observation in getattr(evidence, "observations", []) or []:
            finding = str(getattr(observation, "finding", "") or "")
            if finding not in target_ids:
                continue
            if str(getattr(observation, "source", "") or "") == "reasoning_inference":
                reasoning_only = observation
                continue
            source_span = _span_for_observation(observation)
            if not source_span:
                return self._base_result(
                    task,
                    UNRESOLVED,
                    source_section=str(getattr(observation, "source", "") or ""),
                    source_field=str(getattr(observation, "field_path", "") or ""),
                    reason="verified evidence requires source span",
                )
            polarity = str(getattr(observation, "polarity", "positive") or "positive")
            status = VERIFIED_NEGATIVE if polarity == "negative" else VERIFIED_POSITIVE
            return self._base_result(
                task,
                status,
                polarity=polarity,
                source_section=str(getattr(observation, "source", "") or ""),
                source_field=str(getattr(observation, "field_path", "") or ""),
                source_span=source_span,
                value=getattr(observation, "value", None),
                unit=str(getattr(observation, "unit", "") or ""),
                confidence=max(task.confidence, _float(getattr(observation, "confidence", 0.0), 0.0)),
                reason="canonical structured evidence found",
            )
        if reasoning_only is not None:
            return self._base_result(
                task,
                UNRESOLVED,
                reason="reasoning_inference cannot verify evidence claim",
            )
        return None

    def _span_match(
        self,
        task: EvidenceQueryTask,
        evidence: EvidenceBundle,
    ) -> Optional[VerificationResult]:
        aliases = _text_list(task.aliases)
        if not aliases:
            aliases = self.registry.aliases_for(task.target_evidence_id)
        for observation in getattr(evidence, "observations", []) or []:
            if str(getattr(observation, "source", "") or "") == "reasoning_inference":
                continue
            section = str(getattr(observation, "source", "") or "")
            field = str(getattr(observation, "field_path", "") or "")
            span = _span_for_observation(observation)
            haystack = " ".join([section, field, span])
            compact_haystack = _compact(haystack)
            if not compact_haystack:
                continue
            if any(_compact(term) in compact_haystack for term in task.ambiguous_terms):
                return self._base_result(
                    task,
                    UNRESOLVED,
                    source_section=section,
                    source_field=field,
                    source_span=span,
                    reason="source text contains an ambiguous evidence phrase",
                )
            alias_hit = any(_compact(alias) in compact_haystack for alias in aliases)
            if not alias_hit:
                continue
            negative = _has_term(task.negative_terms, haystack)
            positive = _has_term(task.positive_terms, haystack)
            if negative and not positive:
                return self._base_result(
                    task,
                    VERIFIED_NEGATIVE,
                    polarity="negative",
                    source_section=section,
                    source_field=field,
                    source_span=span,
                    reason="negative source semantics verified",
                    confidence=max(task.confidence, 0.76),
                )
            if positive or not task.polarity_required:
                return self._base_result(
                    task,
                    VERIFIED_POSITIVE,
                    polarity="positive",
                    source_section=section,
                    source_field=field,
                    source_span=span,
                    reason="positive source semantics verified",
                    confidence=max(task.confidence, 0.76),
                )
            return self._base_result(
                task,
                UNRESOLVED,
                source_section=section,
                source_field=field,
                source_span=span,
                reason="alias found but polarity is unclear",
            )
        return None

    def _base_result(
        self,
        task: EvidenceQueryTask,
        verification_status: str,
        *,
        polarity: str = "positive",
        source_section: str = "",
        source_field: str = "",
        source_span: str = "",
        value: Optional[float] = None,
        unit: str = "",
        reason: str = "",
        derived_from: Optional[Sequence[str]] = None,
        confidence: float = 0.0,
    ) -> VerificationResult:
        return VerificationResult(
            hypothesis_id=task.hypothesis_id,
            query_task_id=task.query_task_id,
            target_evidence_id=task.target_evidence_id,
            candidate=task.candidate,
            claim_type=task.claim_type,
            verification_status=verification_status,
            polarity=polarity,
            source_section=source_section,
            source_field=source_field,
            source_span=source_span,
            value=value,
            unit=unit,
            reason=reason,
            recommended_exam=task.recommended_exam,
            importance=task.importance,
            expected_effect=task.expected_effect,
            required_inputs=list(task.required_inputs),
            derived_from=list(derived_from or []),
            case_version=task.case_version,
            confidence=confidence or task.confidence,
            entity_id=task.entity_id,
        )

    def _verified_positive_finding_set(self, evidence: EvidenceBundle) -> set[str]:
        findings = set()
        for item in getattr(evidence, "observations", []) or []:
            if str(getattr(item, "source", "") or "") == "reasoning_inference":
                continue
            if str(getattr(item, "polarity", "positive") or "positive") != "positive":
                continue
            finding = str(getattr(item, "finding", "") or "")
            if finding:
                findings.add(finding)
                findings.update(self._finding_aliases(finding))
        return findings

    def _finding_set(self, evidence: EvidenceBundle) -> set[str]:
        findings = set()
        for item in getattr(evidence, "observations", []) or []:
            if str(getattr(item, "source", "") or "") == "reasoning_inference":
                continue
            finding = str(getattr(item, "finding", "") or "")
            if finding:
                findings.add(finding)
        return findings

    def _finding_aliases(self, finding: str) -> set[str]:
        canonical = self.registry.normalize_evidence_id(finding)
        aliases = {
            "hemoglobin_low": {"anemia", "low_hemoglobin"},
            "anemia": {"hemoglobin_low", "low_hemoglobin"},
            "thrombocytopenia": {"platelet_low"},
            "platelet_low": {"thrombocytopenia"},
            "leukocytosis": {"white_blood_cell_abnormal"},
            "leukopenia": {"white_blood_cell_abnormal"},
            "white_blood_cell_abnormal": {"leukocytosis", "leukopenia", "wbc_abnormal"},
            "pulmonary_vascular_shunt": {"right_to_left_shunt"},
            "right_to_left_shunt": {"pulmonary_vascular_shunt"},
            "pulmonary_avm_imaging": {"pulmonary_cta_positive", "enhanced_ct_vascular_malformation"},
        }
        return set(aliases.get(canonical, set()))

    def _alias_present(self, finding: str, findings: set[str]) -> bool:
        candidates = {finding, self.registry.normalize_evidence_id(finding)}
        candidates.update(self._finding_aliases(finding))
        return bool(candidates & findings)


def _legacy_status(status: str) -> str:
    normalized = _normalize_status(status)
    if normalized in {VERIFIED_POSITIVE, VERIFIED_NEGATIVE}:
        return "Verified"
    if normalized == DERIVED:
        return "Derived"
    if normalized == INVALID_CLAIM:
        return "Invalid"
    return "Unresolved"


def _normalize_status(value: Any) -> str:
    text = str(value or "").strip()
    lowered = text.lower()
    if lowered in {
        VERIFIED_POSITIVE,
        VERIFIED_NEGATIVE,
        UNSUPPORTED,
        UNRESOLVED,
        INVALID_CLAIM,
        DERIVED,
    }:
        return lowered
    if text == "Verified":
        return VERIFIED_POSITIVE
    if text == "Derived":
        return DERIVED
    if text == "Invalid":
        return INVALID_CLAIM
    return UNRESOLVED if text else UNSUPPORTED


def _span_for_observation(observation: Observation) -> str:
    return str(
        getattr(observation, "source_text", "")
        or getattr(observation, "raw_text", "")
        or ""
    ).strip()


def _has_term(terms: Sequence[Any], text: str) -> bool:
    compact_text = _compact(text)
    return any(_compact(term) in compact_text for term in terms or [] if _compact(term))


def _compact(value: Any) -> str:
    return "".join(str(value or "").lower().split())


def _text_list(values: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(str(item).strip() for item in values or [] if str(item).strip()))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
