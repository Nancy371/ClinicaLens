"""Controlled LLM clinical-pattern hypothesis recall.

This module keeps LLM-generated pattern proposals outside the evidence layer.
Verified hypotheses can only emit recall signals; they do not create
observations, derived findings, eligibility anchors, or active gaps.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .clinical_evidence import EvidenceBundle, Observation


RECALL_NONE = "none"
RECALL_QUERY_EXPANSION = "query_expansion"
RECALL_BOOST = "recall_boost"
RECALL_PROTECTED = "protected_recall"

STATUS_VERIFIED = "verified"
STATUS_REJECTED = "rejected"
STATUS_UNRESOLVED = "unresolved"

_DEFAULT_CONFIG = {
    "enabled": True,
    "max_hypotheses": 3,
    "use_thinking_snapshots": True,
    "max_recall_candidates": 3,
    "max_protected_candidates": 2,
    "active_min_strength": 0.55,
    "protection_min_strength": 0.75,
    "min_entity_link_confidence": 0.85,
    "max_expansion_rounds": 1,
    "max_proposals_per_source": 3,
    "max_active_schemas_per_snapshot": 5,
    "max_bindings_per_schema": 1,
    "max_family_expansion_entities": 5,
    "allow_direct_gap_creation": False,
    "allow_judge_score_contribution": False,
    "allow_eligibility_evidence_contribution": False,
}

_FORBIDDEN_LINEAGE_TOKENS = (
    "reasoning",
    "candidate",
    "hypothesis",
    "judge",
    "ordered_exam",
    "exam_request",
    "diagnosis_support",
    "llm_summary",
)

_RELATION_SLOTS = {
    "exposure",
    "temporal_relation",
    "organ_manifestation",
    "imaging_or_objective_finding",
    "exclusion_or_contradiction",
    "structure_or_credible_sign",
    "function_impairment",
    "regurgitation_specific",
    "support",
    "context",
}

_ROLE_ALIASES = {
    "causal_exposure": "exposure",
    "exposure": "exposure",
    "temporal_relation": "temporal_relation",
    "structural_abnormality": "structure_or_credible_sign",
    "structure_or_credible_sign": "structure_or_credible_sign",
    "functional_consequence": "function_impairment",
    "function_impairment": "function_impairment",
    "objective_organ_injury": "imaging_or_objective_finding",
    "imaging_or_objective_finding": "imaging_or_objective_finding",
    "organ_manifestation": "organ_manifestation",
    "manifestation": "organ_manifestation",
    "regurgitation_specific": "regurgitation_specific",
}

_SCHEMA_REQUIRED_SLOTS = {
    "exposure_temporal_organ_injury": [
        "exposure",
        "organ_manifestation",
        "imaging_or_objective_finding",
    ],
    "structural_function_abnormality": [
        "structure_or_credible_sign",
        "function_impairment",
    ],
    "left_sided_valvular_disease": [
        "structure_or_credible_sign",
        "function_impairment",
    ],
    "left_sided_valvular_regurgitation": [
        "structure_or_credible_sign",
        "function_impairment",
        "regurgitation_specific",
    ],
}

_SCHEMA_CRITICAL_CONSTRAINTS = {
    "exposure_temporal_organ_injury": [
        "temporal_after",
        "anatomical_consistency",
    ],
    "structural_function_abnormality": [
        "structural_function_consistency",
    ],
    "left_sided_valvular_disease": [
        "structural_function_consistency",
    ],
    "left_sided_valvular_regurgitation": [
        "structural_function_consistency",
    ],
}

_SOURCE_ANCHOR_CONCEPTS = {
    "thoracic_radiotherapy": (
        "胸部放疗",
        "胸部放射治疗",
        "肺部放疗",
        "thoracic radiotherapy",
        "chest radiotherapy",
    ),
    "post_radiotherapy_time_window": (
        "放疗后",
        "放疗结束后",
        "after radiotherapy",
        "post radiotherapy",
    ),
    "ground_glass_opacity": (
        "磨玻璃",
        "ground glass",
        "ground-glass",
    ),
    "pulmonary_inflammatory_change": (
        "肺部炎性",
        "肺部炎症",
        "炎性改变",
    ),
    "pink_frothy_sputum": (
        "粉红色泡沫痰",
        "粉红色泡沫样痰",
    ),
    "mitral_valve_prolapse": (
        "二尖瓣脱垂",
        "mitral valve prolapse",
    ),
}

_RADIATION_PATTERN_HINTS = (
    "radiation",
    "radiotherapy",
    "post_radiotherapy",
    "radiation_pneumonitis",
    "radiation_lung",
)

_FAMILY_ENTITY_HINTS = {
    "radiation_related_lung_injury": ["D100058"],
    "radiation_induced_lung_injury": ["D100058"],
    "left_sided_valvular_disease": ["D100012"],
    "left_sided_valvular_regurgitation": ["D100012"],
    "valvular_left_heart": ["D100012"],
    "valvular_regurgitation": ["D100012"],
    "pulmonary_vascular_shunt_family": ["D100055"],
    "pulmonary_vascular_shunt": ["D100055"],
    "pulmonary_vascular_abnormality": ["D100055"],
}

_CONTROLLED_RELATION_TYPES = {
    "temporal_after",
    "temporal_before",
    "anatomical_consistency",
    "causal_exposure",
    "cross_system_cooccurrence",
    "vascular_shunt_pattern",
    "structural_function_abnormality",
}

_TRUST_TIER_STRUCTURED = "structured_relation"
_TRUST_TIER_EVIDENCE_BOUND = "evidence_bound_hint"
_TRUST_TIER_QUERY_ONLY = "query_only_hint"


@dataclass
class EvidenceBinding:
    evidence_id: str
    role: str = "support"
    expected_polarity: str = "positive"
    relation_slot: str = "support"

    @classmethod
    def from_any(cls, value: Any) -> "EvidenceBinding":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                evidence_id=str(value.get("evidence_id") or value.get("id") or "").strip(),
                role=str(value.get("role") or "support").strip() or "support",
                expected_polarity=str(value.get("expected_polarity") or "positive").strip() or "positive",
                relation_slot=str(value.get("relation_slot") or value.get("slot") or "support").strip() or "support",
            )
        return cls(evidence_id=str(value or "").strip())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceReference:
    raw_ref: str
    resolved_observation_ref: str = ""
    canonical_concept: str = ""
    binding_method: str = "unresolved"
    binding_confidence: float = 0.0
    binding_status: str = "unresolved"
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    candidate_matches: List[Dict[str, Any]] = field(default_factory=list)
    failure_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRoleBinding:
    relation_schema_id: str
    slot_id: str
    observation_ref: str
    canonical_concept: str
    binding_rule: str
    binding_confidence: float
    polarity: str
    evidence_level: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationConstraintResult:
    constraint_type: str
    from_observation_ref: str = ""
    to_observation_ref: str = ""
    status: str = "unresolved"
    reason: str = ""
    interval: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SuggestedDisease:
    name: str
    canonical_id: str = ""
    hypothesis_confidence: float = 0.0

    @classmethod
    def from_any(cls, value: Any) -> "SuggestedDisease":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(
                name=str(value.get("name") or value.get("canonical_name") or value.get("disease") or "").strip(),
                canonical_id=str(value.get("canonical_id") or value.get("entity_id") or "").strip(),
                hypothesis_confidence=_safe_float(value.get("hypothesis_confidence", value.get("confidence", 0.0))),
            )
        return cls(name=str(value or "").strip())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalPatternHypothesis:
    pattern_hypothesis_id: str
    case_id: str = ""
    case_version: int = 0
    evidence_snapshot_id: str = ""
    pattern_name: str = ""
    pattern_type: str = ""
    suggested_family: str = ""
    relation_schema_id: str = ""
    evidence_bindings: List[EvidenceBinding] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    suggested_diseases: List[SuggestedDisease] = field(default_factory=list)
    missing_evidence_requests: List[Dict[str, Any]] = field(default_factory=list)
    relation_activation_audit: Dict[str, Any] = field(default_factory=dict)
    model_confidence: float = 0.0
    generator_source: str = "diagnosis_llm_draft"
    proposal_trust_tier: str = _TRUST_TIER_QUERY_ONLY
    model_id: str = ""
    prompt_version: str = ""
    created_at: str = ""

    @classmethod
    def from_dict(
        cls,
        value: Dict[str, Any],
        *,
        index: int = 0,
        case_id: str = "",
        case_version: int = 0,
        evidence_snapshot_id: str = "",
    ) -> "ClinicalPatternHypothesis":
        raw_id = str(
            value.get("pattern_hypothesis_id")
            or value.get("hypothesis_id")
            or value.get("id")
            or ""
        ).strip()
        pattern_name = str(value.get("pattern_name") or value.get("name") or "").strip()
        if not raw_id:
            raw_id = _stable_id("PH", f"{case_id}|{evidence_snapshot_id}|{index}|{pattern_name}")
        return cls(
            pattern_hypothesis_id=raw_id,
            case_id=str(value.get("case_id") or case_id or "").strip(),
            case_version=int(value.get("case_version") or case_version or 0),
            evidence_snapshot_id=str(value.get("evidence_snapshot_id") or evidence_snapshot_id or "").strip(),
            pattern_name=pattern_name,
            pattern_type=str(value.get("pattern_type") or "").strip(),
            suggested_family=str(value.get("suggested_family") or value.get("family_id") or "").strip(),
            relation_schema_id=str(
                value.get("relation_schema_id")
                or value.get("pattern_schema_id")
                or value.get("pattern_type")
                or ""
            ).strip(),
            evidence_bindings=[
                binding
                for binding in (
                    EvidenceBinding.from_any(item)
                    for item in value.get("evidence_bindings")
                    or value.get("source_evidence_ids")
                    or []
                )
                if binding.evidence_id
            ],
            relations=[
                _normalize_relation(item)
                for item in value.get("relations") or []
                if isinstance(item, dict)
            ],
            suggested_diseases=[
                disease
                for disease in (
                    SuggestedDisease.from_any(item)
                    for item in value.get("suggested_diseases") or []
                )
                if disease.name or disease.canonical_id
            ],
            missing_evidence_requests=[
                dict(item) if isinstance(item, dict) else {"target_evidence": str(item)}
                for item in (
                    value.get("missing_evidence_requests")
                    or value.get("missing_evidence")
                    or []
                )
                if item
            ],
            relation_activation_audit=dict(value.get("relation_activation_audit") or {}),
            model_confidence=_safe_float(value.get("model_confidence", value.get("confidence", 0.0))),
            generator_source=str(value.get("generator_source") or "diagnosis_llm_draft"),
            proposal_trust_tier=str(
                value.get("proposal_trust_tier")
                or value.get("_proposal_trust_tier")
                or _TRUST_TIER_QUERY_ONLY
            ),
            model_id=str(value.get("model_id") or ""),
            prompt_version=str(value.get("prompt_version") or ""),
            created_at=str(value.get("created_at") or _utc_now()),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence_bindings"] = [item.to_dict() for item in self.evidence_bindings]
        data["suggested_diseases"] = [item.to_dict() for item in self.suggested_diseases]
        return data


@dataclass
class ThinkingSnapshot:
    snapshot_id: str
    case_id: str = ""
    patient_id: str = ""
    phase: str = ""
    round_id: str = "round_00"
    case_version: int = 0
    evidence_snapshot_id: str = ""
    schema_version: str = "thinking_snapshot_v1"
    differential_diagnoses: List[Dict[str, Any]] = field(default_factory=list)
    supporting_evidence_refs: List[str] = field(default_factory=list)
    key_unknowns: List[Any] = field(default_factory=list)
    excluded_diagnoses: List[Any] = field(default_factory=list)
    body_system_hints: List[str] = field(default_factory=list)
    family_hints: List[str] = field(default_factory=list)
    mechanism_hints: List[str] = field(default_factory=list)
    action_plan: str = ""
    planned_exams: List[Any] = field(default_factory=list)
    clinical_pattern_proposals: List[Dict[str, Any]] = field(default_factory=list)
    raw_output_hash: str = ""
    created_at: str = ""
    superseded: bool = False

    @classmethod
    def from_thinking(
        cls,
        thinking: Dict[str, Any],
        *,
        case_id: str = "",
        patient_id: str = "",
        phase: str = "",
        round_id: str = "round_00",
        case_version: int = 0,
        evidence_snapshot_id: str = "",
    ) -> "ThinkingSnapshot":
        if not isinstance(thinking, dict):
            thinking = {}
        raw_hash = _stable_hash(thinking)
        differentials = [
            dict(item)
            for item in thinking.get("differential_diagnosis")
            or thinking.get("candidate_diseases")
            or []
            if isinstance(item, dict)
        ]
        supporting_refs: List[str] = []
        for item in differentials:
            for value in _as_list(
                item.get("supporting_evidence_refs")
                or item.get("evidence_refs")
                or item.get("source_evidence_ids")
            ):
                text = str(value or "").strip()
                if text:
                    supporting_refs.append(text)
        for value in _as_list(thinking.get("supporting_evidence_refs")):
            text = str(value or "").strip()
            if text:
                supporting_refs.append(text)
        proposals = [
            dict(item)
            for item in thinking.get("clinical_pattern_proposals")
            or thinking.get("clinical_pattern_hypotheses")
            or []
            if isinstance(item, dict)
        ]
        snapshot_id = _stable_id("TS", f"{case_id}|{patient_id}|{phase}|{round_id}|{evidence_snapshot_id}|{raw_hash}")
        return cls(
            snapshot_id=snapshot_id,
            case_id=str(case_id or "").strip(),
            patient_id=str(patient_id or case_id or "").strip(),
            phase=str(phase or "").strip(),
            round_id=str(round_id or "round_00").strip() or "round_00",
            case_version=int(case_version or 0),
            evidence_snapshot_id=str(evidence_snapshot_id or "").strip(),
            differential_diagnoses=differentials,
            supporting_evidence_refs=list(dict.fromkeys(supporting_refs)),
            key_unknowns=list(thinking.get("key_unknowns") or []),
            excluded_diagnoses=list(thinking.get("excluded_diagnoses") or []),
            body_system_hints=[str(item).strip() for item in _as_list(thinking.get("body_system_hints")) if str(item).strip()],
            family_hints=[str(item).strip() for item in _as_list(thinking.get("family_hints")) if str(item).strip()],
            mechanism_hints=[str(item).strip() for item in _as_list(thinking.get("mechanism_hints")) if str(item).strip()],
            action_plan=str(thinking.get("next_action") or thinking.get("action_plan") or "").strip(),
            planned_exams=list(thinking.get("planned_exams") or []),
            clinical_pattern_proposals=proposals,
            raw_output_hash=raw_hash,
            created_at=_utc_now(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatternEntityLink:
    entity_id: str
    canonical_name: str
    submission_name: str
    raw_name: str
    link_confidence: float
    resolution_status: str
    submittable: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatternVerificationResult:
    pattern_hypothesis_id: str
    verification_status: str
    valid_source_evidence_ids: List[str] = field(default_factory=list)
    invalid_source_evidence_ids: List[str] = field(default_factory=list)
    support_strength: float = 0.0
    contradiction_strength: float = 0.0
    critical_anchor_completeness: bool = False
    relation_completeness: bool = False
    entity_links: List[PatternEntityLink] = field(default_factory=list)
    hard_gate_results: Dict[str, Any] = field(default_factory=dict)
    net_pattern_strength: float = 0.0
    rejection_reasons: List[str] = field(default_factory=list)
    verified_at: str = ""
    verifier_version: str = "pattern_hypothesis_verifier_v1"
    source_groups: Dict[str, List[str]] = field(default_factory=dict)
    missing_evidence_requests: List[Dict[str, Any]] = field(default_factory=list)
    hypothesis: Dict[str, Any] = field(default_factory=dict)
    ref_resolution_audit: List[Dict[str, Any]] = field(default_factory=list)
    slot_binding_audit: Dict[str, Any] = field(default_factory=dict)
    relation_activation_audit: Dict[str, Any] = field(default_factory=dict)
    admission_level: str = "family_expansion"
    verified_specificity: str = "family"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["entity_links"] = [item.to_dict() for item in self.entity_links]
        return data


@dataclass
class PatternRecallSignal:
    pattern_hypothesis_id: str
    entity_id: str
    entity_link_confidence: float
    recall_mode: str
    recall_strength: float
    protected_pool_slot: bool
    source_evidence_ids: List[str] = field(default_factory=list)
    missing_evidence_requests: List[Dict[str, Any]] = field(default_factory=list)
    canonical_name: str = ""
    submission_name: str = ""
    raw_name: str = ""
    judge_evidence_weight: float = 0.0
    eligibility_evidence_weight: float = 0.0
    gap_suggestion_only: bool = True
    active_gap_write_permission: str = "none"
    admission_level: str = "family_expansion"
    verified_specificity: str = "family"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceRefResolver:
    """Resolve proposal refs to existing observations without creating evidence."""

    def __init__(self, observations: Sequence[Observation], ontology: Optional[Dict[str, Any]] = None):
        self.observations = list(observations or [])
        self.ontology = dict(ontology or {})
        self.ref_index, self.ambiguous_refs = _build_observation_lookup(self.observations)
        self.obs_ref_index = {_observation_ref(item): item for item in self.observations}
        self.finding_index: Dict[str, List[Observation]] = {}
        for item in self.observations:
            self.finding_index.setdefault(_normalize_token(item.finding), []).append(item)
        self.alias_to_concept: Dict[str, str] = {}
        for concept, spec in self.ontology.items():
            self.alias_to_concept[_normalize_token(concept)] = concept
            for alias in spec.get("aliases", []) or []:
                self.alias_to_concept[_normalize_token(alias)] = concept

    def resolve(self, raw_ref: Any, *, expected_slot: str = "") -> EvidenceReference:
        text = str(raw_ref or "").strip()
        result = EvidenceReference(raw_ref=text)
        if not text:
            result.failure_reason = "empty_ref"
            return result
        exact = self._resolve_exact_observation_ref(text, result)
        if exact:
            return exact
        canonical = self._resolve_canonical_exact(text, result)
        if canonical:
            return canonical
        alias = self._resolve_alias_or_parent(text, result)
        if alias:
            return alias
        source_anchor = self._resolve_source_anchor(text, result)
        if source_anchor:
            return source_anchor
        result.binding_status = "unresolved"
        if not result.failure_reason:
            result.failure_reason = "ontology_mapping_missing"
        return result

    def _resolve_exact_observation_ref(self, raw_ref: str, result: EvidenceReference) -> Optional[EvidenceReference]:
        result.attempts.append({"method": "observation_ref_exact", "matched": False})
        obs = self.obs_ref_index.get(raw_ref) or self.ref_index.get(raw_ref)
        if not obs:
            return None
        return self._resolved(result, obs, "observation_ref_exact", 1.0)

    def _resolve_canonical_exact(self, raw_ref: str, result: EvidenceReference) -> Optional[EvidenceReference]:
        key = _normalize_token(raw_ref)
        matches = list(self.finding_index.get(key) or [])
        result.attempts.append({"method": "canonical_exact", "matched": bool(matches)})
        if not matches:
            return None
        return self._coerce_matches(result, matches, "canonical_exact", 0.98)

    def _resolve_alias_or_parent(self, raw_ref: str, result: EvidenceReference) -> Optional[EvidenceReference]:
        key = _normalize_token(raw_ref)
        concept = self.alias_to_concept.get(key)
        if not concept and key in self.ontology:
            concept = key
        result.attempts.append({"method": "controlled_alias", "matched": bool(concept and concept != raw_ref)})
        if not concept:
            return None
        exact = list(self.finding_index.get(_normalize_token(concept)) or [])
        if exact:
            return self._coerce_matches(result, exact, "controlled_alias", 0.94, canonical=concept)
        children = [
            str(item)
            for item in (self.ontology.get(concept, {}) or {}).get("children", []) or []
            if str(item)
        ]
        matches: List[Observation] = []
        child_set = {_normalize_token(item) for item in children}
        for item in self.observations:
            if _normalize_token(item.finding) in child_set:
                matches.append(item)
        result.attempts.append({"method": "ontology_parent", "matched": bool(matches), "parent": concept})
        if not matches:
            result.failure_reason = "ontology_child_observation_missing"
            return None
        return self._coerce_matches(result, matches, "ontology_parent", 0.88, canonical=concept)

    def _resolve_source_anchor(self, raw_ref: str, result: EvidenceReference) -> Optional[EvidenceReference]:
        concept = self.alias_to_concept.get(_normalize_token(raw_ref)) or raw_ref
        anchors = _SOURCE_ANCHOR_CONCEPTS.get(concept, ())
        result.attempts.append({"method": "source_provenance_anchor", "matched": False})
        if not anchors:
            return None
        matches: List[Observation] = []
        for item in self.observations:
            text = f"{item.raw_text} {item.source_text} {item.field_path}".lower()
            if any(str(anchor).lower() in text for anchor in anchors):
                matches.append(item)
        if not matches:
            result.failure_reason = "requires_evidence_normalization"
            return None
        result.attempts[-1]["matched"] = True
        return self._coerce_matches(result, matches, "source_provenance_anchor", 0.72, canonical=concept)

    def _coerce_matches(
        self,
        result: EvidenceReference,
        matches: Sequence[Observation],
        method: str,
        confidence: float,
        *,
        canonical: str = "",
    ) -> EvidenceReference:
        unique: Dict[str, Observation] = {}
        for item in matches:
            unique.setdefault(_evidence_group_id(item), item)
        result.candidate_matches = [_observation_summary(item) for item in unique.values()]
        if len(unique) != 1:
            result.binding_status = "ambiguous"
            result.binding_method = method
            result.binding_confidence = confidence
            result.failure_reason = "ambiguous_evidence_binding"
            return result
        return self._resolved(result, next(iter(unique.values())), method, confidence, canonical=canonical)

    def _resolved(
        self,
        result: EvidenceReference,
        obs: Observation,
        method: str,
        confidence: float,
        *,
        canonical: str = "",
    ) -> EvidenceReference:
        result.resolved_observation_ref = _observation_ref(obs)
        result.canonical_concept = canonical or obs.finding
        result.binding_method = method
        result.binding_confidence = confidence
        result.binding_status = "resolved"
        result.failure_reason = ""
        result.candidate_matches = [_observation_summary(obs)]
        return result


class EvidenceRelationBinder:
    """Bind existing observations to relation-schema roles."""

    def __init__(self, observations: Sequence[Observation], ontology: Optional[Dict[str, Any]] = None):
        self.observations = list(observations or [])
        self.ontology = dict(ontology or {})

    def bind(self, relation_schema_id: str) -> Dict[str, Any]:
        schema = _normalize_schema_id(relation_schema_id)
        required = list(_SCHEMA_REQUIRED_SLOTS.get(schema) or [])
        slot_candidates: Dict[str, List[EvidenceRoleBinding]] = {}
        for item in self.observations:
            if item.polarity != "positive" or item.shadowed_by or _lineage_rejection_reason(item):
                continue
            for slot, rule, confidence in _role_candidates_for_observation(schema, item, self.ontology):
                slot_candidates.setdefault(slot, []).append(
                    EvidenceRoleBinding(
                        relation_schema_id=schema,
                        slot_id=slot,
                        observation_ref=_observation_ref(item),
                        canonical_concept=item.finding,
                        binding_rule=rule,
                        binding_confidence=round(confidence, 4),
                        polarity=item.polarity or "positive",
                        evidence_level=item.evidence_level or "",
                        source=item.source or "",
                    )
                )
        bound_slots: Dict[str, EvidenceRoleBinding] = {}
        ambiguous_slots: List[str] = []
        for slot, candidates in slot_candidates.items():
            unique_groups: Dict[str, EvidenceRoleBinding] = {}
            for candidate in candidates:
                obs = _observation_by_ref(self.observations, candidate.observation_ref)
                if obs:
                    unique_groups.setdefault(_evidence_group_id(obs), candidate)
            ranked = sorted(
                unique_groups.values(),
                key=lambda item: _role_binding_rank(item, self.observations),
                reverse=True,
            )
            if ranked:
                bound_slots[slot] = ranked[0]
            if len(ranked) > 1 and slot in required:
                ambiguous_slots.append(slot)
        missing = [slot for slot in required if slot not in bound_slots]
        constraints = _evaluate_constraints(schema, bound_slots, self.observations)
        critical = list(_SCHEMA_CRITICAL_CONSTRAINTS.get(schema) or [])
        critical_failed = [
            item.constraint_type
            for item in constraints
            if item.constraint_type in critical and item.status != "satisfied"
        ]
        if missing:
            activation_status = "partial"
        elif critical_failed:
            activation_status = "partial"
        else:
            activation_status = "activated"
        score = _activation_score(required, bound_slots, constraints)
        audit = {
            "relation_schema_id": schema,
            "required_slots": required,
            "bound_slots": sorted(bound_slots),
            "missing_slots": missing,
            "ambiguous_slots": ambiguous_slots,
            "supporting_evidence": {
                slot: bound.to_dict()
                for slot, bound in bound_slots.items()
            },
            "critical_constraints": critical,
            "constraint_results": [item.to_dict() for item in constraints],
            "activation_status": activation_status,
            "activation_score": round(score, 4),
            "rejection_reasons": (
                [f"missing_slot:{slot}" for slot in missing]
                + [f"constraint_unmet:{name}" for name in critical_failed]
            ),
        }
        return {"bindings": bound_slots, "constraints": constraints, "audit": audit}


class PatternProposalAdapter:
    """Normalize Thinking and diagnosis-draft proposals before verification.

    The adapter only builds hypotheses. It deliberately avoids validating
    medical relationships or creating evidence/candidates/gaps.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        section = ((config or {}).get("diagnosis", {}) or {}).get("llm_pattern_hypothesis", {})
        merged = dict(_DEFAULT_CONFIG)
        merged.update(dict(section or {}))
        self.config = merged
        ref_dir = str((config or {}).get("ref_data_dir") or "data/ref_data")
        self.relation_registry = _load_relation_registry(ref_dir)
        self.evidence_ontology = _load_evidence_ontology(ref_dir)
        self.last_audit: Dict[str, Any] = {}

    def propose(
        self,
        thinking_snapshots: Optional[Sequence[Any]],
        diagnosis_draft: Any,
        evidence: EvidenceBundle,
        *,
        case_id: str = "",
        case_version: int = 0,
        evidence_snapshot_id: str = "",
    ) -> List[ClinicalPatternHypothesis]:
        if not self.config.get("enabled", True):
            self.last_audit = _empty_compiler_audit(
                enabled=False,
                evidence_snapshot_id=evidence_snapshot_id or evidence_snapshot_hash(evidence),
            )
            return []
        snapshot_id = evidence_snapshot_id or evidence_snapshot_hash(evidence)
        source_audit = _empty_source_audit()
        hypotheses: List[ClinicalPatternHypothesis] = []
        if self.config.get("use_thinking_snapshots", True):
            snapshot_count = 0
            for snapshot in thinking_snapshots or []:
                snapshot_obj = _coerce_thinking_snapshot(snapshot)
                if not snapshot_obj:
                    continue
                if case_id and snapshot_obj.case_id and snapshot_obj.case_id != case_id:
                    continue
                snapshot_count += 1
                structured, reasoning = self._from_thinking_snapshot(
                    snapshot_obj,
                    evidence,
                    case_id=case_id,
                    case_version=case_version,
                    evidence_snapshot_id=snapshot_id,
                )
                hypotheses.extend(structured)
                hypotheses.extend(reasoning)
                _record_source_audit(
                    source_audit,
                    "structured_thinking",
                    input_present=bool(snapshot_obj.clinical_pattern_proposals),
                    generated=len(structured),
                    skip_reason="" if structured else _thinking_skip_reason(snapshot_obj, "structured_thinking"),
                )
                _record_source_audit(
                    source_audit,
                    "reasoning_adapter",
                    input_present=bool(snapshot_obj.differential_diagnoses),
                    generated=len(reasoning),
                    skip_reason="" if reasoning else _thinking_skip_reason(snapshot_obj, "reasoning_adapter"),
                )
            if snapshot_count == 0:
                _record_source_audit(
                    source_audit,
                    "structured_thinking",
                    input_present=False,
                    generated=0,
                    skip_reason="no_thinking_snapshot",
                )
                _record_source_audit(
                    source_audit,
                    "reasoning_adapter",
                    input_present=False,
                    generated=0,
                    skip_reason="no_thinking_snapshot",
                )
        else:
            _record_source_audit(
                source_audit,
                "structured_thinking",
                input_present=False,
                generated=0,
                skip_reason="thinking_snapshots_disabled",
            )
            _record_source_audit(
                source_audit,
                "reasoning_adapter",
                input_present=False,
                generated=0,
                skip_reason="thinking_snapshots_disabled",
            )
        draft_hypotheses = self._from_diagnosis_draft(
            diagnosis_draft,
            case_id=case_id,
            case_version=case_version,
            evidence_snapshot_id=snapshot_id,
        )
        hypotheses.extend(draft_hypotheses)
        _record_source_audit(
            source_audit,
            "diagnosis_draft",
            input_present=bool(
                isinstance(diagnosis_draft, dict)
                and (
                    diagnosis_draft.get("clinical_pattern_hypotheses")
                    or diagnosis_draft.get("_clinical_pattern_hypotheses")
                )
            ),
            generated=len(draft_hypotheses),
            skip_reason="" if draft_hypotheses else "no_structured_pattern_hypotheses",
        )
        deterministic = self._from_deterministic_relations(
            evidence,
            case_id=case_id,
            case_version=case_version,
            evidence_snapshot_id=snapshot_id,
        )
        hypotheses.extend(deterministic)
        _record_source_audit(
            source_audit,
            "deterministic_relation",
            input_present=bool(evidence and evidence.observations),
            generated=len(deterministic),
            skip_reason="" if deterministic else "missing_required_slot",
        )
        deduped = self._dedupe(hypotheses)
        max_total = int(self.config.get("max_hypotheses", 3) or 3)
        result = deduped[:max_total]
        self.last_audit = {
            "compiler_enabled": True,
            "compiler_version": "pattern_proposal_compiler_v2",
            "evidence_snapshot_id": snapshot_id,
            "sources": source_audit,
            "proposal_count_before_dedup": len(hypotheses),
            "proposal_count_after_dedup": len(deduped),
            "proposal_count_after_budget": len(result),
            "deduplicated_count": max(0, len(hypotheses) - len(deduped)),
            "budget_truncated_count": max(0, len(deduped) - len(result)),
        }
        return result

    def _from_thinking_snapshot(
        self,
        snapshot: ThinkingSnapshot,
        evidence: EvidenceBundle,
        *,
        case_id: str,
        case_version: int,
        evidence_snapshot_id: str,
    ) -> Tuple[List[ClinicalPatternHypothesis], List[ClinicalPatternHypothesis]]:
        structured: List[ClinicalPatternHypothesis] = []
        for index, proposal in enumerate(snapshot.clinical_pattern_proposals or []):
            if not isinstance(proposal, dict):
                continue
            payload = dict(proposal)
            payload.setdefault("generator_source", "thinking_structured")
            payload.setdefault("created_at", snapshot.created_at or _utc_now())
            payload.setdefault("evidence_snapshot_id", evidence_snapshot_id)
            payload.setdefault("case_version", case_version or snapshot.case_version)
            payload.setdefault("case_id", case_id or snapshot.case_id)
            _expand_family_entity_hints(payload)
            _normalize_relation_schema_hint(payload)
            payload.setdefault("_proposal_trust_tier", self._proposal_trust_tier(payload))
            hypothesis = ClinicalPatternHypothesis.from_dict(
                payload,
                index=index,
                case_id=case_id or snapshot.case_id,
                case_version=case_version or snapshot.case_version,
                evidence_snapshot_id=evidence_snapshot_id,
            )
            if hypothesis.evidence_bindings and (hypothesis.suggested_diseases or hypothesis.suggested_family):
                structured.append(hypothesis)
        reasoning = self._from_differentials_with_refs(
            snapshot,
            evidence,
            case_id=case_id or snapshot.case_id,
            case_version=case_version or snapshot.case_version,
            evidence_snapshot_id=evidence_snapshot_id,
        )
        return structured[: int(self.config.get("max_proposals_per_source", 3) or 3)], reasoning[
            : int(self.config.get("max_proposals_per_source", 3) or 3)
        ]

    def _from_diagnosis_draft(
        self,
        diagnosis_draft: Any,
        *,
        case_id: str,
        case_version: int,
        evidence_snapshot_id: str,
    ) -> List[ClinicalPatternHypothesis]:
        if not isinstance(diagnosis_draft, dict):
            return []
        raw_items = (
            diagnosis_draft.get("clinical_pattern_hypotheses")
            or diagnosis_draft.get("_clinical_pattern_hypotheses")
            or []
        )
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        result: List[ClinicalPatternHypothesis] = []
        for index, item in enumerate(raw_items or []):
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            payload.setdefault("generator_source", "diagnosis_llm_draft")
            _expand_family_entity_hints(payload)
            _normalize_relation_schema_hint(payload)
            payload.setdefault("_proposal_trust_tier", self._proposal_trust_tier(payload))
            hypothesis = ClinicalPatternHypothesis.from_dict(
                payload,
                index=index,
                case_id=case_id,
                case_version=case_version,
                evidence_snapshot_id=evidence_snapshot_id,
            )
            if hypothesis.evidence_bindings and (hypothesis.suggested_diseases or hypothesis.suggested_family):
                result.append(hypothesis)
        return result

    def _from_differentials_with_refs(
        self,
        snapshot: ThinkingSnapshot,
        evidence: EvidenceBundle,
        *,
        case_id: str,
        case_version: int,
        evidence_snapshot_id: str,
    ) -> List[ClinicalPatternHypothesis]:
        result: List[ClinicalPatternHypothesis] = []
        observations = list(evidence.observations if evidence else [])
        resolver = EvidenceRefResolver(observations, self.evidence_ontology)
        for index, item in enumerate(snapshot.differential_diagnoses or []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("diagnosis") or item.get("name") or "").strip()
            refs = _as_list(
                item.get("supporting_evidence_refs")
                or item.get("evidence_refs")
                or item.get("source_evidence_ids")
            )
            refs = [str(ref).strip() for ref in refs if str(ref).strip()]
            if not name or len(set(refs)) < 2:
                continue
            bound_observations: List[Observation] = []
            for ref in refs:
                resolved = resolver.resolve(ref)
                if resolved.binding_status != "resolved":
                    continue
                observation = _observation_by_ref(observations, resolved.resolved_observation_ref)
                if observation:
                    bound_observations.append(observation)
            if len({_evidence_group_id(item) for item in bound_observations}) < 2:
                continue
            relation_payload = self._infer_relation_from_differential(
                name,
                bound_observations,
            )
            if not relation_payload:
                continue
            payload = {
                "pattern_hypothesis_id": _stable_id("PH_THINK", f"{snapshot.snapshot_id}|{index}|{name}|{refs}"),
                "pattern_name": relation_payload["pattern_name"],
                "pattern_type": relation_payload["pattern_type"],
                "relation_schema_id": relation_payload["relation_schema_id"],
                "suggested_family": relation_payload["suggested_family"],
                "evidence_bindings": relation_payload["evidence_bindings"],
                "relations": relation_payload["relations"],
                "suggested_diseases": relation_payload["suggested_diseases"]
                or [{"name": name, "hypothesis_confidence": _safe_float(item.get("likelihood", 0.0))}],
                "missing_evidence_requests": [
                    {"target_evidence": str(value)}
                    for value in _as_list(item.get("differentiating_info"))
                    if str(value).strip()
                ],
                "generator_source": "thinking_structured_legacy",
                "_proposal_trust_tier": _TRUST_TIER_EVIDENCE_BOUND,
            }
            result.append(
                ClinicalPatternHypothesis.from_dict(
                    payload,
                    index=index,
                    case_id=case_id,
                    case_version=case_version,
                    evidence_snapshot_id=evidence_snapshot_id,
                )
            )
        return result

    def _infer_relation_from_differential(
        self,
        name: str,
        observations: Sequence[Observation],
    ) -> Dict[str, Any]:
        normalized_name = _normalize_token(name)
        bound = list(observations or [])
        if len({_evidence_group_id(item) for item in bound}) < 2:
            return {}
        if any(token in normalized_name for token in {"d100012", "mitralregurgitation", "mitral_regurgitation", "mr"}):
            return self._build_valvular_relation(bound, source_prefix="reasoning_bound")
        if any(token in normalized_name for token in {"d100058", "radiationpneumonitis", "radiation_pneumonitis"}):
            return self._build_radiation_relation(bound, source_prefix="reasoning_bound")
        return {}

    def _from_deterministic_relations(
        self,
        evidence: EvidenceBundle,
        *,
        case_id: str,
        case_version: int,
        evidence_snapshot_id: str,
    ) -> List[ClinicalPatternHypothesis]:
        if not evidence:
            return []
        observations = list(evidence.observations or [])
        results: List[ClinicalPatternHypothesis] = []
        max_schemas = int(self.config.get("max_active_schemas_per_snapshot", 5) or 5)
        binders = (
            ("exposure_temporal_organ_injury", self._build_radiation_relation),
            ("structural_function_abnormality", self._build_valvular_relation),
        )
        for schema_id, builder in binders:
            if len(results) >= max_schemas:
                break
            relation_binding = EvidenceRelationBinder(observations, self.evidence_ontology).bind(schema_id)
            payload = builder(
                observations,
                source_prefix="deterministic_relation",
                relation_binding=relation_binding,
            )
            if not payload:
                continue
            payload["pattern_hypothesis_id"] = _stable_id(
                "PH_DET",
                f"{evidence_snapshot_id}|{payload.get('relation_schema_id')}|{payload.get('suggested_family')}",
            )
            payload.setdefault("generator_source", "deterministic_relation")
            payload.setdefault("_proposal_trust_tier", _TRUST_TIER_STRUCTURED)
            hypothesis = ClinicalPatternHypothesis.from_dict(
                payload,
                index=len(results),
                case_id=case_id,
                case_version=case_version,
                evidence_snapshot_id=evidence_snapshot_id,
            )
            if hypothesis.evidence_bindings and (hypothesis.suggested_diseases or hypothesis.suggested_family):
                results.append(hypothesis)
        return results[: int(self.config.get("max_proposals_per_source", 3) or 3)]

    def _build_radiation_relation(
        self,
        observations: Sequence[Observation],
        *,
        source_prefix: str,
        relation_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        relation_binding = relation_binding or EvidenceRelationBinder(
            observations,
            self.evidence_ontology,
        ).bind("exposure_temporal_organ_injury")
        audit = relation_binding.get("audit") or {}
        if audit.get("activation_status") != "activated":
            return {}
        bound = relation_binding.get("bindings") or {}
        exposure = _observation_by_ref(observations, bound["exposure"].observation_ref)
        manifestation = _observation_by_ref(observations, bound["organ_manifestation"].observation_ref)
        objective = _observation_by_ref(observations, bound["imaging_or_objective_finding"].observation_ref)
        if not exposure or not manifestation or not objective:
            return {}
        return {
            "pattern_name": "post_thoracic_radiotherapy_lung_injury_pattern",
            "pattern_type": "exposure_temporal_organ_injury",
            "relation_schema_id": "exposure_temporal_organ_injury",
            "suggested_family": "radiation_related_lung_injury",
            "evidence_bindings": [
                _binding_for(exposure, "exposure"),
                _binding_for(manifestation, "organ_manifestation"),
                _binding_for(objective, "imaging_or_objective_finding"),
            ],
            "relations": [
                {"type": "temporal_after", "from": _observation_ref(exposure), "to": _observation_ref(manifestation)},
                {"type": "anatomical_consistency", "from": _observation_ref(exposure), "to": _observation_ref(objective)},
            ],
            "suggested_diseases": [{"name": "D100058", "canonical_id": "D100058"}],
            "missing_evidence_requests": [{"target_evidence": "infection_exclusion", "importance": "supportive"}],
            "generator_source": source_prefix,
            "relation_activation_audit": audit,
        }

    def _build_valvular_relation(
        self,
        observations: Sequence[Observation],
        *,
        source_prefix: str,
        relation_binding: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        relation_binding = relation_binding or EvidenceRelationBinder(
            observations,
            self.evidence_ontology,
        ).bind("structural_function_abnormality")
        audit = relation_binding.get("audit") or {}
        if audit.get("activation_status") != "activated":
            return {}
        bound = relation_binding.get("bindings") or {}
        structure = _observation_by_ref(observations, bound["structure_or_credible_sign"].observation_ref)
        function = _observation_by_ref(observations, bound["function_impairment"].observation_ref)
        regurgitation_binding = bound.get("regurgitation_specific")
        regurgitation = (
            _observation_by_ref(observations, regurgitation_binding.observation_ref)
            if regurgitation_binding
            else None
        )
        if not structure or not function:
            return {}
        bindings = [_binding_for(structure, "structure_or_credible_sign"), _binding_for(function, "function_impairment")]
        family = "valvular_left_heart"
        pattern_type = "structural_function_abnormality"
        pattern_name = "left_sided_valvular_disease_pattern"
        relations = [{"type": "structural_function_abnormality", "from": _observation_ref(structure), "to": _observation_ref(function)}]
        if regurgitation:
            bindings.append(_binding_for(regurgitation, "regurgitation_specific"))
            family = "valvular_left_heart"
            pattern_type = "left_sided_valvular_regurgitation"
            pattern_name = "left_sided_valvular_regurgitation_pattern"
            relations.append({"type": "anatomical_consistency", "from": _observation_ref(regurgitation), "to": _observation_ref(structure)})
        return {
            "pattern_name": pattern_name,
            "pattern_type": pattern_type,
            "relation_schema_id": pattern_type,
            "suggested_family": family,
            "evidence_bindings": bindings,
            "relations": relations,
            "suggested_diseases": [{"name": "D100012", "canonical_id": "D100012"}],
            "missing_evidence_requests": [{"target_evidence": "echo_regurgitant_jet", "importance": "confirmatory"}],
            "generator_source": source_prefix,
            "relation_activation_audit": audit,
        }

    def _proposal_trust_tier(self, payload: Dict[str, Any]) -> str:
        bindings = payload.get("evidence_bindings") or payload.get("source_evidence_ids") or []
        relations = [item for item in payload.get("relations") or [] if isinstance(item, dict)]
        controlled_relations = [
            item
            for item in relations
            if str(item.get("type") or "") in _CONTROLLED_RELATION_TYPES
        ]
        if bindings and controlled_relations and len(bindings) >= 2:
            return _TRUST_TIER_STRUCTURED
        if bindings:
            return _TRUST_TIER_EVIDENCE_BOUND
        return _TRUST_TIER_QUERY_ONLY

    def _dedupe(self, hypotheses: Sequence[ClinicalPatternHypothesis]) -> List[ClinicalPatternHypothesis]:
        by_key: Dict[str, ClinicalPatternHypothesis] = {}
        order: List[str] = []

        def priority(item: ClinicalPatternHypothesis) -> int:
            source = str(item.generator_source or "")
            if source == "deterministic_relation":
                return 3
            if source == "thinking_structured":
                return 2
            if source == "diagnosis_llm_draft":
                return 1
            return 0

        for hypothesis in hypotheses or []:
            key = _proposal_signature(hypothesis)
            if key not in by_key:
                order.append(key)
                by_key[key] = hypothesis
                continue
            if priority(hypothesis) > priority(by_key[key]):
                by_key[key] = hypothesis
        return [by_key[key] for key in order if key in by_key]


class PatternProposalCompiler(PatternProposalAdapter):
    """Canonical v2 compiler name; the adapter remains as a compatibility API."""


class PatternHypothesisVerifier:
    """Validate LLM pattern hypotheses and emit recall-only signals."""

    def __init__(self, knowledge: Any = None, config: Optional[Dict[str, Any]] = None):
        self.knowledge = knowledge
        section = ((config or {}).get("diagnosis", {}) or {}).get("llm_pattern_hypothesis", {})
        merged = dict(_DEFAULT_CONFIG)
        merged.update(dict(section or {}))
        self.config = merged
        ref_dir = str((config or {}).get("ref_data_dir") or "data/ref_data")
        self.evidence_ontology = _load_evidence_ontology(ref_dir)

    def parse_hypotheses(
        self,
        llm_result: Any,
        *,
        case_id: str = "",
        case_version: int = 0,
        evidence_snapshot_id: str = "",
    ) -> List[ClinicalPatternHypothesis]:
        if not self.config.get("enabled", True) or not isinstance(llm_result, dict):
            return []
        raw_items = (
            llm_result.get("clinical_pattern_hypotheses")
            or llm_result.get("_clinical_pattern_hypotheses")
            or []
        )
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        result: List[ClinicalPatternHypothesis] = []
        for index, item in enumerate(raw_items or []):
            if not isinstance(item, dict):
                continue
            hypothesis = ClinicalPatternHypothesis.from_dict(
                item,
                index=index,
                case_id=case_id,
                case_version=case_version,
                evidence_snapshot_id=evidence_snapshot_id,
            )
            if hypothesis.evidence_bindings and hypothesis.suggested_diseases:
                result.append(hypothesis)
            if len(result) >= int(self.config.get("max_hypotheses", 3) or 3):
                break
        return result

    def verify_all(
        self,
        hypotheses: Sequence[ClinicalPatternHypothesis],
        evidence: EvidenceBundle,
        *,
        case_version: int = 0,
        evidence_snapshot_id: str = "",
    ) -> List[PatternVerificationResult]:
        return [
            self.verify(
                hypothesis,
                evidence,
                case_version=case_version,
                evidence_snapshot_id=evidence_snapshot_id,
            )
            for hypothesis in hypotheses or []
        ]

    def verify(
        self,
        hypothesis: ClinicalPatternHypothesis,
        evidence: EvidenceBundle,
        *,
        case_version: int = 0,
        evidence_snapshot_id: str = "",
    ) -> PatternVerificationResult:
        observations = list(evidence.observations if evidence else [])
        observation_index, _ambiguous_refs = _build_observation_lookup(observations)
        resolver = EvidenceRefResolver(observations, self.evidence_ontology)
        valid_ids: List[str] = []
        invalid_ids: List[str] = []
        source_groups: Dict[str, List[str]] = {}
        source_group_ids: Dict[str, List[str]] = {}
        ref_resolution_audit: List[Dict[str, Any]] = []
        rejection_reasons: List[str] = []
        contradiction_strength = 0.0

        if hypothesis.evidence_snapshot_id and evidence_snapshot_id and hypothesis.evidence_snapshot_id != evidence_snapshot_id:
            rejection_reasons.append("stale_evidence_snapshot")
        if case_version and hypothesis.case_version and int(hypothesis.case_version) != int(case_version):
            rejection_reasons.append("stale_case_version")

        for binding in hypothesis.evidence_bindings:
            evidence_id = str(binding.evidence_id or "").strip()
            group = _normal_group(binding.relation_slot)
            resolved = resolver.resolve(evidence_id, expected_slot=group)
            ref_resolution_audit.append(
                {
                    "raw_ref": evidence_id,
                    "expected_slot": group,
                    "resolution": resolved.to_dict(),
                }
            )
            if resolved.binding_status == "ambiguous":
                invalid_ids.append(evidence_id)
                rejection_reasons.append("ambiguous_evidence_binding")
                continue
            observation = _observation_by_ref(observations, resolved.resolved_observation_ref)
            if not observation:
                invalid_ids.append(evidence_id)
                continue
            lineage_reason = _lineage_rejection_reason(observation)
            if lineage_reason:
                invalid_ids.append(evidence_id)
                rejection_reasons.append(lineage_reason)
                continue
            if evidence_id == hypothesis.pattern_hypothesis_id:
                invalid_ids.append(evidence_id)
                rejection_reasons.append("circular_evidence_lineage")
                continue
            expected = str(binding.expected_polarity or "positive").strip()
            if expected and expected != str(observation.polarity or "positive"):
                invalid_ids.append(evidence_id)
                rejection_reasons.append("polarity_mismatch")
                continue
            valid_ids.append(resolved.resolved_observation_ref)
            if binding.role == "contradiction":
                group = "exclusion_or_contradiction"
                contradiction_strength = max(
                    contradiction_strength,
                    max(0.25, min(1.0, observation.confidence * _information_value(observation))),
                )
            source_groups.setdefault(group, [])
            if observation.finding not in source_groups[group]:
                source_groups[group].append(observation.finding)
            source_group_ids.setdefault(group, [])
            group_id = _evidence_group_id(observation)
            if group_id not in source_group_ids[group]:
                source_group_ids[group].append(group_id)

        if invalid_ids:
            rejection_reasons.append("unsupported_source_evidence")
        if not valid_ids:
            rejection_reasons.append("no_valid_source_evidence")

        valid_ref_set = set(valid_ids)
        resolved_relations: List[Dict[str, Any]] = []
        for relation in hypothesis.relations or []:
            relation_type = str(relation.get("type") or "").strip()
            if relation_type and relation_type not in _CONTROLLED_RELATION_TYPES:
                rejection_reasons.append("unsupported_relation_type")
            resolved_relation = dict(relation)
            for key in ("from", "to"):
                raw_ref = str(
                    relation.get(key)
                    or relation.get(f"{key}_evidence_ref")
                    or ""
                ).strip()
                if not raw_ref:
                    continue
                ref_result = resolver.resolve(raw_ref)
                ref_resolution_audit.append(
                    {
                        "raw_ref": raw_ref,
                        "expected_slot": "relation_endpoint",
                        "relation_type": relation_type,
                        "resolution": ref_result.to_dict(),
                    }
                )
                if ref_result.binding_status == "ambiguous":
                    rejection_reasons.append("ambiguous_evidence_binding")
                    continue
                if ref_result.binding_status != "resolved":
                    rejection_reasons.append("relation_endpoint_unbound")
                    continue
                resolved_relation[key] = ref_result.resolved_observation_ref
                if ref_result.resolved_observation_ref not in valid_ref_set:
                    rejection_reasons.append("relation_endpoint_unbound")
            resolved_relations.append(resolved_relation)

        resolved_hypothesis = ClinicalPatternHypothesis(
            pattern_hypothesis_id=hypothesis.pattern_hypothesis_id,
            case_id=hypothesis.case_id,
            case_version=hypothesis.case_version,
            evidence_snapshot_id=hypothesis.evidence_snapshot_id,
            pattern_name=hypothesis.pattern_name,
            pattern_type=hypothesis.pattern_type,
            suggested_family=hypothesis.suggested_family,
            relation_schema_id=hypothesis.relation_schema_id,
            evidence_bindings=list(hypothesis.evidence_bindings),
            relations=resolved_relations,
            suggested_diseases=list(hypothesis.suggested_diseases),
            missing_evidence_requests=list(hypothesis.missing_evidence_requests),
            relation_activation_audit=dict(hypothesis.relation_activation_audit),
            model_confidence=hypothesis.model_confidence,
            generator_source=hypothesis.generator_source,
            proposal_trust_tier=hypothesis.proposal_trust_tier,
            model_id=hypothesis.model_id,
            prompt_version=hypothesis.prompt_version,
            created_at=hypothesis.created_at,
        )
        relation_activation_audit = dict(hypothesis.relation_activation_audit or {})
        relation_results = _verify_relation_claims(
            resolved_hypothesis,
            observation_index,
            valid_ref_set,
            source_groups=source_groups,
            relation_activation_audit=relation_activation_audit,
        )
        for relation_result in relation_results:
            status = str(relation_result.get("status") or "")
            if status == "contradicted":
                rejection_reasons.append("relation_contradicted")
            elif status == "unresolved":
                rejection_reasons.append("relation_unresolved")

        entity_links = self._link_entities(hypothesis)
        if not entity_links:
            rejection_reasons.append("entity_unresolved")

        support_strength = _support_strength(source_groups, observations)
        relation_complete = _relation_complete(resolved_hypothesis, source_groups, source_group_ids, relation_results)
        critical_complete = _critical_anchor_complete(resolved_hypothesis, source_groups, source_group_ids, relation_results)
        audit_activated = str(relation_activation_audit.get("activation_status") or "") == "activated"
        if audit_activated and source_groups:
            relation_complete = True
            critical_complete = True
            rejection_reasons = [
                reason
                for reason in rejection_reasons
                if reason not in {"relation_unresolved", "critical_anchor_incomplete", "relation_incomplete"}
            ]
        if not critical_complete:
            rejection_reasons.append("critical_anchor_incomplete")
        if not relation_complete:
            rejection_reasons.append("relation_incomplete")

        contradiction_penalty = min(0.65, contradiction_strength)
        missing_penalty = 0.18 if not critical_complete else 0.0
        relation_penalty = 0.12 if not relation_complete else 0.0
        net = max(0.0, min(1.0, support_strength - contradiction_penalty - missing_penalty - relation_penalty))

        critical_contradiction = contradiction_strength >= 0.55
        source_lineage_ok = bool(valid_ids) and not any(
            reason in rejection_reasons
            for reason in {
                "reasoning_inference_source",
                "candidate_derived_source",
                "hypothesis_derived_source",
                "judge_derived_source",
                "ordered_exam_name_source",
                "exam_request_source",
                "llm_generated_summary_without_source",
                "circular_evidence_lineage",
                "stale_evidence_snapshot",
            }
        )
        hard_gate_results = {
            "source_lineage_ok": source_lineage_ok,
            "critical_contradiction": critical_contradiction,
            "independent_evidence_group_count": len(
                [key for key, values in source_groups.items() if values and key != "exclusion_or_contradiction"]
            ),
            "entity_link_threshold_met": any(
                item.link_confidence >= float(self.config.get("min_entity_link_confidence", 0.85) or 0.85)
                for item in entity_links
            ),
            "allow_direct_gap_creation": bool(self.config.get("allow_direct_gap_creation", False)),
            "allow_judge_score_contribution": bool(self.config.get("allow_judge_score_contribution", False)),
            "allow_eligibility_evidence_contribution": bool(
                self.config.get("allow_eligibility_evidence_contribution", False)
            ),
        }
        slot_binding_audit = {
            "relation_schema_id": _normalize_schema_id(
                hypothesis.relation_schema_id or hypothesis.pattern_type
            ),
            "required_slots": list(
                _SCHEMA_REQUIRED_SLOTS.get(
                    _normalize_schema_id(hypothesis.relation_schema_id or hypothesis.pattern_type),
                    [],
                )
            ),
            "bound_slots": sorted(
                key for key, values in source_groups.items() if values and key != "exclusion_or_contradiction"
            ),
            "missing_slots": [
                slot
                for slot in _SCHEMA_REQUIRED_SLOTS.get(
                    _normalize_schema_id(hypothesis.relation_schema_id or hypothesis.pattern_type),
                    [],
                )
                if not source_groups.get(slot)
            ],
            "ambiguous_slots": [],
            "supporting_evidence": {
                slot: list(values)
                for slot, values in source_groups.items()
                if values
            },
        }
        if not relation_activation_audit:
            relation_activation_audit = {
                "relation_schema_id": slot_binding_audit["relation_schema_id"],
                "required_slots": list(slot_binding_audit["required_slots"]),
                "bound_slots": list(slot_binding_audit["bound_slots"]),
                "missing_slots": list(slot_binding_audit["missing_slots"]),
                "critical_constraints": list(
                    _SCHEMA_CRITICAL_CONSTRAINTS.get(slot_binding_audit["relation_schema_id"], [])
                ),
                "constraint_results": [
                    {
                        "constraint_type": item.get("type"),
                        "from_observation_ref": item.get("from"),
                        "to_observation_ref": item.get("to"),
                        "status": "satisfied" if item.get("status") == "verified" else item.get("status"),
                        "reason": item.get("reason", ""),
                    }
                    for item in relation_results
                ],
                "activation_status": "activated" if relation_complete and critical_complete else "partial",
                "activation_score": round(support_strength, 4),
                "rejection_reasons": [
                    reason
                    for reason in {
                        *[f"missing_slot:{slot}" for slot in slot_binding_audit["missing_slots"]],
                        *[str(item.get("reason") or "") for item in relation_results if item.get("reason")],
                    }
                    if reason
                ],
            }
        specificity = "entity" if source_groups.get("regurgitation_specific") else "family"

        status = STATUS_VERIFIED
        hard_rejections = {
            "unsupported_source_evidence",
            "no_valid_source_evidence",
            "stale_evidence_snapshot",
            "stale_case_version",
            "reasoning_inference_source",
            "candidate_derived_source",
            "hypothesis_derived_source",
            "judge_derived_source",
            "ordered_exam_name_source",
            "exam_request_source",
            "llm_generated_summary_without_source",
            "circular_evidence_lineage",
            "entity_unresolved",
                "unsupported_relation_type",
                "relation_endpoint_unbound",
                "ambiguous_evidence_binding",
                "relation_contradicted",
            }
        activated_relation = (
            (relation_activation_audit or {}).get("activation_status") == "activated"
            and bool(relation_complete)
            and bool(critical_complete)
        )
        if any(reason in hard_rejections for reason in rejection_reasons):
            status = STATUS_REJECTED
        elif net < float(self.config.get("active_min_strength", 0.55) or 0.55) and not activated_relation:
            status = STATUS_UNRESOLVED

        return PatternVerificationResult(
            pattern_hypothesis_id=hypothesis.pattern_hypothesis_id,
            verification_status=status,
            valid_source_evidence_ids=list(dict.fromkeys(valid_ids)),
            invalid_source_evidence_ids=list(dict.fromkeys(invalid_ids)),
            support_strength=round(support_strength, 4),
            contradiction_strength=round(contradiction_strength, 4),
            critical_anchor_completeness=bool(critical_complete),
            relation_completeness=bool(relation_complete),
            entity_links=entity_links,
            hard_gate_results=hard_gate_results,
            net_pattern_strength=round(net, 4),
            rejection_reasons=list(dict.fromkeys(rejection_reasons)),
            verified_at=_utc_now(),
            source_groups=source_groups,
            missing_evidence_requests=list(hypothesis.missing_evidence_requests),
            hypothesis=resolved_hypothesis.to_dict(),
            ref_resolution_audit=ref_resolution_audit,
            slot_binding_audit=slot_binding_audit,
            relation_activation_audit=relation_activation_audit,
            admission_level="entity_specific" if specificity == "entity" else "family_expansion",
            verified_specificity=specificity,
        )

    def signals_from_results(
        self,
        results: Sequence[PatternVerificationResult],
    ) -> List[PatternRecallSignal]:
        signals: List[PatternRecallSignal] = []
        protected_used = 0
        recall_used = 0
        active_min = float(self.config.get("active_min_strength", 0.55) or 0.55)
        protected_min = float(self.config.get("protection_min_strength", 0.75) or 0.75)
        entity_min = float(self.config.get("min_entity_link_confidence", 0.85) or 0.85)
        for result in results or []:
            if result.verification_status != STATUS_VERIFIED:
                continue
            for link in result.entity_links:
                if link.link_confidence < entity_min or not link.entity_id:
                    continue
                source_ceiling = _source_permission_ceiling(result.hypothesis)
                activated_relation = _has_activated_relation(result)
                independent_group_count = int(result.hard_gate_results.get("independent_evidence_group_count") or 0)
                relation_protected_floor = (
                    activated_relation
                    and independent_group_count >= 3
                    and _schema_id_from_result(result) == "exposure_temporal_organ_injury"
                )
                protected = (
                    (result.net_pattern_strength >= protected_min or relation_protected_floor)
                    and result.critical_anchor_completeness
                    and result.relation_completeness
                    and not bool(result.hard_gate_results.get("critical_contradiction"))
                    and bool(result.hard_gate_results.get("source_lineage_ok"))
                    and independent_group_count >= 3
                    and protected_used < int(self.config.get("max_protected_candidates", 2) or 2)
                    and source_ceiling == RECALL_PROTECTED
                )
                mode = RECALL_PROTECTED if protected else RECALL_BOOST
                if source_ceiling == RECALL_QUERY_EXPANSION:
                    mode = RECALL_QUERY_EXPANSION
                if result.net_pattern_strength < active_min and not activated_relation:
                    mode = RECALL_QUERY_EXPANSION
                if mode == RECALL_PROTECTED:
                    protected_used += 1
                if mode in {RECALL_BOOST, RECALL_PROTECTED}:
                    if recall_used >= int(self.config.get("max_recall_candidates", 3) or 3):
                        continue
                    recall_used += 1
                signals.append(
                    PatternRecallSignal(
                        pattern_hypothesis_id=result.pattern_hypothesis_id,
                        entity_id=link.entity_id,
                        entity_link_confidence=round(link.link_confidence, 4),
                        recall_mode=mode,
                        recall_strength=round(result.net_pattern_strength, 4),
                        protected_pool_slot=bool(protected),
                        source_evidence_ids=list(result.valid_source_evidence_ids),
                        missing_evidence_requests=list(result.missing_evidence_requests),
                        canonical_name=link.canonical_name,
                        submission_name=link.submission_name,
                        raw_name=link.raw_name,
                        admission_level=result.admission_level,
                        verified_specificity=result.verified_specificity,
                    )
                )
        return signals

    def _link_entities(self, hypothesis: ClinicalPatternHypothesis) -> List[PatternEntityLink]:
        links: List[PatternEntityLink] = []
        seen: set[str] = set()
        diseases = list(hypothesis.suggested_diseases or [])
        for entity in _entities_for_family(self.knowledge, hypothesis.suggested_family):
            diseases.append(SuggestedDisease(name=entity.entity_id, canonical_id=entity.entity_id))
        for disease in diseases or []:
            entity = None
            raw_name = disease.name or disease.canonical_id
            for value in (disease.canonical_id, disease.name):
                if not value or not self.knowledge:
                    continue
                entity = self.knowledge.resolve_entity(value)
                if entity:
                    break
            if not entity:
                continue
            if entity.entity_id in seen:
                continue
            seen.add(entity.entity_id)
            links.append(
                PatternEntityLink(
                    entity_id=entity.entity_id,
                    canonical_name=entity.canonical_name,
                    submission_name=entity.display_name,
                    raw_name=raw_name,
                    link_confidence=1.0,
                    resolution_status="controlled_entity",
                    submittable=bool(entity.submittable),
                )
            )
        return links


def build_pattern_recall_context(
    verifier: PatternHypothesisVerifier,
    llm_result: Any,
    evidence: EvidenceBundle,
    *,
    case_id: str = "",
    case_version: int = 0,
    evidence_snapshot_id: str = "",
    thinking_snapshots: Optional[Sequence[Any]] = None,
    adapter: Optional[PatternProposalAdapter] = None,
) -> Dict[str, Any]:
    snapshot_id = evidence_snapshot_id or evidence_snapshot_hash(evidence)
    if adapter is None:
        adapter = PatternProposalAdapter({"diagnosis": {"llm_pattern_hypothesis": verifier.config}})
    hypotheses = adapter.propose(
        thinking_snapshots or [],
        llm_result,
        evidence,
        case_id=case_id,
        case_version=case_version,
        evidence_snapshot_id=snapshot_id,
    )
    verification_results = verifier.verify_all(
        hypotheses,
        evidence,
        case_version=case_version,
        evidence_snapshot_id=snapshot_id,
    )
    signals = verifier.signals_from_results(verification_results)
    audit = _pattern_recall_audit(
        hypotheses,
        verification_results,
        signals,
        thinking_snapshot_count=len(list(thinking_snapshots or [])),
        evidence_snapshot_id=snapshot_id,
        compiler_audit=getattr(adapter, "last_audit", {}),
    )
    return {
        "pattern_hypotheses": [item.to_dict() for item in hypotheses],
        "pattern_verification_results": [item.to_dict() for item in verification_results],
        "pattern_recall_signals": [item.to_dict() for item in signals],
        "pattern_recall_audit": audit,
        "pattern_driven_candidate_recall": [
            item.to_dict() for item in signals if item.recall_mode in {RECALL_BOOST, RECALL_PROTECTED}
        ],
        "pattern_protected_candidate_recall": [
            item.to_dict() for item in signals if item.recall_mode == RECALL_PROTECTED
        ],
        "pattern_gap_suggestions": _gap_suggestions(signals),
        "unverified_pattern_leakage_count": 0,
        "pattern_generated_active_gaps": 0,
        "pattern_expansion_round_count": 1 if hypotheses else 0,
        "evidence_snapshot_id": snapshot_id,
        "thinking_snapshot_count": len(list(thinking_snapshots or [])),
    }


def evidence_snapshot_hash(evidence: EvidenceBundle) -> str:
    payload = [
        {
            "finding": item.finding,
            "source": item.source,
            "polarity": item.polarity,
            "field_path": item.field_path,
            "raw_text": item.raw_text,
            "value": item.value,
            "unit": item.unit,
        }
        for item in (evidence.observations if evidence else [])
    ]
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    return f"evidence_snapshot:{digest[:16]}"


def coerce_pattern_recall_context(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "pattern_hypotheses": list(value.get("pattern_hypotheses") or []),
        "pattern_verification_results": list(value.get("pattern_verification_results") or []),
        "pattern_recall_signals": list(value.get("pattern_recall_signals") or []),
        "pattern_recall_audit": dict(value.get("pattern_recall_audit") or {}),
        "pattern_driven_candidate_recall": list(value.get("pattern_driven_candidate_recall") or []),
        "pattern_protected_candidate_recall": list(value.get("pattern_protected_candidate_recall") or []),
        "pattern_gap_suggestions": list(value.get("pattern_gap_suggestions") or []),
        "unverified_pattern_leakage_count": int(value.get("unverified_pattern_leakage_count") or 0),
        "pattern_generated_active_gaps": int(value.get("pattern_generated_active_gaps") or 0),
        "pattern_expansion_round_count": min(1, int(value.get("pattern_expansion_round_count") or 0)),
        "evidence_snapshot_id": str(value.get("evidence_snapshot_id") or ""),
        "thinking_snapshot_count": int(value.get("thinking_snapshot_count") or 0),
    }


def _pattern_recall_audit(
    hypotheses: Sequence[ClinicalPatternHypothesis],
    verification_results: Sequence[PatternVerificationResult],
    signals: Sequence[PatternRecallSignal],
    *,
    thinking_snapshot_count: int,
    evidence_snapshot_id: str,
    compiler_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compact stage-by-stage audit for Pattern Recall failure attribution."""
    proposal_sources: Dict[str, int] = {}
    proposal_trust_tiers: Dict[str, int] = {}
    for hypothesis in hypotheses or []:
        _count_key(proposal_sources, hypothesis.generator_source or "unknown")
        _count_key(proposal_trust_tiers, hypothesis.proposal_trust_tier or "unknown")

    verification_statuses: Dict[str, int] = {}
    rejection_reasons: Dict[str, int] = {}
    linked_entity_ids: List[str] = []
    unresolved_entity_count = 0
    verification_records: List[Dict[str, Any]] = []
    relation_activation_results: List[Dict[str, Any]] = []
    slot_binding_results: List[Dict[str, Any]] = []
    ref_resolution_results: List[Dict[str, Any]] = []
    for result in verification_results or []:
        _count_key(verification_statuses, result.verification_status or "unknown")
        for reason in result.rejection_reasons or []:
            _count_key(rejection_reasons, str(reason or "unknown"))
        entity_links = list(result.entity_links or [])
        if not entity_links:
            unresolved_entity_count += 1
        for link in entity_links:
            if link.entity_id:
                linked_entity_ids.append(link.entity_id)
        verification_records.append(
            {
                "pattern_hypothesis_id": result.pattern_hypothesis_id,
                "verification_status": result.verification_status,
                "rejection_reasons": list(result.rejection_reasons or []),
                "source_groups": dict(result.source_groups or {}),
                "net_pattern_strength": result.net_pattern_strength,
                "critical_anchor_completeness": result.critical_anchor_completeness,
                "relation_completeness": result.relation_completeness,
            }
        )
        if result.ref_resolution_audit:
            ref_resolution_results.append(
                {
                    "pattern_hypothesis_id": result.pattern_hypothesis_id,
                    "records": list(result.ref_resolution_audit or []),
                }
            )
        if result.slot_binding_audit:
            slot_binding_results.append(
                {
                    "pattern_hypothesis_id": result.pattern_hypothesis_id,
                    **dict(result.slot_binding_audit or {}),
                }
            )
        if result.relation_activation_audit:
            relation_activation_results.append(
                {
                    "pattern_hypothesis_id": result.pattern_hypothesis_id,
                    **dict(result.relation_activation_audit or {}),
                }
            )

    signal_modes: Dict[str, int] = {}
    signal_entity_ids: List[str] = []
    protected_signal_count = 0
    for signal in signals or []:
        _count_key(signal_modes, signal.recall_mode or "unknown")
        if signal.entity_id:
            signal_entity_ids.append(signal.entity_id)
        if signal.protected_pool_slot:
            protected_signal_count += 1

    return {
        "evidence_snapshot_id": evidence_snapshot_id,
        "thinking_snapshot_count": int(thinking_snapshot_count or 0),
        "proposal_count": len(list(hypotheses or [])),
        "proposal_sources": proposal_sources,
        "proposal_trust_tiers": proposal_trust_tiers,
        "verification_count": len(list(verification_results or [])),
        "verification_statuses": verification_statuses,
        "rejection_reasons": rejection_reasons,
        "entity_link_count": len(list(dict.fromkeys(linked_entity_ids))),
        "linked_entity_ids": list(dict.fromkeys(linked_entity_ids)),
        "unresolved_entity_count": unresolved_entity_count,
        "signal_count": len(list(signals or [])),
        "signal_modes": signal_modes,
        "signal_entity_ids": list(dict.fromkeys(signal_entity_ids)),
        "protected_signal_count": protected_signal_count,
        "unverified_pattern_leakage_count": 0,
        "pattern_generated_active_gaps": 0,
        "compiler_enabled": bool((compiler_audit or {}).get("compiler_enabled", True)),
        "compiler_audit": dict(compiler_audit or _empty_compiler_audit(evidence_snapshot_id=evidence_snapshot_id)),
        "pattern_pipeline_audit": {
            "proposal_count_by_source": proposal_sources,
            "proposal_rejection_by_source": rejection_reasons,
            "verification_records": verification_records,
            "ref_resolution_results": ref_resolution_results,
            "slot_binding_results": slot_binding_results,
            "relation_activation_results": relation_activation_results,
            "entity_expansion_results": [
                {"entity_id": entity_id, "source": "pattern_recall_signal"}
                for entity_id in list(dict.fromkeys(signal_entity_ids))
            ],
            "protected_recall_count": protected_signal_count,
            "recall_boost_count": int(signal_modes.get(RECALL_BOOST, 0)),
            "query_expansion_count": int(signal_modes.get(RECALL_QUERY_EXPANSION, 0)),
        },
    }


def _count_key(target: Dict[str, int], key: str) -> None:
    text = str(key or "unknown")
    target[text] = int(target.get(text, 0)) + 1


def _build_observation_index(observations: Sequence[Observation]) -> Dict[str, Observation]:
    return _build_observation_lookup(observations)[0]


def _build_observation_lookup(observations: Sequence[Observation]) -> Tuple[Dict[str, Observation], set[str]]:
    index: Dict[str, Observation] = {}
    ref_to_group: Dict[str, str] = {}
    ambiguous: set[str] = set()
    for item in observations or []:
        refs = [
            _observation_ref(item),
            item.finding,
            f"finding:{item.finding}",
            _evidence_group_id(item),
        ]
        if item.field_path:
            refs.append(item.field_path)
        for ref in refs:
            text = str(ref or "").strip()
            if not text:
                continue
            group_id = _evidence_group_id(item)
            existing_group = ref_to_group.get(text)
            if existing_group and existing_group != group_id:
                ambiguous.add(text)
                continue
            ref_to_group[text] = group_id
            index.setdefault(text, item)
    for ref in ambiguous:
        index.pop(ref, None)
    return index, ambiguous


def _observation_ref(item: Observation) -> str:
    parts = [item.finding, item.source, item.field_path, item.polarity]
    return "obs:" + hashlib.sha1("|".join(str(part or "") for part in parts).encode("utf-8")).hexdigest()[:12]


def _evidence_group_id(item: Observation) -> str:
    parts = [
        _normalize_token(item.finding),
        str(item.polarity or "positive").lower(),
        _normalize_token(getattr(item, "anatomy", "")),
        _time_bucket(getattr(item, "temporality", "")),
        _source_bucket(getattr(item, "source", "")),
    ]
    return "eg:" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _binding_for(item: Observation, slot: str) -> Dict[str, Any]:
    return {
        "evidence_id": _observation_ref(item),
        "role": "support",
        "expected_polarity": item.polarity or "positive",
        "relation_slot": slot,
    }


def _best_observation(observations: Sequence[Observation], findings: set[str]) -> Optional[Observation]:
    candidates = [
        item
        for item in observations or []
        if item.polarity == "positive" and item.finding in findings and not _lineage_rejection_reason(item)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.confidence or 0.0) * _information_value(item))


def _lineage_rejection_reason(item: Observation) -> str:
    fields = [
        item.source,
        item.evidence_level,
        item.verification_method,
        item.parser_profile,
        item.raw_text,
        item.source_text,
    ]
    lowered = " ".join(str(value or "").lower() for value in fields)
    if "reasoning_inference" in lowered or "reasoning" == str(item.source or "").lower():
        return "reasoning_inference_source"
    if "candidate" in lowered:
        return "candidate_derived_source"
    if "hypothesis" in lowered:
        return "hypothesis_derived_source"
    if "judge" in lowered:
        return "judge_derived_source"
    if "ordered_exam" in lowered:
        return "ordered_exam_name_source"
    if "exam_request" in lowered:
        return "exam_request_source"
    if "llm_generated_summary_without_source" in lowered:
        return "llm_generated_summary_without_source"
    for token in _FORBIDDEN_LINEAGE_TOKENS:
        if token in lowered:
            return f"{token}_source"
    return ""


def _normal_group(value: Any) -> str:
    text = str(value or "support").strip() or "support"
    text = _ROLE_ALIASES.get(text, text)
    return text if text in _RELATION_SLOTS else "support"


def _information_value(item: Observation) -> float:
    try:
        value = float(item.information_value or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0:
        value = 0.5
    return max(0.1, min(1.0, value))


def _support_strength(source_groups: Dict[str, List[str]], observations: Sequence[Observation]) -> float:
    finding_values = {
        item.finding: max(0.1, min(1.0, float(item.confidence or 0.0) * _information_value(item)))
        for item in observations or []
    }
    weights = {
        "exposure": 0.24,
        "temporal_relation": 0.20,
        "organ_manifestation": 0.18,
        "imaging_or_objective_finding": 0.24,
        "structure_or_credible_sign": 0.32,
        "function_impairment": 0.40,
        "regurgitation_specific": 0.24,
        "support": 0.14,
        "context": 0.06,
    }
    score = 0.0
    for group, findings in source_groups.items():
        if group == "exclusion_or_contradiction" or not findings:
            continue
        group_value = max(finding_values.get(finding, 0.5) for finding in findings)
        score += weights.get(group, 0.08) * group_value
    independent_groups = len([key for key, values in source_groups.items() if values and key != "exclusion_or_contradiction"])
    if independent_groups >= 3:
        score += 0.12
    elif independent_groups >= 2:
        score += 0.06
    return max(0.0, min(1.0, score))


def _relation_complete(
    hypothesis: ClinicalPatternHypothesis,
    source_groups: Dict[str, List[str]],
    source_group_ids: Dict[str, List[str]],
    relation_results: Sequence[Dict[str, Any]],
) -> bool:
    if any(item.get("status") == "contradicted" for item in relation_results or []):
        return False
    schema = str(hypothesis.relation_schema_id or hypothesis.pattern_type or "").lower()
    verified_relation_types = {
        str(item.get("type") or "")
        for item in relation_results or []
        if item.get("status") == "verified"
    }
    if schema == "exposure_temporal_organ_injury":
        return bool(
            source_groups.get("exposure")
            and source_groups.get("organ_manifestation")
            and source_groups.get("imaging_or_objective_finding")
            and "temporal_after" in verified_relation_types
            and "anatomical_consistency" in verified_relation_types
            and _independent_group_count(source_group_ids) >= 3
        )
    if schema in {"structural_function_abnormality", "left_sided_valvular_disease"}:
        return bool(
            source_groups.get("structure_or_credible_sign")
            and source_groups.get("function_impairment")
            and _independent_group_count(source_group_ids) >= 2
        )
    if schema == "left_sided_valvular_regurgitation":
        return bool(
            source_groups.get("structure_or_credible_sign")
            and source_groups.get("function_impairment")
            and source_groups.get("regurgitation_specific")
            and _independent_group_count(source_group_ids) >= 3
        )
    if source_groups.get("temporal_relation"):
        return True
    if verified_relation_types & {"anatomical_consistency", "vascular_shunt_pattern"}:
        return _independent_group_count(source_group_ids) >= 2
    for relation in hypothesis.relations or []:
        relation_type = str(relation.get("type") or "").lower()
        if relation_type in {"structural_function_abnormality", "cross_system_cooccurrence"}:
            if _independent_group_count(source_group_ids) >= 2:
                return True
    return False


def _critical_anchor_complete(
    hypothesis: ClinicalPatternHypothesis,
    source_groups: Dict[str, List[str]],
    source_group_ids: Dict[str, List[str]],
    relation_results: Optional[Sequence[Dict[str, Any]]] = None,
) -> bool:
    text = f"{hypothesis.pattern_name} {hypothesis.pattern_type} {hypothesis.relation_schema_id} {hypothesis.suggested_family}".lower()
    groups = {key for key, values in source_groups.items() if values}
    schema = str(hypothesis.relation_schema_id or hypothesis.pattern_type or "").lower()
    verified_relation_types = {
        str(item.get("type") or "")
        for item in relation_results or []
        if item.get("status") == "verified"
    }
    if schema == "exposure_temporal_organ_injury" or any(token in text for token in _RADIATION_PATTERN_HINTS):
        exposure_findings = set(source_groups.get("exposure") or [])
        objective_findings = set(source_groups.get("imaging_or_objective_finding") or [])
        manifestation_findings = set(source_groups.get("organ_manifestation") or [])
        return (
            "thoracic_radiotherapy" in exposure_findings
            and "temporal_after" in verified_relation_types
            and bool(
                objective_findings
                & {
                    "ground_glass_opacity",
                    "pulmonary_infiltrative_opacity",
                    "pulmonary_infiltrate",
                    "lung_opacity",
                    "pulmonary_consolidation",
                    "interstitial_opacity",
                    "atelectasis",
                }
            )
            and bool(manifestation_findings & {"dyspnea", "cough", "hypoxemia"})
            and _independent_group_count(source_group_ids) >= 3
        )
    if schema in {"structural_function_abnormality", "left_sided_valvular_disease"}:
        return (
            bool(source_groups.get("structure_or_credible_sign"))
            and bool(source_groups.get("function_impairment"))
            and _independent_group_count(source_group_ids) >= 2
        )
    if schema == "left_sided_valvular_regurgitation":
        return (
            bool(source_groups.get("structure_or_credible_sign"))
            and bool(source_groups.get("function_impairment"))
            and bool(source_groups.get("regurgitation_specific"))
            and _independent_group_count(source_group_ids) >= 3
        )
    return len(groups - {"exclusion_or_contradiction"}) >= 2


def _gap_suggestions(signals: Sequence[PatternRecallSignal]) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []
    for signal in signals or []:
        for item in signal.missing_evidence_requests or []:
            payload = dict(item)
            payload.setdefault("pattern_hypothesis_id", signal.pattern_hypothesis_id)
            payload.setdefault("entity_id", signal.entity_id)
            payload.setdefault("gap_suggestion_only", True)
            payload.setdefault("active_gap_write_permission", "none")
            suggestions.append(payload)
    return suggestions


def _has_activated_relation(result: PatternVerificationResult) -> bool:
    audit = result.relation_activation_audit or {}
    return (
        audit.get("activation_status") == "activated"
        and bool(result.critical_anchor_completeness)
        and bool(result.relation_completeness)
    )


def _schema_id_from_result(result: PatternVerificationResult) -> str:
    hypothesis = result.hypothesis if isinstance(result.hypothesis, dict) else {}
    return str(
        hypothesis.get("relation_schema_id")
        or hypothesis.get("pattern_type")
        or (result.relation_activation_audit or {}).get("relation_schema_id")
        or ""
    ).lower()


def _verify_relation_claims(
    hypothesis: ClinicalPatternHypothesis,
    observation_index: Dict[str, Observation],
    valid_refs: set[str],
    *,
    source_groups: Optional[Dict[str, List[str]]] = None,
    relation_activation_audit: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for relation in hypothesis.relations or []:
        relation_type = str(relation.get("type") or "").strip()
        left_ref = str(relation.get("from") or relation.get("from_evidence_ref") or "").strip()
        right_ref = str(relation.get("to") or relation.get("to_evidence_ref") or "").strip()
        left = observation_index.get(left_ref)
        right = observation_index.get(right_ref)
        status = "verified"
        reason = ""
        if relation_type not in _CONTROLLED_RELATION_TYPES:
            status = "contradicted"
            reason = "unsupported_relation_type"
        elif (left_ref and left_ref not in valid_refs) or (right_ref and right_ref not in valid_refs):
            status = "unresolved"
            reason = "relation_endpoint_unbound"
        elif relation_type == "temporal_after":
            if not (
                _temporal_after_verified(left, right)
                or bool((source_groups or {}).get("temporal_relation"))
                or _audit_constraint_satisfied(relation_activation_audit, "temporal_after")
            ):
                status = "unresolved"
                reason = "temporal_relation_unresolved"
        elif relation_type == "anatomical_consistency":
            if not (
                _anatomical_consistency_verified(left, right)
                or _audit_constraint_satisfied(relation_activation_audit, "anatomical_consistency")
            ):
                status = "unresolved"
                reason = "anatomical_relation_unresolved"
        elif relation_type == "structural_function_abnormality":
            if not left or not right or _evidence_group_id(left) == _evidence_group_id(right):
                status = "unresolved"
                reason = "structure_function_not_independent"
        elif relation_type == "cross_system_cooccurrence":
            if not left or not right or _source_bucket(left.source) == _source_bucket(right.source):
                status = "unresolved"
                reason = "cross_system_not_independent"
        results.append(
            {
                "type": relation_type,
                "from": left_ref,
                "to": right_ref,
                "status": status,
                "reason": reason,
            }
        )
    return results


def _audit_constraint_satisfied(audit: Optional[Dict[str, Any]], constraint_type: str) -> bool:
    if not isinstance(audit, dict):
        return False
    for item in audit.get("constraint_results") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("constraint_type") or item.get("type") or "")
        if name == constraint_type and str(item.get("status") or "") in {"satisfied", "verified"}:
            return True
    return False


def _temporal_after_verified(left: Optional[Observation], right: Optional[Observation]) -> bool:
    if not left or not right:
        return False
    left_text = f"{left.finding} {left.temporality} {left.raw_text} {left.source_text}".lower()
    right_text = f"{right.finding} {right.temporality} {right.raw_text} {right.source_text}".lower()
    if any(token in right_text for token in ("\u540e", "\u4e4b\u540e", "\u968f\u540e")):
        return True
    if any(token in left_text for token in ("\u524d", "\u65e2\u5f80", "\u66fe")):
        return True
    if left.finding in {"post_radiotherapy_time_window", "symptom_onset_after_radiotherapy", "radiotherapy_before_symptoms"}:
        return True
    if "after" in right_text or "post" in right_text or "later" in right_text or "之后" in right_text or "后" in right_text:
        return True
    if "ago" in left_text or "prior" in left_text or "history" in left_text or "既往" in left_text or "曾" in left_text:
        return True
    return False


def _anatomical_consistency_verified(left: Optional[Observation], right: Optional[Observation]) -> bool:
    if not left or not right:
        return False
    left_text = f"{left.finding} {left.anatomy} {left.source_text}".lower()
    right_text = f"{right.finding} {right.anatomy} {right.source_text}".lower()
    incompatible_exposure_sites = ("pelvis", "pelvic", "brain", "cranial", "\u76c6\u8154", "\u8111\u90e8")
    if any(token in left_text for token in incompatible_exposure_sites):
        return False
    pulmonary_exposure_markers = (
        "thoracic",
        "thorax",
        "chest",
        "lung",
        "mediastinal",
        "breast",
        "\u80f8",
        "\u80ba",
        "\u7eb5\u9694",
        "\u4e73\u817a",
    )
    pulmonary_target_markers = ("lung", "pulmonary", "ground_glass", "\u80ba")
    if any(token in left_text for token in pulmonary_exposure_markers) and any(
        token in right_text for token in pulmonary_target_markers
    ):
        return True
    merged = f"{left.finding} {right.finding} {left.anatomy} {right.anatomy} {left.source_text} {right.source_text}".lower()
    pulmonary_markers = ("thoracic", "chest", "lung", "pulmonary", "ground_glass")
    cardiac_markers = ("cardiac", "heart", "mitral", "left_heart", "valvular")
    if any(token in merged for token in pulmonary_markers):
        return True
    if any(token in merged for token in cardiac_markers):
        return True
    return False


def _independent_group_count(source_group_ids: Dict[str, List[str]]) -> int:
    groups: set[str] = set()
    for key, values in source_group_ids.items():
        if key == "exclusion_or_contradiction":
            continue
        groups.update(str(value) for value in values or [] if str(value))
    return len(groups)


def _entities_for_family(knowledge: Any, family: Any) -> List[Any]:
    family_key = str(family or "").strip()
    if not family_key or not knowledge:
        return []
    registry = getattr(knowledge, "entity_registry", None)
    entities = list(getattr(registry, "entities_by_id", {}).values()) if registry else []
    matched = [
        entity
        for entity in entities
        if family_key in {str(getattr(entity, "family", "") or ""), str(getattr(entity, "disease_family", "") or "")}
    ]
    if matched:
        return matched[:5]
    for entity_id in _FAMILY_ENTITY_HINTS.get(family_key.strip().lower().replace(" ", "_").replace("-", "_"), [])[:5]:
        entity = registry.get(entity_id) if registry else None
        if entity:
            matched.append(entity)
    return matched[:5]


def _source_permission_ceiling(hypothesis: Dict[str, Any]) -> str:
    source = str(hypothesis.get("generator_source") or "").lower()
    tier = str(hypothesis.get("proposal_trust_tier") or "").lower()
    if source in {"thinking_structured_legacy", "reasoning_bound"} or tier == _TRUST_TIER_EVIDENCE_BOUND:
        return RECALL_BOOST
    if tier == _TRUST_TIER_QUERY_ONLY:
        return RECALL_QUERY_EXPANSION
    return RECALL_PROTECTED


def _normalize_token(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if not ch.isspace())


def _time_bucket(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "time_unknown"
    for token in ("current", "acute", "subacute", "chronic", "ago", "post", "after"):
        if token in text:
            return token
    return text[:24]


def _source_bucket(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "lab" in text or "laboratory" in text:
        return "lab"
    if "imag" in text or "ct" in text or "xray" in text:
        return "imaging"
    if "exam" in text or "clinician" in text:
        return "exam"
    if "patient" in text or "history" in text:
        return "history"
    return text or "unknown_source"


def _load_relation_registry(ref_dir: str) -> Dict[str, Any]:
    path = os.path.join(ref_dir, "clinical_patterns.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "relation_schemas": dict(payload.get("relation_schemas") or {}),
        "family_links": dict(payload.get("family_links") or {}),
    }


def _load_evidence_ontology(ref_dir: str) -> Dict[str, Any]:
    path = os.path.join(ref_dir, "evidence_ontology.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    concepts = payload.get("concepts") if isinstance(payload.get("concepts"), dict) else payload
    return dict(concepts or {})


def _normalize_schema_id(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if text == "left_sided_valvular_disease":
        return "structural_function_abnormality"
    return text


def _observation_summary(item: Observation) -> Dict[str, Any]:
    return {
        "observation_ref": _observation_ref(item),
        "canonical_concept": item.finding,
        "polarity": item.polarity or "positive",
        "source": item.source or "",
        "evidence_group_id": _evidence_group_id(item),
        "field_path": item.field_path or "",
    }


def _observation_by_ref(observations: Sequence[Observation], ref: Any) -> Optional[Observation]:
    text = str(ref or "").strip()
    if not text:
        return None
    for item in observations or []:
        if _observation_ref(item) == text:
            return item
    return None


def _ontology_concept_for_finding(finding: str, ontology: Dict[str, Any]) -> str:
    key = _normalize_token(finding)
    for concept, spec in (ontology or {}).items():
        if key == _normalize_token(concept):
            return concept
        aliases = {_normalize_token(item) for item in (spec.get("aliases") or []) if str(item)}
        children = {_normalize_token(item) for item in (spec.get("children") or []) if str(item)}
        if key in aliases or key in children:
            return concept
    return ""


def _ontology_role_hints(finding: str, ontology: Dict[str, Any]) -> List[str]:
    concept = _ontology_concept_for_finding(finding, ontology)
    if not concept:
        return []
    return [
        _ROLE_ALIASES.get(str(item or "").strip(), str(item or "").strip())
        for item in (ontology.get(concept, {}) or {}).get("role_hints", []) or []
        if str(item or "").strip()
    ]


def _role_candidates_for_observation(
    schema: str,
    item: Observation,
    ontology: Dict[str, Any],
) -> List[Tuple[str, str, float]]:
    finding = _normalize_token(item.finding)
    merged = f"{item.finding} {item.raw_text} {item.source_text} {item.anatomy}".lower()
    hints = set(_ontology_role_hints(item.finding, ontology))
    observation_type = str(getattr(item, "observation_type", "") or "").strip()
    semantic_level = str(getattr(item, "semantic_level", "") or "fact").strip() or "fact"
    result: List[Tuple[str, str, float]] = []

    def add(slot: str, rule: str, confidence: float) -> None:
        normalized = _ROLE_ALIASES.get(slot, slot)
        result.append((normalized, rule, confidence))

    if schema == "exposure_temporal_organ_injury":
        if (
            semantic_level == "fact"
            and finding == "thoracic_radiotherapy"
            and observation_type in {"", "treatment_history", "exposure"}
        ):
            add("exposure", "canonical_or_ontology_exposure", 0.96)
        if semantic_level == "fact" and (
            finding in {"dyspnea", "cough", "hypoxemia", "wheeze", "orthopnea"}
            or "organ_manifestation" in hints
            or observation_type == "symptom"
        ):
            add("organ_manifestation", "respiratory_manifestation", 0.9)
        if (
            semantic_level == "fact"
            and not finding.startswith("field:")
            and _is_pulmonary_objective_finding(finding, observation_type, merged)
        ):
            add("imaging_or_objective_finding", "pulmonary_objective_abnormality", 0.92)
        return result

    if schema in {
        "structural_function_abnormality",
        "left_sided_valvular_regurgitation",
    }:
        if (
            finding in {
                "mitral_valve_prolapse",
                "cardiac_murmur",
                "left_heart_enlargement",
                "valvular_structural_abnormality",
                "holosystolic_apical_murmur",
                "flail_leaflet",
                "leaflet_prolapse",
            }
            or "structure_or_credible_sign" in hints
        ):
            add("structure_or_credible_sign", "left_valvular_structural_or_sign", 0.92)
        if (
            finding in {
                "dyspnea",
                "pulmonary_edema",
                "heart_failure_state",
                "orthopnea",
                "acute_heart_failure",
                "pink_frothy_sputum",
            }
            or "function_impairment" in hints
            or "pink frothy" in merged
        ):
            add("function_impairment", "left_heart_functional_consequence", 0.9)
        if finding in {
            "mitral_regurgitation",
            "echo_mitral_regurgitation",
            "regurgitant_jet",
            "flail_leaflet",
            "leaflet_prolapse",
            "holosystolic_apical_murmur",
        }:
            add("regurgitation_specific", "regurgitation_specific_concept", 0.95)
        return result

    return result


def _is_pulmonary_objective_finding(
    finding: str,
    observation_type: str,
    merged_text: str,
) -> bool:
    if finding in {"pneumonia_infiltrate", "pulmonary_inflammatory_change"}:
        return False
    if finding in {
        "ground_glass_opacity",
        "pulmonary_abnormality",
        "pulmonary_infiltrative_opacity",
        "pulmonary_infiltrate",
        "lung_opacity",
        "pulmonary_consolidation",
        "interstitial_opacity",
        "atelectasis",
    }:
        return True
    if observation_type != "imaging_finding":
        return False
    pulmonary_markers = {
        "lung",
        "pulmonary",
        "chest",
        "thorax",
        "thoracic",
        "肺",
        "胸",
    }
    return any(marker in merged_text for marker in pulmonary_markers)


def _role_binding_rank(binding: EvidenceRoleBinding, observations: Sequence[Observation]) -> Tuple[float, float, float]:
    observation = _observation_by_ref(observations, binding.observation_ref)
    finding = str(getattr(observation, "finding", "") or "")
    specificity_bonus = 0 if finding.startswith("field:") else 1
    return (
        float(specificity_bonus),
        _information_value(observation) if observation else 0.0,
        float(binding.binding_confidence or 0.0),
    )


def _evaluate_constraints(
    schema: str,
    bound_slots: Dict[str, EvidenceRoleBinding],
    observations: Sequence[Observation],
) -> List[RelationConstraintResult]:
    result: List[RelationConstraintResult] = []
    if schema == "exposure_temporal_organ_injury":
        exposure = _observation_by_ref(observations, getattr(bound_slots.get("exposure"), "observation_ref", ""))
        manifestation = _observation_by_ref(
            observations,
            getattr(bound_slots.get("organ_manifestation"), "observation_ref", ""),
        )
        objective = _observation_by_ref(
            observations,
            getattr(bound_slots.get("imaging_or_objective_finding"), "observation_ref", ""),
        )
        explicit_temporal = next(
            (
                item
                for item in observations or []
                if item.polarity == "positive"
                and item.finding
                in {
                    "post_radiotherapy_time_window",
                    "symptom_onset_after_radiotherapy",
                    "radiotherapy_before_symptoms",
                }
                and not _lineage_rejection_reason(item)
            ),
            None,
        )
        temporal_status = (
            "satisfied"
            if explicit_temporal or _temporal_after_verified(exposure, manifestation)
            else "unresolved"
        )
        result.append(
            RelationConstraintResult(
                constraint_type="temporal_after",
                from_observation_ref=_observation_ref(exposure) if exposure else "",
                to_observation_ref=_observation_ref(manifestation) if manifestation else "",
                status=temporal_status,
                reason=(
                    "explicit_temporal_relation"
                    if explicit_temporal
                    else ("" if temporal_status == "satisfied" else "temporal_relation_unresolved")
                ),
            )
        )
        anatomy_status = "satisfied" if _anatomical_consistency_verified(exposure, objective) else "unresolved"
        result.append(
            RelationConstraintResult(
                constraint_type="anatomical_consistency",
                from_observation_ref=_observation_ref(exposure) if exposure else "",
                to_observation_ref=_observation_ref(objective) if objective else "",
                status=anatomy_status,
                reason="" if anatomy_status == "satisfied" else "anatomical_relation_unresolved",
            )
        )
        return result
    if schema in {"structural_function_abnormality", "left_sided_valvular_regurgitation"}:
        structure = _observation_by_ref(
            observations,
            getattr(bound_slots.get("structure_or_credible_sign"), "observation_ref", ""),
        )
        function = _observation_by_ref(observations, getattr(bound_slots.get("function_impairment"), "observation_ref", ""))
        satisfied = bool(structure and function and _evidence_group_id(structure) != _evidence_group_id(function))
        result.append(
            RelationConstraintResult(
                constraint_type="structural_function_consistency",
                from_observation_ref=_observation_ref(structure) if structure else "",
                to_observation_ref=_observation_ref(function) if function else "",
                status="satisfied" if satisfied else "unresolved",
                reason="" if satisfied else "structure_function_not_independent",
            )
        )
        return result
    return result


def _activation_score(
    required_slots: Sequence[str],
    bound_slots: Dict[str, EvidenceRoleBinding],
    constraints: Sequence[RelationConstraintResult],
) -> float:
    slot_total = max(1, len(list(required_slots or [])))
    slot_score = len([slot for slot in required_slots if slot in bound_slots]) / slot_total
    if not constraints:
        return slot_score
    constraint_score = len([item for item in constraints if item.status == "satisfied"]) / max(1, len(constraints))
    return 0.7 * slot_score + 0.3 * constraint_score


def _empty_source_audit() -> Dict[str, Dict[str, Any]]:
    return {
        key: {"input_present": False, "generated": 0, "skip_reason": "not_evaluated"}
        for key in ("structured_thinking", "reasoning_adapter", "diagnosis_draft", "deterministic_relation")
    }


def _record_source_audit(
    audit: Dict[str, Dict[str, Any]],
    source: str,
    *,
    input_present: bool,
    generated: int,
    skip_reason: str,
) -> None:
    previous = dict(audit.get(source) or {})
    audit[source] = {
        "input_present": bool(previous.get("input_present") or input_present),
        "generated": int(previous.get("generated") or 0) + int(generated or 0),
        "skip_reason": "" if (int(previous.get("generated") or 0) + int(generated or 0)) else (skip_reason or previous.get("skip_reason") or ""),
    }


def _thinking_skip_reason(snapshot: ThinkingSnapshot, source: str) -> str:
    if not snapshot:
        return "no_thinking_snapshot"
    if source == "structured_thinking":
        return "no_structured_pattern_proposals"
    if source == "reasoning_adapter":
        if not snapshot.differential_diagnoses:
            return "no_structured_differential"
        return "no_verifiable_relation"
    return "not_generated"


def _empty_compiler_audit(*, enabled: bool = True, evidence_snapshot_id: str = "") -> Dict[str, Any]:
    return {
        "compiler_enabled": bool(enabled),
        "compiler_version": "pattern_proposal_compiler_v2",
        "evidence_snapshot_id": evidence_snapshot_id,
        "sources": _empty_source_audit(),
        "proposal_count_before_dedup": 0,
        "proposal_count_after_dedup": 0,
        "proposal_count_after_budget": 0,
        "deduplicated_count": 0,
        "budget_truncated_count": 0,
    }


def _proposal_signature(hypothesis: ClinicalPatternHypothesis) -> str:
    binding_parts = sorted(
        f"{binding.evidence_id}:{binding.relation_slot}"
        for binding in hypothesis.evidence_bindings
    )
    return repr(
        (
            hypothesis.relation_schema_id or hypothesis.pattern_type,
            hypothesis.suggested_family,
            tuple(binding_parts),
        )
    )


def _expand_family_entity_hints(payload: Dict[str, Any]) -> None:
    family = str(payload.get("suggested_family") or payload.get("family_id") or "").strip()
    if not family:
        return
    normalized = family.strip().lower().replace(" ", "_").replace("-", "_")
    entity_ids = list(_FAMILY_ENTITY_HINTS.get(normalized) or [])
    if not entity_ids:
        return
    diseases = list(payload.get("suggested_diseases") or [])
    existing = {
        str((item or {}).get("canonical_id") or (item or {}).get("entity_id") or "").strip()
        for item in diseases
        if isinstance(item, dict)
    }
    for entity_id in entity_ids[:5]:
        if entity_id in existing:
            continue
        diseases.append({"name": entity_id, "canonical_id": entity_id, "hypothesis_confidence": 0.0})
    payload["suggested_diseases"] = diseases


def _normalize_relation_schema_hint(payload: Dict[str, Any]) -> None:
    if payload.get("relation_schema_id"):
        return
    text = " ".join(
        [
            str(payload.get("pattern_name") or ""),
            str(payload.get("pattern_type") or ""),
            str(payload.get("suggested_family") or payload.get("family_id") or ""),
            " ".join(
                str((item or {}).get("canonical_id") or (item or {}).get("name") or "")
                for item in payload.get("suggested_diseases") or []
                if isinstance(item, dict)
            ),
        ]
    ).lower()
    if any(token in text for token in _RADIATION_PATTERN_HINTS):
        payload["relation_schema_id"] = "exposure_temporal_organ_injury"
        payload.setdefault("suggested_family", "radiation_related_lung_injury")
        return
    if "left_sided_valvular_regurgitation" in text or "mitral" in text or "二尖瓣" in text:
        payload["relation_schema_id"] = "structural_function_abnormality"
        payload.setdefault("suggested_family", "valvular_left_heart")
        for binding in payload.get("evidence_bindings") or []:
            if not isinstance(binding, dict):
                continue
            finding = str(binding.get("evidence_id") or binding.get("id") or "")
            if finding in {"cardiac_murmur", "left_heart_enlargement", "valvular_structural_abnormality"}:
                binding["relation_slot"] = "structure_or_credible_sign"
            elif finding in {"dyspnea", "pulmonary_edema", "heart_failure_state", "orthopnea"}:
                binding["relation_slot"] = "function_impairment"
            elif finding in {"mitral_regurgitation", "echo_mitral_regurgitation", "regurgitant_jet"}:
                binding["relation_slot"] = "regurgitation_specific"
        return
    if "vascular_shunt" in text or "pulmonary_vascular" in text:
        payload["relation_schema_id"] = "vascular_shunt"


def _coerce_thinking_snapshot(value: Any) -> Optional[ThinkingSnapshot]:
    if isinstance(value, ThinkingSnapshot):
        return value
    if not isinstance(value, dict):
        return None
    fields = set(ThinkingSnapshot.__dataclass_fields__)
    payload = {key: value[key] for key in fields if key in value}
    try:
        return ThinkingSnapshot(**payload)
    except TypeError:
        return None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _stable_hash(value: Any) -> str:
    try:
        payload = repr(value)
    except Exception:
        payload = str(type(value))
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in text.split("_") if part)[:48] or "pattern"


def _normalize_relation(value: Dict[str, Any]) -> Dict[str, Any]:
    relation = dict(value)
    if "from" not in relation and "from_evidence_ref" in relation:
        relation["from"] = relation.get("from_evidence_ref")
    if "to" not in relation and "to_evidence_ref" in relation:
        relation["to"] = relation.get("to_evidence_ref")
    return relation


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0
