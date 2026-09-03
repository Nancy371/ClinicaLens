"""Evidence-first candidate scoring and final diagnosis adjudication."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .candidate_generator import CandidateGenerator, CandidatePool
from .case_board import (
    ConsultationEvidencePipeline,
    StaleJudgeDecisionError,
    evidence_snapshot_hash,
    judge_decision_is_stale,
)
from .clinical_evidence import EvidenceBundle, Observation
from .clinical_pattern_bridge import BridgePatternValidator
from .clinical_pattern_compiler import ClinicalPatternCompiler
from .diagnosis_eligibility import (
    DEFERRED,
    DIFFERENTIAL_ONLY,
    EXCLUDED,
    PRIMARY_ELIGIBLE,
    DiagnosisEligibilityGate,
)
from .disease_entity import DiseaseEntityRegistry
from .diagnosis_judge import DiagnosisJudge, DiagnosisSubmitter
from .diagnosis_resolver import DiagnosisResolution, OpenWorldDiagnosisResolver
from .evidence_conflicts import EvidenceConflictArbiter
from .mechanism_reasoner import MechanismReasoner
from .claim_resolution import AnchorEvaluator, normalize_ledger
from .pattern_hypothesis import (
    build_pattern_recall_context,
    coerce_pattern_recall_context,
    evidence_snapshot_hash as pattern_evidence_snapshot_hash,
    PatternProposalAdapter,
    PatternHypothesisVerifier,
)
from .root_cause_arbitration import RootCauseArbiter
from .submission_authorization import (
    AUTH_AUTHORIZED,
    SubmissionAuthorizationLayer,
)


_SECONDARY_MANIFESTATION_DIAGNOSES = {
    "心律失常",
    "心力衰竭",
    "肺动脉高压",
}

_GENERIC_EXPLANATORY_FINDINGS = {
    "acute_course",
    "chronic_course",
    "cough",
    "dizziness",
    "dyspnea",
    "fatigue",
    "fever",
    "pain",
    "pruritus",
    "rash",
    "visual_blurring",
    "weakness",
}

_CORE_EXPLANATORY_FINDINGS = {
    "age_related_near_blur",
    "ambiguous_genitalia",
    "anogenital_warts",
    "bradycardia",
    "cauliflower_lesions",
    "childcare_exposure",
    "crusted_exudative_skin_ulcer",
    "dark_urine",
    "deep_skin_ulcer",
    "dermatomal_vesicles",
    "distance_vision_relatively_preserved",
    "dyspnea_on_exertion",
    "exercise_intolerance",
    "fluid_retention_pattern",
    "gradual_onset",
    "hemoptysis",
    "iris_coloboma",
    "lens_dislocation",
    "midline_suprapubic_cyst",
    "midline_suprapubic_pain",
    "near_vision_difficulty",
    "night_vision_decline",
    "night_sweats",
    "nyctalopia_pattern",
    "ocular_pain",
    "ocular_redness",
    "orthopnea",
    "ovotesticular_tissue",
    "paroxysmal_nocturnal_dyspnea",
    "periorbital_edema",
    "periostitis",
    "polydipsia",
    "postprandial_nausea",
    "presbyopia_pattern",
    "refractive_correction_improves_near_vision",
    "regional_lymphadenopathy",
    "rural_child_contact",
    "sex_development_disorder",
    "tb_exposure",
    "treponema_positive",
    "treponemal_disease_pattern",
    "treponemal_serology_positive",
    "treponemal_skin_lesion",
    "tropical_exposure",
    "tuberculosis_pattern",
    "tuberculosis_exposure",
    "umbilical_discharge",
    "umbilical_mass",
    "urachal_remnant_pattern",
    "urachal_cyst_imaging",
    "worse_in_dim_light",
}

_DIAGNOSTIC_EXPLANATORY_FINDINGS = {
    "afb_positive",
    "tb_naat_positive",
    "naat_positive",
    "xpert_mtb_positive",
    "ugt1a1_positive",
    "echo_vsd",
    "ventricular_septal_defect",
    "left_to_right_shunt",
    "anca_positive",
    "mpo_anca_positive",
    "mri_cavitary_lesion",
    "cavitary_lesion",
    "urachal_cyst_imaging",
    "second_degree_av_block",
    "av_block",
    "treponema_positive",
    "treponemal_serology_positive",
    "treponemal_disease_pattern",
    "accommodation_failure_pattern",
}

_SPECIFIC_GENERIC_SUPPRESSIONS = {
    "\u80ba\u7ed3\u6838": {
        "\u80ba\u708e",
        "\u652f\u6c14\u7ba1\u80ba\u708e",
        "\u652f\u6c14\u7ba1\u708e",
    },
    "\u96c5\u53f8\u75c5": {
        "\u6e7f\u75b9",
        "\u76ae\u708e",
    },
    "\u8110\u5c3f\u7ba1\u56ca\u80bf": {
        "\u5c3f\u9053\u7efc\u5408\u5f81",
        "\u6025\u6027\u7ec6\u83cc\u6027\u524d\u5217\u817a\u708e",
    },
    "\u4e8c\u5ea6\u623f\u5ba4\u4f20\u5bfc\u963b\u6ede": {
        "\u5fc3\u5f8b\u5931\u5e38",
    },
    "\u5ba4\u95f4\u9694\u7f3a\u635f\uff08VSD\uff09": {
        "\u5148\u5929\u6027\u5fc3\u810f\u75c5",
        "\u4e09\u623f\u5fc3",
        "\u5fc3\u5185\u819c\u57ab\u7f3a\u635f",
    },
}

_GENERIC_PARENT_DIAGNOSES = {
    "\u80ba\u708e",
    "\u652f\u6c14\u7ba1\u80ba\u708e",
    "\u652f\u6c14\u7ba1\u708e",
    "\u4e0a\u547c\u5438\u9053\u611f\u67d3",
    "\u5c3f\u9053\u7efc\u5408\u5f81",
    "\u6ccc\u5c3f\u7cfb\u611f\u67d3",
    "\u6e7f\u75b9",
    "\u76ae\u708e",
    "\u5fc3\u5f8b\u5931\u5e38",
    "\u5fc3\u529b\u8870\u7aed",
    "\u9aa8\u6298",
    "\u5148\u5929\u6027\u5fc3\u810f\u75c5",
}


@dataclass
class CandidateScore:
    diagnosis: str
    score: float
    support_score: float
    source_prior: float
    explanation_score: float
    coverage_score: float
    residual_score: float
    contradiction_penalty: float
    required_met: bool
    hard_contradiction: bool
    matched_evidence: List[str] = field(default_factory=list)
    contradicted_evidence: List[str] = field(default_factory=list)
    soft_contradicted_evidence: List[str] = field(default_factory=list)
    hard_contradicted_evidence: List[str] = field(default_factory=list)
    required_gaps: List[str] = field(default_factory=list)
    residual_evidence: List[str] = field(default_factory=list)
    component_scores: Dict[str, float] = field(default_factory=dict)
    candidate_sources: List[Dict[str, Any]] = field(default_factory=list)
    diagnosis_type: str = "disease"
    parent_diagnosis: str = ""
    specificity: float = 0.5
    causal_relation_to_selected: str = ""
    differential_only: bool = False
    differential_only_reason: str = ""
    required_gap_authorized: bool = False
    required_gap_state: str = ""
    explanatory_coverage: float = 0.0
    core_explanatory_coverage: float = 0.0
    residual_evidence_score: float = 0.0
    residual_core_evidence_count: int = 0
    explained_evidence: List[str] = field(default_factory=list)
    unexplained_core_evidence: List[str] = field(default_factory=list)
    explanatory_rank_reason: str = ""
    generic_matched_evidence: List[str] = field(default_factory=list)
    core_matched_evidence: List[str] = field(default_factory=list)
    diagnostic_matched_evidence: List[str] = field(default_factory=list)
    generic_coverage_score: float = 0.0
    core_evidence_score: float = 0.0
    diagnostic_evidence_score: float = 0.0
    evidence_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_evidence_conflict: bool = False
    conflict_adjudication_exams: List[str] = field(default_factory=list)
    root_cause_coverage: float = 0.0
    explains_candidates: List[str] = field(default_factory=list)
    explained_by_root_cause: str = ""
    root_cause_role: str = ""
    root_cause_submit_as_final: bool = False
    eligibility_status: str = ""
    eligibility_reason: str = ""
    missing_required_anchors: List[str] = field(default_factory=list)
    satisfied_required_anchors: List[str] = field(default_factory=list)
    eligibility_blockers: List[str] = field(default_factory=list)
    eligibility_anchor_status: str = ""
    eligibility_anchor_policy: Dict[str, Any] = field(default_factory=dict)
    eligibility_anchor_policy_audit: Dict[str, Any] = field(default_factory=dict)
    evidence_contributions: List[Dict[str, Any]] = field(default_factory=list)
    evidence_pattern_matches: List[Dict[str, Any]] = field(default_factory=list)
    clinical_pattern_matches: List[Dict[str, Any]] = field(default_factory=list)
    derived_pattern_assertions: List[Dict[str, Any]] = field(default_factory=list)
    bridge_validation_results: List[Dict[str, Any]] = field(default_factory=list)
    bridge_protection_decisions: List[Dict[str, Any]] = field(default_factory=list)
    positive_evidence_score: float = 0.0
    evidence_specificity_score: float = 0.0
    entity_id: str = ""
    canonical_name: str = ""
    submission_name: str = ""
    raw_names: List[str] = field(default_factory=list)
    submittable: bool = True
    unresolved_high_value: bool = False
    exam_followup_authorized: bool = False
    submission_authorized: bool = False
    submission_role: str = ""
    submission_authorization: str = ""
    submission_authorization_reasons: List[str] = field(default_factory=list)
    eligibility_substatus: str = ""
    evidence_gaps: List[Dict[str, Any]] = field(default_factory=list)
    gap_values: List[Dict[str, Any]] = field(default_factory=list)
    max_gap_value: float = 0.0
    actionable_gap_count: int = 0
    deferred_priority: float = 0.0
    deferred_priority_components: Dict[str, float] = field(default_factory=dict)
    exam_priority_override: bool = False
    exam_priority_override_reason: str = ""
    deferred_priority_status: str = ""
    deferred_rounds: int = 0
    gap_closure_attempts: int = 0
    evidence_claims: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_critical_evidence_claims: List[Dict[str, Any]] = field(default_factory=list)
    claim_followup_exams: List[str] = field(default_factory=list)
    claim_verification_status: str = ""

    @property
    def trusted(self) -> bool:
        if self.eligibility_status:
            return self.eligibility_status == PRIMARY_ELIGIBLE
        return (
            self.required_met
            and not self.hard_contradiction
            and bool(self.matched_evidence)
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisDecision:
    final_diagnoses: List[str]
    trusted_diagnoses: List[str]
    candidates: List[CandidateScore]
    unexplained_evidence: List[str]
    confidence: float
    margin: float
    low_confidence: bool
    evidence_reasoning: str = ""
    entity_resolutions: List[Dict[str, Any]] = field(default_factory=list)
    name_resolutions: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_candidates: List[str] = field(default_factory=list)
    differential_only_diagnoses: List[Dict[str, Any]] = field(default_factory=list)
    pre_authorization_diagnoses: List[str] = field(default_factory=list)
    authorized_diagnoses: List[str] = field(default_factory=list)
    blocked_diagnoses: List[Dict[str, Any]] = field(default_factory=list)
    submission_override_count: int = 0
    submission_authorization_records: List[Dict[str, Any]] = field(default_factory=list)
    submission_dependency_edges: List[Dict[str, Any]] = field(default_factory=list)
    submission_authorization_bypass_count: int = 0
    associated_finding_block_count: int = 0
    authorized_primary_count: int = 0
    authorized_secondary_count: int = 0
    retriever_top1: str = ""
    judge_primary: str = ""
    submitter_final: List[str] = field(default_factory=list)
    decision_override: bool = False
    required_gap_authorized_diagnoses: List[str] = field(default_factory=list)
    judge_decision: Dict[str, Any] = field(default_factory=dict)
    open_world_candidates: List[Dict[str, Any]] = field(default_factory=list)
    mechanism_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    clinical_patterns: List[Dict[str, Any]] = field(default_factory=list)
    clinical_pattern_matches: List[Dict[str, Any]] = field(default_factory=list)
    llm_pattern_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    verified_pattern_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    rejected_pattern_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    pattern_recall_signals: List[Dict[str, Any]] = field(default_factory=list)
    pattern_recall_audit: Dict[str, Any] = field(default_factory=dict)
    pattern_candidate_admissions: List[Dict[str, Any]] = field(default_factory=list)
    pattern_driven_candidate_recall: List[Dict[str, Any]] = field(default_factory=list)
    pattern_protected_candidate_recall: List[Dict[str, Any]] = field(default_factory=list)
    pattern_gap_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    pattern_generated_active_gaps: int = 0
    unverified_pattern_leakage_count: int = 0
    pattern_expansion_round_count: int = 0
    derived_pattern_assertions: List[Dict[str, Any]] = field(default_factory=list)
    bridge_validation_results: List[Dict[str, Any]] = field(default_factory=list)
    bridge_protection_decisions: List[Dict[str, Any]] = field(default_factory=list)
    bridge_protected_candidates: List[str] = field(default_factory=list)
    retrieval_views: List[Dict[str, Any]] = field(default_factory=list)
    evidence_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    conflict_affected_diagnoses: List[str] = field(default_factory=list)
    root_cause_arbitration: Dict[str, Any] = field(default_factory=dict)
    root_cause_primary: str = ""
    root_cause_secondary: List[str] = field(default_factory=list)
    candidate_explanation_edges: List[Dict[str, Any]] = field(default_factory=list)
    eligibility_distribution: Dict[str, int] = field(default_factory=dict)
    deferred_anchor_candidates: List[str] = field(default_factory=list)
    excluded_candidates: List[str] = field(default_factory=list)
    primary_eligible_candidates: List[str] = field(default_factory=list)
    case_board: Dict[str, Any] = field(default_factory=dict)
    case_version: int = 0
    evidence_snapshot_hash: str = ""
    claim_state_version: int = 0
    diagnostic_state_version: int = 0
    knowledge_profile_version: str = ""
    decision_policy_version: str = ""
    exam_catalog_version: str = ""
    stale_decision: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_diagnoses": list(self.final_diagnoses),
            "trusted_diagnoses": list(self.trusted_diagnoses),
            "candidates": [item.to_dict() for item in self.candidates],
            "unexplained_evidence": list(self.unexplained_evidence),
            "confidence": self.confidence,
            "margin": self.margin,
            "low_confidence": self.low_confidence,
            "evidence_reasoning": self.evidence_reasoning,
            "entity_resolutions": list(self.entity_resolutions),
            "name_resolutions": list(self.name_resolutions),
            "unresolved_candidates": list(self.unresolved_candidates),
            "differential_only_diagnoses": list(self.differential_only_diagnoses),
            "pre_authorization_diagnoses": list(self.pre_authorization_diagnoses),
            "authorized_diagnoses": list(self.authorized_diagnoses),
            "blocked_diagnoses": list(self.blocked_diagnoses),
            "submission_override_count": int(self.submission_override_count),
            "submission_authorization_records": list(
                self.submission_authorization_records
            ),
            "submission_dependency_edges": list(self.submission_dependency_edges),
            "submission_authorization_bypass_count": int(
                self.submission_authorization_bypass_count or 0
            ),
            "associated_finding_block_count": int(
                self.associated_finding_block_count or 0
            ),
            "authorized_primary_count": int(self.authorized_primary_count or 0),
            "authorized_secondary_count": int(self.authorized_secondary_count or 0),
            "retriever_top1": self.retriever_top1,
            "judge_primary": self.judge_primary,
            "submitter_final": list(self.submitter_final),
            "decision_override": bool(self.decision_override),
            "required_gap_authorized_diagnoses": list(
                self.required_gap_authorized_diagnoses
            ),
            "judge_decision": dict(self.judge_decision),
            "open_world_candidates": list(self.open_world_candidates),
            "mechanism_hypotheses": list(self.mechanism_hypotheses),
            "clinical_patterns": list(self.clinical_patterns),
            "clinical_pattern_matches": list(self.clinical_pattern_matches),
            "llm_pattern_hypotheses": list(self.llm_pattern_hypotheses),
            "verified_pattern_hypotheses": list(self.verified_pattern_hypotheses),
            "rejected_pattern_hypotheses": list(self.rejected_pattern_hypotheses),
            "pattern_recall_signals": list(self.pattern_recall_signals),
            "pattern_recall_audit": dict(self.pattern_recall_audit),
            "pattern_candidate_admissions": list(self.pattern_candidate_admissions),
            "pattern_driven_candidate_recall": list(self.pattern_driven_candidate_recall),
            "pattern_protected_candidate_recall": list(self.pattern_protected_candidate_recall),
            "pattern_gap_suggestions": list(self.pattern_gap_suggestions),
            "pattern_generated_active_gaps": int(self.pattern_generated_active_gaps or 0),
            "unverified_pattern_leakage_count": int(self.unverified_pattern_leakage_count or 0),
            "pattern_expansion_round_count": int(self.pattern_expansion_round_count or 0),
            "derived_pattern_assertions": list(self.derived_pattern_assertions),
            "bridge_validation_results": list(self.bridge_validation_results),
            "bridge_protection_decisions": list(self.bridge_protection_decisions),
            "bridge_protected_candidates": list(self.bridge_protected_candidates),
            "retrieval_views": list(self.retrieval_views),
            "evidence_conflicts": list(self.evidence_conflicts),
            "conflict_affected_diagnoses": list(self.conflict_affected_diagnoses),
            "root_cause_arbitration": dict(self.root_cause_arbitration),
            "root_cause_primary": self.root_cause_primary,
            "root_cause_secondary": list(self.root_cause_secondary),
            "candidate_explanation_edges": list(self.candidate_explanation_edges),
            "eligibility_distribution": dict(self.eligibility_distribution),
            "deferred_anchor_candidates": list(self.deferred_anchor_candidates),
            "excluded_candidates": list(self.excluded_candidates),
            "primary_eligible_candidates": list(self.primary_eligible_candidates),
            "case_board": dict(self.case_board),
            "case_version": int(self.case_version or 0),
            "evidence_snapshot_hash": self.evidence_snapshot_hash,
            "claim_state_version": int(self.claim_state_version or 0),
            "diagnostic_state_version": int(self.diagnostic_state_version or 0),
            "knowledge_profile_version": self.knowledge_profile_version,
            "decision_policy_version": self.decision_policy_version,
            "exam_catalog_version": self.exam_catalog_version,
            "stale_decision": bool(self.stale_decision),
        }


class DiagnosticKnowledgeBase:
    """Load official names, controlled extensions, profiles, and evidence rules."""

    def __init__(self, ref_dir: str = "data/ref_data"):
        self.ref_dir = ref_dir
        self.catalog_path = os.path.join(ref_dir, "diseases_catalog.json")
        self.extensions_path = os.path.join(ref_dir, "submission_diagnosis_extensions.json")
        self.graph_path = os.path.join(ref_dir, "disease_graph.json")
        self.knowledge_path = os.path.join(ref_dir, "diagnostic_knowledge.json")
        self.entity_registry = DiseaseEntityRegistry(ref_dir)
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.aliases: Dict[str, str] = {}
        self.official_names: Set[str] = set()
        self.extension_names: Set[str] = set()
        self.entity_id_by_name: Dict[str, str] = {}
        self.knowledge_version = ""
        self.source_registry: Dict[str, Any] = {}
        self._load()

    @property
    def allowed_names(self) -> List[str]:
        return sorted(self.official_names | self.extension_names)

    def normalize_name(self, value: Any) -> Optional[str]:
        text = str(value or "").strip()
        if not text:
            return None
        entity = self.entity_registry.resolve(text)
        if entity and entity.submittable:
            return entity.display_name
        if text in self.entries and text in (self.official_names | self.extension_names):
            return text
        if text in self.aliases:
            return self.aliases[text]
        lowered = text.lower()
        for alias, name in self.aliases.items():
            if alias.lower() == lowered:
                return name
        # Avoid broad substring normalization except for explicit suffixes from LLM output.
        stripped = text.replace("诊断", "").replace("可能", "").strip(" ：:，,。")
        return self.aliases.get(stripped)

    def is_allowed(self, name: Any) -> bool:
        normalized = self.normalize_name(name)
        return bool(normalized and normalized in (self.official_names | self.extension_names))

    def get(self, name: Any) -> Dict[str, Any]:
        entity = self.entity_registry.get(name)
        if entity and entity.display_name in self.entries:
            return self.entries.get(entity.display_name, {})
        normalized = self.normalize_name(name) or str(name or "")
        return self.entries.get(normalized, {})

    def resolve_entity(self, value: Any):
        return self.entity_registry.resolve(value)

    def entity_id_for(self, value: Any) -> str:
        entity = self.entity_registry.get(value)
        if entity:
            return entity.entity_id
        normalized = self.normalize_name(value)
        if normalized:
            return self.entity_id_by_name.get(normalized, "")
        return ""

    def submission_name_for(self, value: Any) -> str:
        entity = self.entity_registry.get(value)
        if entity:
            return entity.display_name
        normalized = self.normalize_name(value)
        return normalized or str(value or "")

    def canonical_name_for(self, value: Any) -> str:
        entity = self.entity_registry.get(value)
        if entity:
            return entity.canonical_name
        normalized = self.normalize_name(value)
        return normalized or str(value or "")

    def is_submittable_entity(self, value: Any) -> bool:
        entity = self.entity_registry.get(value)
        if entity:
            return bool(entity.submittable)
        normalized = self.normalize_name(value)
        return bool(normalized and normalized in (self.official_names | self.extension_names))

    def get_exam_bundle(self, value: Any) -> List[str]:
        return self.entity_registry.exam_bundle_for(value)

    def get_discriminating_exam_bundle(self, value: Any) -> List[str]:
        return self.entity_registry.discriminating_exam_bundle_for(value)

    def get_treatment_protocols(self, diagnoses: Iterable[Any]) -> List[str]:
        protocols: List[str] = []
        for diagnosis in diagnoses or []:
            for item in self.get(diagnosis).get("treatment_protocol", []) or []:
                text = str(item).strip()
                if text and text not in protocols:
                    protocols.append(text)
        return protocols

    def get_contraindications(self, diagnoses: Iterable[Any]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for diagnosis in diagnoses or []:
            for item in self.get(diagnosis).get("contraindications", []) or []:
                if isinstance(item, str):
                    item = {"term": item}
                if isinstance(item, dict) and item not in result:
                    result.append(dict(item))
        return result

    def _load(self) -> None:
        catalog = _read_json(self.catalog_path, {}).get("diseases", [])
        for item in catalog:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            self.official_names.add(name)
            self.aliases[name] = name
            self.entries[name] = self._base_entry(name)

        extensions = _read_json(self.extensions_path, {}).get("extensions", [])
        extension_meta: Dict[str, Dict[str, Any]] = {}
        for item in extensions:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            self.extension_names.add(name)
            extension_meta[name] = dict(item)
            self.aliases[name] = name
            for alias in item.get("aliases", []) or []:
                if str(alias).strip():
                    self.aliases[str(alias).strip()] = name
            entry = self._base_entry(name)
            entry["parent_diagnosis"] = str(item.get("parent_catalog_name") or "")
            entry["specificity"] = float(item.get("specificity", 0.8) or 0.8)
            entry["sources"] = list(item.get("sources", []) or [])
            self.entries[name] = entry

        self._merge_disease_graph()

        profiles: Dict[str, Dict[str, Any]] = {}
        if os.path.isdir(self.ref_dir):
            for filename in sorted(os.listdir(self.ref_dir)):
                if not filename.startswith("disease_profiles") or not filename.endswith(".json"):
                    continue
                for profile in _read_json(os.path.join(self.ref_dir, filename), {}).get("profiles", []):
                    name = str(profile.get("name") or "").strip()
                    if name:
                        profiles[name] = dict(profile)

        for name, entry in list(self.entries.items()):
            profile = profiles.get(name, {})
            if profile.get("department"):
                entry["department"] = str(profile.get("department") or "")
            entry["aliases"] = list(dict.fromkeys(profile.get("aliases", []) or []))
            for alias in entry["aliases"]:
                if str(alias).strip():
                    self.aliases[str(alias).strip()] = name
            entry["discriminating_exams"] = list(
                dict.fromkeys(
                    list(profile.get("strong_verification_exams", []) or [])
                    + list(profile.get("required_exams", []) or [])
                )
            )
            entry["treatment_protocol"] = list(profile.get("treatment_principles", []) or [])
            entry["avoid_mistakes"] = list(profile.get("avoid_mistakes", []) or [])
            entry["supporting_evidence"] = self._profile_support(profile, name)
            entry["category"] = str(profile.get("category") or entry.get("category") or "")
            if profile.get("diagnosis_type"):
                entry["diagnosis_type"] = str(profile.get("diagnosis_type") or "")
            if profile.get("specificity") is not None:
                entry["specificity"] = float(profile.get("specificity") or entry.get("specificity") or 0.5)
            if profile.get("parent_diagnosis") is not None:
                entry["parent_diagnosis"] = str(profile.get("parent_diagnosis") or entry.get("parent_diagnosis") or "")
            entry["body_system"] = str(profile.get("body_system") or entry.get("body_system") or "")
            entry["disease_family"] = str(
                profile.get("disease_family")
                or profile.get("family")
                or entry.get("disease_family")
                or entry.get("family")
                or ""
            )
            entry["family"] = entry["disease_family"]
            if profile.get("required_groups"):
                entry["required_groups"] = list(profile.get("required_groups") or [])
            for key in ("causes", "caused_by", "suppress_diagnoses", "related_complications"):
                entry[key] = _dedupe_objects(
                    list(entry.get(key, []) or []) + list(profile.get(key, []) or [])
                )
            if profile.get("diagnostic_patterns"):
                entry["diagnostic_patterns"] = _dedupe_objects(
                    list(entry.get("diagnostic_patterns", []) or [])
                    + list(profile.get("diagnostic_patterns", []) or [])
                )
            entry["generalization_suppressions"] = list(
                dict.fromkeys(
                    list(entry.get("generalization_suppressions", []) or [])
                    + list(profile.get("generalization_suppressions", []) or [])
                    + list(profile.get("common_confusions", []) or [])
                    + list(profile.get("generic_suppressions", []) or [])
                )
            )
            if profile.get("negative_features"):
                entry["contradictions"] = _dedupe_objects(
                    list(entry.get("contradictions", []) or [])
                    + [
                        dict(item, hard=bool(item.get("hard", False)))
                        if isinstance(item, dict)
                        else {"terms": [str(item)], "hard": False}
                        for item in profile.get("negative_features", []) or []
                    ]
                )

        self._merge_disease_graph()

        for knowledge_payload in self._iter_knowledge_payloads():
            if knowledge_payload.get("knowledge_version"):
                self.knowledge_version = str(knowledge_payload.get("knowledge_version") or "")
            self.source_registry.update(dict(knowledge_payload.get("source_registry") or {}))
            overrides = knowledge_payload.get("diseases", [])
            for override in overrides:
                name = str(override.get("name") or "").strip()
                if name not in self.entries:
                    continue
                merged = dict(self.entries[name])
                for key, value in override.items():
                    if key == "name":
                        continue
                    if key == "supporting_evidence":
                        merged[key] = _dedupe_specs(
                            list(merged.get(key, [])) + list(value or [])
                        )
                    elif key in {
                        "treatment_protocol",
                        "contraindications",
                        "sources",
                        "contradictions",
                        "causes",
                        "caused_by",
                        "suppress_diagnoses",
                        "generalization_suppressions",
                        "diagnostic_patterns",
                        "accepted_bridge_patterns",
                    }:
                        merged[key] = _dedupe_objects(
                            list(merged.get(key, [])) + list(value or [])
                        )
                    elif key == "eligibility_anchor_policy":
                        merged[key] = dict(value or {})
                    else:
                        merged[key] = value
                self.entries[name] = merged

        self._merge_disease_graph()

        self._merge_entity_registry()
        self._apply_consultation_pattern_overlays()

        # A direct positive or negative mention is a generic evidence source for every disease.
        for name, entry in self.entries.items():
            entry["supporting_evidence"] = _dedupe_specs(
                [{"finding": f"diagnosis:{name}", "weight": 0.65}]
                + list(entry.get("supporting_evidence", []))
            )

    def _iter_knowledge_payloads(self) -> Iterable[Dict[str, Any]]:
        paths = [self.knowledge_path]
        if os.path.isdir(self.ref_dir):
            for filename in sorted(os.listdir(self.ref_dir)):
                if (
                    filename.startswith("diagnostic_knowledge_")
                    and filename.endswith(".json")
                ):
                    paths.append(os.path.join(self.ref_dir, filename))
        for path in paths:
            payload = _read_json(path, {})
            if isinstance(payload, dict):
                yield payload

    @staticmethod
    def _base_entry(name: str) -> Dict[str, Any]:
        return {
            "name": name,
            "diagnosis_type": "disease",
            "parent_diagnosis": "",
            "supporting_evidence": [],
            "required_groups": [],
            "contradictions": [],
            "discriminating_exams": [],
            "specificity": 0.5,
            "body_system": "",
            "disease_family": "",
            "family": "",
            "treatment_protocol": [],
            "contraindications": [],
            "suppress_diagnoses": [],
            "causes": [],
            "caused_by": [],
            "related_complications": [],
            "category": "",
            "generalization_suppressions": [],
            "diagnostic_patterns": [],
            "accepted_bridge_patterns": [],
            "eligibility_anchor_policy": {},
            "sources": [],
            "source_version": "",
            "department": "",
        }

    @staticmethod
    def _profile_support(profile: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for symptom in profile.get("common_symptoms", []) or []:
            text = str(symptom).strip()
            if text:
                specs.append({"terms": [text], "weight": 0.2})
        for red_flag in profile.get("red_flags", []) or []:
            text = str(red_flag).strip()
            if text:
                specs.append({"terms": [text], "weight": 0.24})
        for item in profile.get("hallmark_findings", []) or []:
            spec = _coerce_profile_evidence_spec(item, default_weight=0.55)
            if spec:
                specs.append(spec)
        for item in profile.get("discriminating_features", []) or []:
            spec = _coerce_profile_evidence_spec(item, default_weight=0.34)
            if spec:
                specs.append(spec)
        return specs

    def _merge_disease_graph(self) -> None:
        payload = _read_json(self.graph_path, {})
        nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            name = str(node.get("name") or "").strip()
            if not name or name not in self.entries:
                continue
            entry = self.entries[name]
            entry["aliases"] = _dedupe_objects(
                list(entry.get("aliases", []) or []) + list(node.get("aliases", []) or [])
            )
            self.aliases[name] = name
            for alias in entry.get("aliases", []) or []:
                text = str(alias).strip()
                if text:
                    self.aliases[text] = name
            if node.get("body_system"):
                entry["body_system"] = str(node.get("body_system") or "")
            family = str(node.get("family") or node.get("disease_family") or "")
            if family:
                entry["disease_family"] = family
                entry["family"] = family
            if node.get("specificity") is not None:
                entry["specificity"] = float(node.get("specificity") or entry.get("specificity") or 0.5)
            if node.get("parent_diagnosis") is not None:
                entry["parent_diagnosis"] = str(node.get("parent_diagnosis") or entry.get("parent_diagnosis") or "")
            for key in ("generic_suppressions", "common_confusions"):
                if node.get(key):
                    entry["generalization_suppressions"] = _dedupe_objects(
                        list(entry.get("generalization_suppressions", []) or [])
                        + list(node.get(key, []) or [])
                    )
            for key in ("related_complications", "causes", "caused_by", "suppress_diagnoses"):
                if node.get(key):
                    entry[key] = _dedupe_objects(
                        list(entry.get(key, []) or []) + list(node.get(key, []) or [])
                    )

    def _merge_entity_registry(self) -> None:
        """Overlay canonical entity metadata onto legacy disease entries."""
        for entity in self.entity_registry.entities_by_id.values():
            name = entity.display_name
            if not name:
                continue
            if entity.source_kind == "official_catalog":
                self.official_names.add(name)
            elif entity.submittable:
                self.extension_names.add(name)
            if name not in self.entries:
                self.entries[name] = self._base_entry(name)
            entry = self.entries[name]
            entry["name"] = name
            entry["entity_id"] = entity.entity_id
            entry["canonical_name"] = entity.canonical_name
            entry["submission_name"] = entity.display_name
            entry["submittable"] = bool(entity.submittable)
            entry["source_kind"] = entity.source_kind
            if entity.parent_name:
                entry["parent_diagnosis"] = entity.parent_name
            if entity.department:
                entry["department"] = entity.department
            if entity.icd10:
                entry["icd10"] = entity.icd10
            if entity.diagnosis_type:
                entry["diagnosis_type"] = entity.diagnosis_type
            if entity.body_system:
                entry["body_system"] = entity.body_system
            if entity.disease_family or entity.family:
                entry["disease_family"] = entity.disease_family or entity.family
                entry["family"] = entity.family or entity.disease_family
            entry["aliases"] = _dedupe_objects(
                list(entry.get("aliases", []) or [])
                + [entity.canonical_name, entity.display_name]
                + list(entity.aliases or [])
            )
            entry["discriminating_exams"] = list(
                dict.fromkeys(
                    list(entry.get("discriminating_exams", []) or [])
                    + list(entity.discriminating_exam_bundle or entity.exam_bundle or [])
                )
            )
            entry["entity_exam_bundle"] = list(entity.exam_bundle or [])
            entry["entity_discriminating_exam_bundle"] = list(
                entity.discriminating_exam_bundle or entity.exam_bundle or []
            )
            evidence_profile = dict(entity.evidence_profile or {})
            if evidence_profile.get("supporting_evidence"):
                entry["supporting_evidence"] = _dedupe_specs(
                    list(entry.get("supporting_evidence", []) or [])
                    + list(evidence_profile.get("supporting_evidence") or [])
                )
            if evidence_profile.get("required_groups") and not entry.get("required_groups"):
                entry["required_groups"] = list(evidence_profile.get("required_groups") or [])
            if evidence_profile.get("contradictions"):
                entry["contradictions"] = _dedupe_objects(
                    list(entry.get("contradictions", []) or [])
                    + list(evidence_profile.get("contradictions") or [])
                )
            if evidence_profile.get("diagnostic_patterns"):
                entry["diagnostic_patterns"] = _dedupe_objects(
                    list(entry.get("diagnostic_patterns", []) or [])
                    + list(evidence_profile.get("diagnostic_patterns") or [])
                )
            if evidence_profile.get("accepted_bridge_patterns"):
                entry["accepted_bridge_patterns"] = _dedupe_objects(
                    list(entry.get("accepted_bridge_patterns", []) or [])
                    + list(evidence_profile.get("accepted_bridge_patterns") or [])
                )
            if evidence_profile.get("eligibility_anchor_policy"):
                entry["eligibility_anchor_policy"] = dict(
                    evidence_profile.get("eligibility_anchor_policy") or {}
                )
            if evidence_profile.get("claim_anchor_contract"):
                entry["claim_anchor_contract"] = dict(
                    evidence_profile.get("claim_anchor_contract") or {}
                )
            if evidence_profile.get("submission_dependency_policy"):
                entry["submission_dependency_policy"] = dict(
                    evidence_profile.get("submission_dependency_policy") or {}
                )
            self.entity_id_by_name[name] = entity.entity_id
            self.aliases[name] = name
            for alias in [entity.canonical_name, entity.display_name] + list(entity.aliases or []):
                text = str(alias or "").strip()
                if text:
                    self.aliases[text] = name

    def _apply_consultation_pattern_overlays(self) -> None:
        leukemia = "\u767d\u8840\u75c5"
        bph = "\u524d\u5217\u817a\u589e\u751f"
        pavm = "\u80ba\u52a8\u9759\u8109\u7618"
        cor_triatriatum = "\u4e09\u623f\u5fc3"
        avsd = "\u5fc3\u5185\u819c\u57ab\u7f3a\u635f"
        self._overlay_entry(
            leukemia,
            supporting_evidence=[
                {"finding": "blast_present", "weight": 0.9},
                {"finding": "multilineage_cytopenia", "weight": 0.78},
                {"finding": "acute_leukemia_pattern", "weight": 0.96},
                {"finding": "hemoglobin_low", "weight": 0.45},
                {"finding": "anemia", "weight": 0.36},
                {"finding": "platelet_low", "weight": 0.45},
                {"finding": "thrombocytopenia", "weight": 0.36},
                {"finding": "white_blood_cell_abnormal", "weight": 0.42},
                {"finding": "leukocytosis", "weight": 0.32},
                {"finding": "leukopenia", "weight": 0.32},
                {"finding": "bleeding_tendency", "weight": 0.24},
                {"finding": "bone_pain", "weight": 0.24},
            ],
            discriminating_exams=[
                "\u5168\u8840\u7ec6\u80de\u8ba1\u6570\uff08CBC\uff09",
                "\u5916\u5468\u8840\u6d82\u7247",
                "\u9aa8\u9ad3\u7a7f\u523a\u548c\u6d3b\u68c0\uff08BMAB\uff09",
                "\u6d41\u5f0f\u7ec6\u80de\u672f\u514d\u75ab\u5206\u578b",
                "\u9aa8\u9ad3\u6d41\u5f0f\u7ec6\u80de\u514d\u75ab\u8868\u578b\u5206\u6790",
                "\u7ec6\u80de\u9057\u4f20\u5b66\u5206\u6790",
                "\u767d\u8840\u75c5\u878d\u5408\u57fa\u56e0\u68c0\u6d4b",
                "\u7ec4\u7ec7\u75c5\u7406\u5b66\u68c0\u67e5",
            ],
            diagnostic_patterns=[
                {
                    "pattern_id": "acute_leukemia_confirmed_pattern",
                    "pattern_type": "anchor_pattern",
                    "logic": "all_of",
                    "required": [
                        {"any_of": ["blast_present", "acute_leukemia_pattern"]},
                        {
                            "any_of": [
                                "multilineage_cytopenia",
                                "hemoglobin_low",
                                "platelet_low",
                                "white_blood_cell_abnormal",
                            ]
                        },
                    ],
                    "requires_objective_source": True,
                    "effect": {"eligibility": "PrimaryEligible"},
                },
                {
                    "pattern_id": "acute_leukemia_suspected_workup_pattern",
                    "pattern_type": "anchor_pattern",
                    "logic": "min_count",
                    "min_count": 2,
                    "required": [
                        "fever",
                        "fatigue",
                        "weakness",
                        "bleeding_tendency",
                        "bone_pain",
                        "hemoglobin_low",
                        "platelet_low",
                        "white_blood_cell_abnormal",
                    ],
                    "effect": {
                        "eligibility": "Deferred",
                        "reason": "NeedsAnchor",
                    },
                },
            ],
        )
        self._overlay_entry(
            bph,
            supporting_evidence=[
                {"finding": "prostate_enlargement", "weight": 0.42},
                {"finding": "urinary_frequency", "weight": 0.28},
                {"finding": "urinary_urgency", "weight": 0.28},
                {"finding": "nocturia", "weight": 0.28},
                {"finding": "urinary_retention", "weight": 0.38},
                {"finding": "difficulty_urinating", "weight": 0.32},
                {"finding": "weak_stream", "weight": 0.3},
                {"finding": "postvoid_residual_high", "weight": 0.35},
            ],
            diagnostic_patterns=[
                {
                    "pattern_id": "bph_luts_obstruction_pattern",
                    "pattern_type": "anchor_pattern",
                    "logic": "all_of",
                    "required": [
                        {
                            "any_of": [
                                "prostate_enlargement",
                                f"diagnosis:{bph}",
                            ]
                        },
                        {
                            "any_of": [
                                "urinary_frequency",
                                "urinary_urgency",
                                "nocturia",
                                "urinary_retention",
                                "difficulty_urinating",
                                "weak_stream",
                                "incomplete_emptying",
                                "postvoid_residual_high",
                            ]
                        },
                    ],
                    "effect": {"eligibility": "PrimaryEligible"},
                }
            ],
        )
        self._overlay_entry(
            cor_triatriatum,
            supporting_evidence=[
                {"finding": "left_atrial_membrane", "weight": 0.88},
                {"finding": "cor_triatriatum", "weight": 0.92},
                {"finding": "restrictive_fenestration", "weight": 0.72},
                {"finding": "congenital_heart_defect", "weight": 0.22},
                {"finding": "cyanosis", "weight": 0.18},
            ],
            discriminating_exams=[
                "\u8d85\u58f0\u5fc3\u52a8\u56fe",
                "\u7ecf\u98df\u7ba1\u8d85\u58f0\u5fc3\u52a8\u56fe\uff08TEE\uff09",
                "\u5fc3\u810fMRI\uff08CMR\uff09",
                "\u5fc3\u5bfc\u7ba1\u68c0\u67e5",
            ],
            diagnostic_patterns=[
                {
                    "pattern_id": "cor_triatriatum_structural_anchor_pattern",
                    "pattern_type": "anchor_pattern",
                    "logic": "all_of",
                    "required": [
                        {
                            "any_of": [
                                "left_atrial_membrane",
                                "cor_triatriatum",
                                "restrictive_fenestration",
                            ]
                        }
                    ],
                    "requires_objective_source": True,
                    "effect": {"eligibility": "PrimaryEligible"},
                }
            ],
        )
        self._overlay_entry(
            avsd,
            supporting_evidence=[
                {"finding": "atrioventricular_septal_defect", "weight": 0.9},
                {"finding": "common_atrioventricular_valve", "weight": 0.72},
                {"finding": "av_valve_regurgitation", "weight": 0.48},
                {"finding": "congenital_heart_defect", "weight": 0.22},
                {"finding": "cyanosis", "weight": 0.18},
            ],
            discriminating_exams=[
                "\u8d85\u58f0\u5fc3\u52a8\u56fe",
                "\u4e09\u7ef4\u8d85\u58f0\u5fc3\u52a8\u56fe\uff083D Echo\uff09",
                "\u7ecf\u98df\u7ba1\u8d85\u58f0\u5fc3\u52a8\u56fe\uff08TEE\uff09",
                "\u5fc3\u810fMRI\uff08CMR\uff09",
                "\u5fc3\u5bfc\u7ba1\u68c0\u67e5",
            ],
            diagnostic_patterns=[
                {
                    "pattern_id": "avsd_structural_anchor_pattern",
                    "pattern_type": "anchor_pattern",
                    "logic": "all_of",
                    "required": [
                        {
                            "any_of": [
                                "atrioventricular_septal_defect",
                                "common_atrioventricular_valve",
                            ]
                        }
                    ],
                    "requires_objective_source": True,
                    "effect": {"eligibility": "PrimaryEligible"},
                }
            ],
        )
        self._replace_diagnostic_patterns(
            pavm,
            {
                "pulmonary_avm_initial_shunt_pattern",
                "pulmonary_avm_confirmed_vascular_pattern",
            },
            [
                {
                    "pattern_id": "pulmonary_avm_initial_shunt_pattern",
                    "pattern_type": "anchor_pattern",
                    "logic": "all_of",
                    "required": [
                        {"any_of": ["hemoptysis", "hypoxemia", "cyanosis"]},
                        {
                            "any_of": [
                                "right_to_left_shunt",
                                "pulmonary_vascular_shunt",
                                "pulmonary_avm_mechanism",
                                "pulmonary_avm_imaging",
                                "pulmonary_cta_positive",
                                "enhanced_ct_vascular_malformation",
                                "bubble_echo_right_to_left_shunt",
                            ]
                        },
                    ],
                    "effect": {
                        "eligibility": "Deferred",
                        "reason": "NeedsAnchor",
                    },
                },
                {
                    "pattern_id": "pulmonary_avm_confirmed_vascular_pattern",
                    "pattern_type": "anchor_pattern",
                    "logic": "all_of",
                    "required": [
                        {"any_of": ["hemoptysis", "hypoxemia", "cyanosis"]},
                        {
                            "any_of": [
                                "right_to_left_shunt",
                                "pulmonary_vascular_shunt",
                                "pulmonary_avm_mechanism",
                                "pulmonary_avm_imaging",
                            ]
                        },
                        {
                            "any_of": [
                                "pulmonary_cta_positive",
                                "enhanced_ct_vascular_malformation",
                                "bubble_echo_right_to_left_shunt",
                                "pulmonary_av_fistula_pattern",
                            ]
                        },
                    ],
                    "requires_objective_source": True,
                    "effect": {"eligibility": "PrimaryEligible"},
                },
            ],
        )

    def _overlay_entry(
        self,
        name: str,
        *,
        supporting_evidence: Optional[Sequence[Dict[str, Any]]] = None,
        discriminating_exams: Optional[Sequence[str]] = None,
        diagnostic_patterns: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        normalized = self.normalize_name(name) or name
        entry = self.entries.get(normalized)
        if not entry:
            return
        if supporting_evidence:
            entry["supporting_evidence"] = _dedupe_specs(
                list(entry.get("supporting_evidence", []) or [])
                + list(supporting_evidence)
            )
        if discriminating_exams:
            entry["discriminating_exams"] = list(
                dict.fromkeys(
                    list(entry.get("discriminating_exams", []) or [])
                    + [str(item) for item in discriminating_exams if str(item)]
                )
            )
        if diagnostic_patterns:
            entry["diagnostic_patterns"] = _dedupe_objects(
                list(entry.get("diagnostic_patterns", []) or [])
                + list(diagnostic_patterns)
            )

    def _replace_diagnostic_patterns(
        self,
        name: str,
        pattern_ids: Set[str],
        replacements: Sequence[Dict[str, Any]],
    ) -> None:
        normalized = self.normalize_name(name) or name
        entry = self.entries.get(normalized)
        if not entry:
            return
        blocked = {str(item) for item in pattern_ids if str(item)}
        entry["diagnostic_patterns"] = _dedupe_objects(
            [
                item for item in list(entry.get("diagnostic_patterns", []) or [])
                if str((item or {}).get("pattern_id") or "") not in blocked
            ]
            + list(replacements or [])
        )


class DiagnosisDecisionEngine:
    """Score every allowed diagnosis and arbitrate a small final diagnosis set."""

    def __init__(self, config: Dict[str, Any], ref_dir: str = "data/ref_data"):
        section = config.get("diagnosis", {}) or {}
        self.trusted_threshold = float(section.get("trusted_threshold", 0.65) or 0.65)
        self.differential_threshold = float(section.get("differential_threshold", 0.35) or 0.35)
        self.margin_threshold = float(section.get("margin_threshold", 0.12) or 0.12)
        self.max_final_diagnoses = int(section.get("max_final_diagnoses", 3) or 3)
        self.candidate_limit = int(section.get("candidate_limit", 12) or 12)
        self.etiology_priority_bonus = float(
            section.get("etiology_priority_bonus", 0.08) or 0.08
        )
        self.etiology_close_margin = float(
            section.get("etiology_close_margin", 0.12) or 0.12
        )
        self.max_evidence_gap_targets = int(
            section.get("max_evidence_gap_targets", 2) or 2
        )
        self.residual_drop_threshold = float(
            section.get("residual_drop_threshold", 0.58) or 0.58
        )
        self.evidence_gap_coverage_threshold = float(
            section.get("evidence_gap_coverage_threshold", 0.32) or 0.32
        )
        self.evidence_gap_residual_threshold = float(
            section.get("evidence_gap_residual_threshold", 0.72) or 0.72
        )
        self.required_group_policy = str(section.get("required_group_policy") or "gap_only")
        self.weights = self._load_weights(section.get("weights") or {})
        self.knowledge = DiagnosticKnowledgeBase(ref_dir=ref_dir)
        self.resolver = OpenWorldDiagnosisResolver(self.knowledge, config=config)
        self.candidate_generator = CandidateGenerator(self.knowledge, self.resolver)
        self.mechanism_reasoner = MechanismReasoner()
        self.clinical_pattern_compiler = ClinicalPatternCompiler(ref_dir)
        self.pattern_hypothesis_verifier = PatternHypothesisVerifier(self.knowledge, config=config)
        self.pattern_proposal_adapter = PatternProposalAdapter(config)
        self.bridge_pattern_validator = BridgePatternValidator()
        self.judge = DiagnosisJudge(config=config, knowledge=self.knowledge)
        self.submitter = DiagnosisSubmitter(knowledge=self.knowledge)
        self.submission_authorizer = SubmissionAuthorizationLayer(
            self.knowledge,
            max_final_diagnoses=self.max_final_diagnoses,
        )
        self.conflict_arbiter = EvidenceConflictArbiter(self.knowledge)
        self.root_cause_arbiter = RootCauseArbiter(self.knowledge, ref_dir=ref_dir)
        self.eligibility_gate = DiagnosisEligibilityGate(self.knowledge)
        self.anchor_evaluator = AnchorEvaluator()
        self.consultation_pipeline = ConsultationEvidencePipeline()
        self.decision_policy_version = "judge_single_authority_v1"
        self.exam_catalog_version = "exam_resolver_v1"

    @staticmethod
    def _load_weights(configured: Dict[str, Any]) -> Dict[str, float]:
        defaults = {
            "evidence": 0.52,
            "prior": 0.10,
            "specificity": 0.0,
            "explain": 0.16,
            "exam_match": 0.06,
            "temporal": 0.03,
            "age": 0.02,
            "risk": 0.03,
            "residual": 0.10,
            "core_explain": 0.12,
            "core_residual": 0.08,
            "generic_evidence": 0.06,
            "core_evidence": 0.24,
            "diagnostic_evidence": 0.36,
            "etiology_structural": 0.07,
            "generic_parent": 0.16,
            "contradiction": 1.0,
        }
        for key, default in list(defaults.items()):
            try:
                defaults[key] = float(configured.get(key, default))
            except (AttributeError, TypeError, ValueError):
                defaults[key] = default
        defaults["specificity"] = 0.0
        return defaults

    def decide(
        self,
        llm_result: Optional[Dict[str, Any]],
        rag_chunks: Optional[Sequence[Dict[str, Any]]],
        evidence: EvidenceBundle,
        pattern_recall_context: Optional[Dict[str, Any]] = None,
    ) -> DiagnosisDecision:
        if pattern_recall_context is None:
            pattern_recall_context = self.build_pattern_recall_context(
                llm_result or {},
                evidence,
                case_id=str((llm_result or {}).get("patient_id") or (llm_result or {}).get("case_id") or ""),
            )
        pattern_recall_context = coerce_pattern_recall_context(pattern_recall_context)
        candidate_pool = self.candidate_generator.generate(
            evidence_graph=evidence.to_graph(),
            llm_result=llm_result or {},
            rag_chunks=rag_chunks or [],
            evidence=evidence,
            pattern_recall_context=pattern_recall_context,
        )
        return self.rank(
            candidate_pool,
            evidence,
            llm_result=llm_result or {},
            pattern_recall_context=pattern_recall_context,
        )

    def rank(
        self,
        candidate_pool: CandidatePool,
        evidence: EvidenceBundle,
        llm_result: Optional[Dict[str, Any]] = None,
        pattern_recall_context: Optional[Dict[str, Any]] = None,
    ) -> DiagnosisDecision:
        pattern_recall_context = coerce_pattern_recall_context(pattern_recall_context)
        case_id = str((llm_result or {}).get("patient_id") or (llm_result or {}).get("case_id") or "")
        case_board, evidence = self.consultation_pipeline.run(
            evidence,
            llm_result=llm_result or {},
            candidate_pool=candidate_pool,
            case_id=case_id,
            knowledge_profile_version=self.knowledge.knowledge_version,
            decision_policy_version=self.decision_policy_version,
            exam_catalog_version=self.exam_catalog_version,
        )
        self._attach_claim_resolution_ledger(case_board, llm_result or {})
        priors = candidate_pool.priors()
        sources_by_name = candidate_pool.sources_by_name()
        mechanism_hypotheses = list(candidate_pool.mechanism_hypotheses)
        clinical_patterns = list(candidate_pool.clinical_patterns)
        pattern_hypotheses = list(
            pattern_recall_context.get("pattern_hypotheses")
            or candidate_pool.pattern_hypotheses
            or []
        )
        pattern_verification_results = list(
            pattern_recall_context.get("pattern_verification_results")
            or candidate_pool.pattern_verification_results
            or []
        )
        pattern_recall_signals = list(
            pattern_recall_context.get("pattern_recall_signals")
            or candidate_pool.pattern_recall_signals
            or []
        )
        verified_pattern_hypotheses = [
            item
            for item in pattern_verification_results
            if isinstance(item, dict) and item.get("verification_status") == "verified"
        ]
        rejected_pattern_hypotheses = [
            item
            for item in pattern_verification_results
            if isinstance(item, dict) and item.get("verification_status") == "rejected"
        ]
        open_world_candidates = self._annotate_open_world_candidates(
            list(candidate_pool.open_world_candidates),
            mechanism_hypotheses,
        )
        retrieval_views = (
            self.clinical_pattern_compiler.retrieval_views(evidence)
            + [
                item.to_dict()
                for item in self.mechanism_reasoner.retrieval_views(evidence)
            ]
        )
        scores = []
        for name, entry in self.knowledge.entries.items():
            entity_id = str(entry.get("entity_id") or "")
            source_key = entity_id or name
            scores.append(
                self._score_entry(
                    entry,
                    max(priors.get(source_key, 0.0), priors.get(name, 0.0)),
                    evidence,
                    candidate_sources=sources_by_name.get(source_key, sources_by_name.get(name, [])),
                )
            )
        self._apply_case_board_claims(scores, case_board)
        self._apply_claim_resolution_ledger(scores, case_board)
        self._apply_competitive_specificity(scores)
        self._clear_submission_marks(scores)
        evidence_conflicts = self.conflict_arbiter.detect(
            llm_result or {},
            evidence,
            scores,
        )
        self._apply_evidence_conflicts(scores, evidence_conflicts)
        eligibility_summary = self.eligibility_gate.evaluate_all(scores, evidence)
        bridge_summary = self.bridge_pattern_validator.validate_all(
            scores,
            clinical_patterns,
            self.knowledge,
            case_version=case_board.case_version,
        )
        scores = self._sort_candidates(scores)

        trusted_pool = [
            item for item in scores
            if item.trusted and item.score >= self.trusted_threshold
        ]
        selected = self._select_final(trusted_pool)
        selected = self._append_independent_states(selected, scores)
        trusted_names = [item.diagnosis for item in selected]

        if not selected:
            supported = [
                item
                for item in scores
                if item.trusted and item.score >= self.differential_threshold
            ]
            if supported:
                selected = [supported[0]]
                selected = self._append_independent_states(selected, scores)
            else:
                fallback = self._fallback_candidate(priors, scores)
                selected = [fallback] if fallback else []

        self._annotate_causal_relations(scores, selected)
        differential_only = self.differential_only_details(scores)
        final_names = [item.diagnosis for item in selected][: self.max_final_diagnoses]
        top_score = scores[0].score if scores else 0.0
        margin = top_score - scores[1].score if len(scores) > 1 else top_score
        explained = set()
        for item in selected:
            explained.update(item.matched_evidence)
        unexplained = [
            item.finding for item in evidence.major()
            if item.finding not in explained
        ]
        unexplained = list(dict.fromkeys(unexplained))[:12]
        low_confidence = (
            not trusted_names
            or top_score < self.trusted_threshold
            or margin < self.margin_threshold
            or bool(unexplained)
            or any(item.hard_contradiction for item in selected)
            or bool(self._strong_open_world_contenders(open_world_candidates))
        )
        reasoning = self._reasoning(selected, unexplained)
        open_world_reason = self._open_world_reasoning(open_world_candidates, mechanism_hypotheses)
        if open_world_reason:
            reasoning = (reasoning.rstrip() + " " + open_world_reason).strip()
        decision = DiagnosisDecision(
            final_diagnoses=final_names,
            trusted_diagnoses=trusted_names,
            # Keep all scored diagnoses in the local audit object. Prompt and
            # log renderers apply their own display limits, while Critic and
            # replay can still inspect a low-ranked hard contradiction.
            candidates=scores,
            unexplained_evidence=unexplained,
            confidence=round(top_score, 4),
            margin=round(margin, 4),
            low_confidence=low_confidence,
            evidence_reasoning=reasoning,
            entity_resolutions=self._entity_resolution_audit(candidate_pool, scores),
            name_resolutions=list(candidate_pool.name_resolutions),
            unresolved_candidates=list(candidate_pool.unresolved_candidates),
            differential_only_diagnoses=differential_only,
            claim_state_version=int(getattr(case_board, "claim_state_version", 0) or 0),
            diagnostic_state_version=int(getattr(case_board, "diagnostic_state_version", 0) or 0),
            open_world_candidates=open_world_candidates,
            mechanism_hypotheses=mechanism_hypotheses,
            clinical_patterns=clinical_patterns,
            llm_pattern_hypotheses=pattern_hypotheses,
            verified_pattern_hypotheses=verified_pattern_hypotheses,
            rejected_pattern_hypotheses=rejected_pattern_hypotheses,
            pattern_recall_signals=pattern_recall_signals,
            pattern_recall_audit=dict(pattern_recall_context.get("pattern_recall_audit") or {}),
            pattern_candidate_admissions=list(
                getattr(candidate_pool, "pattern_candidate_admissions", []) or []
            ),
            pattern_driven_candidate_recall=list(
                pattern_recall_context.get("pattern_driven_candidate_recall") or []
            ),
            pattern_protected_candidate_recall=list(
                pattern_recall_context.get("pattern_protected_candidate_recall") or []
            ),
            pattern_gap_suggestions=list(
                pattern_recall_context.get("pattern_gap_suggestions") or []
            ),
            pattern_generated_active_gaps=int(
                pattern_recall_context.get("pattern_generated_active_gaps") or 0
            ),
            unverified_pattern_leakage_count=int(
                pattern_recall_context.get("unverified_pattern_leakage_count") or 0
            ),
            pattern_expansion_round_count=int(
                pattern_recall_context.get("pattern_expansion_round_count") or 0
            ),
            clinical_pattern_matches=self._candidate_bridge_records(
                scores,
                "clinical_pattern_matches",
            ),
            derived_pattern_assertions=self._candidate_bridge_records(
                scores,
                "derived_pattern_assertions",
            ),
            bridge_validation_results=list(
                bridge_summary.get("bridge_validation_results") or []
            ),
            bridge_protection_decisions=self._candidate_bridge_records(
                scores,
                "bridge_protection_decisions",
            ),
            bridge_protected_candidates=list(
                bridge_summary.get("bridge_protected_candidates") or []
            ),
            retrieval_views=retrieval_views,
            evidence_conflicts=evidence_conflicts,
            conflict_affected_diagnoses=[
                str(item.get("affected_diagnosis") or "")
                for item in evidence_conflicts
                if str(item.get("affected_diagnosis") or "")
            ],
            eligibility_distribution=dict(
                eligibility_summary.get("eligibility_distribution") or {}
            ),
            deferred_anchor_candidates=list(
                eligibility_summary.get("deferred_anchor_candidates") or []
            ),
            excluded_candidates=list(
                eligibility_summary.get("excluded_candidates") or []
            ),
            primary_eligible_candidates=list(
                eligibility_summary.get("primary_eligible_candidates") or []
            ),
            case_board=case_board.to_dict(),
            case_version=case_board.case_version,
            evidence_snapshot_hash=case_board.evidence_snapshot_hash,
            knowledge_profile_version=case_board.knowledge_profile_version,
            decision_policy_version=case_board.decision_policy_version,
            exam_catalog_version=case_board.exam_catalog_version,
        )
        self.judge_and_submit(decision)
        return decision

    @staticmethod
    def _apply_evidence_conflicts(
        scores: Sequence[CandidateScore],
        evidence_conflicts: Sequence[Dict[str, Any]],
    ) -> None:
        if not evidence_conflicts:
            return
        by_name = {item.diagnosis: item for item in scores}
        by_entity = {item.entity_id: item for item in scores if getattr(item, "entity_id", "")}
        for conflict in evidence_conflicts:
            diagnosis = str(conflict.get("affected_diagnosis") or "").strip()
            entity_id = str(conflict.get("entity_id") or "").strip()
            candidate = by_entity.get(entity_id) if entity_id else None
            candidate = candidate or by_name.get(diagnosis)
            if candidate is None:
                continue
            candidate.evidence_conflicts.append(dict(conflict))
            if str(conflict.get("status") or "unresolved") != "resolved":
                candidate.unresolved_evidence_conflict = True
            exams = list(candidate.conflict_adjudication_exams)
            for exam in conflict.get("adjudication_exams") or []:
                text = str(exam or "").strip()
                if text and text not in exams:
                    exams.append(text)
            candidate.conflict_adjudication_exams = exams

    def _apply_case_board_claims(
        self,
        scores: Sequence[CandidateScore],
        case_board: Any,
    ) -> None:
        if not scores or not case_board:
            return
        latest_claims: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for event in getattr(case_board, "events", []) or []:
            event_type = str(getattr(event, "event_type", "") or "")
            if event_type not in {"evidence_claim", "evidence_claim_verification"}:
                continue
            payload = dict(getattr(event, "payload", {}) or {})
            claim_id = str(payload.get("claim_id") or "").strip()
            target = str(payload.get("target_evidence") or "").strip()
            hypothesis = str(payload.get("diagnosis_hypothesis") or "").strip()
            if not claim_id or not target:
                continue
            latest_claims[(hypothesis, claim_id, target)] = payload
        if not latest_claims:
            return

        for claim in latest_claims.values():
            targets = self._claim_target_candidates(scores, claim)
            for candidate in targets:
                self._append_candidate_claim(candidate, claim)

    @staticmethod
    def _attach_claim_resolution_ledger(case_board: Any, llm_result: Dict[str, Any]) -> None:
        if not case_board or not isinstance(llm_result, dict):
            return
        ledger = normalize_ledger(llm_result.get("_claim_resolution_ledger") or {})
        if not ledger:
            return
        claim_state_version = int(llm_result.get("_claim_state_version") or 0)
        diagnostic_state_version = int(llm_result.get("_diagnostic_state_version") or 0)
        setattr(case_board, "claim_resolution_ledger", ledger)
        setattr(case_board, "claim_state_version", claim_state_version)
        setattr(case_board, "diagnostic_state_version", diagnostic_state_version)
        try:
            case_board.append_event(
                "claim_resolution_ledger",
                "case_board",
                {
                    "claim_resolution_ledger": ledger,
                    "claim_state_version": claim_state_version,
                    "diagnostic_state_version": diagnostic_state_version,
                },
                created_at_stage="claim_resolution_persistence",
            )
        except Exception:
            # Ledger is already on the board; event emission is audit-only.
            return

    def _apply_claim_resolution_ledger(
        self,
        scores: Sequence[CandidateScore],
        case_board: Any,
    ) -> None:
        if not scores or not case_board:
            return
        ledger = normalize_ledger(
            getattr(case_board, "claim_resolution_ledger", None)
            or (case_board.view().get("claim_resolution_ledger") if hasattr(case_board, "view") else {})
            or {}
        )
        if not ledger:
            return
        for candidate in scores or []:
            entity_id = str(getattr(candidate, "entity_id", "") or "").strip()
            if not entity_id:
                continue
            entry = self.knowledge.get(str(getattr(candidate, "diagnosis", "") or ""))
            contract = dict(entry.get("claim_anchor_contract") or {})
            if not contract:
                continue
            evaluation = self.anchor_evaluator.evaluate(
                entity_id=entity_id,
                anchor_contract=contract,
                ledger=ledger,
                previous_status=str(getattr(candidate, "eligibility_anchor_status", "") or ""),
            )
            setattr(candidate, "claim_resolution_ledger", ledger)
            setattr(candidate, "claim_anchor_contract", contract)
            setattr(candidate, "claim_anchor_evaluation", evaluation)
            setattr(candidate, "claim_resolution_status", evaluation.get("anchor_status_after"))

    def _claim_target_candidates(
        self,
        scores: Sequence[CandidateScore],
        claim: Dict[str, Any],
    ) -> List[CandidateScore]:
        hypothesis = str(claim.get("diagnosis_hypothesis") or "").strip()
        if not hypothesis:
            return []
        normalized = self.knowledge.normalize_name(hypothesis) if self.knowledge else None
        entity_id = self.knowledge.entity_id_for(hypothesis) if self.knowledge else ""
        candidates: List[CandidateScore] = []
        lower_hypothesis = hypothesis.lower()
        for candidate in scores or []:
            candidate_keys = self._claim_candidate_keys(candidate)
            if entity_id and entity_id == str(getattr(candidate, "entity_id", "") or ""):
                candidates.append(candidate)
                continue
            if normalized and normalized in candidate_keys:
                candidates.append(candidate)
                continue
            if self._hypothesis_mentions_candidate(lower_hypothesis, candidate, candidate_keys):
                candidates.append(candidate)
        result: List[CandidateScore] = []
        seen: set[int] = set()
        for candidate in candidates:
            marker = id(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            result.append(candidate)
        return result

    def _claim_candidate_keys(self, candidate: CandidateScore) -> set[str]:
        keys = {
            str(getattr(candidate, "diagnosis", "") or ""),
            str(getattr(candidate, "canonical_name", "") or ""),
            str(getattr(candidate, "submission_name", "") or ""),
            str(getattr(candidate, "entity_id", "") or ""),
        }
        keys.update(str(item or "") for item in getattr(candidate, "raw_names", []) or [])
        if self.knowledge:
            entry = self.knowledge.get(str(getattr(candidate, "diagnosis", "") or ""))
            keys.update(str(item or "") for item in entry.get("aliases", []) or [])
        return {item.strip() for item in keys if item and item.strip()}

    @staticmethod
    def _hypothesis_mentions_candidate(
        lower_hypothesis: str,
        candidate: CandidateScore,
        candidate_keys: set[str],
    ) -> bool:
        if not lower_hypothesis:
            return False
        for key in candidate_keys:
            text = str(key or "").strip()
            if not text or len(text) < 2:
                continue
            lower_key = text.lower()
            if lower_key in lower_hypothesis or lower_hypothesis == lower_key:
                return True
        entity_id = str(getattr(candidate, "entity_id", "") or "")
        target = str((getattr(candidate, "canonical_name", "") or getattr(candidate, "diagnosis", "")) or "")
        target_lower = target.lower()
        if entity_id == "D000025" and any(token in lower_hypothesis for token in ("leukemia", " aml", " all")):
            return True
        if entity_id == "D100055" and any(token in lower_hypothesis for token in ("pavm", "pulmonary avm")):
            return True
        if "leukemia" in lower_hypothesis and "leukemia" in target_lower:
            return True
        return False

    @staticmethod
    def _append_candidate_claim(candidate: CandidateScore, claim: Dict[str, Any]) -> None:
        payload = dict(claim)
        existing = list(getattr(candidate, "evidence_claims", []) or [])
        key = (
            str(payload.get("claim_id") or ""),
            str(payload.get("target_evidence") or ""),
        )
        replaced = False
        for index, item in enumerate(existing):
            current_key = (
                str(item.get("claim_id") or ""),
                str(item.get("target_evidence") or ""),
            )
            if current_key == key:
                existing[index] = payload
                replaced = True
                break
        if not replaced:
            existing.append(payload)
        candidate.evidence_claims = existing

        status = str(payload.get("status") or "Unresolved")
        importance = str(payload.get("importance") or "")
        if status not in {"Verified", "Derived"} and importance == "critical":
            unresolved = list(getattr(candidate, "unresolved_critical_evidence_claims", []) or [])
            if not any(
                str(item.get("claim_id") or "") == str(payload.get("claim_id") or "")
                for item in unresolved
            ):
                unresolved.append(payload)
            candidate.unresolved_critical_evidence_claims = unresolved
            exam = str(payload.get("recommended_exam") or "").strip()
            if exam:
                exams = list(getattr(candidate, "claim_followup_exams", []) or [])
                if exam not in exams:
                    exams.append(exam)
                candidate.claim_followup_exams = exams
            candidate.claim_verification_status = "unresolved_critical_claims"
            candidate.exam_followup_authorized = True
        elif existing:
            unresolved_left = [
                item for item in existing
                if str(item.get("status") or "Unresolved") not in {"Verified", "Derived"}
                and str(item.get("importance") or "") == "critical"
            ]
            candidate.unresolved_critical_evidence_claims = unresolved_left
            candidate.claim_verification_status = (
                "unresolved_critical_claims" if unresolved_left else "claims_verified_or_noncritical"
            )

    def _annotate_open_world_candidates(
        self,
        open_world_candidates: Sequence[Dict[str, Any]],
        mechanism_hypotheses: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not open_world_candidates:
            return []
        sources_by_identity: Dict[str, set[str]] = {}
        for item in open_world_candidates or []:
            if not isinstance(item, dict):
                continue
            identity = self._open_world_identity(item)
            if not identity:
                continue
            sources_by_identity.setdefault(identity, set()).add(str(item.get("source") or ""))
        mechanism_ids = {
            str(item.get("mechanism_id") or "")
            for item in mechanism_hypotheses or []
            if isinstance(item, dict) and str(item.get("mechanism_id") or "")
        }
        result: List[Dict[str, Any]] = []
        for raw in open_world_candidates:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            if bool(item.get("submittable", False)):
                item["open_world_status"] = "ResolvedSubmittableEntity"
                result.append(item)
                continue
            identity = self._open_world_identity(item)
            metadata = item.get("metadata") or {}
            sources = sources_by_identity.get(identity, set())
            stable_identity = bool(identity)
            high_specific = bool(
                item.get("evidence_links")
                or metadata.get("mechanism_id")
                or metadata.get("unreviewed_external")
                or (metadata.get("mechanism_id") in mechanism_ids)
            )
            actionable = bool(
                item.get("recommended_exams")
                or metadata.get("recommended_exams")
                or self.knowledge.get_discriminating_exam_bundle(
                    item.get("entity_id")
                    or item.get("canonical_name")
                    or item.get("raw_name")
                    or item.get("name")
                )
            )
            independent = len({source for source in sources if source}) >= 2
            if independent and high_specific and actionable and stable_identity:
                item["open_world_status"] = "UnresolvedHighValue"
                item["blocks_low_confidence_lock"] = True
            else:
                item["open_world_status"] = "OpenWorldDifferential"
                item["blocks_low_confidence_lock"] = False
            item["independent_source_count"] = len({source for source in sources if source})
            item["stable_canonical_identity"] = stable_identity
            item["high_specific_signal"] = high_specific
            item["actionable_exam_available"] = actionable
            result.append(item)
        return result

    @staticmethod
    def _open_world_identity(item: Dict[str, Any]) -> str:
        return str(
            item.get("entity_id")
            or item.get("canonical_name")
            or item.get("submission_name")
            or item.get("raw_name")
            or item.get("name")
            or ""
        ).strip()

    @staticmethod
    def _candidate_bridge_records(
        candidates: Sequence[CandidateScore],
        attribute: str,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates or []:
            diagnosis = str(getattr(candidate, "diagnosis", "") or "")
            entity_id = str(getattr(candidate, "entity_id", "") or "")
            for item in getattr(candidate, attribute, []) or []:
                if not isinstance(item, dict):
                    continue
                record = dict(item)
                record.setdefault("candidate", diagnosis)
                record.setdefault("entity_id", entity_id)
                marker = "|".join(
                    [
                        attribute,
                        str(record.get("candidate") or ""),
                        str(record.get("entity_id") or ""),
                        str(record.get("assertion_id") or record.get("match_id") or record.get("source_assertion_id") or record.get("pattern_id") or record),
                    ]
                )
                if marker in seen:
                    continue
                seen.add(marker)
                records.append(record)
        return records

    def _entity_resolution_audit(
        self,
        candidate_pool: CandidatePool,
        scores: Sequence[CandidateScore],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()

        def add_record(raw_name: Any, entity_id: Any, canonical_name: Any, submission_name: Any, source: Any, submittable: Any) -> None:
            raw = str(raw_name or "").strip()
            eid = str(entity_id or "").strip()
            canonical = str(canonical_name or "").strip()
            submission = str(submission_name or canonical).strip()
            if not (raw or eid or canonical or submission):
                return
            key = (raw, eid, str(source or ""))
            if key in seen:
                return
            seen.add(key)
            records.append(
                {
                    "raw_name": raw,
                    "entity_id": eid,
                    "canonical_name": canonical,
                    "submission_name": submission,
                    "source": str(source or ""),
                    "submittable": bool(submittable),
                }
            )

        for resolution in candidate_pool.name_resolutions or []:
            add_record(
                resolution.get("raw_name"),
                resolution.get("entity_id"),
                resolution.get("canonical_name"),
                resolution.get("submission_name"),
                "resolver",
                resolution.get("submittable"),
            )
        for item in candidate_pool.items or []:
            add_record(
                item.raw_name,
                item.entity_id,
                item.canonical_name,
                item.submission_name,
                item.source,
                item.submittable,
            )
        for item in candidate_pool.open_world_candidates or []:
            add_record(
                item.get("raw_name"),
                item.get("entity_id"),
                item.get("canonical_name"),
                item.get("submission_name"),
                item.get("source"),
                item.get("submittable"),
            )
        for score in scores or []:
            if not getattr(score, "entity_id", ""):
                continue
            add_record(
                score.diagnosis,
                score.entity_id,
                score.canonical_name,
                score.submission_name,
                "scored_candidate",
                score.submittable,
            )
        return records

    def judge_and_submit(self, decision: DiagnosisDecision) -> DiagnosisDecision:
        """Run the replayable judge and submitter over an existing decision."""
        if not decision:
            return decision
        self._ensure_decision_metadata(decision)
        if not any(
            str(getattr(item, "eligibility_status", "") or "")
            for item in decision.candidates or []
        ):
            eligibility_summary = self.eligibility_gate.evaluate_all(
                decision.candidates,
                None,
            )
            decision.eligibility_distribution = dict(
                eligibility_summary.get("eligibility_distribution") or {}
            )
            decision.deferred_anchor_candidates = list(
                eligibility_summary.get("deferred_anchor_candidates") or []
            )
            decision.excluded_candidates = list(
                eligibility_summary.get("excluded_candidates") or []
            )
            decision.primary_eligible_candidates = list(
                eligibility_summary.get("primary_eligible_candidates") or []
            )
        judge_decision = self.judge.judge(
            decision.candidates,
            preselected=decision.final_diagnoses,
            max_final_diagnoses=self.max_final_diagnoses,
        )
        self._bind_judge_metadata(decision, judge_decision)
        root_cause = self.root_cause_arbiter.arbitrate(
            judge_decision,
            decision.candidates,
            mechanism_hypotheses=decision.mechanism_hypotheses,
            max_final_diagnoses=self.max_final_diagnoses,
        )
        self.judge.apply_root_cause_arbitration(
            judge_decision,
            root_cause,
            max_final_diagnoses=self.max_final_diagnoses,
        )
        self._bind_judge_metadata(decision, judge_decision)
        self.submitter.apply(decision, judge_decision)
        self.authorize_final_diagnoses(
            decision,
            decision.pre_authorization_diagnoses or decision.final_diagnoses,
            respect_differential_only=True,
        )
        return decision

    def _ensure_decision_metadata(self, decision: DiagnosisDecision) -> None:
        if not getattr(decision, "case_version", 0):
            decision.case_version = 1
        if not getattr(decision, "evidence_snapshot_hash", ""):
            decision.evidence_snapshot_hash = evidence_snapshot_hash(EvidenceBundle([]))
        if not getattr(decision, "knowledge_profile_version", ""):
            decision.knowledge_profile_version = self.knowledge.knowledge_version
        if not getattr(decision, "decision_policy_version", ""):
            decision.decision_policy_version = self.decision_policy_version
        if not getattr(decision, "exam_catalog_version", ""):
            decision.exam_catalog_version = self.exam_catalog_version

    @staticmethod
    def _bind_judge_metadata(
        decision: DiagnosisDecision,
        judge_decision: Any,
    ) -> None:
        if not decision or not judge_decision:
            return
        for field_name in (
            "case_version",
            "evidence_snapshot_hash",
            "knowledge_profile_version",
            "decision_policy_version",
            "exam_catalog_version",
        ):
            setattr(judge_decision, field_name, getattr(decision, field_name, ""))
        for field_name in (
            "clinical_pattern_matches",
            "derived_pattern_assertions",
            "bridge_validation_results",
            "bridge_protection_decisions",
            "bridge_protected_candidates",
        ):
            setattr(judge_decision, field_name, list(getattr(decision, field_name, []) or []))

    @staticmethod
    def _decision_judge_stale(decision: DiagnosisDecision) -> bool:
        judge_payload = getattr(decision, "judge_decision", None)
        if not judge_payload:
            return False
        return judge_decision_is_stale(decision, judge_payload)

    def filter_final_diagnoses(
        self,
        diagnosis_names: Sequence[str],
        scores: Sequence[CandidateScore],
        respect_differential_only: bool = False,
    ) -> List[CandidateScore]:
        if not respect_differential_only:
            self._clear_submission_marks(scores)
        score_by_name = {item.diagnosis: item for item in scores}
        score_by_entity = {item.entity_id: item for item in scores if getattr(item, "entity_id", "")}
        pool: List[CandidateScore] = []
        for name in dict.fromkeys(str(item).strip() for item in diagnosis_names if str(item).strip()):
            entity_id = self.knowledge.entity_id_for(name)
            candidate = (score_by_entity.get(entity_id) if entity_id else None) or score_by_name.get(name)
            if candidate is None:
                continue
            if candidate.eligibility_status != PRIMARY_ELIGIBLE:
                continue
            if candidate.hard_contradiction:
                continue
            if respect_differential_only and candidate.differential_only:
                continue
            if not candidate.submittable:
                continue
            pool.append(candidate)
        selected = self._select_final(self._sort_candidates(pool))
        selected = self._append_independent_states(selected, scores)
        self._annotate_causal_relations(scores, selected)
        return selected[: self.max_final_diagnoses]

    def authorize_final_diagnoses(
        self,
        decision: DiagnosisDecision,
        requested_names: Optional[Sequence[str]] = None,
        respect_differential_only: bool = True,
    ) -> DiagnosisDecision:
        """Apply the final deterministic submission gate.

        Ranking decides what is clinically plausible. Authorization decides what is
        allowed to enter the strict final `diagnosis` payload.
        """
        if not decision:
            return decision
        if self._decision_judge_stale(decision):
            pre_names = list(
                dict.fromkeys(
                    str(item).strip()
                    for item in (
                        requested_names
                        if requested_names is not None
                        else decision.final_diagnoses
                    )
                    if str(item).strip()
                )
            )
            decision.stale_decision = True
            decision.pre_authorization_diagnoses = pre_names
            decision.authorized_diagnoses = []
            decision.final_diagnoses = []
            decision.trusted_diagnoses = []
            decision.confidence = 0.0
            decision.blocked_diagnoses = [
                {
                    "diagnosis": name,
                    "reason": "stale judge decision for current evidence snapshot",
                }
                for name in pre_names
            ]
            decision.submission_override_count = len(pre_names)
            return decision

        judge_decision = dict(getattr(decision, "judge_decision", {}) or {})
        judge_status = str(judge_decision.get("primary_status") or "").strip()
        if judge_status and (
            judge_status != "locked"
            or bool(judge_decision.get("needs_discriminating_exams", False))
        ):
            pre_names = list(
                dict.fromkeys(
                    str(item).strip()
                    for item in (
                        requested_names
                        if requested_names is not None
                        else decision.final_diagnoses
                    )
                    if str(item).strip()
                )
            )
            reason = "judge decision deferred pending discriminating exams"
            decision.pre_authorization_diagnoses = pre_names
            decision.authorized_diagnoses = []
            decision.final_diagnoses = []
            decision.trusted_diagnoses = []
            decision.confidence = 0.0
            decision.blocked_diagnoses = [
                self._authorization_block_record(
                    name,
                    self._candidate_for_name(decision.candidates, name),
                    reason,
                )
                for name in pre_names
            ]
            for candidate in decision.candidates or []:
                candidate.submission_authorized = False
                if (
                    candidate.eligibility_status == DEFERRED
                    or candidate.diagnosis in set(decision.deferred_anchor_candidates)
                    or getattr(candidate, "unresolved_critical_evidence_claims", None)
                ):
                    candidate.exam_followup_authorized = True
            decision.submission_override_count = len(pre_names)
            decision.differential_only_diagnoses = self.differential_only_details(
                decision.candidates
            )
            return decision

        existing_blocked = list(decision.blocked_diagnoses or [])
        pre_names = list(
            dict.fromkeys(
                str(item).strip()
                for item in (
                    requested_names
                    if requested_names is not None
                    else decision.final_diagnoses
                )
                if str(item).strip()
            )
        )
        if not pre_names:
            pre_names = list(decision.final_diagnoses or [])

        authorization = self.submission_authorizer.authorize(
            decision,
            pre_names,
            policy=self,
            respect_differential_only=respect_differential_only,
        )
        blocked = list(authorization.blocked_diagnoses or [])
        if not authorization.authorized_candidates:
            if not blocked and not pre_names:
                blocked = existing_blocked
            for candidate in decision.candidates or []:
                candidate.submission_authorized = False
                candidate.submission_role = ""
                candidate.submission_authorization = ""
                candidate.submission_authorization_reasons = []
                if candidate.eligibility_status == DEFERRED or candidate.diagnosis in set(decision.deferred_anchor_candidates):
                    candidate.exam_followup_authorized = True
            decision.pre_authorization_diagnoses = list(authorization.pre_authorization_diagnoses)
            decision.authorized_diagnoses = []
            decision.blocked_diagnoses = blocked
            decision.submission_authorization_records = authorization.record_dicts()
            decision.submission_dependency_edges = authorization.edge_dicts()
            decision.submission_authorization_bypass_count = int(
                authorization.submission_authorization_bypass_count or 0
            )
            decision.associated_finding_block_count = int(
                authorization.associated_finding_block_count or 0
            )
            decision.authorized_primary_count = 0
            decision.authorized_secondary_count = 0
            decision.submission_override_count = len(pre_names)
            decision.final_diagnoses = []
            decision.trusted_diagnoses = []
            decision.confidence = 0.0
            decision.differential_only_diagnoses = self.differential_only_details(
                decision.candidates
            )
            self._annotate_causal_relations(decision.candidates, [])
            return decision

        authorized: List[CandidateScore] = list(authorization.authorized_candidates)
        authorized_names = list(authorization.authorized_diagnoses)
        for candidate in decision.candidates or []:
            candidate.submission_authorized = False
            candidate.submission_role = ""
            candidate.submission_authorization = ""
            candidate.submission_authorization_reasons = []
            if candidate.eligibility_status == DEFERRED or candidate.diagnosis in set(decision.deferred_anchor_candidates):
                candidate.exam_followup_authorized = True
        records_by_name = {
            str(item.get("diagnosis_name") or ""): item
            for item in authorization.record_dicts()
        }
        for candidate in decision.candidates or []:
            record = records_by_name.get(candidate.diagnosis)
            if not record:
                continue
            candidate.submission_role = str(record.get("submission_role") or "")
            candidate.submission_authorization = str(
                record.get("submission_authorization") or ""
            )
            candidate.submission_authorization_reasons = list(
                record.get("reason_codes") or []
            )
            candidate.submission_authorized = (
                candidate.submission_authorization == AUTH_AUTHORIZED
            )
        requested_set = set(pre_names)
        authorized_set = set(authorized_names)
        decision.pre_authorization_diagnoses = list(authorization.pre_authorization_diagnoses)
        decision.authorized_diagnoses = authorized_names
        decision.blocked_diagnoses = blocked
        decision.submission_authorization_records = authorization.record_dicts()
        decision.submission_dependency_edges = authorization.edge_dicts()
        decision.submission_authorization_bypass_count = int(
            authorization.submission_authorization_bypass_count or 0
        )
        decision.associated_finding_block_count = int(
            authorization.associated_finding_block_count or 0
        )
        decision.authorized_primary_count = int(authorization.authorized_primary_count or 0)
        decision.authorized_secondary_count = int(authorization.authorized_secondary_count or 0)
        decision.submission_override_count = max(
            0,
            len(requested_set.symmetric_difference(authorized_set)),
        )
        decision.final_diagnoses = authorized_names
        decision.trusted_diagnoses = [
            item.diagnosis
            for item in authorized
            if item.score >= self.trusted_threshold or item.trusted
        ]
        decision.confidence = authorized[0].score if authorized else 0.0
        self._annotate_causal_relations(decision.candidates, authorized)
        decision.differential_only_diagnoses = self.differential_only_details(
            decision.candidates
        )
        return decision

    def _authorization_ineligible_reason(
        self,
        candidate: Optional[CandidateScore],
        respect_differential_only: bool = True,
        decision: Optional[DiagnosisDecision] = None,
    ) -> str:
        if candidate is None:
            return "not present in evidence-first candidate table"
        if not bool(getattr(candidate, "submittable", True)):
            return "entity is not submittable"
        if respect_differential_only and candidate.differential_only:
            return candidate.differential_only_reason or "differential only"
        if candidate.hard_contradiction:
            return "hard contradiction present"
        if getattr(candidate, "unresolved_evidence_conflict", False):
            return "unresolved reasoning-structured evidence conflict"
        if bool(getattr(candidate, "required_gap_authorized", False)):
            return "required_gap_authorized is audit-only, not submission authorization"
        status = str(getattr(candidate, "eligibility_status", "") or "")
        if status and status != PRIMARY_ELIGIBLE:
            if status == DEFERRED:
                return "candidate deferred pending required anchor evidence"
            if status == EXCLUDED:
                return "candidate excluded by eligibility gate"
            return "candidate is differential-only by eligibility gate"
        if candidate.required_gaps and not self._gap_candidate_has_submission_evidence(candidate):
            return "required evidence gap lacks objective confirmation"
        if (
            decision
            and self._strong_open_world_contenders(decision.open_world_candidates)
            and not candidate.required_met
            and not self._gap_candidate_has_submission_evidence(candidate)
        ):
            return "strong open-world contender remains unresolved"
        if not candidate.matched_evidence:
            return "no matched supporting evidence"
        if not candidate.trusted and candidate.score <= 0:
            return "not trusted by evidence-first decision"
        return ""

    def _choose_authorized_primary(
        self,
        eligible: Sequence[CandidateScore],
        decision: DiagnosisDecision,
    ) -> CandidateScore:
        current_names = list(decision.final_diagnoses or [])
        for name in current_names:
            entity_id = self.knowledge.entity_id_for(name)
            for candidate in eligible:
                if candidate.diagnosis == name or (
                    entity_id and getattr(candidate, "entity_id", "") == entity_id
                ):
                    return candidate
        return self._sort_candidates(eligible)[0]

    def _candidate_for_name(
        self,
        candidates: Sequence[CandidateScore],
        name: Any,
    ) -> Optional[CandidateScore]:
        text = str(name or "").strip()
        if not text:
            return None
        score_by_name = {item.diagnosis: item for item in candidates or []}
        score_by_entity = {
            item.entity_id: item
            for item in candidates or []
            if getattr(item, "entity_id", "")
        }
        entity_id = self.knowledge.entity_id_for(text) if self.knowledge else ""
        return (score_by_entity.get(entity_id) if entity_id else None) or score_by_name.get(text)

    def _secondary_authorization_block_reason(
        self,
        candidate: CandidateScore,
        primary: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> str:
        if candidate.hard_contradiction:
            return "secondary diagnosis has hard contradiction"
        if getattr(candidate, "explained_by_root_cause", "") == primary.diagnosis:
            if not bool(getattr(candidate, "root_cause_submit_as_final", False)):
                return "root-cause downstream diagnosis retained for audit only"
            if self._has_authorized_independent_objective_evidence(candidate):
                return ""
            return "root-cause downstream diagnosis lacks independent objective evidence"
        if self._is_generic_parent_of_selected(candidate, selected):
            return "generic parent suppressed by a more specific primary diagnosis"
        if self._is_suppressed_by_selected(candidate, selected):
            return "suppressed by selected primary diagnosis"
        if self._diagnosis_causes(primary.diagnosis, candidate.diagnosis):
            if (
                self._is_secondary_manifestation(candidate)
                and self._has_independent_state_evidence(candidate)
            ):
                return ""
            return "downstream manifestation is fully explained by primary diagnosis"
        if self._diagnosis_causes(candidate.diagnosis, primary.diagnosis):
            if self._has_authorized_independent_objective_evidence(candidate):
                return ""
            return "upstream etiology remains audit-only without independent objective evidence"
        if self._diagnoses_submission_related(candidate.diagnosis, primary.diagnosis):
            if self._has_authorized_independent_objective_evidence(candidate):
                return ""
            return "related diagnosis lacks independent objective evidence"
        if self._explains_selected_residual(candidate, selected) and (
            self._has_authorized_independent_objective_evidence(candidate)
        ):
            return ""
        return "differential-only or low-explainability companion diagnosis"

    def _is_structural_comorbidity_candidate(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        if not selected:
            return False
        dtype = candidate.diagnosis_type.lower()
        if dtype not in {"structural", "etiology", "metabolic"}:
            return False
        return any(
            self._same_family_or_explicit_related(candidate.diagnosis, item.diagnosis)
            for item in selected
        )

    def _same_family_or_explicit_related(self, left: str, right: str) -> bool:
        if left == right:
            return True
        left_entry = self.knowledge.get(left)
        right_entry = self.knowledge.get(right)
        left_related = set(str(item) for item in left_entry.get("related_complications", []) or [])
        right_related = set(str(item) for item in right_entry.get("related_complications", []) or [])
        if right in left_related or left in right_related:
            return True
        left_system = str(left_entry.get("body_system") or "")
        right_system = str(right_entry.get("body_system") or "")
        left_family = str(left_entry.get("disease_family") or left_entry.get("family") or "")
        right_family = str(right_entry.get("disease_family") or right_entry.get("family") or "")
        return bool(left_system and left_system == right_system and left_family and left_family == right_family)

    def _is_suppressed_by_selected(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        selected_names = {item.diagnosis for item in selected}
        for item in selected:
            entry = self.knowledge.get(item.diagnosis)
            suppressed = set(str(value) for value in entry.get("suppress_diagnoses", []) or [])
            suppressed.update(
                str(value) for value in entry.get("generalization_suppressions", []) or []
            )
            if candidate.diagnosis in suppressed:
                return True
            if candidate.parent_diagnosis and candidate.parent_diagnosis in selected_names:
                return True
        return False

    def _has_authorized_independent_objective_evidence(
        self,
        candidate: CandidateScore,
    ) -> bool:
        if f"diagnosis:{candidate.diagnosis}" in set(candidate.matched_evidence or []):
            return True
        if (
            getattr(candidate, "root_cause_role", "") == "secondary"
            and getattr(candidate, "explained_by_root_cause", "")
        ):
            return bool(getattr(candidate, "root_cause_submit_as_final", False))
        return self._has_independent_state_evidence(candidate) or bool(
            candidate.component_scores.get("objective_evidence", 0.0) >= 1.0
        )

    def _gap_candidate_has_submission_evidence(self, candidate: CandidateScore) -> bool:
        if candidate.required_met:
            return True
        components = candidate.component_scores or {}
        try:
            objective = float(components.get("objective_evidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            objective = 0.0
        if objective >= 1.0:
            return True
        try:
            diagnostic = float(candidate.diagnostic_evidence_score or 0.0)
        except (TypeError, ValueError):
            diagnostic = 0.0
        if diagnostic >= 0.45:
            return True
        core_count = len(set(candidate.core_matched_evidence or []))
        try:
            core_coverage = float(candidate.core_explanatory_coverage or 0.0)
        except (TypeError, ValueError):
            core_coverage = 0.0
        return bool(
            core_count >= 3
            and core_coverage >= 0.65
            and int(candidate.residual_core_evidence_count or 0) <= 0
        )

    @staticmethod
    def _strong_open_world_contenders(
        open_world_candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        contenders: List[Dict[str, Any]] = []
        for item in open_world_candidates or []:
            try:
                prior = float(item.get("prior", 0.0) or 0.0)
            except (TypeError, ValueError):
                prior = 0.0
            metadata = item.get("metadata") or {}
            source = str(item.get("source") or "")
            if bool(item.get("submittable", False)):
                continue
            status = str(item.get("open_world_status") or "")
            if status == "OpenWorldDifferential":
                continue
            if status == "UnresolvedHighValue":
                contenders.append(dict(item))
                continue
            if prior < 0.62:
                continue
            if source not in {"mechanism_reasoner", "external_retrieval", "llm_unresolved"}:
                continue
            if not (item.get("evidence_links") or metadata.get("mechanism_id") or metadata.get("unreviewed_external")):
                continue
            contenders.append(dict(item))
        contenders.sort(key=lambda value: float(value.get("prior", 0.0) or 0.0), reverse=True)
        return contenders

    def _open_world_reasoning(
        self,
        open_world_candidates: Sequence[Dict[str, Any]],
        mechanism_hypotheses: Sequence[Dict[str, Any]],
    ) -> str:
        contenders = self._strong_open_world_contenders(open_world_candidates)
        if not contenders and not mechanism_hypotheses:
            return ""
        parts: List[str] = []
        if mechanism_hypotheses:
            names = [
                str(item.get("mechanism_id") or "")
                for item in mechanism_hypotheses[:3]
                if str(item.get("mechanism_id") or "")
            ]
            if names:
                parts.append("机制/疾病家族假设: " + ", ".join(names))
        if contenders:
            names = [
                str(item.get("raw_name") or "")
                for item in contenders[:3]
                if str(item.get("raw_name") or "")
            ]
            if names:
                parts.append("未映射开放候选仅用于鉴别和检查，不直接提交: " + ", ".join(names))
        return " ".join(parts)

    @staticmethod
    def _authorization_block_record(
        name: str,
        candidate: Optional[CandidateScore],
        reason: str,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {"diagnosis": name, "reason": reason}
        if candidate is not None:
            record.update(
                {
                    "score": candidate.score,
                    "entity_id": str(getattr(candidate, "entity_id", "") or ""),
                    "canonical_name": str(getattr(candidate, "canonical_name", "") or ""),
                    "submission_name": str(getattr(candidate, "submission_name", "") or ""),
                    "submittable": bool(getattr(candidate, "submittable", True)),
                    "coverage_score": candidate.coverage_score,
                    "residual_score": candidate.residual_score,
                    "required_met": candidate.required_met,
                    "required_gap_authorized": bool(
                        getattr(candidate, "required_gap_authorized", False)
                    ),
                    "hard_contradiction": candidate.hard_contradiction,
                    "matched_evidence": list(candidate.matched_evidence[:6]),
                    "contradicted_evidence": list(candidate.contradicted_evidence[:6]),
                    "soft_contradicted_evidence": list(
                        candidate.soft_contradicted_evidence[:6]
                    ),
                    "hard_contradicted_evidence": list(
                        candidate.hard_contradicted_evidence[:6]
                    ),
                    "evidence_conflicts": list(
                        getattr(candidate, "evidence_conflicts", []) or []
                    ),
                    "unresolved_evidence_conflict": bool(
                        getattr(candidate, "unresolved_evidence_conflict", False)
                    ),
                    "conflict_adjudication_exams": list(
                        getattr(candidate, "conflict_adjudication_exams", []) or []
                    ),
                    "eligibility_status": str(
                        getattr(candidate, "eligibility_status", "") or ""
                    ),
                    "eligibility_reason": str(
                        getattr(candidate, "eligibility_reason", "") or ""
                    ),
                    "eligibility_substatus": str(
                        getattr(candidate, "eligibility_substatus", "") or ""
                    ),
                    "missing_required_anchors": list(
                        getattr(candidate, "missing_required_anchors", []) or []
                    ),
                    "eligibility_blockers": list(
                        getattr(candidate, "eligibility_blockers", []) or []
                    ),
                    "root_cause_role": str(
                        getattr(candidate, "root_cause_role", "") or ""
                    ),
                    "explained_by_root_cause": str(
                        getattr(candidate, "explained_by_root_cause", "") or ""
                    ),
                    "root_cause_submit_as_final": bool(
                        getattr(candidate, "root_cause_submit_as_final", False)
                    ),
                    "explains_candidates": list(
                        getattr(candidate, "explains_candidates", []) or []
                    ),
                    "root_cause_coverage": float(
                        getattr(candidate, "root_cause_coverage", 0.0) or 0.0
                    ),
                    "evidence_claims": list(
                        getattr(candidate, "evidence_claims", []) or []
                    )[:6],
                    "unresolved_critical_evidence_claims": list(
                        getattr(candidate, "unresolved_critical_evidence_claims", []) or []
                    )[:6],
                    "claim_followup_exams": list(
                        getattr(candidate, "claim_followup_exams", []) or []
                    ),
                    "claim_verification_status": str(
                        getattr(candidate, "claim_verification_status", "") or ""
                    ),
                    "evidence_gaps": list(
                        getattr(candidate, "evidence_gaps", []) or []
                    )[:6],
                    "deferred_priority": float(
                        getattr(candidate, "deferred_priority", 0.0) or 0.0
                    ),
                    "deferred_priority_components": dict(
                        getattr(candidate, "deferred_priority_components", {}) or {}
                    ),
                    "exam_priority_override": bool(
                        getattr(candidate, "exam_priority_override", False)
                    ),
                    "exam_priority_override_reason": str(
                        getattr(candidate, "exam_priority_override_reason", "") or ""
                    ),
                }
            )
        return record

    def apply_to_result(
        self,
        result: Optional[Dict[str, Any]],
        decision: DiagnosisDecision,
        evidence: EvidenceBundle,
    ) -> Dict[str, Any]:
        fixed = dict(result or {})
        self.authorize_final_diagnoses(
            decision,
            decision.final_diagnoses,
            respect_differential_only=True,
        )
        decision.differential_only_diagnoses = self.differential_only_details(decision.candidates)
        final_submission_names = [
            self.knowledge.submission_name_for(item)
            for item in decision.final_diagnoses
            if self.knowledge.submission_name_for(item)
        ]
        fixed["diagnosis"] = list(dict.fromkeys(final_submission_names))
        fixed["_trusted_diagnoses"] = [
            self.knowledge.submission_name_for(item)
            for item in decision.trusted_diagnoses
        ]
        fixed["_authorized_diagnoses"] = [
            self.knowledge.submission_name_for(item)
            for item in (decision.authorized_diagnoses or decision.final_diagnoses)
        ]
        fixed["_blocked_diagnoses"] = list(decision.blocked_diagnoses)
        fixed["_retriever_top1"] = decision.retriever_top1
        fixed["_judge_primary"] = decision.judge_primary
        fixed["_submitter_final"] = list(decision.submitter_final or decision.final_diagnoses)
        fixed["_required_gap_authorized_diagnoses"] = list(
            decision.required_gap_authorized_diagnoses
        )
        fixed["_authorization_locked"] = True
        fixed["_diagnosis_decision"] = decision.to_dict()
        fixed["_diagnosis_entity_resolution"] = list(decision.entity_resolutions)
        fixed["_diagnosis_name_resolution"] = list(decision.name_resolutions)
        fixed["_unresolved_diagnosis_candidates"] = list(decision.unresolved_candidates)
        fixed["_open_world_diagnosis_candidates"] = list(decision.open_world_candidates)
        fixed["_mechanism_hypotheses"] = list(decision.mechanism_hypotheses)
        fixed["_llm_pattern_hypotheses"] = list(decision.llm_pattern_hypotheses)
        fixed["_verified_pattern_hypotheses"] = list(decision.verified_pattern_hypotheses)
        fixed["_rejected_pattern_hypotheses"] = list(decision.rejected_pattern_hypotheses)
        fixed["_pattern_recall_signals"] = list(decision.pattern_recall_signals)
        fixed["_pattern_driven_candidate_recall"] = list(decision.pattern_driven_candidate_recall)
        fixed["_pattern_protected_candidate_recall"] = list(decision.pattern_protected_candidate_recall)
        fixed["_pattern_gap_suggestions"] = list(decision.pattern_gap_suggestions)
        fixed["_unverified_pattern_leakage_count"] = int(
            decision.unverified_pattern_leakage_count or 0
        )
        fixed["_pattern_generated_active_gaps"] = int(
            decision.pattern_generated_active_gaps or 0
        )
        fixed["_pattern_expansion_round_count"] = int(
            decision.pattern_expansion_round_count or 0
        )
        fixed["_retrieval_views"] = list(decision.retrieval_views)
        fixed["_evidence_items"] = [item.to_dict() for item in evidence.observations]
        reasoning = str(fixed.get("reasoning") or "").strip()
        if decision.evidence_reasoning and decision.evidence_reasoning not in reasoning:
            reasoning = (reasoning.rstrip("。") + "。" if reasoning else "") + decision.evidence_reasoning
        fixed["reasoning"] = reasoning
        return fixed

    def _sort_candidates(
        self, candidates: Sequence[CandidateScore]
    ) -> List[CandidateScore]:
        return sorted(candidates, key=self._candidate_sort_key, reverse=True)

    def _candidate_sort_key(self, candidate: CandidateScore) -> Tuple[float, ...]:
        audit_visibility_bonus = 0.0
        if candidate.source_prior >= 0.5 and candidate.matched_evidence:
            audit_visibility_bonus += 0.04
        if self._is_secondary_manifestation(candidate) and candidate.matched_evidence:
            audit_visibility_bonus += 0.20
        eligibility_rank = {
            PRIMARY_ELIGIBLE: 3.0,
            DEFERRED: 2.0,
            DIFFERENTIAL_ONLY: 1.0,
            EXCLUDED: 0.0,
        }.get(candidate.eligibility_status, 1.0)
        return (
            eligibility_rank,
            self._adjudication_score(candidate) + audit_visibility_bonus,
            self._diagnosis_type_rank(candidate),
            candidate.source_prior,
            1.0 - candidate.residual_score,
            candidate.evidence_specificity_score,
            candidate.score,
        )

    def _adjudication_score(self, candidate: CandidateScore) -> float:
        score = candidate.score
        score += 0.08 * candidate.core_evidence_score
        score += 0.12 * candidate.diagnostic_evidence_score
        score -= 0.08 * float(
            (candidate.component_scores or {}).get("generic_parent_penalty", 0.0) or 0.0
        )
        score -= 0.05 * float(
            (candidate.component_scores or {}).get("specific_over_generic_penalty", 0.0) or 0.0
        )
        if (
            candidate.required_met
            and not candidate.hard_contradiction
            and candidate.matched_evidence
            and self._is_etiology_priority_candidate(candidate)
        ):
            score += self.etiology_priority_bonus
        elif (
            candidate.required_gaps
            and not candidate.hard_contradiction
            and candidate.matched_evidence
            and self._is_etiology_priority_candidate(candidate)
            and (
                candidate.coverage_score >= self.evidence_gap_coverage_threshold
                or candidate.residual_score <= self.evidence_gap_residual_threshold
            )
        ):
            score += self.etiology_priority_bonus * 0.5
        return round(min(1.0, score), 4)

    def _diagnosis_type_rank(self, candidate: CandidateScore) -> int:
        dtype = candidate.diagnosis_type.lower()
        if dtype in {"etiology", "metabolic", "structural"}:
            return 4
        if dtype == "disease":
            return 2
        if dtype in {"syndrome", "state", "complication"}:
            return 1
        return 0

    @staticmethod
    def _is_etiology_priority_candidate(candidate: CandidateScore) -> bool:
        dtype = candidate.diagnosis_type.lower()
        return dtype in {"etiology", "metabolic", "structural"}

    def render_candidate_table(self, decision: DiagnosisDecision, limit: int = 8) -> str:
        lines = ["【证据评分候选】"]
        for item in decision.candidates[:limit]:
            lines.append(
                f"- {item.diagnosis}: score={item.score:.3f}, support={item.support_score:.3f}, "
                f"prior={item.source_prior:.3f}, coverage={item.coverage_score:.3f}, "
                f"residual={item.residual_score:.3f}, required={item.required_met}, "
                f"hard_contradiction={item.hard_contradiction}, gaps={item.required_gaps[:3]}, "
                f"residual_evidence={item.residual_evidence[:4]}, evidence={item.matched_evidence[:5]}"
            )
        resolved = [
            f"{item.get('raw_name')}→{item.get('canonical_name')}({item.get('method')})"
            for item in decision.name_resolutions
            if item.get("canonical_name")
        ]
        if resolved:
            lines.append("【开放候选标准化】" + "；".join(resolved[:limit]))
        if decision.unresolved_candidates:
            lines.append("【未解析候选】" + "、".join(decision.unresolved_candidates[:limit]))
        return "\n".join(lines)

    @staticmethod
    def _clear_submission_marks(scores: Sequence[CandidateScore]) -> None:
        for item in scores:
            item.differential_only = False
            item.differential_only_reason = ""
            item.required_gap_authorized = False
            item.exam_followup_authorized = False
            item.submission_authorized = False

    def differential_only_details(
        self,
        scores: Sequence[CandidateScore],
    ) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        for item in scores:
            if not item.differential_only:
                continue
            details.append(
                {
                    "diagnosis": item.diagnosis,
                    "reason": item.differential_only_reason,
                    "score": item.score,
                    "coverage_score": item.coverage_score,
                    "residual_score": item.residual_score,
                    "causal_relation_to_selected": item.causal_relation_to_selected,
                    "eligibility_status": item.eligibility_status,
                    "eligibility_reason": item.eligibility_reason,
                    "missing_required_anchors": list(item.missing_required_anchors),
                }
            )
        return details

    def differential_only_reasoning(
        self,
        scores: Sequence[CandidateScore],
        limit: int = 3,
    ) -> str:
        details = self.differential_only_details(scores)
        if not details:
            return ""
        parts = [
            f"{item['diagnosis']}：{item['reason']}"
            for item in details[:limit]
        ]
        return "仅鉴别诊断不提交：" + "；".join(parts) + "。"

    def resolve_open_candidates(self, result: Any) -> List[DiagnosisResolution]:
        return self.resolver.resolve_result(result)

    def build_pattern_recall_context(
        self,
        llm_result: Any,
        evidence: EvidenceBundle,
        *,
        case_id: str = "",
        case_version: int = 0,
        evidence_snapshot_id: str = "",
        thinking_snapshots: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(llm_result, dict):
            llm_result = {}
        return build_pattern_recall_context(
            self.pattern_hypothesis_verifier,
            llm_result,
            evidence,
            case_id=case_id,
            case_version=case_version,
            evidence_snapshot_id=evidence_snapshot_id
            or pattern_evidence_snapshot_hash(evidence),
            thinking_snapshots=thinking_snapshots,
            adapter=self.pattern_proposal_adapter,
        )

    def build_retrieval_views(self, evidence: EvidenceBundle) -> List[Dict[str, Any]]:
        mechanisms = self.mechanism_reasoner.evaluate(evidence)
        return self.clinical_pattern_compiler.retrieval_views(evidence) + [
            item.to_dict()
            for item in self.mechanism_reasoner.retrieval_views(evidence, mechanisms)
        ]

    def _score_entry(
        self,
        entry: Dict[str, Any],
        prior: float,
        evidence: EvidenceBundle,
        candidate_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> CandidateScore:
        support_specs = list(entry.get("supporting_evidence", []) or [])
        matched: List[str] = []
        matched_observations: List[Observation] = []
        matched_weight = 0.0
        for spec in support_specs:
            hits = _matching_observations(spec, evidence.observations, polarity="positive")
            if not hits:
                continue
            weight = float(spec.get("weight", 0.2) or 0.2)
            best_confidence = max(item.confidence * _information_multiplier(item) for item in hits)
            matched_weight += weight * best_confidence
            matched.extend(item.finding for item in hits)
            matched_observations.extend(hits)

        support_score = min(1.0, matched_weight)
        required_groups = entry.get("required_groups", []) or []
        required_gaps: List[str] = []
        for group in required_groups:
            if not isinstance(group, list) or not group:
                continue
            group_hits: List[Observation] = []
            for spec in group:
                group_hits.extend(
                    _matching_observations(
                        _coerce_spec(spec),
                        evidence.observations,
                        polarity="positive",
                    )
                )
            if group_hits:
                matched_observations.extend(group_hits)
                matched.extend(item.finding for item in group_hits)
                matched_weight += 0.18 * max(
                    item.confidence * _information_multiplier(item)
                    for item in group_hits
                )
            else:
                required_gaps.append(_render_required_group(group))
        required_met = not required_gaps
        support_score = min(1.0, matched_weight)

        soft_contradicted: List[str] = []
        hard_contradicted: List[str] = []
        contradiction_penalty = 0.0
        hard_contradiction = False
        direct_negative = [
            item for item in evidence.observations
            if item.finding == f"diagnosis:{entry['name']}" and item.polarity == "negative"
        ]
        if direct_negative:
            direct_name = f"diagnosis:{entry['name']}"
            direct_positive = any(
                item.finding == direct_name and item.polarity == "positive"
                for item in evidence.observations
            )
            if direct_positive or (required_met and support_score >= 0.5):
                soft_contradicted.append(direct_name)
                contradiction_penalty += 0.25
            else:
                hard_contradicted.append(direct_name)
                contradiction_penalty += 0.65
                hard_contradiction = True

        for spec in entry.get("contradictions", []) or []:
            spec = _coerce_spec(spec)
            hits = _matching_observations(spec, evidence.observations, polarity=spec.get("polarity", "positive"))
            if not hits:
                continue
            target = hard_contradicted if bool(spec.get("hard", False)) else soft_contradicted
            target.extend(item.finding for item in hits)
            contradiction_penalty += float(spec.get("penalty", 0.35) or 0.35)
            hard_contradiction = hard_contradiction or bool(spec.get("hard", False))

        # Negative evidence against a required/supporting finding lowers the score even
        # when the knowledge entry did not repeat it as an explicit contradiction.
        supporting_findings = {str(spec.get("finding")) for spec in support_specs if spec.get("finding")}
        for item in evidence.observations:
            if item.polarity == "negative" and item.finding in supporting_findings:
                contradiction_penalty += 0.2
                soft_contradicted.append(item.finding)

        explainability = self._explainability(
            support_specs,
            evidence,
            bool(matched),
        )
        coverage_score = explainability["coverage"]
        residual_score = explainability["residual_score"]
        core_coverage = explainability["core_coverage"]
        residual_evidence_score = explainability["residual_evidence_score"]
        residual_core_evidence = explainability["unexplained_core_evidence"]
        explained_evidence = explainability["explained_evidence"]
        residual_evidence = explainability["residual_evidence"]
        explanation = coverage_score
        tiered = self._tiered_matched_evidence(matched_observations)
        generic_evidence_score = tiered["generic_score"]
        core_evidence_score = tiered["core_score"]
        diagnostic_evidence_score = tiered["diagnostic_score"]
        specificity = float(entry.get("specificity", 0.5) or 0.5)
        has_signal = bool(matched or prior > 0)
        specificity_score = 0.0
        dtype = str(entry.get("diagnosis_type") or "disease").lower()
        etiology_structural_score = (
            1.0
            if has_signal
            and dtype in {"etiology", "metabolic", "structural", "systemic"}
            and (core_evidence_score >= 0.20 or diagnostic_evidence_score > 0.0 or required_met)
            else 0.0
        )
        exam_match = self._expected_exam_match(entry, evidence) if has_signal else 0.0
        temporal = self._temporal_consistency(matched, evidence) if has_signal else 0.0
        age = self._age_match(entry, evidence) if has_signal else 0.0
        risk = self._risk_factor_match(entry, evidence) if has_signal else 0.0
        objective_evidence = self._has_objective_evidence(matched, evidence)
        gap_penalty = 0.0
        if required_gaps:
            if objective_evidence:
                gap_penalty = min(0.12, 0.04 * len(required_gaps))
            else:
                gap_penalty = min(0.28, 0.18 + 0.05 * (len(required_gaps) - 1))
        generic_parent_penalty = self._generic_parent_penalty(
            entry,
            core_evidence_score=core_evidence_score,
            diagnostic_evidence_score=diagnostic_evidence_score,
            residual_core_count=len(residual_core_evidence),
            required_met=required_met,
        )
        raw_score = (
            self.weights["evidence"] * support_score * 0.45
            + self.weights["prior"] * max(0.0, min(1.0, prior))
            + self.weights["explain"] * explanation
            + self.weights["core_explain"] * core_coverage
            + self.weights["generic_evidence"] * generic_evidence_score
            + self.weights["core_evidence"] * core_evidence_score
            + self.weights["diagnostic_evidence"] * diagnostic_evidence_score
            + self.weights["etiology_structural"] * etiology_structural_score
            + self.weights["exam_match"] * exam_match
            + self.weights["temporal"] * temporal
            + self.weights["age"] * age
            + self.weights["risk"] * risk
            - self.weights["residual"] * residual_score
            - self.weights["core_residual"] * min(1.0, 0.28 * len(residual_core_evidence))
            - self.weights["generic_parent"] * generic_parent_penalty
            - self.weights["contradiction"] * contradiction_penalty
            - gap_penalty
        )
        if not required_met and self.required_group_policy != "gap_only":
            raw_score = min(raw_score, self.differential_threshold - 0.01)
        score = max(0.0, min(1.0, raw_score))
        required_gap_state = self._required_gap_state(
            entry=entry,
            required_met=required_met,
            hard_contradiction=hard_contradiction,
            matched_evidence=matched,
            required_gaps=required_gaps,
            coverage_score=coverage_score,
            core_coverage=core_coverage,
            residual_score=residual_score,
            residual_core_count=len(residual_core_evidence),
        )
        return CandidateScore(
            diagnosis=entry["name"],
            score=round(score, 4),
            support_score=round(support_score, 4),
            source_prior=round(prior, 4),
            explanation_score=round(explanation, 4),
            coverage_score=round(coverage_score, 4),
            residual_score=round(residual_score, 4),
            explanatory_coverage=round(coverage_score, 4),
            core_explanatory_coverage=round(core_coverage, 4),
            residual_evidence_score=round(residual_evidence_score, 4),
            residual_core_evidence_count=len(residual_core_evidence),
            explained_evidence=list(dict.fromkeys(explained_evidence))[:12],
            unexplained_core_evidence=list(dict.fromkeys(residual_core_evidence))[:12],
            explanatory_rank_reason=self._explanatory_rank_reason(
                coverage_score,
                core_coverage,
                residual_evidence_score,
                residual_core_evidence,
            ),
            contradiction_penalty=round(contradiction_penalty, 4),
            required_met=required_met,
            hard_contradiction=hard_contradiction,
            required_gap_state=required_gap_state,
            matched_evidence=list(dict.fromkeys(matched)),
            generic_matched_evidence=list(dict.fromkeys(tiered["generic_evidence"])),
            core_matched_evidence=list(dict.fromkeys(tiered["core_evidence"])),
            diagnostic_matched_evidence=list(dict.fromkeys(tiered["diagnostic_evidence"])),
            generic_coverage_score=round(generic_evidence_score, 4),
            core_evidence_score=round(core_evidence_score, 4),
            diagnostic_evidence_score=round(diagnostic_evidence_score, 4),
            contradicted_evidence=list(
                dict.fromkeys(hard_contradicted + soft_contradicted)
            ),
            soft_contradicted_evidence=list(dict.fromkeys(soft_contradicted)),
            hard_contradicted_evidence=list(dict.fromkeys(hard_contradicted)),
            required_gaps=list(dict.fromkeys(required_gaps)),
            residual_evidence=list(dict.fromkeys(residual_evidence))[:12],
            component_scores={
                "evidence": round(support_score, 4),
                "prior": round(max(0.0, min(1.0, prior)), 4),
                "specificity": 0.0,
                "disease_specificity_metadata": round(specificity, 4),
                "explain": round(explanation, 4),
                "coverage": round(coverage_score, 4),
                "explanatory_coverage": round(coverage_score, 4),
                "core_explanatory_coverage": round(core_coverage, 4),
                "generic_coverage_score": round(generic_evidence_score, 4),
                "core_evidence_score": round(core_evidence_score, 4),
                "diagnostic_evidence_score": round(diagnostic_evidence_score, 4),
                "etiology_structural_bonus": round(etiology_structural_score, 4),
                "generic_parent_penalty": round(generic_parent_penalty, 4),
                "residual": round(residual_score, 4),
                "residual_evidence_score": round(residual_evidence_score, 4),
                "residual_core_evidence_count": float(len(residual_core_evidence)),
                "residual_core_penalty": round(
                    min(1.0, 0.28 * len(residual_core_evidence)),
                    4,
                ),
                "exam_match": round(exam_match, 4),
                "temporal": round(temporal, 4),
                "age": round(age, 4),
                "risk": round(risk, 4),
                "contradiction": round(contradiction_penalty, 4),
                "soft_contradiction_count": float(
                    len(set(soft_contradicted))
                ),
                "hard_contradiction_count": float(
                    len(set(hard_contradicted))
                ),
                "required_gap_penalty": round(gap_penalty, 4),
                "required_gap_state": required_gap_state,
                "objective_evidence": 1.0 if objective_evidence else 0.0,
            },
            candidate_sources=list(candidate_sources or []),
            diagnosis_type=str(entry.get("diagnosis_type") or "disease"),
            parent_diagnosis=str(entry.get("parent_diagnosis") or ""),
            specificity=specificity,
            entity_id=str(entry.get("entity_id") or ""),
            canonical_name=str(entry.get("canonical_name") or entry.get("name") or ""),
            submission_name=str(entry.get("submission_name") or entry.get("name") or ""),
            raw_names=list(
                dict.fromkeys(
                    str(item.get("raw_name") or "").strip()
                    for item in (candidate_sources or [])
                    if str(item.get("raw_name") or "").strip()
                )
            ),
            submittable=bool(entry.get("submittable", True)),
        )

    @classmethod
    def _tiered_matched_evidence(
        cls,
        observations: Sequence[Observation],
    ) -> Dict[str, Any]:
        buckets = {
            "generic": {},
            "core": {},
            "diagnostic": {},
        }
        for item in observations:
            if item.polarity != "positive" or not item.finding:
                continue
            tier = cls._observation_evidence_tier(item)
            value = max(0.0, min(1.0, item.confidence * _information_multiplier(item)))
            existing = buckets[tier].get(item.finding, 0.0)
            buckets[tier][item.finding] = max(existing, value)
        return {
            "generic_evidence": list(buckets["generic"].keys()),
            "core_evidence": list(buckets["core"].keys()),
            "diagnostic_evidence": list(buckets["diagnostic"].keys()),
            "generic_score": min(1.0, sum(buckets["generic"].values()) / 4.0),
            "core_score": min(1.0, sum(buckets["core"].values()) / 3.0),
            "diagnostic_score": min(1.0, sum(buckets["diagnostic"].values()) / 2.0),
        }

    @classmethod
    def _observation_evidence_tier(cls, observation: Observation) -> str:
        finding = str(observation.finding or "")
        evidence_level = str(getattr(observation, "evidence_level", "") or "")
        if finding.startswith("diagnosis:"):
            return "diagnostic"
        if finding in _DIAGNOSTIC_EXPLANATORY_FINDINGS:
            return "diagnostic"
        if evidence_level == "diagnostic_pattern":
            return "diagnostic"
        if observation.value is not None or observation.direction:
            return "diagnostic"
        if finding in _CORE_EXPLANATORY_FINDINGS:
            return "core"
        if evidence_level == "specific":
            return "core"
        try:
            information_value = float(getattr(observation, "information_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            information_value = 0.0
        if information_value >= 0.75:
            return "core"
        if evidence_level == "generic" or finding in _GENERIC_EXPLANATORY_FINDINGS:
            return "generic"
        if finding.startswith(("field:", "symptom:")):
            return "generic"
        return "core"

    @staticmethod
    def _generic_parent_penalty(
        entry: Dict[str, Any],
        *,
        core_evidence_score: float,
        diagnostic_evidence_score: float,
        residual_core_count: int,
        required_met: bool,
    ) -> float:
        name = str(entry.get("name") or "")
        dtype = str(entry.get("diagnosis_type") or "").lower()
        is_generic = (
            name in _GENERIC_PARENT_DIAGNOSES
            or dtype in {"syndrome", "state", "complication"}
        )
        if not is_generic:
            return 0.0
        penalty = 0.25
        if core_evidence_score <= 0.05 and diagnostic_evidence_score <= 0.05:
            penalty += 0.30
        if diagnostic_evidence_score <= 0.05:
            penalty += 0.10
        if required_met:
            penalty += 0.08
        penalty += min(0.30, 0.10 * max(0, residual_core_count))
        return round(min(1.0, penalty), 4)

    @staticmethod
    def _required_gap_state(
        entry: Dict[str, Any],
        required_met: bool,
        hard_contradiction: bool,
        matched_evidence: Sequence[str],
        required_gaps: Sequence[str],
        coverage_score: float,
        core_coverage: float,
        residual_score: float,
        residual_core_count: int,
    ) -> str:
        if hard_contradiction:
            return "hard_contradiction"
        if required_met and not required_gaps:
            return "satisfied"
        if not matched_evidence:
            return "unsupported_gap"
        if required_gaps:
            has_actionable_exam = bool(
                entry.get("discriminating_exams")
                or entry.get("strong_verification_exams")
                or entry.get("required_exams")
            )
            if has_actionable_exam:
                return "actionable_gap"
            if core_coverage >= 0.40 or (
                coverage_score >= 0.52 and residual_score <= 0.48
            ):
                return "nonblocking_gap"
            if coverage_score >= 0.28 or residual_core_count <= 2:
                return "partially_satisfied"
            return "unsupported_gap"
        if core_coverage >= 0.40 or coverage_score >= 0.52:
            return "nonblocking_gap"
        return "partially_satisfied"

    def _apply_competitive_specificity(self, scores: Sequence[CandidateScore]) -> None:
        by_name = {item.diagnosis: item for item in scores}
        for specific in scores:
            if specific.hard_contradiction or not specific.matched_evidence:
                continue
            has_core_or_diagnostic_signal = bool(
                specific.core_matched_evidence
                or specific.diagnostic_matched_evidence
                or specific.core_evidence_score >= 0.20
                or specific.diagnostic_evidence_score > 0.0
            )
            if not (
                specific.required_met
                or specific.source_prior >= 0.45
                or specific.coverage_score >= self.evidence_gap_coverage_threshold
                or has_core_or_diagnostic_signal
            ):
                continue
            entry = self.knowledge.get(specific.diagnosis)
            explicit_generic_names = set(
                _SPECIFIC_GENERIC_SUPPRESSIONS.get(specific.diagnosis, set())
            )
            generic_names = set(str(item) for item in entry.get("generalization_suppressions", []) or [])
            generic_names.update(str(item) for item in entry.get("suppress_diagnoses", []) or [])
            generic_names.update(explicit_generic_names)
            if specific.parent_diagnosis:
                generic_names.add(specific.parent_diagnosis)
            for generic_name in generic_names:
                generic = by_name.get(generic_name)
                if not generic or generic.hard_contradiction:
                    continue
                if (
                    generic_name not in explicit_generic_names
                    and generic.parent_diagnosis != specific.diagnosis
                ):
                    continue
                penalty = 0.16 if specific.required_met else 0.12
                if specific.diagnostic_evidence_score > 0.0:
                    penalty += 0.10
                elif specific.core_evidence_score >= 0.20:
                    penalty += 0.06
                if generic.residual_core_evidence_count > specific.residual_core_evidence_count:
                    penalty += min(0.08, 0.03 * generic.residual_core_evidence_count)
                generic.score = round(max(0.0, generic.score - penalty), 4)
                generic.component_scores["generalization_penalty"] = round(
                    generic.component_scores.get("generalization_penalty", 0.0) + penalty,
                    4,
                )
                generic.component_scores["specific_over_generic_penalty"] = round(
                    generic.component_scores.get("specific_over_generic_penalty", 0.0) + penalty,
                    4,
                )

        av_block = by_name.get("二度房室传导阻滞")
        low_mag = by_name.get("低镁血症")
        if av_block and low_mag and av_block.matched_evidence:
            conduction_signal = {
                "second_degree_av_block",
                "av_block",
                "bradycardia",
                "pr_prolongation",
                "dropped_beats",
            }
            low_mag_direct = {"low_magnesium", "magnesium_depletion", "magnesium_load_retention_high"}
            if (
                conduction_signal & set(av_block.matched_evidence)
                and not (low_mag_direct & set(low_mag.matched_evidence))
            ):
                low_mag.score = round(max(0.0, low_mag.score - 0.16), 4)
                low_mag.component_scores["anchoring_penalty"] = round(
                    low_mag.component_scores.get("anchoring_penalty", 0.0) + 0.16,
                    4,
                )

    def _explainability(
        self,
        support_specs: Sequence[Dict[str, Any]],
        evidence: EvidenceBundle,
        has_matched_signal: bool,
    ) -> Dict[str, Any]:
        major = evidence.major()
        if not major:
            coverage = 0.5 if has_matched_signal else 0.0
            return {
                "coverage": coverage,
                "core_coverage": coverage,
                "residual_score": 0.0,
                "residual_evidence_score": 0.0,
                "residual_evidence": [],
                "explained_evidence": [],
                "unexplained_core_evidence": [],
            }

        total_weight = 0.0
        explained_weight = 0.0
        core_weight = 0.0
        explained_core_weight = 0.0
        residual: List[str] = []
        explained: List[str] = []
        core_residual: List[str] = []
        for observation in major:
            weight = self._observation_explainability_weight(observation)
            total_weight += weight
            is_core = self._is_core_explanatory_observation(observation)
            if is_core:
                core_weight += weight
            if any(_observation_matches(spec, observation) for spec in support_specs):
                explained_weight += weight
                explained.append(observation.finding)
                if is_core:
                    explained_core_weight += weight
            else:
                residual.append(observation.finding)
                if is_core:
                    core_residual.append(observation.finding)

        if total_weight <= 0:
            return {
                "coverage": 0.0,
                "core_coverage": 0.0,
                "residual_score": 0.0,
                "residual_evidence_score": 0.0,
                "residual_evidence": [],
                "explained_evidence": [],
                "unexplained_core_evidence": [],
            }
        coverage = max(0.0, min(1.0, explained_weight / total_weight))
        residual_score = max(0.0, min(1.0, 1.0 - coverage))
        core_coverage = (
            max(0.0, min(1.0, explained_core_weight / core_weight))
            if core_weight > 0
            else coverage
        )
        return {
            "coverage": coverage,
            "core_coverage": core_coverage,
            "residual_score": residual_score,
            "residual_evidence_score": residual_score,
            "residual_evidence": list(dict.fromkeys(residual)),
            "explained_evidence": list(dict.fromkeys(explained)),
            "unexplained_core_evidence": list(dict.fromkeys(core_residual)),
        }

    @staticmethod
    def _is_core_explanatory_observation(observation: Observation) -> bool:
        finding = str(observation.finding or "")
        if not finding or finding.startswith("field:"):
            return False
        if getattr(observation, "shadowed_by", ""):
            return False
        evidence_level = str(getattr(observation, "evidence_level", "") or "")
        if evidence_level == "generic":
            return False
        try:
            information_value = float(getattr(observation, "information_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            information_value = 0.0
        if finding.startswith("diagnosis:"):
            return True
        if finding in _DIAGNOSTIC_EXPLANATORY_FINDINGS:
            return True
        if finding in _CORE_EXPLANATORY_FINDINGS:
            return True
        if finding in _GENERIC_EXPLANATORY_FINDINGS:
            return False
        if evidence_level in {"specific", "diagnostic_pattern"} or information_value >= 0.75:
            return True
        if observation.value is not None or observation.direction:
            return True
        if observation.source != "问诊" and not finding.startswith("symptom:"):
            return True
        return not finding.startswith("symptom:")

    @staticmethod
    def _explanatory_rank_reason(
        coverage: float,
        core_coverage: float,
        residual_score: float,
        core_residual: Sequence[str],
    ) -> str:
        if core_residual:
            return (
                "core residual evidence remains: "
                + ", ".join(str(item) for item in core_residual[:4])
            )
        if core_coverage >= 0.75 and coverage >= 0.55:
            return "high explanatory coverage with low core residual evidence"
        if residual_score >= 0.55:
            return "low explanatory coverage with high residual evidence"
        return "moderate explanatory coverage"

    @staticmethod
    def _observation_explainability_weight(observation: Observation) -> float:
        if observation.finding.startswith("diagnosis:"):
            return 1.15
        if observation.value is not None or observation.direction:
            return 1.05
        if observation.source != "问诊":
            return 1.0
        if observation.finding.startswith("symptom:"):
            return 0.65
        return 0.8

    def _expected_exam_match(self, entry: Dict[str, Any], evidence: EvidenceBundle) -> float:
        exams = [str(item).strip() for item in entry.get("discriminating_exams", []) or [] if str(item).strip()]
        if not exams:
            return 0.0
        sources = {_compact_text(item.source) for item in evidence.observations if item.source}
        if not sources:
            return 0.0
        matched = 0
        for exam in exams:
            compact_exam = _compact_text(exam)
            if any(compact_exam in source or source in compact_exam for source in sources):
                matched += 1
        denominator = max(1, min(3, len(exams)))
        return min(1.0, matched / denominator)

    @staticmethod
    def _has_objective_evidence(matched: Sequence[str], evidence: EvidenceBundle) -> bool:
        matched_set = set(matched or [])
        if not matched_set:
            return False
        for item in evidence.observations:
            if item.finding not in matched_set or item.polarity != "positive":
                continue
            if item.finding.startswith("diagnosis:"):
                return True
            if str(item.source or "").strip() not in {
                "\u95ee\u8bca",
                "raw_case_finding",
                "evidence_interpreter",
                "reasoning_inference",
            }:
                return True
        return False

    @staticmethod
    def _temporal_consistency(matched: Sequence[str], evidence: EvidenceBundle) -> float:
        if not matched:
            return 0.0
        matched_set = set(matched)
        observations = [
            item for item in evidence.observations
            if item.finding in matched_set and item.polarity == "positive"
        ]
        if not observations:
            return 0.0
        if any(item.temporality for item in observations):
            return 0.7
        return 0.4

    @staticmethod
    def _age_match(entry: Dict[str, Any], evidence: EvidenceBundle) -> float:
        diagnosis = str(entry.get("name") or "")
        department = str(entry.get("department") or "")
        age_values = [
            item.value for item in evidence.observations
            if item.finding == "field:age" and item.value is not None
        ]
        if not age_values:
            return 0.0
        age = age_values[0]
        if age < 18 and ("儿" in department or diagnosis in {"先天性心脏病", "房间隔缺损"}):
            return 1.0
        if age >= 60 and diagnosis in {"骨质疏松症", "冠心病", "终末期肾病"}:
            return 0.7
        if age < 18 and diagnosis in {"骨质疏松症", "冠心病"}:
            return 0.0
        return 0.35

    @staticmethod
    def _risk_factor_match(entry: Dict[str, Any], evidence: EvidenceBundle) -> float:
        diagnosis = str(entry.get("name") or "")
        findings = set(evidence.findings("positive"))
        disease_signals = {
            "先天性心脏病": {
                "cyanosis", "feeding_diaphoresis", "congenital_heart_defect",
                "ventricular_septal_defect", "right_to_left_shunt", "pulmonary_hypertension",
            },
            "肺动脉瓣狭窄": {"pulmonary_valve_gradient", "pulmonary_valve_stenosis", "cyanosis"},
            "肺不张": {"atelectasis", "choking_event", "aspiration_risk"},
            "终末期肾病": {"renal_impairment", "egfr_low", "urea_elevated", "oliguria", "hyperkalemia"},
            "卵巢过度刺激综合征": {
                "ohss_risk", "ovarian_enlargement", "ascites", "hemoconcentration", "hypoalbuminemia",
            },
            "门静脉高压": {"portal_vein_dilation", "portal_flow_abnormal", "splenomegaly", "ascites", "varices"},
        }
        expected = disease_signals.get(diagnosis, set())
        if not expected:
            return 0.0
        overlap = len(expected & findings)
        return min(1.0, overlap / max(1, min(3, len(expected))))

    def _candidate_priors(
        self,
        llm_result: Dict[str, Any],
        rag_chunks: Sequence[Dict[str, Any]],
        resolutions: Optional[Sequence[DiagnosisResolution]] = None,
    ) -> Dict[str, float]:
        priors: Dict[str, float] = {}
        resolved_items = list(resolutions or self.resolver.resolve_result(llm_result))
        for index, item in enumerate(resolved_items):
            name = item.canonical_name
            if not name:
                continue
            rank_prior = max(0.65, 1.0 - index * 0.10)
            mapping_confidence = item.confidence * item.model_confidence
            prior = rank_prior * mapping_confidence
            priors[name] = max(priors.get(name, 0.0), prior)

        for chunk in rag_chunks:
            if chunk.get("type") != "disease_profile":
                continue
            resolution = self.resolver.resolve(chunk.get("title"))
            name = resolution.canonical_name
            if not name:
                continue
            try:
                score = float(chunk.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            priors[name] = max(priors.get(name, 0.0), score)
        return priors

    def _select_final(self, pool: Sequence[CandidateScore]) -> List[CandidateScore]:
        selected: List[CandidateScore] = []
        suppressed: Set[str] = set()
        for candidate in pool:
            if candidate.differential_only:
                continue
            if not candidate.trusted:
                continue
            if candidate.diagnosis in suppressed:
                continue
            entry = self.knowledge.get(candidate.diagnosis)
            parent = candidate.parent_diagnosis
            if parent:
                selected = [
                    item for item in selected
                    if item.diagnosis != parent
                    or self._can_keep_parent_with_specific_child(item)
                ]
            if self._is_generic_parent_of_selected(candidate, selected):
                self._mark_differential_only(
                    candidate,
                    "作为更泛化的父诊断保留鉴别，但已有更具体诊断且缺少独立状态证据，不作为最终诊断提交。",
                )
                continue
            if self._is_explained_secondary_manifestation(candidate, selected):
                self._mark_differential_only(
                    candidate,
                    "作为已选病因/结构诊断可解释的表现或并发状态保留鉴别，缺少独立客观证据，不作为最终诊断提交。",
                )
                continue
            if self._is_unrelated_high_residual(candidate, selected):
                self._mark_differential_only(
                    candidate,
                    self._differential_only_reason(candidate, selected),
                )
                continue
            if not self._is_final_companion_eligible(candidate, selected):
                self._mark_differential_only(
                    candidate,
                    self._differential_only_reason(candidate, selected),
                )
                continue
            selected.append(candidate)
            suppressed.update(str(item) for item in entry.get("suppress_diagnoses", []) or [])
            if len(selected) >= self.max_final_diagnoses:
                break
        return selected

    def _append_independent_states(
        self,
        selected: Sequence[CandidateScore],
        scores: Sequence[CandidateScore],
    ) -> List[CandidateScore]:
        result = list(selected)
        if len(result) >= self.max_final_diagnoses:
            return result
        if not any(self._is_causal_primary(item) for item in result):
            return result
        selected_names = {item.diagnosis for item in result}
        for candidate in scores:
            if candidate.differential_only:
                continue
            if candidate.diagnosis in selected_names:
                continue
            if self._is_generic_parent_of_selected(candidate, result):
                self._mark_differential_only(
                    candidate,
                    "作为更泛化的父诊断保留鉴别，但已有更具体诊断且缺少独立状态证据，不作为最终诊断提交。",
                )
                continue
            if not candidate.trusted:
                continue
            if candidate.hard_contradiction or not candidate.matched_evidence:
                continue
            if candidate.score < self.differential_threshold:
                continue
            if self._is_unrelated_high_residual(candidate, result):
                continue
            if not self._is_final_companion_eligible(candidate, result):
                continue
            if not (
                self._has_independent_state_evidence(candidate)
                or (
                    self._is_structural_comorbidity_candidate(candidate, result)
                    and self._has_authorized_independent_objective_evidence(candidate)
                )
            ):
                continue
            result.append(candidate)
            selected_names.add(candidate.diagnosis)
            if len(result) >= self.max_final_diagnoses:
                break
        return result

    def _is_final_companion_eligible(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        if not selected:
            return True
        if not candidate.trusted or candidate.hard_contradiction:
            return False
        if self._is_related_to_selected(candidate, selected):
            if self._is_secondary_manifestation(candidate):
                return self._has_independent_state_evidence(candidate)
            return True
        if (
            self._is_secondary_manifestation(candidate)
            and self._has_independent_state_evidence(candidate)
            and any(self._diagnoses_submission_related(candidate.diagnosis, item.diagnosis) for item in selected)
        ):
            return True
        return self._explains_selected_residual(candidate, selected)

    def _is_related_to_selected(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        return any(
            self._diagnoses_submission_related(candidate.diagnosis, item.diagnosis)
            for item in selected
        )

    def _explains_selected_residual(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        residual = set()
        for item in selected:
            residual.update(item.residual_evidence)
        if not residual:
            return False
        if not residual.intersection(candidate.matched_evidence):
            return False
        best_coverage = max(item.coverage_score for item in selected)
        best_residual = min(item.residual_score for item in selected)
        coverage_ok = candidate.coverage_score + 0.18 >= best_coverage
        residual_ok = candidate.residual_score <= best_residual + 0.18
        return coverage_ok and residual_ok

    def _mark_differential_only(self, candidate: CandidateScore, reason: str) -> None:
        candidate.differential_only = True
        candidate.differential_only_reason = reason

    def _differential_only_reason(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> str:
        primary = selected[0] if selected else None
        if not primary:
            return "作为鉴别保留，但未满足最终提交条件。"
        selected_residual = list(dict.fromkeys(
            item for diagnosis in selected for item in diagnosis.residual_evidence
        ))
        unexplained = [
            item for item in selected_residual
            if item not in set(candidate.matched_evidence)
        ][:3]
        evidence_gap = ""
        if "癌" in candidate.diagnosis or "肿瘤" in candidate.diagnosis:
            tumor_specific = any(
                token in evidence
                for evidence in candidate.matched_evidence
                for token in ("肿瘤", "癌", "占位", "结节", "肿块", "tumor")
            )
            if not tumor_specific and f"diagnosis:{candidate.diagnosis}" not in candidate.matched_evidence:
                evidence_gap = "且缺少肿瘤特异证据，"
        residual_text = "、".join(unexplained) if unexplained else "主诊断残余核心证据"
        return (
            f"作为鉴别保留，但相对{primary.diagnosis}解释力不足"
            f"（coverage {candidate.coverage_score:.2f} vs {primary.coverage_score:.2f}，"
            f"residual {candidate.residual_score:.2f} vs {primary.residual_score:.2f}），"
            f"未解释{residual_text}，{evidence_gap}且无明确因果/并发关系，不作为最终诊断提交。"
        )

    def _is_explained_secondary_manifestation(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        if not selected:
            return False
        if any(self._diagnosis_causes(item.diagnosis, candidate.diagnosis) for item in selected):
            return not self._has_independent_state_evidence(candidate)
        if not self._is_secondary_manifestation(candidate):
            return False
        if not any(self._is_causal_primary(item) for item in selected):
            return False
        return not self._has_independent_state_evidence(candidate)

    def _is_unrelated_high_residual(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        if not selected:
            return False
        if candidate.residual_score < self.residual_drop_threshold:
            return False
        if candidate.coverage_score >= self.evidence_gap_coverage_threshold:
            return False
        if self._is_secondary_manifestation(candidate) and self._has_independent_state_evidence(candidate):
            return False
        return not any(
            self._diagnoses_causally_related(candidate.diagnosis, item.diagnosis)
            or self._same_family_or_explicit_related(candidate.diagnosis, item.diagnosis)
            for item in selected
        )

    @staticmethod
    def _is_secondary_manifestation(candidate: CandidateScore) -> bool:
        dtype = candidate.diagnosis_type.lower()
        return (
            dtype in {"syndrome", "state", "complication"}
            or candidate.diagnosis in _SECONDARY_MANIFESTATION_DIAGNOSES
        )

    def _can_keep_parent_with_specific_child(self, candidate: CandidateScore) -> bool:
        return (
            self._is_secondary_manifestation(candidate)
            and self._has_independent_state_evidence(candidate)
        )

    def _is_generic_parent_of_selected(
        self,
        candidate: CandidateScore,
        selected: Sequence[CandidateScore],
    ) -> bool:
        if not selected:
            return False
        if not any(item.parent_diagnosis == candidate.diagnosis for item in selected):
            return False
        return not self._can_keep_parent_with_specific_child(candidate)

    @staticmethod
    def _is_causal_primary(candidate: CandidateScore) -> bool:
        dtype = candidate.diagnosis_type.lower()
        return dtype in {"etiology", "metabolic", "structural"}

    @staticmethod
    def _has_independent_state_evidence(candidate: CandidateScore) -> bool:
        if candidate.diagnosis == "心律失常":
            return any(
                finding in candidate.matched_evidence
                for finding in ("symptom:晕厥", "symptom:低血压", "symptom:持续心动过速")
            )
        if f"diagnosis:{candidate.diagnosis}" in candidate.matched_evidence:
            return True
        if candidate.diagnosis != "心力衰竭":
            return False
        matched = set(candidate.matched_evidence or [])
        return DiagnosisDecisionEngine._heart_failure_state_evidence(matched)

    @staticmethod
    def _heart_failure_state_evidence(matched: set[str]) -> bool:
        if "heart_failure_state" in matched:
            return True
        congestion = bool(
            matched
            & {
                "fluid_retention_pattern",
                "leg_edema",
                "symptom:下肢水肿",
                "symptom:脚踝水肿",
            }
        )
        positional_dyspnea = bool(
            matched
            & {
                "orthopnea",
                "paroxysmal_nocturnal_dyspnea",
                "symptom:端坐呼吸",
                "symptom:夜间阵发性呼吸困难",
            }
        )
        return congestion and positional_dyspnea

    def _diagnosis_causes(self, cause: str, effect: str) -> bool:
        cause_entry = self.knowledge.get(cause)
        effect_entry = self.knowledge.get(effect)
        return (
            effect in set(str(item) for item in cause_entry.get("causes", []) or [])
            or cause in set(str(item) for item in effect_entry.get("caused_by", []) or [])
        )

    def _diagnoses_causally_related(self, left: str, right: str) -> bool:
        if left == right:
            return True
        return self._diagnosis_causes(left, right) or self._diagnosis_causes(right, left)

    def _diagnoses_submission_related(self, left: str, right: str) -> bool:
        if self._diagnoses_causally_related(left, right):
            return True
        left_entry = self.knowledge.get(left)
        right_entry = self.knowledge.get(right)
        left_related = set(str(item) for item in left_entry.get("related_complications", []) or [])
        right_related = set(str(item) for item in right_entry.get("related_complications", []) or [])
        if right in left_related or left in right_related:
            return True
        left_system = str(left_entry.get("body_system") or "")
        right_system = str(right_entry.get("body_system") or "")
        left_family = str(left_entry.get("disease_family") or left_entry.get("family") or "")
        right_family = str(right_entry.get("disease_family") or right_entry.get("family") or "")
        if left_system and left_system == right_system and left_family and left_family == right_family:
            return True
        left_parent = str(left_entry.get("parent_diagnosis") or "")
        right_parent = str(right_entry.get("parent_diagnosis") or "")
        return (
            bool(left_parent and left_parent == right)
            or bool(right_parent and right_parent == left)
            or bool(left_parent and right_parent and left_parent == right_parent)
        )

    def _annotate_causal_relations(
        self,
        scores: Sequence[CandidateScore],
        selected: Sequence[CandidateScore],
    ) -> None:
        selected_names = [item.diagnosis for item in selected]
        for candidate in scores:
            if candidate.diagnosis in selected_names:
                candidate.causal_relation_to_selected = "selected"
                continue
            relation = ""
            for name in selected_names:
                if self._diagnosis_causes(name, candidate.diagnosis):
                    relation = f"caused_by:{name}"
                    break
                if self._diagnosis_causes(candidate.diagnosis, name):
                    relation = f"causes:{name}"
                    break
                if self._diagnoses_submission_related(candidate.diagnosis, name):
                    relation = f"related:{name}"
                    break
            candidate.causal_relation_to_selected = relation or (
                "unrelated_to_selected" if selected_names else ""
            )

    def _fallback_candidate(
        self,
        priors: Dict[str, float],
        scores: Sequence[CandidateScore],
    ) -> Optional[CandidateScore]:
        for name, _ in sorted(priors.items(), key=lambda item: item[1], reverse=True):
            for score in scores:
                if (
                    score.diagnosis == name
                    and not score.hard_contradiction
                    and score.matched_evidence
                    and score.trusted
                ):
                    return score
        # Do not silently turn an all-zero evidence set into whichever disease
        # happens to be first in the catalog. The caller can ask for more data
        # or submit a controlled low-confidence candidate supplied by LLM/RAG.
        return None

    @staticmethod
    def _reasoning(selected: Sequence[CandidateScore], unexplained: Sequence[str]) -> str:
        if not selected:
            return "证据裁决未找到可提交的标准诊断。"
        parts = []
        for item in selected:
            evidence = "、".join(item.matched_evidence[:6]) or "有限临床证据"
            parts.append(
                f"{item.diagnosis}由{evidence}支持"
                f"（证据评分{item.score:.2f}，解释覆盖{item.coverage_score:.2f}，残余{item.residual_score:.2f}）"
            )
        text = "证据裁决：" + "；".join(parts) + "。"
        if unexplained:
            text += "仍需关注未完全解释的证据：" + "、".join(unexplained[:6]) + "。"
        return text


def _matching_observations(
    spec: Dict[str, Any],
    observations: Sequence[Observation],
    polarity: str = "positive",
) -> List[Observation]:
    return [
        item for item in observations
        if item.polarity == polarity
        and not (polarity == "positive" and getattr(item, "shadowed_by", ""))
        and _observation_matches(spec, item)
    ]


def _observation_matches(spec: Dict[str, Any], item: Observation) -> bool:
    finding = str(spec.get("finding") or "")
    if finding and finding != item.finding:
        return False
    direction = str(spec.get("direction") or "")
    if direction and direction != item.direction:
        return False
    source_contains = str(spec.get("source_contains") or "")
    if source_contains and source_contains.lower() not in item.source.lower():
        return False
    terms = spec.get("terms") or []
    if isinstance(terms, str):
        terms = [terms]
    if terms and not any(str(term).lower() in item.raw_text.lower() for term in terms):
        return False
    if spec.get("min_value") is not None:
        if item.value is None or item.value < float(spec["min_value"]):
            return False
    if spec.get("max_value") is not None:
        if item.value is None or item.value > float(spec["max_value"]):
            return False
    return bool(finding or direction or source_contains or terms or spec.get("min_value") is not None or spec.get("max_value") is not None)


def _information_multiplier(item: Observation) -> float:
    try:
        value = float(getattr(item, "information_value", 0.0) or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0.0:
        return 1.0
    return max(0.35, min(1.45, 0.65 + value))


def _coerce_profile_evidence_spec(value: Any, default_weight: float) -> Dict[str, Any]:
    if isinstance(value, dict):
        spec = dict(value)
    else:
        text = str(value or "").strip()
        spec = {"terms": [text]} if text else {}
    if not spec:
        return {}
    if not (spec.get("finding") or spec.get("terms") or spec.get("source_contains")):
        return {}
    spec["weight"] = float(spec.get("weight", default_weight) or default_weight)
    return spec


def _coerce_spec(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"finding": str(value)}


def _render_required_group(group: Sequence[Any]) -> str:
    labels: List[str] = []
    for item in group:
        spec = _coerce_spec(item)
        label = str(spec.get("finding") or "")
        if not label and spec.get("terms"):
            terms = spec.get("terms")
            if isinstance(terms, str):
                terms = [terms]
            label = "/".join(str(term) for term in terms[:3])
        if not label and spec.get("source_contains"):
            label = f"source:{spec.get('source_contains')}"
        if label:
            labels.append(label)
    return "|".join(labels) or "required_evidence"


def _compact_text(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _dedupe_specs(items: Iterable[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        spec = _coerce_spec(item)
        key = json.dumps(spec, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(spec)
    return result


def _dedupe_objects(items: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    seen = set()
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return default
