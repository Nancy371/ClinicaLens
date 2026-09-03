"""Replayable diagnosis judge and submitter.

The retriever/ranker owns candidate generation. The judge only arbitrates an
existing candidate table into primary, secondary, differential, and evidence-gap
roles. The submitter then writes that authorization back to DiagnosisDecision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .case_board import StaleJudgeDecisionError, judge_decision_is_stale
from .claim_resolution import hydrate_gap_with_claim_state
from .diagnosis_eligibility import (
    DEFERRED,
    DEFERRED_NEEDS_CONFIRMATORY_EXAM,
    DEFERRED_NEEDS_DERIVED_PATTERN,
    DEFERRED_NEEDS_OBSERVED_EVIDENCE,
    DIFFERENTIAL_ONLY,
    EXCLUDED,
    PRIMARY_ELIGIBLE,
)
from .clinical_pattern_bridge import BRIDGE_REASON, CROSS_SYSTEM_SCOPE, has_active_bridge_protection
from .clinical_reasoning_comparator import (
    ClinicalReasoningComparator,
    KEEP_CURRENT_AND_DEFER_CONTENDER,
    KEEP_CURRENT_PRIMARY,
    NO_MATERIAL_DIFFERENCE,
    REJECT_CONTENDER,
    SWITCH_PRIMARY,
    UNLOCK_AND_DEFER,
)
from .exam_resolver import ALIAS, EQUIVALENT, EXACT, PARTIAL_SUBSTITUTE, ExamResolver


_DEFERRED_EXAM_OVERRIDE_SUBSTATUSES = {
    DEFERRED_NEEDS_CONFIRMATORY_EXAM,
    DEFERRED_NEEDS_DERIVED_PATTERN,
    DEFERRED_NEEDS_OBSERVED_EVIDENCE,
}
_FULL_CLOSURE_RESOLUTION_TYPES = {EXACT, ALIAS, EQUIVALENT}
_USABLE_CLOSURE_RESOLUTION_TYPES = {
    EXACT,
    ALIAS,
    EQUIVALENT,
    PARTIAL_SUBSTITUTE,
}

LIFECYCLE_WORKUP_REQUIRED = "WORKUP_REQUIRED"
LIFECYCLE_READY_FOR_ARBITRATION = "READY_FOR_ARBITRATION"
LIFECYCLE_PRIMARY = "PRIMARY"
LIFECYCLE_SECONDARY = "SECONDARY"
LIFECYCLE_REJECTED = "REJECTED"
LIFECYCLE_DIFFERENTIAL_ONLY = "DIFFERENTIAL_ONLY"

INVARIANT_VALID = "VALID"
INVARIANT_DEADLOCK = "DEADLOCK"
INVARIANT_INCONSISTENT = "INCONSISTENT"

REASON_PRIMARY_ELIGIBLE = "PRIMARY_ELIGIBLE"
REASON_ANCHOR_SATISFIED = "ANCHOR_SATISFIED"
REASON_PROTECTED_CONTENDER = "PROTECTED_CONTENDER"
REASON_PAIRWISE_ALLOWED = "PAIRWISE_ALLOWED"


def _compact_text(value: Any) -> str:
    return "".join(str(value or "").strip().lower().split())


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


_PRIORITY_TYPES = {"etiology", "metabolic", "structural", "systemic"}
_MANIFESTATION_TYPES = {"syndrome", "state", "complication"}
_MANIFESTATION_NAMES = {"心力衰竭", "心律失常", "肺动脉高压"}
_SYSTEMIC_PRIMARY_NAMES = {"卵巢过度刺激综合征", "门静脉高压", "终末期肾病"}
_PARENT_FALLBACK_NAMES = {"先天性心脏病"}
_DIFFERENTIAL_EXAM_HINTS = {
    "雅司病": ["体格检查", "梅毒血清学检查", "暗视野显微镜检查", "组织病理学检查"],
    "湿疹": ["体格检查", "血清学抗体检测"],
    "白血病": [
        "全血细胞计数（CBC）",
        "外周血涂片",
        "骨髓穿刺和活检（BMAB）",
        "流式细胞术免疫分型",
        "白血病融合基因检测",
    ],
    "肺结核": ["胸部CT扫描（Chest CT）", "痰培养", "抗酸杆菌染色（AFB）", "核酸扩增检测（NAAT）"],
    "肺炎": ["胸部X线检查（CXR）", "全血细胞计数（CBC）", "C反应蛋白（CRP）", "痰培养"],
    "支气管肺炎": ["胸部X线检查（CXR）", "全血细胞计数（CBC）", "C反应蛋白（CRP）", "痰培养"],
    "肺癌": ["胸部CT扫描（Chest CT）", "组织病理学检查", "支气管镜检查"],
    "显微镜下多血管炎": [
        "胸部CT扫描（Chest CT）",
        "尿液分析（UA）",
        "肾功能检查（RFTs）",
        "抗中性粒细胞胞质抗体（ANCA）谱",
        "MPO-ANCA",
    ],
    "老视": ["视力检查", "屈光检查"],
    "晶状体脱位": ["裂隙灯检查", "眼压测量", "眼部B超检查"],
    "青光眼": ["眼压测量", "裂隙灯检查", "眼底镜检查"],
}
_DIFFERENTIAL_SET_EXAM_HINTS = [
    (
        {"雅司病", "湿疹", "白血病"},
        ["全血细胞计数（CBC）", "外周血涂片", "梅毒血清学检查", "体格检查"],
    ),
    (
        {"肺结核", "肺炎", "肺癌"},
        ["胸部CT扫描（Chest CT）", "痰培养", "抗酸杆菌染色（AFB）", "核酸扩增检测（NAAT）"],
    ),
    (
        {"肺结核", "支气管肺炎", "肺癌"},
        ["胸部CT扫描（Chest CT）", "痰培养", "抗酸杆菌染色（AFB）", "核酸扩增检测（NAAT）"],
    ),
    (
        {"肺结核", "显微镜下多血管炎", "肺癌"},
        [
            "胸部CT扫描（Chest CT）",
            "痰培养",
            "抗酸杆菌染色（AFB）",
            "核酸扩增检测（NAAT）",
            "抗中性粒细胞胞质抗体（ANCA）谱",
            "尿液分析（UA）",
            "肾功能检查（RFTs）",
        ],
    ),
    (
        {"老视", "晶状体脱位", "青光眼"},
        ["视力检查", "屈光检查", "裂隙灯检查", "眼压测量"],
    ),
]
_OBJECTIVE_GAP_FINDINGS = {
    "ascites",
    "bilirubin_high",
    "dextrocardia",
    "egfr_low",
    "hemoconcentration",
    "hyperkalemia",
    "hypoalbuminemia",
    "left_to_right_shunt",
    "low_magnesium",
    "magnesium_depletion",
    "magnesium_load_retention_high",
    "oliguria",
    "portal_flow_abnormal",
    "portal_vein_dilation",
    "pulmonary_valve_gradient",
    "pulmonary_valve_stenosis",
    "renal_impairment",
    "right_heart_strain",
    "splenomegaly",
    "treponema_positive",
    "treponemal_skin_lesion",
    "tuberculosis_exposure",
    "hemoptysis",
    "urea_elevated",
    "uremia",
    "ventricular_septal_defect",
}

_BROAD_EVIDENCE_TOKENS = {
    "acute_course",
    "chronic_course",
    "congenital_onset",
    "fever",
    "cough",
    "fatigue",
    "pain",
    "pruritus",
    "rash",
    "dyspnea",
    "visual_blurring",
    "abdominal_pain",
    "symptom:发热",
    "symptom:咳嗽",
    "symptom:乏力",
    "symptom:疼痛",
    "symptom:皮疹",
    "symptom:瘙痒",
}
_CORE_EVIDENCE_TOKENS = _OBJECTIVE_GAP_FINDINGS | {
    "abnormal_genitalia",
    "age_related_near_blur",
    "ambiguous_genitalia",
    "bone_pain",
    "cardiopulmonary_exertional_pattern",
    "crusted_exudative_skin_ulcer",
    "crusted_skin_lesion",
    "deep_skin_ulcer",
    "dermatomal_vesicles",
    "diagnostic_imaging",
    "dyspnea_on_exertion",
    "exercise_intolerance",
    "fluid_retention_pattern",
    "hemoptysis",
    "lens_dislocation",
    "midline_suprapubic_cyst",
    "midline_suprapubic_pain",
    "near_vision_difficulty",
    "night_vision_decline",
    "nyctalopia_pattern",
    "optic_pressure_high",
    "periorbital_edema",
    "periostitis",
    "polydipsia",
    "postprandial_nausea",
    "presbyopia_refraction",
    "presbyopia_pattern",
    "refractive_error",
    "refractive_correction_improves_near_vision",
    "regional_lymphadenopathy",
    "rural_child_contact",
    "sex_development_disorder",
    "treponemal_disease_pattern",
    "tropical_exposure",
    "tb_exposure",
    "tuberculosis_pattern",
    "tuberculosis_exposure",
    "umbilical_discharge",
    "umbilical_mass",
    "urachal_remnant_pattern",
    "urachal_cyst_imaging",
    "vesicular_rash",
}
_CONTEXTUAL_CORE_FINDINGS = {
    "bradycardia",
    "crusted_exudative_skin_ulcer",
    "anca_positive",
    "mpo_anca_positive",
    "p_anca_positive",
    "microscopic_hematuria",
    "proteinuria",
    "pulmonary_hemorrhage",
    "low_magnesium",
    "low_urine_magnesium",
    "magnesium_depletion",
    "magnesium_load_retention_high",
    "regional_lymphadenopathy",
    "rural_child_contact",
    "near_vision_difficulty",
    "age_related_near_blur",
    "refractive_correction_improves_near_vision",
    "presbyopia_pattern",
    "night_vision_decline",
    "nyctalopia_pattern",
    "midline_suprapubic_pain",
    "urachal_remnant_pattern",
    "tb_exposure",
    "tuberculosis_pattern",
    "tuberculosis_exposure",
}
_CORE_SYMPTOM_KEYWORDS = (
    "脐部",
    "脐周",
    "脐下",
    "外生殖器",
    "尿道下裂",
    "隐睾",
    "看近",
    "阅读困难",
    "视物模糊",
    "血痰",
    "咯血",
    "盗汗",
    "骨痛",
    "关节痛",
    "结痂",
    "渗出",
    "黄水",
    "腹股沟",
    "水疱",
    "疱疹",
)
_GENERIC_EVIDENCE_PREFIXES = ("field:",)
_GENERIC_INFLAMMATION_EXAM_MARKERS = (
    "CBC",
    "CRP",
    "ESR",
    "PCT",
    "全血细胞计数",
    "C反应蛋白",
    "红细胞沉降率",
    "降钙素原",
)
_SPECIAL_DISCRIMINATOR_EXAM_MARKERS = (
    "AFB",
    "NAAT",
    "Xpert",
    "ANCA",
    "MPO",
    "p-ANCA",
    "CT",
    "MRI",
    "血清学",
    "涂片",
    "病理",
    "活检",
    "支气管镜",
    "尿液分析",
    "肾功能",
    "屈光",
    "眼压",
    "裂隙灯",
    "痰培养",
)
_KNOWN_DIFFERENTIAL_GROUPS = (
    (
        "dermatology_eruptive_systemic",
        {"雅司病", "水痘", "湿疹", "白血病", "尖锐湿疣"},
    ),
    (
        "pulmonary_infection_mass",
        {"肺结核", "肺炎", "支气管肺炎", "肺癌", "肺隐球菌病", "肺念珠菌病", "支原体肺炎"},
    ),
    (
        "ophthalmology_visual",
        {"老视", "晶状体脱位", "青光眼", "白内障", "虹膜缺损"},
    ),
    (
        "congenital_genitourinary_dsd",
        {"卵睾性别发育异常（Ovotesticular DSD）", "X三体综合征（47,XXX）"},
    ),
    (
        "urachal_midline_urinary",
        {"脐尿管囊肿", "泌尿系感染", "尿道综合征", "急性细菌性前列腺炎"},
    ),
)
_KNOWN_CLUSTER_BY_NAME = {
    name: cluster
    for cluster, names in _KNOWN_DIFFERENTIAL_GROUPS
    for name in names
}
_FALLBACK_DISEASE_METADATA = {
    "门静脉高压": ("gastrointestinal", "portal_hypertension"),
    "白血病": ("hematology", "leukemia"),
    "带状疱疹": ("dermatology_infectious", "dermatomal_viral"),
    "骨折": ("musculoskeletal", "acute_trauma"),
    "前列腺增生": ("genitourinary", "prostate_obstruction"),
    "泌尿系感染": ("genitourinary", "urinary_tract_infection"),
    "尿道综合征": ("genitourinary", "urethral_syndrome"),
    "肺隐球菌病": ("respiratory", "opportunistic_fungal_pneumonia"),
    "肺念珠菌病": ("respiratory", "opportunistic_fungal_pneumonia"),
    "肺不张": ("respiratory", "atelectasis"),
    "二尖瓣反流": ("cardiovascular", "valvular_left_heart"),
    "三尖瓣反流": ("cardiovascular", "valvular_right_heart"),
    "心力衰竭": ("cardiovascular", "heart_failure_state"),
    "晶状体脱位": ("ophthalmology", "ocular_structural"),
    "青光眼": ("ophthalmology", "glaucoma"),
    "白内障": ("ophthalmology", "lens_opacity"),
}


@dataclass
class EvidenceGapValue:
    gap_id: str
    entity_id: str
    candidate: str
    target_evidence: str
    gap_type: str
    gap_value: float
    gap_value_components: Dict[str, float] = field(default_factory=dict)
    expected_transition: Dict[str, Any] = field(default_factory=dict)
    closure_exams: List[str] = field(default_factory=list)
    hard_contradiction: bool = False
    already_attempted_exams: List[str] = field(default_factory=list)
    candidate_score_at_decision: float = 0.0
    score_gap_decoupled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.metadata or {})
        data = asdict(self)
        metadata = data.pop("metadata", {})
        payload.update(metadata)
        payload.update(data)
        return payload


@dataclass
class JudgeCandidateReview:
    diagnosis: str
    role: str
    reason: str
    entity_id: str = ""
    canonical_name: str = ""
    submission_name: str = ""
    submittable: bool = True
    score: float = 0.0
    judge_score: float = 0.0
    required_met: bool = False
    required_gap_authorized: bool = False
    hard_contradiction: bool = False
    coverage_score: float = 0.0
    residual_score: float = 0.0
    explanatory_coverage: float = 0.0
    core_explanatory_coverage: float = 0.0
    residual_evidence_score: float = 0.0
    residual_core_evidence_count: int = 0
    diagnosis_type: str = ""
    specificity: float = 0.0
    required_gaps: List[str] = field(default_factory=list)
    matched_evidence: List[str] = field(default_factory=list)
    explained_evidence: List[str] = field(default_factory=list)
    unexplained_core_evidence: List[str] = field(default_factory=list)
    explanatory_rank_reason: str = ""
    required_gap_state: str = ""
    eligibility_status: str = ""
    eligibility_reason: str = ""
    eligibility_substatus: str = ""
    eligibility_anchor_status: str = ""
    eligibility_anchor_policy_audit: Dict[str, Any] = field(default_factory=dict)
    missing_required_anchors: List[str] = field(default_factory=list)
    evidence_pattern_matches: List[Dict[str, Any]] = field(default_factory=list)
    clinical_pattern_matches: List[Dict[str, Any]] = field(default_factory=list)
    derived_pattern_assertions: List[Dict[str, Any]] = field(default_factory=list)
    bridge_validation_results: List[Dict[str, Any]] = field(default_factory=list)
    bridge_protection_decisions: List[Dict[str, Any]] = field(default_factory=list)
    exam_followup_authorized: bool = False
    submission_authorized: bool = False
    evidence_gaps: List[Dict[str, Any]] = field(default_factory=list)
    gap_values: List[Dict[str, Any]] = field(default_factory=list)
    max_gap_value: float = 0.0
    actionable_gap_count: int = 0
    deferred_priority: float = 0.0
    deferred_priority_components: Dict[str, float] = field(default_factory=dict)
    exam_priority_override: bool = False
    exam_priority_override_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeDecision:
    retriever_top1: str = ""
    retriever_top1_entity_id: str = ""
    judge_primary: str = ""
    judge_primary_entity_id: str = ""
    primary: str = ""
    primary_entity_id: str = ""
    primary_status: str = "locked"
    needs_discriminating_exams: bool = False
    provisional_primary: str = ""
    locked_primary: str = ""
    defer_reason: str = ""
    pre_discrimination_primary: str = ""
    fallback_primary: str = ""
    fallback_reason: str = ""
    discrimination_attempted: bool = False
    discrimination_resolved: bool = False
    fallback_to_pre_discrimination_primary: bool = False
    differential_pool_source: Dict[str, str] = field(default_factory=dict)
    secondary: List[str] = field(default_factory=list)
    secondary_entity_ids: List[str] = field(default_factory=list)
    differential: List[str] = field(default_factory=list)
    differential_entity_ids: List[str] = field(default_factory=list)
    evidence_gap_targets: List[str] = field(default_factory=list)
    evidence_gap_target_entity_ids: List[str] = field(default_factory=list)
    final_diagnoses: List[str] = field(default_factory=list)
    final_entity_ids: List[str] = field(default_factory=list)
    required_gap_authorized_diagnoses: List[str] = field(default_factory=list)
    blocked_diagnoses: List[Dict[str, Any]] = field(default_factory=list)
    reviews: List[JudgeCandidateReview] = field(default_factory=list)
    differential_candidates: List[str] = field(default_factory=list)
    pairwise_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    excluded_from_pairwise: List[Dict[str, Any]] = field(default_factory=list)
    pool_filter_reasons: Dict[str, str] = field(default_factory=dict)
    cluster_assignments: Dict[str, str] = field(default_factory=dict)
    pairwise_allowed_matrix: List[Dict[str, Any]] = field(default_factory=list)
    core_evidence_by_candidate: Dict[str, List[str]] = field(default_factory=dict)
    generic_evidence_by_candidate: Dict[str, List[str]] = field(default_factory=dict)
    pool_filter_summary: Dict[str, Any] = field(default_factory=dict)
    discriminating_findings: List[str] = field(default_factory=list)
    discriminating_exams: List[str] = field(default_factory=list)
    discriminating_exam_tasks: List[Dict[str, Any]] = field(default_factory=list)
    required_gap_by_candidate: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    high_value_gap_candidates: List[str] = field(default_factory=list)
    active_evidence_gaps: List[Dict[str, Any]] = field(default_factory=list)
    deferred_evidence_gaps: List[Dict[str, Any]] = field(default_factory=list)
    exam_priority_overrides: List[Dict[str, Any]] = field(default_factory=list)
    deferred_gap_closure_tasks: List[Dict[str, Any]] = field(default_factory=list)
    deferred_gap_closure_exam_coverage: float = 0.0
    exam_priority_alignment: float = 0.0
    wrong_primary_exam_drift: float = 0.0
    explanatory_coverage: float = 0.0
    core_explanatory_coverage: float = 0.0
    residual_evidence_score: float = 0.0
    residual_core_evidence_count: int = 0
    dynamic_rerank_trace: List[Dict[str, Any]] = field(default_factory=list)
    decision_override: bool = False
    required_gap_state_by_candidate: Dict[str, str] = field(default_factory=dict)
    gap_state_distribution: Dict[str, int] = field(default_factory=dict)
    primary_unlock_reason: str = ""
    explanation_score_changed_ranking: bool = False
    evidence_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    conflict_affected_diagnoses: List[str] = field(default_factory=list)
    root_cause_arbitration: Dict[str, Any] = field(default_factory=dict)
    root_cause_primary: str = ""
    root_cause_secondary: List[str] = field(default_factory=list)
    root_cause_primary_override: bool = False
    root_cause_coverage: float = 0.0
    candidate_explanation_edges: List[Dict[str, Any]] = field(default_factory=list)
    primary_override_source: str = ""
    clinical_reasoning_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    contender_admission_audit: List[Dict[str, Any]] = field(default_factory=list)
    material_contender_filter: List[Dict[str, Any]] = field(default_factory=list)
    candidate_disposition_audit: List[Dict[str, Any]] = field(default_factory=list)
    candidate_lifecycle_transitions: List[Dict[str, Any]] = field(default_factory=list)
    lifecycle_recoveries: List[Dict[str, Any]] = field(default_factory=list)
    arbitration_deadlocks: List[Dict[str, Any]] = field(default_factory=list)
    primary_arbitration_candidates: List[Dict[str, Any]] = field(default_factory=list)
    primary_arbitration_decision: Dict[str, Any] = field(default_factory=dict)
    primary_arbitration_summary: Dict[str, Any] = field(default_factory=dict)
    primary_anchor_revalidation: Dict[str, Any] = field(default_factory=dict)
    arbitration_winner: str = ""
    arbitration_loser: str = ""
    arbitration_action: str = ""
    arbitration_reason_codes: List[str] = field(default_factory=list)
    pairwise_discriminating_gaps: List[Dict[str, Any]] = field(default_factory=list)
    eligibility_distribution: Dict[str, int] = field(default_factory=dict)
    deferred_substatus_distribution: Dict[str, int] = field(default_factory=dict)
    deferred_anchor_candidates: List[str] = field(default_factory=list)
    excluded_candidates: List[str] = field(default_factory=list)
    primary_eligible_candidates: List[str] = field(default_factory=list)
    entity_resolutions: List[Dict[str, Any]] = field(default_factory=list)
    clinical_pattern_matches: List[Dict[str, Any]] = field(default_factory=list)
    derived_pattern_assertions: List[Dict[str, Any]] = field(default_factory=list)
    bridge_validation_results: List[Dict[str, Any]] = field(default_factory=list)
    bridge_protection_decisions: List[Dict[str, Any]] = field(default_factory=list)
    bridge_protected_candidates: List[str] = field(default_factory=list)
    bridge_pairwise_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    bridge_candidate_final_dispositions: List[Dict[str, Any]] = field(default_factory=list)
    bridge_generated_gaps: List[Dict[str, Any]] = field(default_factory=list)
    case_version: int = 0
    evidence_snapshot_hash: str = ""
    knowledge_profile_version: str = ""
    decision_policy_version: str = ""
    exam_catalog_version: str = ""
    stale_decision: bool = False
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["reviews"] = [item.to_dict() for item in self.reviews]
        return data


@dataclass
class DifferentialPoolFilterResult:
    candidates: List[Any] = field(default_factory=list)
    excluded: List[Dict[str, Any]] = field(default_factory=list)
    pool_filter_reasons: Dict[str, str] = field(default_factory=dict)
    cluster_assignments: Dict[str, str] = field(default_factory=dict)
    pairwise_allowed_matrix: List[Dict[str, Any]] = field(default_factory=list)
    core_evidence_by_candidate: Dict[str, List[str]] = field(default_factory=dict)
    generic_evidence_by_candidate: Dict[str, List[str]] = field(default_factory=dict)
    pool_source: Dict[str, str] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


class DifferentialPoolFilter:
    """Filter Top20 candidates into clinically meaningful pairwise DDx."""

    def __init__(self, judge: "DiagnosisJudge"):
        self.judge = judge

    def filter(
        self,
        pool: Sequence[Any],
        source_by_name: Optional[Dict[str, str]] = None,
        force_names: Optional[Sequence[str]] = None,
    ) -> DifferentialPoolFilterResult:
        source_by_name = dict(source_by_name or {})
        force_names = {
            str(item or "").strip()
            for item in (force_names or [])
            if str(item or "").strip()
        }
        candidates = [item for item in pool or [] if item]
        result = DifferentialPoolFilterResult()
        if not candidates:
            return result

        relevance = {self._name(item): self._relevance(item) for item in candidates}
        result.cluster_assignments = {
            self._name(item): relevance[self._name(item)]["cluster"]
            for item in candidates
        }
        result.core_evidence_by_candidate = {
            self._name(item): list(relevance[self._name(item)]["core"])
            for item in candidates
        }
        result.generic_evidence_by_candidate = {
            self._name(item): list(relevance[self._name(item)]["generic"])
            for item in candidates
        }

        dominant_clusters = self._dominant_clusters(candidates, relevance)
        retained: List[Any] = []
        excluded: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            name = self._name(candidate)
            if name in force_names and not getattr(candidate, "hard_contradiction", False):
                retained.append(candidate)
                result.pool_filter_reasons[name] = (
                    BRIDGE_REASON
                    if self._bridge_protected(candidate)
                    else "protected_recall_arbitration"
                    if self._protected_recall(candidate)
                    else "forced_conflict_or_top_candidate"
                )
                continue
            reason = self._keep_reason(
                candidate,
                index,
                candidates,
                retained,
                relevance,
                dominant_clusters,
            )
            if reason:
                retained.append(candidate)
                result.pool_filter_reasons[name] = reason
                continue
            exclude_reason = self._exclude_reason(
                candidate,
                relevance,
                dominant_clusters,
            )
            excluded.append(
                {
                    "diagnosis": name,
                    "reason": exclude_reason,
                    "cluster": relevance[name]["cluster"],
                    "core_evidence": list(relevance[name]["core"])[:6],
                    "generic_evidence": list(relevance[name]["generic"])[:6],
                    "source": source_by_name.get(name, ""),
                }
            )

        if not retained:
            retained = [candidates[0]]
            result.pool_filter_reasons[self._name(candidates[0])] = "fallback_top_candidate"

        retained = self._limit_pool(
            retained,
            relevance,
            candidates,
            force_names=force_names,
        )
        retained_names = {self._name(item) for item in retained}
        for candidate in candidates:
            name = self._name(candidate)
            if name in retained_names:
                continue
            if any(item.get("diagnosis") == name for item in excluded):
                continue
            excluded.append(
                {
                    "diagnosis": name,
                    "reason": "pool_size_limit",
                    "cluster": relevance[name]["cluster"],
                    "core_evidence": list(relevance[name]["core"])[:6],
                    "generic_evidence": list(relevance[name]["generic"])[:6],
                    "source": source_by_name.get(name, ""),
                }
            )

        matrix = self._allowed_matrix(retained, relevance)
        result.candidates = retained
        result.excluded = excluded
        result.pairwise_allowed_matrix = matrix
        result.pool_source = {
            self._name(item): source_by_name.get(self._name(item), "filtered")
            for item in retained
        }
        result.summary = self._summary(candidates, retained, excluded, matrix, relevance)
        return result

    def allowed_pair_names(
        self,
        result: DifferentialPoolFilterResult,
    ) -> set[tuple[str, str]]:
        allowed: set[tuple[str, str]] = set()
        for item in result.pairwise_allowed_matrix:
            if item.get("allowed"):
                left = str(item.get("left") or "")
                right = str(item.get("right") or "")
                if left and right:
                    allowed.add(tuple(sorted((left, right))))
        return allowed

    def _keep_reason(
        self,
        candidate: Any,
        index: int,
        candidates: Sequence[Any],
        retained: Sequence[Any],
        relevance: Dict[str, Dict[str, Any]],
        dominant_clusters: set[str],
    ) -> str:
        name = self._name(candidate)
        data = relevance[name]
        if getattr(candidate, "hard_contradiction", False):
            return ""
        if self._bridge_protected(candidate):
            return BRIDGE_REASON
        if self._high_explanatory_candidate(candidate, data):
            return "high_explanatory_primary_candidate"
        if self._direct_diagnosis(candidate):
            non_diagnosis_core = [
                item for item in data["core"] if not str(item).startswith("diagnosis:")
            ]
            if (
                dominant_clusters
                and data["cluster"] not in dominant_clusters
                and (self._direct_low_explainability(candidate) or not non_diagnosis_core)
            ):
                return ""
            if not non_diagnosis_core and index >= max(1, self.judge.differential_top_k):
                return ""
            if (
                getattr(candidate, "required_met", False)
                or non_diagnosis_core
                or self.judge._objective_signal(candidate)
            ):
                return "direct_diagnosis_evidence"
            if (
                not dominant_clusters
                or data["cluster"] in dominant_clusters
                or any(
                    self._can_form_differential(candidate, item, relevance)
                    for item in retained
                )
            ):
                return "direct_diagnosis_evidence"
            return ""
        if data["core"]:
            if (
                float(getattr(candidate, "source_prior", 0.0) or 0.0) >= 0.75
                and float(getattr(candidate, "coverage_score", 0.0) or 0.0) >= 0.55
                and self.judge._residual(candidate) <= 0.45
            ):
                return "high_explanatory_source_candidate"
            if data["cluster"] in dominant_clusters:
                return "dominant_cluster_core_evidence"
            if not dominant_clusters and self._strong_specific_candidate(candidate):
                return "standalone_core_specific_candidate"
            if any(
                self._can_form_differential(candidate, item, relevance)
                for item in retained
            ):
                return "clinically_related_core_evidence"
        if index < max(2, self.judge.differential_top_k):
            if data["cluster"] in dominant_clusters:
                return "top_k_same_dominant_cluster"
            if any(
                self._can_form_differential(candidate, item, relevance)
                for item in retained
            ):
                return "top_k_clinically_related"
            if not dominant_clusters and data["generic"]:
                return "top_k_no_dominant_cluster"
        if self._strong_specific_candidate(candidate):
            if any(
                self._can_form_differential(candidate, item, relevance)
                for item in retained
            ):
                return "top20_specific_related_tail"
            if data["core"] and (
                not dominant_clusters or data["cluster"] in dominant_clusters
            ) and not any(
                relevance[self._name(item)]["core"] for item in retained
            ):
                return "top20_specific_core_tail"
        return ""

    def _exclude_reason(
        self,
        candidate: Any,
        relevance: Dict[str, Dict[str, Any]],
        dominant_clusters: set[str],
    ) -> str:
        name = self._name(candidate)
        data = relevance[name]
        if getattr(candidate, "hard_contradiction", False):
            return "negative_feature"
        if self._bridge_protected(candidate):
            return BRIDGE_REASON
        if not data["core"] and data["generic"]:
            return "generic_only_evidence"
        if data["cluster"] not in dominant_clusters and dominant_clusters:
            return "cross_system_no_shared_core_evidence"
        if (
            float(getattr(candidate, "coverage_score", 0.0) or 0.0) < 0.24
            and self.judge._residual(candidate) > 0.55
        ):
            return "low_core_coverage_high_residual"
        if not self.judge._candidate_discriminating_exams(candidate):
            return "no_discriminating_exam"
        return "cross_system_no_shared_core_evidence"

    def _dominant_clusters(
        self,
        candidates: Sequence[Any],
        relevance: Dict[str, Dict[str, Any]],
    ) -> set[str]:
        cluster_scores: Dict[str, float] = {}
        cluster_counts: Dict[str, int] = {}
        for candidate in candidates:
            name = self._name(candidate)
            cluster = str(relevance[name]["cluster"] or "")
            if not cluster or cluster == "unknown":
                continue
            core_count = len(relevance[name]["core"])
            direct = 1.0 if self._direct_diagnosis(candidate) else 0.0
            if not core_count and not direct:
                continue
            if direct and self._direct_low_explainability(candidate):
                continue
            score = (
                0.42 * min(core_count, 3)
                + 0.35 * float(getattr(candidate, "coverage_score", 0.0) or 0.0)
                + 0.16 * float(getattr(candidate, "source_prior", 0.0) or 0.0)
                + 0.20 * direct
            )
            if direct and not [
                item
                for item in relevance[name]["core"]
                if not str(item).startswith("diagnosis:")
            ]:
                score -= 0.24
            if self._strong_specific_candidate(candidate):
                score += 0.08
            cluster_scores[cluster] = cluster_scores.get(cluster, 0.0) + score
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        if not cluster_scores:
            return set()
        best = max(cluster_scores.values())
        threshold = max(0.42, best * 0.62)
        multi_candidate_cluster_exists = any(count >= 2 for count in cluster_counts.values())
        return {
            cluster
            for cluster, score in cluster_scores.items()
            if score >= threshold
            and (
                cluster_counts.get(cluster, 0) >= 2
                or not multi_candidate_cluster_exists
                or score >= best * 0.92
            )
        }

    def _allowed_matrix(
        self,
        retained: Sequence[Any],
        relevance: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        matrix: List[Dict[str, Any]] = []
        for left_index, left in enumerate(retained):
            for right in retained[left_index + 1 :]:
                allowed, reason = self._pair_allowed_reason(left, right, relevance)
                matrix.append(
                    {
                        "left": self._name(left),
                        "right": self._name(right),
                        "allowed": allowed,
                        "reason": reason,
                    }
                )
        return matrix

    def _pair_allowed_reason(
        self,
        left: Any,
        right: Any,
        relevance: Dict[str, Dict[str, Any]],
    ) -> tuple[bool, str]:
        if getattr(left, "hard_contradiction", False) or getattr(
            right, "hard_contradiction", False
        ):
            return False, "negative_feature"
        left_data = relevance[self._name(left)]
        right_data = relevance[self._name(right)]
        if left_data["cluster"] and left_data["cluster"] == right_data["cluster"]:
            return True, "same_clinical_cluster"
        shared_core = set(left_data["core"]) & set(right_data["core"])
        if shared_core:
            return True, "shared_core_evidence"
        if self.judge._same_family(left, right):
            return True, "same_family"
        if self.judge._causally_related(left, right):
            return True, "causal_or_graph_relation"
        if self._bridge_protected(left) or self._bridge_protected(right):
            return True, BRIDGE_REASON
        if self._protected_recall(left) or self._protected_recall(right):
            return True, "protected_recall_arbitration"
        if self.judge._same_body_system(left, right) and (
            left_data["core"] or right_data["core"]
        ):
            return True, "same_body_system_with_core_evidence"
        return False, "cross_system_no_shared_core_evidence"

    @staticmethod
    def _bridge_protected(candidate: Any) -> bool:
        return has_active_bridge_protection(candidate, CROSS_SYSTEM_SCOPE)

    def _protected_recall(self, candidate: Any) -> bool:
        return self.judge._protected_recall_candidate(candidate)

    def _can_form_differential(
        self,
        left: Any,
        right: Any,
        relevance: Dict[str, Dict[str, Any]],
    ) -> bool:
        allowed, _ = self._pair_allowed_reason(left, right, relevance)
        return allowed

    def _limit_pool(
        self,
        retained: Sequence[Any],
        relevance: Dict[str, Dict[str, Any]],
        original: Sequence[Any],
        force_names: Optional[set[str]] = None,
    ) -> List[Any]:
        force_names = set(force_names or set())
        limit = int(
            getattr(self.judge, "filtered_pool_max_size", 0)
            or max(self.judge.differential_top_k, 6)
        )
        if len(retained) <= limit:
            return list(retained)
        original_index = {id(item): index for index, item in enumerate(original)}

        def key(item: Any) -> tuple:
            data = relevance[self._name(item)]
            index = original_index.get(id(item), 999)
            return (
                1 if self._high_explanatory_candidate(item, data) else 0,
                1 if self._bridge_protected(item) else 0,
                1 if index < self.judge.differential_top_k else 0,
                1 if data["core"] else 0,
                1 if self._direct_diagnosis(item) else 0,
                self.judge._judge_score(item),
                -index,
            )

        forced = [
            item
            for item in retained
            if self._name(item) in force_names
            and not getattr(item, "hard_contradiction", False)
        ]
        forced_ids = {id(item) for item in forced}
        remaining_limit = max(0, limit - len(forced))
        selected = forced + [
            item
            for item in sorted(
                [item for item in retained if id(item) not in forced_ids],
                key=key,
                reverse=True,
            )[:remaining_limit]
        ]
        selected_ids = {id(item) for item in selected}
        return [item for item in original if id(item) in selected_ids]

    def _summary(
        self,
        original: Sequence[Any],
        retained: Sequence[Any],
        excluded: Sequence[Dict[str, Any]],
        matrix: Sequence[Dict[str, Any]],
        relevance: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        retained_names = [self._name(item) for item in retained]
        noise_rejections = sum(
            1
            for item in excluded
            if item.get("reason")
            in {
                "cross_system_no_shared_core_evidence",
                "generic_only_evidence",
                "low_core_coverage_high_residual",
                "negative_feature",
                "no_discriminating_exam",
                "pool_size_limit",
            }
        )
        cluster_rejections = sum(
            1
            for item in excluded
            if item.get("reason") == "cross_system_no_shared_core_evidence"
        )
        generic_only_candidates = sum(
            1
            for data in relevance.values()
            if data.get("generic") and not data.get("core")
        )
        core_hits = sum(
            1 for name in retained_names if relevance.get(name, {}).get("core")
        )
        return {
            "initial_pool_count": len(original),
            "filtered_pool_count": len(retained),
            "excluded_count": len(excluded),
            "pairwise_allowed_count": sum(1 for item in matrix if item.get("allowed")),
            "pairwise_blocked_count": sum(1 for item in matrix if not item.get("allowed")),
            "pairwise_noise_rejection_count": noise_rejections,
            "cluster_gate_rejection_count": cluster_rejections,
            "generic_only_candidate_count": generic_only_candidates,
            "core_evidence_coverage": (
                round(core_hits / max(1, len(retained)), 4) if retained else None
            ),
        }

    def _relevance(self, candidate: Any) -> Dict[str, Any]:
        core: List[str] = []
        generic: List[str] = []
        entry = self._entry(candidate)
        entry_findings = self._entry_findings(entry)
        for item in getattr(candidate, "matched_evidence", []) or []:
            text = str(item or "").strip()
            if not text:
                continue
            if self._is_core_evidence(text, candidate, entry_findings):
                if text not in core:
                    core.append(text)
            else:
                if text not in generic:
                    generic.append(text)
        return {
            "core": core,
            "generic": generic,
            "cluster": self._clinical_cluster(candidate, entry, core, generic),
        }

    def _is_core_evidence(
        self,
        text: str,
        candidate: Any,
        entry_findings: set[str],
    ) -> bool:
        if text.startswith(_GENERIC_EVIDENCE_PREFIXES):
            return False
        if text.startswith("diagnosis:"):
            return True
        if text in _BROAD_EVIDENCE_TOKENS:
            return False
        if text in _CONTEXTUAL_CORE_FINDINGS:
            return self._contextual_core_allowed(text, candidate)
        if text in _CORE_EVIDENCE_TOKENS or text in entry_findings:
            return True
        if text.startswith("symptom:"):
            symptom = text.split(":", 1)[1]
            return self._symptom_core_allowed(symptom, candidate)
        lower = text.lower()
        if lower in _BROAD_EVIDENCE_TOKENS:
            return False
        return any(
            marker in lower
            for marker in (
                "_positive",
                "_abnormal",
                "_high",
                "_low",
                "_defect",
                "_stenosis",
                "_imaging",
                "_cyst",
                "_mass",
                "_discharge",
            )
        )

    def _contextual_core_allowed(self, text: str, candidate: Any) -> bool:
        name = self._name(candidate)
        entry = self._entry(candidate)
        body, family = self._metadata(candidate, entry)
        cluster = _KNOWN_CLUSTER_BY_NAME.get(name, "")
        if text in {
            "near_vision_difficulty",
            "age_related_near_blur",
            "refractive_correction_improves_near_vision",
            "presbyopia_pattern",
            "night_vision_decline",
            "nyctalopia_pattern",
        }:
            return body == "ophthalmology" or cluster == "ophthalmology_visual"
        if text in {
            "umbilical_discharge",
            "umbilical_mass",
            "midline_suprapubic_pain",
            "midline_suprapubic_cyst",
            "urachal_remnant_pattern",
            "urachal_cyst_imaging",
        }:
            return (
                cluster == "urachal_midline_urinary"
                or family == "urachal_remnant"
                or body in {"urology", "genitourinary"}
            )
        if text in {
            "chronic_cough_pattern",
            "tb_exposure",
            "tuberculosis_exposure",
            "tuberculosis_pattern",
            "night_sweats",
            "hemoptysis",
        }:
            return body == "respiratory" or cluster == "pulmonary_infection_mass"
        if text in {
            "crusted_exudative_skin_ulcer",
            "regional_lymphadenopathy",
            "rural_child_contact",
        }:
            return (
                cluster == "dermatology_eruptive_systemic"
                or body.startswith("dermatology")
                or body == "hematology"
                or family in {"treponemal_skin_bone_infection", "vesicular_viral_exanthem"}
            )
        if text == "bradycardia":
            return body == "cardiovascular" and family == "cardiovascular_conduction"
        if text in {
            "anca_positive",
            "mpo_anca_positive",
            "p_anca_positive",
            "microscopic_hematuria",
            "proteinuria",
            "pulmonary_hemorrhage",
        }:
            return (
                "\u8840\u7ba1\u708e" in name
                or "\u80ba\u80be" in name
                or body in {"immune", "rheumatology", "nephrology", "immune_renal_pulmonary"}
                or "vasculitis" in family.lower()
                or "glomerulonephritis" in family.lower()
            )
        if text in {
            "low_magnesium",
            "low_urine_magnesium",
            "magnesium_depletion",
            "magnesium_load_retention_high",
        }:
            return (
                "\u4f4e\u9541" in name
                or "magnesium" in family.lower()
                or "electrolyte" in family.lower()
                or body in {"metabolic", "endocrine_metabolic"}
            )
        return True

    def _symptom_core_allowed(self, symptom: str, candidate: Any) -> bool:
        if not any(keyword in symptom for keyword in _CORE_SYMPTOM_KEYWORDS):
            return False
        name = self._name(candidate)
        entry = self._entry(candidate)
        body, family = self._metadata(candidate, entry)
        cluster = _KNOWN_CLUSTER_BY_NAME.get(name, "")
        if any(keyword in symptom for keyword in ("脐部", "脐周", "脐下")):
            return cluster == "urachal_midline_urinary" or family == "urachal_remnant"
        if any(keyword in symptom for keyword in ("外生殖器", "尿道下裂", "隐睾")):
            return cluster == "congenital_genitourinary_dsd" or family == "sex_development_disorder"
        if any(keyword in symptom for keyword in ("看近", "阅读困难", "视物模糊")):
            return body == "ophthalmology"
        if any(keyword in symptom for keyword in ("血痰", "咯血", "盗汗")):
            return body == "respiratory" or cluster == "pulmonary_infection_mass"
        if any(keyword in symptom for keyword in ("结痂", "渗出", "黄水", "腹股沟", "水疱", "疱疹", "关节痛", "骨痛")):
            return (
                cluster == "dermatology_eruptive_systemic"
                or body.startswith("dermatology")
                or body == "hematology"
            )
        return True

    def _entry_findings(self, entry: Dict[str, Any]) -> set[str]:
        findings: set[str] = set()
        for spec in entry.get("supporting_evidence", []) or []:
            if isinstance(spec, dict):
                finding = str(spec.get("finding") or "").strip()
                if finding:
                    findings.add(finding)
                for term in spec.get("terms", []) or []:
                    text = str(term or "").strip()
                    if text:
                        findings.add(text)
        for group in entry.get("required_groups", []) or []:
            for spec in group or []:
                if isinstance(spec, dict):
                    finding = str(spec.get("finding") or "").strip()
                    if finding:
                        findings.add(finding)
        return findings

    def _clinical_cluster(
        self,
        candidate: Any,
        entry: Dict[str, Any],
        core: Sequence[str],
        generic: Sequence[str],
    ) -> str:
        name = self._name(candidate)
        if name in _KNOWN_CLUSTER_BY_NAME:
            return _KNOWN_CLUSTER_BY_NAME[name]
        if "\u4f4e\u9541" in name:
            return "metabolic_electrolyte"
        if "\u5375\u5de2\u8fc7\u5ea6\u523a\u6fc0" in name:
            return "gynecology_ovarian_hyperstimulation"
        if "\u8840\u7ba1\u708e" in name or "\u80ba\u80be" in name:
            return "pulmonary_renal_vasculitis"
        body, family = self._metadata(candidate, entry)
        if family in {
            "treponemal_skin_bone_infection",
            "vesicular_viral_exanthem",
            "dermatitis",
            "anogenital_hpv_infection",
        }:
            return "dermatology_eruptive_systemic"
        if family in {
            "pulmonary_tuberculosis",
            "opportunistic_fungal_pneumonia",
            "pneumonia",
            "lung_malignancy",
        } or body == "respiratory":
            return "pulmonary_infection_mass"
        if body == "ophthalmology":
            return "ophthalmology_visual"
        if family == "sex_development_disorder" or body in {"endocrine_genetic", "genetic"}:
            return "congenital_genitourinary_dsd"
        if family == "urachal_remnant" or set(core) & {
            "umbilical_discharge",
            "umbilical_mass",
            "midline_suprapubic_cyst",
            "urachal_cyst_imaging",
        }:
            return "urachal_midline_urinary"
        if body and family:
            return f"{body}:{family}"
        return body or family or "unknown"

    def _metadata(self, candidate: Any, entry: Dict[str, Any]) -> tuple[str, str]:
        name = self._name(candidate)
        body = str(entry.get("body_system") or "")
        family = str(entry.get("disease_family") or entry.get("family") or "")
        fallback = _FALLBACK_DISEASE_METADATA.get(name)
        if fallback:
            body = body or fallback[0]
            family = family or fallback[1]
        return body, family

    def _entry(self, candidate: Any) -> Dict[str, Any]:
        if self.judge.knowledge and candidate:
            return self.judge.knowledge.get(self._name(candidate)) or {}
        return {}

    def _direct_diagnosis(self, candidate: Any) -> bool:
        name = self._name(candidate)
        return f"diagnosis:{name}" in set(getattr(candidate, "matched_evidence", []) or [])

    def _direct_low_explainability(self, candidate: Any) -> bool:
        return bool(
            candidate
            and float(getattr(candidate, "coverage_score", 0.0) or 0.0) < 0.22
            and self.judge._residual(candidate) > 0.65
        )

    def _strong_specific_candidate(self, candidate: Any) -> bool:
        return bool(
            candidate
            and (
                self.judge._priority(candidate)
                or self.judge._gap_authorizable(candidate)
            )
        )

    def _high_explanatory_candidate(self, candidate: Any, data: Dict[str, Any]) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        if not data.get("core"):
            return False
        coverage = float(getattr(candidate, "coverage_score", 0.0) or 0.0)
        core_coverage = float(
            getattr(candidate, "core_explanatory_coverage", 0.0) or 0.0
        )
        residual = self.judge._residual(candidate)
        score = float(getattr(candidate, "score", 0.0) or 0.0)
        return bool(
            coverage >= 0.70
            and core_coverage >= 0.65
            and residual <= 0.25
            and (
                score >= 0.65
                or getattr(candidate, "required_met", False)
                or self.judge._priority(candidate)
                or self.judge._systemic_primary(candidate)
            )
        )

    @staticmethod
    def _name(candidate: Any) -> str:
        return str(getattr(candidate, "diagnosis", "") or "")


class DiagnosisJudge:
    """Deterministic judge for an already-generated diagnosis candidate table."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, knowledge: Any = None):
        diagnosis_section = (config or {}).get("diagnosis") or {}
        section = diagnosis_section.get("judge") or {}
        self.knowledge = knowledge
        self.exam_resolver = ExamResolver(knowledge) if knowledge is not None else None
        self.clinical_comparator = ClinicalReasoningComparator()
        self.top_k = int(section.get("top_k", 20) or 20)
        self.differential_top_k = int(
            section.get(
                "judge_top_k",
                diagnosis_section.get("judge_top_k", 5),
            )
            or 5
        )
        self.max_reviews = int(section.get("max_reviews", 20) or 20)
        self.pairwise_close_margin = float(
            section.get(
                "pairwise_close_margin",
                diagnosis_section.get("pairwise_close_margin", 0.18),
            )
            or 0.18
        )
        self.discriminating_exam_max_items = int(
            section.get(
                "discriminating_exam_max_items",
                diagnosis_section.get("discriminating_exam_max_items", 6),
            )
            or 6
        )
        self.filtered_pool_max_size = int(
            section.get(
                "filtered_pool_max_size",
                diagnosis_section.get("filtered_pool_max_size", 8),
            )
            or 8
        )
        self.differential_score_margin = float(
            section.get(
                "differential_score_margin",
                diagnosis_section.get("differential_score_margin", 0.12),
            )
            or 0.12
        )
        self.gap_authorization_min_score = float(
            section.get("gap_authorization_min_score", 0.42) or 0.42
        )
        self.gap_authorization_min_coverage = float(
            section.get("gap_authorization_min_coverage", 0.18) or 0.18
        )
        self.gap_authorization_max_residual = float(
            section.get("gap_authorization_max_residual", 0.88) or 0.88
        )
        self.priority_gap_bonus = float(section.get("priority_gap_bonus", 0.14) or 0.14)
        self.required_met_bonus = float(section.get("required_met_bonus", 0.005) or 0.005)
        self.coverage_bonus = float(section.get("coverage_bonus", 0.24) or 0.24)
        self.residual_penalty = float(section.get("residual_penalty", 0.18) or 0.18)
        self.core_coverage_bonus = float(
            section.get("core_coverage_bonus", 0.30) or 0.30
        )
        self.residual_core_penalty = float(
            section.get("residual_core_penalty", 0.09) or 0.09
        )
        self.explanatory_preference_margin = float(
            section.get("explanatory_preference_margin", 0.10) or 0.10
        )
        self.gap_authorization_min_explanatory_score = float(
            section.get("gap_authorization_min_explanatory_score", 0.46) or 0.46
        )
        self.gap_authorization_min_core_coverage = float(
            section.get("gap_authorization_min_core_coverage", 0.40) or 0.40
        )
        self.gap_authorization_max_core_residual = int(
            section.get("gap_authorization_max_core_residual", 1) or 1
        )
        self.parent_fallback_margin = float(
            section.get("parent_fallback_margin", 0.10) or 0.10
        )
        self.secondary_min_score = float(section.get("secondary_min_score", 0.45) or 0.45)
        self.gap_target_limit = int(section.get("evidence_gap_target_limit", 2) or 2)
        self.pool_filter = DifferentialPoolFilter(self)

    def judge(
        self,
        candidates: Sequence[Any],
        preselected: Optional[Sequence[Any]] = None,
        max_final_diagnoses: int = 3,
    ) -> JudgeDecision:
        ranked_all = [item for item in candidates or [] if self._has_signal(item)]
        ranked_all = sorted(ranked_all, key=self._sort_key, reverse=True)
        ranked = [
            item
            for item in ranked_all
            if not getattr(item, "hard_contradiction", False)
            and self._eligibility_status(item) != EXCLUDED
        ]
        retriever_top1 = self._name(candidates[0]) if candidates else ""

        decision = JudgeDecision(retriever_top1=retriever_top1)
        self._apply_eligibility_audit(decision, ranked_all)
        if not ranked:
            decision.reasoning = "Judge found no supported candidate."
            return decision

        evidence_conflicts = self._collect_evidence_conflicts(ranked)
        conflict_affected_diagnoses = self._conflict_affected_diagnoses(
            evidence_conflicts
        )
        force_names = self._forced_pool_names_for_conflicts(
            ranked,
            conflict_affected_diagnoses,
        )
        self._annotate_deferred_gap_priorities(ranked)
        pattern_force_names = self._forced_pool_names_for_pattern_deferred(ranked)
        claim_force_names = self._forced_pool_names_for_claims(ranked)
        gap_force_names = self._forced_pool_names_for_deferred_gap_override(ranked)
        bridge_force_names = self._forced_pool_names_for_bridge_protection(ranked)
        protected_recall_force_names = self._forced_pool_names_for_protected_recall(
            ranked
        )
        force_names = list(
            dict.fromkeys(
                list(force_names)
                + pattern_force_names
                + claim_force_names
                + gap_force_names
                + bridge_force_names
                + protected_recall_force_names
            )
        )
        candidate_pool_for_workup = [
            item
            for item in ranked
            if self._eligibility_status(item) in {PRIMARY_ELIGIBLE, DEFERRED}
        ] or ranked
        raw_differential_pool = self._differential_pool(candidate_pool_for_workup)
        raw_differential_pool = self._extend_forced_pool(
            raw_differential_pool,
            candidate_pool_for_workup,
            force_names,
        )
        raw_pool_source = self._differential_pool_source(raw_differential_pool)
        for name in pattern_force_names:
            raw_pool_source[name] = "pattern_deferred_workup"
        for name in claim_force_names:
            raw_pool_source[name] = "critical_evidence_claim_followup"
        for name in gap_force_names:
            raw_pool_source[name] = "deferred_gap_priority_override"
        for name in bridge_force_names:
            raw_pool_source[name] = "bridge_protection"
        for name in protected_recall_force_names:
            raw_pool_source[name] = "protected_recall_arbitration"
        for name in force_names:
            raw_pool_source.setdefault(
                name,
                "forced_conflict_or_top_candidate",
            )
        pool_filter = self.pool_filter.filter(
            raw_differential_pool,
            raw_pool_source,
            force_names=force_names,
        )
        differential_pool = pool_filter.candidates or raw_differential_pool
        allowed_pairs = self.pool_filter.allowed_pair_names(pool_filter)
        pairwise = self._pairwise_comparisons(differential_pool, allowed_pairs)
        required_gap_by_candidate = self._required_gap_by_candidate(differential_pool)
        discriminating_findings = self._discriminating_findings(
            differential_pool, required_gap_by_candidate
        )
        base_discriminating_exam_tasks = self._discriminating_exam_tasks(
            differential_pool, pairwise, discriminating_findings
        )
        pattern_exam_tasks = self._pattern_anchor_workup_exam_tasks(differential_pool)
        claim_exam_tasks = self._claim_followup_exam_tasks(differential_pool)
        conflict_exam_tasks = self._conflict_adjudication_exam_tasks(
            differential_pool
        )
        deferred_gap_exam_tasks = self._deferred_gap_closure_exam_tasks(
            differential_pool
        )
        discriminating_exam_tasks = self._merge_discriminating_exam_tasks(
            list(conflict_exam_tasks)
            + list(deferred_gap_exam_tasks)
            + list(pattern_exam_tasks)
            + list(claim_exam_tasks),
            base_discriminating_exam_tasks,
        )
        discriminating_exams = [
            str(task.get("exam") or "").strip()
            for task in discriminating_exam_tasks
            if str(task.get("exam") or "").strip()
        ]
        primary_candidates = [
            item
            for item in differential_pool
            if self._eligibility_status(item) == PRIMARY_ELIGIBLE
        ] or [
            item
            for item in ranked[: self.top_k]
            if self._eligibility_status(item) == PRIMARY_ELIGIBLE
        ]
        primary = self._choose_primary(primary_candidates)
        arbitration = self._primary_arbitration(
            primary,
            differential_pool,
            pairwise,
            full_pool=ranked,
        )
        if arbitration.get("selected_candidate") is not None:
            primary = arbitration["selected_candidate"]
        pairwise_gap_tasks = self._pairwise_gap_exam_tasks(
            arbitration.get("pairwise_discriminating_gaps") or [],
            differential_pool,
        )
        if pairwise_gap_tasks:
            discriminating_exam_tasks = self._merge_discriminating_exam_tasks(
                pairwise_gap_tasks,
                discriminating_exam_tasks,
            )
            discriminating_exams = [
                str(task.get("exam") or "").strip()
                for task in discriminating_exam_tasks
                if str(task.get("exam") or "").strip()
            ]
        gap_state_by_candidate = self._gap_state_by_candidate(ranked_all)
        gap_state_distribution = self._gap_state_distribution(gap_state_by_candidate)
        if primary is None:
            provisional = self._best_deferred_candidate(differential_pool or ranked)
            evidence_gap_targets = (
                self._deferred_evidence_gap_targets(provisional, differential_pool)
                if provisional
                else []
            )
            blocked = self._blocked_records(ranked_all, [])
            reviews = self._reviews(
                ranked_all,
                provisional,
                [],
                evidence_gap_targets,
                blocked,
            )
            if provisional:
                decision.judge_primary = provisional.diagnosis
                decision.primary = provisional.diagnosis
                decision.provisional_primary = provisional.diagnosis
                decision.pre_discrimination_primary = provisional.diagnosis
                decision.fallback_primary = provisional.diagnosis
                decision.explanatory_coverage = self._coverage(provisional)
                decision.core_explanatory_coverage = self._core_coverage(provisional)
                decision.residual_evidence_score = self._residual(provisional)
                decision.residual_core_evidence_count = self._residual_core_count(provisional)
            decision.primary_status = "deferred"
            decision.needs_discriminating_exams = bool(evidence_gap_targets)
            decision.defer_reason = "no PrimaryEligible candidate; deferred candidates need required anchor evidence"
            decision.discrimination_attempted = bool(evidence_gap_targets)
            decision.discrimination_resolved = False
            decision.evidence_gap_targets = evidence_gap_targets
            decision.final_diagnoses = []
            decision.required_gap_authorized_diagnoses = []
            decision.blocked_diagnoses = blocked
            decision.reviews = reviews
            decision.required_gap_state_by_candidate = gap_state_by_candidate
            decision.gap_state_distribution = gap_state_distribution
            decision.differential_candidates = [item.diagnosis for item in differential_pool]
            decision.pairwise_comparisons = pairwise
            decision.excluded_from_pairwise = list(pool_filter.excluded)
            decision.pool_filter_reasons = dict(pool_filter.pool_filter_reasons)
            decision.discriminating_findings = discriminating_findings
            decision.discriminating_exams = discriminating_exams
            decision.discriminating_exam_tasks = discriminating_exam_tasks
            self._apply_deferred_gap_decision_audit(
                decision,
                differential_pool,
                discriminating_exam_tasks,
            )
            self._apply_bridge_decision_audit(
                decision,
                ranked_all,
                differential_pool,
                pairwise,
                discriminating_exam_tasks,
            )
            decision.required_gap_by_candidate = required_gap_by_candidate
            decision.differential_pool_source = dict(pool_filter.pool_source)
            decision.evidence_conflicts = evidence_conflicts
            decision.conflict_affected_diagnoses = conflict_affected_diagnoses
            self._apply_entity_audit(decision, ranked_all)
            decision.reasoning = self._reasoning(decision)
            return decision

        defer_reason = str(arbitration.get("defer_reason") or "") or self._defer_primary_lock_reason(
            primary,
            differential_pool,
            pairwise,
            discriminating_exams,
        )
        needs_discriminating = bool(defer_reason)
        primary_status = "deferred" if needs_discriminating else "locked"

        eligible_ranked = [
            item for item in ranked if self._eligibility_status(item) == PRIMARY_ELIGIBLE
        ]
        secondary = (
            self._select_secondary(primary, eligible_ranked, max_final_diagnoses)
            if self._eligibility_status(primary) == PRIMARY_ELIGIBLE and not needs_discriminating
            else []
        )
        final = (
            [primary.diagnosis] + [item.diagnosis for item in secondary]
            if self._eligibility_status(primary) == PRIMARY_ELIGIBLE and not needs_discriminating
            else []
        )

        if needs_discriminating:
            evidence_gap_targets = self._deferred_evidence_gap_targets(
                primary,
                differential_pool,
            )
        else:
            evidence_gap_targets = self._evidence_gap_targets(primary, ranked, final)
        blocked = self._blocked_records(ranked_all, final)
        reviews = self._reviews(ranked_all, primary, secondary, evidence_gap_targets, blocked)

        decision.judge_primary = primary.diagnosis
        decision.primary = primary.diagnosis
        decision.primary_status = primary_status
        decision.needs_discriminating_exams = needs_discriminating
        decision.provisional_primary = primary.diagnosis if needs_discriminating else ""
        decision.locked_primary = primary.diagnosis if not needs_discriminating else ""
        decision.defer_reason = defer_reason
        decision.pre_discrimination_primary = primary.diagnosis
        decision.fallback_primary = primary.diagnosis
        decision.discrimination_attempted = needs_discriminating
        decision.discrimination_resolved = not needs_discriminating
        decision.secondary = [item.diagnosis for item in secondary]
        decision.differential = [item["diagnosis"] for item in blocked[: self.max_reviews]]
        decision.evidence_gap_targets = evidence_gap_targets
        decision.final_diagnoses = final[:max(1, int(max_final_diagnoses or 1))]
        decision.required_gap_authorized_diagnoses = []
        decision.blocked_diagnoses = blocked
        decision.reviews = reviews
        decision.high_value_gap_candidates = [
            item.diagnosis
            for item in differential_pool
            if item.diagnosis != primary.diagnosis
            and (
                self._high_value_unresolved_contender(primary, item)
                or self._pattern_deferred_workup_candidate(item)
                or self._exam_priority_override_candidate(item)
            )
        ][: self.gap_target_limit]
        decision.explanatory_coverage = self._coverage(primary)
        decision.core_explanatory_coverage = self._core_coverage(primary)
        decision.residual_evidence_score = self._residual(primary)
        decision.residual_core_evidence_count = self._residual_core_count(primary)
        decision.required_gap_state_by_candidate = gap_state_by_candidate
        decision.gap_state_distribution = gap_state_distribution
        previous_primary = str((preselected or [""])[0] if preselected else "")
        if previous_primary and previous_primary != primary.diagnosis:
            decision.primary_unlock_reason = self._primary_unlock_reason(
                previous_primary,
                primary,
                ranked,
            )
        model_primary = self._model_score_primary(ranked)
        decision.explanation_score_changed_ranking = bool(
            model_primary and model_primary != primary.diagnosis
        )
        decision.differential_candidates = [item.diagnosis for item in differential_pool]
        decision.pairwise_comparisons = pairwise
        decision.excluded_from_pairwise = list(pool_filter.excluded)
        decision.pool_filter_reasons = dict(pool_filter.pool_filter_reasons)
        decision.cluster_assignments = dict(pool_filter.cluster_assignments)
        decision.pairwise_allowed_matrix = list(pool_filter.pairwise_allowed_matrix)
        decision.core_evidence_by_candidate = dict(pool_filter.core_evidence_by_candidate)
        decision.generic_evidence_by_candidate = dict(
            pool_filter.generic_evidence_by_candidate
        )
        decision.pool_filter_summary = dict(pool_filter.summary)
        decision.discriminating_findings = discriminating_findings
        decision.discriminating_exams = discriminating_exams
        decision.discriminating_exam_tasks = discriminating_exam_tasks
        self._apply_primary_arbitration_audit(decision, arbitration)
        self._apply_deferred_gap_decision_audit(
            decision,
            differential_pool,
            discriminating_exam_tasks,
        )
        self._apply_bridge_decision_audit(
            decision,
            ranked_all,
            differential_pool,
            pairwise,
            discriminating_exam_tasks,
        )
        decision.required_gap_by_candidate = required_gap_by_candidate
        decision.differential_pool_source = dict(pool_filter.pool_source)
        decision.evidence_conflicts = evidence_conflicts
        decision.conflict_affected_diagnoses = conflict_affected_diagnoses
        decision.dynamic_rerank_trace = [
            {
                "stage": "initial_judge",
                "primary": primary.diagnosis,
                "primary_status": primary_status,
                "needs_discriminating_exams": needs_discriminating,
                "defer_reason": defer_reason,
                "evidence_conflicts": list(evidence_conflicts),
                "conflict_affected_diagnoses": list(conflict_affected_diagnoses),
                "required_gap_state": gap_state_by_candidate.get(primary.diagnosis, ""),
                "gap_state_distribution": dict(gap_state_distribution),
                "primary_unlock_reason": decision.primary_unlock_reason,
                "explanation_score_changed_ranking": decision.explanation_score_changed_ranking,
                "pool_filter_summary": dict(pool_filter.summary),
                "ranked": [
                    {
                        "diagnosis": item.diagnosis,
                        "entity_id": str(getattr(item, "entity_id", "") or ""),
                        "primary_eligibility_score": round(
                            self._primary_eligibility_score(item),
                            4,
                        ),
                        "judge_score": round(self._judge_score(item), 4),
                        "required_gap_state": gap_state_by_candidate.get(
                            item.diagnosis,
                            self._required_gap_state(item),
                        ),
                        "required_met": bool(getattr(item, "required_met", False)),
                        "explanatory_coverage": round(self._coverage(item), 4),
                        "core_explanatory_coverage": round(
                            self._core_coverage(item),
                            4,
                        ),
                        "residual_evidence_score": round(self._residual(item), 4),
                        "residual_core_evidence_count": self._residual_core_count(item),
                        "eligibility_status": self._eligibility_status(item),
                        "eligibility_reason": str(
                            getattr(item, "eligibility_reason", "") or ""
                        ),
                        "missing_required_anchors": list(
                            getattr(item, "missing_required_anchors", []) or []
                        )[:6],
                        "evidence_pattern_matches": list(
                            getattr(item, "evidence_pattern_matches", []) or []
                        )[:4],
                    }
                    for item in differential_pool
                ],
            }
        ]
        decision.decision_override = bool(
            decision.retriever_top1
            and decision.judge_primary
            and decision.retriever_top1 != decision.judge_primary
        )
        decision.reasoning = self._reasoning(decision)
        decision.evidence_conflicts = evidence_conflicts
        decision.conflict_affected_diagnoses = conflict_affected_diagnoses
        self._apply_entity_audit(decision, ranked_all)
        return decision

    def _differential_pool(self, ranked: Sequence[Any]) -> List[Any]:
        pool: List[Any] = []
        top_score = self._judge_score(ranked[0]) if ranked else 0.0
        for index, item in enumerate(ranked[: self.top_k]):
            if item and not getattr(item, "hard_contradiction", False):
                if (
                    index < max(2, self.differential_top_k)
                    or self._judge_score(item) >= top_score - self.differential_score_margin
                    or self._tail_differential_candidate(item)
                ):
                    pool.append(item)
        limit = max(self.differential_top_k, min(self.top_k, 12))
        for item in ranked[self.differential_top_k : self.top_k]:
            if len(pool) >= limit:
                break
            if not self._tail_differential_candidate(item):
                continue
            if item not in pool:
                pool.append(item)
        result: List[Any] = []
        seen_ids: set[int] = set()
        for item in pool:
            marker = id(item)
            if marker in seen_ids:
                continue
            seen_ids.add(marker)
            result.append(item)
        return result

    def _tail_differential_candidate(self, candidate: Any) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        if not self._has_signal(candidate):
            return False
        if not (
            self._priority(candidate)
            or self._gap_authorizable(candidate)
        ):
            return False
        return bool(
            getattr(candidate, "required_gaps", None)
            or self._objective_signal(candidate)
            or float(getattr(candidate, "coverage_score", 0.0) or 0.0) >= 0.18
            or float(getattr(candidate, "source_prior", 0.0) or 0.0) >= 0.45
        )

    def _differential_pool_source(self, pool: Sequence[Any]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for index, item in enumerate(pool):
            if not item:
                continue
            result[item.diagnosis] = (
                "top_k" if index < self.differential_top_k else "top20_priority_tail"
            )
        return result

    @staticmethod
    def _collect_evidence_conflicts(ranked: Sequence[Any]) -> List[Dict[str, Any]]:
        conflicts: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for candidate in ranked or []:
            for conflict in getattr(candidate, "evidence_conflicts", []) or []:
                if not isinstance(conflict, dict):
                    continue
                diagnosis = str(
                    conflict.get("affected_diagnosis")
                    or getattr(candidate, "diagnosis", "")
                    or ""
                )
                finding = str(conflict.get("finding") or "")
                key = (diagnosis, finding)
                if key in seen:
                    continue
                seen.add(key)
                conflicts.append(dict(conflict))
        return conflicts

    @staticmethod
    def _conflict_affected_diagnoses(
        evidence_conflicts: Sequence[Dict[str, Any]],
    ) -> List[str]:
        return list(
            dict.fromkeys(
                str(item.get("affected_diagnosis") or "").strip()
                for item in evidence_conflicts or []
                if str(item.get("affected_diagnosis") or "").strip()
            )
        )

    def _forced_pool_names_for_conflicts(
        self,
        ranked: Sequence[Any],
        conflict_affected_diagnoses: Sequence[str],
    ) -> List[str]:
        names: List[str] = []

        def add(name: str) -> None:
            text = str(name or "").strip()
            if text and text not in names:
                names.append(text)

        if ranked:
            add(str(getattr(ranked[0], "diagnosis", "") or ""))
        affected = set(conflict_affected_diagnoses or [])
        if not affected:
            return []
        for item in ranked:
            name = str(getattr(item, "diagnosis", "") or "")
            if name in affected:
                add(name)

        conflict_candidates = [
            item
            for item in ranked
            if str(getattr(item, "diagnosis", "") or "") in affected
        ]
        if not conflict_candidates:
            return names
        conflict_score = max(self._judge_score(item) for item in conflict_candidates)
        for index, item in enumerate(ranked[: self.top_k]):
            if not item or getattr(item, "hard_contradiction", False):
                continue
            name = str(getattr(item, "diagnosis", "") or "")
            if name in names:
                continue
            high_value = (
                index < max(3, self.differential_top_k)
                or self._priority(item)
                or self._systemic_primary(item)
                or self._core_or_diagnostic_signal(item)
                or float(getattr(item, "specificity", 0.0) or 0.0) >= 0.85
            )
            if index < max(3, self.differential_top_k) and high_value:
                add(name)
                continue
            if high_value and self._judge_score(item) >= conflict_score - 0.26:
                add(name)
        return names[:8]

    def _forced_pool_names_for_pattern_deferred(
        self,
        ranked: Sequence[Any],
    ) -> List[str]:
        candidates = [
            item
            for item in ranked or []
            if self._pattern_deferred_workup_candidate(item)
        ]
        names: List[str] = []
        limit = max(
            4,
            int(getattr(self, "filtered_pool_max_size", 0) or 0),
            int(getattr(self, "differential_top_k", 0) or 0),
        )
        for item in sorted(
            candidates,
            key=self._pattern_deferred_workup_sort_key,
            reverse=True,
        ):
            name = str(getattr(item, "diagnosis", "") or "")
            if name and name not in names:
                names.append(name)
            if len(names) >= limit:
                break
        return names

    def _forced_pool_names_for_claims(
        self,
        ranked: Sequence[Any],
    ) -> List[str]:
        names: List[str] = []
        limit = max(
            4,
            int(getattr(self, "filtered_pool_max_size", 0) or 0),
            int(getattr(self, "differential_top_k", 0) or 0),
        )
        candidates = [
            item for item in ranked or []
            if self._critical_claim_followup_candidate(item)
        ]
        for item in sorted(
            candidates,
            key=self._claim_followup_sort_key,
            reverse=True,
        ):
            name = str(getattr(item, "diagnosis", "") or "")
            if name and name not in names:
                names.append(name)
            if len(names) >= limit:
                break
        return names

    def _critical_claim_followup_candidate(self, candidate: Any) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        claims = [
            item for item in getattr(candidate, "unresolved_critical_evidence_claims", []) or []
            if isinstance(item, dict)
        ]
        if not claims:
            return False
        if getattr(candidate, "claim_followup_exams", None):
            return True
        return any(str(item.get("recommended_exam") or "").strip() for item in claims)

    def _claim_followup_sort_key(self, candidate: Any) -> tuple:
        claims = [
            item for item in getattr(candidate, "unresolved_critical_evidence_claims", []) or []
            if isinstance(item, dict)
        ]
        confidence = 0.0
        for claim in claims:
            try:
                confidence = max(confidence, float(claim.get("confidence", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        return (
            len(claims),
            confidence,
            1 if self._priority(candidate) else 0,
            1 if self._systemic_primary(candidate) else 0,
            self._judge_score(candidate),
        )

    def _pattern_deferred_workup_candidate(self, candidate: Any) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        if self._eligibility_status(candidate) != DEFERRED:
            return False
        pattern_matches = [
            item
            for item in getattr(candidate, "evidence_pattern_matches", []) or []
            if isinstance(item, dict)
        ]
        if not self._pattern_deferred_workup_progress(pattern_matches):
            return False
        if not self._candidate_discriminating_exams(candidate):
            return False
        if not (
            getattr(candidate, "required_gaps", None)
            or pattern_matches
        ):
            return False
        if self._priority(candidate) or self._systemic_primary(candidate):
            return True
        if self._core_or_diagnostic_signal(candidate):
            return True
        if float(getattr(candidate, "evidence_specificity_score", 0.0) or 0.0) >= 0.35:
            return True
        return bool(getattr(candidate, "matched_evidence", None))

    @staticmethod
    def _pattern_deferred_workup_progress(
        pattern_matches: Sequence[Dict[str, Any]],
    ) -> bool:
        for item in pattern_matches or []:
            if not isinstance(item, dict):
                continue
            effect_status = str((item.get("effect") or {}).get("eligibility") or "")
            matched_required = list(item.get("matched_required_groups") or [])
            missing_required = list(item.get("missing_required_groups") or [])
            if effect_status == DEFERRED and (
                bool(item.get("matched")) or bool(matched_required)
            ):
                return True
            if (
                effect_status == PRIMARY_ELIGIBLE
                and not bool(item.get("matched"))
                and matched_required
                and missing_required
            ):
                return True
        return False

    def _pattern_deferred_workup_sort_key(self, candidate: Any) -> tuple:
        best_progress = 0.0
        has_deferred_match = 0
        for item in getattr(candidate, "evidence_pattern_matches", []) or []:
            if not isinstance(item, dict):
                continue
            matched_count = len(list(item.get("matched_required_groups") or []))
            missing_count = len(list(item.get("missing_required_groups") or []))
            total = max(1, matched_count + missing_count)
            best_progress = max(best_progress, matched_count / total)
            if (
                str((item.get("effect") or {}).get("eligibility") or "") == DEFERRED
                and bool(item.get("matched"))
            ):
                has_deferred_match = 1
        return (
            has_deferred_match,
            best_progress,
            1 if self._core_or_diagnostic_signal(candidate) else 0,
            float(getattr(candidate, "evidence_specificity_score", 0.0) or 0.0),
            self._judge_score(candidate),
        )

    @staticmethod
    def _extend_forced_pool(
        pool: Sequence[Any],
        ranked: Sequence[Any],
        force_names: Sequence[str],
    ) -> List[Any]:
        result: List[Any] = []
        seen_names: set[str] = set()
        for item in pool or []:
            name = str(getattr(item, "diagnosis", "") or "")
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            result.append(item)
        for item in ranked or []:
            name = str(getattr(item, "diagnosis", "") or "")
            if name in force_names and name not in seen_names:
                seen_names.add(name)
                result.append(item)
        return result

    def _pairwise_comparisons(
        self,
        pool: Sequence[Any],
        allowed_pairs: Optional[set[tuple[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        comparisons: List[Dict[str, Any]] = []
        for left_index, left in enumerate(pool):
            for right in pool[left_index + 1 :]:
                pair_key = tuple(sorted((left.diagnosis, right.diagnosis)))
                if allowed_pairs is not None and pair_key not in allowed_pairs:
                    continue
                preferred = self._pairwise_preferred(left, right)
                alternate = right if preferred is left else left
                left_gaps = list(getattr(left, "required_gaps", []) or [])
                right_gaps = list(getattr(right, "required_gaps", []) or [])
                shared_evidence = sorted(
                    set(getattr(left, "matched_evidence", []) or [])
                    & set(getattr(right, "matched_evidence", []) or [])
                )[:5]
                comparisons.append(
                    {
                        "left": left.diagnosis,
                        "right": right.diagnosis,
                        "preferred": preferred.diagnosis,
                        "reason": self._pairwise_reason(preferred, alternate),
                        "score_delta": round(
                            self._judge_score(preferred) - self._judge_score(alternate),
                            4,
                        ),
                        "close_call": abs(
                            self._judge_score(left) - self._judge_score(right)
                        )
                        <= self.pairwise_close_margin,
                        "left_required_gaps": left_gaps[:4],
                        "right_required_gaps": right_gaps[:4],
                        "shared_matched_evidence": shared_evidence,
                        "discriminating_findings": self._pairwise_discriminating_findings(
                            left, right
                        ),
                        "discriminating_exams": self._pairwise_discriminating_exams(
                            left, right
                        ),
                    }
                )
        return comparisons

    def _pairwise_preferred(self, left: Any, right: Any) -> Any:
        left_key = self._pairwise_key(left)
        right_key = self._pairwise_key(right)
        return left if left_key >= right_key else right

    def _pairwise_reason(self, preferred: Any, alternate: Any) -> str:
        if self._core_coverage(preferred) > self._core_coverage(alternate) + 0.10:
            return "preferred because it explains more core evidence"
        if self._residual_core_count(preferred) < self._residual_core_count(alternate):
            return "preferred because unexplained core evidence is lower"
        if self._coverage(preferred) > self._coverage(alternate) + 0.08:
            return "preferred because it explains more evidence"
        if self._residual(preferred) + 0.08 < self._residual(alternate):
            return "preferred because residual evidence is lower"
        if self._gap_authorizable(preferred) and not self._gap_authorizable(alternate):
            return "preferred as high-priority candidate with actionable evidence gap"
        if self._priority(preferred) and not self._priority(alternate):
            return "preferred because etiology/structural/specific diagnosis has priority"
        if self._trusted(preferred) and not self._trusted(alternate):
            return "preferred because required evidence is met"
        return "preferred by evidence-weighted judge score"

    def _pairwise_key(self, candidate: Any) -> tuple:
        return (
            self._core_coverage(candidate),
            -self._residual_core_count(candidate),
            self._coverage(candidate),
            1.0 - self._residual(candidate),
            self._judge_score(candidate),
            1 if self._priority(candidate) else 0,
            1 if self._trusted(candidate) else 0,
            float(getattr(candidate, "score", 0.0) or 0.0),
        )

    def _pairwise_discriminating_findings(self, left: Any, right: Any) -> List[str]:
        left_gaps = set(getattr(left, "required_gaps", []) or [])
        right_gaps = set(getattr(right, "required_gaps", []) or [])
        findings = list(dict.fromkeys(list(left_gaps - right_gaps) + list(right_gaps - left_gaps)))
        if not findings:
            findings = list(
                dict.fromkeys(
                    list(getattr(left, "residual_evidence", []) or [])
                    + list(getattr(right, "residual_evidence", []) or [])
                )
            )
        return findings[:6]

    def _pairwise_discriminating_exams(self, left: Any, right: Any) -> List[str]:
        return self._candidate_exam_union([left, right])[: self.discriminating_exam_max_items]

    def _candidate_key(self, candidate: Any) -> str:
        entity_id = str(getattr(candidate, "entity_id", "") or "").strip()
        if entity_id:
            return f"entity:{entity_id}"
        return f"name:{self._name(candidate)}"

    def _unique_candidate_sequence(
        self,
        *pools: Sequence[Any],
    ) -> List[Any]:
        result: List[Any] = []
        seen: set[str] = set()
        for pool in pools:
            for candidate in pool or []:
                if not candidate:
                    continue
                key = self._candidate_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                result.append(candidate)
        return result

    def _arbitration_admission_reason(
        self,
        candidate: Any,
        *,
        pairwise_allowed: bool,
        protected_entry: bool,
        primary_eligible_entry: bool,
        candidate_anchor: str,
    ) -> str:
        if primary_eligible_entry:
            return REASON_PRIMARY_ELIGIBLE
        if candidate_anchor == "AnchorSatisfied":
            return REASON_ANCHOR_SATISFIED
        if protected_entry:
            return REASON_PROTECTED_CONTENDER
        if pairwise_allowed:
            return REASON_PAIRWISE_ALLOWED
        return "PAIRWISE_NOT_ALLOWED"

    def _primary_arbitration(
        self,
        primary: Any,
        pool: Sequence[Any],
        pairwise: Sequence[Dict[str, Any]],
        *,
        full_pool: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        if not primary:
            return {
                "comparisons": [],
                "candidates": [],
                "decision": {},
                "pairwise_discriminating_gaps": [],
            }
        pair_names = {
            tuple(sorted((str(item.get("left") or ""), str(item.get("right") or ""))))
            for item in pairwise or []
            if isinstance(item, dict)
        }
        differential_names = {self._name(item) for item in pool or [] if item}
        admission_pool = self._unique_candidate_sequence(full_pool or [], pool or [])
        contenders = []
        admission_audit: List[Dict[str, Any]] = []
        material_filter_audit: List[Dict[str, Any]] = []
        arbitration_pool_names: set[str] = set()
        arbitration_admission_reason_by_name: Dict[str, str] = {}
        lifecycle_recoveries: List[Dict[str, Any]] = []
        for index, item in enumerate(admission_pool or [], start=1):
            if not item or item is primary or self._name(item) == self._name(primary):
                continue
            pairwise_allowed = (
                tuple(sorted((self._name(primary), self._name(item)))) in pair_names
            )
            protected_entry = (
                has_active_bridge_protection(item, CROSS_SYSTEM_SCOPE)
                or self._protected_recall_candidate(item)
            )
            candidate_anchor = self.clinical_comparator.anchor_status(item)
            current_anchor = self.clinical_comparator.anchor_status(primary)
            primary_eligible_entry = self._primary_eligible_arbitration_entry(item)
            has_core_or_diagnostic = self._core_or_diagnostic_signal(item)
            matched_pattern = bool(
                self.clinical_comparator._matched_bridge_patterns(item)
                or self.clinical_comparator._matched_diagnostic_patterns(item)
            )
            admission_decision = bool(
                pairwise_allowed or protected_entry or primary_eligible_entry
            )
            admission_reason = self._arbitration_admission_reason(
                item,
                pairwise_allowed=pairwise_allowed,
                protected_entry=protected_entry,
                primary_eligible_entry=primary_eligible_entry,
                candidate_anchor=candidate_anchor,
            )
            if not admission_decision:
                audit = self._material_contender_filter_record(
                    item,
                    primary,
                    rank=index,
                    pairwise_allowed=pairwise_allowed,
                    protected_entry=protected_entry,
                    primary_eligible_entry=primary_eligible_entry,
                    candidate_anchor=candidate_anchor,
                    current_anchor=current_anchor,
                    has_core_or_diagnostic=has_core_or_diagnostic,
                    matched_pattern=matched_pattern,
                    material_contender=False,
                    admission_decision=False,
                    admission_reason=admission_reason,
                    filtered_reason="pairwise_not_allowed_and_no_protection",
                    differential_pool_member=self._name(item) in differential_names,
                    arbitration_pool_member=False,
                )
                admission_audit.append(audit)
                material_filter_audit.append(audit)
                continue
            material_contender = self.clinical_comparator.material_contender(item, primary)
            if primary_eligible_entry and not material_contender:
                material_contender = True
                lifecycle_recoveries.append(
                    {
                        "candidate": self._name(item),
                        "entity_id": str(getattr(item, "entity_id", "") or ""),
                        "lifecycle_recovery_applied": True,
                        "recovery_reason": "PRIMARY_ELIGIBLE_MATERIALITY_OVERRIDE",
                        "score_unchanged": True,
                        "evidence_version_unchanged": True,
                        "claim_state_unchanged": True,
                    }
                )
            filtered_reason = ""
            if not material_contender:
                filtered_reason = self._material_contender_filtered_reason(
                    item,
                    primary,
                    candidate_anchor=candidate_anchor,
                    current_anchor=current_anchor,
                    protected_entry=protected_entry,
                    matched_pattern=matched_pattern,
                    primary_eligible_entry=primary_eligible_entry,
                )
            audit = self._material_contender_filter_record(
                item,
                primary,
                rank=index,
                pairwise_allowed=pairwise_allowed,
                protected_entry=protected_entry,
                primary_eligible_entry=primary_eligible_entry,
                candidate_anchor=candidate_anchor,
                current_anchor=current_anchor,
                has_core_or_diagnostic=has_core_or_diagnostic,
                matched_pattern=matched_pattern,
                material_contender=material_contender,
                admission_decision=admission_decision,
                admission_reason=admission_reason,
                filtered_reason=filtered_reason,
                differential_pool_member=self._name(item) in differential_names,
                arbitration_pool_member=bool(material_contender),
            )
            admission_audit.append(audit)
            material_filter_audit.append(audit)
            if not material_contender:
                continue
            arbitration_pool_names.add(self._name(item))
            arbitration_admission_reason_by_name[self._name(item)] = admission_reason
            contenders.append(item)
        contenders = sorted(
            contenders,
            key=lambda item: (
                1 if has_active_bridge_protection(item, CROSS_SYSTEM_SCOPE) else 0,
                self._bridge_strength_rank(item),
                len(self.clinical_comparator.pair_high_value_evidence(primary, item)),
                float(getattr(item, "max_gap_value", 0.0) or 0.0),
                self._core_coverage(item),
                self._judge_score(item),
            ),
            reverse=True,
        )
        mandatory_contenders = [
            item
            for item in contenders
            if arbitration_admission_reason_by_name.get(self._name(item))
            in {
                REASON_PRIMARY_ELIGIBLE,
                REASON_ANCHOR_SATISFIED,
                REASON_PROTECTED_CONTENDER,
            }
        ]
        optional_limit = max(0, 3 - len(mandatory_contenders))
        optional_contenders = [
            item
            for item in contenders
            if item not in mandatory_contenders
        ][:optional_limit]
        contenders = self._unique_candidate_sequence(
            mandatory_contenders,
            optional_contenders,
        )
        records: List[Dict[str, Any]] = []
        candidate_audits: List[Dict[str, Any]] = []
        gaps: List[Dict[str, Any]] = []
        comparison_action_by_name: Dict[str, str] = {}
        comparison_reason_by_name: Dict[str, List[str]] = {}
        selected = primary
        selected_action = KEEP_CURRENT_PRIMARY
        selected_reason_codes: List[str] = []
        defer_reason = ""
        switch_candidate_count = 0

        for contender in contenders:
            high_value = self.clinical_comparator.pair_high_value_evidence(
                primary,
                contender,
            )
            record = self.clinical_comparator.compare(
                primary,
                contender,
                judge_score_current=self._judge_score(primary),
                judge_score_contender=self._judge_score(contender),
                high_value_evidence=high_value,
            )
            records.append(record)
            comparison_action_by_name[self._name(contender)] = str(
                record.get("recommended_action") or ""
            )
            comparison_reason_by_name[self._name(contender)] = list(
                record.get("decision_reason_codes") or []
            )
            candidate_audits.append(
                {
                    "candidate": self._name(contender),
                    "entity_id": str(getattr(contender, "entity_id", "") or ""),
                    "entered_by": self._arbitration_entry_reason(contender),
                    "anchor_status": record.get("candidate_b_analysis", {}).get(
                        "anchor_status",
                        "",
                    ),
                    "matched_bridge_patterns": list(
                        record.get("candidate_b_analysis", {}).get(
                            "matched_bridge_patterns",
                            [],
                        )
                    ),
                    "matched_diagnostic_patterns": list(
                        record.get("candidate_b_analysis", {}).get(
                            "matched_diagnostic_patterns",
                            [],
                        )
                    ),
                    "recommended_action": record.get("recommended_action", ""),
                }
            )
            action = str(record.get("recommended_action") or "")
            if action == SWITCH_PRIMARY:
                switch_candidate_count += 1
                if selected_action != SWITCH_PRIMARY:
                    selected = contender
                    selected_action = action
                    selected_reason_codes = list(record.get("decision_reason_codes") or [])
                continue
            if action == UNLOCK_AND_DEFER:
                if selected_action not in {SWITCH_PRIMARY, UNLOCK_AND_DEFER}:
                    selected = contender
                    selected_action = action
                    selected_reason_codes = list(record.get("decision_reason_codes") or [])
                    defer_reason = (
                        "better_explanatory_candidate_requires_gap_closure:"
                        f" {self._name(contender)} challenges {self._name(primary)}"
                    )
                gaps.append(self._pairwise_discriminating_gap(primary, contender, record))
                continue
            if action == KEEP_CURRENT_AND_DEFER_CONTENDER:
                gaps.append(self._pairwise_discriminating_gap(primary, contender, record))
                if selected_action == KEEP_CURRENT_PRIMARY:
                    selected_action = action
                    selected_reason_codes = list(record.get("decision_reason_codes") or [])
            elif (
                selected_action == KEEP_CURRENT_PRIMARY
                and action not in {REJECT_CONTENDER, NO_MATERIAL_DIFFERENCE}
            ):
                selected_action = action
                selected_reason_codes = list(record.get("decision_reason_codes") or [])

        disposition_audit = self._candidate_disposition_audit(
            primary,
            admission_pool,
            admission_audit,
            comparison_action_by_name,
            comparison_reason_by_name,
            differential_pool_names=differential_names,
            arbitration_pool_names=arbitration_pool_names,
            arbitration_admission_reason_by_name=arbitration_admission_reason_by_name,
        )
        deadlocks = [
            item for item in disposition_audit if item.get("invariant_status") == INVARIANT_DEADLOCK
        ]
        lifecycle_transitions = self._candidate_lifecycle_transitions(disposition_audit)
        summary = self._primary_arbitration_summary(
            admission_pool,
            records,
            candidate_audits,
            disposition_audit,
            deadlocks,
        )
        if not records:
            eligible_challenger_count = sum(
                1
                for item in admission_audit
                if item.get("primary_eligible")
                and item.get("candidate") != self._name(primary)
            )
            if deadlocks:
                reason_codes = ["ARBITRATION_DEADLOCK"]
            elif eligible_challenger_count <= 0:
                reason_codes = ["NO_ELIGIBLE_CHALLENGER"]
            else:
                reason_codes = ["NO_MATERIAL_ARBITRATION_CONTENDER"]
            return {
                "comparisons": [],
                "candidates": [],
                "contender_admission_audit": admission_audit,
                "material_contender_filter": material_filter_audit,
                "candidate_disposition_audit": disposition_audit,
                "arbitration_deadlocks": deadlocks,
                "candidate_lifecycle_transitions": lifecycle_transitions,
                "lifecycle_recoveries": lifecycle_recoveries,
                "summary": summary,
                "decision": {
                    "action": KEEP_CURRENT_PRIMARY,
                    "selected_primary": self._name(primary),
                    "reason_codes": reason_codes,
                },
                "selected_candidate": primary,
                "pairwise_discriminating_gaps": [],
            }
        if selected_action == SWITCH_PRIMARY and switch_candidate_count > 1:
            selected_reason_codes = list(
                dict.fromkeys(
                    selected_reason_codes
                    + ["PRIMARY_SWITCH_WINNER_SELECTION_REASON"]
                )
            )
        return {
            "comparisons": records,
            "candidates": candidate_audits,
            "contender_admission_audit": admission_audit,
            "material_contender_filter": material_filter_audit,
            "candidate_disposition_audit": disposition_audit,
            "arbitration_deadlocks": deadlocks,
            "candidate_lifecycle_transitions": lifecycle_transitions,
            "lifecycle_recoveries": lifecycle_recoveries,
            "summary": summary,
            "decision": {
                "action": selected_action,
                "selected_primary": self._name(selected),
                "original_primary": self._name(primary),
                "reason_codes": selected_reason_codes,
            },
            "selected_candidate": selected,
            "defer_reason": defer_reason,
            "pairwise_discriminating_gaps": gaps,
        }

    def _primary_eligible_arbitration_entry(self, candidate: Any) -> bool:
        if not candidate or bool(getattr(candidate, "hard_contradiction", False)):
            return False
        if self._eligibility_status(candidate) == EXCLUDED:
            return False
        anchor = self.clinical_comparator.anchor_status(candidate)
        if anchor != "AnchorSatisfied" and self._eligibility_status(candidate) != PRIMARY_ELIGIBLE:
            return False
        return bool(
            getattr(candidate, "required_met", False)
            or self._core_or_diagnostic_signal(candidate)
            or self.clinical_comparator._matched_diagnostic_patterns(candidate)
        )

    def _material_contender_filter_record(
        self,
        candidate: Any,
        primary: Any,
        *,
        rank: int,
        pairwise_allowed: bool,
        protected_entry: bool,
        primary_eligible_entry: bool,
        candidate_anchor: str,
        current_anchor: str,
        has_core_or_diagnostic: bool,
        matched_pattern: bool,
        material_contender: bool,
        admission_decision: bool,
        admission_reason: str,
        filtered_reason: str,
        differential_pool_member: bool = False,
        arbitration_pool_member: bool = False,
    ) -> Dict[str, Any]:
        eligibility = self._eligibility_status(candidate)
        return {
            "candidate": self._name(candidate),
            "candidate_id": str(getattr(candidate, "entity_id", "") or ""),
            "entity_id": str(getattr(candidate, "entity_id", "") or ""),
            "rank": int(rank),
            "pairwise_allowed": bool(pairwise_allowed),
            "candidate_anchor_status": str(candidate_anchor or ""),
            "current_primary_anchor_status": str(current_anchor or ""),
            "eligibility_status": eligibility,
            "primary_eligible": bool(
                eligibility == PRIMARY_ELIGIBLE or candidate_anchor == "AnchorSatisfied"
            ),
            "required_met": bool(getattr(candidate, "required_met", False)),
            "has_core_or_diagnostic_evidence": bool(has_core_or_diagnostic),
            "matched_pattern": bool(matched_pattern),
            "protected_entry": bool(protected_entry),
            "primary_eligible_entry": bool(primary_eligible_entry),
            "admission_decision": bool(admission_decision),
            "admission_reason": str(admission_reason or ""),
            "differential_pool_member": bool(differential_pool_member),
            "arbitration_pool_member": bool(arbitration_pool_member),
            "arbitration_admission_reason": str(admission_reason or ""),
            "material_contender": bool(material_contender),
            "filtered_reason": str(filtered_reason or ""),
        }

    def _material_contender_filtered_reason(
        self,
        candidate: Any,
        primary: Any,
        *,
        candidate_anchor: str,
        current_anchor: str,
        protected_entry: bool,
        matched_pattern: bool,
        primary_eligible_entry: bool,
    ) -> str:
        if bool(getattr(candidate, "hard_contradiction", False)):
            return "hard_contradiction"
        if self._eligibility_status(candidate) == EXCLUDED:
            return "excluded_candidate"
        if primary_eligible_entry:
            return "primary_eligible_contender_unexpectedly_non_material"
        if (
            current_anchor != "NoValidAnchor"
            and not protected_entry
            and not matched_pattern
            and candidate_anchor != "AnchorSatisfied"
        ):
            return "current_primary_has_anchor_and_contender_lacks_pattern"
        return "non_material_after_admission"

    def _candidate_disposition_audit(
        self,
        primary: Any,
        pool: Sequence[Any],
        admission_audit: Sequence[Dict[str, Any]],
        comparison_action_by_name: Dict[str, str],
        comparison_reason_by_name: Dict[str, List[str]],
        *,
        differential_pool_names: Optional[set[str]] = None,
        arbitration_pool_names: Optional[set[str]] = None,
        arbitration_admission_reason_by_name: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        admission_by_name = {
            str(item.get("candidate") or ""): dict(item)
            for item in admission_audit or []
            if str(item.get("candidate") or "")
        }
        differential_pool_names = set(differential_pool_names or [])
        arbitration_pool_names = set(arbitration_pool_names or [])
        arbitration_admission_reason_by_name = dict(
            arbitration_admission_reason_by_name or {}
        )
        records: List[Dict[str, Any]] = []
        for index, candidate in enumerate(pool or [], start=1):
            if not candidate:
                continue
            name = self._name(candidate)
            if not name:
                continue
            admission = admission_by_name.get(name, {})
            is_primary = name == self._name(primary)
            anchor = (
                str(admission.get("candidate_anchor_status") or "")
                or self.clinical_comparator.anchor_status(candidate)
            )
            eligibility = (
                str(admission.get("eligibility_status") or "")
                or self._eligibility_status(candidate)
            )
            primary_eligible = bool(
                eligibility == PRIMARY_ELIGIBLE or anchor == "AnchorSatisfied"
            )
            gap_count = self._actionable_workup_count(candidate)
            comparison_action = str(comparison_action_by_name.get(name) or "")
            compared = bool(comparison_action)
            differential_member = bool(
                name in differential_pool_names
                or admission.get("differential_pool_member")
            )
            arbitration_member = bool(
                is_primary
                or name in arbitration_pool_names
                or admission.get("arbitration_pool_member")
            )
            arbitration_reason = (
                str(arbitration_admission_reason_by_name.get(name) or "")
                or str(admission.get("arbitration_admission_reason") or "")
            )
            rejection_reason = ""
            lifecycle_state = ""
            lifecycle_reason = ""
            required_action = ""
            arbitration_status = ""
            invariant_status = INVARIANT_VALID
            deadlock_codes: List[str] = []
            failure_stage = ""
            if is_primary:
                lifecycle_state = LIFECYCLE_PRIMARY
                lifecycle_reason = "CURRENT_PRIMARY"
                required_action = "PRIMARY_DECIDED"
                arbitration_status = "CURRENT_PRIMARY"
            elif compared:
                if comparison_action == REJECT_CONTENDER:
                    lifecycle_state = LIFECYCLE_REJECTED
                    lifecycle_reason = "COMPARATOR_REJECTED"
                    rejection_reason = "clinical_reasoning_comparator_rejected"
                    arbitration_status = "COMPARED_REJECT"
                elif comparison_action == NO_MATERIAL_DIFFERENCE:
                    lifecycle_state = (
                        LIFECYCLE_READY_FOR_ARBITRATION
                        if primary_eligible
                        else LIFECYCLE_DIFFERENTIAL_ONLY
                    )
                    lifecycle_reason = "NO_MATERIAL_DIFFERENCE_AFTER_COMPARISON"
                    arbitration_status = "COMPARED_DEFER"
                else:
                    lifecycle_state = LIFECYCLE_READY_FOR_ARBITRATION
                    lifecycle_reason = "COMPARISON_COMPLETED"
                    arbitration_status = (
                        "COMPARED_SWITCH"
                        if comparison_action == SWITCH_PRIMARY
                        else "COMPARED_DEFER"
                        if comparison_action in {
                            UNLOCK_AND_DEFER,
                            KEEP_CURRENT_AND_DEFER_CONTENDER,
                        }
                        else "COMPARED_KEEP"
                    )
                required_action = "ARBITRATION_RESOLVED"
            elif bool(getattr(candidate, "hard_contradiction", False)):
                lifecycle_state = LIFECYCLE_REJECTED
                lifecycle_reason = "HARD_CONTRADICTION"
                required_action = "NONE"
                rejection_reason = "hard_contradiction"
            elif eligibility == EXCLUDED:
                lifecycle_state = LIFECYCLE_REJECTED
                lifecycle_reason = "EXCLUDED"
                required_action = "NONE"
                rejection_reason = "excluded_candidate"
            elif primary_eligible:
                lifecycle_state = LIFECYCLE_READY_FOR_ARBITRATION
                lifecycle_reason = arbitration_reason or (
                    REASON_PRIMARY_ELIGIBLE
                    if eligibility == PRIMARY_ELIGIBLE
                    else REASON_ANCHOR_SATISFIED
                )
                required_action = "ARBITRATE"
                if not arbitration_member:
                    invariant_status = INVARIANT_DEADLOCK
                    deadlock_codes.append("PRIMARY_ELIGIBLE_NOT_IN_ARBITRATION_POOL")
                    failure_stage = "candidate_routing"
                else:
                    invariant_status = INVARIANT_DEADLOCK
                    deadlock_codes.append("ARBITRATION_MEMBER_NOT_RESOLVED")
                    failure_stage = "arbitration_resolution"
            elif gap_count > 0:
                lifecycle_state = LIFECYCLE_WORKUP_REQUIRED
                lifecycle_reason = "ACTIONABLE_WORKUP_AVAILABLE"
                required_action = "WORKUP"
                arbitration_status = "NOT_REQUIRED"
            elif eligibility == DEFERRED or anchor == "PatternSupportedButUnconfirmed":
                lifecycle_state = LIFECYCLE_WORKUP_REQUIRED
                lifecycle_reason = "DEFERRED_WITHOUT_ACTIONABLE_WORKUP"
                required_action = "WORKUP"
                invariant_status = INVARIANT_DEADLOCK
                deadlock_codes.append("DEFERRED_WITHOUT_ACTIONABLE_WORKUP")
                if not getattr(candidate, "required_gaps", None):
                    deadlock_codes.append("GAPLESS_DEFERRED_CANDIDATE")
                failure_stage = "candidate_routing"
            elif eligibility == DIFFERENTIAL_ONLY or getattr(candidate, "differential_only", False):
                lifecycle_state = LIFECYCLE_DIFFERENTIAL_ONLY
                lifecycle_reason = "DIFFERENTIAL_ONLY_STILL_ALIVE"
                required_action = "MONITOR"
                arbitration_status = "NOT_REQUIRED"
            else:
                lifecycle_state = LIFECYCLE_DIFFERENTIAL_ONLY
                lifecycle_reason = "SEARCH_SPACE_ONLY"
                required_action = "MONITOR"
                arbitration_status = "NOT_REQUIRED"
            if (
                arbitration_member
                and not is_primary
                and not compared
                and not rejection_reason
                and "ARBITRATION_MEMBER_NOT_RESOLVED" not in deadlock_codes
            ):
                invariant_status = INVARIANT_DEADLOCK
                deadlock_codes.append("ARBITRATION_MEMBER_NOT_RESOLVED")
                failure_stage = failure_stage or "arbitration_resolution"
            deadlock_code = "ARBITRATION_DEADLOCK" if deadlock_codes else ""
            final_disposition = self._legacy_disposition(
                lifecycle_state,
                compared=compared,
                comparison_action=comparison_action,
                rejection_reason=rejection_reason,
                gap_count=gap_count,
                is_primary=is_primary,
            )
            records.append(
                {
                    "candidate": name,
                    "candidate_id": str(getattr(candidate, "entity_id", "") or ""),
                    "entity_id": str(getattr(candidate, "entity_id", "") or ""),
                    "rank": index,
                    "eligibility_status": eligibility,
                    "anchor_status": anchor,
                    "primary_eligible": primary_eligible,
                    "pairwise_status": bool(admission.get("pairwise_allowed", False)),
                    "differential_pool_member": differential_member,
                    "arbitration_pool_member": arbitration_member,
                    "arbitration_admission_reason": arbitration_reason,
                    "material_contender_status": bool(
                        admission.get("material_contender", False)
                    ),
                    "comparison_present": compared,
                    "comparison_status": "compared" if compared else "not_compared",
                    "comparison_outcome": comparison_action,
                    "comparison_reason_codes": list(
                        comparison_reason_by_name.get(name) or []
                    ),
                    "rejection_reason": rejection_reason,
                    "active_gap_count": gap_count,
                    "deferred_gap_count": len(getattr(candidate, "required_gaps", []) or []),
                    "lifecycle_state": lifecycle_state,
                    "lifecycle_reason": lifecycle_reason,
                    "required_action": required_action,
                    "arbitration_status": arbitration_status,
                    "source_state_version": int(
                        getattr(candidate, "diagnostic_state_version", 0) or 0
                    ),
                    "disposition_version": int(
                        getattr(candidate, "diagnostic_state_version", 0) or 0
                    ),
                    "invariant_status": invariant_status,
                    "deadlock_codes": deadlock_codes,
                    "final_disposition": final_disposition,
                    "deadlock_code": deadlock_code,
                    "failure_stage": failure_stage,
                }
            )
        return records

    def _legacy_disposition(
        self,
        lifecycle_state: str,
        *,
        compared: bool,
        comparison_action: str,
        rejection_reason: str,
        gap_count: int,
        is_primary: bool,
    ) -> str:
        if is_primary:
            return "CURRENT_PRIMARY"
        if rejection_reason or lifecycle_state == LIFECYCLE_REJECTED:
            return "REJECTED_WITH_REASON"
        if compared:
            if comparison_action == NO_MATERIAL_DIFFERENCE:
                return "NON_MATERIAL_AFTER_COMPARISON"
            return "ARBITRATED"
        if lifecycle_state == LIFECYCLE_WORKUP_REQUIRED or gap_count > 0:
            return "ACTIONABLE_GAP_CREATED"
        if lifecycle_state == LIFECYCLE_READY_FOR_ARBITRATION:
            return "READY_FOR_ARBITRATION"
        if lifecycle_state == LIFECYCLE_DIFFERENTIAL_ONLY:
            return "DIFFERENTIAL_ONLY"
        return "NONE"

    def _actionable_workup_count(self, candidate: Any) -> int:
        count = int(getattr(candidate, "actionable_gap_count", 0) or 0)
        count += len(getattr(candidate, "required_gaps", []) or [])
        count += len(getattr(candidate, "evidence_gaps", []) or [])
        count += len(getattr(candidate, "claim_closure_plan", []) or [])
        count += len(getattr(candidate, "claim_closure_plans", []) or [])
        count += len(getattr(candidate, "pending_exam_bindings", []) or [])
        count += len(getattr(candidate, "pending_history_inquiries", []) or [])
        count += len(getattr(candidate, "pending_relation_resolutions", []) or [])
        if bool(getattr(candidate, "pending_exam_result", False)):
            count += 1
        if self._critical_claim_followup_candidate(candidate):
            count += 1
        return count

    @staticmethod
    def _candidate_lifecycle_transitions(
        disposition_audit: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        transitions: List[Dict[str, Any]] = []
        for item in disposition_audit or []:
            if not isinstance(item, dict):
                continue
            transitions.append(
                {
                    "candidate": str(item.get("candidate") or ""),
                    "entity_id": str(item.get("entity_id") or ""),
                    "diagnostic_state_version": int(
                        item.get("source_state_version") or 0
                    ),
                    "from": "",
                    "to": str(item.get("lifecycle_state") or ""),
                    "trigger": "candidate_lifecycle_projection",
                    "eligibility_status": str(item.get("eligibility_status") or ""),
                    "anchor_status": str(item.get("anchor_status") or ""),
                    "reason": str(item.get("lifecycle_reason") or ""),
                    "required_action": str(item.get("required_action") or ""),
                    "invariant_status": str(item.get("invariant_status") or ""),
                    "deadlock_codes": list(item.get("deadlock_codes") or []),
                }
            )
        return transitions

    @staticmethod
    def _primary_arbitration_summary(
        pool: Sequence[Any],
        records: Sequence[Dict[str, Any]],
        candidate_audits: Sequence[Dict[str, Any]],
        disposition_audit: Sequence[Dict[str, Any]],
        deadlocks: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "primary_eligible_candidate_count": sum(
                1 for item in disposition_audit or [] if item.get("primary_eligible")
            ),
            "admitted_contender_count": len(candidate_audits or []),
            "clinical_reasoning_comparison_count": len(records or []),
            "rejected_contender_count": sum(
                1
                for item in disposition_audit or []
                if item.get("final_disposition") == "REJECTED_WITH_REASON"
            ),
            "gap_routed_contender_count": sum(
                1
                for item in disposition_audit or []
                if item.get("final_disposition") == "ACTIONABLE_GAP_CREATED"
            ),
            "arbitration_deadlock_count": len(deadlocks or []),
            "candidate_count": len([item for item in pool or [] if item]),
        }

    def _arbitration_entry_reason(self, contender: Any) -> str:
        if has_active_bridge_protection(contender, CROSS_SYSTEM_SCOPE):
            return "bridge_protection"
        if self._protected_recall_candidate(contender):
            return "protected_recall"
        return "diagnostic_pattern_or_high_value_evidence"

    def _pairwise_discriminating_gap(
        self,
        current_primary: Any,
        contender: Any,
        comparison: Dict[str, Any],
    ) -> Dict[str, Any]:
        left = self._name(current_primary)
        right = self._name(contender)
        left_analysis = dict(comparison.get("candidate_a_analysis") or {})
        right_analysis = dict(comparison.get("candidate_b_analysis") or {})
        target_evidence = list(
            dict.fromkeys(
                list(right_analysis.get("actionable_gaps") or [])
                + list(right_analysis.get("unexplained_high_value_evidence") or [])
                + list(left_analysis.get("unexplained_high_value_evidence") or [])
                + list(right_analysis.get("explained_high_value_evidence") or [])
            )
        )[:8]
        closure_exams = self._candidate_exam_union([current_primary, contender])[
            : self.discriminating_exam_max_items
        ]
        gap_id = "PWG-" + (
            str(getattr(current_primary, "entity_id", "") or left or "primary")
            + "-"
            + str(getattr(contender, "entity_id", "") or right or "contender")
        ).replace(" ", "_")
        return {
            "gap_id": gap_id,
            "gap_type": "pairwise_discrimination",
            "candidate_a": left,
            "candidate_a_entity_id": str(getattr(current_primary, "entity_id", "") or ""),
            "candidate_b": right,
            "candidate_b_entity_id": str(getattr(contender, "entity_id", "") or ""),
            "target_question": (
                f"distinguish whether {right} or {left} better explains the "
                "verified high-value evidence pattern"
            ),
            "target_evidence": target_evidence,
            "closure_exams": closure_exams,
            "expected_effect_on_arbitration": {
                "contender_pattern_confirmed": "favor_candidate_b",
                "current_primary_anchor_confirmed": "favor_candidate_a",
                "both_unconfirmed": "remain_deferred",
            },
            "source_comparison_id": str(comparison.get("comparison_id") or ""),
            "reason_codes": list(comparison.get("decision_reason_codes") or []),
        }

    def _pairwise_gap_exam_tasks(
        self,
        gaps: Sequence[Dict[str, Any]],
        pool: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        if not gaps:
            return []
        pool_size = max(1, len([item for item in pool or [] if item]))
        tasks: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for gap in gaps or []:
            if not isinstance(gap, dict):
                continue
            gap_id = str(gap.get("gap_id") or "")
            targets = [str(gap.get("candidate_a") or ""), str(gap.get("candidate_b") or "")]
            targets = [item for item in targets if item]
            findings = [str(item) for item in gap.get("target_evidence") or [] if str(item)]
            target_claim = findings[0] if findings else ""
            for exam in gap.get("closure_exams") or []:
                text = str(exam or "").strip()
                if not text:
                    continue
                key = (gap_id, text)
                if key in seen:
                    continue
                seen.add(key)
                tasks.append(
                    {
                        "exam": text,
                        "target_candidates": list(dict.fromkeys(targets)),
                        "target_findings": findings[:12],
                        "target_claims": [target_claim] if target_claim else [],
                        "exam_type": "pairwise_discrimination",
                        "expected_effect": "resolve_primary_arbitration_pairwise_gap",
                        "source": ["clinical_reasoning_primary_arbitration"],
                        "pool_candidate_count": pool_size,
                        "target_candidate_count": len(set(targets)),
                        "information_gain_hint": 1.02,
                        "exam_source": "pairwise_discrimination_exam",
                        "priority_bucket": "high_value_pairwise_gap_closure",
                        "source_gap_id": gap_id,
                        "target_pair": list(dict.fromkeys(targets)),
                        "target_question": str(gap.get("target_question") or ""),
                        "target_claim": target_claim,
                        "exam_role": self._pairwise_exam_role(text, target_claim),
                        "expected_arbitration_effect": dict(
                            gap.get("expected_effect_on_arbitration") or {}
                        ),
                    }
                )
        return tasks

    @staticmethod
    def _pairwise_exam_role(exam: str, target_claim: str = "") -> str:
        text = f"{exam} {target_claim}".lower()
        compact = "".join(text.split())
        if any(token in compact for token in ("hla-b27", "hlab27", "esr", "crp")):
            return "supportive"
        if any(token in text for token in ("衣原体", "淋球菌", "核酸", "培养", "病原")):
            return "trigger_evidence"
        if any(token in text for token in ("眼", "裂隙灯", "结膜", "葡萄膜")):
            return "manifestation_evidence"
        if any(token in text for token in ("关节", "滑膜", "积液", "晶体")):
            return "manifestation_or_exclusion_evidence"
        if any(token in text for token in ("皮肤", "水疱", "疱疹", "皮疹")):
            return "current_primary_anchor_evidence"
        return "pairwise_discrimination"

    @staticmethod
    def _bridge_strength_rank(candidate: Any) -> int:
        ranks = {"weak": 0, "probable": 1, "strong": 2}
        best = 0
        for item in getattr(candidate, "bridge_protection_decisions", []) or []:
            if not isinstance(item, dict):
                continue
            best = max(best, ranks.get(str(item.get("strength") or "weak").lower(), 0))
        for item in getattr(candidate, "bridge_validation_results", []) or []:
            if not isinstance(item, dict):
                continue
            best = max(best, ranks.get(str(item.get("strength") or "weak").lower(), 0))
        return best

    def _apply_primary_arbitration_audit(
        self,
        decision: JudgeDecision,
        arbitration: Dict[str, Any],
    ) -> None:
        decision.clinical_reasoning_comparisons = list(
            arbitration.get("comparisons") or []
        )
        decision.contender_admission_audit = list(
            arbitration.get("contender_admission_audit") or []
        )
        decision.material_contender_filter = list(
            arbitration.get("material_contender_filter") or []
        )
        decision.candidate_disposition_audit = list(
            arbitration.get("candidate_disposition_audit") or []
        )
        decision.candidate_lifecycle_transitions = list(
            arbitration.get("candidate_lifecycle_transitions") or []
        )
        decision.lifecycle_recoveries = list(
            arbitration.get("lifecycle_recoveries") or []
        )
        decision.arbitration_deadlocks = list(
            arbitration.get("arbitration_deadlocks") or []
        )
        decision.primary_arbitration_candidates = list(
            arbitration.get("candidates") or []
        )
        decision.primary_arbitration_decision = dict(
            arbitration.get("decision") or {}
        )
        decision.primary_arbitration_summary = dict(
            arbitration.get("summary") or {}
        )
        if decision.clinical_reasoning_comparisons:
            first = decision.clinical_reasoning_comparisons[0]
            decision.primary_anchor_revalidation = dict(
                first.get("candidate_a_analysis") or {}
            )
        decision.pairwise_discriminating_gaps = list(
            arbitration.get("pairwise_discriminating_gaps") or []
        )
        decision.arbitration_action = str(
            decision.primary_arbitration_decision.get("action") or ""
        )
        decision.arbitration_reason_codes = list(
            decision.primary_arbitration_decision.get("reason_codes") or []
        )
        decision.arbitration_winner = str(
            decision.primary_arbitration_decision.get("selected_primary") or ""
        )
        original = str(
            decision.primary_arbitration_decision.get("original_primary") or ""
        )
        if original and original != decision.arbitration_winner:
            decision.arbitration_loser = original
            decision.primary_unlock_reason = (
                decision.defer_reason
                or "clinical_reasoning_primary_arbitration_changed_primary"
            )

    def _defer_primary_lock_reason(
        self,
        primary: Any,
        pool: Sequence[Any],
        pairwise: Sequence[Dict[str, Any]],
        discriminating_exams: Sequence[str],
    ) -> str:
        conflict_reason = self._conflict_defer_reason(primary, pool)
        if conflict_reason:
            return conflict_reason
        if not primary or not discriminating_exams:
            return ""
        if getattr(primary, "hard_contradiction", False):
            return ""
        contenders = [
            item
            for item in pool
            if item
            and item.diagnosis != primary.diagnosis
            and self._high_value_unresolved_contender(primary, item)
        ]
        if not contenders:
            return ""
        if self._primary_lock_allowed(primary, contenders, pairwise):
            return ""
        names = ", ".join(item.diagnosis for item in contenders[:3])
        return (
            "defer_for_discrimination: high-value unresolved contender(s) "
            f"remain before primary lock: {names}"
        )

    def _conflict_defer_reason(
        self,
        primary: Any,
        pool: Sequence[Any],
    ) -> str:
        if not primary or getattr(primary, "hard_contradiction", False):
            return ""
        if getattr(primary, "unresolved_evidence_conflict", False):
            return (
                "defer_for_conflict: unresolved reasoning-structured evidence "
                f"conflict affects primary {primary.diagnosis}"
            )
        primary_score = self._judge_score(primary)
        contenders = [
            item
            for item in pool or []
            if item
            and item.diagnosis != primary.diagnosis
            and getattr(item, "unresolved_evidence_conflict", False)
            and self._judge_score(item)
            >= primary_score - max(self.pairwise_close_margin, 0.26)
        ]
        if not contenders:
            return ""
        names = ", ".join(item.diagnosis for item in contenders[:3])
        return (
            "defer_for_conflict: unresolved reasoning-structured evidence "
            f"conflict remains for close candidate(s): {names}"
        )

    def _primary_lock_allowed(
        self,
        primary: Any,
        contenders: Sequence[Any],
        pairwise: Sequence[Dict[str, Any]],
    ) -> bool:
        if not self._trusted(primary):
            return False
        gap_state = self._required_gap_state(primary)
        if gap_state in {"hard_contradiction", "unsupported_gap", "actionable_gap"}:
            return False
        if gap_state == "partially_satisfied" and contenders:
            return False
        if self._residual_core_count(primary) > 0 and contenders:
            return False
        if self._core_coverage(primary) < 0.45 and contenders:
            return False
        if self._component_score(primary, "generic_parent_penalty") > 0.0:
            for item in contenders:
                if self._core_or_diagnostic_signal(item):
                    return False
        primary_score = self._judge_score(primary)
        if any(primary_score <= self._judge_score(item) + self.pairwise_close_margin for item in contenders):
            return False
        for comparison in pairwise:
            if not comparison.get("close_call"):
                continue
            if primary.diagnosis in {comparison.get("left"), comparison.get("right")}:
                return False
        return True

    def _high_value_unresolved_contender(self, primary: Any, contender: Any) -> bool:
        if not contender or getattr(contender, "hard_contradiction", False):
            return False
        if getattr(contender, "differential_only", False):
            return False
        if not self._has_signal(contender):
            return False
        if (
            str(getattr(contender, "entity_id", "") or "") == "D100055"
            and self._intracardiac_shunt_primary(primary)
        ):
            return False
        if self._generic_parent_of(primary, contender):
            return False
        if (
            self._stable_structural_primary(primary)
            and self._eligibility_status(contender) == DEFERRED
            and not self._same_family(primary, contender)
            and not self._same_body_system(primary, contender)
        ):
            return False
        if not self._high_value_contender_competes_with_primary(primary, contender):
            return False
        if self._exam_priority_override_candidate(contender):
            return True
        if self._critical_claim_followup_candidate(contender):
            return self._claim_followup_blocks_primary(primary, contender)
        if (
            primary
            and self._trusted(primary)
            and self._eligibility_status(contender) == DEFERRED
            and f"diagnosis:{getattr(primary, 'diagnosis', '')}" in set(getattr(primary, "matched_evidence", []) or [])
            and not self._pattern_deferred_workup_candidate(contender)
        ):
            return False
        if self._trusted(contender) and self._judge_score(contender) > self._judge_score(primary):
            return True
        priority_or_specific = (
            self._priority(contender)
            or self._systemic_primary(contender)
        )
        if not priority_or_specific:
            return False
        actionable_gap = bool(getattr(contender, "required_gaps", None)) or self._gap_authorizable(contender)
        if not actionable_gap and not self._candidate_discriminating_exams(contender):
            return False
        contender_score = self._judge_score(contender)
        primary_score = self._judge_score(primary)
        if self._is_manifestation(primary) or self._generic_parent_of(contender, primary):
            return contender_score >= primary_score - 0.24
        if (
            self._component_score(primary, "generic_parent_penalty") > 0.0
            and self._core_or_diagnostic_signal(contender)
        ):
            return contender_score >= primary_score - max(self.pairwise_close_margin, 0.30)
        if (
            self._coverage(contender)
            >= self._coverage(primary) - 0.12
            and self._residual(contender)
            <= self._residual(primary) + 0.18
        ):
            return contender_score >= primary_score - max(self.pairwise_close_margin, 0.24)
        if (
            self._core_coverage(contender) >= self._core_coverage(primary) + 0.12
            or self._residual_core_count(contender) < self._residual_core_count(primary)
        ):
            return contender_score >= primary_score - max(self.pairwise_close_margin, 0.26)
        return contender_score >= primary_score - self.pairwise_close_margin

    def _claim_followup_blocks_primary(self, primary: Any, contender: Any) -> bool:
        if not self._critical_claim_followup_candidate(contender):
            return False
        if not self._high_value_contender_competes_with_primary(primary, contender):
            return False
        entity_id = str(getattr(contender, "entity_id", "") or "")
        if entity_id in {"D000025", "D100055"}:
            return self._critical_context_signal(contender)
        if not (self._priority(contender) or self._systemic_primary(contender)):
            return False
        if not primary:
            return True
        try:
            source_prior = float(getattr(contender, "source_prior", 0.0) or 0.0)
        except (TypeError, ValueError):
            source_prior = 0.0
        if source_prior < 0.65 and self._judge_score(contender) < self._judge_score(primary) - 0.18:
            return False
        return bool(
            self._candidate_discriminating_exams(contender)
            and self._core_or_diagnostic_signal(contender)
            and source_prior >= 0.45
        )

    def _high_value_contender_competes_with_primary(
        self,
        primary: Any,
        contender: Any,
    ) -> bool:
        if not contender:
            return False
        if not primary:
            return True
        if self._same_family(primary, contender) or self._causally_related(primary, contender):
            return True
        if self._same_body_system(primary, contender):
            return bool(
                self._core_or_diagnostic_signal(primary)
                or self._core_or_diagnostic_signal(contender)
            )
        primary_signals = self._candidate_core_diagnostic_signals(primary)
        contender_signals = self._candidate_core_diagnostic_signals(contender)
        return bool(primary_signals & contender_signals)

    @staticmethod
    def _candidate_core_diagnostic_signals(candidate: Any) -> set[str]:
        if not candidate:
            return set()
        signals = set(getattr(candidate, "core_matched_evidence", []) or [])
        signals.update(getattr(candidate, "diagnostic_matched_evidence", []) or [])
        return {
            str(item)
            for item in signals
            if item
            and not str(item).startswith(("field:", "diagnosis:"))
            and str(item) not in _BROAD_EVIDENCE_TOKENS
        }

    def _stable_structural_primary(self, candidate: Any) -> bool:
        if not candidate or not self._trusted(candidate):
            return False
        dtype = str(getattr(candidate, "diagnosis_type", "") or "").lower()
        if dtype not in {"structural", "anatomical_diagnosis"}:
            return False
        return bool(
            self._component_score(candidate, "objective_evidence") >= 1.0
            or getattr(candidate, "diagnostic_matched_evidence", None)
            or getattr(candidate, "satisfied_required_anchors", None)
        )

    @staticmethod
    def _intracardiac_shunt_primary(primary: Any) -> bool:
        if not primary:
            return False
        matched = set(getattr(primary, "matched_evidence", []) or [])
        if "right_to_left_shunt" not in matched:
            return False
        return bool(
            matched
            & {
                "ventricular_septal_defect",
                "atrial_septal_defect",
                "congenital_heart_defect",
                "diagnosis:\u5ba4\u95f4\u9694\u7f3a\u635f\uff08VSD\uff09",
                "diagnosis:\u623f\u95f4\u9694\u7f3a\u635f",
                "diagnosis:\u5148\u5929\u6027\u5fc3\u810f\u75c5",
            }
        )

    def _required_gap_by_candidate(
        self, pool: Sequence[Any]
    ) -> Dict[str, Dict[str, List[str]]]:
        all_gap_sets = {
            item.diagnosis: set(getattr(item, "required_gaps", []) or [])
            for item in pool
        }
        result: Dict[str, Dict[str, List[str]]] = {}
        for item in pool:
            gaps = list(getattr(item, "required_gaps", []) or [])
            other_gaps = set()
            for name, gap_set in all_gap_sets.items():
                if name != item.diagnosis:
                    other_gaps.update(gap_set)
            discriminating = [gap for gap in gaps if gap not in other_gaps]
            if not discriminating:
                discriminating = gaps[:2]
            result[item.diagnosis] = {
                "confirmatory_gap": gaps[:6],
                "discriminating_gap": discriminating[:6],
            }
        return result

    def _discriminating_findings(
        self,
        pool: Sequence[Any],
        gap_by_candidate: Dict[str, Dict[str, List[str]]],
    ) -> List[str]:
        findings: List[str] = []
        for item in pool:
            payload = gap_by_candidate.get(item.diagnosis) or {}
            for finding in payload.get("discriminating_gap") or []:
                if finding and finding not in findings:
                    findings.append(finding)
            for finding in getattr(item, "residual_evidence", []) or []:
                if finding and finding not in findings:
                    findings.append(finding)
        return findings[:12]

    def _discriminating_exams(
        self,
        pool: Sequence[Any],
        pairwise: Sequence[Dict[str, Any]],
        findings: Sequence[str],
    ) -> List[str]:
        return [
            str(task.get("exam") or "").strip()
            for task in self._discriminating_exam_tasks(pool, pairwise, findings)
            if str(task.get("exam") or "").strip()
        ]

    def _discriminating_exam_tasks(
        self,
        pool: Sequence[Any],
        pairwise: Sequence[Dict[str, Any]],
        findings: Sequence[str],
    ) -> List[Dict[str, Any]]:
        exam_targets: Dict[str, set] = {}
        exam_findings: Dict[str, List[str]] = {}
        exam_sources: Dict[str, set] = {}
        pool_names = {str(getattr(item, "diagnosis", "") or "") for item in pool}

        def add_exam(
            exam: str,
            targets: Sequence[str],
            source: str,
            target_findings: Optional[Sequence[str]] = None,
        ) -> None:
            text = str(exam or "").strip()
            if not text:
                return
            target_names = [
                str(name).strip()
                for name in targets
                if str(name).strip() in pool_names
            ]
            if not target_names:
                target_names = list(pool_names)
            exam_targets.setdefault(text, set()).update(target_names)
            exam_sources.setdefault(text, set()).add(source)
            bucket = exam_findings.setdefault(text, [])
            for finding in target_findings or []:
                value = str(finding or "").strip()
                if value and value not in bucket:
                    bucket.append(value)

        high_prior_candidates = sorted(
            [item for item in pool if self._high_prior_specific_exam_candidate(item)],
            key=self._specific_exam_candidate_key,
            reverse=True,
        )
        for candidate in high_prior_candidates:
            name = str(getattr(candidate, "diagnosis", "") or "")
            candidate_findings = list(getattr(candidate, "required_gaps", []) or []) + list(
                getattr(candidate, "residual_evidence", []) or []
            )
            for item in self._candidate_discriminating_exams(candidate):
                add_exam(item, [name], "high_value_candidate", candidate_findings)
        for names, hinted_exams in _DIFFERENTIAL_SET_EXAM_HINTS:
            if names.issubset(pool_names):
                for item in hinted_exams:
                    add_exam(item, list(names), "differential_set_hint", findings)
        for comparison in pairwise:
            if not comparison.get("close_call"):
                continue
            targets = [
                str(comparison.get("left") or "").strip(),
                str(comparison.get("right") or "").strip(),
            ]
            for item in comparison.get("discriminating_exams") or []:
                add_exam(
                    item,
                    targets,
                    "pairwise_close_call",
                    comparison.get("discriminating_findings") or findings,
                )
        for candidate in pool:
            name = str(getattr(candidate, "diagnosis", "") or "")
            candidate_findings = list(getattr(candidate, "required_gaps", []) or []) + list(
                getattr(candidate, "residual_evidence", []) or []
            )
            for item in self._candidate_discriminating_exams(candidate):
                add_exam(item, [name], "candidate_profile", candidate_findings)

        pool_size = max(1, len(pool_names))
        tasks: List[Dict[str, Any]] = []
        for exam, targets in exam_targets.items():
            target_names = sorted(targets)
            exam_type = self._exam_task_type(exam, target_names, pool_size)
            task_findings = list(dict.fromkeys(exam_findings.get(exam, []) + list(findings)))[:12]
            tasks.append(
                {
                    "exam": exam,
                    "target_candidates": target_names,
                    "target_findings": task_findings,
                    "exam_type": exam_type,
                    "expected_effect": self._exam_expected_effect(exam_type, target_names),
                    "source": sorted(exam_sources.get(exam, [])),
                    "pool_candidate_count": pool_size,
                    "target_candidate_count": len(target_names),
                    "information_gain_hint": round(
                        self._exam_task_score(exam, target_names, pool_size),
                        4,
                    ),
                }
            )

        tasks.sort(
            key=lambda task: (
                self._exam_type_priority(str(task.get("exam_type") or "")),
                float(task.get("information_gain_hint") or 0.0),
                int(task.get("target_candidate_count") or 0),
            ),
            reverse=True,
        )
        return tasks[: self.discriminating_exam_max_items]

    def _conflict_adjudication_exam_tasks(
        self,
        pool: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        seen: set[str] = set()
        pool_size = max(1, len([item for item in pool or [] if item]))
        for candidate in pool or []:
            diagnosis = str(getattr(candidate, "diagnosis", "") or "")
            if not diagnosis or not getattr(candidate, "unresolved_evidence_conflict", False):
                continue
            conflicts = [
                item
                for item in getattr(candidate, "evidence_conflicts", []) or []
                if isinstance(item, dict)
                and str(item.get("status") or "unresolved") != "resolved"
            ]
            if not conflicts:
                continue
            findings = [
                str(item.get("finding") or "").strip()
                for item in conflicts
                if str(item.get("finding") or "").strip()
            ]
            exams = list(getattr(candidate, "conflict_adjudication_exams", []) or [])
            for conflict in conflicts:
                for exam in conflict.get("adjudication_exams") or []:
                    text = str(exam or "").strip()
                    if text and text not in exams:
                        exams.append(text)
            for exam in exams:
                text = str(exam or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                tasks.append(
                    {
                        "exam": text,
                        "target_candidates": [diagnosis],
                        "target_findings": list(dict.fromkeys(findings))[:12],
                        "exam_type": "conflict_adjudication",
                        "expected_effect": "resolve_reasoning_structured_polarity_conflict",
                        "source": ["evidence_conflict_arbiter"],
                        "pool_candidate_count": pool_size,
                        "target_candidate_count": 1,
                        "information_gain_hint": 0.98,
                        "exam_source": "conflict_adjudication_exam",
                    }
                )
        return tasks

    def _pattern_anchor_workup_exam_tasks(
        self,
        pool: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        pool_size = max(1, len([item for item in pool or [] if item]))
        for candidate in sorted(
            [item for item in pool or [] if self._pattern_deferred_workup_candidate(item)],
            key=self._pattern_deferred_workup_sort_key,
            reverse=True,
        ):
            diagnosis = str(getattr(candidate, "diagnosis", "") or "")
            if not diagnosis:
                continue
            target_findings = list(
                dict.fromkeys(
                    list(getattr(candidate, "required_gaps", []) or [])
                    + [
                        finding
                        for pattern in getattr(candidate, "evidence_pattern_matches", []) or []
                        if isinstance(pattern, dict)
                        for group in pattern.get("missing_required_groups", []) or []
                        if isinstance(group, dict)
                        for finding in group.get("missing_findings", []) or []
                    ]
                )
            )[:12]
            for exam in self._candidate_discriminating_exams(candidate)[:4]:
                text = str(exam or "").strip()
                if not text:
                    continue
                key = (diagnosis, text)
                if key in seen:
                    continue
                seen.add(key)
                tasks.append(
                    {
                        "exam": text,
                        "target_candidates": [diagnosis],
                        "target_findings": target_findings,
                        "exam_type": "pattern_anchor_workup",
                        "expected_effect": "close_missing_diagnostic_pattern_anchor",
                        "source": ["diagnostic_pattern_workup"],
                        "pool_candidate_count": pool_size,
                        "target_candidate_count": 1,
                        "information_gain_hint": 0.99,
                        "exam_source": "pattern_anchor_workup_exam",
                    }
                )
        return tasks

    def _claim_followup_exam_tasks(
        self,
        pool: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        pool_size = max(1, len([item for item in pool or [] if item]))
        for candidate in sorted(
            [item for item in pool or [] if self._critical_claim_followup_candidate(item)],
            key=self._claim_followup_sort_key,
            reverse=True,
        ):
            diagnosis = str(getattr(candidate, "diagnosis", "") or "")
            if not diagnosis:
                continue
            claims = [
                item for item in getattr(candidate, "unresolved_critical_evidence_claims", []) or []
                if isinstance(item, dict)
            ]
            for claim in claims:
                exam = str(claim.get("recommended_exam") or "").strip()
                finding = str(claim.get("target_evidence") or "").strip()
                claim_id = str(claim.get("claim_id") or "").strip()
                if not exam:
                    continue
                key = (diagnosis, claim_id, exam)
                if key in seen:
                    continue
                seen.add(key)
                tasks.append(
                    {
                        "exam": exam,
                        "target_candidates": [diagnosis],
                        "target_findings": [finding] if finding else [],
                        "target_claims": [claim_id] if claim_id else [],
                        "exam_type": "evidence_claim_verification",
                        "expected_effect": "verify_or_reject_reasoner_evidence_claim",
                        "source": ["evidence_claim_followup"],
                        "pool_candidate_count": pool_size,
                        "target_candidate_count": 1,
                        "information_gain_hint": 0.99,
                        "exam_source": "evidence_claim_followup_exam",
                    }
                )
        return tasks

    def _annotate_deferred_gap_priorities(self, candidates: Sequence[Any]) -> None:
        for candidate in candidates or []:
            if not candidate:
                continue
            if self._eligibility_status(candidate) != DEFERRED:
                setattr(candidate, "evidence_gaps", [])
                setattr(candidate, "gap_values", [])
                setattr(candidate, "max_gap_value", 0.0)
                setattr(candidate, "actionable_gap_count", 0)
                setattr(candidate, "deferred_priority", 0.0)
                setattr(candidate, "deferred_priority_components", {})
                setattr(candidate, "exam_priority_override", False)
                setattr(candidate, "exam_priority_override_reason", "")
                setattr(candidate, "deferred_priority_status", "")
                continue
            gaps = self._candidate_evidence_gaps(candidate)
            valued_gaps = [self._with_gap_value(candidate, gap) for gap in gaps]
            actionable_gaps = [
                gap for gap in valued_gaps if self._actionable_gap_value(candidate, gap)
            ]
            max_gap_value = max(
                [float(gap.get("gap_value") or 0.0) for gap in actionable_gaps],
                default=0.0,
            )
            components = self._candidate_gap_value_components(actionable_gaps)
            priority = max_gap_value
            override = self._deferred_gap_priority_override(
                candidate,
                actionable_gaps,
                components,
            )
            setattr(candidate, "evidence_gaps", gaps)
            setattr(candidate, "gap_values", valued_gaps)
            setattr(candidate, "max_gap_value", max_gap_value)
            setattr(candidate, "actionable_gap_count", len(actionable_gaps))
            setattr(candidate, "deferred_priority", priority)
            setattr(candidate, "deferred_priority_components", components)
            setattr(candidate, "exam_priority_override", override)
            setattr(
                candidate,
                "exam_priority_override_reason",
                (
                    "high-value Deferred candidate has a closable critical evidence gap"
                    if override
                    else ""
                ),
            )
            setattr(
                candidate,
                "deferred_priority_status",
                "active" if override else "inactive",
            )
            if override:
                setattr(candidate, "exam_followup_authorized", True)

    def _candidate_evidence_gaps(self, candidate: Any) -> List[Dict[str, Any]]:
        diagnosis = self._name(candidate)
        entity_id = str(getattr(candidate, "entity_id", "") or "")
        base_exams = self._candidate_discriminating_exams(candidate)
        gaps: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def add_gap(
            *,
            target: str,
            gap_type: str,
            importance: str = "high",
            closure_exams: Optional[Sequence[str]] = None,
            source: str = "",
            source_claims: Optional[Sequence[str]] = None,
        ) -> None:
            text = str(target or "").strip()
            if not text:
                return
            key = f"{diagnosis}:{gap_type}:{text}"
            if key in seen:
                return
            seen.add(key)
            exams = list(
                dict.fromkeys(
                    str(item or "").strip()
                    for item in list(closure_exams or []) + list(base_exams or [])
                    if str(item or "").strip()
                )
            )
            gap_payload = {
                "gap_id": self._gap_id(candidate, gap_type, text, len(gaps) + 1),
                "candidate": diagnosis,
                "entity_id": entity_id,
                "target_evidence": text,
                "importance": importance,
                "gap_type": gap_type,
                "closure_exams": exams[:6],
                "source": source,
                "source_claims": list(source_claims or []),
                "expected_transition": {
                    "positive": PRIMARY_ELIGIBLE,
                    "negative": DIFFERENTIAL_ONLY,
                },
            }
            claim_plan = self._claim_closure_plan_for_gap(
                candidate,
                gap_payload,
                base_exams=exams[:6],
            )
            if claim_plan:
                gap_payload.update(claim_plan)
            ledger = getattr(candidate, "claim_resolution_ledger", None)
            if ledger and gap_payload.get("claim_requirements"):
                gap_payload = hydrate_gap_with_claim_state(gap_payload, ledger)
            gaps.append(gap_payload)

        for claim in getattr(candidate, "unresolved_critical_evidence_claims", []) or []:
            if not isinstance(claim, dict):
                continue
            claim_type = str(claim.get("claim_type") or "")
            if claim_type == "derived_pattern" or claim.get("required_inputs"):
                gap_type = "derived_pattern_gap"
            else:
                gap_type = "observed_evidence_gap"
            target = str(claim.get("target_evidence") or "").strip()
            exam = str(claim.get("recommended_exam") or "").strip()
            add_gap(
                target=target,
                gap_type=gap_type,
                importance=str(claim.get("importance") or "critical"),
                closure_exams=[exam] if exam else [],
                source="evidence_claim",
                source_claims=[str(claim.get("claim_id") or "").strip()],
            )

        for pattern in getattr(candidate, "evidence_pattern_matches", []) or []:
            if not isinstance(pattern, dict):
                continue
            pattern_id = str(pattern.get("pattern_id") or "").strip()
            for index, group in enumerate(pattern.get("missing_required_groups", []) or []):
                if not isinstance(group, dict):
                    continue
                findings = [
                    str(item or "").strip()
                    for item in group.get("missing_findings", []) or []
                    if str(item or "").strip()
                ]
                target = "|".join(findings) or f"{pattern_id}:missing_group:{index + 1}"
                add_gap(
                    target=target,
                    gap_type="derived_pattern_gap",
                    importance="critical",
                    closure_exams=base_exams,
                    source="diagnostic_pattern",
                    source_claims=[pattern_id] if pattern_id else [],
                )

        for gap in getattr(candidate, "required_gaps", []) or []:
            add_gap(
                target=str(gap or "").strip(),
                gap_type="confirmation_gap",
                importance="critical" if self._priority(candidate) else "high",
                closure_exams=base_exams,
                source="required_anchor",
            )
        return gaps

    def _claim_closure_plan_for_gap(
        self,
        candidate: Any,
        gap: Dict[str, Any],
        *,
        base_exams: Sequence[str],
    ) -> Dict[str, Any]:
        text = " ".join(
            str(item or "")
            for item in (
                getattr(candidate, "entity_id", ""),
                getattr(candidate, "diagnosis", ""),
                gap.get("target_evidence"),
                gap.get("gap_id"),
            )
        )
        compact = "".join(text.lower().split())
        is_radiation_lung = any(
            marker in compact
            for marker in (
                "d100058",
                "radiation",
                "radiotherapy",
                "post_radiotherapy",
                "放射",
                "放疗",
            )
        )
        if not is_radiation_lung:
            return {}
        chest_ct = [
            str(exam)
            for exam in base_exams or []
            if any(marker in str(exam).lower() for marker in ("ct", "胸部", "chest"))
        ]
        route_exam = chest_ct[0] if chest_ct else "胸部CT扫描（Chest CT）"
        return {
            "claim_closure_plan_version": "claim_closure_plan_v1",
            "claim_requirements": [
                {
                    "claim_id": "pulmonary_morphology",
                    "claim_type": "composite_observation",
                    "fulfillment_rule": "ANY",
                    "expected_evidence_concepts": [
                        "ground_glass_opacity",
                        "pulmonary_consolidation",
                        "patchy_pulmonary_opacity",
                        "pulmonary_opacity",
                        "pulmonary_infiltrative_opacity",
                    ],
                    "allowed_source_types": ["imaging_result", "exam_result_observation"],
                    "required_for_anchor": True,
                    "route_ids": ["route_pulmonary_morphology_ct"],
                },
                {
                    "claim_id": "radiation_field_lung_consistency",
                    "claim_type": "spatial_relation",
                    "fulfillment_rule": "SUPPORTED_RELATION",
                    "expected_evidence_concepts": [
                        "lesion_within_prior_radiation_field",
                    ],
                    "allowed_source_types": ["imaging_result", "explicit_relation"],
                    "required_for_anchor": True,
                    "route_ids": ["route_radiation_field_ct"],
                },
                {
                    "claim_id": "post_radiotherapy_time_window",
                    "claim_type": "temporal_relation",
                    "fulfillment_rule": "DERIVED_TEMPORAL_RELATION",
                    "expected_evidence_concepts": [
                        "radiotherapy_end_date",
                        "pulmonary_symptom_onset",
                        "radiotherapy_before_pulmonary_onset",
                    ],
                    "allowed_source_types": [
                        "patient_reported_observation",
                        "treatment_history",
                        "derived_relation",
                    ],
                    "required_for_anchor": True,
                    "route_ids": [
                        "route_radiotherapy_timing_history",
                        "route_post_radiotherapy_temporal_relation",
                    ],
                },
            ],
            "closure_routes": [
                {
                    "route_id": "route_pulmonary_morphology_ct",
                    "route_type": "exam_result",
                    "exam": route_exam,
                    "target_claims": ["pulmonary_morphology"],
                    "expected_evidence_concepts": [
                        "ground_glass_opacity",
                        "pulmonary_consolidation",
                        "patchy_pulmonary_opacity",
                        "pulmonary_opacity",
                        "pulmonary_infiltrative_opacity",
                    ],
                },
                {
                    "route_id": "route_radiation_field_ct",
                    "route_type": "exam_result",
                    "exam": route_exam,
                    "target_claims": ["radiation_field_lung_consistency"],
                    "expected_evidence_concepts": [
                        "lesion_within_prior_radiation_field",
                        "lesion_outside_prior_radiation_field",
                    ],
                },
                {
                    "route_id": "route_radiotherapy_timing_history",
                    "route_type": "history_inquiry",
                    "inquiry_targets": [
                        "radiotherapy_end_date",
                        "pulmonary_symptom_onset",
                    ],
                    "target_claims": ["post_radiotherapy_time_window"],
                },
                {
                    "route_id": "route_post_radiotherapy_temporal_relation",
                    "route_type": "temporal_relation",
                    "inputs": [
                        "radiotherapy_end_date",
                        "pulmonary_symptom_onset",
                    ],
                    "target_claims": ["post_radiotherapy_time_window"],
                },
            ],
            "claim_resolutions": [],
        }

    @staticmethod
    def _gap_id(candidate: Any, gap_type: str, target: str, index: int) -> str:
        entity_id = str(getattr(candidate, "entity_id", "") or "").strip()
        name = entity_id or str(getattr(candidate, "diagnosis", "") or "candidate")
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in name)[:32] or "candidate"
        safe_type = "".join(ch if ch.isalnum() else "_" for ch in gap_type)[:24]
        safe_target = "".join(ch if ch.isalnum() else "_" for ch in target)[:32]
        return f"G-{safe_name}-{safe_type}-{index}-{safe_target}"

    def _with_gap_value(self, candidate: Any, gap: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(gap or {})
        closure_exams = [
            str(exam or "").strip()
            for exam in result.get("closure_exams", []) or []
            if str(exam or "").strip()
        ]
        components = self._gap_value_components(candidate, result, closure_exams)
        gap_value = round(
            max(
                0.0,
                min(
                    1.0,
                    components["clinical_impact"]
                    + components["decision_change_potential"]
                    + components["evidence_specificity"]
                    + components["uncertainty_reduction"]
                    + components["closure_exam_quality"]
                    + components["missed_diagnosis_risk"]
                    - components["redundancy"]
                    - components["exam_cost"]
                    - components["exam_risk"],
                ),
            ),
            4,
        )
        return EvidenceGapValue(
            gap_id=str(result.get("gap_id") or ""),
            entity_id=str(result.get("entity_id") or getattr(candidate, "entity_id", "") or ""),
            candidate=str(result.get("candidate") or self._name(candidate)),
            target_evidence=str(result.get("target_evidence") or ""),
            gap_type=str(result.get("gap_type") or ""),
            gap_value=gap_value,
            gap_value_components=components,
            expected_transition=dict(result.get("expected_transition") or {}),
            closure_exams=closure_exams,
            hard_contradiction=bool(
                result.get("hard_contradiction")
                or getattr(candidate, "hard_contradiction", False)
            ),
            already_attempted_exams=list(result.get("already_attempted_exams") or []),
            candidate_score_at_decision=float(getattr(candidate, "score", 0.0) or 0.0),
            score_gap_decoupled=True,
            metadata=result,
        ).to_dict()

    def _gap_value_components(
        self,
        candidate: Any,
        gap: Dict[str, Any],
        closure_exams: Sequence[str],
    ) -> Dict[str, float]:
        best_closure = self._best_gap_closure_quality(candidate, gap, closure_exams)
        return {
            "clinical_impact": self._gap_clinical_impact(candidate, gap),
            "decision_change_potential": self._gap_decision_change_potential(candidate, gap),
            "evidence_specificity": self._gap_evidence_specificity(candidate, gap),
            "uncertainty_reduction": self._gap_uncertainty_reduction(candidate, gap),
            "closure_exam_quality": best_closure,
            "missed_diagnosis_risk": self._gap_missed_diagnosis_risk(candidate, gap),
            "redundancy": self._gap_redundancy(candidate, gap, closure_exams),
            "exam_cost": min(0.05, 0.14 * self._exam_cost(closure_exams)),
            "exam_risk": min(0.05, 0.18 * self._exam_risk(closure_exams)),
        }

    def _candidate_gap_value_components(
        self,
        gaps: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        if not gaps:
            return {}
        best = max(gaps, key=lambda item: float(item.get("gap_value") or 0.0))
        return dict(best.get("gap_value_components") or {})

    def _actionable_gap_value(self, candidate: Any, gap: Dict[str, Any]) -> bool:
        if not gap or getattr(candidate, "hard_contradiction", False):
            return False
        if not gap.get("closure_exams"):
            return False
        if float(gap.get("gap_value") or 0.0) <= 0.0:
            return False
        transition = gap.get("expected_transition") or {}
        return bool(transition.get("positive") or transition.get("negative"))

    def _best_gap_closure_quality(
        self,
        candidate: Any,
        gap: Dict[str, Any],
        closure_exams: Sequence[str],
    ) -> float:
        values: List[float] = []
        for exam in closure_exams or []:
            resolution = self._gap_closure_exam_resolution(exam, candidate)
            coverage = self._gap_specific_exam_coverage(candidate, gap, exam, resolution)
            resolution_type = str(resolution.get("resolution_type") or "")
            resolution_bonus = (
                0.03
                if resolution_type in _FULL_CLOSURE_RESOLUTION_TYPES
                else 0.015
                if resolution_type == PARTIAL_SUBSTITUTE
                else 0.0
            )
            values.append(min(0.18, 0.15 * float(coverage or 0.0) + resolution_bonus))
        return round(max(values, default=0.0), 4)

    def _gap_clinical_impact(self, candidate: Any, gap: Dict[str, Any]) -> float:
        if self._critical_risk_candidate(candidate):
            return 0.16
        if self._priority(candidate):
            return 0.12
        if str((gap or {}).get("importance") or "").lower() == "critical":
            return 0.10
        return 0.06

    @staticmethod
    def _gap_decision_change_potential(candidate: Any, gap: Dict[str, Any]) -> float:
        transition = gap.get("expected_transition") or {}
        positive = str(transition.get("positive") or "").lower()
        negative = str(transition.get("negative") or "").lower()
        if "primary" in positive and (
            "differential" in negative or "excluded" in negative or "reject" in negative
        ):
            return 0.22
        if "primary" in positive:
            return 0.18
        if "differential" in negative or "excluded" in negative:
            return 0.14
        return 0.08

    def _gap_evidence_specificity(self, candidate: Any, gap: Dict[str, Any]) -> float:
        target = str((gap or {}).get("target_evidence") or "").lower()
        if any(
            token in target
            for token in (
                "cta",
                "vascular",
                "shunt",
                "blast",
                "bone_marrow",
                "vitamin_d_low",
                "bone_deformity",
                "culture_positive",
                "anca_positive",
            )
        ):
            return 0.15
        if str((gap or {}).get("gap_type") or "") in {
            "confirmation_gap",
            "derived_pattern_gap",
        }:
            return 0.12
        return 0.07

    @staticmethod
    def _gap_uncertainty_reduction(candidate: Any, gap: Dict[str, Any]) -> float:
        gap_type = str((gap or {}).get("gap_type") or "")
        if gap_type in {"confirmation_gap", "derived_pattern_gap"}:
            return 0.15
        if gap_type == "observed_evidence_gap":
            return 0.13
        return 0.08

    def _gap_missed_diagnosis_risk(self, candidate: Any, gap: Dict[str, Any]) -> float:
        if self._critical_risk_candidate(candidate):
            return 0.13
        if self._priority(candidate):
            return 0.09
        return 0.04

    @staticmethod
    def _gap_redundancy(candidate: Any, gap: Dict[str, Any], closure_exams: Sequence[str]) -> float:
        attempts = len(gap.get("already_attempted_exams") or [])
        duplicate_exam_penalty = max(0, len(list(closure_exams or [])) - len(set(closure_exams or [])))
        return round(min(0.08, 0.025 * attempts + 0.02 * duplicate_exam_penalty), 4)

    def _deferred_priority_components(
        self,
        candidate: Any,
        gaps: Sequence[Dict[str, Any]],
    ) -> Dict[str, float]:
        closure_exams = [
            exam
            for gap in gaps or []
            for exam in gap.get("closure_exams", []) or []
            if str(exam or "").strip()
        ]
        return {
            "clinical_value": self._deferred_clinical_value(candidate),
            "evidence_support": self._verified_support_score(candidate),
            "expected_information_gain": 1.0 if closure_exams else 0.0,
            "eligibility_change_potential": 1.0 if gaps else 0.0,
            "missed_diagnosis_risk": self._missed_diagnosis_risk(candidate),
            "exam_cost": self._exam_cost(closure_exams),
            "exam_risk": self._exam_risk(closure_exams),
            "redundancy": min(0.35, 0.06 * max(0, len(closure_exams) - 3)),
        }

    def _deferred_gap_priority_override(
        self,
        candidate: Any,
        gaps: Sequence[Dict[str, Any]],
        components: Dict[str, float],
    ) -> bool:
        if self._eligibility_status(candidate) != DEFERRED:
            return False
        if getattr(candidate, "hard_contradiction", False):
            return False
        if getattr(candidate, "unresolved_evidence_conflict", False):
            return False
        substatus = str(getattr(candidate, "eligibility_substatus", "") or "")
        if substatus not in _DEFERRED_EXAM_OVERRIDE_SUBSTATUSES:
            return False
        if not gaps or not any(gap.get("closure_exams") for gap in gaps):
            return False
        support_score = self._verified_support_score(candidate)
        if support_score <= 0.0:
            return False
        if self._critical_risk_candidate(candidate):
            if not self._critical_context_signal(candidate):
                return False
        elif support_score < 0.25:
            return False
        if not self._deferred_high_value_candidate(candidate):
            return False
        return max(float(gap.get("gap_value") or 0.0) for gap in gaps) >= 0.58

    def _deferred_high_value_candidate(self, candidate: Any) -> bool:
        if self._explicit_deferred_high_value(candidate):
            return True
        if self._critical_risk_candidate(candidate):
            return True
        if self._critical_claim_followup_candidate(candidate) and self._verified_support_score(candidate) >= 0.25:
            return True
        return False

    @staticmethod
    def _explicit_deferred_high_value(candidate: Any) -> bool:
        if bool(getattr(candidate, "unresolved_high_value", False)):
            return True
        value = str(getattr(candidate, "candidate_value", "") or "").lower()
        risk = str(getattr(candidate, "risk_level", "") or "").lower()
        return value in {"high", "critical"} or risk in {"high", "critical"}

    @staticmethod
    def _critical_risk_candidate(candidate: Any) -> bool:
        entity_id = str(getattr(candidate, "entity_id", "") or "")
        name = str(getattr(candidate, "diagnosis", "") or "")
        canonical = str(getattr(candidate, "canonical_name", "") or "")
        text = f"{name} {canonical}".lower()
        critical_names = (
            "\u767d\u8840\u75c5",
            "\u80ba\u52a8\u9759\u8109\u7618",
            "\u663e\u5fae\u955c\u4e0b\u591a\u8840\u7ba1\u708e",
            "\u7ed3\u6838\u6027\u5fc3\u5305\u708e",
        )
        return bool(
            entity_id in {"D000025", "D100055"}
            or "leukemia" in text
            or "pavm" in text
            or any(item in name or item in canonical for item in critical_names)
        )

    def _critical_context_signal(self, candidate: Any) -> bool:
        entity_id = str(getattr(candidate, "entity_id", "") or "")
        name = str(getattr(candidate, "diagnosis", "") or "")
        canonical = str(getattr(candidate, "canonical_name", "") or "")
        text = f"{name} {canonical}".lower()
        support = {
            str(item or "").strip()
            for item in self._specific_support_evidence(candidate)
            if str(item or "").strip()
        }
        if entity_id == "D100055" or "pavm" in text or "\u80ba\u52a8\u9759\u8109\u7618" in name:
            return bool(
                support
                & {
                    "hemoptysis",
                    "hypoxemia",
                    "pulmonary_nodule",
                    "pulmonary_vascular_shunt",
                    "pulmonary_avm_mechanism",
                    "vascular_pulmonary_nodule_suspected",
                    "enhanced_ct_vascular_malformation",
                    "pulmonary_cta_positive",
                    "bubble_echo_right_to_left_shunt",
                }
            )
        if entity_id == "D000025" or "leukemia" in text or "\u767d\u8840\u75c5" in name:
            return bool(
                support
                & {
                    "blast_present",
                    "blast_percentage_high",
                    "multilineage_cytopenia",
                    "acute_leukemia_pattern",
                    "anemia",
                    "platelet_low",
                    "white_blood_cell_abnormal",
                    "bleeding_tendency",
                }
            )
        if "\u7ed3\u6838\u6027\u5fc3\u5305\u708e" in name or "tuberculous pericarditis" in text:
            return bool(
                any(item.startswith("pericard") for item in support)
                or support
                & {
                    "tb_exposure",
                    "tuberculosis_exposure",
                    "tuberculosis_pattern",
                    "tb_naat_positive",
                    "afb_positive",
                    "xpert_mtb_positive",
                    "diagnosis:\u80ba\u7ed3\u6838",
                }
            )
        if "\u663e\u5fae\u955c\u4e0b\u591a\u8840\u7ba1\u708e" in name or "polyangiitis" in text:
            return bool(
                support
                & {
                    "anca_positive",
                    "microscopic_hematuria",
                    "red_cell_casts",
                    "renal_impairment",
                    "pulmonary_hemorrhage",
                    "hemoptysis",
                }
            )
        return bool(support)

    def _verified_support_score(self, candidate: Any) -> float:
        matched = [
            str(item or "")
            for item in getattr(candidate, "matched_evidence", []) or []
            if str(item or "")
        ]
        concrete = [
            item
            for item in matched
            if self._specific_evidence_token(item)
        ]
        core = [
            str(item or "")
            for item in getattr(candidate, "core_matched_evidence", []) or []
            if self._specific_evidence_token(str(item or ""))
        ]
        diagnostic = [
            str(item or "")
            for item in getattr(candidate, "diagnostic_matched_evidence", []) or []
            if self._specific_evidence_token(str(item or ""))
        ]
        verified_claims = [
            item
            for item in getattr(candidate, "evidence_claims", []) or []
            if isinstance(item, dict)
            and str(item.get("status") or "") in {"Verified", "Derived"}
        ]
        score = 0.0
        if concrete:
            score += 0.25
        score += min(0.30, 0.10 * len(core))
        score += min(0.30, 0.15 * len(diagnostic))
        score += min(0.25, 0.12 * len(verified_claims))
        return round(min(1.0, score), 4)

    @staticmethod
    def _specific_evidence_token(item: str) -> bool:
        text = str(item or "").strip()
        if not text:
            return False
        if text.startswith("field:") or text.startswith("diagnosis:"):
            return False
        lower = text.lower()
        return text not in _BROAD_EVIDENCE_TOKENS and lower not in _BROAD_EVIDENCE_TOKENS

    def _specific_support_evidence(self, candidate: Any) -> List[str]:
        items = list(getattr(candidate, "matched_evidence", []) or [])
        items.extend(getattr(candidate, "core_matched_evidence", []) or [])
        items.extend(getattr(candidate, "diagnostic_matched_evidence", []) or [])
        return [
            str(item or "").strip()
            for item in items
            if self._specific_evidence_token(str(item or ""))
        ]

    def _deferred_clinical_value(self, candidate: Any) -> float:
        if self._critical_risk_candidate(candidate):
            return 1.0
        if self._priority(candidate):
            return 0.72
        if self._critical_claim_followup_candidate(candidate):
            return 0.68
        return 0.35

    def _missed_diagnosis_risk(self, candidate: Any) -> float:
        if self._critical_risk_candidate(candidate):
            return 0.9
        if self._priority(candidate):
            return 0.55
        return 0.25

    @staticmethod
    def _exam_cost(exams: Sequence[str]) -> float:
        if not exams:
            return 0.0
        cost = 0.0
        for exam in exams:
            text = str(exam or "")
            if any(token in text for token in ("\u9aa8\u9ad3", "\u6d3b\u68c0", "\u9020\u5f71", "\u5bfc\u7ba1")):
                cost += 0.12
            elif any(token in text for token in ("CT", "CTA", "MRI", "\u589e\u5f3a")):
                cost += 0.07
            else:
                cost += 0.03
        return round(min(0.35, cost), 4)

    @staticmethod
    def _exam_risk(exams: Sequence[str]) -> float:
        if not exams:
            return 0.0
        risk = 0.0
        for exam in exams:
            text = str(exam or "")
            if any(token in text for token in ("\u9aa8\u9ad3", "\u6d3b\u68c0", "\u9020\u5f71", "\u5bfc\u7ba1")):
                risk += 0.08
            elif any(token in text for token in ("CTA", "\u589e\u5f3a")):
                risk += 0.04
            else:
                risk += 0.01
        return round(min(0.25, risk), 4)

    def _exam_priority_override_candidate(self, candidate: Any) -> bool:
        return bool(getattr(candidate, "exam_priority_override", False))

    def _forced_pool_names_for_deferred_gap_override(
        self,
        candidates: Sequence[Any],
    ) -> List[str]:
        return [
            self._name(item)
            for item in candidates or []
            if self._exam_priority_override_candidate(item) and self._name(item)
        ][: self.gap_target_limit]

    def _forced_pool_names_for_bridge_protection(
        self,
        candidates: Sequence[Any],
    ) -> List[str]:
        protected = [
            item
            for item in candidates or []
            if has_active_bridge_protection(item, CROSS_SYSTEM_SCOPE)
            and self._name(item)
            and not getattr(item, "hard_contradiction", False)
        ]
        protected.sort(key=self._bridge_protection_sort_key, reverse=True)
        return [self._name(item) for item in protected[:3]]

    def _forced_pool_names_for_protected_recall(
        self,
        candidates: Sequence[Any],
    ) -> List[str]:
        protected = [
            item
            for item in candidates or []
            if self._protected_recall_candidate(item)
            and self._name(item)
            and not getattr(item, "hard_contradiction", False)
        ]
        protected.sort(
            key=lambda item: (
                self._core_coverage(item),
                float(getattr(item, "max_gap_value", 0.0) or 0.0),
                self._judge_score(item),
            ),
            reverse=True,
        )
        return [self._name(item) for item in protected[:3]]

    @staticmethod
    def _protected_recall_candidate(candidate: Any) -> bool:
        for source in getattr(candidate, "candidate_sources", []) or []:
            if not isinstance(source, dict):
                continue
            metadata = dict(source.get("metadata") or {})
            if (
                str(source.get("source") or "") == "llm_pattern_hypothesis"
                and bool(metadata.get("protected_pool_slot"))
                and str(metadata.get("recall_mode") or "") == "protected_recall"
            ):
                return True
        return False

    def _bridge_protection_sort_key(self, candidate: Any) -> tuple:
        strength_rank = 0
        for item in getattr(candidate, "bridge_protection_decisions", []) or []:
            if not isinstance(item, dict):
                continue
            strength = str(item.get("strength") or "").lower()
            strength_rank = max(
                strength_rank,
                {"weak": 0, "probable": 1, "strong": 2}.get(strength, 0),
            )
        return (
            strength_rank,
            self._core_coverage(candidate),
            float(getattr(candidate, "max_gap_value", 0.0) or 0.0),
            self._judge_score(candidate),
        )

    def _deferred_gap_closure_exam_tasks(
        self,
        pool: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        pool_size = max(1, len([item for item in pool or [] if item]))
        candidates = sorted(
            [item for item in pool or [] if self._exam_priority_override_candidate(item)],
            key=lambda item: (
                float(getattr(item, "max_gap_value", 0.0) or 0.0),
                int(getattr(item, "actionable_gap_count", 0) or 0),
                self._verified_support_score(item),
            ),
            reverse=True,
        )
        for candidate in candidates[: self.gap_target_limit]:
            diagnosis = self._name(candidate)
            candidate_task_count = 0
            candidate_task_limit = min(3, max(1, self.discriminating_exam_max_items))
            candidate_gaps = sorted(
                [
                    gap
                    for gap in getattr(candidate, "gap_values", []) or []
                    if isinstance(gap, dict)
                    and self._actionable_gap_value(candidate, gap)
                ],
                key=lambda gap: float(gap.get("gap_value") or 0.0),
                reverse=True,
            )
            for gap_value_rank, gap in enumerate(candidate_gaps, start=1):
                if candidate_task_count >= candidate_task_limit:
                    break
                gap_id = str(gap.get("gap_id") or "").strip()
                target = str(gap.get("target_evidence") or "").strip()
                closure_exam_items = [
                    (index, str(exam or "").strip())
                    for index, exam in enumerate(gap.get("closure_exams", []) or [])
                    if str(exam or "").strip()
                ]
                closure_exam_items.sort(
                    key=lambda item: self._gap_closure_exam_sort_key(
                        candidate,
                        gap,
                        item[1],
                        item[0],
                    ),
                    reverse=True,
                )
                for closure_rank, (original_index, exam) in enumerate(
                    closure_exam_items,
                    start=1,
                ):
                    if candidate_task_count >= candidate_task_limit:
                        break
                    key = (diagnosis, gap_id, exam)
                    if key in seen:
                        continue
                    seen.add(key)
                    resolution = self._gap_closure_exam_resolution(exam, candidate)
                    resolution_type = str(resolution.get("resolution_type") or "")
                    diagnostic_coverage = float(
                        resolution.get("diagnostic_coverage") or 0.0
                    )
                    gap_coverage = self._gap_specific_exam_coverage(
                        candidate,
                        gap,
                        exam,
                        resolution,
                    )
                    closure_priority = self._gap_specific_exam_preference(
                        candidate,
                        gap,
                        exam,
                        resolution,
                    )
                    candidate_task_count += 1
                    tasks.append(
                        {
                            "exam": exam,
                            "target_candidates": [diagnosis],
                            "target_findings": [target] if target else [],
                            "target_gap": gap_id,
                            "target_gaps": [gap_id] if gap_id else [],
                            "evidence_gap": dict(gap),
                            "exam_type": "deferred_gap_closure",
                            "expected_effect": "close_high_value_deferred_evidence_gap",
                            "expected_transition": dict(gap.get("expected_transition") or {}),
                            "source": ["deferred_gap_closure"],
                            "pool_candidate_count": pool_size,
                            "target_candidate_count": 1,
                            "information_gain_hint": min(
                                1.0,
                                max(0.50, float(gap.get("gap_value") or 0.0)),
                            ),
                            "exam_source": "deferred_gap_closure_exam",
                            "priority_override": True,
                            "priority_bucket": "high_value_deferred_gap_closure",
                            "closure_rank": closure_rank,
                            "closure_priority": closure_priority,
                            "gap_value_rank": gap_value_rank,
                            "source_gap_value": float(gap.get("gap_value") or 0.0),
                            "gap_value_components": dict(
                                gap.get("gap_value_components") or {}
                            ),
                            "candidate_score_at_decision": float(
                                getattr(candidate, "score", 0.0) or 0.0
                            ),
                            "score_gap_decoupled": True,
                            "original_closure_exam_index": original_index,
                            "requested_exam": str(
                                resolution.get("requested_exam") or exam
                            ),
                            "resolved_exam": str(resolution.get("resolved_exam") or ""),
                            "resolution_type": resolution_type,
                            "diagnostic_coverage": diagnostic_coverage,
                            "gap_diagnostic_coverage": gap_coverage,
                            "exam_gap_closure_value": round(
                                min(
                                    1.0,
                                    0.62 * float(gap.get("gap_value") or 0.0)
                                    + 0.28 * float(gap_coverage or 0.0)
                                    + 0.10 * min(1.0, closure_priority / 100.0),
                                ),
                                4,
                            ),
                            "exam_resolution": dict(resolution),
                            "override_reason": str(
                                getattr(candidate, "exam_priority_override_reason", "") or ""
                            ),
                        }
                    )
        return tasks

    def _gap_closure_exam_resolution(
        self,
        exam: str,
        candidate: Any,
    ) -> Dict[str, Any]:
        if self.exam_resolver is None:
            return {
                "requested_exam": str(exam or ""),
                "resolved_exam": str(exam or ""),
                "resolution_type": "unresolved",
                "diagnostic_coverage": 0.0,
                "reason": "no exam resolver available",
                "candidate": self._name(candidate),
            }
        return self.exam_resolver.resolve(exam, candidate=self._name(candidate)).to_dict()

    def _gap_closure_exam_sort_key(
        self,
        candidate: Any,
        gap: Dict[str, Any],
        exam: str,
        original_index: int,
    ) -> tuple:
        resolution = self._gap_closure_exam_resolution(exam, candidate)
        resolution_type = str(resolution.get("resolution_type") or "")
        resolution_rank = (
            3
            if resolution_type in _FULL_CLOSURE_RESOLUTION_TYPES
            else 2
            if resolution_type == PARTIAL_SUBSTITUTE
            else 0
        )
        diagnostic_coverage = float(resolution.get("diagnostic_coverage") or 0.0)
        closure_priority = self._gap_specific_exam_preference(
            candidate,
            gap,
            exam,
            resolution,
        )
        gap_coverage = self._gap_specific_exam_coverage(
            candidate,
            gap,
            exam,
            resolution,
        )
        return (
            closure_priority,
            gap_coverage,
            resolution_rank,
            diagnostic_coverage,
            -original_index,
        )

    def _gap_specific_exam_coverage(
        self,
        candidate: Any,
        gap: Dict[str, Any],
        exam: str,
        resolution: Dict[str, Any],
    ) -> float:
        preference = self._gap_specific_exam_preference(candidate, gap, exam, resolution)
        if self._pulmonary_avm_gap(candidate, gap):
            if preference >= 90:
                return round(preference / 100.0, 4)
            if preference >= 80:
                return 0.82
            if preference >= 40:
                return 0.45
            if preference >= 20:
                return 0.25
        if self._hematologic_malignancy_gap(candidate, gap):
            if preference >= 90:
                return round(preference / 100.0, 4)
            if preference >= 80:
                return 0.82
            if preference >= 60:
                return 0.62
            if preference >= 35:
                return 0.35
        return float(resolution.get("diagnostic_coverage") or 0.0)

    def _gap_specific_exam_preference(
        self,
        candidate: Any,
        gap: Dict[str, Any],
        exam: str,
        resolution: Dict[str, Any],
    ) -> int:
        requested = str(resolution.get("requested_exam") or exam or "")
        resolved = str(resolution.get("resolved_exam") or "")
        text = f"{requested} {resolved}"
        compact = _compact_text(text)
        if self._pulmonary_avm_gap(candidate, gap):
            if "cta" in compact or "\u80ba\u52a8\u8109ct" in compact or "\u80ba\u8840\u7ba1cta" in compact:
                return 100
            if "bubble" in compact or "\u53f3\u5fc3\u58f0\u5b66\u9020\u5f71" in compact:
                return 96
            if "\u589e\u5f3a" in compact or "cect" in compact:
                return 92
            if "\u8840\u7ba1\u9020\u5f71" in compact:
                return 86
            if "\u80f8\u90e8ct" in compact or "chestct" in compact:
                return 42
            if "ct" in compact:
                return 35
            if "abg" in compact or "\u8840\u6c14" in compact or "spo2" in compact:
                return 24
            return 20
        if self._hematologic_malignancy_gap(candidate, gap):
            if "\u9aa8\u9ad3\u7a7f\u523a" in compact or "\u9aa8\u9ad3\u6d3b\u68c0" in compact or "bmab" in compact:
                return 100
            if "\u9aa8\u9ad3\u6d41\u5f0f" in compact or "\u6d41\u5f0f\u7ec6\u80de" in compact or "\u514d\u75ab\u5206\u578b" in compact:
                return 96
            if "\u767d\u8840\u75c5\u878d\u5408\u57fa\u56e0" in compact or "\u878d\u5408\u57fa\u56e0" in compact:
                return 92
            if "\u7ec6\u80de\u9057\u4f20" in compact or "\u67d3\u8272\u4f53\u6838\u578b" in compact:
                return 88
            if "\u5916\u5468\u8840\u6d82\u7247" in compact:
                return 72
            if "\u7ec4\u7ec7\u75c5\u7406" in compact or "\u7a7f\u523a\u6d3b\u68c0" in compact:
                return 58
            if "\u5168\u8840\u7ec6\u80de\u8ba1\u6570" in compact or "\u8840\u5e38\u89c4" in compact or "cbc" in compact:
                return 38
            if "\u8840\u6c89" in compact or "esr" in compact or "\u809d\u529f\u80fd" in compact or "\u80be\u529f\u80fd" in compact:
                return 22
            return 18
        resolution_type = str(resolution.get("resolution_type") or "")
        if resolution_type in _FULL_CLOSURE_RESOLUTION_TYPES:
            return 80
        if resolution_type == PARTIAL_SUBSTITUTE:
            return 45
        return 10

    @staticmethod
    def _pulmonary_avm_gap(candidate: Any, gap: Dict[str, Any]) -> bool:
        entity_id = str(getattr(candidate, "entity_id", "") or "")
        name = str(getattr(candidate, "diagnosis", "") or "")
        canonical = str(getattr(candidate, "canonical_name", "") or "")
        target = str((gap or {}).get("target_evidence") or "")
        text = f"{entity_id} {name} {canonical} {target}".lower()
        return bool(
            entity_id == "D100055"
            or "pavm" in text
            or "\u80ba\u52a8\u9759\u8109\u7618" in text
            or "pulmonary_avm" in text
            or "pulmonary_vascular" in text
        )

    @staticmethod
    def _hematologic_malignancy_gap(candidate: Any, gap: Dict[str, Any]) -> bool:
        entity_id = str(getattr(candidate, "entity_id", "") or "")
        name = str(getattr(candidate, "diagnosis", "") or "")
        canonical = str(getattr(candidate, "canonical_name", "") or "")
        target = str((gap or {}).get("target_evidence") or "")
        text = f"{entity_id} {name} {canonical} {target}".lower()
        return bool(
            entity_id == "D000025"
            or "leukemia" in text
            or "blast" in text
            or "acute_leukemia" in text
            or "\u767d\u8840\u75c5" in text
        )

    def _apply_deferred_gap_decision_audit(
        self,
        decision: JudgeDecision,
        pool: Sequence[Any],
        tasks: Sequence[Dict[str, Any]],
    ) -> None:
        active_gaps = [
            dict(gap)
            for candidate in pool or []
            for gap in getattr(candidate, "gap_values", []) or []
            if isinstance(gap, dict)
        ]
        gaps = [
            dict(gap)
            for candidate in pool or []
            for gap in getattr(candidate, "evidence_gaps", []) or []
            if isinstance(gap, dict)
        ]
        overrides = []
        for candidate in pool or []:
            if not self._exam_priority_override_candidate(candidate):
                continue
            overrides.append(
                {
                    "candidate": self._name(candidate),
                    "entity_id": str(getattr(candidate, "entity_id", "") or ""),
                    "eligibility_status": self._eligibility_status(candidate),
                    "eligibility_substatus": str(
                        getattr(candidate, "eligibility_substatus", "") or ""
                    ),
                    "deferred_priority": float(
                        getattr(candidate, "deferred_priority", 0.0) or 0.0
                    ),
                    "deferred_priority_components": dict(
                        getattr(candidate, "deferred_priority_components", {}) or {}
                    ),
                    "max_gap_value": float(getattr(candidate, "max_gap_value", 0.0) or 0.0),
                    "actionable_gap_count": int(
                        getattr(candidate, "actionable_gap_count", 0) or 0
                    ),
                    "gap_values": list(getattr(candidate, "gap_values", []) or []),
                    "evidence_gaps": list(
                        getattr(candidate, "evidence_gaps", []) or []
                    ),
                    "override_reason": str(
                        getattr(candidate, "exam_priority_override_reason", "") or ""
                    ),
                    "priority_status": str(
                        getattr(candidate, "deferred_priority_status", "") or ""
                    ),
                }
            )
        closure_tasks = [
            dict(task)
            for task in tasks or []
            if isinstance(task, dict)
            and str(task.get("exam_source") or "") == "deferred_gap_closure_exam"
        ]
        override_gap_ids = {
            str(gap.get("gap_id") or "")
            for item in overrides
            for gap in item.get("evidence_gaps", []) or []
            if str(gap.get("gap_id") or "")
        }
        covered_gap_ids = {
            str(gap_id or "")
            for task in closure_tasks
            for gap_id in task.get("target_gaps", []) or []
            if str(gap_id or "")
        }
        ranked_active_gaps = sorted(
            active_gaps,
            key=lambda item: float(item.get("gap_value") or 0.0),
            reverse=True,
        )
        for index, gap in enumerate(ranked_active_gaps, start=1):
            gap["gap_value_rank"] = index
        decision.active_evidence_gaps = ranked_active_gaps
        decision.deferred_evidence_gaps = gaps
        decision.exam_priority_overrides = overrides
        decision.deferred_gap_closure_tasks = closure_tasks
        decision.deferred_gap_closure_exam_coverage = round(
            len(override_gap_ids & covered_gap_ids) / max(1, len(override_gap_ids)),
            4,
        ) if override_gap_ids else 0.0
        decision.exam_priority_alignment = decision.deferred_gap_closure_exam_coverage
        decision.wrong_primary_exam_drift = 0.0

    def _apply_bridge_decision_audit(
        self,
        decision: JudgeDecision,
        ranked_all: Sequence[Any],
        differential_pool: Sequence[Any],
        pairwise: Sequence[Dict[str, Any]],
        discriminating_exam_tasks: Sequence[Dict[str, Any]],
    ) -> None:
        protected = [
            item
            for item in ranked_all or []
            if has_active_bridge_protection(item, CROSS_SYSTEM_SCOPE)
        ]
        if not protected:
            return
        pool_names = {self._name(item) for item in differential_pool or []}
        final_names = set(getattr(decision, "final_diagnoses", []) or [])
        gap_targets = set(getattr(decision, "evidence_gap_targets", []) or [])
        blocked_reasons = {
            str(item.get("diagnosis") or ""): str(item.get("reason") or "")
            for item in getattr(decision, "blocked_diagnoses", []) or []
            if isinstance(item, dict)
        }
        decision.bridge_protected_candidates = list(
            dict.fromkeys(
                list(getattr(decision, "bridge_protected_candidates", []) or [])
                + [self._name(item) for item in protected if self._name(item)]
            )
        )
        decision.bridge_pairwise_comparisons = [
            dict(item)
            for item in pairwise or []
            if str(item.get("left") or "") in decision.bridge_protected_candidates
            or str(item.get("right") or "") in decision.bridge_protected_candidates
        ]
        dispositions: List[Dict[str, Any]] = []
        for candidate in protected:
            name = self._name(candidate)
            if not name:
                continue
            if name in final_names:
                disposition = "SelectedPrimary" if name == decision.primary else "SelectedSecondary"
            elif name in gap_targets or self._eligibility_status(candidate) == DEFERRED:
                disposition = "DeferredNeedsConfirmatoryEvidence"
            elif name in pool_names:
                disposition = "RejectedAfterComparison"
            else:
                disposition = "NotInJudgePool"
            dispositions.append(
                {
                    "candidate": name,
                    "entity_id": str(getattr(candidate, "entity_id", "") or ""),
                    "entered_by": "bridge_protection",
                    "eligibility_status": self._eligibility_status(candidate),
                    "pool_filter_reason": decision.pool_filter_reasons.get(name, ""),
                    "decision": disposition,
                    "decision_reason": blocked_reasons.get(name, disposition),
                    "bridge_protection_decisions": list(
                        getattr(candidate, "bridge_protection_decisions", []) or []
                    ),
                }
            )
        decision.bridge_candidate_final_dispositions = dispositions
        bridge_names = set(decision.bridge_protected_candidates)
        decision.bridge_generated_gaps = [
            dict(task)
            for task in discriminating_exam_tasks or []
            if bridge_names & set(task.get("target_candidates") or [])
        ]

    def _merge_discriminating_exam_tasks(
        self,
        conflict_tasks: Sequence[Dict[str, Any]],
        base_tasks: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        by_exam: Dict[str, Dict[str, Any]] = {}

        def add(task: Dict[str, Any]) -> None:
            exam = str((task or {}).get("exam") or "").strip()
            if not exam:
                return
            current = by_exam.get(exam)
            if current is None:
                current = dict(task)
                current["target_candidates"] = list(
                    dict.fromkeys(current.get("target_candidates") or [])
                )
                current["target_findings"] = list(
                    dict.fromkeys(current.get("target_findings") or [])
                )
                current["source"] = list(dict.fromkeys(current.get("source") or []))
                by_exam[exam] = current
                merged.append(current)
                return
            current["target_candidates"] = list(
                dict.fromkeys(
                    list(current.get("target_candidates") or [])
                    + list(task.get("target_candidates") or [])
                )
            )
            current["target_findings"] = list(
                dict.fromkeys(
                    list(current.get("target_findings") or [])
                    + list(task.get("target_findings") or [])
                )
            )[:12]
            current["source"] = list(
                dict.fromkeys(
                    list(current.get("source") or []) + list(task.get("source") or [])
                )
            )
            current["target_candidate_count"] = len(current["target_candidates"])
            current["information_gain_hint"] = max(
                float(current.get("information_gain_hint") or 0.0),
                float(task.get("information_gain_hint") or 0.0),
            )
            if (
                task.get("exam_source") == "pairwise_discrimination_exam"
                and current.get("exam_source") not in {
                    "deferred_gap_closure_exam",
                    "conflict_adjudication_exam",
                }
            ):
                current["exam_source"] = "pairwise_discrimination_exam"
                current["exam_type"] = "pairwise_discrimination"
                current["expected_effect"] = "resolve_primary_arbitration_pairwise_gap"
                for key in (
                    "priority_bucket",
                    "source_gap_id",
                    "target_pair",
                    "target_question",
                    "target_claim",
                    "exam_role",
                    "expected_arbitration_effect",
                ):
                    if task.get(key) not in (None, "", [], {}):
                        current[key] = (
                            dict(task.get(key))
                            if isinstance(task.get(key), dict)
                            else task.get(key)
                        )
            elif (
                task.get("exam_source") == "conflict_adjudication_exam"
                and current.get("exam_source") != "deferred_gap_closure_exam"
            ):
                current["exam_source"] = "conflict_adjudication_exam"
                current["exam_type"] = "conflict_adjudication"
                current["expected_effect"] = (
                    "resolve_reasoning_structured_polarity_conflict"
                )
            elif (
                task.get("exam_source") == "deferred_gap_closure_exam"
                and not current.get("urgent_safety")
            ):
                current["exam_source"] = "deferred_gap_closure_exam"
                current["exam_type"] = "deferred_gap_closure"
                current["expected_effect"] = "close_high_value_deferred_evidence_gap"
                current["priority_override"] = True
                current["override_reason"] = str(
                    task.get("override_reason")
                    or current.get("override_reason")
                    or ""
                )
                current["target_gaps"] = list(
                    dict.fromkeys(
                        list(current.get("target_gaps") or [])
                        + list(task.get("target_gaps") or [])
                    )
                )
                if task.get("evidence_gap") and not current.get("evidence_gap"):
                    current["evidence_gap"] = dict(task.get("evidence_gap") or {})
                for key in (
                    "priority_bucket",
                    "requested_exam",
                    "resolved_exam",
                    "resolution_type",
                    "diagnostic_coverage",
                    "gap_diagnostic_coverage",
                    "exam_resolution",
                    "target_gap",
                    "expected_transition",
                ):
                    if task.get(key) not in (None, "", [], {}):
                        current[key] = (
                            dict(task.get(key))
                            if isinstance(task.get(key), dict)
                            else task.get(key)
                        )
                current["closure_priority"] = max(
                    int(current.get("closure_priority") or 0),
                    int(task.get("closure_priority") or 0),
                )
                incoming_rank = int(task.get("closure_rank") or 9999)
                current_rank = int(current.get("closure_rank") or 9999)
                current["closure_rank"] = min(current_rank, incoming_rank)
            elif (
                task.get("exam_source") == "evidence_claim_followup_exam"
                and current.get("exam_source") not in {
                    "conflict_adjudication_exam",
                    "deferred_gap_closure_exam",
                }
            ):
                current["exam_source"] = "evidence_claim_followup_exam"
                current["exam_type"] = "evidence_claim_verification"
                current["expected_effect"] = "verify_or_reject_reasoner_evidence_claim"
            elif (
                task.get("exam_source") == "pattern_anchor_workup_exam"
                and current.get("exam_source") not in {
                    "conflict_adjudication_exam",
                    "deferred_gap_closure_exam",
                    "evidence_claim_followup_exam",
                }
            ):
                current["exam_source"] = "pattern_anchor_workup_exam"
                current["exam_type"] = "pattern_anchor_workup"
                current["expected_effect"] = "close_missing_diagnostic_pattern_anchor"

        for task in conflict_tasks or []:
            add(task)
        for task in base_tasks or []:
            add(task)
        merged.sort(key=self._merged_exam_task_priority_key, reverse=True)
        return merged[: self.discriminating_exam_max_items]

    @staticmethod
    def _merged_exam_task_priority_key(task: Dict[str, Any]) -> tuple:
        source = str((task or {}).get("exam_source") or "")
        exam_type = str((task or {}).get("exam_type") or "")
        if bool((task or {}).get("urgent_safety")):
            bucket = 7
        elif source == "deferred_gap_closure_exam" and bool(
            (task or {}).get("priority_override")
        ):
            bucket = 6
        elif source == "deferred_gap_closure_exam":
            bucket = 5
        elif source in {
            "evidence_claim_followup_exam",
            "pattern_anchor_workup_exam",
        }:
            bucket = 4
        elif source == "conflict_adjudication_exam":
            bucket = 3
        else:
            bucket = DiagnosisJudge._exam_type_priority(exam_type)
        closure_priority = int((task or {}).get("closure_priority") or 0)
        closure_rank = int((task or {}).get("closure_rank") or 9999)
        source_gap_value = float((task or {}).get("source_gap_value") or 0.0)
        exam_gap_closure_value = float(
            (task or {}).get("exam_gap_closure_value") or 0.0
        )
        return (
            bucket,
            source_gap_value,
            exam_gap_closure_value,
            closure_priority,
            -closure_rank,
            float((task or {}).get("information_gain_hint") or 0.0),
            int((task or {}).get("target_candidate_count") or 0),
        )

    def _exam_task_type(
        self,
        exam: str,
        target_names: Sequence[str],
        pool_size: int,
    ) -> str:
        if self._generic_inflammation_exam(exam):
            return "generic_inflammation"
        if self._special_discriminator_exam(exam):
            return "special_discriminator"
        if len(set(target_names)) >= min(2, max(1, pool_size)):
            return "shared_discriminator"
        return "confirmatory"

    @staticmethod
    def _generic_inflammation_exam(exam: str) -> bool:
        text = str(exam or "")
        return any(marker in text for marker in _GENERIC_INFLAMMATION_EXAM_MARKERS)

    @staticmethod
    def _special_discriminator_exam(exam: str) -> bool:
        text = str(exam or "")
        return any(marker in text for marker in _SPECIAL_DISCRIMINATOR_EXAM_MARKERS)

    @staticmethod
    def _exam_type_priority(exam_type: str) -> int:
        return {
            "conflict_adjudication": 6,
            "deferred_gap_closure": 5,
            "evidence_claim_verification": 5,
            "pattern_anchor_workup": 5,
            "special_discriminator": 4,
            "shared_discriminator": 3,
            "confirmatory": 2,
            "generic_inflammation": 1,
        }.get(str(exam_type or ""), 0)

    def _exam_task_score(
        self,
        exam: str,
        target_names: Sequence[str],
        pool_size: int,
    ) -> float:
        target_count = len(set(target_names))
        coverage = target_count / max(1, pool_size)
        multi_candidate = 1.0 if target_count >= 2 else 0.0
        if 0 < target_count < pool_size:
            separation = 1.0
        elif target_count >= pool_size and pool_size > 1:
            separation = 0.95
        else:
            separation = 0.25
        exam_type = self._exam_task_type(exam, target_names, pool_size)
        type_score = self._exam_type_priority(exam_type) / 4.0
        score = 0.30 * separation + 0.25 * multi_candidate + 0.25 * type_score + 0.20 * coverage
        if exam_type == "generic_inflammation" and target_count < 2:
            score *= 0.55
        return score

    @staticmethod
    def _exam_expected_effect(exam_type: str, target_names: Sequence[str]) -> str:
        if exam_type in {"special_discriminator", "shared_discriminator"}:
            return "shift_probabilities_across_differential_pool"
        if exam_type == "confirmatory":
            target = ", ".join(target_names[:2])
            return f"close_confirmatory_gap:{target}" if target else "close_confirmatory_gap"
        return "low_priority_generic_context"

    def _high_prior_specific_exam_candidate(self, candidate: Any) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        status = self._eligibility_status(candidate)
        if status and status not in {PRIMARY_ELIGIBLE, DEFERRED}:
            return False
        evidence_specificity = float(
            getattr(candidate, "evidence_specificity_score", 0.0) or 0.0
        )
        if (
            evidence_specificity < 0.65
            and not getattr(candidate, "required_gaps", None)
            and not self._priority(candidate)
        ):
            return False
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        source_prior = float(getattr(candidate, "source_prior", 0.0) or 0.0)
        if source_prior < 0.55 and f"diagnosis:{candidate.diagnosis}" not in matched:
            return False
        return bool(
            getattr(candidate, "required_gaps", None)
            or self._priority(candidate)
            or f"diagnosis:{candidate.diagnosis}" in matched
        )

    def _specific_exam_candidate_key(self, candidate: Any) -> tuple:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        return (
            1 if f"diagnosis:{candidate.diagnosis}" in matched else 0,
            float(getattr(candidate, "source_prior", 0.0) or 0.0),
            1 if getattr(candidate, "required_gaps", None) else 0,
            float(getattr(candidate, "coverage_score", 0.0) or 0.0),
            float(getattr(candidate, "evidence_specificity_score", 0.0) or 0.0),
        )

    def _candidate_exam_union(self, candidates: Sequence[Any]) -> List[str]:
        exams: List[str] = []
        for candidate in candidates:
            for exam in self._candidate_discriminating_exams(candidate):
                if exam and exam not in exams:
                    exams.append(exam)
        return exams

    def _candidate_discriminating_exams(self, candidate: Any) -> List[str]:
        if not candidate:
            return []
        name = str(getattr(candidate, "diagnosis", "") or "")
        exams: List[str] = []
        entry: Dict[str, Any] = self._knowledge_entry(name)
        for field_name in (
            "discriminating_exams",
            "strong_verification_exams",
            "required_exams",
        ):
            for exam in entry.get(field_name, []) or []:
                text = str(exam).strip()
                if text and text not in exams:
                    exams.append(text)
        if self.knowledge and hasattr(self.knowledge, "get_discriminating_exam_bundle"):
            lookups = [
                str(getattr(candidate, "entity_id", "") or ""),
                name,
                str(getattr(candidate, "canonical_name", "") or ""),
                str(getattr(candidate, "submission_name", "") or ""),
            ]
            for lookup in list(dict.fromkeys(item for item in lookups if item)):
                before_count = len(exams)
                for exam in self.knowledge.get_discriminating_exam_bundle(lookup) or []:
                    text = str(exam).strip()
                    if text and text not in exams:
                        exams.append(text)
                if len(exams) > before_count:
                    break
        for exam in _DIFFERENTIAL_EXAM_HINTS.get(name, []):
            if exam and exam not in exams:
                exams.append(exam)
        for exam in getattr(candidate, "claim_followup_exams", []) or []:
            text = str(exam).strip()
            if text and text not in exams:
                exams.append(text)
        return exams[:6]

    def _knowledge_entry(self, diagnosis: Any) -> Dict[str, Any]:
        if not self.knowledge:
            return {}
        name = str(diagnosis or "")
        if hasattr(self.knowledge, "get_disease_profile"):
            return self.knowledge.get_disease_profile(name) or {}
        if hasattr(self.knowledge, "get"):
            return self.knowledge.get(name) or {}
        return {}

    def _choose_primary(self, ranked: Sequence[Any]) -> Optional[Any]:
        if not ranked:
            return None
        ranked = sorted(ranked, key=self._sort_key, reverse=True)
        best = ranked[0]
        for item in ranked[1:]:
            if self._should_prefer_explanatory_primary(item, best):
                best = item
        gap_candidates = [
            item
            for item in ranked
            if self._gap_authorizable(item)
            and self._should_prefer_explanatory_primary(item, best, allow_tie=True)
        ]
        if gap_candidates:
            best = max(gap_candidates, key=self._sort_key)

        best = self._maybe_prefer_systemic_primary(best, ranked)
        best = self._maybe_prefer_parent_primary(best, ranked[: self.top_k])
        return best

    def _should_prefer_explanatory_primary(
        self,
        candidate: Any,
        selected: Any,
        allow_tie: bool = False,
    ) -> bool:
        if not candidate or not selected:
            return False
        if getattr(candidate, "hard_contradiction", False):
            return False
        if not (self._trusted(candidate) or self._gap_authorizable(candidate)):
            return False
        candidate_score = self._judge_score(candidate)
        selected_score = self._judge_score(selected)
        margin = 0.0 if allow_tie else self.explanatory_preference_margin
        if candidate_score >= selected_score + margin:
            return True
        candidate_core = self._core_coverage(candidate)
        selected_core = self._core_coverage(selected)
        candidate_residual = self._residual(candidate)
        selected_residual = self._residual(selected)
        if (
            self._priority(candidate)
            and not self._priority(selected)
            and candidate_core + 0.10 >= selected_core
            and candidate_residual <= selected_residual + 0.10
        ):
            return True
        if (
            candidate_core >= selected_core + 0.18
            and candidate_residual <= selected_residual + 0.12
            and candidate_score >= selected_score - self.pairwise_close_margin
        ):
            return True
        if (
            self._residual_core_count(selected) > self._residual_core_count(candidate)
            and candidate_score >= selected_score - self.pairwise_close_margin
        ):
            return True
        return False

    def _maybe_prefer_systemic_primary(self, selected: Any, ranked: Sequence[Any]) -> Any:
        if self._systemic_primary(selected):
            return selected
        for candidate in ranked:
            if candidate is selected:
                continue
            if not self._systemic_primary(candidate):
                continue
            if not (self._trusted(candidate) or self._gap_authorizable(candidate)):
                continue
            if self._judge_score(candidate) + 0.06 < self._judge_score(selected):
                continue
            if self._is_manifestation(selected) or self._generic_parent_of(candidate, selected):
                return candidate
            if self._coverage(candidate) >= self._coverage(selected) and self._residual(candidate) <= self._residual(selected) + 0.12:
                return candidate
        return selected

    def _maybe_prefer_parent_primary(self, selected: Any, ranked: Sequence[Any]) -> Any:
        parent = str(getattr(selected, "parent_diagnosis", "") or "")
        if parent not in _PARENT_FALLBACK_NAMES:
            direct_parent = self._direct_parent_fallback_candidate(selected, ranked)
            return direct_parent or selected
        parent_candidate = next((item for item in ranked if item.diagnosis == parent), None)
        if not parent_candidate or getattr(parent_candidate, "hard_contradiction", False):
            direct_parent = self._direct_parent_fallback_candidate(selected, ranked)
            return direct_parent or selected
        selected_matched = set(getattr(selected, "matched_evidence", []) or [])
        parent_matched = set(getattr(parent_candidate, "matched_evidence", []) or [])
        if f"diagnosis:{selected.diagnosis}" in selected_matched:
            return selected
        if (
            f"diagnosis:{parent}" in parent_matched
            and (
                not getattr(selected, "required_met", False)
                or not self._objective_signal(selected)
            )
        ):
            return parent_candidate
        if f"diagnosis:{parent}" in parent_matched:
            if self._judge_score(parent_candidate) + self.parent_fallback_margin >= self._judge_score(selected):
                return parent_candidate
        if (
            self._trusted(parent_candidate)
            and self._judge_score(parent_candidate) + self.parent_fallback_margin >= self._judge_score(selected)
            and not self._trusted(selected)
        ):
            return parent_candidate
        return selected

    def _direct_parent_fallback_candidate(self, selected: Any, ranked: Sequence[Any]) -> Optional[Any]:
        if not (
            selected
            and self._eligibility_status(selected) == DEFERRED
            and not f"diagnosis:{selected.diagnosis}"
            in set(getattr(selected, "matched_evidence", []) or [])
        ):
            return None
        for candidate in ranked:
            if candidate.diagnosis not in _PARENT_FALLBACK_NAMES:
                continue
            if getattr(candidate, "hard_contradiction", False):
                continue
            if f"diagnosis:{candidate.diagnosis}" not in set(getattr(candidate, "matched_evidence", []) or []):
                continue
            if self._trusted(candidate):
                return candidate
            if not (self._same_family(candidate, selected) or self._same_body_system(candidate, selected)):
                continue
            if self._trusted(candidate) or self._judge_score(candidate) + self.parent_fallback_margin >= self._judge_score(selected):
                return candidate
        return None

    def _select_secondary(
        self,
        primary: Any,
        ranked: Sequence[Any],
        max_final_diagnoses: int,
    ) -> List[Any]:
        result: List[Any] = []
        limit = max(0, int(max_final_diagnoses or 1) - 1)
        if limit <= 0:
            return result
        secondary_pool = sorted(
            [item for item in ranked if item and item.diagnosis != primary.diagnosis],
            key=lambda item: (
                1
                if (
                    self._secondary_causally_related(primary, item)
                    and self._is_manifestation(item)
                    and self._independent_objective(item)
                )
                else 0,
                1
                if (
                    self._is_manifestation(item)
                    and f"diagnosis:{item.diagnosis}"
                    in set(getattr(primary, "matched_evidence", []) or [])
                )
                else 0,
                1 if self._same_family(primary, item) and self._independent_objective(item) else 0,
                self._judge_score(item),
            ),
            reverse=True,
        )
        for candidate in secondary_pool:
            if candidate.diagnosis == primary.diagnosis:
                continue
            if len(result) >= limit:
                break
            if getattr(candidate, "hard_contradiction", False):
                continue
            if not self._trusted(candidate):
                continue
            if (
                self._is_manifestation(candidate)
                and f"diagnosis:{candidate.diagnosis}"
                in set(getattr(primary, "matched_evidence", []) or [])
            ):
                result.append(candidate)
                continue
            if self._generic_parent_of(primary, candidate) or self._generic_parent_of(candidate, primary):
                if self._same_family(primary, candidate) and self._independent_objective(candidate):
                    result.append(candidate)
                continue
            if self._secondary_causally_related(primary, candidate):
                if self._is_manifestation(candidate):
                    if self._independent_objective(candidate):
                        result.append(candidate)
                    continue
                if self._independent_objective(candidate):
                    result.append(candidate)
                continue
            if float(getattr(candidate, "score", 0.0) or 0.0) < self.secondary_min_score:
                continue
            if self._same_family(primary, candidate) and self._independent_objective(candidate):
                result.append(candidate)
        return result

    def _secondary_causally_related(self, primary: Any, candidate: Any) -> bool:
        if self._causally_related(primary, candidate):
            return True
        primary_name = str(getattr(primary, "diagnosis", "") or "")
        candidate_name = str(getattr(candidate, "diagnosis", "") or "")
        relation = str(getattr(candidate, "causal_relation_to_selected", "") or "")
        if primary_name and relation in {
            f"caused_by:{primary_name}",
            f"complication_of:{primary_name}",
            f"downstream_of:{primary_name}",
        }:
            return True
        primary_relation = str(getattr(primary, "causal_relation_to_selected", "") or "")
        if candidate_name and primary_relation in {
            f"causes:{candidate_name}",
            f"explains:{candidate_name}",
        }:
            return True
        return False

    def _evidence_gap_targets(
        self,
        primary: Any,
        ranked: Sequence[Any],
        final: Sequence[str],
    ) -> List[str]:
        targets: List[str] = []
        final_set = set(final or [])
        if (
            getattr(primary, "required_gaps", None)
            or getattr(primary, "evidence_gaps", None)
            or self._critical_claim_followup_candidate(primary)
        ):
            targets.append(primary.diagnosis)
        for candidate in ranked:
            if len(targets) >= self.gap_target_limit:
                break
            if candidate.diagnosis in final_set or candidate.diagnosis in targets:
                continue
            if self._eligibility_status(candidate) != DEFERRED:
                continue
            if not (
                getattr(candidate, "required_gaps", None)
                or getattr(candidate, "evidence_gaps", None)
                or self._critical_claim_followup_candidate(candidate)
            ):
                continue
            if (
                self._deferred_explains_better_than_primary(primary, candidate)
                or self._exam_priority_override_candidate(candidate)
            ):
                targets.append(candidate.diagnosis)
        for candidate in ranked:
            if len(targets) >= self.gap_target_limit:
                break
            if candidate.diagnosis in final_set or candidate.diagnosis in targets:
                continue
            if self._eligibility_status(candidate) != DEFERRED:
                continue
            if not (
                getattr(candidate, "required_gaps", None)
                or getattr(candidate, "evidence_gaps", None)
                or self._critical_claim_followup_candidate(candidate)
            ):
                continue
            if (
                self._same_family(primary, candidate)
                or self._causally_related(primary, candidate)
                or self._high_value_unresolved_contender(primary, candidate)
                or self._exam_priority_override_candidate(candidate)
            ):
                targets.append(candidate.diagnosis)
        for candidate in ranked:
            if len(targets) >= self.gap_target_limit:
                break
            if candidate.diagnosis in final_set or candidate.diagnosis in targets:
                continue
            if not self._gap_authorizable(candidate):
                continue
            if self._same_family(primary, candidate) or self._causally_related(primary, candidate):
                targets.append(candidate.diagnosis)
        pattern_targets: List[str] = []
        for candidate in sorted(
            ranked,
            key=self._pattern_deferred_workup_sort_key,
            reverse=True,
        ):
            if candidate.diagnosis in final_set or candidate.diagnosis in targets:
                continue
            if self._pattern_deferred_workup_candidate(candidate):
                pattern_targets.append(candidate.diagnosis)
        for name in pattern_targets:
            if len(targets) < self.gap_target_limit or not any(
                target in pattern_targets for target in targets
            ):
                targets.append(name)
            if len(targets) >= self.gap_target_limit and any(
                target in pattern_targets for target in targets
            ):
                break
        return targets

    def _deferred_explains_better_than_primary(self, primary: Any, candidate: Any) -> bool:
        if not primary or not candidate:
            return False
        if getattr(candidate, "hard_contradiction", False):
            return False
        if not self._has_signal(candidate):
            return False
        if self._coverage(candidate) >= self._coverage(primary) + 0.08:
            return True
        if self._residual(candidate) <= self._residual(primary) - 0.10:
            return True
        if self._core_coverage(candidate) >= self._core_coverage(primary) + 0.10:
            return True
        if self._residual_core_count(candidate) < self._residual_core_count(primary):
            return True
        return False

    def _deferred_evidence_gap_targets(
        self,
        primary: Any,
        pool: Sequence[Any],
    ) -> List[str]:
        targets: List[str] = []
        ordered_pool: List[Any] = []
        seen_ids: set[int] = set()
        for item in [primary] + list(pool or []):
            if not item:
                continue
            marker = id(item)
            if marker in seen_ids:
                continue
            seen_ids.add(marker)
            ordered_pool.append(item)
        for candidate in ordered_pool:
            if not candidate or candidate.diagnosis in targets:
                continue
            if getattr(candidate, "hard_contradiction", False):
                continue
            if not (
                getattr(candidate, "required_gaps", None)
                or getattr(candidate, "evidence_gaps", None)
                or self._candidate_discriminating_exams(candidate)
                or self._critical_claim_followup_candidate(candidate)
            ):
                continue
            if candidate.diagnosis == getattr(primary, "diagnosis", ""):
                targets.append(candidate.diagnosis)
                break
        for candidate in ordered_pool:
            if len(targets) >= self.gap_target_limit:
                break
            if not candidate or candidate.diagnosis in targets:
                continue
            if self._eligibility_status(candidate) != DEFERRED:
                continue
            if not (
                getattr(candidate, "required_gaps", None)
                or getattr(candidate, "evidence_gaps", None)
                or self._critical_claim_followup_candidate(candidate)
            ):
                continue
            if (
                self._deferred_explains_better_than_primary(primary, candidate)
                or self._high_value_unresolved_contender(primary, candidate)
                or self._exam_priority_override_candidate(candidate)
            ):
                targets.append(candidate.diagnosis)
        pattern_targets: List[str] = []
        for candidate in sorted(
            ordered_pool,
            key=self._pattern_deferred_workup_sort_key,
            reverse=True,
        ):
            if not candidate or candidate.diagnosis in targets:
                continue
            if self._pattern_deferred_workup_candidate(candidate):
                pattern_targets.append(candidate.diagnosis)
        for name in pattern_targets:
            if len(targets) < self.gap_target_limit or not any(
                target in pattern_targets for target in targets
            ):
                targets.append(name)
            if len(targets) >= self.gap_target_limit and any(
                target in pattern_targets for target in targets
            ):
                break
        for candidate in ordered_pool:
            if len(targets) >= self.gap_target_limit:
                break
            if not candidate or candidate.diagnosis in targets:
                continue
            if getattr(candidate, "hard_contradiction", False):
                continue
            if self._eligibility_status(candidate) == DEFERRED and (
                getattr(candidate, "required_gaps", None)
                or getattr(candidate, "evidence_gaps", None)
                or self._candidate_discriminating_exams(candidate)
                or self._critical_claim_followup_candidate(candidate)
            ):
                targets.append(candidate.diagnosis)
                continue
        for candidate in ordered_pool:
            if self._high_value_unresolved_contender(primary, candidate) or self._exam_priority_override_candidate(candidate):
                targets.append(candidate.diagnosis)
            if len(targets) >= self.gap_target_limit:
                break
        return targets[: max(1, self.gap_target_limit)]

    def _gap_state_by_candidate(self, ranked: Sequence[Any]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for candidate in ranked[: self.max_reviews]:
            name = self._name(candidate)
            if name:
                result[name] = self._required_gap_state(candidate)
        return result

    @staticmethod
    def _gap_state_distribution(states: Dict[str, str]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for state in states.values():
            result[state] = result.get(state, 0) + 1
        return result

    def _apply_eligibility_audit(
        self,
        decision: JudgeDecision,
        ranked: Sequence[Any],
    ) -> None:
        distribution: Dict[str, int] = {}
        substatus_distribution: Dict[str, int] = {}
        primary: List[str] = []
        deferred: List[str] = []
        excluded: List[str] = []
        for candidate in ranked or []:
            status = self._eligibility_status(candidate)
            if not status:
                continue
            distribution[status] = distribution.get(status, 0) + 1
            if status == PRIMARY_ELIGIBLE:
                primary.append(candidate.diagnosis)
            elif status == DEFERRED:
                deferred.append(candidate.diagnosis)
                substatus = str(getattr(candidate, "eligibility_substatus", "") or "")
                if substatus:
                    substatus_distribution[substatus] = substatus_distribution.get(substatus, 0) + 1
            elif status == EXCLUDED:
                excluded.append(candidate.diagnosis)
        decision.eligibility_distribution = distribution
        decision.deferred_substatus_distribution = substatus_distribution
        decision.primary_eligible_candidates = primary
        decision.deferred_anchor_candidates = deferred
        decision.excluded_candidates = excluded

    @staticmethod
    def _eligibility_status(candidate: Any) -> str:
        return str(getattr(candidate, "eligibility_status", "") or "")

    def _best_deferred_candidate(self, candidates: Sequence[Any]) -> Optional[Any]:
        deferred = [
            item for item in candidates or [] if self._eligibility_status(item) == DEFERRED
        ]
        if not deferred:
            return None
        return sorted(deferred, key=self._sort_key, reverse=True)[0]

    def _required_gap_state(self, candidate: Any) -> str:
        if not candidate:
            return "unsupported_gap"
        status = self._eligibility_status(candidate)
        if status == PRIMARY_ELIGIBLE:
            return "satisfied"
        if status == DEFERRED:
            return (
                "actionable_gap"
                if getattr(candidate, "required_gaps", None)
                or getattr(candidate, "evidence_gaps", None)
                or self._critical_claim_followup_candidate(candidate)
                else "partially_satisfied"
            )
        if status == DIFFERENTIAL_ONLY:
            return "unsupported_gap"
        if status == EXCLUDED:
            return "hard_contradiction"
        if getattr(candidate, "hard_contradiction", False):
            return "hard_contradiction"
        existing = str(getattr(candidate, "required_gap_state", "") or "")
        if existing in {
            "satisfied",
            "partially_satisfied",
            "actionable_gap",
            "nonblocking_gap",
            "unsupported_gap",
            "hard_contradiction",
        }:
            return existing
        if getattr(candidate, "required_met", False) and not getattr(
            candidate, "required_gaps", None
        ):
            return "satisfied"
        if not getattr(candidate, "matched_evidence", None):
            return "unsupported_gap"
        if getattr(candidate, "required_gaps", None):
            if self._candidate_discriminating_exams(candidate):
                return "actionable_gap"
            if self._explanatory_gap_authorizable(candidate):
                return "nonblocking_gap"
            return "partially_satisfied"
        if self._explanatory_gap_authorizable(candidate):
            return "nonblocking_gap"
        return "partially_satisfied"

    def _explanatory_gap_authorizable(self, candidate: Any) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        if getattr(candidate, "required_met", False):
            return False
        if not getattr(candidate, "matched_evidence", None):
            return False
        if not (
            self._priority(candidate)
            or self._systemic_primary(candidate)
        ):
            return False
        if not self._core_support_signal(candidate):
            return False
        explanatory_score = (
            0.45 * self._coverage(candidate)
            + 0.40 * self._core_coverage(candidate)
            + 0.15 * (1.0 - self._residual(candidate))
            + 0.10 * float(getattr(candidate, "source_prior", 0.0) or 0.0)
        )
        if explanatory_score >= self.gap_authorization_min_explanatory_score:
            return True
        if self._core_coverage(candidate) >= self.gap_authorization_min_core_coverage:
            return self._residual_core_count(candidate) <= self.gap_authorization_max_core_residual
        if (
            self._coverage(candidate) >= max(0.52, self.gap_authorization_min_coverage)
            and self._residual(candidate) <= self.gap_authorization_max_residual
        ):
            return True
        return False

    @staticmethod
    def _core_support_signal(candidate: Any) -> bool:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        if f"diagnosis:{getattr(candidate, 'diagnosis', '')}" in matched:
            return True
        core = matched & _CORE_EVIDENCE_TOKENS
        if core:
            return True
        if matched & _OBJECTIVE_GAP_FINDINGS:
            return True
        explained = set(getattr(candidate, "explained_evidence", []) or [])
        if explained & _CORE_EVIDENCE_TOKENS:
            return True
        return False

    def _model_score_primary(self, ranked: Sequence[Any]) -> str:
        pool = [item for item in ranked or [] if item and not getattr(item, "hard_contradiction", False)]
        if not pool:
            return ""
        return self._name(
            max(pool, key=lambda item: float(getattr(item, "score", 0.0) or 0.0))
        )

    def _primary_unlock_reason(
        self,
        previous_primary: str,
        primary: Any,
        ranked: Sequence[Any],
    ) -> str:
        if not previous_primary or not primary:
            return ""
        previous = next(
            (item for item in ranked if self._name(item) == previous_primary),
            None,
        )
        if not previous:
            return "previous primary no longer appears in candidate table"
        if getattr(previous, "hard_contradiction", False):
            return "previous primary gained hard contradiction"
        if self._residual_core_count(primary) < self._residual_core_count(previous):
            return "new primary leaves fewer unexplained core findings"
        if self._core_coverage(primary) > self._core_coverage(previous) + 0.10:
            return "new primary explains more core findings"
        if self._judge_score(primary) > self._judge_score(previous):
            return "new primary has higher evidence-authority judge score"
        return "primary changed after evidence-authority rerank"

    def _blocked_records(self, ranked: Sequence[Any], final: Sequence[str]) -> List[Dict[str, Any]]:
        final_set = set(final or [])
        blocked: List[Dict[str, Any]] = []
        for candidate in ranked[: self.max_reviews]:
            if candidate.diagnosis in final_set:
                continue
            reason = "differential_only"
            gap_state = self._required_gap_state(candidate)
            status = self._eligibility_status(candidate)
            eligibility_reason = str(getattr(candidate, "eligibility_reason", "") or "")
            if status:
                if status == DEFERRED:
                    reason = f"Deferred:{eligibility_reason or 'NeedsAnchor'}"
                elif status == DIFFERENTIAL_ONLY:
                    reason = f"DifferentialOnly:{eligibility_reason or 'not primary eligible'}"
                elif status == EXCLUDED:
                    reason = f"Excluded:{eligibility_reason or 'eligibility gate'}"
            if getattr(candidate, "hard_contradiction", False):
                reason = "hard_contradiction"
            elif status:
                pass
            elif not getattr(candidate, "matched_evidence", None):
                reason = "no_supporting_evidence"
            elif not getattr(candidate, "required_met", False):
                reason = gap_state
            elif self._is_manifestation(candidate):
                reason = "manifestation_or_complication_not_primary"
            elif self._generic_parent_name(candidate):
                reason = "generic_or_parent_diagnosis"
            blocked.append(
                {
                    "diagnosis": candidate.diagnosis,
                    "entity_id": str(getattr(candidate, "entity_id", "") or ""),
                    "canonical_name": str(getattr(candidate, "canonical_name", "") or candidate.diagnosis),
                    "submission_name": str(getattr(candidate, "submission_name", "") or candidate.diagnosis),
                    "submittable": bool(getattr(candidate, "submittable", True)),
                    "reason": reason,
                    "score": getattr(candidate, "score", 0.0),
                    "judge_score": round(self._judge_score(candidate), 4),
                    "explanatory_coverage": round(self._coverage(candidate), 4),
                    "core_explanatory_coverage": round(
                        self._core_coverage(candidate),
                        4,
                    ),
                    "residual_evidence_score": round(self._residual(candidate), 4),
                    "residual_core_evidence_count": self._residual_core_count(candidate),
                    "required_met": bool(getattr(candidate, "required_met", False)),
                    "required_gap_state": gap_state,
                    "required_gaps": list(getattr(candidate, "required_gaps", []) or [])[:4],
                    "eligibility_status": status,
                    "eligibility_reason": eligibility_reason,
                    "eligibility_substatus": str(
                        getattr(candidate, "eligibility_substatus", "") or ""
                    ),
                    "missing_required_anchors": list(
                        getattr(candidate, "missing_required_anchors", []) or []
                    )[:6],
                    "satisfied_required_anchors": list(
                        getattr(candidate, "satisfied_required_anchors", []) or []
                    )[:6],
                    "eligibility_blockers": list(
                        getattr(candidate, "eligibility_blockers", []) or []
                    )[:6],
                    "evidence_gaps": list(
                        getattr(candidate, "evidence_gaps", []) or []
                    )[:4],
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
                    "hard_contradiction": bool(getattr(candidate, "hard_contradiction", False)),
                }
                )
        return blocked

    def _apply_entity_audit(
        self,
        decision: JudgeDecision,
        candidates: Sequence[Any],
    ) -> None:
        by_name = {self._name(item): item for item in candidates or [] if self._name(item)}
        by_entity = {
            str(getattr(item, "entity_id", "") or ""): item
            for item in candidates or []
            if str(getattr(item, "entity_id", "") or "")
        }

        def candidate_for_name(name: Any) -> Any:
            text = str(name or "").strip()
            if not text:
                return None
            entity_id = ""
            if self.knowledge and hasattr(self.knowledge, "entity_id_for"):
                entity_id = self.knowledge.entity_id_for(text)
            return (by_entity.get(entity_id) if entity_id else None) or by_name.get(text)

        def entity_id_for_name(name: Any) -> str:
            candidate = candidate_for_name(name)
            if candidate is not None:
                return str(getattr(candidate, "entity_id", "") or "")
            if self.knowledge and hasattr(self.knowledge, "entity_id_for"):
                return self.knowledge.entity_id_for(name)
            return ""

        decision.retriever_top1_entity_id = entity_id_for_name(decision.retriever_top1)
        decision.judge_primary_entity_id = entity_id_for_name(decision.judge_primary)
        decision.primary_entity_id = entity_id_for_name(decision.primary)
        decision.secondary_entity_ids = [
            entity_id_for_name(name) for name in decision.secondary if entity_id_for_name(name)
        ]
        decision.differential_entity_ids = [
            entity_id_for_name(name) for name in decision.differential if entity_id_for_name(name)
        ]
        decision.evidence_gap_target_entity_ids = [
            entity_id_for_name(name) for name in decision.evidence_gap_targets if entity_id_for_name(name)
        ]
        decision.final_entity_ids = [
            entity_id_for_name(name) for name in decision.final_diagnoses if entity_id_for_name(name)
        ]
        records: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates or []:
            entity_id = str(getattr(candidate, "entity_id", "") or "")
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            records.append(
                {
                    "entity_id": entity_id,
                    "diagnosis": self._name(candidate),
                    "canonical_name": str(getattr(candidate, "canonical_name", "") or self._name(candidate)),
                    "submission_name": str(getattr(candidate, "submission_name", "") or self._name(candidate)),
                    "submittable": bool(getattr(candidate, "submittable", True)),
                }
            )
        decision.entity_resolutions = records

    def _reviews(
        self,
        ranked: Sequence[Any],
        primary: Any,
        secondary: Sequence[Any],
        gap_targets: Sequence[str],
        blocked: Sequence[Dict[str, Any]],
    ) -> List[JudgeCandidateReview]:
        roles: Dict[str, str] = {}
        if primary:
            primary_role = (
                "primary"
                if self._eligibility_status(primary) == PRIMARY_ELIGIBLE
                else "evidence_gap"
            )
            roles[primary.diagnosis] = primary_role
        roles.update({item.diagnosis: "secondary" for item in secondary})
        roles.update({name: "evidence_gap" for name in gap_targets if name not in roles})
        blocked_reasons = {item["diagnosis"]: item["reason"] for item in blocked}
        reviews: List[JudgeCandidateReview] = []
        for candidate in ranked[: self.max_reviews]:
            role = roles.get(candidate.diagnosis, "differential")
            reason = blocked_reasons.get(candidate.diagnosis, role)
            if role == "primary" and self._requires_gap_authorization(candidate):
                reason = "provisional primary: strong explainability with required evidence gap"
            reviews.append(
                JudgeCandidateReview(
                    diagnosis=candidate.diagnosis,
                    role=role,
                    reason=reason,
                    entity_id=str(getattr(candidate, "entity_id", "") or ""),
                    canonical_name=str(getattr(candidate, "canonical_name", "") or candidate.diagnosis),
                    submission_name=str(getattr(candidate, "submission_name", "") or candidate.diagnosis),
                    submittable=bool(getattr(candidate, "submittable", True)),
                    score=float(getattr(candidate, "score", 0.0) or 0.0),
                    judge_score=round(self._judge_score(candidate), 4),
                    required_met=bool(getattr(candidate, "required_met", False)),
                    required_gap_authorized=False,
                    hard_contradiction=bool(getattr(candidate, "hard_contradiction", False)),
                    coverage_score=float(getattr(candidate, "coverage_score", 0.0) or 0.0),
                    residual_score=float(getattr(candidate, "residual_score", 0.0) or 0.0),
                    explanatory_coverage=self._coverage(candidate),
                    core_explanatory_coverage=self._core_coverage(candidate),
                    residual_evidence_score=self._residual(candidate),
                    residual_core_evidence_count=self._residual_core_count(candidate),
                    diagnosis_type=str(getattr(candidate, "diagnosis_type", "") or ""),
                    specificity=float(getattr(candidate, "specificity", 0.0) or 0.0),
                    required_gaps=list(getattr(candidate, "required_gaps", []) or [])[:4],
                    matched_evidence=list(getattr(candidate, "matched_evidence", []) or [])[:6],
                    explained_evidence=list(getattr(candidate, "explained_evidence", []) or [])[:6],
                    unexplained_core_evidence=list(
                        getattr(candidate, "unexplained_core_evidence", []) or []
                    )[:6],
                    explanatory_rank_reason=str(
                        getattr(candidate, "explanatory_rank_reason", "") or ""
                    ),
                    required_gap_state=self._required_gap_state(candidate),
                    eligibility_status=self._eligibility_status(candidate),
                    eligibility_reason=str(
                        getattr(candidate, "eligibility_reason", "") or ""
                    ),
                    eligibility_substatus=str(
                        getattr(candidate, "eligibility_substatus", "") or ""
                    ),
                    eligibility_anchor_status=str(
                        getattr(candidate, "eligibility_anchor_status", "") or ""
                    ),
                    eligibility_anchor_policy_audit=dict(
                        getattr(candidate, "eligibility_anchor_policy_audit", {}) or {}
                    ),
                    missing_required_anchors=list(
                        getattr(candidate, "missing_required_anchors", []) or []
                    )[:6],
                    evidence_pattern_matches=list(
                        getattr(candidate, "evidence_pattern_matches", []) or []
                    )[:4],
                    clinical_pattern_matches=list(
                        getattr(candidate, "clinical_pattern_matches", []) or []
                    )[:4],
                    derived_pattern_assertions=list(
                        getattr(candidate, "derived_pattern_assertions", []) or []
                    )[:4],
                    bridge_validation_results=list(
                        getattr(candidate, "bridge_validation_results", []) or []
                    )[:4],
                    bridge_protection_decisions=list(
                        getattr(candidate, "bridge_protection_decisions", []) or []
                    )[:4],
                    exam_followup_authorized=bool(role == "evidence_gap"),
                    submission_authorized=bool(
                        role in {"primary", "secondary"}
                        and self._eligibility_status(candidate) == PRIMARY_ELIGIBLE
                    ),
                    evidence_gaps=list(getattr(candidate, "evidence_gaps", []) or [])[:4],
                    gap_values=list(getattr(candidate, "gap_values", []) or [])[:4],
                    max_gap_value=float(
                        getattr(candidate, "max_gap_value", 0.0) or 0.0
                    ),
                    actionable_gap_count=int(
                        getattr(candidate, "actionable_gap_count", 0) or 0
                    ),
                    deferred_priority=float(
                        getattr(candidate, "deferred_priority", 0.0) or 0.0
                    ),
                    deferred_priority_components=dict(
                        getattr(candidate, "deferred_priority_components", {}) or {}
                    ),
                    exam_priority_override=bool(
                        getattr(candidate, "exam_priority_override", False)
                    ),
                    exam_priority_override_reason=str(
                        getattr(candidate, "exam_priority_override_reason", "") or ""
                    ),
                )
            )
        return reviews

    def _requires_gap_authorization(self, candidate: Any) -> bool:
        return False

    def _gap_authorizable(self, candidate: Any) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        if getattr(candidate, "required_met", False):
            return False
        if not getattr(candidate, "required_gaps", None):
            return False
        if not self._priority(candidate):
            return False
        if not getattr(candidate, "matched_evidence", None):
            return False
        if not self._objective_signal(candidate) and not self._explanatory_gap_authorizable(candidate):
            return False
        explanatory_score = (
            0.55 * self._coverage(candidate)
            + 0.35 * self._core_coverage(candidate)
            + 0.10 * float(getattr(candidate, "source_prior", 0.0) or 0.0)
            - 0.20 * self._residual(candidate)
        )
        if (
            float(getattr(candidate, "score", 0.0) or 0.0) < self.gap_authorization_min_score
            and explanatory_score < self.gap_authorization_min_explanatory_score
        ):
            return False
        return (
            self._coverage(candidate) >= self.gap_authorization_min_coverage
            or self._residual(candidate) <= self.gap_authorization_max_residual
            or self._core_coverage(candidate) >= self.gap_authorization_min_coverage
            or float(getattr(candidate, "source_prior", 0.0) or 0.0) >= 0.45
        )

    def _trusted(self, candidate: Any) -> bool:
        if not candidate:
            return False
        status = self._eligibility_status(candidate)
        if status:
            return status == PRIMARY_ELIGIBLE
        return bool(
            not getattr(candidate, "hard_contradiction", False)
            and getattr(candidate, "matched_evidence", None)
            and getattr(candidate, "required_met", False)
        )

    def _has_signal(self, candidate: Any) -> bool:
        return bool(
            candidate
            and (
                getattr(candidate, "matched_evidence", None)
                or float(getattr(candidate, "source_prior", 0.0) or 0.0) > 0.0
            )
        )

    def _sort_key(self, candidate: Any) -> tuple:
        status_rank = {
            PRIMARY_ELIGIBLE: 3,
            DEFERRED: 2,
            DIFFERENTIAL_ONLY: 1,
            EXCLUDED: 0,
        }.get(self._eligibility_status(candidate), 1)
        return (
            status_rank,
            self._primary_eligibility_score(candidate),
            self._core_coverage(candidate),
            -self._residual_core_count(candidate),
            1.0 - self._residual(candidate),
            self._coverage(candidate),
            self._judge_score(candidate),
            1 if self._priority(candidate) else 0,
            float(getattr(candidate, "evidence_specificity_score", 0.0) or 0.0),
            float(getattr(candidate, "score", 0.0) or 0.0),
        )

    def _primary_eligibility_score(self, candidate: Any) -> float:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return -1.0
        status = self._eligibility_status(candidate)
        if status == EXCLUDED:
            return -1.0
        if status == DIFFERENTIAL_ONLY:
            return -0.2
        gap_state = self._required_gap_state(candidate)
        gap_penalty = {
            "satisfied": 0.0,
            "nonblocking_gap": 0.03,
            "partially_satisfied": 0.08,
            "actionable_gap": 0.10,
            "unsupported_gap": 0.28,
            "hard_contradiction": 1.0,
        }.get(gap_state, 0.18)
        score = (
            0.32 * self._core_coverage(candidate)
            + 0.24 * self._coverage(candidate)
            + 0.14 * self._component_score(candidate, "core_evidence_score")
            + 0.22 * self._component_score(candidate, "diagnostic_evidence_score")
            + 0.14 * float(getattr(candidate, "evidence_specificity_score", 0.0) or 0.0)
            + 0.12 * float(getattr(candidate, "source_prior", 0.0) or 0.0)
            + 0.10 * float(getattr(candidate, "score", 0.0) or 0.0)
            - 0.18 * min(1.0, 0.25 * self._residual_core_count(candidate))
            - 0.14 * self._residual(candidate)
            - 0.16 * self._component_score(candidate, "generic_parent_penalty")
            - 0.08 * self._component_score(candidate, "specific_over_generic_penalty")
            - gap_penalty
            - 0.10 * float(getattr(candidate, "contradiction_penalty", 0.0) or 0.0)
        )
        if self._priority(candidate):
            score += 0.06
        if self._systemic_primary(candidate):
            score += 0.04
        if self._is_manifestation(candidate):
            score -= 0.08
        if self._generic_parent_name(candidate):
            score -= 0.05
        return round(score, 4)

    def _judge_score(self, candidate: Any) -> float:
        score = float(getattr(candidate, "score", 0.0) or 0.0)
        score += self.coverage_bonus * self._coverage(candidate)
        score += self.core_coverage_bonus * self._core_coverage(candidate)
        score += 0.10 * self._component_score(candidate, "core_evidence_score")
        score += 0.16 * self._component_score(candidate, "diagnostic_evidence_score")
        score -= self.residual_penalty * self._residual(candidate)
        score -= self.residual_core_penalty * min(
            4,
            self._residual_core_count(candidate),
        )
        score -= 0.14 * self._component_score(candidate, "generic_parent_penalty")
        score -= 0.06 * self._component_score(candidate, "specific_over_generic_penalty")
        gap_state = self._required_gap_state(candidate)
        if gap_state == "satisfied":
            score += self.required_met_bonus
        elif gap_state in {"actionable_gap", "nonblocking_gap", "partially_satisfied"} and self._gap_authorizable(candidate):
            score += self.priority_gap_bonus
        elif gap_state == "unsupported_gap":
            score -= 0.15
        if self._priority(candidate):
            score += 0.05
        if not self._disease_specific_priority_allowed(candidate):
            score -= 0.25
        if self._systemic_primary(candidate):
            score += 0.05
        if self._is_manifestation(candidate):
            score -= 0.08
        if self._generic_parent_name(candidate):
            score -= 0.04
        if getattr(candidate, "hard_contradiction", False):
            score -= 1.0
        return round(score, 4)

    @staticmethod
    def _component_score(candidate: Any, name: str) -> float:
        try:
            return float((getattr(candidate, "component_scores", {}) or {}).get(name, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _core_or_diagnostic_signal(self, candidate: Any) -> bool:
        return bool(
            getattr(candidate, "core_matched_evidence", None)
            or getattr(candidate, "diagnostic_matched_evidence", None)
            or self._component_score(candidate, "core_evidence_score") >= 0.20
            or self._component_score(candidate, "diagnostic_evidence_score") > 0.0
            or self._core_coverage(candidate) >= 0.50
        )

    @staticmethod
    def _coverage(candidate: Any) -> float:
        value = getattr(candidate, "explanatory_coverage", None)
        legacy = getattr(candidate, "coverage_score", 0.0)
        if value is None or (float(value or 0.0) == 0.0 and float(legacy or 0.0) > 0.0):
            value = legacy
        return float(value or 0.0)

    @staticmethod
    def _core_coverage(candidate: Any) -> float:
        value = getattr(candidate, "core_explanatory_coverage", None)
        component_value = (getattr(candidate, "component_scores", {}) or {}).get(
            "core_explanatory_coverage"
        )
        legacy = getattr(candidate, "coverage_score", 0.0)
        if component_value is not None and (
            value is None or float(value or 0.0) == 0.0
        ):
            value = component_value
        if value is None or (float(value or 0.0) == 0.0 and float(legacy or 0.0) > 0.0):
            value = legacy
        return float(value or 0.0)

    @staticmethod
    def _residual(candidate: Any) -> float:
        value = getattr(candidate, "residual_evidence_score", None)
        legacy = getattr(candidate, "residual_score", 1.0)
        if value is None or (float(value or 0.0) == 0.0 and float(legacy or 0.0) > 0.0):
            value = legacy
        return float(value or 0.0)

    @staticmethod
    def _residual_core_count(candidate: Any) -> int:
        value = getattr(candidate, "residual_core_evidence_count", None)
        component_value = (getattr(candidate, "component_scores", {}) or {}).get(
            "residual_core_evidence_count"
        )
        try:
            current = int(float(value or 0))
        except (TypeError, ValueError):
            current = 0
        if component_value is not None and (value is None or current == 0):
            value = component_value
        try:
            return int(float(value or 0))
        except (TypeError, ValueError):
            return 0

    def _priority(self, candidate: Any) -> bool:
        dtype = str(getattr(candidate, "diagnosis_type", "") or "").lower()
        if not self._disease_specific_priority_allowed(candidate):
            return False
        return dtype in _PRIORITY_TYPES

    @staticmethod
    def _disease_specific_priority_allowed(candidate: Any) -> bool:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        if getattr(candidate, "diagnosis", "") == "克里格勒-纳贾尔综合征":
            return bool(
                matched
                & {
                    "ugt1a1_positive",
                    "genetic_suspicion",
                    "neonatal_jaundice",
                    "diagnosis:克里格勒-纳贾尔综合征",
                }
            )
        if getattr(candidate, "diagnosis", "") != "压力性尿失禁":
            return True
        return bool(
            matched
            & {
                "stress_urinary_incontinence",
                "urine_leak_with_pressure",
                "diagnosis:压力性尿失禁",
                "symptom:压力性尿失禁",
            }
        )

    def _systemic_primary(self, candidate: Any) -> bool:
        return bool(
            candidate
            and (
                getattr(candidate, "diagnosis", "") in _SYSTEMIC_PRIMARY_NAMES
                or str(getattr(candidate, "diagnosis_type", "") or "").lower() == "systemic"
            )
        )

    def _is_manifestation(self, candidate: Any) -> bool:
        if self._systemic_primary(candidate):
            return False
        dtype = str(getattr(candidate, "diagnosis_type", "") or "").lower()
        return dtype in _MANIFESTATION_TYPES or getattr(candidate, "diagnosis", "") in _MANIFESTATION_NAMES

    def _generic_parent_name(self, candidate: Any) -> str:
        if not candidate:
            return ""
        diagnosis = getattr(candidate, "diagnosis", "")
        if diagnosis in _GENERIC_PARENT_DIAGNOSES:
            return diagnosis
        if self._component_score(candidate, "generic_parent_penalty") > 0.0:
            return diagnosis
        if not self.knowledge:
            return ""
        entry = self._knowledge_entry(diagnosis)
        for item in getattr(self.knowledge, "entries", {}).values():
            if str(item.get("parent_diagnosis") or "") == diagnosis:
                return diagnosis
        if entry.get("generic_suppressions") or entry.get("generalization_suppressions"):
            return ""
        return ""

    def _generic_parent_of(self, specific: Any, parent: Any) -> bool:
        return bool(
            specific
            and parent
            and str(getattr(specific, "parent_diagnosis", "") or "") == getattr(parent, "diagnosis", "")
        )

    def _same_family(self, left: Any, right: Any) -> bool:
        if not self.knowledge or not left or not right:
            return False
        left_entry = self._knowledge_entry(left.diagnosis)
        right_entry = self._knowledge_entry(right.diagnosis)
        left_system = str(left_entry.get("body_system") or "")
        right_system = str(right_entry.get("body_system") or "")
        left_family = str(left_entry.get("disease_family") or left_entry.get("family") or "")
        right_family = str(right_entry.get("disease_family") or right_entry.get("family") or "")
        if left_system and right_system and left_system == right_system:
            return bool(left_family and right_family and left_family == right_family)
        left_parent = str(left_entry.get("parent_diagnosis") or "")
        right_parent = str(right_entry.get("parent_diagnosis") or "")
        return bool(left_parent and right_parent and left_parent == right_parent)

    def _same_body_system(self, left: Any, right: Any) -> bool:
        if not self.knowledge or not left or not right:
            return False
        left_entry = self._knowledge_entry(left.diagnosis)
        right_entry = self._knowledge_entry(right.diagnosis)
        left_system = str(left_entry.get("body_system") or "")
        right_system = str(right_entry.get("body_system") or "")
        return bool(left_system and right_system and left_system == right_system)

    def _causally_related(self, left: Any, right: Any) -> bool:
        if not self.knowledge or not left or not right:
            return False
        left_entry = self._knowledge_entry(left.diagnosis)
        right_entry = self._knowledge_entry(right.diagnosis)
        return (
            right.diagnosis in set(str(item) for item in left_entry.get("causes", []) or [])
            or left.diagnosis in set(str(item) for item in right_entry.get("caused_by", []) or [])
            or left.diagnosis in set(str(item) for item in right_entry.get("causes", []) or [])
            or right.diagnosis in set(str(item) for item in left_entry.get("caused_by", []) or [])
            or self._same_family(left, right)
        )

    @staticmethod
    def _independent_objective(candidate: Any) -> bool:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        if f"diagnosis:{candidate.diagnosis}" in matched:
            return True
        components = getattr(candidate, "component_scores", {}) or {}
        if float(components.get("objective_evidence", 0.0) or 0.0) >= 1.0:
            return True
        if candidate.diagnosis == "心力衰竭":
            return DiagnosisJudge._heart_failure_state_evidence(matched)
        return bool(
            matched
            & {
                "heart_failure_state",
                "renal_impairment",
                "egfr_low",
                "urea_elevated",
                "portal_vein_dilation",
                "portal_flow_abnormal",
                "ventricular_septal_defect",
                "pulmonary_valve_stenosis",
            }
        )

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

    @staticmethod
    def _objective_signal(candidate: Any) -> bool:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        if f"diagnosis:{candidate.diagnosis}" in matched:
            return True
        if matched & _OBJECTIVE_GAP_FINDINGS:
            return True
        components = getattr(candidate, "component_scores", {}) or {}
        return float(components.get("objective_evidence", 0.0) or 0.0) >= 1.0

    @staticmethod
    def _name(candidate: Any) -> str:
        return str(getattr(candidate, "diagnosis", "") or "")

    @staticmethod
    def _reasoning(decision: JudgeDecision) -> str:
        if not decision.primary:
            return "Judge did not authorize a final diagnosis."
        text = (
            f"Judge authorized primary={decision.primary}; "
            f"retriever_top1={decision.retriever_top1 or 'none'}; "
            f"override={int(decision.decision_override)}."
        )
        if decision.needs_discriminating_exams:
            text += (
                " Primary lock deferred for discriminating exams; "
                f"provisional_primary={decision.provisional_primary}."
            )
        if decision.primary_unlock_reason:
            text += " Primary unlock: " + decision.primary_unlock_reason + "."
        if decision.gap_state_distribution:
            text += " Gap states: " + ", ".join(
                f"{key}={value}"
                for key, value in sorted(decision.gap_state_distribution.items())
            ) + "."
        if decision.evidence_gap_targets:
            text += " Evidence-gap targets: " + ", ".join(decision.evidence_gap_targets) + "."
        return text

    def apply_root_cause_arbitration(
        self,
        judge_decision: JudgeDecision,
        result: Any,
        *,
        max_final_diagnoses: int = 3,
    ) -> JudgeDecision:
        """Judge-owned adoption of root-cause arbitration opinions."""
        if not judge_decision or not result:
            return judge_decision
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
        judge_decision.root_cause_arbitration = dict(payload)
        judge_decision.root_cause_primary = str(
            getattr(result, "root_cause_primary", "") or payload.get("root_cause_primary") or ""
        )
        judge_decision.root_cause_secondary = list(
            getattr(result, "root_cause_secondary", None)
            or payload.get("root_cause_secondary")
            or []
        )
        judge_decision.candidate_explanation_edges = list(
            getattr(result, "candidate_explanation_edges", None)
            or payload.get("candidate_explanation_edges")
            or []
        )
        if not bool(getattr(result, "applied", False) or payload.get("applied")):
            return judge_decision
        if (
            str(getattr(judge_decision, "primary_status", "") or "") != "locked"
            or bool(getattr(judge_decision, "needs_discriminating_exams", False))
        ):
            payload["applied"] = False
            payload["blocked_reason"] = "root_cause_arbitration_requires_locked_judge_primary"
            judge_decision.root_cause_arbitration = dict(payload)
            return judge_decision

        primary = str(
            getattr(result, "root_cause_primary", "") or payload.get("root_cause_primary") or ""
        )
        final_secondary = list(
            getattr(result, "root_cause_final_secondary", None)
            or payload.get("root_cause_final_secondary")
            or []
        )
        audit_secondary = list(
            getattr(result, "root_cause_secondary", None)
            or payload.get("root_cause_secondary")
            or []
        )
        blocked_secondary = set(audit_secondary) - set(final_secondary)
        existing_final = [
            name
            for name in list(judge_decision.final_diagnoses or [])
            if name not in blocked_secondary and name != primary
        ]
        final = list(
            dict.fromkeys(
                [primary]
                + [name for name in final_secondary if name != primary]
                + existing_final
            )
        )[: max(1, int(max_final_diagnoses or 1))]
        if not primary or not final:
            return judge_decision

        judge_decision.primary = primary
        judge_decision.judge_primary = primary
        judge_decision.locked_primary = primary
        judge_decision.provisional_primary = ""
        judge_decision.primary_status = "locked"
        judge_decision.needs_discriminating_exams = False
        judge_decision.defer_reason = ""
        judge_decision.secondary = [name for name in final if name != primary]
        judge_decision.final_diagnoses = final
        judge_decision.root_cause_primary_override = bool(
            getattr(result, "primary_override", False) or payload.get("primary_override")
        )
        judge_decision.primary_override_source = "judge_root_cause_arbitration"
        judge_decision.root_cause_coverage = float(
            getattr(result, "root_cause_coverage", 0.0)
            or payload.get("root_cause_coverage")
            or 0.0
        )
        judge_decision.required_gap_authorized_diagnoses = []
        trace = list(judge_decision.dynamic_rerank_trace or [])
        trace.append(
            {
                "stage": "judge_root_cause_arbitration",
                "primary_before": str(
                    getattr(result, "primary_before", "") or payload.get("primary_before") or ""
                ),
                "primary_after": str(
                    getattr(result, "primary_after", "") or payload.get("primary_after") or primary
                ),
                "primary_override": bool(judge_decision.root_cause_primary_override),
                "root_cause_secondary": list(audit_secondary),
                "root_cause_final_secondary": list(final_secondary),
                "root_cause_coverage": judge_decision.root_cause_coverage,
                "candidate_explanation_edges": list(judge_decision.candidate_explanation_edges),
            }
        )
        judge_decision.dynamic_rerank_trace = trace
        judge_decision.reasoning = self._reasoning(judge_decision)
        return judge_decision


class DiagnosisSubmitter:
    """Write a JudgeDecision into the mutable DiagnosisDecision audit object."""

    def __init__(self, knowledge: Any = None):
        self.knowledge = knowledge

    def apply(self, decision: Any, judge_decision: JudgeDecision) -> Any:
        if not decision or not judge_decision:
            return decision
        if judge_decision_is_stale(decision, judge_decision):
            setattr(decision, "stale_decision", True)
            raise StaleJudgeDecisionError(
                "JudgeDecision is stale for the current evidence snapshot"
            )
        score_by_name = {item.diagnosis: item for item in getattr(decision, "candidates", []) or []}
        score_by_entity = {
            str(getattr(item, "entity_id", "") or ""): item
            for item in getattr(decision, "candidates", []) or []
            if str(getattr(item, "entity_id", "") or "")
        }
        def candidate_for_name(name: Any) -> Any:
            text = str(name or "").strip()
            if not text:
                return None
            entity_id = ""
            if self.knowledge and hasattr(self.knowledge, "entity_id_for"):
                entity_id = self.knowledge.entity_id_for(text)
            return (score_by_entity.get(entity_id) if entity_id else None) or score_by_name.get(text)
        for candidate in list(getattr(decision, "candidates", []) or []):
            setattr(candidate, "submission_authorized", False)
            setattr(candidate, "exam_followup_authorized", False)
        submission_allowed = (
            str(getattr(judge_decision, "primary_status", "") or "") == "locked"
            and not bool(getattr(judge_decision, "needs_discriminating_exams", False))
        )
        final_from_judge = list(judge_decision.final_diagnoses) if submission_allowed else []
        for name in final_from_judge:
            candidate = candidate_for_name(name)
            if candidate:
                candidate.differential_only = False
                candidate.differential_only_reason = ""
                candidate.submission_authorized = True
        for name in list(getattr(judge_decision, "evidence_gap_targets", []) or []):
            candidate = candidate_for_name(name)
            if candidate:
                candidate.exam_followup_authorized = True
        for item in judge_decision.blocked_diagnoses:
            candidate = candidate_for_name(item.get("diagnosis"))
            if candidate and candidate.diagnosis not in set(final_from_judge):
                candidate.differential_only = True
                candidate.differential_only_reason = str(item.get("reason") or "differential_only")

        decision.retriever_top1 = judge_decision.retriever_top1
        decision.judge_primary = judge_decision.judge_primary
        decision.submitter_final = list(final_from_judge)
        decision.decision_override = bool(judge_decision.decision_override)
        decision.required_gap_authorized_diagnoses = []
        decision.evidence_conflicts = list(judge_decision.evidence_conflicts)
        decision.conflict_affected_diagnoses = list(
            judge_decision.conflict_affected_diagnoses
        )
        decision.eligibility_distribution = dict(
            getattr(judge_decision, "eligibility_distribution", {}) or {}
        )
        decision.deferred_anchor_candidates = list(
            getattr(judge_decision, "deferred_anchor_candidates", []) or []
        )
        decision.excluded_candidates = list(
            getattr(judge_decision, "excluded_candidates", []) or []
        )
        decision.primary_eligible_candidates = list(
            getattr(judge_decision, "primary_eligible_candidates", []) or []
        )
        decision.root_cause_arbitration = dict(
            getattr(judge_decision, "root_cause_arbitration", {}) or {}
        )
        decision.root_cause_primary = str(
            getattr(judge_decision, "root_cause_primary", "") or ""
        )
        decision.root_cause_secondary = list(
            getattr(judge_decision, "root_cause_secondary", []) or []
        )
        decision.candidate_explanation_edges = list(
            getattr(judge_decision, "candidate_explanation_edges", []) or []
        )
        decision.case_version = int(getattr(judge_decision, "case_version", 0) or getattr(decision, "case_version", 0) or 0)
        decision.evidence_snapshot_hash = str(
            getattr(judge_decision, "evidence_snapshot_hash", "")
            or getattr(decision, "evidence_snapshot_hash", "")
            or ""
        )
        decision.knowledge_profile_version = str(
            getattr(judge_decision, "knowledge_profile_version", "")
            or getattr(decision, "knowledge_profile_version", "")
            or ""
        )
        decision.decision_policy_version = str(
            getattr(judge_decision, "decision_policy_version", "")
            or getattr(decision, "decision_policy_version", "")
            or ""
        )
        decision.exam_catalog_version = str(
            getattr(judge_decision, "exam_catalog_version", "")
            or getattr(decision, "exam_catalog_version", "")
            or ""
        )
        decision.judge_decision = judge_decision.to_dict()
        if isinstance(getattr(decision, "case_board", None), dict):
            case_board = dict(decision.case_board)
            case_board["case_version"] = decision.case_version
            case_board["evidence_snapshot_hash"] = decision.evidence_snapshot_hash
            case_board["knowledge_profile_version"] = decision.knowledge_profile_version
            case_board["decision_policy_version"] = decision.decision_policy_version
            case_board["exam_catalog_version"] = decision.exam_catalog_version
            case_board["judge_decision"] = judge_decision.to_dict()
            case_board["candidate_decisions"] = [
                item.to_dict() if hasattr(item, "to_dict") else dict(item)
                for item in list(getattr(judge_decision, "reviews", []) or [])
            ]
            decision.case_board = case_board
        decision.pre_authorization_diagnoses = list(judge_decision.final_diagnoses)
        decision.final_diagnoses = list(final_from_judge)
        decision.authorized_diagnoses = list(final_from_judge)
        decision.trusted_diagnoses = [
            name
            for name in final_from_judge
            if candidate_for_name(name)
            and getattr(candidate_for_name(name), "trusted", False)
        ]
        decision.blocked_diagnoses = list(judge_decision.blocked_diagnoses)
        decision.submission_override_count = int(judge_decision.decision_override)
        if judge_decision.reasoning and judge_decision.reasoning not in str(
            getattr(decision, "evidence_reasoning", "") or ""
        ):
            decision.evidence_reasoning = (
                str(getattr(decision, "evidence_reasoning", "") or "").rstrip()
                + " "
                + judge_decision.reasoning
            ).strip()
        return decision
