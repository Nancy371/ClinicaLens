"""Canonical evidence definitions for targeted claim verification."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class EvidenceDefinition:
    evidence_id: str
    aliases: List[str] = field(default_factory=list)
    preferred_sections: List[str] = field(default_factory=list)
    verification_type: str = "presence"
    positive_terms: List[str] = field(default_factory=list)
    negative_terms: List[str] = field(default_factory=list)
    ambiguous_terms: List[str] = field(default_factory=list)
    followup_exam: str = ""
    requires_value: bool = False
    evidence_level: str = "specific"
    information_value: float = 0.75

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceDefinitionRegistry:
    """Load and normalize evidence definitions from ref_data.

    The registry owns aliases and verification hints so LLM hypotheses can stay
    canonical: a Reasoner asks for `blast_present`; this class decides which
    source terms can verify it.
    """

    def __init__(self, ref_dir: str = "data/ref_data"):
        self.ref_dir = ref_dir
        self.path = os.path.join(ref_dir, "evidence_registry.json")
        self.definitions: Dict[str, EvidenceDefinition] = {}
        self.alias_index: Dict[str, str] = {}
        self._load()

    def get(self, evidence_id: Any) -> Optional[EvidenceDefinition]:
        canonical = self.normalize_evidence_id(evidence_id)
        if not canonical:
            return None
        return self.definitions.get(canonical)

    def require(self, evidence_id: Any) -> EvidenceDefinition:
        definition = self.get(evidence_id)
        if definition is not None:
            return definition
        text = str(evidence_id or "").strip()
        return EvidenceDefinition(evidence_id=text, aliases=[text] if text else [])

    def normalize_evidence_id(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if text in self.definitions:
            return text
        return self.alias_index.get(_compact(text), text)

    def aliases_for(self, evidence_id: Any) -> List[str]:
        definition = self.require(evidence_id)
        return _dedupe([definition.evidence_id] + list(definition.aliases))

    def followup_exam_for(self, evidence_id: Any) -> str:
        definition = self.get(evidence_id)
        return str(definition.followup_exam or "") if definition else ""

    def known_ids(self) -> List[str]:
        return sorted(self.definitions)

    def _load(self) -> None:
        payload = _read_json(self.path, {})
        raw_items = payload.get("evidence") if isinstance(payload, dict) else None
        if not raw_items:
            raw_items = _BUILTIN_EVIDENCE
        merged: Dict[str, Dict[str, Any]] = {
            str(item.get("evidence_id") or ""): dict(item)
            for item in _BUILTIN_EVIDENCE
            if str(item.get("evidence_id") or "")
        }
        for item in raw_items or []:
            if isinstance(item, dict) and item.get("evidence_id"):
                merged[str(item["evidence_id"])] = {**merged.get(str(item["evidence_id"]), {}), **dict(item)}
        for item in merged.values():
            evidence_id = str(item.get("evidence_id") or "").strip()
            if not evidence_id:
                continue
            definition = EvidenceDefinition(
                evidence_id=evidence_id,
                aliases=_text_list(item.get("aliases") or []),
                preferred_sections=_text_list(item.get("preferred_sections") or []),
                verification_type=str(item.get("verification_type") or "presence"),
                positive_terms=_text_list(item.get("positive_terms") or []),
                negative_terms=_text_list(item.get("negative_terms") or []),
                ambiguous_terms=_text_list(item.get("ambiguous_terms") or []),
                followup_exam=str(item.get("followup_exam") or ""),
                requires_value=bool(item.get("requires_value", False)),
                evidence_level=str(item.get("evidence_level") or "specific"),
                information_value=_float(item.get("information_value"), 0.75),
            )
            self.definitions[evidence_id] = definition
            for alias in [evidence_id] + definition.aliases:
                if alias:
                    self.alias_index[_compact(alias)] = evidence_id


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return default


def _text_list(values: Iterable[Any]) -> List[str]:
    return _dedupe(str(item).strip() for item in values or [] if str(item).strip())


def _dedupe(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _compact(value: Any) -> str:
    return "".join(str(value or "").lower().split())


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_COMMON_POSITIVE_TERMS = [
    "阳性",
    "提示",
    "发现",
    "可见",
    "存在",
    "升高",
    "降低",
    "异常",
    "检出",
    "positive",
    "+",
]

_COMMON_NEGATIVE_TERMS = [
    "阴性",
    "未见",
    "未发现",
    "未提示",
    "未检出",
    "无",
    "正常",
    "排除",
    "negative",
    "no ",
]

_BUILTIN_EVIDENCE: List[Dict[str, Any]] = [
    {
        "evidence_id": "blast_present",
        "aliases": ["原始细胞", "母细胞", "blast", "myeloblast", "lymphoblast"],
        "preferred_sections": ["peripheral_blood_smear", "bone_marrow", "laboratory", "外周血涂片", "骨髓"],
        "verification_type": "presence_or_percentage",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "ambiguous_terms": ["异常幼稚细胞", "可疑幼稚细胞"],
        "followup_exam": "外周血涂片",
        "evidence_level": "diagnostic_pattern",
        "information_value": 0.96,
    },
    {
        "evidence_id": "hemoglobin_low",
        "aliases": ["贫血", "血红蛋白降低", "Hb低", "HGB low", "hemoglobin low"],
        "preferred_sections": ["CBC", "laboratory", "血常规"],
        "verification_type": "lab_direction",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "全血细胞计数（CBC）",
        "information_value": 0.82,
    },
    {
        "evidence_id": "platelet_low",
        "aliases": ["血小板减少", "PLT低", "platelet low", "thrombocytopenia"],
        "preferred_sections": ["CBC", "laboratory", "血常规"],
        "verification_type": "lab_direction",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "全血细胞计数（CBC）",
        "information_value": 0.84,
    },
    {
        "evidence_id": "white_blood_cell_abnormal",
        "aliases": ["白细胞异常", "白细胞升高", "白细胞降低", "WBC abnormal", "leukocytosis", "leukopenia"],
        "preferred_sections": ["CBC", "laboratory", "血常规"],
        "verification_type": "lab_direction",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "全血细胞计数（CBC）",
        "information_value": 0.8,
    },
    {
        "evidence_id": "multilineage_cytopenia",
        "aliases": ["双系减少", "三系减少", "多系血细胞减少", "multilineage cytopenia"],
        "preferred_sections": ["CBC", "laboratory", "血常规"],
        "verification_type": "derived_pattern",
        "followup_exam": "全血细胞计数（CBC）",
        "evidence_level": "diagnostic_pattern",
        "information_value": 0.88,
    },
    {
        "evidence_id": "acute_leukemia_pattern",
        "aliases": ["急性白血病模式", "acute leukemia pattern"],
        "preferred_sections": ["peripheral_blood_smear", "bone_marrow", "laboratory"],
        "verification_type": "derived_pattern",
        "followup_exam": "骨髓穿刺和活检（BMAB）",
        "evidence_level": "diagnostic_pattern",
        "information_value": 0.96,
    },
    {
        "evidence_id": "pyuria",
        "aliases": ["脓尿", "尿白细胞升高", "尿WBC升高", "pyuria"],
        "preferred_sections": ["urinalysis", "laboratory", "尿液分析"],
        "verification_type": "lab_direction",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "尿液分析（UA）",
        "information_value": 0.36,
    },
    {
        "evidence_id": "urine_culture_no_growth",
        "aliases": ["尿培养无生长", "尿培养阴性", "未培养出细菌", "no growth in urine culture"],
        "preferred_sections": ["urine_culture", "microbiology", "尿培养"],
        "verification_type": "negative_fact",
        "positive_terms": ["无生长", "阴性", "未培养出", "未检出", "no growth", "negative"],
        "negative_terms": ["阳性", "生长", "检出", "positive"],
        "followup_exam": "尿培养",
        "information_value": 0.72,
    },
    {
        "evidence_id": "nitrite_negative",
        "aliases": ["亚硝酸盐阴性", "nitrite negative"],
        "preferred_sections": ["urinalysis", "尿液分析"],
        "verification_type": "negative_fact",
        "positive_terms": ["阴性", "negative", "-"],
        "negative_terms": ["阳性", "positive", "+"],
        "followup_exam": "尿液分析（UA）",
        "information_value": 0.62,
    },
    {
        "evidence_id": "leukocyte_esterase_negative",
        "aliases": ["白细胞酯酶阴性", "LE阴性", "leukocyte esterase negative"],
        "preferred_sections": ["urinalysis", "尿液分析"],
        "verification_type": "negative_fact",
        "positive_terms": ["阴性", "negative", "-"],
        "negative_terms": ["阳性", "positive", "+"],
        "followup_exam": "尿液分析（UA）",
        "information_value": 0.64,
    },
    {
        "evidence_id": "urine_wbc_normal",
        "aliases": ["尿白细胞正常", "尿WBC正常", "white cells normal in urine"],
        "preferred_sections": ["urinalysis", "尿液分析"],
        "verification_type": "negative_fact",
        "positive_terms": ["正常", "0-5", "阴性", "normal"],
        "negative_terms": ["升高", "增多", "阳性"],
        "followup_exam": "尿液分析（UA）",
        "information_value": 0.58,
    },
    {
        "evidence_id": "prostate_tenderness",
        "aliases": ["前列腺压痛", "直肠指检压痛", "DRE压痛"],
        "preferred_sections": ["physical_exam", "DRE", "体格检查"],
        "verification_type": "physical_finding",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "直肠指检（DRE）",
        "information_value": 0.86,
    },
    {
        "evidence_id": "right_to_left_shunt",
        "aliases": ["右向左分流", "右至左分流", "肺内分流", "right-to-left shunt", "right to left shunt"],
        "preferred_sections": ["echocardiography", "imaging", "血气", "超声"],
        "verification_type": "presence",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "超声心动图右心声学造影",
        "evidence_level": "diagnostic_pattern",
        "information_value": 0.96,
    },
    {
        "evidence_id": "pulmonary_avm_mechanism",
        "aliases": ["肺动静脉瘘", "肺动静脉畸形", "PAVM", "肺内右向左分流", "pulmonary AVM"],
        "preferred_sections": ["imaging", "CT", "CTA", "胸部增强CT"],
        "verification_type": "presence",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "肺动脉CTA",
        "evidence_level": "diagnostic_pattern",
        "information_value": 0.94,
    },
    {
        "evidence_id": "pulmonary_cta_positive",
        "aliases": ["肺动脉CTA阳性", "CTA提示肺动静脉瘘", "肺血管畸形", "pulmonary CTA positive"],
        "preferred_sections": ["CTA", "CT", "imaging"],
        "verification_type": "imaging_confirmation",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "肺动脉CTA",
        "evidence_level": "diagnostic_pattern",
        "information_value": 0.98,
    },
    {
        "evidence_id": "enhanced_ct_vascular_malformation",
        "aliases": ["胸部增强CT血管畸形", "增强CT提示血管畸形", "强化血管团", "enhanced CT vascular malformation"],
        "preferred_sections": ["CT", "imaging", "胸部增强CT"],
        "verification_type": "imaging_confirmation",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "胸部增强CT",
        "evidence_level": "diagnostic_pattern",
        "information_value": 0.97,
    },
    {
        "evidence_id": "bubble_echo_right_to_left_shunt",
        "aliases": ["右心声学造影阳性", "声学造影右向左分流", "bubble echo right-to-left shunt"],
        "preferred_sections": ["echocardiography", "超声"],
        "verification_type": "imaging_confirmation",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "超声心动图右心声学造影",
        "evidence_level": "diagnostic_pattern",
        "information_value": 0.97,
    },
    {
        "evidence_id": "feeding_pulmonary_artery_present",
        "aliases": ["供血肺动脉", "供血动脉", "feeding pulmonary artery"],
        "preferred_sections": ["CTA", "CT", "imaging", "胸部增强CT"],
        "verification_type": "targeted_exam_observation",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "肺动脉CTA",
        "evidence_level": "observed_imaging",
        "information_value": 0.94,
    },
    {
        "evidence_id": "draining_pulmonary_vein_present",
        "aliases": ["引流肺静脉", "早期引流肺静脉", "draining pulmonary vein"],
        "preferred_sections": ["CTA", "CT", "imaging", "胸部增强CT"],
        "verification_type": "targeted_exam_observation",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "肺动脉CTA",
        "evidence_level": "observed_imaging",
        "information_value": 0.94,
    },
    {
        "evidence_id": "early_pulmonary_venous_enhancement",
        "aliases": ["早期肺静脉显影", "肺静脉早期显影", "early pulmonary venous enhancement"],
        "preferred_sections": ["CTA", "CT", "imaging", "胸部增强CT"],
        "verification_type": "targeted_exam_observation",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "胸部增强CT",
        "evidence_level": "observed_imaging",
        "information_value": 0.93,
    },
    {
        "evidence_id": "abnormal_pulmonary_vascular_cluster",
        "aliases": ["异常肺血管团", "异常血管团", "强化血管团", "迂曲血管"],
        "preferred_sections": ["CTA", "CT", "imaging", "胸部增强CT"],
        "verification_type": "targeted_exam_observation",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "胸部增强CT",
        "evidence_level": "observed_imaging",
        "information_value": 0.82,
    },
    {
        "evidence_id": "abnormal_pulmonary_av_connection_described",
        "aliases": ["肺动静脉交通", "动静脉异常交通", "肺动静脉瘘样血管结构"],
        "preferred_sections": ["CTA", "CT", "imaging", "胸部增强CT"],
        "verification_type": "targeted_exam_observation",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "肺动脉CTA",
        "evidence_level": "observed_imaging",
        "information_value": 0.98,
    },
    {
        "evidence_id": "delayed_bubbles_in_left_heart",
        "aliases": ["微泡延迟进入左心", "左心延迟显影", "delayed bubbles in left heart"],
        "preferred_sections": ["echocardiography", "超声", "右心声学造影"],
        "verification_type": "targeted_exam_observation",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "超声心动图右心声学造影",
        "evidence_level": "observed_imaging",
        "information_value": 0.96,
    },
    {
        "evidence_id": "intrapulmonary_right_to_left_shunt_observed",
        "aliases": ["肺内右向左分流", "肺内分流", "intrapulmonary right-to-left shunt"],
        "preferred_sections": ["echocardiography", "超声", "右心声学造影"],
        "verification_type": "targeted_exam_observation",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "超声心动图右心声学造影",
        "evidence_level": "observed_imaging",
        "information_value": 0.96,
    },
    {
        "evidence_id": "vascular_pulmonary_nodule_suspected",
        "aliases": ["血管性肺结节", "血管影增粗", "肺内血管性病变"],
        "preferred_sections": ["CT", "imaging", "胸部CT"],
        "verification_type": "support_observation",
        "positive_terms": _COMMON_POSITIVE_TERMS,
        "negative_terms": _COMMON_NEGATIVE_TERMS,
        "followup_exam": "胸部增强CT",
        "evidence_level": "support",
        "information_value": 0.58,
    },
]
