"""Evidence-driven diagnosis adjudication.

This module turns raw examination JSON/text into structured clinical evidence,
scores data-driven diagnostic rules, and promotes high-confidence etiologic
diagnoses before final submission.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    """Normalized evidence extracted from patient info or examinations."""

    finding: str
    source_exam: str
    polarity: str = "positive"
    severity: str = ""
    value: Optional[float] = None
    unit: str = ""
    confidence: float = 0.8
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data.get("value") is None:
            data.pop("value", None)
        return data


class EvidenceDiagnosisEngine:
    """Extract evidence, score diagnostic rules, and adjudicate final diagnosis."""

    def __init__(self, ref_dir: str = "data/ref_data"):
        self.ref_dir = ref_dir
        self.rules_path = os.path.join(ref_dir, "diagnostic_rules.json")
        self.pending_rules_path = os.path.join(
            "outputs",
            "runtime_state",
            "pending_diagnostic_rules.json",
        )
        self.rules = self._load_rules()

    def review(
        self,
        result: Dict[str, Any],
        collected_info: Optional[Dict[str, Any]] = None,
        exam_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply evidence adjudication to an LLM result."""
        fixed = dict(result or {}) if isinstance(result, dict) else {}
        collected_info = collected_info or {}
        exam_results = exam_results or {}

        evidence = self.extract_evidence(collected_info, exam_results)
        scores = self.score_rules(evidence, collected_info)
        trusted = self.select_trusted_diagnoses(scores)
        fixed["_evidence_items"] = [item.to_dict() for item in evidence]
        fixed["_evidence_scores"] = scores

        if not trusted:
            return fixed

        original = fixed.get("diagnosis") or fixed.get("diagnoses") or []
        if isinstance(original, str):
            original = [original]
        original = [str(item) for item in original if str(item).strip()]

        suppress = set()
        for diagnosis in trusted:
            rule = self._rule_by_diagnosis(diagnosis)
            suppress.update(str(item) for item in rule.get("suppress_diagnoses", []) if item)

        suppressed = [
            item
            for item in original
            if item not in trusted and item in suppress
        ]
        preserved = [
            item
            for item in original
            if item not in trusted and item not in suppress
        ]
        final_diagnoses = list(dict.fromkeys(trusted + preserved))[:4]

        fixed["diagnosis"] = final_diagnoses
        fixed["_trusted_diagnoses"] = trusted
        fixed["_suppressed_diagnoses"] = list(dict.fromkeys(suppressed))
        fixed["_evidence_reasoning"] = self._build_evidence_reasoning(trusted, scores)

        reasoning = str(fixed.get("reasoning") or "")
        if fixed["_evidence_reasoning"] and fixed["_evidence_reasoning"] not in reasoning:
            reasoning = (reasoning.rstrip("。") + "。" if reasoning else "") + fixed["_evidence_reasoning"]
        if fixed["_suppressed_diagnoses"]:
            note = (
                "证据裁决层认为上述强证据病因诊断优先；"
                f"{'、'.join(fixed['_suppressed_diagnoses'])}作为表现、并发状态或鉴别诊断记录。"
            )
            if note not in reasoning:
                reasoning = (reasoning.rstrip("。") + "。" if reasoning else "") + note
        fixed["reasoning"] = reasoning
        return fixed

    def extract_evidence(
        self,
        collected_info: Optional[Dict[str, Any]] = None,
        exam_results: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        """Extract structured evidence from symptoms and examination results."""
        evidence: List[EvidenceItem] = []
        collected_info = collected_info or {}
        exam_results = exam_results or {}

        evidence.extend(self._extract_symptom_evidence(collected_info))
        for source_exam, payload in exam_results.items():
            text = self._payload_text(payload)
            source = str(source_exam)
            evidence.extend(self._extract_cardiac_evidence(source, text))
            evidence.extend(self._extract_urinary_evidence(source, text))
            evidence.extend(self._extract_trauma_evidence(source, text))
            evidence.extend(self._extract_eye_evidence(source, text))
            evidence.extend(self._extract_infection_immune_evidence(source, text))
            evidence.extend(self._extract_congenital_cardiac_evidence(source, text))
            evidence.extend(self._extract_legacy_structural_evidence(source, text))

        return self._dedupe_evidence(evidence)

    def score_rules(
        self,
        evidence: List[EvidenceItem],
        collected_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Score each diagnostic rule against extracted evidence."""
        index = self._evidence_index(evidence)
        scores: Dict[str, Dict[str, Any]] = {}
        for rule in self.rules:
            diagnosis = str(rule.get("diagnosis") or "").strip()
            if not diagnosis:
                continue
            positive_hits, positive_score = self._score_evidence_list(
                rule.get("positive_evidence", []), index
            )
            negative_hits, negative_score = self._score_evidence_list(
                rule.get("negative_evidence", []), index
            )
            raw_score = max(0.0, positive_score - negative_score)
            required_any_ok = self._required_any_ok(rule.get("required_any", []), index)
            required_all_ok = self._required_all_ok(rule.get("required_all", []), index)
            threshold = float(rule.get("score_threshold", 1.0) or 1.0)
            passed = raw_score >= threshold and required_any_ok and required_all_ok
            scores[diagnosis] = {
                "diagnosis": diagnosis,
                "diagnosis_type": rule.get("diagnosis_type", ""),
                "score": round(raw_score, 4),
                "threshold": threshold,
                "passed": passed,
                "positive_hits": positive_hits,
                "negative_hits": negative_hits,
                "required_any_ok": required_any_ok,
                "required_all_ok": required_all_ok,
                "priority": int(rule.get("priority", 50) or 50),
            }
        return scores

    def select_trusted_diagnoses(self, scores: Dict[str, Dict[str, Any]]) -> List[str]:
        """Select high-confidence etiologic diagnoses from rule scores."""
        passed = [item for item in scores.values() if item.get("passed")]
        if not passed:
            return []
        type_rank = {
            "etiology": 5,
            "structural": 4,
            "syndrome": 3,
            "infection": 2,
            "state": 1,
        }
        passed.sort(
            key=lambda item: (
                type_rank.get(str(item.get("diagnosis_type")), 0),
                int(item.get("priority", 0)),
                float(item.get("score", 0.0)),
            ),
            reverse=True,
        )

        selected: List[str] = []
        suppressed = set()
        for item in passed:
            diagnosis = str(item.get("diagnosis") or "")
            if not diagnosis or diagnosis in suppressed:
                continue
            selected.append(diagnosis)
            rule = self._rule_by_diagnosis(diagnosis)
            suppressed.update(str(x) for x in rule.get("suppress_diagnoses", []) if x)
        return selected[:4]

    def get_treatment_protocols(self, diagnoses: Iterable[Any]) -> List[str]:
        """Return unique treatment protocol steps for diagnoses."""
        protocols: List[str] = []
        for diagnosis in diagnoses or []:
            rule = self._rule_by_diagnosis(str(diagnosis))
            for item in rule.get("treatment_protocol", []) or []:
                text = str(item).strip()
                if text and text not in protocols:
                    protocols.append(text)
        return protocols

    def record_feedback(
        self,
        patient_id: str,
        report: Dict[str, Any],
        collected_info: Optional[Dict[str, Any]] = None,
        exam_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """Write low-score evaluation feedback as pending diagnostic rule evidence."""
        if not isinstance(report, dict):
            return {"pending": 0}
        detail = report.get("diagnosisDetail") or report.get("diagnosis_detail") or {}
        if not isinstance(detail, dict):
            detail = {}
        expected = self._as_text_list(detail.get("expected") or report.get("finalDiagnosis"))
        submitted = self._as_text_list(detail.get("submitted") or report.get("diagnosis"))
        matched = self._as_text_list(detail.get("matched"))
        if not expected:
            return {"pending": 0}
        missing = [item for item in expected if item not in matched]
        if not missing:
            return {"pending": 0}

        data = self._read_json(self.pending_rules_path, {"candidates": []})
        if not isinstance(data, dict):
            data = {"candidates": []}
        candidates = data.setdefault("candidates", [])
        evidence = [item.to_dict() for item in self.extract_evidence(collected_info, exam_results)]
        now = datetime.now(timezone.utc).isoformat()
        added = 0
        for diagnosis in missing:
            candidates.append(
                {
                    "id": "diag_rule_" + uuid.uuid4().hex[:10],
                    "status": "pending",
                    "patient_id": patient_id,
                    "diagnosis": diagnosis,
                    "submitted": submitted,
                    "expected": expected,
                    "evidence_items": evidence,
                    "examination_detail": report.get("examinationDetail") or {},
                    "treatment_detail": report.get("treatmentDetail") or {},
                    "created_at": now,
                }
            )
            added += 1
        self._write_json(self.pending_rules_path, data)
        return {"pending": added}

    def _extract_symptom_evidence(self, collected_info: Dict[str, Any]) -> List[EvidenceItem]:
        text = self._payload_text(collected_info)
        mappings = [
            ("urinary_urgency", ("尿急",)),
            ("urinary_frequency", ("尿频",)),
            ("dysuria", ("尿痛", "排尿烧灼", "烧灼感")),
            ("suprapubic_discomfort", ("耻骨上", "下腹胀", "膀胱区不适")),
            ("dyspnea", ("气短", "呼吸困难", "喘不上气", "气促")),
            ("leg_edema", ("下肢水肿", "腿肿", "水肿")),
            ("palpitation", ("心慌", "心悸")),
            ("chest_wall_pain", ("胸壁疼痛", "肋部疼痛", "肋骨疼痛", "左侧胸壁")),
            ("trauma_history", ("外伤", "夹伤", "撞伤", "摔伤", "车门夹")),
            ("pleuritic_pain", ("深呼吸", "咳嗽时加重", "转身时加重")),
            ("leukocoria", ("瞳孔发白", "白瞳", "猫眼反光", "白色反光")),
            ("strabismus", ("内斜视", "斜视")),
            ("eye_rubbing_irritability", ("揉眼", "烦躁")),
            ("prolonged_respiratory_infection", ("6周", "六周", "迁延", "反复发热", "反复咳嗽")),
            ("purulent_sputum", ("脓痰", "浓痰", "黄痰")),
            ("otalgia", ("耳痛", "耳朵痛")),
            ("wheezing", ("喘息", "喘鸣")),
            ("recurrent_pneumonia", ("反复肺炎", "两次肺炎", "多次肺炎")),
            ("exertional_dyspnea", ("活动后喘", "活动后气短", "玩耍时", "运动后", "跑几步")),
            ("pediatric_caregiver", ("孩子", "患儿", "宝宝", "家长", "照护者")),
            ("rheumatoid_history", ("类风湿", "晨僵", "关节肿痛")),
            ("immunosuppression", ("糖皮质激素", "激素", "免疫抑制", "免疫抑制剂")),
        ]
        evidence = []
        for finding, tokens in mappings:
            if any(token in text for token in tokens):
                evidence.append(
                    EvidenceItem(
                        finding=finding,
                        source_exam="问诊",
                        confidence=0.75,
                        text=self._snippet(text, tokens),
                    )
                )
        if (
            any(token in text for token in ("胸壁疼痛", "肋部疼痛", "左侧胸壁"))
            and any(token in text for token in ("夹伤", "撞伤", "摔伤", "车门", "外伤"))
            and any(token in text for token in ("深呼吸", "咳嗽", "转身"))
        ):
            evidence.append(
                EvidenceItem(
                    finding="chest_wall_trauma_pattern",
                    source_exam="问诊",
                    confidence=0.86,
                    text=self._snippet(text, ("胸壁", "夹伤", "深呼吸")),
                )
            )
        if (
            any(token in text for token in ("反复发热", "发热", "咳嗽"))
            and any(token in text for token in ("6周", "六周", "迁延"))
            and any(token in text for token in ("耳痛", "脓痰", "浓痰", "喘息"))
        ):
            evidence.append(
                EvidenceItem(
                    finding="hib_clinical_pattern",
                    source_exam="问诊",
                    confidence=0.86,
                    text=self._snippet(text, ("发热", "咳嗽", "耳痛", "脓痰")),
                )
            )
        if (
            any(token in text for token in ("反复肺炎", "两次肺炎", "多次肺炎"))
            and any(token in text for token in ("活动后", "玩耍", "运动", "心悸", "喘息", "乏力"))
        ):
            evidence.append(
                EvidenceItem(
                    finding="congenital_shunt_pattern",
                    source_exam="问诊",
                    confidence=0.82,
                    text=self._snippet(text, ("反复肺炎", "活动", "心悸", "喘息")),
                )
            )
        return evidence

    def _extract_cardiac_evidence(self, source: str, text: str) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        if "三尖瓣反流" in text:
            severity = self._severity(text)
            confidence = 0.96 if severity in ("重度", "中重度") else 0.86
            evidence.append(
                EvidenceItem(
                    finding="tricuspid_regurgitation",
                    source_exam=source,
                    severity=severity,
                    confidence=confidence,
                    text=self._snippet(text, ("三尖瓣反流",)),
                )
            )
        if "肺动脉瓣狭窄" in text:
            evidence.append(
                EvidenceItem(
                    finding="pulmonary_valve_stenosis",
                    source_exam=source,
                    confidence=0.95,
                    text=self._snippet(text, ("肺动脉瓣狭窄",)),
                )
            )
        gradient = self._extract_gradient(text)
        if gradient is not None:
            evidence.append(
                EvidenceItem(
                    finding="pulmonary_valve_gradient",
                    source_exam=source,
                    severity="显著升高" if gradient >= 40 else "升高",
                    value=gradient,
                    unit="mmHg",
                    confidence=0.95 if gradient >= 40 else 0.8,
                    text=self._snippet(text, ("肺动脉瓣", "峰值压差", "压差")),
                )
            )
        if any(token in text for token in ("右心房增大", "右心室扩大", "右心室扩张", "右心扩大", "RVD")):
            evidence.append(
                EvidenceItem(
                    finding="right_heart_enlargement",
                    source_exam=source,
                    confidence=0.85,
                    text=self._snippet(text, ("右心房", "右心室", "RVD")),
                )
            )
        if any(token in text for token in ("肺动脉高压", "肺动脉压升高", "PASP", "肺动脉收缩压")):
            evidence.append(
                EvidenceItem(
                    finding="pulmonary_hypertension",
                    source_exam=source,
                    confidence=0.82,
                    text=self._snippet(text, ("肺动脉高压", "PASP", "肺动脉收缩压")),
                )
            )
        return evidence

    def _extract_urinary_evidence(self, source: str, text: str) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        if "逼尿肌过度活动" in text or ("逼尿肌" in text and "过度活动" in text):
            evidence.append(
                EvidenceItem(
                    finding="detrusor_overactivity",
                    source_exam=source,
                    confidence=0.96,
                    text=self._snippet(text, ("逼尿肌", "过度活动")),
                )
            )
        if "残余尿" in text and any(token in text for token in ("正常", "不多", "未见明显残余")):
            evidence.append(
                EvidenceItem(
                    finding="normal_postvoid_residual",
                    source_exam=source,
                    confidence=0.78,
                    text=self._snippet(text, ("残余尿",)),
                )
            )
        if "尿培养" in source or "尿培养" in text:
            if any(token in text for token in ("无生长", "未见细菌", "阴性", "未培养出")):
                evidence.append(
                    EvidenceItem(
                        finding="urine_culture_no_growth",
                        source_exam=source,
                        polarity="negative",
                        confidence=0.92,
                        text=self._snippet(text, ("尿培养", "无生长", "阴性")),
                    )
                )
            if any(token in text for token in ("阳性", "生长", "检出")) and "无生长" not in text:
                evidence.append(
                    EvidenceItem(
                        finding="urine_culture_positive",
                        source_exam=source,
                        confidence=0.92,
                        text=self._snippet(text, ("尿培养", "阳性", "生长")),
                    )
                )
        if "白细胞酯酶" in text:
            evidence.append(self._binary_urine_marker("leukocyte_esterase", source, text, "白细胞酯酶"))
        if "亚硝酸盐" in text:
            evidence.append(self._binary_urine_marker("nitrite", source, text, "亚硝酸盐"))
        if any(token in text for token in ("尿白细胞", "白细胞计数")):
            if any(token in text for token in ("0-5", "0～5", "阴性", "正常")):
                evidence.append(
                    EvidenceItem(
                        finding="urine_wbc_normal",
                        source_exam=source,
                        polarity="negative",
                        confidence=0.78,
                        text=self._snippet(text, ("尿白细胞", "白细胞计数")),
                    )
                )
        return evidence

    def _extract_trauma_evidence(self, source: str, text: str) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        if any(token in text for token in ("肋骨骨折", "肋骨皮质连续性中断", "肋骨线样透亮影")):
            evidence.append(
                EvidenceItem(
                    finding="rib_fracture_imaging",
                    source_exam=source,
                    confidence=0.96,
                    text=self._snippet(text, ("肋骨骨折", "肋骨", "骨折")),
                )
            )
        elif "骨折" in text and any(token in text for token in ("肋", "胸壁", "胸廓")):
            evidence.append(
                EvidenceItem(
                    finding="rib_fracture_imaging",
                    source_exam=source,
                    confidence=0.88,
                    text=self._snippet(text, ("骨折", "肋", "胸壁")),
                )
            )
        if any(token in text for token in ("气胸", "胸腔积液", "肺挫伤")):
            evidence.append(
                EvidenceItem(
                    finding="thoracic_trauma_complication",
                    source_exam=source,
                    confidence=0.82,
                    text=self._snippet(text, ("气胸", "胸腔积液", "肺挫伤")),
                )
            )
        return evidence

    def _extract_eye_evidence(self, source: str, text: str) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        if any(token in text for token in ("视网膜母细胞瘤", "眼内肿瘤", "视网膜肿瘤", "眼球内肿块")):
            evidence.append(
                EvidenceItem(
                    finding="eye_tumor_imaging",
                    source_exam=source,
                    confidence=0.97,
                    text=self._snippet(text, ("视网膜母细胞瘤", "眼内肿瘤", "眼球内肿块")),
                )
            )
        if any(token in text for token in ("视神经受累", "视神经侵犯", "筛板后视神经")):
            evidence.append(
                EvidenceItem(
                    finding="optic_nerve_involvement",
                    source_exam=source,
                    confidence=0.9,
                    text=self._snippet(text, ("视神经", "筛板后")),
                )
            )
        if any(token in text for token in ("钙化", "眼内钙化", "肿块内钙化")) and any(
            token in text for token in ("眼", "视网膜", "眶")
        ):
            evidence.append(
                EvidenceItem(
                    finding="intraocular_calcification",
                    source_exam=source,
                    confidence=0.82,
                    text=self._snippet(text, ("钙化", "视网膜", "眼")),
                )
            )
        return evidence

    def _extract_infection_immune_evidence(self, source: str, text: str) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        if any(token in text for token in ("流感嗜血杆菌", "Hib", "Haemophilus")):
            evidence.append(
                EvidenceItem(
                    finding="hib_pathogen_detected",
                    source_exam=source,
                    confidence=0.98,
                    text=self._snippet(text, ("流感嗜血杆菌", "Hib", "Haemophilus")),
                )
            )
        if any(token in text for token in ("肺炎", "实变", "浸润影", "支气管肺炎")):
            evidence.append(
                EvidenceItem(
                    finding="pneumonia_imaging",
                    source_exam=source,
                    confidence=0.78,
                    text=self._snippet(text, ("肺炎", "实变", "浸润影")),
                )
            )
        if any(token in text for token in ("中性粒细胞减少", "粒细胞减少", "白细胞减少", "白细胞计数降低")):
            evidence.append(
                EvidenceItem(
                    finding="neutropenia",
                    source_exam=source,
                    confidence=0.9,
                    text=self._snippet(text, ("中性粒细胞", "粒细胞减少", "白细胞")),
                )
            )
        if any(token in text for token in ("脾肿大", "脾大", "脾脏增大")):
            evidence.append(
                EvidenceItem(
                    finding="splenomegaly",
                    source_exam=source,
                    confidence=0.9,
                    text=self._snippet(text, ("脾肿大", "脾大", "脾脏")),
                )
            )
        if any(token in text for token in ("类风湿因子阳性", "RF阳性", "类风湿关节炎")):
            evidence.append(
                EvidenceItem(
                    finding="rheumatoid_history",
                    source_exam=source,
                    confidence=0.86,
                    text=self._snippet(text, ("类风湿", "RF", "风湿因子")),
                )
            )
        return evidence

    def _extract_congenital_cardiac_evidence(self, source: str, text: str) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        negative_asd = any(
            token in text for token in ("无 ASD", "无ASD", "未见ASD", "未见 ASD", "房间隔完整")
        )
        strong_positive_asd = any(
            token in text for token in ("房间隔缺损", "继发孔型", "继发孔房缺")
        )
        if strong_positive_asd or ("ASD" in text and not negative_asd):
            evidence.append(
                EvidenceItem(
                    finding="atrial_septal_defect",
                    source_exam=source,
                    confidence=0.96,
                    text=self._snippet(text, ("房间隔缺损", "ASD", "继发孔型")),
                )
            )
        if any(token in text for token in ("左向右分流", "左至右分流", "双向分流")):
            evidence.append(
                EvidenceItem(
                    finding="left_to_right_shunt",
                    source_exam=source,
                    confidence=0.86,
                    text=self._snippet(text, ("左向右分流", "左至右分流", "分流")),
                )
            )
        qpqs = self._extract_qpqs(text)
        if qpqs is not None:
            evidence.append(
                EvidenceItem(
                    finding="qpqs_elevated",
                    source_exam=source,
                    severity="显著升高" if qpqs >= 1.5 else "升高",
                    value=qpqs,
                    unit=":1",
                    confidence=0.9 if qpqs >= 1.5 else 0.75,
                    text=self._snippet(text, ("Qp:Qs", "肺体循环血流比")),
                )
            )
        return evidence

    def _extract_legacy_structural_evidence(self, source: str, text: str) -> List[EvidenceItem]:
        mappings = [
            ("left_atrial_membrane", ("左心房隔膜", "左房隔膜", "三房心")),
            ("restrictive_fenestration", ("限制性开窗",)),
            ("atrioventricular_septal_defect", ("完全性房室间隔缺损", "房室间隔缺损", "心内膜垫缺损")),
            ("common_atrioventricular_valve", ("共同房室瓣", "房室瓣反流")),
        ]
        evidence: List[EvidenceItem] = []
        for finding, tokens in mappings:
            if any(token in text for token in tokens):
                evidence.append(
                    EvidenceItem(
                        finding=finding,
                        source_exam=source,
                        confidence=0.92,
                        text=self._snippet(text, tokens),
                    )
                )
        return evidence

    @staticmethod
    def _binary_urine_marker(finding_prefix: str, source: str, text: str, marker: str) -> EvidenceItem:
        snippet = EvidenceDiagnosisEngine._snippet(text, (marker,))
        if "阳性" in snippet or "+" in snippet:
            return EvidenceItem(
                finding=f"{finding_prefix}_positive",
                source_exam=source,
                confidence=0.86,
                text=snippet,
            )
        return EvidenceItem(
            finding=f"{finding_prefix}_negative",
            source_exam=source,
            polarity="negative",
            confidence=0.86,
            text=snippet,
        )

    @staticmethod
    def _score_evidence_list(
        specs: Any,
        index: Dict[str, EvidenceItem],
    ) -> Tuple[List[Dict[str, Any]], float]:
        hits: List[Dict[str, Any]] = []
        score = 0.0
        if not isinstance(specs, list):
            return hits, score
        for spec in specs:
            if isinstance(spec, str):
                finding = spec
                weight = 1.0
            elif isinstance(spec, dict):
                finding = str(spec.get("finding") or "").strip()
                weight = float(spec.get("weight", 1.0) or 1.0)
            else:
                continue
            item = index.get(finding)
            if not item:
                continue
            contribution = weight * float(item.confidence or 0.0)
            score += contribution
            hits.append(
                {
                    "finding": finding,
                    "source_exam": item.source_exam,
                    "severity": item.severity,
                    "value": item.value,
                    "unit": item.unit,
                    "confidence": item.confidence,
                    "weight": weight,
                    "contribution": round(contribution, 4),
                }
            )
        return hits, score

    @staticmethod
    def _required_any_ok(required: Any, index: Dict[str, EvidenceItem]) -> bool:
        items = [str(item) for item in required or [] if str(item).strip()]
        return True if not items else any(item in index for item in items)

    @staticmethod
    def _required_all_ok(required: Any, index: Dict[str, EvidenceItem]) -> bool:
        items = [str(item) for item in required or [] if str(item).strip()]
        return all(item in index for item in items)

    @staticmethod
    def _evidence_index(evidence: List[EvidenceItem]) -> Dict[str, EvidenceItem]:
        index: Dict[str, EvidenceItem] = {}
        for item in evidence:
            old = index.get(item.finding)
            if old is None or item.confidence > old.confidence:
                index[item.finding] = item
        return index

    @staticmethod
    def _dedupe_evidence(evidence: List[EvidenceItem]) -> List[EvidenceItem]:
        merged: Dict[Tuple[str, str, str], EvidenceItem] = {}
        for item in evidence:
            key = (item.finding, item.source_exam, item.polarity)
            old = merged.get(key)
            if old is None or item.confidence > old.confidence:
                merged[key] = item
        return list(merged.values())

    def _rule_by_diagnosis(self, diagnosis: str) -> Dict[str, Any]:
        for rule in self.rules:
            if str(rule.get("diagnosis") or "") == diagnosis:
                return rule
        return {}

    def _build_evidence_reasoning(
        self,
        trusted: List[str],
        scores: Dict[str, Dict[str, Any]],
    ) -> str:
        parts = []
        for diagnosis in trusted:
            score = scores.get(diagnosis, {})
            hits = score.get("positive_hits", []) or []
            hit_text = []
            for hit in hits[:4]:
                label = str(hit.get("finding") or "")
                source = str(hit.get("source_exam") or "")
                value = hit.get("value")
                unit = hit.get("unit") or ""
                if value is not None:
                    label = f"{label}={value}{unit}"
                if source:
                    label = f"{source}:{label}"
                hit_text.append(label)
            if hit_text:
                parts.append(f"{diagnosis}由{ '、'.join(hit_text) }支持")
            else:
                parts.append(f"{diagnosis}达到证据规则阈值")
        if not parts:
            return ""
        return "证据裁决层基于检查结果优先提交病因/结构诊断：" + "；".join(parts) + "。"

    @staticmethod
    def _severity(text: str) -> str:
        for token in ("极重度", "重度", "中重度", "中度", "轻度"):
            if token in text:
                return token
        return ""

    @staticmethod
    def _extract_gradient(text: str) -> Optional[float]:
        if "肺动脉瓣" not in text and "跨肺动脉瓣" not in text:
            return None
        patterns = [
            r"(?:峰值压差|压差|跨瓣压差)[^\d]{0,12}(\d+(?:\.\d+)?)\s*(?:mmHg|毫米汞柱)?",
            r"(\d+(?:\.\d+)?)\s*(?:mmHg|毫米汞柱)[^\n\r。；;]{0,12}(?:峰值压差|压差|跨瓣压差)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _extract_qpqs(text: str) -> Optional[float]:
        patterns = [
            r"Qp\s*[:：/]\s*Qs[^\d]{0,12}(\d+(?:\.\d+)?)\s*[:：]?\s*1?",
            r"肺体循环血流比[^\d]{0,12}(\d+(?:\.\d+)?)\s*[:：]?\s*1?",
            r"(\d+(?:\.\d+)?)\s*[:：]\s*1[^\n\r。；;]{0,12}(?:Qp|肺体循环)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _payload_text(payload: Any) -> str:
        if payload is None:
            return ""
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(payload)

    @staticmethod
    def _snippet(text: str, tokens: Iterable[str], width: int = 90) -> str:
        if not text:
            return ""
        positions = [text.find(token) for token in tokens if token and text.find(token) >= 0]
        if not positions:
            return text[:width]
        start = max(0, min(positions) - width // 3)
        end = min(len(text), start + width)
        return text[start:end]

    @staticmethod
    def _as_text_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            result: List[str] = []
            for item in value:
                result.extend(EvidenceDiagnosisEngine._as_text_list(item))
            return list(dict.fromkeys(item for item in result if item))
        return [str(value)] if str(value).strip() else []

    def _load_rules(self) -> List[Dict[str, Any]]:
        data = self._read_json(self.rules_path, {"rules": []})
        if isinstance(data, dict) and isinstance(data.get("rules"), list):
            return [rule for rule in data["rules"] if isinstance(rule, dict)]
        return []

    @staticmethod
    def _read_json(path: str, default: Any) -> Any:
        try:
            if not os.path.exists(path):
                return default
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("[Evidence] failed to read %s: %s", path, exc)
            return default

    @staticmethod
    def _write_json(path: str, data: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
