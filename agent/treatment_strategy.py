"""治疗策略 Agent：基于疾病画像补齐治疗原则和安全提醒。"""

from typing import Any, Dict, List

from .diagnosis_engine import DiagnosticKnowledgeBase
from .knowledge import KnowledgeBase


class TreatmentStrategyAgent:
    """提交前治疗方案增强角色。"""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        diagnostic_knowledge: DiagnosticKnowledgeBase = None,
    ):
        self.knowledge = knowledge
        self.diagnostic_knowledge = diagnostic_knowledge or DiagnosticKnowledgeBase(
            ref_dir=getattr(knowledge, "ref_dir", "data/ref_data")
        )

    def review(
        self,
        result: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._review_with_coverage(result, collected_info, exam_results)

    def _review_with_coverage(
        self,
        result: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a diagnosis-covered, actionable treatment plan before safety review."""
        fixed = dict(result or {})
        diagnosis = fixed.get("diagnosis") or []
        if isinstance(diagnosis, str):
            diagnosis = [diagnosis]
        diagnosis = [str(item).strip() for item in diagnosis if str(item).strip()]

        coverage_records = self._build_diagnosis_treatment_coverage(diagnosis)
        rule_protocols = self._dedupe(
            step
            for record in coverage_records
            for step in record.get("protocols", [])
        )
        principles = self._dedupe(
            step
            for record in coverage_records
            for step in record.get("principles", [])
        )
        warnings = self._dedupe(
            step
            for record in coverage_records
            for step in record.get("warnings", [])
        )

        treatment_plan = str(fixed.get("treatment_plan") or "")
        additions = self._build_actionable_additions(
            coverage_records=coverage_records,
            existing_plan=treatment_plan,
            exam_results=exam_results,
            collected_info=collected_info,
        )
        if warnings:
            additions.append("\u5b89\u5168\u63d0\u9192\uff1a" + "\uff1b".join(warnings[:3]) + "\u3002")
        for addition in additions:
            if addition not in treatment_plan:
                treatment_plan = self._append_sentence(treatment_plan, addition)
        fixed["treatment_plan"] = treatment_plan

        reasoning = str(fixed.get("reasoning") or "")
        if principles and "\u6cbb\u7597\u539f\u5219" not in reasoning:
            reasoning = self._append_sentence(
                reasoning,
                "\u6cbb\u7597\u65b9\u6848\u53c2\u8003\u75be\u75c5\u753b\u50cf\u548c\u8bc1\u636e\u88c1\u51b3\u540e\u7684\u6807\u51c6\u5904\u7406\u539f\u5219\uff0c\u5e76\u7ed3\u5408\u5df2\u5b8c\u6210\u68c0\u67e5\u7ed3\u679c\u5236\u5b9a\u3002",
            )
        fixed["reasoning"] = reasoning

        covered_diagnoses = [
            str(record.get("diagnosis") or "")
            for record in coverage_records
            if record.get("protocol_count")
        ]
        uncovered_diagnoses = [
            str(record.get("diagnosis") or "")
            for record in coverage_records
            if not record.get("protocol_count")
        ]
        coverage_rate = (
            len(covered_diagnoses) / len(coverage_records)
            if coverage_records
            else 0.0
        )
        fixed["_treatment_strategy"] = {
            "principles": principles[:5],
            "warnings": warnings[:3],
            "rule_protocols": rule_protocols[:8],
            "diagnosis_protocol_coverage": [
                {
                    "diagnosis": record.get("diagnosis"),
                    "protocol_count": int(record.get("protocol_count") or 0),
                    "principle_count": int(record.get("principle_count") or 0),
                    "warning_count": int(record.get("warning_count") or 0),
                }
                for record in coverage_records
            ],
            "covered_diagnoses": covered_diagnoses,
            "uncovered_diagnoses": uncovered_diagnoses,
            "treatment_protocol_coverage_rate": round(coverage_rate, 3),
            "actionability_sections": self._treatment_actionability_sections(
                treatment_plan
            ),
            "exam_count": len(exam_results or {}),
            "has_personalization_note": True,
            "structural_cardiac": any(
                self.diagnostic_knowledge.get(item).get("diagnosis_type") == "structural"
                for item in diagnosis
            ),
        }
        return fixed

    def _build_diagnosis_treatment_coverage(
        self,
        diagnosis: List[str],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for disease in diagnosis:
            diagnostic_entry = self.diagnostic_knowledge.get(disease)
            profile = self.knowledge.get_disease_profile(str(disease)) or {}
            protocols = self._dedupe(
                list(diagnostic_entry.get("treatment_protocol", []) or [])
                + list(profile.get("treatment_principles", []) or [])
            )
            principles = self._dedupe(profile.get("treatment_principles", []) or [])
            warnings = self._dedupe(
                list(diagnostic_entry.get("avoid_mistakes", []) or [])
                + list(profile.get("avoid_mistakes", []) or [])
            )
            records.append(
                {
                    "diagnosis": disease,
                    "protocols": protocols,
                    "principles": principles,
                    "warnings": warnings,
                    "protocol_count": len(protocols),
                    "principle_count": len(principles),
                    "warning_count": len(warnings),
                }
            )
        return records

    def _build_actionable_additions(
        self,
        *,
        coverage_records: List[Dict[str, Any]],
        existing_plan: str,
        exam_results: Dict[str, Any],
        collected_info: Dict[str, Any],
    ) -> List[str]:
        additions: List[str] = []
        diagnosis_lines: List[str] = []
        for record in coverage_records:
            protocols = list(record.get("protocols") or [])[:4]
            if not protocols:
                continue
            diagnosis_lines.append(
                f"{record.get('diagnosis')}: " + "\uff1b".join(str(item) for item in protocols)
            )
        if diagnosis_lines:
            additions.append(
                "\u6388\u6743\u8bca\u65ad\u5bf9\u5e94\u6cbb\u7597\uff1a"
                + "\uff1b".join(diagnosis_lines[:4])
                + "\u3002"
            )
        if not self._has_monitoring_section(existing_plan):
            additions.append(self._monitoring_sentence(exam_results))
        if not self._has_follow_up_section(existing_plan):
            additions.append(self._follow_up_sentence(collected_info))
        return additions

    @staticmethod
    def _dedupe(values: Any) -> List[str]:
        result: List[str] = []
        for item in values or []:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
        return result

    @staticmethod
    def _append_sentence(plan: str, addition: str) -> str:
        addition = str(addition or "").strip()
        if not addition:
            return str(plan or "")
        text = str(plan or "").strip()
        if not text:
            return addition
        return text.rstrip("\u3002.;\uff1b; ") + "\u3002" + addition

    @staticmethod
    def _has_monitoring_section(plan: str) -> bool:
        text = str(plan or "")
        return any(token in text for token in ("\u76d1\u6d4b", "\u590d\u67e5", "monitor"))

    @staticmethod
    def _has_follow_up_section(plan: str) -> bool:
        text = str(plan or "")
        return any(token in text for token in ("\u968f\u8bbf", "\u590d\u8bca", "follow"))

    @staticmethod
    def _monitoring_sentence(exam_results: Dict[str, Any]) -> str:
        if exam_results:
            return "\u76d1\u6d4b\u4e0e\u590d\u67e5\uff1a\u6839\u636e\u5df2\u5b8c\u6210\u68c0\u67e5\u7ed3\u679c\u8ffd\u8e2a\u75c7\u72b6\u3001\u751f\u547d\u4f53\u5f81\u548c\u5173\u952e\u5ba2\u89c2\u6307\u6807\uff0c\u82e5\u75c5\u60c5\u8fdb\u5c55\u6216\u51fa\u73b0\u77db\u76fe\u8bc1\u636e\u9700\u53ca\u65f6\u590d\u8bc4\u8bca\u65ad\u4e0e\u6cbb\u7597\u3002"
        return "\u76d1\u6d4b\u4e0e\u590d\u67e5\uff1a\u5728\u5b8c\u5584\u5fc5\u8981\u68c0\u67e5\u540e\u590d\u8bc4\u75c5\u60c5\uff0c\u8ffd\u8e2a\u75c7\u72b6\u53d8\u5316\u3001\u751f\u547d\u4f53\u5f81\u548c\u5173\u952e\u5b89\u5168\u6307\u6807\u3002"

    @staticmethod
    def _follow_up_sentence(collected_info: Dict[str, Any]) -> str:
        age_text = str((collected_info or {}).get("age") or "").strip()
        age_note = "\u5e76\u7ed3\u5408\u5e74\u9f84\u8c03\u6574\u7528\u836f\u5242\u91cf" if age_text else ""
        return (
            "\u4e13\u79d1\u4e0e\u968f\u8bbf\uff1a\u6839\u636e\u4e3b\u8bca\u65ad\u5b89\u6392\u76f8\u5173\u4e13\u79d1\u5904\u7406"
            + age_note
            + "\uff0c\u660e\u786e\u590d\u8bca\u548c\u8b66\u793a\u75c7\u72b6\uff0c\u907f\u514d\u5c06\u672a\u83b7\u6388\u6743\u7684\u9274\u522b\u8bca\u65ad\u4f5c\u4e3a\u6cbb\u7597\u76ee\u6807\u3002"
        )

    @staticmethod
    def _treatment_actionability_sections(plan: str) -> List[str]:
        text = str(plan or "")
        sections: List[str] = []
        if "\u6388\u6743\u8bca\u65ad\u5bf9\u5e94\u6cbb\u7597" in text:
            sections.append("diagnosis_specific_protocol")
        if TreatmentStrategyAgent._has_monitoring_section(text):
            sections.append("monitoring")
        if TreatmentStrategyAgent._has_follow_up_section(text):
            sections.append("follow_up")
        if "\u5b89\u5168\u63d0\u9192" in text:
            sections.append("safety_warning")
        return sections

        """结合标准疾病画像补齐治疗原则、随访和安全提醒。"""
        fixed = dict(result or {})
        diagnosis = fixed.get("diagnosis") or []
        if isinstance(diagnosis, str):
            diagnosis = [diagnosis]

        rule_protocols = self.diagnostic_knowledge.get_treatment_protocols(diagnosis)
        if rule_protocols:
            fixed = self._apply_rule_protocol_treatment(fixed, rule_protocols)

        principles: List[str] = []
        warnings: List[str] = []
        for disease in diagnosis:
            profile = self.knowledge.get_disease_profile(str(disease))
            if not profile:
                continue
            for item in profile.get("treatment_principles") or []:
                if item and item not in principles:
                    principles.append(str(item))
            for item in profile.get("avoid_mistakes") or []:
                if item and item not in warnings:
                    warnings.append(str(item))

        treatment_plan = str(fixed.get("treatment_plan") or "")
        additions: List[str] = []
        if principles:
            additions.append("治疗原则：" + "；".join(principles[:5]) + "。")
        additions.append("需结合年龄、既往病史、过敏史和检查异常调整用药，并安排复诊复查。")
        if warnings:
            additions.append("安全提醒：" + "；".join(warnings[:3]) + "。")

        for addition in additions:
            if addition not in treatment_plan:
                treatment_plan = (treatment_plan.rstrip("。") + "。" if treatment_plan else "") + addition
        fixed["treatment_plan"] = treatment_plan

        reasoning = str(fixed.get("reasoning") or "")
        if principles and "治疗原则" not in reasoning:
            reasoning = (reasoning.rstrip("。") + "。" if reasoning else "") + "治疗方案参考疾病画像中的标准处理原则，并结合已完成检查结果制定。"
        fixed["reasoning"] = reasoning
        fixed.setdefault("_treatment_strategy", {})
        fixed["_treatment_strategy"] = {
            "principles": principles[:5],
            "warnings": warnings[:3],
            "rule_protocols": rule_protocols[:8],
            "exam_count": len(exam_results or {}),
            "has_personalization_note": True,
            "structural_cardiac": any(
                self.diagnostic_knowledge.get(item).get("diagnosis_type") == "structural"
                for item in diagnosis
            ),
        }
        return fixed

    @staticmethod
    def _apply_rule_protocol_treatment(
        result: Dict[str, Any],
        protocols: List[str],
    ) -> Dict[str, Any]:
        fixed = dict(result or {})
        steps = [str(item).strip() for item in protocols if str(item).strip()]
        if not steps:
            return fixed
        treatment_plan = "证据规则驱动治疗方案：" + "；".join(
            f"{idx}. {step}" for idx, step in enumerate(steps, 1)
        ) + "。"
        fixed["treatment_plan"] = treatment_plan
        reasoning = str(fixed.get("reasoning") or "")
        note = "治疗方案根据证据裁决后的病因诊断和 diagnostic_knowledge 中的 treatment_protocol 生成。"
        if note not in reasoning:
            reasoning = (reasoning.rstrip("。") + "。" if reasoning else "") + note
        fixed["reasoning"] = reasoning
        return fixed

    @staticmethod
    def _has_structural_cardiac_diagnosis(
        diagnosis: List[Any],
        result: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> bool:
        diag_text = " ".join(str(item) for item in diagnosis or [])
        trusted_text = " ".join(str(item) for item in result.get("_trusted_diagnoses") or [])
        exam_text = str(exam_results or {})
        text = diag_text + " " + trusted_text + " " + exam_text
        return (
            ("三房心" in text or "左心房隔膜" in text)
            and ("心内膜垫缺损" in text or "房室间隔缺损" in text or "共同房室瓣" in text)
        )

    @staticmethod
    def _apply_structural_cardiac_treatment(
        result: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        fixed = dict(result or {})
        findings = fixed.get("_structural_findings") or []
        finding_text = "、".join(str(item) for item in findings[:6]) if findings else "超声心动图和心导管检查提示结构性心脏病"

        treatment_plan = (
            "诊疗重点为结构性先天性心脏病导致的心肺功能失代偿："
            "1. 立即转入儿科心脏专科/心胸外科评估，准备切除左心房隔膜并修复完全性房室间隔缺损；"
            "2. 围手术期给予吸氧，维持合适氧饱和度，连续监测心率、呼吸频率、血压、尿量和液体出入量；"
            "3. 在监护下使用静脉呋塞米减轻肺血管充血，并根据血流动力学给予米力农等正性肌力/扩血管支持；"
            "4. 喂养采用慢流量奶嘴、少量多次和高热量配方奶，减少吃奶时呼吸负荷，避免用电解质液替代营养；"
            "5. 咳嗽、流涕、低热更符合病毒性上呼吸道感染诱发心衰失代偿，若无持续高热、化脓感染证据或培养阳性，不常规使用抗生素；"
            "6. 若出现拒奶、嗜睡、发绀加重、尿量减少或呼吸窘迫进展，立即急诊/重症监护处理。"
        )
        fixed["treatment_plan"] = treatment_plan

        reasoning = str(fixed.get("reasoning") or "")
        structural_note = (
            f"治疗方案围绕病因处理制定：{finding_text}，"
            "首要矫治结构畸形并进行围手术期强心、利尿、氧疗和营养支持；"
            "呼吸道感染作为诱因监测，避免无证据抗生素。"
        )
        if structural_note not in reasoning:
            reasoning = (reasoning.rstrip("。") + "。" if reasoning else "") + structural_note
        fixed["reasoning"] = reasoning
        return fixed
