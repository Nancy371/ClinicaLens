"""Structural diagnosis rules derived from objective examination results."""

import json
from typing import Any, Dict, List


class StructuralDiagnosisAgent:
    """Promote concrete structural disease names when exams directly support them."""

    STRUCTURAL_PRIORITY = [
        "三房心",
        "心内膜垫缺损",
    ]

    def review(
        self,
        result: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        fixed = dict(result or {}) if isinstance(result, dict) else {}
        trusted = self.detect_trusted_diagnoses(exam_results)
        if not trusted:
            return fixed

        original = fixed.get("diagnosis") or []
        if isinstance(original, str):
            original = [original]

        # Use concrete etiologic diagnoses as submitted diagnoses; keep syndromes
        # and triggers in reasoning instead of letting them displace the cause.
        fixed["diagnosis"] = trusted
        fixed["_trusted_diagnoses"] = trusted
        fixed["_structural_findings"] = self.extract_key_findings(exam_results)
        fixed["_suppressed_diagnoses"] = [
            str(item)
            for item in original
            if str(item) and str(item) not in trusted
        ]

        reasoning = str(fixed.get("reasoning") or "")
        evidence = self._structural_reasoning_text(trusted, fixed["_structural_findings"])
        if evidence and evidence not in reasoning:
            reasoning = (reasoning.rstrip("。") + "。" if reasoning else "") + evidence
        if fixed["_suppressed_diagnoses"]:
            note = "心力衰竭、上呼吸道感染等作为并发状态或诱因记录在治疗依据中，不作为首要病因诊断提交。"
            if note not in reasoning:
                reasoning = (reasoning.rstrip("。") + "。" if reasoning else "") + note
        fixed["reasoning"] = reasoning
        return fixed

    def detect_trusted_diagnoses(self, exam_results: Dict[str, Any]) -> List[str]:
        text = self._flatten_exam_text(exam_results)
        diagnoses: List[str] = []

        if any(token in text for token in ("左心房隔膜", "左房隔膜", "三房心")):
            diagnoses.append("三房心")

        if any(
            token in text
            for token in (
                "完全性房室间隔缺损",
                "房室间隔缺损",
                "心内膜垫缺损",
                "共同房室瓣",
                "房室瓣反流",
            )
        ):
            diagnoses.append("心内膜垫缺损")

        if "限制性开窗" in text and "三房心" not in diagnoses:
            diagnoses.append("三房心")

        return [item for item in self.STRUCTURAL_PRIORITY if item in diagnoses]

    def extract_key_findings(self, exam_results: Dict[str, Any]) -> List[str]:
        text = self._flatten_exam_text(exam_results)
        findings = []
        for token in (
            "左心房隔膜",
            "限制性开窗",
            "完全性房室间隔缺损",
            "共同房室瓣反流",
            "右心扩张",
            "肺动脉高压",
            "肺血管充血",
            "肺门周围间质性水肿",
            "心脏增大",
        ):
            if token in text:
                findings.append(token)
        return findings

    @staticmethod
    def _flatten_exam_text(exam_results: Dict[str, Any]) -> str:
        if not exam_results:
            return ""
        try:
            return json.dumps(exam_results, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(exam_results)

    @staticmethod
    def _structural_reasoning_text(diagnoses: List[str], findings: List[str]) -> str:
        if not diagnoses:
            return ""
        finding_text = "、".join(findings[:8]) if findings else "结构性心脏异常"
        return (
            "客观检查结果直接支持结构性心脏病病因诊断："
            f"{finding_text}，因此优先提交{ '、'.join(diagnoses) }。"
        )
