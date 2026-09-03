"""Deterministic treatment safety checks applied after diagnosis adjudication."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Set

from .diagnosis_engine import DiagnosticKnowledgeBase


_MEDICATION_ALIASES = {
    "阿司匹林": ("阿司匹林", "水杨酸盐", "aspirin"),
    "青霉素": ("青霉素", "阿莫西林", "氨苄西林", "哌拉西林"),
    "头孢菌素": ("头孢", "ceftriaxone", "头孢曲松"),
    "磺胺类": ("磺胺", "复方新诺明", "甲氧苄啶"),
}
_RENAL_RISK_TERMS = ("布洛芬", "双氯芬酸", "吲哚美辛", "NSAID", "二甲双胍")
_POTASSIUM_RISK_TERMS = ("氯化钾", "补钾", "螺内酯")
_PREGNANCY_RISK_TERMS = ("ACEI", "ARB", "依那普利", "贝那普利", "缬沙坦", "华法林")
_YOUNG_CHILD_RISK_TERMS = ("四环素", "多西环素")


class TreatmentSafetyGate:
    """Remove protocol steps that conflict with explicit patient constraints."""

    def __init__(self, knowledge: DiagnosticKnowledgeBase):
        self.knowledge = knowledge

    def review(
        self,
        result: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        fixed = dict(result or {})
        diagnoses = fixed.get("diagnosis") or []
        if isinstance(diagnoses, str):
            diagnoses = [diagnoses]
        text = json.dumps(collected_info or {}, ensure_ascii=False)
        blocked = self._explicit_blocked_terms(text)
        blocked.update(self._knowledge_blocked_terms(diagnoses, collected_info))
        blocked.update(self._patient_blocked_terms(collected_info))
        blocked.update(self._exam_blocked_terms(exam_results))

        plan = str(fixed.get("treatment_plan") or "")
        segments = [item.strip() for item in re.split(r"(?<=[。；;])", plan) if item.strip()]
        removed: List[str] = []
        kept: List[str] = []
        for segment in segments:
            conflict = next((term for term in blocked if term and term.lower() in segment.lower()), None)
            if conflict:
                removed.append(f"{conflict}:{segment[:120]}")
            else:
                kept.append(segment)

        if removed:
            plan = "".join(kept).strip()
            if not plan:
                plan = "当前治疗需由相应专科结合年龄、过敏史、禁忌证及检查结果制定，避免使用已识别的禁忌药物。"
            warning = "治疗安全门控已移除与患者禁忌证冲突的方案，需由临床团队选择替代治疗。"
            if warning not in plan:
                plan = plan.rstrip("。") + "。" + warning
            fixed["treatment_plan"] = plan

        fixed["_treatment_safety"] = {
            "blocked_terms": sorted(blocked),
            "removed_segments": removed,
            "safe": not removed,
            "exam_count": len(exam_results or {}),
        }
        return fixed

    @staticmethod
    def _explicit_blocked_terms(text: str) -> Set[str]:
        blocked: Set[str] = set()
        lowered = text.lower()
        for canonical, aliases in _MEDICATION_ALIASES.items():
            for alias in aliases:
                index = lowered.find(alias.lower())
                while index >= 0:
                    window = lowered[max(0, index - 24):index + len(alias) + 24]
                    if any(token in window for token in ("过敏", "禁忌", "不能使用", "避免使用")):
                        blocked.update(aliases)
                        blocked.add(canonical)
                        break
                    index = lowered.find(alias.lower(), index + len(alias))
        return blocked

    @staticmethod
    def _patient_blocked_terms(collected_info: Dict[str, Any]) -> Set[str]:
        blocked: Set[str] = set()
        age = _age_number((collected_info or {}).get("age"))
        text = json.dumps(collected_info or {}, ensure_ascii=False)
        if age is not None and age < 8:
            blocked.update(_YOUNG_CHILD_RISK_TERMS)
        if any(token in text for token in ("妊娠", "怀孕", "孕妇")):
            blocked.update(_PREGNANCY_RISK_TERMS)
        return blocked

    @staticmethod
    def _exam_blocked_terms(exam_results: Dict[str, Any]) -> Set[str]:
        blocked: Set[str] = set()
        leaves = " ".join(_leaf_texts(exam_results or {}))
        if any(
            token in leaves
            for token in ("严重肾功能不全", "肾功能衰竭", "肾衰竭", "eGFR<30", "eGFR <30")
        ):
            blocked.update(_RENAL_RISK_TERMS)
        if "高钾血症" in leaves or (
            "血钾" in leaves and any(token in leaves for token in ("升高", "偏高", "增高"))
        ):
            blocked.update(_POTASSIUM_RISK_TERMS)
        return blocked

    def _knowledge_blocked_terms(
        self,
        diagnoses: Sequence[Any],
        collected_info: Dict[str, Any],
    ) -> Set[str]:
        blocked: Set[str] = set()
        age = _age_number(collected_info.get("age"))
        text = json.dumps(collected_info or {}, ensure_ascii=False)
        for item in self.knowledge.get_contraindications(diagnoses):
            term = str(item.get("term") or "").strip()
            if not term:
                continue
            condition = str(item.get("condition") or "always")
            if condition == "always":
                blocked.add(term)
            elif condition == "pediatric" and age is not None and age < 18:
                blocked.add(term)
            elif condition.startswith("text_contains:"):
                marker = condition.split(":", 1)[1]
                if marker and marker in text:
                    blocked.add(term)
        return blocked


def _age_number(value: Any):
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _leaf_texts(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"reference_range", "参考范围", "参考值"}:
                continue
            yield str(key)
            yield from _leaf_texts(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaf_texts(item)
    elif value is not None:
        yield str(value)
