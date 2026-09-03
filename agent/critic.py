"""缺陷检测器：从案例评估报告与执行轨迹中自动识别缺陷并归因。

DefectDetector 是自迭代闭环的第一环。它接收一次诊疗结束后的：
  - report：平台评估报告（含 diagnosisAccuracy / examinationPrecision / treatmentOverallScore 等）
  - collected_info：问诊收集到的患者信息
  - exam_results：检查结果
  - action_history：执行轨迹
并输出结构化的缺陷列表（defects），每一项包含：
  - subsystem：inquiry / examination / treatment / boundary
  - severity：high / medium / low
  - signal：触发规则名（便于溯源）
  - evidence：触发时的关键字段
  - failure_stage / failure_type / root_cause：先定位错误层级和可泛化根因
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# 分数阈值：低于此值视为该子系统失分
_THRESH_DIAGNOSIS = 0.80
_THRESH_EXAM = 0.80
_THRESH_TREATMENT = 0.80

# 检查覆盖度：<该值视为漏检
_THRESH_EXAM_COVERAGE = 0.50


class DefectDetector:
    """规则通道 + LLM 通道（可选）双路缺陷识别。

    规则通道零成本、可解释，先行判定。LLM 通道用于规则未捕获的复杂归因，
    需要显式启用（通过传入 llm_chat 回调）。
    """

    def __init__(self, knowledge=None, llm_chat=None):
        """
        Args:
            knowledge: KnowledgeBase 实例，用于覆盖度评分等结构化判定
            llm_chat: 可选，异步 LLM 回调（messages, temperature) -> str，
                      若提供则启用 LLM 归因分支
        """
        self.knowledge = knowledge
        self.llm_chat = llm_chat

    # ---------------- 规则通道----------------

    def detect(
        self,
        report: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        action_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """规则通道检测。返回 defects 列表。"""
        defects: List[Dict[str, Any]] = []
        report = report or {}
        collected_info = collected_info or {}
        exam_results = exam_results or {}

        diag_acc = _safe_float(report.get("diagnosisAccuracy"))
        exam_prec = _safe_float(report.get("examinationPrecision"))
        treat_score = _safe_float(report.get("treatmentOverallScore"))
        final_dx = report.get("finalDiagnosis") or report.get("diagnosis") or ""
        symptoms = collected_info.get("symptoms") or []
        age = collected_info.get("age")
        gender = collected_info.get("gender")

        # R1: 诊断准确率偏低 → 问诊+推理策略问题
        if diag_acc is not None and diag_acc < _THRESH_DIAGNOSIS:
            defects.append({
                "subsystem": "inquiry",
                "severity": "high" if diag_acc < 0.6 else "medium",
                "signal": "low_diagnosis_accuracy",
                "evidence": {"diagnosisAccuracy": diag_acc, "final_dx": final_dx,
                             "symptoms": symptoms[:6]},
                "suggested_fix": {
                    "type": "inquiry_deepen",
                    "trigger": {"symptoms_any": symptoms[:3]},
                    "action": "扩展鉴别诊断维度，追问诱因/伴随/加重缓解因素/既往史",
                },
            })

        # R2: 检查精确率偏低 → 检查选择泛滥或遗漏
        if exam_prec is not None and exam_prec < _THRESH_EXAM:
            defects.append({
                "subsystem": "examination",
                "severity": "medium",
                "signal": "low_examination_precision",
                "evidence": {"examinationPrecision": exam_prec,
                             "ordered": list(exam_results.keys())[:8]},
                "suggested_fix": {
                    "type": "exam_prune",
                    "trigger": {"final_dx": final_dx},
                    "action": "严格按 KB 推荐清单开检查，禁止兜底式全套",
                },
            })

        # R3: 治疗分偏低 → 治疗方案策略问题
        if treat_score is not None and treat_score < _THRESH_TREATMENT:
            defects.append({
                "subsystem": "treatment",
                "severity": "medium",
                "signal": "low_treatment_score",
                "evidence": {"treatmentOverallScore": treat_score, "final_dx": final_dx},
                "suggested_fix": {
                    "type": "treatment_personalize",
                    "trigger": {"final_dx": final_dx},
                    "action": "结合年龄/性别/合并症调整用药与剂量，补充随访计划",
                },
            })

        # R4: KB 覆盖度不足 → 漏检
        if self.knowledge and final_dx:
            try:
                cov = self.knowledge.score_examination_coverage(
                    list(exam_results.keys()), candidate_diseases=[final_dx]
                ) or {}
                coverage = _safe_float(cov.get("coverage"))
                supplements = cov.get("recommended_supplements") or []
                if coverage is not None and coverage < _THRESH_EXAM_COVERAGE and supplements:
                    defects.append({
                        "subsystem": "examination",
                        "severity": "high",
                        "signal": "kb_coverage_gap",
                        "evidence": {"coverage": coverage,
                                     "final_dx": final_dx,
                                     "missing": supplements[:5]},
                        "suggested_fix": {
                            "type": "exam_mandatory",
                            "trigger": {"final_dx": final_dx},
                            "action": f"命中 {final_dx} 时强制补充检查: {supplements[:3]}",
                            "items": supplements[:3],
                        },
                    })
            except Exception as e:
                logger.debug(f"[detector] KB 覆盖度评分失败: {e}")

        # R5: 边界条件 —— 老年+胸痛必须心电图/心肌酶
        try:
            age_num = _to_age_int(age)
            has_chest_pain = any("胸痛" in str(s) for s in symptoms)
            if age_num is not None and age_num >= 60 and has_chest_pain:
                ordered = [str(k) for k in exam_results.keys()]
                needed = ["心电图", "心肌酶", "肌钙蛋白"]
                if not any(any(n in o for o in ordered) for n in needed):
                    defects.append({
                        "subsystem": "boundary",
                        "severity": "high",
                        "signal": "elderly_chest_pain_missing_ecg",
                        "evidence": {"age": age, "symptoms": symptoms[:5],
                                     "ordered": ordered[:8]},
                        "suggested_fix": {
                            "type": "exam_mandatory",
                            "trigger": {"age_min": 60, "symptoms_any": ["胸痛"]},
                            "action": "老年+胸痛必须开心电图/心肌酶/肌钙蛋白 排除 ACS",
                            "items": ["心电图", "心肌酶", "肌钙蛋白"],
                        },
                    })
        except Exception:
            pass

        # R6: 症状为空却给出诊断 → 数据链路问题
        if final_dx and not symptoms:
            defects.append({
                "subsystem": "inquiry",
                "severity": "high",
                "signal": "diagnosis_without_symptoms",
                "evidence": {"final_dx": final_dx},
                "suggested_fix": {
                    "type": "inquiry_deepen",
                    "trigger": {"always": True},
                    "action": "确保 collected_info.symptoms 至少 1 条再进入诊断阶段",
                },
            })

        defects = [self._normalize_failure_attribution(item) for item in defects]

        if defects:
            logger.info(f"[detector] 规则通道识别缺陷 {len(defects)} 项: "
                        f"{[d['signal'] for d in defects]}")
        return defects

    # ---------------- LLM 归因通道（P1） ----------------

    async def attribute_with_llm(
        self,
        report: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        rule_defects: List[Dict[str, Any]],
        max_extra: int = 3,
    ) -> List[Dict[str, Any]]:
        """LLM 归因：让 LLM 补充规则未覆盖的深层原因。

        约定 llm_chat 签名: `async llm_chat(messages: List[dict], temperature: float) -> str`
        返回 JSON 数组，每项形如:
          {"failure_stage": "...", "failure_type": "...",
           "affected_candidate": "...", "root_cause": "...",
           "generalizable_pattern": "...", "evidence_refs": [...]}
        """
        if not self.llm_chat:
            return []

        # 规则已覆盖的 signal 集合，避免重复
        covered = {d.get("signal") for d in (rule_defects or [])}
        rule_summary = [
            {"signal": d.get("signal"), "subsystem": d.get("subsystem"),
             "severity": d.get("severity")}
            for d in (rule_defects or [])
        ]

        system_prompt = (
            "你是医疗 Agent 的复盘专家。给定一次诊疗的评估报告、问诊信息、检查结果、"
            "以及规则通道已识别的缺陷,请**仅**输出规则未覆盖的**深层归因**缺陷。\n"
            "输出严格 JSON 数组,每项字段: failure_stage(evidence_mapping|candidate_recall|"
            "eligibility|ranking|exam_selection|submission), failure_type, affected_candidate, "
            "root_cause, generalizable_pattern, evidence_refs。\n"
            "不要输出 suggested_fix、疾病加分、疾病降分或病例答案补丁。\n"
            f"最多输出 {max_extra} 项。如无新增缺陷,输出 []。禁止解释,禁止 markdown。"
        )
        user_prompt = _build_llm_user_prompt(
            report, collected_info, exam_results, rule_summary
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = await self.llm_chat(messages, temperature=0.2)
        except Exception as e:
            logger.warning(f"[detector.llm] LLM 调用失败: {e}")
            return []

        extra = _parse_llm_json_array(raw)
        # 过滤重复 signal + 强校验必需字段
        out: List[Dict[str, Any]] = []
        for item in extra[:max_extra]:
            if not isinstance(item, dict):
                continue
            sig = item.get("failure_type") or item.get("signal")
            if not sig or sig in covered:
                continue
            item.setdefault("signal", sig)
            item.setdefault("failure_stage", item.get("subsystem") or "ranking")
            item.setdefault("failure_type", sig)
            item.setdefault("affected_candidate", "")
            item.setdefault("root_cause", sig)
            item.setdefault("generalizable_pattern", item.get("root_cause") or sig)
            refs = item.get("evidence_refs") or item.get("evidence") or []
            if isinstance(refs, dict):
                refs = list(refs.keys())
            elif isinstance(refs, str):
                refs = [refs]
            elif not isinstance(refs, list):
                refs = []
            item["evidence_refs"] = refs
            item.setdefault("severity", "medium")
            item.setdefault("evidence", {})
            item.pop("suggested_fix", None)
            item["source_channel"] = "llm"
            out.append(self._normalize_failure_attribution(item))
            covered.add(sig)

        if out:
            logger.info(f"[detector.llm] LLM 通道补充缺陷 {len(out)} 项: "
                        f"{[d['signal'] for d in out]}")
        return out

    async def detect_all(
        self,
        report: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        action_history: Optional[List[Dict[str, Any]]] = None,
        use_llm: bool = True,
    ) -> List[Dict[str, Any]]:
        """规则通道 + LLM 通道合并入口(异步)。use_llm=False 或未提供 llm_chat 时退化为规则通道。"""
        rule_defects = self.detect(report, collected_info, exam_results, action_history)
        if use_llm and self.llm_chat:
            extra = await self.attribute_with_llm(
                report, collected_info, exam_results, rule_defects
            )
            rule_defects.extend(extra)
        return [self._normalize_failure_attribution(item) for item in rule_defects]

    @staticmethod
    def _normalize_failure_attribution(item: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(item or {})
        signal = str(item.get("signal") or item.get("failure_type") or "unknown_failure").strip()
        stage = _failure_stage(item.get("failure_stage") or item.get("subsystem") or signal)
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        final_dx = str(evidence.get("final_dx") or evidence.get("finalDiagnosis") or "").strip()
        root_cause, pattern = _failure_text(stage, signal, final_dx)
        item["signal"] = signal
        item["failure_stage"] = stage
        item["failure_type"] = str(item.get("failure_type") or signal)
        item["affected_candidate"] = str(item.get("affected_candidate") or final_dx or "").strip()
        item["root_cause"] = str(item.get("root_cause") or root_cause)
        item["generalizable_pattern"] = str(item.get("generalizable_pattern") or pattern)
        refs = item.get("evidence_refs")
        if not refs:
            refs = list(evidence.keys()) if isinstance(evidence, dict) else []
        if isinstance(refs, str):
            refs = [refs]
        elif not isinstance(refs, list):
            refs = []
        item["evidence_refs"] = [str(ref) for ref in refs if str(ref)]
        item["source_case"] = str(item.get("source_case") or evidence.get("patient_id") or "")
        item.pop("suggested_fix", None)
        return item


# ---------------- helpers ----------------

def _failure_stage(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "inquiry": "candidate_recall",
        "low_diagnosis_accuracy": "ranking",
        "diagnosis_without_symptoms": "candidate_recall",
        "examination": "exam_selection",
        "low_examination_precision": "exam_selection",
        "kb_coverage_gap": "exam_selection",
        "treatment": "submission",
        "low_treatment_score": "submission",
        "boundary": "submission",
        "elderly_chest_pain_missing_ecg": "exam_selection",
        "reasoning": "ranking",
    }
    text = aliases.get(text, text)
    if text in {
        "evidence_mapping",
        "candidate_recall",
        "eligibility",
        "ranking",
        "exam_selection",
        "submission",
    }:
        return text
    return "ranking"


def _failure_text(stage: str, signal: str, final_dx: str = "") -> tuple:
    affected = f" for {final_dx}" if final_dx else ""
    if signal == "low_diagnosis_accuracy":
        return (
            "diagnosis failed; locate recall, eligibility, ranking, or submission before changing scores",
            "diagnosis accuracy failures must become layer-specific principles, not disease score patches",
        )
    if signal == "low_examination_precision":
        return (
            "exam selection produced low-precision examinations",
            "exam policies must target missing discriminating anchors and avoid broad panels",
        )
    if signal == "low_treatment_score":
        return (
            "submitted treatment plan underperformed",
            "treatment adjustments belong to submission safety and personalization rules",
        )
    if signal == "kb_coverage_gap":
        return (
            f"required examination coverage was incomplete{affected}",
            "missing required tests should generate exam-selection candidates, not final diagnosis authorization",
        )
    if signal == "elderly_chest_pain_missing_ecg":
        return (
            "red-flag presentation lacked required cardiac exclusion tests",
            "safety-critical symptoms require specific exclusion exams before low-risk submission",
        )
    if signal == "diagnosis_without_symptoms":
        return (
            "diagnosis was attempted before enough patient evidence was collected",
            "candidate recall must preserve evidence intake before diagnosis generation",
        )
    return (
        f"{stage} failure detected by {signal}",
        f"{stage} failures should be generalized as evidence conditions and layer-local actions",
    )


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_age_int(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip()
    if not s:
        return None
    # 提取数字
    import re
    m = re.search(r"\d+", s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def _build_llm_user_prompt(report, collected_info, exam_results, rule_summary) -> str:
    """构造 LLM归��的 user prompt(裁剪长字段,避免超 token)。"""
    import json as _json
    def _trunc(obj, limit=800):
        try:
            s = _json.dumps(obj, ensure_ascii=False)
        except Exception:
            s = str(obj)
        return s if len(s) <= limit else s[:limit] + "...(截断)"

    return (
        "【评估报告】\n" + _trunc({
            "diagnosisAccuracy": report.get("diagnosisAccuracy"),
            "examinationPrecision": report.get("examinationPrecision"),
            "treatmentOverallScore": report.get("treatmentOverallScore"),
            "finalDiagnosis": report.get("finalDiagnosis") or report.get("diagnosis"),
            "expectedDiagnosis": report.get("expectedDiagnosis"),
            "feedback": report.get("feedback"),
        }, 600) +
        "\n\n【问诊信息】\n" + _trunc(collected_info, 800) +
        "\n\n【检查结果 keys】\n" + _trunc(list((exam_results or {}).keys()), 400) +
        "\n\n【规则通道已识别】\n" + _trunc(rule_summary, 400) +
        "\n\n请仅输出 JSON 数组,不要解释。"
    )


def _parse_llm_json_array(raw: str) -> List[Dict[str, Any]]:
    """从 LLM 输出中鲁棒地提取 JSON 数组。支持带 markdown code fence / 前后有噪声文字。"""
    import json as _json
    if not raw or not isinstance(raw, str):
        return []
    s = raw.strip()
    # 去除 ```json ... ``` 包裹
    if s.startswith("```"):
        # 去掉首行 fence
        lines = s.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # 直接尝试
    try:
        val = _json.loads(s)
        if isinstance(val, list):
            return val
        if isinstance(val, dict) and "defects" in val and isinstance(val["defects"], list):
            return val["defects"]
    except Exception:
        pass
    # 兜底:找第一个 '[' 到最后一个 ']'
    l = s.find("[")
    r = s.rfind("]")
    if 0 <= l < r:
        try:
            val = _json.loads(s[l:r + 1])
            if isinstance(val, list):
                return val
        except Exception:
            return []
    return []
