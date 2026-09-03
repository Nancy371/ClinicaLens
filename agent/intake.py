"""Deterministic, resumable consultation intake for the patient journey."""

from __future__ import annotations

import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


REQUIRED_FIELDS = (
    "chief_complaint",
    "onset",
    "progression",
    "severity",
    "associated_symptoms",
    "medical_history",
    "surgery_history",
    "medication",
    "allergy",
    "family_history",
    "exposure_history",
)

FIELD_LABELS = {
    "chief_complaint": "主要不适",
    "onset": "发病时间",
    "progression": "变化过程",
    "severity": "最严重程度",
    "associated_symptoms": "伴随症状",
    "medical_history": "既往疾病",
    "surgery_history": "手术史",
    "medication": "当前用药",
    "allergy": "过敏情况",
    "family_history": "家族史",
    "exposure_history": "吸烟、职业与暴露",
}

QUESTION_GROUPS = (
    {
        "id": "chief_complaint",
        "stage": "CHIEF_COMPLAINT",
        "fields": ["chief_complaint"],
        "text": "请先告诉我，这次最让你担心的不舒服是什么？",
        "why": "先找到最需要解决的问题。",
    },
    {
        "id": "symptom_characterization",
        "stage": "SYMPTOM_CHARACTERIZATION",
        "fields": ["onset", "progression", "severity"],
        "text": "这次不舒服是什么时候开始的？是突然出现还是慢慢加重？最严重时 0–10 分大约几分？",
        "why": "时间和程度影响就医紧急性。",
    },
    {
        "id": "associated_symptoms",
        "stage": "ASSOCIATED_SYMPTOMS",
        "fields": ["associated_symptoms"],
        "text": "同时有没有发热、胸痛、咯血、呼吸困难、晕厥、腿肿，或尿色和尿量变化？没有也请明确告诉我。",
        "why": "伴随表现可提示危险原因。",
    },
    {
        "id": "medical_and_surgery_history",
        "stage": "MEDICAL_HISTORY",
        "fields": ["medical_history", "surgery_history"],
        "text": "以前确诊过哪些疾病、住过院或做过哪些手术？如果都没有，也请明确说没有。",
        "why": "既往情况会改变风险判断。",
    },
    {
        "id": "medication_and_allergy",
        "stage": "MEDICATION",
        "fields": ["medication", "allergy"],
        "text": "现在正在用哪些处方药、非处方药或保健品？有无药物、食物或造影剂过敏？",
        "why": "用药和过敏影响检查安全。",
    },
    {
        "id": "family_and_exposure",
        "stage": "FAMILY_HISTORY",
        "fields": ["family_history", "exposure_history"],
        "text": "家人有无类似疾病？请再告诉我吸烟、职业、粉尘、动物或近期旅行接触情况。",
        "why": "家族和接触史帮助补全原因。",
    },
)

DANGER_SIGN_LABELS = {
    "active_hemoptysis": "大量或持续咯血",
    "severe_dyspnea": "静息时明显呼吸困难",
    "altered_consciousness": "意识改变或晕厥",
    "low_oxygen": "可靠指氧低于 93%",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_consultation_state(*, session_id: str = "") -> Dict[str, Any]:
    state = {
        "schema_version": "consultation-state.v1",
        "session_id": session_id or str(uuid.uuid4()),
        "current_stage": "SAFETY_SCREEN",
        "known_facts": {},
        "missing_required_fields": list(REQUIRED_FIELDS),
        "optional_missing_fields": ["pregnancy_status", "recent_travel"],
        "red_flags": [],
        "answered_questions": [],
        "pending_question": _safety_question(),
        "corrections": [],
        "summary": {},
        "completion_status": "in_progress",
        "safety_screened": False,
        "safety_checked_at": None,
        "safety_recheck_reason": None,
        "summary_confirmed": False,
        "summary_confirmed_at": None,
    }
    return state


def sample_consultation_state() -> Dict[str, Any]:
    state = new_consultation_state(session_id="sample-consultation-state")
    state["known_facts"] = {
        "chief_complaint": "2 周乏力与关节痛，3 天血丝痰、活动后呼吸困难",
        "onset": "乏力与关节痛 2 周，呼吸道症状 3 天",
        "progression": "近 3 天出现血丝痰并逐渐活动后气短",
        "severity": "活动后明显，静息时尚可",
        "associated_symptoms": "血丝痰、活动后气短；无大量咯血、静息气促或晕厥",
        "medical_history": "高血压 3 年；否认糖尿病及既往慢性肾病",
        "surgery_history": "2012 年阑尾切除术",
        "medication": "氨氯地平 5 mg 每日一次；布洛芬 200 mg 偶尔服用",
        "allergy": "无已知药物、食物或造影剂过敏",
        "family_history": "无自身免疫病或遗传性肾病家族史",
        "exposure_history": "既往 10 包年吸烟史，2 年前戒烟；偶有装修粉尘暴露",
    }
    state["safety_screened"] = True
    state["safety_checked_at"] = "2026-09-02T08:10:00+08:00"
    state["summary_confirmed"] = True
    state["summary_confirmed_at"] = "2026-09-02T08:12:00+08:00"
    state["answered_questions"] = [
        {"question_id": item["id"], "fields": item["fields"], "answered_at": f"2026-09-02T08:{index:02d}:00+08:00"}
        for index, item in enumerate(QUESTION_GROUPS, start=1)
    ]
    return recompute_state(state)


def _safety_question(reason: str = "") -> Dict[str, Any]:
    return {
        "id": "safety_screen",
        "stage": "SAFETY_SCREEN",
        "fields": ["red_flags"],
        "text": "先确认现在有没有持续大量咯血、静息时明显呼吸困难、晕厥/意识模糊，或可靠指氧低于 93%。",
        "why": "先排除需要立即急诊的情况。",
        "response_type": "safety_screen",
        "reason": reason,
    }


def _contains_unknown(text: str) -> bool:
    return any(word in text for word in ("不知道", "不清楚", "不了解", "记不清"))


def _extract_characterization(text: str) -> Dict[str, str]:
    updates: Dict[str, str] = {}
    onset = re.search(r"(?:约|大概|已经|持续)?\s*\d+(?:\.\d+)?\s*(?:分钟|小时|天|周|个月|月|年)(?:前|了|左右)?", text)
    if onset:
        updates["onset"] = onset.group(0).strip()
    progression_terms = [word for word in ("突然", "慢慢", "逐渐", "加重", "缓解", "反复", "没有变化", "时好时坏") if word in text]
    if progression_terms:
        updates["progression"] = "、".join(progression_terms)
    severity = re.search(r"(?:最严重(?:时)?|大约|约)?\s*([0-9]|10)\s*分", text)
    if severity:
        updates["severity"] = f"{severity.group(1)} 分（0–10 分）"
    elif any(word in text for word in ("轻微", "不严重", "一般", "比较严重", "很严重", "无法忍受")):
        updates["severity"] = text[:160]
    if _contains_unknown(text):
        for field in ("onset", "progression", "severity"):
            updates.setdefault(field, "不了解")
    return updates


def _new_safety_signal(text: str) -> bool:
    return any(_positive_term(text, word) for word in ("刚才开始咯血", "刚才开始咳血", "突然喘", "喘不过气", "晕倒", "晕厥", "意识模糊", "指氧"))


def _positive_term(text: str, term: str) -> bool:
    start = text.find(term)
    if start < 0:
        return False
    clause_start = max(text.rfind(mark, 0, start) for mark in ("，", "。", "；", "!", "！", "?", "？")) + 1
    prefix = text[clause_start:start]
    return not any(negation in prefix for negation in ("没有", "无", "否认", "未出现", "不伴"))


def _strong_red_flags(text: str) -> List[str]:
    flags: List[str] = []
    if any(_positive_term(text, word) for word in ("大量咯血", "持续咯血", "大口咳血")):
        flags.append("active_hemoptysis")
    if any(_positive_term(text, word) for word in ("静息呼吸困难", "坐着也喘", "喘不过气", "无法完整说话")):
        flags.append("severe_dyspnea")
    if any(_positive_term(text, word) for word in ("意识模糊", "晕厥", "晕倒")):
        flags.append("altered_consciousness")
    oxygen = re.search(r"(?:指氧|血氧)[^0-9]{0,5}(\d{2,3})", text)
    if oxygen and int(oxygen.group(1)) < 93:
        flags.append("low_oxygen")
    return flags


def _question_for_missing(missing: Iterable[str]) -> Optional[Dict[str, Any]]:
    missing_set = set(missing)
    for group in QUESTION_GROUPS:
        fields = [field for field in group["fields"] if field in missing_set]
        if not fields:
            continue
        question = deepcopy(group)
        question["fields"] = fields
        if group["id"] == "symptom_characterization" and fields != group["fields"]:
            question["text"] = "还需要补充：" + "、".join(FIELD_LABELS[field] for field in fields) + "。请按实际情况告诉我，不清楚也可以明确说不知道。"
        return question
    return None


def _build_summary(facts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "主要不适": facts.get("chief_complaint", "未提供"),
        "开始与变化": "；".join(str(facts.get(field)) for field in ("onset", "progression", "severity") if facts.get(field)) or "未提供",
        "伴随情况": facts.get("associated_symptoms", "未提供"),
        "既往疾病与手术": "；".join(str(facts.get(field)) for field in ("medical_history", "surgery_history") if facts.get(field)) or "未提供",
        "用药与过敏": "；".join(str(facts.get(field)) for field in ("medication", "allergy") if facts.get(field)) or "未提供",
        "家族与接触": "；".join(str(facts.get(field)) for field in ("family_history", "exposure_history") if facts.get(field)) or "未提供",
    }


def recompute_state(raw_state: Dict[str, Any]) -> Dict[str, Any]:
    state = deepcopy(raw_state)
    facts = state.setdefault("known_facts", {})
    missing = [field for field in REQUIRED_FIELDS if not str(facts.get(field) or "").strip()]
    state["missing_required_fields"] = missing
    state["summary"] = _build_summary(facts)
    if state.get("red_flags"):
        state["current_stage"] = "SAFETY_SCREEN"
        state["pending_question"] = None
        state["completion_status"] = "emergency_interrupted"
        state["summary_confirmed"] = False
    elif not state.get("safety_screened"):
        state["current_stage"] = "SAFETY_SCREEN"
        state["pending_question"] = _safety_question(str(state.get("safety_recheck_reason") or ""))
        state["completion_status"] = "in_progress"
        state["summary_confirmed"] = False
    elif missing:
        question = _question_for_missing(missing)
        state["current_stage"] = question["stage"] if question else "SUMMARY_CONFIRMATION"
        state["pending_question"] = question
        state["completion_status"] = "in_progress"
        state["summary_confirmed"] = False
    elif not state.get("summary_confirmed"):
        state["current_stage"] = "SUMMARY_CONFIRMATION"
        state["pending_question"] = {
            "id": "summary_confirmation",
            "stage": "SUMMARY_CONFIRMATION",
            "fields": [],
            "text": "请核对下面的信息。确认无误后，系统才会用于辅助判断。",
            "why": "避免错误信息直接进入判断。",
            "response_type": "summary_confirmation",
        }
        state["completion_status"] = "awaiting_summary_confirmation"
    else:
        state["current_stage"] = "READY_FOR_ASSESSMENT"
        state["pending_question"] = None
        state["completion_status"] = "ready_for_assessment"
    completed = {group["id"]: all(field not in missing for field in group["fields"]) for group in QUESTION_GROUPS}
    state["progress"] = [
        {"id": "safety_screen", "label": "危险信号", "complete": bool(state.get("safety_screened"))},
        {"id": "chief_complaint", "label": "主要不适", "complete": completed["chief_complaint"]},
        {"id": "symptoms", "label": "发病时间与程度", "complete": completed["symptom_characterization"] and completed["associated_symptoms"]},
        {"id": "history", "label": "既往疾病与手术", "complete": completed["medical_and_surgery_history"]},
        {"id": "medication", "label": "用药与过敏", "complete": completed["medication_and_allergy"]},
        {"id": "context", "label": "家族与接触史", "complete": completed["family_and_exposure"]},
    ]
    return state


def apply_consultation_turn(
    raw_state: Dict[str, Any],
    text: str,
    *,
    danger_signs: Optional[Iterable[str]] = None,
    confirm_summary: bool = False,
) -> Dict[str, Any]:
    state = recompute_state(raw_state or new_consultation_state())
    clean = str(text or "").strip()[:500]
    now = _now()
    if confirm_summary:
        if state.get("completion_status") != "awaiting_summary_confirmation":
            raise ValueError("summary_not_ready")
        state["summary_confirmed"] = True
        state["summary_confirmed_at"] = now
        return recompute_state(state)
    strong_flags = _strong_red_flags(clean)
    if strong_flags:
        state["red_flags"] = sorted(set(state.get("red_flags", []) + strong_flags))
        state["safety_screened"] = True
        state["safety_checked_at"] = now
        return recompute_state(state)
    if danger_signs is not None:
        allowed = set(DANGER_SIGN_LABELS)
        state["red_flags"] = sorted({str(item) for item in danger_signs if str(item) in allowed})
        state["safety_screened"] = True
        state["safety_checked_at"] = now
        state["safety_recheck_reason"] = None
    elif state.get("safety_screened") and _new_safety_signal(clean):
        state["safety_screened"] = False
        state["safety_recheck_reason"] = "你刚补充了可能影响紧急程度的新症状，需要重新确认一次。"
    if clean:
        pending = state.get("pending_question") or {}
        facts = state.setdefault("known_facts", {})
        if not facts.get("chief_complaint"):
            facts["chief_complaint"] = clean
            facts.update(_extract_characterization(clean))
            question_id = "chief_complaint"
            fields = ["chief_complaint"]
        elif pending.get("id") == "symptom_characterization":
            facts.update(_extract_characterization(clean))
            question_id = pending["id"]
            fields = list(pending.get("fields") or [])
        elif pending.get("id") == "associated_symptoms":
            facts["associated_symptoms"] = clean
            question_id, fields = pending["id"], list(pending["fields"])
        elif pending.get("id") == "medical_and_surgery_history":
            facts["medical_history"] = clean
            facts["surgery_history"] = clean
            question_id, fields = pending["id"], list(pending["fields"])
        elif pending.get("id") == "medication_and_allergy":
            facts["medication"] = clean
            facts["allergy"] = clean
            question_id, fields = pending["id"], list(pending["fields"])
        elif pending.get("id") == "family_and_exposure":
            facts["family_history"] = clean
            facts["exposure_history"] = clean
            question_id, fields = pending["id"], list(pending["fields"])
        else:
            question_id, fields = "additional_information", []
        state.setdefault("answered_questions", []).append({
            "question_id": question_id,
            "fields": fields,
            "answer": clean,
            "answered_at": now,
        })
    return recompute_state(state)


def apply_consultation_correction(
    raw_state: Dict[str, Any], field: str, new_value: str, *, reason: str = ""
) -> Dict[str, Any]:
    if field not in REQUIRED_FIELDS:
        raise ValueError("invalid_correction_field")
    clean = str(new_value or "").strip()[:500]
    if not clean:
        raise ValueError("correction_value_required")
    state = recompute_state(raw_state or new_consultation_state())
    old_value = state.setdefault("known_facts", {}).get(field)
    state["known_facts"][field] = clean
    state["summary_confirmed"] = False
    state["summary_confirmed_at"] = None
    state.setdefault("corrections", []).append({
        "id": str(uuid.uuid4()),
        "field": field,
        "field_label": FIELD_LABELS[field],
        "old_value": old_value,
        "new_value": clean,
        "reason": str(reason or "").strip()[:300],
        "timestamp": _now(),
    })
    return recompute_state(state)


def intake_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    pending = state.get("pending_question") or {}
    status = state.get("completion_status")
    if status == "emergency_interrupted":
        labels = [DANGER_SIGN_LABELS.get(item, item) for item in state.get("red_flags", [])]
        return {
            "intent": "emergency",
            "direct_answer": "现在需要立即急诊，不要继续等待在线问诊。",
            "urgency": "emergency",
            "basis": "已报告危险信号：" + "、".join(labels),
            "follow_up_questions": [],
            "next_action": "立即拨打 120 或前往最近急诊。",
            "boundary": "安全分流不替代急诊医生评估。",
            "intake_question": None,
        }
    if status == "ready_for_assessment":
        return {
            "intent": "intake_ready",
            "direct_answer": "你已确认信息摘要，可以进入辅助判断。",
            "urgency": "guided",
            "basis": "必要信息已补全，并由你完成最终核对。",
            "follow_up_questions": [],
            "next_action": "连接医院记录或上传已有检查，然后开始辅助判断。",
            "boundary": "AI 只提供辅助判断，确诊和治疗由医生负责。",
            "intake_question": None,
        }
    if status == "awaiting_summary_confirmation":
        direct = "需要的信息已经收集齐。请先核对摘要；确认前不会进入辅助判断。"
    elif pending.get("response_type") == "safety_screen":
        direct = pending.get("reason") or "我先记录了你的问题。请完成一次危险信号确认。"
    else:
        direct = "我已经记下这部分信息，接下来只追问还缺少的内容。"
    return {
        "intent": "intake",
        "direct_answer": direct,
        "urgency": "guided",
        "basis": pending.get("why") or "补全必要信息可以减少误判。",
        "follow_up_questions": [pending.get("text")] if pending.get("text") else [],
        "next_action": pending.get("text") or "核对已收集的信息。",
        "boundary": "这是信息收集与就医导航，不替代医生诊断。",
        "intake_question": deepcopy(pending) if pending else None,
    }


__all__ = [
    "DANGER_SIGN_LABELS",
    "FIELD_LABELS",
    "QUESTION_GROUPS",
    "REQUIRED_FIELDS",
    "apply_consultation_correction",
    "apply_consultation_turn",
    "intake_answer",
    "new_consultation_state",
    "recompute_state",
    "sample_consultation_state",
]
