"""ClinicaLens consumer product projections and deterministic safety content.

This module contains only the explicitly fictional sandbox case and curated
decision-support copy.  It does not call an LLM or activate a prescription.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_patient_profile() -> Dict[str, Any]:
    return {
        "name": "周予安",
        "sex": "男",
        "age": 46,
        "weight_kg": 70,
        "date_of_birth": "1980-04-18",
        "phone": "138 0000 2468",
        "address": "北京市朝阳区望京街道（虚构）",
        "hospital_record_no": "SBX-20260902-001",
        "emergency_contact": "林女士 · 139 0000 1357（虚构）",
        "is_fictional": True,
        "notice": "以上身份、病历、诊断和处方均为产品沙箱虚构数据，不对应任何真实患者。",
    }


def sample_clinical_history(*, confirmed: bool = False) -> Dict[str, Any]:
    status = "confirmed" if confirmed else "unconfirmed"
    return {
        "conditions": [
            {"name": "高血压", "detail": "3 年，平时血压约 135–145/85–95 mmHg"},
            {"name": "糖尿病", "detail": "否认"},
            {"name": "慢性肾病", "detail": "否认既往诊断"},
        ],
        "surgeries": [{"name": "阑尾切除术", "detail": "2012 年，恢复良好"}],
        "current_medications": [
            {"name": "氨氯地平", "dose": "5 mg", "frequency": "每日一次", "source": "患者自述"},
            {"name": "布洛芬", "dose": "200 mg", "frequency": "近一个月关节痛时偶尔服用", "source": "患者自述"},
        ],
        "allergies": [
            {"allergen": "药物", "reaction": "未发现", "severity": "none", "status": "known_none"},
            {"allergen": "食物", "reaction": "未发现", "severity": "none", "status": "known_none"},
            {"allergen": "造影剂", "reaction": "未发现", "severity": "none", "status": "known_none"},
        ],
        "family_history": ["无明确自身免疫病家族史", "无遗传性肾病家族史"],
        "social_history": {
            "smoking": "既往约 10 包年，2 年前戒烟",
            "alcohol": "偶尔饮酒",
            "occupation": "室内设计项目经理",
            "exposures": "偶有装修粉尘暴露，否认石棉和禽类长期接触",
        },
        "field_statuses": {
            "conditions": status,
            "surgeries": status,
            "current_medications": status,
            "allergies": status,
            "family_history": status,
            "social_history": status,
        },
        "confirmation_status": "confirmed" if confirmed else "unconfirmed",
        "confirmed_at": utc_now() if confirmed else None,
        "source": "问诊生成病例原文 · 用户确认",
    }


def sample_consultation_messages() -> List[Dict[str, Any]]:
    """Fictional intake transcript used to prove where the sample note came from."""
    turns = [
        ("sample-q-1", "我这两周总是乏力，膝盖和手腕酸痛，最近三天开始有血丝痰，活动后有点喘，这严重吗？", "建议今天线下评估；先确认是否存在大量咯血、静息气促、低氧或意识改变。"),
        ("sample-q-2", "现在没有大量咯血，坐着不喘，也没有晕厥。我今年46岁，男，体重70公斤。", "目前未报告列出的急性危险信号，但这组症状仍不能按普通咳嗽等待。"),
        ("sample-q-3", "我有高血压三年，2012年做过阑尾手术，没有糖尿病，也没有听说过慢性肾病。", "这些病史会进入病例整理稿，并需要你逐项确认。"),
        ("sample-q-4", "平时吃氨氯地平5毫克每天一次，最近关节痛偶尔吃布洛芬200毫克。没有已知药物、食物或造影剂过敏。", "用药与过敏信息会影响检查和后续治疗安全核对。"),
        ("sample-q-5", "以前吸烟大约10包年，两年前戒了。我做室内设计项目管理，偶尔接触装修粉尘，家里没有自身免疫病或遗传性肾病。", "职业暴露、吸烟史和家族史已经记录；仍需由医生结合检查判断。"),
    ]
    messages: List[Dict[str, Any]] = []
    for index, (message_id, text, answer) in enumerate(turns, start=1):
        messages.extend([
            {"id": message_id, "role": "user", "kind": "intake", "text": text, "created_at": f"2026-09-02T08:{index:02d}:00+08:00"},
            {"id": f"sample-a-{index}", "role": "assistant", "kind": "intake", "answer": {"direct_answer": answer, "urgency": "same_day", "basis": "问诊安全分流与病史整理", "next_action": "继续补全问诊", "follow_up_questions": [], "boundary": "这是就医导航与信息整理，不是医生诊断或处方。"}, "created_at": f"2026-09-02T08:{index:02d}:30+08:00"},
        ])
    return messages


def sample_raw_case_document() -> Dict[str, Any]:
    message_ids = [item["id"] for item in sample_consultation_messages() if item.get("role") == "user"]
    sections = [
        {"id": "complaint", "title": "主诉与现病史", "text": "2 周乏力、双膝及腕关节酸痛，3 天血丝痰并伴活动后呼吸困难。当前否认大量咯血、静息呼吸困难和晕厥。", "source_message_ids": ["sample-q-1", "sample-q-2"]},
        {"id": "identity", "title": "问诊基本信息", "text": "周予安，男，46 岁，体重 70 kg。身份信息均为完整虚构。", "source_message_ids": ["sample-q-2"]},
        {"id": "history", "title": "疾病史与手术史", "text": "高血压 3 年；否认糖尿病及既往慢性肾病。2012 年行阑尾切除。", "source_message_ids": ["sample-q-3"]},
        {"id": "medication", "title": "用药史与过敏史", "text": "氨氯地平 5 mg 每日一次；近一个月因关节痛间断服用布洛芬 200 mg。无已知药物、食物及造影剂过敏。", "source_message_ids": ["sample-q-4"]},
        {"id": "social", "title": "家族史、个人史与暴露史", "text": "无明确自身免疫病或遗传性肾病家族史。既往约 10 包年吸烟史，2 年前戒烟；职业为室内设计项目经理，偶有装修粉尘暴露。", "source_message_ids": ["sample-q-5"]},
    ]
    return {
        "id": "consultation-case-sample-v1",
        "version": 1,
        "status": "confirmed",
        "origin": "consultation",
        "title": "问诊生成病例原文（AI 整理稿）",
        "is_fictional": True,
        "source": "完整虚构问诊对话",
        "generated_from_message_ids": message_ids,
        "missing_information": ["首次问诊未包含医院检查结果；请在下方医院报告中查看。"],
        "created_on": "2026-09-02",
        "generated_at": "2026-09-02T08:08:00+08:00",
        "confirmed_at": "2026-09-02T08:09:00+08:00",
        "sections": sections,
        "full_text": "\n\n".join(f"【{item['title']}】\n{item['text']}" for item in sections),
        "notice": "本稿由完整虚构问诊生成，只整理用户口述；医院检查与医生结论保持独立来源。",
    }


def build_consultation_case_document(
    messages: Iterable[Dict[str, Any]],
    clinical_history: Dict[str, Any],
    *,
    version: int = 1,
    confirmed: bool = False,
) -> Dict[str, Any]:
    """Compile a traceable intake note without importing exams or diagnoses."""
    user_messages = [item for item in messages if item.get("role") == "user" and str(item.get("text") or "").strip()]
    message_ids = [str(item.get("id")) for item in user_messages if item.get("id")]
    narrative = "；".join(str(item.get("text") or "").strip() for item in user_messages[-8:]) or "尚未完成有效问诊。"

    def history_text(key: str) -> str:
        status = (clinical_history.get("field_statuses") or {}).get(key, "unconfirmed")
        if status == "unknown":
            return "用户明确选择不了解。"
        if status != "confirmed":
            return "尚未在问诊或病史确认中核实。"
        value = clinical_history.get(key)
        if isinstance(value, dict):
            return "；".join(str(item) for item in value.values() if item) or "已确认无补充信息。"
        if isinstance(value, list):
            rows = []
            for item in value:
                if isinstance(item, dict):
                    rows.append(" · ".join(str(value) for value in item.values() if value))
                elif item:
                    rows.append(str(item))
            return "；".join(rows) or "已确认无补充信息。"
        return str(value or "已确认无补充信息。")

    sections = [
        {"id": "complaint", "title": "主诉与问诊经过", "text": narrative, "source_message_ids": message_ids[-8:]},
        {"id": "history", "title": "疾病史与手术史", "text": f"疾病史：{history_text('conditions')} 手术史：{history_text('surgeries')}", "source_message_ids": message_ids},
        {"id": "medication", "title": "用药史与过敏史", "text": f"当前用药：{history_text('current_medications')} 过敏史：{history_text('allergies')}", "source_message_ids": message_ids},
        {"id": "context", "title": "家族史、个人史与暴露史", "text": f"家族史：{history_text('family_history')} 个人与暴露史：{history_text('social_history')}", "source_message_ids": message_ids},
    ]
    now = utc_now()
    status = "confirmed" if confirmed else "draft"
    return {
        "id": f"consultation-case-v{version}-{uuid.uuid4().hex[:8]}",
        "version": version,
        "status": status,
        "origin": "consultation",
        "title": "问诊生成病例原文（AI 整理稿）",
        "is_fictional": False,
        "source": "患者问诊与已确认病史",
        "generated_from_message_ids": message_ids,
        "missing_information": [
            label for key, label in {
                "conditions": "疾病史", "surgeries": "手术史", "current_medications": "用药史",
                "allergies": "过敏史", "family_history": "家族史", "social_history": "个人与暴露史",
            }.items() if (clinical_history.get("field_statuses") or {}).get(key) not in {"confirmed", "unknown"}
        ],
        "generated_at": now,
        "confirmed_at": now if confirmed else None,
        "sections": sections,
        "full_text": "\n\n".join(f"【{item['title']}】\n{item['text']}" for item in sections),
        "notice": "由问诊生成，只整理患者口述；检查结果与医生结论不会写入本稿。",
    }


def sample_record_batches() -> Dict[str, Dict[str, Any]]:
    return {
        "baseline": {
            "order": 1,
            "label": "症状、生命体征与病史",
            "records": [
                _record("record-symptoms", "symptom", "症状演变", ["2 周乏力与关节痛", "3 天血丝痰", "活动后呼吸困难", "当前无大咯血、静息呼吸困难或意识改变"], "主诉与现病史", True),
                _record("record-vitals", "observation", "生命体征", ["体温 37.4℃", "心率 102 次/分", "呼吸 22 次/分", "血压 148/92 mmHg", "静息指氧 94%（空气）"], "生命体征", True),
            ],
        },
        "organ": {
            "order": 2,
            "label": "尿检、肾功能与胸部 CT",
            "records": [
                _record("record-urinalysis", "laboratory", "尿常规与尿沉渣", ["尿红细胞 50 个/HPF（参考值 0–3）", "尿蛋白 2+", "尿沉渣见畸形红细胞"], "基础检验与影像", True),
                _record("record-renal", "laboratory", "肾功能", ["肌酐 220 μmol/L（参考值 44–133）", "eGFR 29 ml/min/1.73m²"], "基础检验与影像", True),
                _record("record-ct", "imaging", "胸部 CT", ["双肺磨玻璃影", "影像考虑弥漫性肺泡出血"], "基础检验与影像", True),
            ],
        },
        "serology": {
            "order": 3,
            "label": "免疫学与感染筛查",
            "records": [
                _record("record-anca", "laboratory", "ANCA 与鉴别抗体", ["MPO-ANCA 86 RU/ml 阳性", "PR3-ANCA 阴性", "抗 GBM 抗体阴性", "补体 C3/C4 正常"], "免疫学与感染筛查", True),
                _record("record-infection", "laboratory", "感染筛查", ["血培养未检出细菌", "痰病原学筛查阴性"], "免疫学与感染筛查", False, "阴性结果降低感染方向，但不能排除全部感染。"),
            ],
        },
        "biopsy": {
            "order": 4,
            "label": "肾活检与医生记录",
            "records": [
                _record("record-biopsy", "pathology", "肾活检", ["少免疫沉积性坏死性新月体性肾小球肾炎", "支持 ANCA 相关小血管炎"], "肾活检", True),
            ],
        },
    }


def _record(record_id: str, kind: str, title: str, items: List[str], locator: str, abnormal: bool, note: str = "") -> Dict[str, Any]:
    result = {
        "id": record_id,
        "kind": kind,
        "title": title,
        "items": items,
        "observed_at": "2026-09-01T09:00:00+08:00",
        "source": {"type": "sandbox", "label": "完整虚构病例原文", "locator": locator},
        "verification_status": "imported",
        "abnormal": abnormal,
    }
    if note:
        result["scenario_note"] = note
    return result


def all_sample_records() -> List[Dict[str, Any]]:
    return [deepcopy(record) for batch in sample_record_batches().values() for record in batch["records"]]


def build_evidence(records: Iterable[Dict[str, Any]], batch_keys: Iterable[str]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for record in records:
        role = "contradicting" if record["id"] == "record-infection" else "supporting"
        for index, item in enumerate(record.get("items") or []):
            evidence.append({
                "id": f"evidence-{record['id']}-{index}", "label": item, "role": role,
                "record_id": record["id"], "source": deepcopy(record["source"]),
                "observed_at": record.get("observed_at"), "verification_status": record.get("verification_status", "imported"),
                "reason": "阴性结果降低感染或抗GBM病方向，但不能单独完成排除。" if role == "contradicting" else "该结果已进入当前诊断版本的证据覆盖检查。",
            })
    batch_set = set(batch_keys)
    if "serology" not in batch_set:
        evidence.append(_gap("gap-serology", "ANCA、抗 GBM 抗体与补体结果", "影响肺肾综合征的进一步鉴别。"))
    if "biopsy" not in batch_set:
        evidence.append(_gap("gap-biopsy", "肾活检或其他组织学依据", "影响医生最终确诊和治疗选择。"))
    evidence.append(_gap("gap-infection", "是否已由医生充分排除感染", "部分阴性结果不等于已排除全部感染。"))
    return evidence


def _gap(item_id: str, label: str, reason: str) -> Dict[str, Any]:
    return {"id": item_id, "label": label, "role": "unresolved", "record_id": None, "source": {"type": "system", "label": "诊断缺口", "locator": "待确认"}, "observed_at": None, "verification_status": "unresolved", "reason": reason}


def treatment_reference() -> Dict[str, Any]:
    return {
        "authority": "decision_support",
        "label": "AI 指南路径与剂量草稿｜待医生签署",
        "status": "requires_doctor_confirmation",
        "goal": "控制可能的器官威胁性炎症、保护肺与肾功能，并降低治疗相关感染和药物毒性。",
        "pathways": [
            {"name": "诱导缓解路径", "description": "糖皮质激素联合利妥昔单抗或环磷酰胺等方案，由专科医生结合肾功能、感染、年龄、生育需求和既往治疗选择。"},
            {"name": "支持与预防", "description": "医生可能同时评估感染预防、骨保护、胃肠道风险和疫苗安排。"},
            {"name": "维持与复发监测", "description": "病情缓解后由医生制定维持治疗、减量与复诊监测计划。"},
        ],
        "prerequisites": ["确认是否存在活动性感染", "确认器官威胁程度与肾功能", "完成乙肝等免疫抑制前筛查", "核对生育计划、疫苗、既往用药与过敏史"],
        "major_risks": ["严重感染", "骨髓抑制或免疫功能下降", "糖皮质激素相关血糖、血压和骨骼风险", "输注反应或药物特异性毒性"],
        "monitoring": ["血常规", "肾功能与电解质", "尿检和尿蛋白", "血糖与血压", "感染征象", "医生指定的免疫学与药物安全监测"],
        "sources": [
            {"title": "KDIGO 2024 ANCA-Associated Vasculitis Guideline", "url": "https://kdigo.org/wp-content/uploads/2024/05/KDIGO-2024-ANCA-Vasculitis-Guideline.pdf", "version": "2024"},
            {"title": "ACR/Vasculitis Foundation ANCA-Associated Vasculitis Guideline", "url": "https://rheumatology.org/vasculitis-guideline", "version": "2021"},
        ],
        "limitations": "AI 可生成带指南来源的医生端剂量草稿；只有医生第二次签署后才能成为沙箱处方并创建患者提醒。",
    }


def assessment_version(journey: Dict[str, Any]) -> Dict[str, Any]:
    batches = list(journey.get("synced_batches") or ["baseline"])
    stage = max((sample_record_batches().get(key, {}).get("order", 0) for key in batches), default=1)
    previous = (journey.get("assessment_versions") or [])[-1] if journey.get("assessment_versions") else None
    definitions = {
        1: ("肺泡出血或其他肺部病变方向", "需要尽快排除危险病因", "症状和低氧提示不能按普通咳嗽处理，但尚无肺肾联合证据。"),
        2: ("肺肾综合征（系统性小血管炎方向）", "重要方向发生改变", "肺泡出血与肾小球性血尿、肾功能损害同时存在，单一肺部疾病无法覆盖全部异常。"),
        3: ("显微镜下多血管炎方向", "需要组织学或医生综合确诊", "MPO-ANCA 阳性与肺肾模式一致；抗GBM阴性及感染筛查阴性使部分鉴别方向下降。"),
        4: ("显微镜下多血管炎", "已有医生确证证据", "肾活检支持 ANCA 相关少免疫沉积性新月体性肾小球肾炎，最终诊断仍以医生记录为准。"),
    }
    name, status, reasoning = definitions[stage]
    version = len(journey.get("assessment_versions") or []) + 1
    evidence = journey.get("evidence") or []
    differentials = _differentials(stage)
    history_labels = {
        "conditions": "疾病史",
        "surgeries": "手术史",
        "current_medications": "当前用药",
        "allergies": "过敏史",
        "family_history": "家族史",
        "social_history": "个人与暴露史",
    }
    unknown_history = [
        f"{history_labels.get(key, key)}明确选择了‘不了解’"
        for key, status in (journey.get("clinical_history", {}).get("field_statuses") or {}).items()
        if status == "unknown"
    ]
    base_uncertainty = "low" if stage == 4 else "medium" if stage >= 2 else "high"
    uncertainty_level = "high" if unknown_history and stage < 4 else "medium" if unknown_history else base_uncertainty
    result = {
        "schema_version": "assessment-version.v1", "version": version, "batch_stage": stage,
        "batch_keys": batches, "created_at": utc_now(), "authority": "decision_support",
        "primary_diagnosis": {"name": name, "status": status, "reasoning": reasoning},
        "leading_direction": {"name": name, "status": status, "reasoning": reasoning},
        "differentials": differentials,
        "dangerous_conditions": _dangerous_conditions(stage),
        "missing_exams": _missing_exams(stage),
        "safety_matrix": _safety_matrix(stage),
        "change_from_previous": {"changed": bool(previous and previous.get("primary_diagnosis", {}).get("name") != name), "previous": previous.get("primary_diagnosis", {}).get("name") if previous else None, "current": name, "why": reasoning},
        "evidence_summary": {"supporting_count": sum(item.get("role") == "supporting" for item in evidence), "contradicting_count": sum(item.get("role") == "contradicting" for item in evidence), "unresolved_count": sum(item.get("role") == "unresolved" for item in evidence)},
        "uncertainty": {
            "level": uncertainty_level,
            "label": ("医生证据已回传，但病史仍有不了解项" if stage == 4 else "病史存在不了解项，需医生继续核对") if unknown_history else ("医生证据已回传" if stage == 4 else "仍需检查与医生确认"),
            "gaps": [item["name"] for item in _missing_exams(stage)] + unknown_history,
        },
        "care_navigation": {"departments": ["风湿免疫科", "肾内科", "呼吸科"], "materials": ["完整症状时间线", "既往史、用药史和过敏史", "尿检、肾功能与胸部 CT 原件", "当前全部药物清单"], "questions_for_doctor": ["肺与肾异常是否由同一病因造成？", "目前还需要排除哪些危险疾病？", "哪些检查会真正改变治疗选择？"], "exam_discussion_items": _exam_discussion_items()},
        "treatment_reference": treatment_reference() if stage >= 2 else None,
        "limitations": ["这是 AI 辅助判断，不是医生确诊。", "诊断版本只使用当时已经同步并确认的证据。", "AI 剂量建议必须经医生二次签署才能成为处方。"],
    }
    result["candidate_history"] = [{"stage": f"诊断版本 v{version}", "value": result["change_from_previous"]["why"], "candidates": [{"name": item["name"], "strength": item["level"], "trend": item["trend"], "reason": item["reason"]} for item in differentials]}]
    result["urgency"] = {"level": "urgent", "label": "建议当日线下评估", "reason": "咯血、低氧及潜在肺肾受累可能涉及器官威胁。"}
    return result


def _differentials(stage: int) -> List[Dict[str, Any]]:
    if stage == 1:
        rows = [("肺部感染", "中", "新出现", "血丝痰和呼吸困难可见于感染。", ["血丝痰", "心率增快"], ["无高热"], ["病原学与影像"]), ("肺栓塞", "中", "必须排查", "呼吸困难和低氧需要结合风险因素排查。", ["活动后呼吸困难", "指氧偏低"], ["无胸痛，仍不能排除"], ["医生风险评估及必要影像"]), ("肺泡出血", "中", "上升", "咯血和低氧需要警惕。", ["血丝痰"], [], ["胸部 CT、血红蛋白"])]
    elif stage == 2:
        rows = [("系统性小血管炎", "强", "上升", "肺泡出血和肾小球损害形成肺肾模式。", ["肺泡出血", "畸形红细胞", "肌酐升高"], [], ["ANCA、组织学"]), ("抗GBM病", "中", "必须排查", "可造成肺出血与快速进展性肾炎。", ["肺肾共同受累"], [], ["抗GBM抗体、肾活检"]), ("严重感染", "中", "待排除", "感染可模拟肺部表现并影响免疫抑制安全。", ["呼吸道表现"], ["肾小球性血尿难以完全解释"], ["培养及病原学"])]
    else:
        rows = [("显微镜下多血管炎", "强", "上升", "MPO-ANCA、肺泡出血与少免疫肾小球肾炎模式一致。", ["MPO-ANCA阳性", "肺肾共同受累"] + (["肾活检支持"] if stage == 4 else []), [], [] if stage == 4 else ["肾活检、医生综合确诊"]), ("抗GBM病", "低", "下降", "抗GBM抗体阴性构成反证，但由医生结合组织学判断。", ["肺肾表现"], ["抗GBM抗体阴性"], ["组织学对照"]), ("严重感染", "低", "下降", "病原学阴性降低感染方向，但不能绝对排除。", ["肺部表现"], ["培养与痰筛查阴性"], ["医生结合临床判断"]), ("肺栓塞", "低", "下降", "现有肺肾免疫模式提供更统一解释。", ["呼吸困难、低氧"], ["肾小球损害和MPO-ANCA不能由肺栓塞解释"], ["如风险评估仍高再检查"])]
    return [{"name": name, "level": level, "trend": trend, "reason": reason, "supporting": supporting, "contradicting": contradicting, "unresolved": unresolved} for name, level, trend, reason, supporting, contradicting, unresolved in rows]


def _dangerous_conditions(stage: int) -> List[Dict[str, str]]:
    return [
        {"name": "肺泡出血/呼吸衰竭", "status": "high_concern" if stage >= 2 else "must_exclude", "evidence": "咯血、低氧" + ("及 CT 肺泡出血表现" if stage >= 2 else ""), "action": "症状加重、静息气促或大量咯血立即急诊"},
        {"name": "肺栓塞", "status": "assessed_not_closed", "evidence": "呼吸困难与低氧需要临床风险评估", "action": "由医生决定是否需要 D-二聚体或肺动脉影像"},
        {"name": "严重感染/脓毒症", "status": "lowered_not_excluded" if stage >= 3 else "must_exclude", "evidence": "感染会影响免疫抑制安全", "action": "持续监测体温、培养和感染征象"},
        {"name": "抗GBM病", "status": "lowered_not_excluded" if stage >= 3 else "must_exclude", "evidence": "可造成肺肾综合征", "action": "结合抗体和组织学由医生判断"},
        {"name": "快速进展性肾损伤", "status": "high_concern" if stage >= 2 else "must_exclude", "evidence": "肌酐趋势和尿沉渣决定风险", "action": "尽快复查肾功能并由肾内科评估"},
    ]


def _safety_matrix(stage: int) -> List[Dict[str, Any]]:
    rows = [
        ("alveolar-hemorrhage", "肺泡出血/呼吸衰竭", "咯血量可能不大，但肺内出血和低氧仍可能快速加重。", "紧急", ["oxygenation", "blood-safety"], ["静息及活动后指氧", "必要时动脉血气", "血红蛋白动态", "胸部影像"], "明显低氧或血红蛋白下降会提高住院与紧急处置优先级。"),
        ("pulmonary-embolism", "肺栓塞", "呼吸困难和低氧并不只见于血管炎，若风险因素被遗漏可能误判。", "条件性高风险", ["pe-risk"], ["临床肺栓塞风险评估", "条件性 D-二聚体", "条件性肺动脉影像"], "风险评估为低风险可避免不必要影像；中高风险结果会改变紧急处置。"),
        ("severe-infection", "严重感染/脓毒症", "感染既可模拟肺部表现，也会显著改变免疫抑制治疗时机。", "必须排查", ["infection-screen", "blood-safety"], ["血常规", "培养", "呼吸道病原学", "免疫抑制前感染筛查"], "阳性结果会推迟或改变免疫抑制路径；阴性只能降低而不能绝对排除。"),
        ("anti-gbm", "抗 GBM 病", "同样可造成肺出血与快速进展性肾炎，遗漏会影响紧急治疗选择。", "必须排查", ["pulmonary-renal-cause", "renal-biopsy"], ["抗 GBM 抗体", "尿沉渣", "肾活检病理对照"], "抗体和组织学结果会改变 AAV 与抗 GBM 病的排序。"),
        ("rapid-kidney-injury", "快速进展性肾损伤", "单次肌酐不能说明变化速度，高钾或酸中毒可能需要立即处理。", "紧急", ["renal-trend", "glomerular-injury", "renal-biopsy"], ["肌酐/eGFR/电解质动态", "尿沉渣", "尿蛋白定量", "肾活检评估"], "快速恶化或危险电解质异常会提高处置紧迫性并影响治疗选择。"),
    ]
    status_by_id = {
        "alveolar-hemorrhage": "高度关注" if stage >= 2 else "尚未排除",
        "pulmonary-embolism": "已评估但未关闭",
        "severe-infection": "已降低但未排除" if stage >= 3 else "尚未排除",
        "anti-gbm": "已降低但未排除" if stage >= 3 else "尚未排除",
        "rapid-kidney-injury": "高度关注" if stage >= 2 else "尚未排除",
    }
    return [
        {
            "condition_id": key,
            "condition_name": name,
            "why_it_might_be_missed": why,
            "risk_level": risk,
            "current_status": status_by_id[key],
            "supporting_evidence": [item["evidence"] for item in _dangerous_conditions(stage) if item["name"] == name][:1],
            "contradicting_evidence": ["现有阴性证据只用于降权，不能单独完成排除。"] if stage >= 3 and key in {"severe-infection", "anti-gbm"} else [],
            "exam_links": [{"recommendation_id": recommendation_id} for recommendation_id in recommendation_ids],
            "exam_items": items,
            "expected_result_effect": impact,
        }
        for key, name, why, risk, recommendation_ids, items, impact in rows
    ]


def _missing_exams(stage: int) -> List[Dict[str, str]]:
    items = _exam_discussion_items()
    if stage == 1:
        return items[:5]
    if stage == 2:
        return items[3:]
    if stage == 3:
        return items[-1:]
    return [{"name": "按医生计划复查血常规、肾功能和尿检", "purpose": "监测病情与治疗安全", "priority": "复诊计划", "status": "doctor_ordered"}]


def _exam_discussion_items() -> List[Dict[str, str]]:
    return [
        {"name": "尿沉渣镜检和尿蛋白定量", "purpose": "确认肾小球性血尿并量化肾损害", "priority": "优先向医生确认", "status": "pending_doctor_confirmation"},
        {"name": "复查肾功能、电解质与 eGFR", "purpose": "观察肾功能变化速度", "priority": "优先向医生确认", "status": "pending_doctor_confirmation"},
        {"name": "胸部影像与氧合评估", "purpose": "评估肺泡出血及呼吸风险", "priority": "当日评估", "status": "pending_doctor_confirmation"},
        {"name": "MPO/PR3-ANCA、抗GBM、补体和ANA谱", "purpose": "鉴别肺肾综合征病因", "priority": "由专科医生选择", "status": "pending_doctor_confirmation"},
        {"name": "感染病原学检查", "purpose": "评估是否存在影响免疫抑制安全的感染", "priority": "就诊时确认", "status": "pending_doctor_confirmation"},
        {"name": "肾活检必要性与可行性评估", "purpose": "获得组织学证据", "priority": "由肾内科决定", "status": "pending_doctor_confirmation"},
    ]


def answer_consultation(text: str, danger_signs: Iterable[str]) -> Dict[str, Any]:
    clean = (text or "").strip()[:500]
    signs = [item for item in danger_signs if item in {"active_hemoptysis", "severe_dyspnea", "altered_consciousness", "low_oxygen"}]
    if signs:
        return _answer("emergency", "现在需要立即急诊，不要等待在线分析。", "存在咯血、静息呼吸困难、意识改变或低氧等危险信号。", ["咯血量是否增加？", "能否完整说话和行走？", "指氧是否低于 93%？"], "立即拨打 120 或前往最近急诊。")
    if any(word in clean for word in ("急诊", "严重", "咯血", "呼吸", "危险")):
        return _answer("urgent", "这组症状不能等到普通体检处理，建议今天完成线下评估。", "血丝痰、呼吸困难和指氧偏低需要先排除肺出血、肺栓塞和严重感染。", ["现在是否仍在咯血？", "静息时是否气促？", "是否有胸痛、晕厥或发热？"], "若出现大量咯血、静息气促、意识异常或指氧下降，立即急诊。")
    if any(word in clean for word in ("挂", "科", "门诊")):
        return _answer("department", "当前优先考虑急诊或呼吸科完成初始评估；发现肾脏或免疫异常后再联合肾内科和风湿免疫科。", "科室选择应随证据变化，而不是在检查前假设唯一诊断。", ["是否已有尿检、肾功能或胸部 CT？"], "携带全部原始报告，由首诊医生决定联合会诊。")
    if any(word in clean for word in ("医生", "问我", "追问")):
        return _answer("doctor_questions", "医生通常会重点追问咯血量、呼吸困难程度、时间线、感染症状、血栓风险和肾脏相关表现。", "这些信息会改变急诊分流和鉴别诊断。", ["咯血是血丝还是鲜血？", "是否发热、胸痛、下肢肿痛？", "尿色、尿量是否变化？", "近期用了哪些药？"], "提前整理症状时间线、全部用药和过敏情况。")
    if any(word in clean for word in ("准备", "检查", "病史", "带什么")):
        return _answer("preparation", "请准备症状时间线、既往病史、全部用药与过敏史，以及已有检验和影像原件。", "完整来源能减少重复检查，并帮助医生判断新结果是否真正改变诊断。", ["是否有既往肾功能基线？", "是否服用止痛药、抗凝药或保健品？"], "不要自行开检查；把清单带给医生确认。")
    return _answer("general", "我可以先帮你判断紧急程度，再整理病史、鉴别诊断和就医准备。", "安全分流必须先于诊断解释。", ["你最担心的症状是什么？", "症状从什么时候开始？", "现在是否存在咯血、静息气促或意识改变？"], "从当前最严重的症状和时间线开始描述。")


def _answer(intent: str, direct: str, basis: str, followups: List[str], action: str) -> Dict[str, Any]:
    return {"intent": intent, "direct_answer": direct, "urgency": "emergency" if intent == "emergency" else "same_day" if intent == "urgent" else "guided", "basis": basis, "follow_up_questions": followups, "next_action": action, "boundary": "这是就医导航与信息整理，不是医生诊断或处方。", "fallback_used": True}


def medication_education(name: str) -> Dict[str, Any]:
    normalized = name.replace("（虚构沙箱处方）", "").strip()
    entries = {
        "利妥昔单抗": {"purpose": "沙箱医生用于 ANCA 相关血管炎诱导缓解的院内输注治疗。", "common_effects": ["输注反应", "疲乏", "感染风险增加"], "urgent_warnings": ["呼吸困难、面唇肿胀或严重皮疹", "持续高热或明显感染症状"], "interactions": ["接种疫苗和其他免疫抑制治疗需由医生统筹"], "monitoring": ["乙肝筛查", "血常规", "免疫球蛋白", "感染征象"], "missed_dose": "院内输注时间变更时联系治疗团队，不自行补用。"},
        "泼尼松": {"purpose": "沙箱医生用于控制炎症；减量必须遵循医生记录。", "common_effects": ["食欲和睡眠变化", "血糖、血压升高", "胃部不适"], "urgent_warnings": ["高热或严重感染症状", "黑便、呕血", "明显情绪或意识改变"], "interactions": ["与布洛芬等 NSAIDs 合用可能增加胃肠道风险"], "monitoring": ["血压", "血糖", "感染征象", "骨骼风险"], "missed_dose": "不要自行加倍或突然停药，联系医生或药师确认。"},
        "复方磺胺甲噁唑": {"purpose": "沙箱医生用于免疫抑制期间的感染预防。", "common_effects": ["恶心", "皮疹"], "urgent_warnings": ["严重皮疹或黏膜损伤", "呼吸困难", "不明原因发热"], "interactions": ["肾功能异常时剂量和监测由医生决定", "与部分影响血钾或凝血的药物可能相互作用"], "monitoring": ["血常规", "肾功能", "血钾", "皮疹和过敏"], "missed_dose": "按处方说明联系医生或药师，不自行补双倍。"},
        "碳酸钙D3": {"purpose": "沙箱医生用于糖皮质激素治疗期间的骨健康支持。", "common_effects": ["便秘", "胃部不适"], "urgent_warnings": ["持续呕吐、意识异常或明显乏力时就医"], "interactions": ["可能影响部分甲状腺药、铁剂或抗菌药吸收，需错开并咨询药师"], "monitoring": ["血钙", "肾功能", "医生认为需要时评估维生素D"], "missed_dose": "不要加倍，按下一次医生处方时间继续。"},
    }
    base = entries.get(normalized)
    if not base:
        return {"review_status": "missing", "notice": "暂无经过审核的药物说明，请仅按医生处方执行并咨询医生或药师。"}
    return {**deepcopy(base), "review_status": "clinician_review_required", "knowledge_source": "ClinicaLens 沙箱静态药物条目", "reviewed_on": "2026-09-02", "notice": "说明不能替代医生或药师指导，也不能用于自行改量、停药或换药。"}


def hydrate_journey(journey: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade persisted v1 journeys without deleting user-owned fields."""
    upgraded = deepcopy(journey)
    upgraded["schema_version"] = "care-journey.v4"
    upgraded.setdefault("patient_profile", sample_patient_profile())
    upgraded.setdefault("clinical_history", sample_clinical_history(confirmed=False))
    upgraded.setdefault("consultation", {"messages": [], "quick_questions": quick_questions()})
    upgraded.setdefault("raw_case_document", sample_raw_case_document())
    if not upgraded["raw_case_document"].get("origin"):
        legacy = deepcopy(upgraded["raw_case_document"])
        legacy["origin"] = "legacy_sample"
        legacy["status"] = "legacy"
        legacy["notice"] = "旧版病例已保留归档；它没有问诊消息级来源，不能作为当前问诊原文。"
        upgraded.setdefault("legacy_case_documents", []).append(legacy)
        upgraded["raw_case_document"] = build_consultation_case_document(
            upgraded["consultation"].get("messages", []), upgraded["clinical_history"], version=1
        )
        upgraded["consultation_case_documents"] = [deepcopy(upgraded["raw_case_document"])]
    else:
        upgraded.setdefault("consultation_case_documents", [deepcopy(upgraded["raw_case_document"])])
    upgraded.setdefault("synced_batches", [])
    upgraded.setdefault("assessment_versions", [])
    if upgraded.get("assessment") and not upgraded["assessment_versions"]:
        old = deepcopy(upgraded["assessment"])
        old.setdefault("version", 1)
        old.setdefault("primary_diagnosis", deepcopy(old.get("leading_direction", {})))
        upgraded["assessment_versions"] = [old]
    upgraded.setdefault("treatment_reference", treatment_reference())
    upgraded.setdefault("doctor_plan", None)
    upgraded.setdefault("medications", [])
    upgraded.setdefault("prescription_drafts", [])
    upgraded.setdefault("signed_prescriptions", [])
    return upgraded


def quick_questions() -> List[str]:
    return ["咯血和呼吸困难严重吗？", "我现在应该去急诊吗？", "应该先看呼吸科、肾内科还是风湿免疫科？", "医生可能会追问什么？", "就医前需要准备哪些病史和检查？"]


def public_sample_journey() -> Dict[str, Any]:
    journey = hydrate_journey({
        "id": "public-fictional-journey", "owner_id": "public-sample", "title": "肺与肾的多项异常需要一起理解",
        "status": "completed_sample", "current_stage": "medication_active", "created_at": "2026-09-02T08:00:00+08:00", "updated_at": "2026-09-02T18:00:00+08:00",
        "hospital_connection": {"display_name": "完整虚构病例", "mode": "sandbox", "status": "connected"},
        "triage": {"status": "stable", "danger_signs": [], "checked_at": "2026-09-02T08:10:00+08:00", "message": "当前无大咯血、静息气促或意识异常；由于低氧和后续肺肾异常，仍需当日线下评估。"},
        "records": [], "evidence": [], "assessment": None, "appointment_plan": None, "followups": [], "reminders": [], "timeline": [], "consents": {"hospital": True, "ai_analysis": True, "push": False},
    })
    journey["clinical_history"] = sample_clinical_history(confirmed=True)
    journey["consultation"]["messages"] = sample_consultation_messages()
    journey["raw_case_document"] = sample_raw_case_document()
    journey["consultation_case_documents"] = [deepcopy(journey["raw_case_document"])]
    journey["synced_batches"] = ["baseline", "organ", "serology", "biopsy"]
    journey["records"] = all_sample_records()
    for record in journey["records"]:
        record["verification_status"] = "user_confirmed"
    journey["evidence"] = build_evidence(journey["records"], journey["synced_batches"])
    for stage in range(1, 5):
        staged = deepcopy(journey)
        staged["synced_batches"] = list(sample_record_batches())[:stage]
        allowed = {record["id"] for key in staged["synced_batches"] for record in sample_record_batches()[key]["records"]}
        staged["records"] = [record for record in journey["records"] if record["id"] in allowed]
        staged["evidence"] = build_evidence(staged["records"], staged["synced_batches"])
        staged["assessment_versions"] = deepcopy(journey["assessment_versions"])
        journey["assessment_versions"].append(assessment_version(staged))
    journey["assessment"] = deepcopy(journey["assessment_versions"][-1])
    journey["treatment_reference"] = treatment_reference()
    doctor_source = {"type": "sandbox_hospital", "label": "完整虚构病例 · 沙箱医生出院记录", "import_id": None}
    journey["doctor_plan"] = {
        "authority": "doctor_plan", "diagnoses": ["显微镜下多血管炎（完整虚构医生诊断）"],
        "source": doctor_source, "verification_status": "doctor_confirmed", "confirmed_at": "2026-09-02T16:00:00+08:00",
        "care_summary": "完成院内诱导治疗后出院，按医生处方执行并由风湿免疫科、肾内科联合复诊。",
        "comparison": {
            "result": "confirmed_with_revision_history",
            "ai_assessment_version_id": "assessment-v4",
            "ai_primary": "显微镜下多血管炎",
            "confirmed_evidence": ["MPO-ANCA 阳性与肺肾共同受累模式一致", "肾活检支持 ANCA 相关小血管炎"],
            "revisions": ["v1 的单一肺部方向在尿检、肾功能进入后被修正为肺肾综合征", "感染筛查阴性只降低感染方向，没有把感染绝对排除"],
        },
        "examination_orders": [{"name": name, "status": "doctor_ordered", "source": "doctor_plan"} for name in ["血常规", "肾功能与电解质", "尿常规与尿蛋白", "血糖与血压", "感染及乙肝筛查"]],
        "treatments": [{"name": "利妥昔单抗", "route": "静脉输注", "schedule": "1 g，第 1、15 天，仅限院内执行", "source": "doctor_plan"}],
    }
    prescriptions = [
        ("利妥昔单抗", "1 g", "第 1、15 天院内输注", "静脉输注", "诱导缓解", "2026-09-16T09:00:00+08:00"),
        ("泼尼松", "30 mg", "每晨一次 7 天，随后 25 mg 每晨一次 14 天，再由复诊医生调整", "口服", "控制炎症", "2026-09-03T08:00:00+08:00"),
        ("复方磺胺甲噁唑", "400/80 mg", "每日一次", "口服", "免疫抑制期间感染预防", "2026-09-03T20:00:00+08:00"),
        ("碳酸钙D3", "600 mg/400 IU", "每日一次", "口服", "骨健康支持", "2026-09-03T12:00:00+08:00"),
    ]
    journey["medications"] = [
        {"id": f"sample-medication-{index}", "name": name, "dose": dose, "frequency": frequency, "course": "按完整虚构医生出院记录执行", "route": route, "purpose": purpose, "prescription_original": f"{name} {dose}，{frequency}", "next_at": next_at, "status": "active", "source": deepcopy(doctor_source), "events": [], "education": medication_education(name), "boundary": "仅执行虚构医生处方；不得据此自行用药。"}
        for index, (name, dose, frequency, route, purpose, next_at) in enumerate(prescriptions, start=1)
    ]
    journey["confirmed_treatment_direction"] = {
        "title": "器官威胁型 ANCA 相关血管炎治疗路径参考", "status": "confirmed",
        "rationale": "沙箱医生结合肺泡出血、肾功能和活检结果确认该路径。",
        "source": "clinician", "confirmed_at": "2026-09-02T16:15:00+08:00",
        "boundary": "医生已确认路径并完成第二次处方签署。",
    }
    signed_items = [
        {"id": f"sample-signed-item-{index}", "medication": medication[0], "dose": medication[1], "frequency": medication[2], "duration": "按完整虚构医生出院记录执行", "route": medication[3], "purpose": medication[4], "dose_source": "KDIGO 2024 Figure 10" if index == 1 else "AI 指南减量路径，经沙箱医生修改" if index == 2 else "沙箱医生依据院内规则补充", "origin": "ai_guideline" if index == 1 else "clinician_modified" if index == 2 else "clinician_added", "requires": [], "schedule": []}
        for index, medication in enumerate(prescriptions, start=1)
    ]
    journey["signed_prescriptions"] = [{
        "id": "sample-signed-prescription-1", "draft_id": "sample-prescription-draft-1",
        "recommendation_id": "aav-guideline-path", "assessment_version_id": "assessment-v4",
        "diagnosis_reference": journey["doctor_plan"]["diagnoses"][0], "items": signed_items,
        "rationale": "结合医生确诊、肾功能和感染筛查完成沙箱签署。", "status": "signed",
        "source": {"type": "clinician_signed_ai_path", "label": "沙箱医生签署的 AI 指南路径处方"},
        "signed_at": "2026-09-02T16:20:00+08:00", "notice": "完整虚构沙箱处方，未向真实医院或药房发送。",
    }]
    signed_source = {"type": "clinician_signed_ai_path", "label": "沙箱医生签署的 AI 指南路径处方", "draft_id": "sample-prescription-draft-1"}
    for medication in journey["medications"]:
        medication["source"] = deepcopy(signed_source)
        medication["boundary"] = "按完整虚构医生签署处方执行；不得据此自行用药。"
    journey["prescription_drafts"] = [{
        "id": "sample-prescription-draft-1", "recommendation_id": "aav-guideline-path",
        "assessment_version_id": "assessment-v4", "diagnosis_reference": journey["doctor_plan"]["diagnoses"][0],
        "items": deepcopy(signed_items), "status": "signed", "clinician_id": "sandbox-clinician",
        "created_at": "2026-09-02T16:17:00+08:00", "signed_at": "2026-09-02T16:20:00+08:00",
        "signed_prescription_id": "sample-signed-prescription-1",
    }]
    journey["treatment_provenance"] = [
        {"type": "ai_guideline_path", "label": "AI 提供指南路径", "at": "2026-09-02T16:12:00+08:00"},
        {"type": "clinician_path_confirmed", "label": "医生确认治疗路径", "at": "2026-09-02T16:15:00+08:00"},
        {"type": "prescription_draft_created", "label": "AI 生成结构化剂量草稿", "at": "2026-09-02T16:17:00+08:00"},
        {"type": "clinician_prescription_signed", "label": "医生修改、补充并签署处方", "at": "2026-09-02T16:20:00+08:00"},
        {"type": "patient_reminders_created", "label": "患者提醒与执行任务已创建", "at": "2026-09-02T16:21:00+08:00"},
    ]
    journey["reminders"] = [
        {"id": f"sample-reminder-{index}", "kind": "infusion" if medication["route"] == "静脉输注" else "medication", "medication_id": medication["id"], "scheduled_at": medication["next_at"], "status": "scheduled", "source": deepcopy(signed_source)}
        for index, medication in enumerate(journey["medications"], start=1)
    ]
    journey["followups"] = [{"id": "sample-followup-1", "title": "风湿免疫科与肾内科联合复诊", "scheduled_at": "2026-09-23T09:00:00+08:00", "status": "scheduled", "source": deepcopy(doctor_source)}]
    return journey


__all__ = ["all_sample_records", "answer_consultation", "assessment_version", "build_consultation_case_document", "build_evidence", "hydrate_journey", "medication_education", "public_sample_journey", "quick_questions", "sample_clinical_history", "sample_consultation_messages", "sample_patient_profile", "sample_raw_case_document", "sample_record_batches", "treatment_reference"]
