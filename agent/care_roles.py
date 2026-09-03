"""Role-aware v4 projections and curated sample clinical content.

The module contains deterministic, reviewable sample data only. Dose options
remain clinician-only drafts until a clinician signs them.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

from agent.care_product import treatment_reference, utc_now


GUIDELINES = [
    {
        "name": "KDIGO 2024 ANCA-Associated Vasculitis Guideline",
        "url": "https://kdigo.org/wp-content/uploads/2024/05/KDIGO-2024-ANCA-Vasculitis-Guideline.pdf",
    },
    {
        "name": "ACR/VF Vasculitis Guideline",
        "url": "https://rheumatology.org/vasculitis-guideline",
    },
]

REPORT_RECORD_IDS = {
    "report-vitals": ["record-vitals"],
    "report-urine": ["record-urinalysis"],
    "report-renal": ["record-renal"],
    "report-ct": ["record-ct"],
    "report-immunology": ["record-anca"],
    "report-infection": ["record-infection"],
    "report-biopsy": ["record-biopsy"],
}


def _source(locator: str) -> Dict[str, str]:
    return {
        "type": "sandbox_hospital",
        "label": "完整虚构病例 · 沙箱医院",
        "locator": locator,
    }


def _observation(
    code: str,
    name: str,
    value: str,
    *,
    unit: str = "",
    reference_range: str = "",
    status: str,
    explanation: str,
    impact: str,
    locator: str,
    evidence_role: str = "supporting",
    trend: str = "当前批次首次记录",
) -> Dict[str, Any]:
    return {
        "id": f"observation-{code}",
        "code": code,
        "name": name,
        "value": value,
        "unit": unit,
        "reference_range": reference_range or None,
        "reference_range_display": reference_range or "原报告未提供",
        "interpretation_status": status,
        "patient_explanation": explanation
        or "暂无经过审核的通俗解释，请咨询医生。",
        "diagnostic_impact": impact,
        "evidence_role": evidence_role,
        "trend": trend,
        "entered_assessment_version": None,
        "source_locator": locator,
        "verification_status": "hospital_confirmed",
        "disputed": False,
    }


def sample_exam_reports() -> List[Dict[str, Any]]:
    """Structured reports matching the single fictional lung-kidney journey."""
    report_specs = [
        (
            "report-vitals", "baseline", "生命体征", "CL-VITAL-20260901", "2026-09-01T08:45:00+08:00",
            [
                _observation("temp", "体温", "37.4", unit="℃", reference_range="36.0–37.3", status="偏高", explanation="体温略高，但单次轻度升高不能单独证明感染。", impact="保留感染方向，需要结合症状和病原学结果。", locator="生命体征第1项"),
                _observation("pulse", "心率", "102", unit="次/分", reference_range="60–100", status="偏高", explanation="心率略快，可见于紧张、缺氧、炎症或其他应激状态。", impact="提示需要结合氧合和全身状态评估严重程度。", locator="生命体征第2项"),
                _observation("rr", "呼吸频率", "22", unit="次/分", reference_range="12–20", status="偏高", explanation="呼吸次数偏快，说明身体可能正在增加呼吸代偿。", impact="与呼吸困难和低氧一起支持尽快线下评估。", locator="生命体征第3项"),
                _observation("bp", "血压", "148/92", unit="mmHg", reference_range="原报告未提供", status="偏高", explanation="本次血压偏高，需要结合既往高血压和重复测量判断。", impact="影响肾脏风险和后续治疗监测，但不能解释全部肺部异常。", locator="生命体征第4项"),
                _observation("spo2", "静息指氧", "94", unit="%（空气）", reference_range="95–100", status="偏低", explanation="血氧低于报告参考范围，说明氧合可能受影响。", impact="提高肺泡出血、感染或肺血管问题的紧急评估优先级。", locator="生命体征第5项"),
            ],
        ),
        (
            "report-urine", "organ", "尿常规与尿沉渣", "CL-URINE-20260901", "2026-09-01T10:10:00+08:00",
            [
                _observation("urine-rbc", "尿红细胞", "50", unit="个/HPF", reference_range="0–3", status="偏高", explanation="尿中红细胞明显增多，提示泌尿系统存在出血来源。", impact="与畸形红细胞同时出现时，更支持肾小球来源的损害。", locator="尿常规第8项"),
                _observation("urine-protein", "尿蛋白", "2+", reference_range="阴性", status="阳性", explanation="尿蛋白阳性说明肾脏过滤屏障可能受损，但仍需定量。", impact="与血尿和肌酐升高一起支持肾脏受累。", locator="尿常规第11项"),
                _observation("dysmorphic-rbc", "畸形红细胞", "可见", reference_range="未见", status="阳性", explanation="红细胞形态异常通常提示血液更可能来自肾小球。", impact="把诊断从单纯肺部疾病推向肺肾共同病因。", locator="尿沉渣镜检结论"),
            ],
        ),
        (
            "report-renal", "organ", "肾功能", "CL-RENAL-20260901", "2026-09-01T10:20:00+08:00",
            [
                _observation("creatinine", "血肌酐", "220", unit="μmol/L", reference_range="44–133", status="偏高", explanation="肌酐升高提示肾脏清除废物的能力下降。", impact="与尿沉渣异常共同提示需要快速评估肾损伤及其进展速度。", locator="生化检验第14项"),
                _observation("egfr", "eGFR", "29", unit="ml/min/1.73m²", reference_range="≥90", status="偏低", explanation="估算肾小球滤过率偏低，表示当前肾功能明显下降。", impact="提高快速进展性肾损伤的关注度，需要动态复查。", locator="生化检验第15项"),
            ],
        ),
        (
            "report-ct", "organ", "胸部 CT", "CL-CT-20260901", "2026-09-01T11:30:00+08:00",
            [
                _observation("ct-ggo", "双肺磨玻璃影", "双肺弥漫分布", status="异常", explanation="磨玻璃影表示肺内有异常密度，但感染、出血等多种原因都可能出现。", impact="结合咯血和低氧，需要重点评估肺泡出血。", locator="胸部CT所见第2段"),
                _observation("ct-dah", "影像印象", "考虑弥漫性肺泡出血", status="异常", explanation="影像医生认为肺泡内可能存在出血，需要结合血红蛋白、氧合和临床表现确认。", impact="与肾小球损害结合后形成肺肾综合征模式。", locator="胸部CT印象第1条"),
            ],
        ),
        (
            "report-immunology", "serology", "ANCA 与鉴别抗体", "CL-IMMUNE-20260901", "2026-09-01T14:10:00+08:00",
            [
                _observation("mpo-anca", "MPO-ANCA", "86", unit="RU/ml", reference_range="报告判定：阳性", status="阳性", explanation="该抗体阳性可见于部分 ANCA 相关小血管炎，但不能脱离临床表现单独确诊。", impact="与肺肾共同受累一致，使显微镜下多血管炎方向升高。", locator="免疫学第3项"),
                _observation("pr3-anca", "PR3-ANCA", "阴性", reference_range="阴性", status="阴性", explanation="本次未检出 PR3-ANCA；阴性不能单独排除所有相关疾病。", impact="使以 PR3-ANCA 为常见关联的方向相对下降。", locator="免疫学第4项", evidence_role="contradicting"),
                _observation("anti-gbm", "抗 GBM 抗体", "阴性", reference_range="阴性", status="阴性", explanation="本次未检出抗 GBM 抗体，降低抗 GBM 病方向，但仍需医生结合组织学。", impact="构成反证，降低抗 GBM 病排序但不做绝对排除。", locator="免疫学第5项", evidence_role="contradicting"),
                _observation("complement", "补体 C3/C4", "正常", reference_range="原报告分项范围见附件", status="正常", explanation="补体结果在医院报告范围内，帮助医生区分部分免疫性疾病。", impact="使低补体相关方向相对下降。", locator="免疫学第8–9项", evidence_role="contradicting"),
            ],
        ),
        (
            "report-infection", "serology", "感染筛查", "CL-INFECT-20260901", "2026-09-01T14:20:00+08:00",
            [
                _observation("blood-culture", "血培养", "未检出细菌", reference_range="未检出", status="阴性", explanation="本次培养没有检出细菌，但一次阴性不能排除全部感染。", impact="降低细菌感染作为统一解释的可能，免疫抑制前仍需医生完成感染评估。", locator="微生物报告结论", evidence_role="contradicting"),
                _observation("sputum-pathogen", "痰病原学筛查", "阴性", reference_range="阴性", status="阴性", explanation="本次痰样本没有检出筛查范围内病原体，仍受样本质量和检测范围限制。", impact="进一步降低感染方向，但不会把感染绝对排除。", locator="呼吸道病原学结论", evidence_role="contradicting"),
            ],
        ),
        (
            "report-biopsy", "biopsy", "肾活检病理", "CL-PATH-20260902", "2026-09-02T10:00:00+08:00",
            [
                _observation("renal-pathology", "组织学结论", "少免疫沉积性坏死性新月体性肾小球肾炎", status="异常", explanation="肾组织显示一种符合 ANCA 相关小血管炎模式的严重肾小球损伤。", impact="提供关键组织学证据，支持医生确认显微镜下多血管炎。", locator="肾活检病理诊断"),
            ],
        ),
    ]
    reports: List[Dict[str, Any]] = []
    version_by_batch = {"baseline": "v1", "organ": "v2", "serology": "v3", "biopsy": "v4"}
    for report_id, batch_key, title, report_no, observed_at, observations in report_specs:
        for observation in observations:
            observation["entered_assessment_version"] = version_by_batch.get(batch_key)
        reports.append({
            "id": report_id,
            "batch_key": batch_key,
            "source": _source(title),
            "hospital": "完整虚构病例 · 沙箱医院",
            "title": title,
            "report_no": report_no,
            "observed_at": observed_at,
            "received_at": observed_at,
            "verification_status": "hospital_confirmed",
            "observations": observations,
            "record_ids": deepcopy(REPORT_RECORD_IDS.get(report_id, [])),
            "dispute": None,
        })
    return reports


def reports_for_batches(batch_keys: Iterable[str]) -> List[Dict[str, Any]]:
    allowed = set(batch_keys)
    return [deepcopy(report) for report in sample_exam_reports() if report["batch_key"] in allowed]


def legacy_reports_from_records(records: Iterable[Dict[str, Any]], covered_record_ids: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Preserve old string observations without inventing ranges or explanations."""
    covered = set(covered_record_ids)
    reports: List[Dict[str, Any]] = []
    for record in records:
        if record.get("id") in covered:
            continue
        if record.get("kind") in {"symptom", "history", "note"}:
            continue
        observations = []
        for index, raw in enumerate(record.get("items") or []):
            observations.append({
                "id": f"legacy-{record.get('id', 'record')}-{index}", "code": "legacy_observation",
                "name": str(raw), "value": str(raw), "unit": "", "reference_range": None,
                "reference_range_display": "原报告未提供", "interpretation_status": "待医生核对",
                "patient_explanation": "暂无经过审核的通俗解释，请咨询医生。",
                "diagnostic_impact": "旧版字符串记录已保留，但需核对原报告后才能解释诊断影响。",
                "evidence_role": "unresolved", "source_locator": record.get("source", {}).get("locator") or "旧版记录",
                "trend": "旧版记录未提供趋势", "entered_assessment_version": "legacy",
                "verification_status": record.get("verification_status") or "unconfirmed", "disputed": False,
            })
        reports.append({
            "id": f"legacy-report-{record.get('id', len(reports))}", "batch_key": "legacy",
            "source": deepcopy(record.get("source") or {"type": "legacy", "label": "旧版记录", "locator": "未提供"}),
            "hospital": "旧版数据（医院信息未提供）", "title": record.get("title") or "旧版检查记录",
            "report_no": "原报告未提供", "observed_at": record.get("observed_at"), "received_at": record.get("observed_at"),
            "verification_status": record.get("verification_status") or "unconfirmed", "observations": observations,
            "record_ids": [record.get("id")], "dispute": None,
        })
    return reports


def patient_explanation(assessment: Dict[str, Any], doctor_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    stage = int(assessment.get("batch_stage") or assessment.get("version") or 1)
    support = {
        1: ["血丝痰和活动后呼吸困难", "静息指氧偏低", "呼吸频率和心率偏快"],
        2: ["CT 提示肺泡出血", "尿中出现畸形红细胞和蛋白", "肌酐升高、eGFR 下降"],
        3: ["MPO-ANCA 阳性", "肺部出血与肾小球损伤同时存在", "感染和抗 GBM 筛查构成反证"],
        4: ["MPO-ANCA 与肺肾模式一致", "肾活检支持 ANCA 相关小血管炎", "医生记录完成最终确认"],
    }[min(stage, 4)]
    contradictions = [
        item for differential in assessment.get("differentials", [])
        for item in differential.get("contradicting", [])
    ][:3]
    missing = [item.get("name") for item in assessment.get("missing_exams", [])][:4]
    dangerous = [
        {
            "name": item.get("name"), "status": item.get("status"), "action": item.get("action"),
            "exams": next((link.get("exam_items", []) for link in assessment.get("safety_matrix", []) if link.get("condition_name") == item.get("name")), []),
        }
        for item in assessment.get("dangerous_conditions", [])
    ]
    confirmed = bool(doctor_plan and doctor_plan.get("verification_status") == "doctor_confirmed")
    medical_name = assessment.get("primary_diagnosis", {}).get("name", "当前仍需补充信息")
    plain_summary = (
        "肺部异常和肾脏损伤可能来自同一个全身性炎症问题。"
        if stage >= 2 else "目前先要确认肺部是否存在需要紧急处理的问题。"
    )
    why_steps = ["血丝痰、活动后气短和血氧偏低提示肺部需要尽快检查。"]
    if stage >= 2:
        why_steps.extend([
            "肺部影像提示异常，尿检和肾功能也提示肾脏同时受到影响。",
            "两个器官同时异常，比两个互不相关的小问题更需要考虑共同原因。",
        ])
    else:
        why_steps.append("目前还没有足够检查说明肺部异常的具体原因。")
    if stage >= 3:
        why_steps.append("血液中的 MPO-ANCA 结果支持免疫系统相关的小血管炎症。")
    else:
        why_steps.append("还需要免疫相关检查帮助判断两处异常是否由同一原因造成。")
    why_steps.append(f"把这些信息放在一起后，目前更支持“{medical_name}”。")
    return {
        "id": f"patient-explanation-v{assessment.get('version', stage)}",
        "assessment_version_id": f"assessment-v{assessment.get('version', stage)}",
        "assessment_version": assessment.get("version", stage),
        "headline": medical_name,
        "summary": plain_summary,
        "language_levels": {
            "level_1": plain_summary,
            "level_2": why_steps,
            "level_3": {
                "medical_name": medical_name,
                "terms": _professional_terms(stage),
                "notice": "专业信息用于核对原始报告，不建议脱离医生解释自行判断。",
            },
        },
        "reasoning_graph": patient_reasoning_graph(assessment),
        "key_evidence": support,
        "contradictions": contradictions or ["目前还没有足够反证降低其他方向"],
        "missing_information": missing or ["按医生计划进行动态复查"],
        "dangerous_conditions": dangerous,
        "doctor_confirmation": {
            "confirmed": confirmed,
            "label": "医生已确认" if confirmed else "尚未经过医生确认",
        },
        "next_action": "如仍有咯血、静息气促、意识改变或指氧继续下降，立即急诊；否则按当日专科评估计划就医。",
        "boundary": "这是 AI 对已确认记录的通俗整理，不是医生确诊，也不替代急诊判断。",
        "created_at": assessment.get("created_at") or utc_now(),
    }


def _professional_terms(stage: int) -> List[Dict[str, str]]:
    terms = [
        {"term": "静息指氧", "value": "94%（空气）", "meaning": "低于报告参考范围", "source": "生命体征第5项"},
    ]
    if stage >= 2:
        terms.extend([
            {"term": "胸部 CT", "value": "双肺弥漫性磨玻璃影", "meaning": "需结合临床评估肺泡出血", "source": "胸部CT所见第2段"},
            {"term": "血肌酐 / eGFR", "value": "220 μmol/L / 29 ml/min/1.73m²", "meaning": "当前肾功能明显下降", "source": "生化检验第14–15项"},
            {"term": "尿沉渣", "value": "畸形红细胞可见、尿蛋白 2+", "meaning": "支持肾小球来源损伤", "source": "尿沉渣镜检结论"},
        ])
    if stage >= 3:
        terms.extend([
            {"term": "MPO-ANCA", "value": "86 RU/ml，阳性", "meaning": "结合肺肾表现支持 ANCA 相关小血管炎", "source": "免疫学第3项"},
            {"term": "抗 GBM / 感染筛查", "value": "阴性", "meaning": "降低部分替代方向，但不能单独完全排除", "source": "免疫学第5项及微生物报告"},
        ])
    if stage >= 4:
        terms.append({"term": "肾活检", "value": "少免疫沉积性坏死性新月体性肾小球肾炎", "meaning": "提供医生确诊所需的组织学依据", "source": "肾活检病理诊断"})
    return terms


def patient_reasoning_graph(assessment: Dict[str, Any]) -> Dict[str, Any]:
    """Project clinical reasoning into a stable, patient-readable graph."""
    stage = int(assessment.get("batch_stage") or assessment.get("version") or 1)
    diagnosis = assessment.get("primary_diagnosis", {}).get("name", "当前仍需补充信息")
    nodes: List[Dict[str, Any]] = [
        {"id": "symptom-respiratory", "type": "symptom", "label": "血丝痰和活动后气短", "plain_text": "这些表现提示肺部需要尽快检查。", "source_ids": ["patient-consultation"]},
        {"id": "exam-oxygen", "type": "exam_result", "label": "血氧偏低", "plain_text": "说明肺部供氧可能受到影响。", "source_ids": ["observation-spo2"]},
        {"id": "hypothesis-infection", "type": "hypothesis", "label": "感染等其他肺部原因", "plain_text": "这是早期需要排查的方向，不能只看一个结果。", "source_ids": []},
    ]
    edges: List[Dict[str, str]] = [
        {"source": "symptom-respiratory", "target": "exam-oxygen", "relation": "supports", "label": "促使检查肺部严重程度"},
        {"source": "symptom-respiratory", "target": "hypothesis-infection", "relation": "supports", "label": "早期仍需考虑"},
    ]
    if stage >= 2:
        nodes.extend([
            {"id": "exam-lung", "type": "exam_result", "label": "肺部影像异常", "plain_text": "影像提示肺内可能存在出血。", "source_ids": ["observation-ct-ggo", "observation-ct-dah"]},
            {"id": "exam-kidney", "type": "exam_result", "label": "尿检和肾功能异常", "plain_text": "提示损伤更可能来自肾脏过滤部位。", "source_ids": ["observation-urine-rbc", "observation-creatinine", "observation-egfr"]},
            {"id": "pattern-lung-kidney", "type": "evidence", "label": "肺部和肾脏同时受影响", "plain_text": "两处异常可能由同一个全身性问题造成。", "source_ids": ["assessment-pattern"]},
        ])
        edges.extend([
            {"source": "exam-lung", "target": "pattern-lung-kidney", "relation": "explains", "label": "构成肺部一侧的证据"},
            {"source": "exam-kidney", "target": "pattern-lung-kidney", "relation": "explains", "label": "构成肾脏一侧的证据"},
        ])
    if stage >= 3:
        nodes.extend([
            {"id": "exam-immune", "type": "exam_result", "label": "MPO-ANCA 阳性", "plain_text": "这项血液检查支持免疫相关小血管炎。", "source_ids": ["observation-mpo-anca"]},
            {"id": "exam-infection-negative", "type": "exam_result", "label": "部分感染检查阴性", "plain_text": "让普通细菌感染作为统一解释的可能性下降。", "source_ids": ["observation-blood-culture", "observation-sputum-pathogen"]},
        ])
        edges.extend([
            {"source": "exam-immune", "target": "diagnosis-primary", "relation": "supports", "label": "支持当前方向"},
            {"source": "exam-infection-negative", "target": "hypothesis-infection", "relation": "contradicts", "label": "降低，但不完全排除"},
        ])
    if stage >= 4:
        nodes.append({"id": "exam-biopsy", "type": "exam_result", "label": "肾活检提供组织证据", "plain_text": "组织检查结果与当前判断一致。", "source_ids": ["observation-renal-pathology"]})
        edges.append({"source": "exam-biopsy", "target": "diagnosis-primary", "relation": "supports", "label": "进一步确认"})
    nodes.append({"id": "diagnosis-primary", "type": "diagnosis", "label": diagnosis, "plain_text": "这是当前最可能的情况，最终以医生结论为准。", "source_ids": [f"assessment-v{assessment.get('version', stage)}"]})
    if stage >= 2:
        edges.append({"source": "pattern-lung-kidney", "target": "diagnosis-primary", "relation": "supports", "label": "共同模式支持当前判断"})
    if stage < 4:
        nodes.append({"id": "evidence-unresolved", "type": "evidence", "label": "仍有信息没有确认", "plain_text": "还需要医生结合检查和病情变化继续核对。", "source_ids": ["assessment-missing-information"]})
        edges.append({"source": "evidence-unresolved", "target": "diagnosis-primary", "relation": "requires_confirmation", "label": "确认后才能提高结论可靠性"})
    return {
        "schema_version": "patient-reasoning-graph.v1",
        "nodes": nodes,
        "edges": edges,
        "legend": {"supports": "支持", "contradicts": "反对或降低", "explains": "帮助解释", "requires_confirmation": "仍需确认"},
    }


def build_patient_explanations(journey: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [patient_explanation(item, journey.get("doctor_plan")) for item in journey.get("assessment_versions", [])]


def sample_exam_recommendations() -> List[Dict[str, Any]]:
    common = {"status": "proposed", "decision": None, "guidelines": deepcopy(GUIDELINES)}
    rows = [
        ("blood-safety", "是否存在肺出血相关贫血，以及检查操作是否安全？", ["血常规", "血红蛋白动态", "血小板计数"], "紧急", "首次评估并动态复查", ["核对近期输血和采血时间"], ["静脉采血不适"], "血红蛋白下降会提高活动性肺出血关注；血小板影响活检等操作安全。"),
        ("oxygenation", "是否已有低氧或呼吸衰竭？", ["静息及活动后指氧", "必要时动脉血气"], "紧急", "现在；血气仅在低氧或呼吸窘迫时", ["先复核指氧波形和吸氧状态"], ["动脉血气有穿刺疼痛和局部出血风险"], "明显低氧或血气异常会触发急诊/住院评估并改变检查优先级。"),
        ("renal-trend", "肾损伤是否在快速进展？", ["血肌酐动态", "eGFR", "尿素氮", "钾、钠、碳酸氢根"], "紧急", "当日并按病情复查", ["核对既往肾功能基线与补液状态"], ["静脉采血不适"], "肌酐快速上升或高钾、酸中毒会提高处置紧迫性。"),
        ("glomerular-injury", "血尿是否来自肾小球，肾损害程度多大？", ["尿沉渣镜检", "红细胞形态", "红细胞管型", "尿蛋白/肌酐比或24小时尿蛋白"], "高", "尽快完成", ["尽量留取新鲜中段尿并记录月经/导尿等干扰"], ["留尿本身通常无创"], "畸形红细胞、管型和蛋白定量会强化或削弱肾小球炎证据。"),
        ("pulmonary-renal-cause", "肺肾综合征更符合哪一类免疫病因？", ["MPO-ANCA", "PR3-ANCA", "抗GBM抗体", "补体C3/C4", "必要时 ANA/相关自身抗体"], "高", "专科评估时", ["结合临床表现解释，不能只凭抗体确诊"], ["假阳性、假阴性或偶然阳性可能造成误读"], "抗体组合与补体模式会改变 AAV、抗GBM病及其他免疫方向的排序。"),
        ("infection-screen", "是否存在会模拟病情或影响免疫抑制安全的感染？", ["血培养/痰培养（有指征时）", "呼吸道病原学", "乙肝、结核等免疫抑制前筛查"], "高", "治疗决策前；危重时并行进行", ["按症状和流行病学选择项目，避免无差别筛查"], ["阴性结果不能排除全部感染"], "阳性结果可能改变治疗时机和抗感染路径；阴性只降低感染方向。"),
        ("pe-risk", "呼吸困难是否仍需单独排查肺栓塞？", ["临床肺栓塞风险评估", "条件性 D-二聚体", "条件性肺动脉影像"], "条件性", "仅在风险评估支持时", ["先评估肾功能、造影剂风险和临床预测规则"], ["造影剂、辐射及假阳性带来的后续检查风险"], "低风险可避免不必要影像；中高风险结果会改变紧急处置。"),
        ("renal-biopsy", "是否需要组织学证据确认病型和活动度？", ["肾活检必要性评估", "凝血与血小板", "血压与穿刺可行性", "病理光镜/免疫荧光/电镜"], "高", "由肾内科评估；不得因等待活检延误紧急处置", ["排查出血风险、严重高血压和其他禁忌"], ["出血、疼痛、极少数需要介入止血"], "组织学可确认少免疫性肾小球肾炎并影响预后判断；不适合时需记录替代依据。"),
    ]
    return [
        {
            "id": key,
            "clinical_question": question,
            "items": items,
            "priority": priority,
            "timing": timing,
            "prerequisites": prerequisites,
            "risks": risks,
            "expected_impact": impact,
            **deepcopy(common),
        }
        for key, question, items, priority, timing, prerequisites, risks, impact in rows
    ]


def sample_treatment_recommendations() -> List[Dict[str, Any]]:
    reference = treatment_reference()
    return [{
        "id": "aav-guideline-path",
        "title": "器官威胁型 ANCA 相关血管炎治疗路径参考",
        "authority": "decision_support",
        "status": "proposed",
        "decision": None,
        "goals": [reference.get("goal")] if reference.get("goal") else [],
        "pathways": deepcopy(reference.get("pathways", [])),
        "prerequisites": deepcopy(reference.get("prerequisites", [])),
        "risks": deepcopy(reference.get("major_risks", [])),
        "monitoring": deepcopy(reference.get("monitoring", [])),
        "dose_options": [
            {
                "id": "rituximab-1g-two-dose",
                "medication": "利妥昔单抗",
                "dose": "1 g",
                "route": "静脉输注",
                "frequency": "第 1 天与第 15 天",
                "duration": "两次院内输注",
                "purpose": "器官威胁型 AAV 诱导缓解路径之一",
                "dose_source": "KDIGO 2024 Figure 10",
                "origin": "ai_guideline",
                "requires": ["医生确认诊断", "感染与乙肝筛查", "血常规与免疫球蛋白基线", "输注条件"],
            },
            {
                "id": "prednisolone-reduced-50-75kg",
                "medication": "泼尼松",
                "dose": "60 mg（第 1 周起始）",
                "route": "口服",
                "frequency": "第 1 周 60 mg/日；第 2 周 30 mg/日；随后按 PEXIVAS 减量表",
                "duration": "按医生签署的分阶段计划",
                "purpose": "联合诱导治疗并减少累计糖皮质激素暴露",
                "dose_source": "KDIGO 2024 Figure 9 · PEXIVAS reduced-dose regimen",
                "origin": "ai_guideline",
                "requires": ["确认体重为 50–75 kg", "评估感染、血糖、血压、骨骼与精神风险", "医生逐阶段核对减量"],
                "schedule": [
                    {"period": "第 1 周", "dose": "60 mg 每日一次"},
                    {"period": "第 2 周", "dose": "30 mg 每日一次"},
                    {"period": "第 3–4 周", "dose": "25 mg 每日一次"},
                    {"period": "第 5–6 周", "dose": "20 mg 每日一次"},
                    {"period": "第 7–8 周", "dose": "15 mg 每日一次"},
                    {"period": "第 9–10 周", "dose": "12.5 mg 每日一次"},
                    {"period": "第 11–12 周", "dose": "10 mg 每日一次"},
                    {"period": "第 13–14 周", "dose": "7.5 mg 每日一次"},
                    {"period": "第 15–52 周", "dose": "5 mg 每日一次；后续由医生决定"},
                ],
            },
        ],
        "non_dosed_support": [
            "肺孢子菌肺炎预防：KDIGO 建议低剂量 TMP-SMX 或替代方案，但本系统不从该指南自动推断具体剂量。",
            "骨保护及其他支持治疗：具体药物与剂量需由医生依据院内规则补充。",
        ],
        "guidelines": deepcopy(GUIDELINES),
        "boundary": "确认路径只生成医生端处方草稿；第二次签署后才创建患者处方与提醒。",
    }]


def hydrate_journey_v3(journey: Dict[str, Any]) -> Dict[str, Any]:
    upgraded = deepcopy(journey)
    upgraded["schema_version"] = "care-journey.v4"
    synced = upgraded.get("synced_batches") or []
    if "exam_reports" not in upgraded:
        upgraded["exam_reports"] = reports_for_batches(synced)
        covered = {record_id for report in upgraded["exam_reports"] for record_id in report.get("record_ids", [])}
        upgraded["exam_reports"].extend(legacy_reports_from_records(upgraded.get("records") or [], covered))
    upgraded.setdefault("patient_explanations", build_patient_explanations(upgraded))
    messages = upgraded.setdefault("consultation", {}).setdefault("messages", [])
    existing_updates = {item.get("assessment_version_id") for item in messages if item.get("kind") == "assessment_update"}
    for explanation in upgraded.get("patient_explanations", []):
        version_id = explanation.get("assessment_version_id")
        if version_id and version_id not in existing_updates:
            messages.append({
                "id": f"assessment-update-{version_id}", "role": "assistant", "kind": "assessment_update",
                "assessment_version_id": version_id, "patient_explanation": deepcopy(explanation),
                "created_at": explanation.get("created_at") or utc_now(),
            })
    if upgraded.get("recommendation_catalog_version") != 2:
        upgraded["exam_recommendations"] = sample_exam_recommendations()
        upgraded["treatment_recommendations"] = sample_treatment_recommendations()
        upgraded["recommendation_catalog_version"] = 2
    upgraded.setdefault("exam_recommendations", sample_exam_recommendations())
    upgraded.setdefault("exam_orders", [])
    upgraded.setdefault("treatment_recommendations", sample_treatment_recommendations())
    upgraded.setdefault("recommendation_decisions", [])
    upgraded.setdefault("consultation_case_documents", [deepcopy(upgraded.get("raw_case_document"))] if upgraded.get("raw_case_document") else [])
    upgraded.setdefault("prescription_drafts", [])
    upgraded.setdefault("signed_prescriptions", [])
    upgraded.setdefault("treatment_provenance", [])
    upgraded.setdefault("care_team_links", [])
    upgraded.setdefault("last_hospital_sync_at", None)
    upgraded.setdefault("hospital_sync_status", "not_started")
    return upgraded


def patient_journey_dto(journey: Dict[str, Any]) -> Dict[str, Any]:
    source = hydrate_journey_v3(journey)
    allowed = {
        "schema_version", "id", "title", "status", "current_stage", "created_at", "updated_at",
        "patient_profile", "clinical_history", "consultation", "hospital_connection",
        "hospital_sync_status", "last_hospital_sync_at", "synced_batches", "exam_reports",
        "records",
        "patient_explanations", "triage", "appointment_plan", "doctor_plan", "followups", "medications",
        "reminders", "timeline", "consents", "confirmed_treatment_direction", "treatment_provenance",
    }
    dto = {key: deepcopy(value) for key, value in source.items() if key in allowed}
    dto["projection"] = "patient"
    dto["schema_version"] = "patient-journey-dto.v1"
    return dto


def clinician_journey_dto(journey: Dict[str, Any], link: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    dto = hydrate_journey_v3(journey)
    dto.pop("owner_id", None)
    dto["projection"] = "clinician"
    dto["schema_version"] = "clinician-journey-dto.v1"
    dto["care_team_link"] = deepcopy(link) if link else None
    return dto


def public_sample_projection(journey: Dict[str, Any], audience: str) -> Dict[str, Any]:
    if audience == "patient":
        return patient_journey_dto(journey)
    return clinician_journey_dto(journey)


__all__ = [
    "GUIDELINES", "build_patient_explanations", "clinician_journey_dto", "hydrate_journey_v3",
    "legacy_reports_from_records", "patient_explanation", "patient_reasoning_graph", "patient_journey_dto", "public_sample_projection", "reports_for_batches",
    "sample_exam_recommendations", "sample_exam_reports", "sample_treatment_recommendations",
]
