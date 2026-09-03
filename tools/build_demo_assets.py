"""Build sanitized, deterministic assets for the public ClinicaLens demo."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.clinical_evidence import ClinicalEvidenceNormalizer
from agent.diagnosis_engine import DiagnosisDecisionEngine
from agent.replay import DiagnosticReplay


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "diagnostic_replay_cases.jsonl"
DATA_DIR = ROOT / "data" / "demo"
WEB_DATA_DIR = ROOT / "web" / "data"

CASE_SPECS: Dict[str, Dict[str, Any]] = {
    "regression-low-magnesium": {
        "key": "electrolyte-signal",
        "number": "01",
        "eyebrow": "单一强证据收敛",
        "title": "抽筋与心悸背后的关键异常",
        "summary": "从常见症状中识别高信息量实验室指标，避免在泛化候选中停留。",
        "focus": "强证据如何快速改变候选排序",
        "tags": ["电解质", "证据评分", "低不确定性"],
        "reasoning": "血镁 0.45 mmol/L 明显低于参考范围，可统一解释手足抽筋、心悸和乏力；相比冠心病、贫血等宽泛候选，低镁血症具有更直接的客观锚点。",
        "navigation": "携带电解质报告尽快线下就医，由医生结合肾功能、其他电解质、用药史和心电图决定检查与治疗。",
        "uncertainty": "诊断证据较充分，但低镁原因及心律风险仍需结合肾功能、用药史和心电图确认。",
        "unresolved": ["低镁的病因与持续时间", "肾功能及其他电解质", "是否伴随心电图异常"],
    },
    "regression-microscopic-polyangiitis": {
        "key": "multi-organ-pattern",
        "number": "02",
        "eyebrow": "多器官证据关联",
        "title": "肺与肾，看似分散的同一条线索",
        "summary": "把呼吸、肾脏和免疫学证据绑定为可验证的疾病模式。",
        "focus": "跨器官证据如何形成统一解释",
        "tags": ["肺肾综合征", "模式识别", "因果解释"],
        "reasoning": "咯血与肺泡出血提示肺部毛细血管损伤，血尿、蛋白尿和肌酐升高提示肾小球损害；MPO-ANCA 阳性把两组证据连接为小血管炎模式。",
        "navigation": "该证据组合提示潜在器官威胁，建议尽快由风湿免疫科与肾内科联合评估；检查医嘱和治疗方案由医生决定。",
        "uncertainty": "模式一致性高，但治疗强度必须结合感染排除、肾脏病理和器官威胁程度决定。",
        "unresolved": ["感染性病因是否充分排除", "肾活检或其他确证依据", "器官威胁程度与治疗禁忌"],
    },
    "Patient_09817": {
        "key": "negative-evidence",
        "number": "03",
        "eyebrow": "反证驱动排除",
        "title": "症状像感染，检查却在说“不”",
        "summary": "用阴性培养和正常尿检约束模型，避免把症状相似当成感染确诊。",
        "focus": "反证如何阻止常见误判",
        "tags": ["阴性证据", "鉴别诊断", "边界意识"],
        "reasoning": "尿急、尿频和排尿烧灼感容易触发感染假设，但尿培养无生长、白细胞酯酶和亚硝酸盐阴性、尿白细胞正常共同削弱感染；尿动力学提示逼尿肌过度活动，更支持尿道综合征。",
        "navigation": "携带尿检、尿培养和尿动力学结果前往泌尿专科，由医生结合症状持续时间和诱因决定后续检查与处理。",
        "uncertainty": "当前证据不支持细菌感染，但仍需结合完整病史和随访排除其他泌尿系统病因。",
        "unresolved": ["症状持续时间与诱因", "既往泌尿系统病史", "随访中尿检是否发生变化"],
    },
}


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _unique(values: Iterable[str], limit: int = 4) -> List[str]:
    result: List[str] = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in result:
            result.append(value)
        if len(result) >= limit:
            break
    return result


def _display_evidence(row: Dict[str, Any], spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for symptom in (row.get("collected_info") or {}).get("symptoms", []) or []:
        result.append(
            {
                "label": str(symptom),
                "value": "患者主诉",
                "source": "问诊",
                "role": "supporting",
                "reason": "进入候选召回与模式匹配。",
            }
        )

    has_negative_target = bool(row.get("negative_diagnoses"))
    for exam_name, exam in (row.get("exam_results") or {}).items():
        status = str((exam or {}).get("status") or "")
        payload = (exam or {}).get("result") or {}
        for field, value in payload.items():
            role = "contradicting" if has_negative_target and status == "normal" else "supporting"
            reason = (
                "阴性结果削弱常见感染假设。"
                if role == "contradicting"
                else "客观检查结果进入证据评分。"
            )
            result.append(
                {
                    "label": str(field),
                    "value": str(value),
                    "source": str(exam_name),
                    "role": role,
                    "reason": reason,
                }
            )

    for item in spec.get("unresolved", []):
        result.append(
            {
                "label": item,
                "value": "尚未确认",
                "source": "信息缺口",
                "role": "unresolved",
                "reason": "影响诊断边界或后续处置。",
            }
        )
    return result


def _evidence_name_map(bundle: Any) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for observation in bundle.observations:
        finding = str(getattr(observation, "finding", "") or "")
        raw = str(
            getattr(observation, "source_text", "")
            or getattr(observation, "raw_text", "")
            or finding
        )
        if finding and raw:
            result[finding] = raw.replace("result.", "")
    return result


def _candidate_projection(decision: Any, evidence_names: Dict[str, str]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    selected = set(decision.final_diagnoses)
    for rank, candidate in enumerate(decision.candidates[:5], start=1):
        supporting = _unique(
            evidence_names.get(item, item.replace("symptom:", ""))
            for item in candidate.matched_evidence
            if not str(item).startswith("field:")
        )
        contradicting = _unique(
            evidence_names.get(item, item)
            for item in candidate.contradicted_evidence
        )
        if candidate.diagnosis in selected:
            status = "selected"
            summary = "证据覆盖充分，且通过提交授权。"
        elif candidate.score <= 0 or contradicting:
            status = "blocked"
            summary = "存在关键反证或缺少必要确证条件。"
        else:
            status = "differential"
            summary = "保留为鉴别方向，但解释覆盖弱于首选结论。"
        candidates.append(
            {
                "rank": rank,
                "diagnosis": candidate.diagnosis,
                "score": round(float(candidate.score or 0), 4),
                "coverage": round(float(candidate.explanatory_coverage or 0), 4),
                "status": status,
                "summary": summary,
                "supporting_evidence": supporting,
                "contradicting_evidence": contradicting,
                "unresolved_gap_count": len(candidate.required_gaps or []),
            }
        )
    return candidates


def _timeline(
    row: Dict[str, Any],
    spec: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    diagnoses: List[str],
) -> List[Dict[str, Any]]:
    symptoms = (row.get("collected_info") or {}).get("symptoms", []) or []
    exams = row.get("exam_results") or {}
    return [
        {
            "step": 1,
            "phase": "信息采集",
            "title": "从主诉建立初始问题空间",
            "summary": "、".join(symptoms),
            "meta": f"识别 {len(symptoms)} 个症状线索",
        },
        {
            "step": 2,
            "phase": "检查解析",
            "title": "把检查结果转换为结构化证据",
            "summary": "、".join(exams.keys()),
            "meta": f"解析 {len(exams)} 组检查",
        },
        {
            "step": 3,
            "phase": "候选仲裁",
            "title": "比较解释力、反证与未决缺口",
            "summary": " → ".join(item["diagnosis"] for item in candidates[:3]),
            "meta": f"展示前 {min(5, len(candidates))} 个候选",
        },
        {
            "step": 4,
            "phase": "结论生成",
            "title": "输出结论，同时保留边界",
            "summary": "、".join(diagnoses),
            "meta": spec["focus"],
        },
    ]


def _build_case(
    row: Dict[str, Any],
    spec: Dict[str, Any],
    engine: DiagnosisDecisionEngine,
    normalizer: ClinicalEvidenceNormalizer,
) -> Dict[str, Any]:
    bundle = normalizer.normalize(
        row.get("collected_info") or {},
        row.get("exam_results") or {},
    )
    decision = engine.decide({}, [], bundle)
    candidates = _candidate_projection(decision, _evidence_name_map(bundle))
    diagnoses = list(decision.final_diagnoses)
    return {
        "schema_version": "sample-assessment.v1",
        "case": {
            "key": spec["key"],
            "number": spec["number"],
            "eyebrow": spec["eyebrow"],
            "title": spec["title"],
            "summary": spec["summary"],
            "focus": spec["focus"],
            "tags": spec["tags"],
        },
        "mode": "offline",
        "status": "completed",
        "timeline": _timeline(row, spec, candidates, diagnoses),
        "evidence": _display_evidence(row, spec),
        "candidates": candidates,
        "conclusion": {
            "leading_direction": diagnoses,
            "status": "requires_doctor_confirmation",
            "care_navigation": spec["navigation"],
            "doctor_plan": None,
            "reasoning": spec["reasoning"],
            "uncertainty": {
                "level": "low" if decision.confidence >= 0.85 else "medium",
                "label": "证据一致性较高" if decision.confidence >= 0.85 else "仍需补充证据",
                "detail": spec["uncertainty"],
            },
            "conversation_rounds": 1,
            "ordered_examinations": list((row.get("exam_results") or {}).keys()),
            "disclaimer": "这是完整虚构病例中的 AI 辅助判断，不是医生确诊；检查、治疗和用药由医生负责。",
        },
        "metrics": {
            "expected": list(row.get("expected") or []),
            "recall_at_5": any(name in [item["diagnosis"] for item in candidates[:5]] for name in row.get("expected", [])),
            "top1_hit": bool(candidates and candidates[0]["diagnosis"] in (row.get("expected") or [])),
            "exact_match": set(diagnoses) == set(row.get("expected") or []),
            "confidence": round(float(decision.confidence or 0), 4),
        },
        "runtime": {
            "engine": "deterministic-evidence-engine",
            "source": "sanitized regression fixture",
            "generated_on": date.today().isoformat(),
            "duration_seconds": 0,
            "timed_out": False,
            "fallback_used": False,
            "snapshot_notice": "确定性完整虚构病例，不是实时模型调用，也不是临床准确率验证。",
        },
    }


def main() -> int:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    ref_dir = str((ROOT / config.get("ref_data_dir", "data/ref_data")).resolve())
    engine = DiagnosisDecisionEngine(config, ref_dir=ref_dir)
    normalizer = ClinicalEvidenceNormalizer(ref_dir=ref_dir)
    rows = _load_rows(FIXTURE_PATH)

    raw_report = DiagnosticReplay(engine, normalizer).evaluate(rows)
    metrics = {
        key: raw_report[key]
        for key in (
            "cases",
            "candidate_recall_at_5",
            "top1_accuracy",
            "exact_match_rate",
            "namespace_legal_rate",
            "negation_false_positive_rate",
            "negation_false_positive_count",
            "targets",
            "maximum_targets",
        )
    }
    metrics.update(
        {
            "generated_on": date.today().isoformat(),
            "dataset": "7 个仓库内确定性回归病例",
            "method": "固定证据输入，不调用 LLM 或远程患者服务",
            "disclaimer": "该结果衡量回归集上的工程行为，不代表真实临床有效性。",
        }
    )

    selected: List[Dict[str, Any]] = []
    by_id = {str(row.get("patient_id")): row for row in rows}
    for source_id, spec in CASE_SPECS.items():
        item = _build_case(by_id[source_id], spec, engine, normalizer)
        selected.append(
            {
                **item["case"],
                "offline_path": f"data/{spec['key']}.json",
                "live_supported": True,
            }
        )
        _write_json(DATA_DIR / f"{spec['key']}.json", item)
        _write_json(WEB_DATA_DIR / f"{spec['key']}.json", item)

    catalog = {
        "product": "ClinicaLens",
        "schema_version": "sample-catalog.v1",
        "default_case": "multi-organ-pattern",
        "cases": selected,
    }
    for base in (DATA_DIR, WEB_DATA_DIR):
        _write_json(base / "cases.json", catalog)
        _write_json(base / "metrics.json", metrics)

    print(json.dumps({"cases": len(selected), "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
