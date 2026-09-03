"""Build the local structured diagnosis knowledge artifact from checked-in sources."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "ref_data"
KNOWLEDGE_VERSION = "2026-07-15"


ADVANCED = {
    "低镁血症": {
        "diagnosis_type": "metabolic",
        "specificity": 0.9,
        "supporting_evidence": [
            {"finding": "low_magnesium", "weight": 1.0},
            {"finding": "magnesium_depletion", "weight": 0.95},
            {"finding": "magnesium_load_retention_high", "weight": 0.9},
            {"finding": "low_urine_magnesium", "weight": 0.45},
            {"finding": "muscle_cramp", "weight": 0.3},
            {"finding": "palpitation", "weight": 0.22},
            {"finding": "weakness", "weight": 0.15},
        ],
        "required_groups": [["low_magnesium", "magnesium_depletion", "magnesium_load_retention_high"]],
        "contradictions": [],
        "suppress_diagnoses": ["心律失常"],
        "causes": ["心律失常"],
        "strong_verification_exams": ["综合代谢面板（CMP）", "24小时尿电解质检测", "镁负荷试验", "心电图（ECG）"],
        "discriminating_exams": ["综合代谢面板（CMP）", "24小时尿电解质检测", "镁负荷试验", "心电图（ECG）"],
        "treatment_protocol": [
            "确认血镁并同步评估血钾、血钙、肾功能和心电图；有症状或明显心律失常时在监护下补镁。",
            "查找胃肠道丢失、利尿剂或其他药物等病因，补镁过程中复查电解质并根据肾功能调整。"
        ],
        "sources": ["evaluation_expected", "local_clinical_profile"]
    },
    "维生素D缺乏性佝偻病": {
        "diagnosis_type": "metabolic",
        "specificity": 0.92,
        "supporting_evidence": [
            {"finding": "vitamin_d_low", "weight": 0.8},
            {"finding": "alp_elevated", "weight": 0.45},
            {"finding": "hypocalcemia", "weight": 0.25},
            {"finding": "bone_deformity", "weight": 0.55},
            {"finding": "waddling_gait", "weight": 0.25},
            {"terms": ["腿痛", "跛行", "活动后下肢功能障碍"], "weight": 0.2}
        ],
        "required_groups": [["vitamin_d_low", "bone_deformity"]],
        "contradictions": [],
        "strong_verification_exams": ["维生素D检测", "血清电解质", "甲状旁腺激素检测（PTH）", "肝功能检查（LFTs）", "骨转换标志物（BTMs）", "X线检查"],
        "discriminating_exams": ["维生素D检测", "血清电解质", "甲状旁腺激素检测（PTH）", "肝功能检查（LFTs）", "骨转换标志物（BTMs）", "X线检查"],
        "treatment_protocol": [
            "由儿科结合25羟维生素D、钙磷和碱性磷酸酶结果进行维生素D与钙补充，并纠正营养和日照不足。",
            "定期复查生化指标和骨骼发育，明显畸形或步态异常时联合儿童骨科评估。"
        ],
        "sources": ["evaluation_expected", "local_clinical_profile"]
    },
    "显微镜下多血管炎": {
        "diagnosis_type": "etiology",
        "specificity": 0.95,
        "supporting_evidence": [
            {"finding": "mpo_anca_positive", "weight": 0.8},
            {"finding": "p_anca_positive", "weight": 0.55},
            {"finding": "anca_positive", "weight": 0.45},
            {"finding": "microscopic_hematuria", "weight": 0.35},
            {"finding": "proteinuria", "weight": 0.3},
            {"finding": "pulmonary_hemorrhage", "weight": 0.55},
            {"finding": "hemoptysis", "weight": 0.2},
            {"finding": "dark_urine", "weight": 0.18},
            {"finding": "leg_edema", "weight": 0.18},
            {"finding": "arthralgia", "weight": 0.18},
            {"terms": ["全身酸痛", "肌肉酸痛"], "weight": 0.16},
            {"finding": "renal_impairment", "weight": 0.3}
        ],
        "required_groups": [
            ["mpo_anca_positive", "p_anca_positive", "anca_positive"],
            ["microscopic_hematuria", "proteinuria", "renal_impairment", "pulmonary_hemorrhage"]
        ],
        "contradictions": [
            {"finding": "diagnosis:系统性红斑狼疮", "polarity": "positive", "penalty": 0.2, "hard": False}
        ],
        "suppress_diagnoses": ["肺炎"],
        "strong_verification_exams": ["抗中性粒细胞胞质抗体（ANCA）谱", "MPO-ANCA", "尿液分析（UA）", "肾功能", "胸部CT扫描（Chest CT）", "血沉", "C反应蛋白（CRP）", "凝血功能", "抗核抗体"],
        "discriminating_exams": ["抗中性粒细胞胞质抗体（ANCA）谱", "MPO-ANCA", "尿液分析（UA）", "肾功能", "胸部CT扫描（Chest CT）", "血沉", "C反应蛋白（CRP）", "凝血功能", "抗核抗体"],
        "treatment_protocol": [
            "肺肾综合征或快速进展性肾损伤需住院，由风湿免疫科和肾内科联合评估疾病活动度与器官威胁程度。",
            "在排除活动性感染后按器官威胁程度选择糖皮质激素联合利妥昔单抗或环磷酰胺等诱导方案，并进行感染预防和实验室监测。"
        ],
        "sources": ["evaluation_expected", "https://rheumatology.org/vasculitis-guideline"]
    },
    "二尖瓣反流": {
        "diagnosis_type": "structural",
        "specificity": 0.9,
        "supporting_evidence": [
            {"finding": "mitral_regurgitation", "weight": 1.0},
            {"finding": "heart_failure_state", "weight": 0.2},
            {"finding": "orthopnea", "weight": 0.18},
            {"finding": "paroxysmal_nocturnal_dyspnea", "weight": 0.22},
            {"finding": "leg_edema", "weight": 0.15}
        ],
        "required_groups": [["mitral_regurgitation"]],
        "contradictions": [],
        "discriminating_exams": ["超声心动图", "心电图", "胸部X线"],
        "suppress_diagnoses": [],
        "causes": ["心力衰竭"],
        "treatment_protocol": [
            "根据超声所示反流严重度、左心室大小和功能及症状，由心脏瓣膜团队评估修复或置换时机。",
            "存在容量超负荷时谨慎利尿并监测肾功能、电解质、血压和体重，同时处理房颤、高血压等诱因。"
        ],
        "sources": ["evaluation_expected", "local_clinical_profile"]
    },
    "肺不张": {
        "diagnosis_type": "structural",
        "specificity": 0.86,
        "supporting_evidence": [
            {"finding": "atelectasis", "weight": 1.0},
            {"finding": "choking_event", "weight": 0.3},
            {"finding": "aspiration_risk", "weight": 0.25},
            {"finding": "hypoxemia", "weight": 0.25},
            {"finding": "dyspnea", "weight": 0.15}
        ],
        "required_groups": [["atelectasis"]],
        "contradictions": [],
        "strong_verification_exams": ["胸部X线检查（CXR）", "胸部CT扫描（Chest CT）", "支气管镜检查", "动脉血气（ABG）"],
        "discriminating_exams": ["胸部X线检查（CXR）", "胸部CT扫描（Chest CT）", "支气管镜检查", "动脉血气（ABG）"],
        "treatment_protocol": [
            "根据低氧程度给予氧疗并实施适龄气道廓清、体位引流和吸痰；怀疑黏液栓或异物阻塞时评估支气管镜。",
            "同步处理误吸、感染或术后低通气等病因，并复查胸部影像确认复张。"
        ],
        "sources": ["evaluation_expected", "local_clinical_profile"]
    },
    "支气管肺炎": {
        "diagnosis_type": "infection",
        "specificity": 0.86,
        "supporting_evidence": [
            {"finding": "bronchopneumonia", "weight": 1.0},
            {"finding": "pneumonia_infiltrate", "weight": 0.55},
            {"finding": "fever", "weight": 0.2},
            {"finding": "cough", "weight": 0.18},
            {"finding": "dyspnea", "weight": 0.18},
            {"finding": "hypoxemia", "weight": 0.25},
            {"finding": "choking_event", "weight": 0.2}
        ],
        "required_groups": [["bronchopneumonia", "pneumonia_infiltrate"]],
        "contradictions": [],
        "suppress_diagnoses": ["肺炎"],
        "strong_verification_exams": ["体格检查", "脉搏血氧饱和度监测（SpO2）", "胸部X线检查（CXR）", "全血细胞计数（CBC）", "C反应蛋白（CRP）", "降钙素原（PCT）", "痰培养", "抗菌药物敏感性试验（AST）"],
        "discriminating_exams": ["体格检查", "脉搏血氧饱和度监测（SpO2）", "胸部X线检查（CXR）", "全血细胞计数（CBC）", "C反应蛋白（CRP）", "降钙素原（PCT）", "痰培养", "抗菌药物敏感性试验（AST）"],
        "treatment_protocol": [
            "结合年龄、严重度、低氧和培养结果选择抗感染方案，必要时住院吸氧、补液并监测呼吸状态。",
            "有误吸线索时同步采取吞咽评估和误吸预防；48至72小时无改善时复评病原、并发症和气道阻塞。"
        ],
        "sources": ["evaluation_expected", "local_clinical_profile"]
    }
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def unique_specs(items):
    seen = set()
    result = []
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def main():
    catalog = read_json(REF / "diseases_catalog.json").get("diseases", [])
    extensions = read_json(REF / "submission_diagnosis_extensions.json").get("extensions", [])
    old_rules = {
        item.get("diagnosis"): item
        for item in read_json(REF / "diagnostic_rules.json").get("rules", [])
        if item.get("diagnosis")
    }
    profiles = {}
    for path in sorted(REF.glob("disease_profiles*.json")):
        for profile in read_json(path).get("profiles", []):
            if profile.get("name"):
                profiles[profile["name"]] = profile

    names = [item["name"] for item in catalog] + [item["name"] for item in extensions]
    extension_meta = {item["name"]: item for item in extensions}
    diseases = []
    for name in names:
        profile = profiles.get(name, {})
        support = []
        for symptom in profile.get("common_symptoms", []) or []:
            support.append({"terms": [symptom], "weight": 0.2})
        for flag in profile.get("red_flags", []) or []:
            support.append({"terms": [flag], "weight": 0.24})
        entry = {
            "name": name,
            "diagnosis_type": "disease",
            "parent_diagnosis": extension_meta.get(name, {}).get("parent_catalog_name") or "",
            "supporting_evidence": unique_specs(support),
            "required_groups": [],
            "contradictions": [],
            "strong_verification_exams": profile.get("strong_verification_exams", []) or [],
            "discriminating_exams": (
                list(profile.get("strong_verification_exams", []) or [])
                + list(profile.get("required_exams", []) or [])
            ),
            "specificity": extension_meta.get(name, {}).get("specificity", 0.5),
            "treatment_protocol": profile.get("treatment_principles", []) or [],
            "contraindications": [],
            "suppress_diagnoses": [],
            "causes": [],
            "caused_by": [],
            "sources": extension_meta.get(name, {}).get(
                "sources", ["official_catalog", "local_clinical_profile"]
            ),
            "source_version": KNOWLEDGE_VERSION,
        }
        old = old_rules.get(name)
        if old:
            entry["diagnosis_type"] = old.get("diagnosis_type", entry["diagnosis_type"])
            entry["supporting_evidence"] = unique_specs(entry["supporting_evidence"] + old.get("positive_evidence", []))
            groups = []
            if old.get("required_any"):
                groups.append(list(old["required_any"]))
            for required in old.get("required_all", []) or []:
                groups.append([required])
            entry["required_groups"] = groups
            entry["contradictions"] = [
                {"finding": item.get("finding"), "penalty": item.get("weight", 0.35), "hard": False}
                for item in old.get("negative_evidence", []) or [] if item.get("finding")
            ]
            entry["treatment_protocol"] = old.get("treatment_protocol", []) or entry["treatment_protocol"]
            entry["suppress_diagnoses"] = old.get("suppress_diagnoses", []) or []
            entry["specificity"] = max(float(entry["specificity"] or 0.5), float(old.get("priority", 50)) / 100.0)
            entry["sources"] = unique_specs(entry["sources"] + ["migrated_diagnostic_rules"])
        if name in ADVANCED:
            override = ADVANCED[name]
            for key, value in override.items():
                if key in {
                    "supporting_evidence",
                    "contradictions",
                    "treatment_protocol",
                    "contraindications",
                    "sources",
                    "causes",
                    "caused_by",
                }:
                    entry[key] = unique_specs(list(entry.get(key, [])) + list(value or []))
                else:
                    entry[key] = value
        diseases.append(entry)

    payload = {
        "schema_version": 2,
        "knowledge_version": KNOWLEDGE_VERSION,
        "source_registry": {
            "who_guidelines": {
                "url": "https://www.who.int/publications/who-guidelines",
                "role": "authoritative_source_index",
                "accessed": KNOWLEDGE_VERSION,
            },
            "acr_vasculitis_2021": {
                "url": "https://rheumatology.org/vasculitis-guideline",
                "version": "2021 ACR/VF ANCA-Associated Vasculitis Guideline",
                "role": "reviewed_source_for_vasculitis_protocol",
                "accessed": KNOWLEDGE_VERSION,
            },
            "esc_valvular_2025": {
                "url": "https://www.escardio.org/guidelines/clinical-practice-guidelines/all-esc-practice-guidelines/valvular-heart-disease/",
                "version": "2025 ESC/EACTS Valvular Heart Disease Guideline",
                "role": "excluded_from_software_transformation_without_license",
                "accessed": KNOWLEDGE_VERSION,
            },
        },
        "generated_from": [
            "diseases_catalog.json",
            "disease_profiles*.json",
            "diagnostic_rules.json",
            "submission_diagnosis_extensions.json"
        ],
        "diseases": diseases
    }
    target = REF / "diagnostic_knowledge.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target} diseases={len(diseases)}")


if __name__ == "__main__":
    main()
