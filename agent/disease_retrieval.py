"""Category-first disease retrieval for evidence-first diagnosis ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .clinical_evidence import EvidenceBundle, Observation


@dataclass
class DiseaseCategoryAssessment:
    category: str
    confidence: float
    evidence_links: List[str] = field(default_factory=list)
    matched_terms: List[str] = field(default_factory=list)
    body_system: str = ""
    family: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiseaseRetrievalHit:
    diagnosis: str
    score: float
    category: str = ""
    evidence_links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_CATEGORY_RULES: Dict[str, Dict[str, Any]] = {
    "cardiovascular_conduction": {
        "target_diseases": ["二度房室传导阻滞"],
        "findings": {
            "second_degree_av_block": 0.50,
            "av_block": 0.42,
            "bradycardia": 0.28,
            "presyncope": 0.24,
            "syncope": 0.24,
            "dizziness": 0.08,
            "pr_prolongation": 0.30,
            "dropped_beats": 0.30,
        },
        "terms": {
            "二度房室传导阻滞": 0.55,
            "房室传导阻滞": 0.40,
            "传导阻滞": 0.30,
            "心动过缓": 0.26,
            "心率40": 0.24,
            "心率 40": 0.24,
            "近晕厥": 0.22,
            "晕厥前兆": 0.22,
            "mobitz": 0.40,
            "wenckebach": 0.35,
        },
        "threshold": 0.30,
    },
    "bilirubin_genetic": {
        "target_diseases": ["克里格勒-纳贾尔综合征"],
        "findings": {
            "jaundice": 0.28,
            "scleral_icterus": 0.24,
            "neonatal_jaundice": 0.34,
            "bilirubin_high": 0.30,
            "unconjugated_hyperbilirubinemia": 0.46,
            "genetic_suspicion": 0.20,
            "poor_feeding": 0.10,
            "lethargy": 0.10,
        },
        "terms": {
            "黄疸": 0.24,
            "胆红素": 0.24,
            "间接胆红素": 0.42,
            "非结合胆红素": 0.42,
            "核黄疸": 0.34,
            "克里格勒": 0.55,
            "纳贾尔": 0.55,
            "crigler": 0.55,
            "najjar": 0.55,
            "遗传": 0.14,
            "基因": 0.14,
        },
        "threshold": 0.30,
    },
    "ent_chronic": {
        "target_diseases": ["慢性鼻咽炎"],
        "findings": {
            "nasopharyngeal_foreign_body_sensation": 0.36,
            "throat_dryness": 0.24,
            "throat_clearing": 0.20,
            "chronic_course": 0.24,
            "nasopharyngoscopy_abnormal": 0.42,
            "cough": 0.06,
        },
        "terms": {
            "慢性鼻咽炎": 0.55,
            "鼻咽": 0.28,
            "咽部异物感": 0.36,
            "鼻后滴漏": 0.24,
            "清嗓": 0.20,
            "咽干": 0.20,
            "反复": 0.08,
            "慢性": 0.16,
        },
        "threshold": 0.28,
    },
    "congenital_ear": {
        "target_diseases": ["小耳畸形"],
        "findings": {
            "microtia": 0.54,
            "auricular_malformation": 0.42,
            "external_auditory_canal_atresia": 0.42,
            "congenital_onset": 0.26,
            "hearing_loss": 0.20,
            "abr_abnormal": 0.30,
            "temporal_bone_ct_abnormal": 0.30,
        },
        "terms": {
            "小耳畸形": 0.60,
            "小耳": 0.50,
            "耳廓畸形": 0.42,
            "耳郭畸形": 0.42,
            "外耳道闭锁": 0.42,
            "出生": 0.24,
            "先天": 0.28,
            "听力": 0.16,
        },
        "threshold": 0.30,
    },
    "acute_bacterial_prostate": {
        "target_diseases": ["急性细菌性前列腺炎"],
        "findings": {
            "dysuria": 0.18,
            "urinary_frequency": 0.12,
            "urinary_urgency": 0.12,
            "fever": 0.18,
            "perineal_pain": 0.28,
            "pelvic_pain": 0.20,
            "prostate_tenderness": 0.44,
            "pyuria": 0.24,
            "bacteriuria": 0.24,
            "urine_culture_positive": 0.24,
        },
        "terms": {
            "急性细菌性前列腺炎": 0.60,
            "前列腺炎": 0.36,
            "前列腺压痛": 0.44,
            "会阴痛": 0.28,
            "寒战": 0.18,
            "发热": 0.16,
            "尿痛": 0.16,
            "尿频": 0.10,
            "尿急": 0.10,
        },
        "threshold": 0.30,
    },
    "infection": {
        "target_diseases": [],
        "findings": {"fever": 0.16, "cough": 0.10, "urine_culture_positive": 0.18},
        "terms": {"感染": 0.12, "脓": 0.12, "培养阳性": 0.16},
        "threshold": 0.34,
    },
    "trauma": {
        "target_diseases": [],
        "findings": {"rib_fracture": 0.36},
        "terms": {"外伤": 0.24, "摔伤": 0.24, "骨折": 0.24},
        "threshold": 0.34,
    },
    "post_traumatic_osteoarthritis": {
        "target_diseases": ["创伤后骨关节炎"],
        "findings": {
            "trauma_history": 0.24,
            "post_traumatic_joint_pain": 0.40,
            "activity_related_joint_pain": 0.24,
            "joint_stiffness": 0.18,
            "joint_space_narrowing": 0.42,
            "osteophyte": 0.38,
            "arthralgia": 0.12,
        },
        "terms": {
            "创伤后骨关节炎": 0.60,
            "外伤后骨关节炎": 0.55,
            "关节间隙": 0.30,
            "骨赘": 0.34,
            "软骨下硬化": 0.30,
            "活动后": 0.14,
            "负重": 0.14,
            "膝关节僵硬": 0.20,
        },
        "threshold": 0.30,
    },
    "congenital_structural_heart": {
        "target_diseases": ["右位心", "室间隔缺损（VSD）"],
        "findings": {
            "dextrocardia": 0.52,
            "right_apex_beat": 0.42,
            "mirror_image_ecg": 0.36,
            "ventricular_septal_defect": 0.52,
            "left_to_right_shunt": 0.42,
            "congenital_heart_defect": 0.26,
            "congenital_onset": 0.18,
            "cardiac_murmur": 0.18,
            "feeding_diaphoresis": 0.12,
            "dyspnea": 0.08,
        },
        "terms": {
            "右位心": 0.55,
            "心脏右位": 0.55,
            "室间隔缺损": 0.55,
            "膜周部室间隔缺损": 0.55,
            "VSD": 0.55,
            "左向右分流": 0.42,
            "肺血增多": 0.24,
            "生长迟缓": 0.16,
            "喂养困难": 0.12,
            "多汗": 0.10,
        },
        "threshold": 0.30,
    },
    "urinary_incontinence": {
        "target_diseases": ["压力性尿失禁"],
        "findings": {
            "stress_urinary_incontinence": 0.52,
            "urine_leak_with_pressure": 0.52,
            "urinary_incontinence": 0.28,
            "urine_culture_no_growth": 0.08,
            "leukocyte_esterase_negative": 0.06,
            "nitrite_negative": 0.06,
            "normal_postvoid_residual": 0.12,
        },
        "terms": {
            "压力性尿失禁": 0.60,
            "压力性漏尿": 0.56,
            "腹压性尿失禁": 0.56,
            "咳嗽漏尿": 0.50,
            "运动漏尿": 0.48,
            "大笑漏尿": 0.46,
            "腹压增加": 0.28,
        },
        "threshold": 0.28,
    },
    "acute_otologic_inflammation": {
        "target_diseases": ["急性鼓膜炎"],
        "findings": {
            "acute_tympanitis": 0.54,
            "tympanic_membrane_inflammation": 0.50,
            "tympanic_bulla": 0.50,
            "ear_pain": 0.28,
            "ear_fullness": 0.16,
            "tinnitus": 0.14,
            "hearing_loss": 0.12,
        },
        "terms": {
            "急性鼓膜炎": 0.60,
            "鼓膜炎": 0.52,
            "大疱性鼓膜炎": 0.56,
            "鼓膜充血": 0.48,
            "鼓膜疱": 0.48,
            "耳痛": 0.26,
            "耳鸣": 0.14,
        },
        "threshold": 0.28,
    },
    "valvular_right_heart": {
        "target_diseases": ["三尖瓣反流"],
        "findings": {
            "tricuspid_regurgitation": 0.54,
            "right_heart_enlargement": 0.30,
            "leg_edema": 0.16,
            "dyspnea": 0.12,
            "abdominal_distension": 0.08,
            "ascites": 0.08,
            "pulmonary_hypertension": 0.18,
        },
        "terms": {
            "三尖瓣反流": 0.58,
            "三尖瓣返流": 0.58,
            "右心扩大": 0.28,
            "右心房增大": 0.28,
            "右心室扩大": 0.28,
        },
        "threshold": 0.30,
    },
    "vesicular_viral_exanthem": {
        "target_diseases": ["水痘"],
        "findings": {
            "vesicular_rash": 0.54,
            "pruritus": 0.18,
            "childcare_exposure": 0.34,
            "fever": 0.14,
            "maculopapular_rash": 0.18,
        },
        "terms": {
            "水痘": 0.60,
            "小泡泡": 0.34,
            "水疱": 0.38,
            "幼儿园": 0.22,
            "同班小朋友": 0.26,
            "接触史": 0.16,
        },
        "threshold": 0.30,
    },
    "tuberculous_pericardial_disease": {
        "target_diseases": ["结核性心包炎"],
        "findings": {
            "pericarditic_chest_pain": 0.46,
            "pericardial_effusion": 0.46,
            "pericardial_thickening": 0.38,
            "tuberculosis_exposure": 0.28,
            "cough": 0.08,
            "fever": 0.10,
            "dyspnea": 0.10,
        },
        "terms": {
            "结核性心包炎": 0.62,
            "心包炎": 0.42,
            "前倾缓解": 0.38,
            "平卧加重": 0.32,
            "深呼吸加重": 0.26,
            "心包积液": 0.42,
            "结核": 0.20,
        },
        "threshold": 0.30,
    },
    "opportunistic_fungal_pneumonia": {
        "target_diseases": ["肺念珠菌病"],
        "findings": {
            "candida_positive": 0.58,
            "fungal_pneumonia": 0.42,
            "post_icu_state": 0.30,
            "immunocompromised": 0.24,
            "neutropenia": 0.24,
            "cough": 0.08,
            "dyspnea": 0.10,
        },
        "terms": {
            "肺念珠菌病": 0.62,
            "念珠菌": 0.45,
            "假丝酵母菌": 0.45,
            "ICU": 0.22,
            "免疫抑制": 0.20,
            "真菌性肺炎": 0.38,
        },
        "threshold": 0.30,
    },
    "lacrimal_gland_inflammation": {
        "target_diseases": ["泪腺炎"],
        "findings": {
            "lacrimal_gland_swelling": 0.52,
            "lacrimal_gland_pain": 0.36,
            "eyelid_edema": 0.24,
            "tearing": 0.20,
            "photophobia": 0.12,
        },
        "terms": {
            "泪腺炎": 0.60,
            "泪腺": 0.36,
            "外上方肿胀": 0.34,
            "流泪": 0.16,
            "眼睑肿胀": 0.22,
        },
        "threshold": 0.28,
    },
    "esophageal_mucosal_injury": {
        "target_diseases": ["食管溃疡"],
        "findings": {
            "esophageal_ulcer": 0.58,
            "odynophagia": 0.32,
            "heartburn": 0.22,
            "retrosternal_burning": 0.28,
            "dysphagia": 0.18,
            "chronic_course": 0.10,
        },
        "terms": {
            "食管溃疡": 0.62,
            "食道溃疡": 0.62,
            "上消化道内镜": 0.22,
            "胃镜": 0.16,
            "吞咽痛": 0.28,
            "烧心": 0.20,
            "反酸": 0.18,
        },
        "threshold": 0.30,
    },
    "early_pregnancy_bleeding": {
        "target_diseases": ["先兆流产"],
        "findings": {
            "vaginal_bleeding": 0.48,
            "early_pregnancy": 0.34,
            "hcg_positive": 0.28,
            "progesterone_low": 0.22,
            "pelvic_pain": 0.16,
        },
        "terms": {
            "先兆流产": 0.62,
            "怀孕": 0.24,
            "停经": 0.22,
            "阴道出血": 0.42,
            "孕酮": 0.16,
            "β-hCG": 0.18,
        },
        "threshold": 0.30,
    },
    "hyperandrogenic_anovulation": {
        "target_diseases": ["多囊卵巢综合征"],
        "findings": {
            "polycystic_ovaries": 0.50,
            "oligomenorrhea": 0.32,
            "hyperandrogenism": 0.32,
            "early_pregnancy": 0.06,
        },
        "terms": {
            "多囊卵巢综合征": 0.62,
            "PCOS": 0.62,
            "多囊卵巢": 0.42,
            "月经稀发": 0.28,
            "高雄激素": 0.28,
            "闭经": 0.20,
        },
        "threshold": 0.30,
    },
    "treponemal_skin_bone_infection": {
        "target_diseases": ["雅司病"],
        "findings": {
            "treponemal_skin_lesion": 0.56,
            "treponema_positive": 0.50,
            "treponemal_serology_positive": 0.50,
            "crusted_exudative_skin_ulcer": 0.42,
            "rural_child_contact": 0.28,
            "regional_lymphadenopathy": 0.22,
            "maculopapular_rash": 0.12,
            "arthralgia": 0.10,
        },
        "terms": {
            "雅司病": 0.62,
            "雅司": 0.50,
            "莓疮": 0.50,
            "螺旋体": 0.28,
            "树莓样": 0.34,
            "农村": 0.12,
            "共用毛巾": 0.20,
            "结痂流黄水": 0.32,
            "腹股沟淋巴结": 0.18,
        },
        "threshold": 0.30,
    },
    "congenital_eye_malformation": {
        "target_diseases": ["虹膜缺损"],
        "findings": {
            "iris_coloboma": 0.58,
            "photophobia": 0.18,
            "night_vision_decline": 0.18,
        },
        "terms": {
            "虹膜缺损": 0.62,
            "虹膜裂隙": 0.52,
            "钥匙孔样瞳孔": 0.52,
            "夜视力下降": 0.16,
        },
        "threshold": 0.28,
    },
    "sex_chromosome_aneuploidy": {
        "target_diseases": ["X三体综合征（47,XXX）"],
        "findings": {
            "triple_x_karyotype": 0.62,
            "genetic_suspicion": 0.18,
            "tall_stature": 0.18,
            "premature_ovarian_insufficiency": 0.22,
        },
        "terms": {
            "X三体": 0.58,
            "47,XXX": 0.62,
            "Triple X": 0.62,
            "超雌": 0.42,
        },
        "threshold": 0.30,
    },
    "adrenal_insufficiency": {
        "target_diseases": ["肾上腺疾病"],
        "findings": {
            "adrenal_insufficiency": 0.48,
            "orthostatic_hypotension": 0.34,
            "cortisol_low": 0.46,
            "acth_high": 0.34,
            "hyponatremia": 0.22,
            "hyperkalemia": 0.18,
            "weakness": 0.10,
        },
        "terms": {
            "肾上腺功能不全": 0.58,
            "Addison": 0.50,
            "站起时头晕": 0.24,
            "直立性低血压": 0.34,
            "皮质醇": 0.24,
            "ACTH": 0.20,
        },
        "threshold": 0.30,
    },
    "renal_failure": {
        "target_diseases": ["终末期肾病"],
        "findings": {
            "renal_impairment": 0.48,
            "egfr_low": 0.48,
            "uremia": 0.48,
            "urea_elevated": 0.30,
            "oliguria": 0.24,
            "leg_edema": 0.12,
            "pruritus": 0.14,
            "hyperkalemia": 0.20,
            "metabolic_acidosis": 0.20,
        },
        "terms": {
            "终末期肾病": 0.62,
            "尿毒症": 0.56,
            "eGFR": 0.26,
            "肌酐": 0.24,
            "少尿": 0.22,
            "透析": 0.32,
        },
        "threshold": 0.30,
    },
    "upper_respiratory_infection": {
        "target_diseases": ["上呼吸道感染"],
        "findings": {
            "rhinorrhea": 0.32,
            "nasal_congestion": 0.24,
            "cough": 0.14,
            "fever": 0.12,
            "acute_course": 0.14,
            "wheeze": 0.06,
        },
        "terms": {
            "流鼻涕": 0.30,
            "流涕": 0.28,
            "鼻塞": 0.26,
            "受凉后": 0.18,
            "2天": 0.10,
            "低热": 0.10,
            "上呼吸道感染": 0.48,
        },
        "threshold": 0.28,
    },
    "pulmonary_tuberculosis": {
        "target_diseases": ["肺结核"],
        "findings": {
            "tuberculosis_exposure": 0.42,
            "tb_exposure": 0.44,
            "tuberculosis_pattern": 0.52,
            "chronic_cough_pattern": 0.28,
            "cough": 0.16,
            "fever": 0.14,
            "hemoptysis": 0.32,
            "dyspnea": 0.08,
            "chronic_course": 0.18,
            "night_sweats": 0.24,
        },
        "terms": {
            "肺结核": 0.56,
            "结核": 0.28,
            "接触确诊患者": 0.34,
            "低热": 0.12,
            "咯血": 0.30,
            "盗汗": 0.24,
            "10天": 0.12,
            "两周": 0.12,
        },
        "threshold": 0.30,
    },
    "age_related_refractive_error": {
        "target_diseases": ["老视"],
        "findings": {
            "near_vision_difficulty": 0.48,
            "age_related_near_blur": 0.46,
            "refractive_correction_improves_near_vision": 0.52,
            "presbyopia_pattern": 0.58,
            "refractive_error": 0.34,
            "visual_blurring": 0.04,
        },
        "terms": {
            "老视": 0.58,
            "老花": 0.58,
            "看近": 0.34,
            "阅读困难": 0.34,
            "填表": 0.22,
            "近距离": 0.22,
            "presbyopia": 0.58,
        },
        "threshold": 0.28,
    },
    "sex_development_disorder": {
        "target_diseases": ["卵睾性别发育异常（Ovotesticular DSD）"],
        "findings": {
            "ambiguous_genitalia": 0.48,
            "sex_development_disorder": 0.46,
            "ovotesticular_tissue": 0.54,
            "karyotype_mosaic": 0.36,
            "hypospadias": 0.22,
            "cryptorchidism": 0.18,
        },
        "terms": {
            "卵睾": 0.62,
            "性别发育异常": 0.44,
            "外生殖器发育异常": 0.44,
            "DSD": 0.38,
            "46,XX/46,XY": 0.42,
            "尿道下裂": 0.22,
        },
        "threshold": 0.30,
    },
    "temporomandibular_joint_disorder": {
        "target_diseases": ["颞下颌关节脱位（TMJ）"],
        "findings": {
            "jaw_locked_open": 0.52,
            "unable_close_mouth": 0.46,
            "preauricular_pain": 0.30,
            "tmj_dislocation": 0.56,
            "malocclusion": 0.20,
        },
        "terms": {
            "颞下颌关节脱位": 0.62,
            "下颌关节脱位": 0.58,
            "嘴巴合不上": 0.46,
            "不能闭口": 0.44,
            "耳前区疼痛": 0.28,
            "TMJ": 0.38,
        },
        "threshold": 0.28,
    },
    "anogenital_hpv_vaginitis": {
        "target_diseases": ["尖锐湿疣", "滴虫性阴道炎"],
        "findings": {
            "anogenital_warts": 0.52,
            "cauliflower_lesions": 0.46,
            "hpv_related_lesions": 0.34,
            "frothy_vaginal_discharge": 0.34,
            "vaginal_pruritus": 0.18,
            "trichomonas_positive": 0.46,
            "strawberry_cervix": 0.32,
            "vaginal_ph_high": 0.18,
        },
        "terms": {
            "尖锐湿疣": 0.62,
            "生殖器疣": 0.52,
            "菜花样": 0.42,
            "HPV": 0.28,
            "滴虫": 0.52,
            "泡沫样": 0.34,
            "外阴瘙痒": 0.20,
            "草莓样宫颈": 0.36,
        },
        "threshold": 0.30,
    },
    "urachal_remnant": {
        "target_diseases": ["脐尿管囊肿"],
        "findings": {
            "umbilical_discharge": 0.46,
            "midline_suprapubic_pain": 0.34,
            "umbilical_mass": 0.36,
            "midline_suprapubic_cyst": 0.42,
            "urachal_remnant_pattern": 0.58,
            "urachal_cyst_imaging": 0.56,
            "pelvic_pain": 0.10,
        },
        "terms": {
            "脐尿管囊肿": 0.62,
            "脐尿管残余": 0.52,
            "脐部流液": 0.42,
            "脐孔流液": 0.42,
            "下腹正中": 0.28,
            "膀胱顶部": 0.34,
        },
        "threshold": 0.28,
    },
}


_CATEGORY_GRAPH_HINTS: Dict[str, Dict[str, str]] = {
    "cardiovascular_conduction": {
        "body_system": "cardiovascular",
        "family": "cardiovascular_conduction",
    },
    "bilirubin_genetic": {
        "body_system": "hepatobiliary",
        "family": "bilirubin_genetic",
    },
    "ent_chronic": {"body_system": "ent", "family": "ent_chronic"},
    "congenital_ear": {"body_system": "ent", "family": "congenital_ear"},
    "acute_bacterial_prostate": {
        "body_system": "genitourinary",
        "family": "acute_bacterial_prostate",
    },
    "infection": {"body_system": "infectious", "family": "general_infection"},
    "trauma": {"body_system": "musculoskeletal", "family": "acute_trauma"},
    "post_traumatic_osteoarthritis": {
        "body_system": "musculoskeletal",
        "family": "post_traumatic_degenerative_joint",
    },
    "congenital_structural_heart": {
        "body_system": "cardiovascular",
        "family": "congenital_structural_heart",
    },
    "urinary_incontinence": {
        "body_system": "genitourinary",
        "family": "urinary_incontinence",
    },
    "acute_otologic_inflammation": {
        "body_system": "ent",
        "family": "acute_otologic_inflammation",
    },
    "valvular_right_heart": {
        "body_system": "cardiovascular",
        "family": "valvular_right_heart",
    },
    "vesicular_viral_exanthem": {
        "body_system": "dermatology_infectious",
        "family": "vesicular_viral_exanthem",
    },
    "tuberculous_pericardial_disease": {
        "body_system": "cardiovascular",
        "family": "tuberculous_pericardial_disease",
    },
    "opportunistic_fungal_pneumonia": {
        "body_system": "respiratory",
        "family": "opportunistic_fungal_pneumonia",
    },
    "lacrimal_gland_inflammation": {
        "body_system": "ophthalmology",
        "family": "lacrimal_gland_inflammation",
    },
    "esophageal_mucosal_injury": {
        "body_system": "gastrointestinal",
        "family": "esophageal_mucosal_injury",
    },
    "early_pregnancy_bleeding": {
        "body_system": "obstetrics_gynecology",
        "family": "early_pregnancy_bleeding",
    },
    "hyperandrogenic_anovulation": {
        "body_system": "endocrine_gynecology",
        "family": "hyperandrogenic_anovulation",
    },
    "treponemal_skin_bone_infection": {
        "body_system": "dermatology_infectious",
        "family": "treponemal_skin_bone_infection",
    },
    "congenital_eye_malformation": {
        "body_system": "ophthalmology",
        "family": "congenital_eye_malformation",
    },
    "sex_chromosome_aneuploidy": {
        "body_system": "genetic",
        "family": "sex_chromosome_aneuploidy",
    },
    "adrenal_insufficiency": {
        "body_system": "endocrine",
        "family": "adrenal_insufficiency",
    },
    "renal_failure": {"body_system": "renal", "family": "renal_failure"},
    "upper_respiratory_infection": {
        "body_system": "respiratory",
        "family": "upper_respiratory_infection",
    },
    "pulmonary_tuberculosis": {
        "body_system": "respiratory",
        "family": "pulmonary_tuberculosis",
    },
    "age_related_refractive_error": {
        "body_system": "ophthalmology",
        "family": "age_related_refractive_error",
    },
    "sex_development_disorder": {
        "body_system": "endocrine_genetic",
        "family": "sex_development_disorder",
    },
    "temporomandibular_joint_disorder": {
        "body_system": "musculoskeletal",
        "family": "temporomandibular_joint_disorder",
    },
    "anogenital_hpv_vaginitis": {
        "body_system": "dermatology_genitourinary",
        "family": "anogenital_hpv_vaginitis",
    },
    "urachal_remnant": {
        "body_system": "genitourinary",
        "family": "urachal_remnant",
    },
}


_GENERIC_PENALTY_BY_CATEGORY: Dict[str, Tuple[str, ...]] = {
    "cardiovascular_conduction": ("低镁血症", "心律失常"),
    "bilirubin_genetic": ("肺炎", "败血症", "胆道感染"),
    "ent_chronic": ("上呼吸道感染", "支气管炎", "肺炎"),
    "congenital_ear": ("骨折", "外伤"),
    "acute_bacterial_prostate": ("前列腺增生", "泌尿系感染"),
    "post_traumatic_osteoarthritis": ("骨折", "骨关节炎", "骨质疏松症"),
    "congenital_structural_heart": ("慢性阻塞性肺疾病", "肺炎", "支气管肺炎", "支气管炎", "上呼吸道感染"),
    "urinary_incontinence": ("支气管炎", "肺癌", "肺结核", "泌尿系感染", "尿道综合征"),
    "acute_otologic_inflammation": ("上呼吸道感染", "中耳炎", "小耳畸形", "急性细菌性前列腺炎"),
    "valvular_right_heart": ("卵巢过度刺激综合征", "门静脉高压", "慢性阻塞性肺疾病", "心力衰竭"),
    "vesicular_viral_exanthem": ("湿疹", "荨麻疹", "带状疱疹", "上呼吸道感染"),
    "tuberculous_pericardial_disease": ("肺结核", "支气管肺炎", "肺隐球菌病", "肺炎", "心律失常"),
    "opportunistic_fungal_pneumonia": ("支气管肺炎", "支原体肺炎", "肺炎", "肺不张"),
    "lacrimal_gland_inflammation": ("骨折", "青光眼", "白内障", "上呼吸道感染"),
    "esophageal_mucosal_injury": ("骨折", "慢性鼻咽炎", "上呼吸道感染", "胃炎"),
    "early_pregnancy_bleeding": ("卵巢过度刺激综合征", "肠系膜淋巴结炎", "胆囊炎", "胃炎"),
    "hyperandrogenic_anovulation": ("卵巢过度刺激综合征", "胃炎", "胆囊炎"),
    "treponemal_skin_bone_infection": ("湿疹", "荨麻疹", "骨折"),
    "congenital_eye_malformation": ("前列腺增生", "青光眼", "白内障", "晶状体脱位"),
    "sex_chromosome_aneuploidy": ("骨质疏松症", "骨折", "癫痫"),
    "adrenal_insufficiency": ("低镁血症", "终末期肾病", "心力衰竭"),
    "renal_failure": ("三尖瓣反流", "心力衰竭", "骨质疏松症"),
    "upper_respiratory_infection": ("肺不张", "肺隐球菌病", "肺结核", "支气管肺炎"),
    "pulmonary_tuberculosis": ("支气管肺炎", "支原体肺炎", "肺炎", "支气管炎"),
    "age_related_refractive_error": ("晶状体脱位", "虹膜缺损", "青光眼", "白内障"),
    "sex_development_disorder": ("肾结石", "尿道综合征", "泌尿系感染", "多囊卵巢综合征"),
    "temporomandibular_joint_disorder": ("带状疱疹", "骨折", "骨质疏松症", "成骨不全症"),
    "anogenital_hpv_vaginitis": ("湿疹", "水痘", "雅司病", "尿道综合征"),
    "urachal_remnant": ("骨折", "带状疱疹", "肺结核", "急性细菌性前列腺炎"),
}


class DiseaseCategoryClassifier:
    """Rule-based first-pass classifier used before candidate ranking."""

    def classify(self, evidence: Optional[EvidenceBundle]) -> List[DiseaseCategoryAssessment]:
        observations = list((evidence or EvidenceBundle()).observations)
        positives = [
            item for item in observations
            if item.polarity == "positive" and not getattr(item, "shadowed_by", "")
        ]
        findings = set(item.finding for item in positives)
        text = _combined_text(positives)
        age = _first_age(observations)

        assessments: List[DiseaseCategoryAssessment] = []
        for category, rule in _CATEGORY_RULES.items():
            score = 0.0
            links: List[str] = []
            terms_hit: List[str] = []
            for finding, weight in (rule.get("findings") or {}).items():
                finding_hits = [item for item in positives if item.finding == finding]
                if finding_hits:
                    score += float(weight) * max(_information_multiplier(item) for item in finding_hits)
                    links.append(finding)
            lower_text = text.lower()
            for term, weight in (rule.get("terms") or {}).items():
                if str(term).lower() in lower_text:
                    score += float(weight)
                    terms_hit.append(str(term))

            if category in {"bilirubin_genetic", "congenital_ear"} and age is not None and age < 18:
                score += 0.10
                links.append("age:pediatric")
            if category == "congenital_ear" and any(term in lower_text for term in ("出生", "先天")):
                score += 0.12
                links.append("onset:congenital")
            if category == "acute_bacterial_prostate" and "female" in lower_text:
                score -= 0.25

            threshold = float(rule.get("threshold", 0.3) or 0.3)
            if score >= threshold:
                graph_hint = _CATEGORY_GRAPH_HINTS.get(category, {})
                assessments.append(
                    DiseaseCategoryAssessment(
                        category=category,
                        confidence=round(max(0.0, min(1.0, score)), 4),
                        evidence_links=list(dict.fromkeys(links)),
                        matched_terms=list(dict.fromkeys(terms_hit)),
                        body_system=str(rule.get("body_system") or graph_hint.get("body_system") or ""),
                        family=str(rule.get("family") or graph_hint.get("family") or category),
                    )
                )
        assessments.sort(key=lambda item: item.confidence, reverse=True)
        return assessments


class DiseaseRetriever:
    """Retrieve concrete diseases from evidence and category cues before ranking."""

    def __init__(self, knowledge: Any, resolver: Any, classifier: Optional[DiseaseCategoryClassifier] = None):
        self.knowledge = knowledge
        self.resolver = resolver
        self.classifier = classifier or DiseaseCategoryClassifier()

    def retrieve(self, evidence: Optional[EvidenceBundle], top_k: int = 20) -> Tuple[List[DiseaseRetrievalHit], List[DiseaseCategoryAssessment]]:
        bundle = evidence or EvidenceBundle()
        categories = self.classifier.classify(bundle)
        category_by_name = {item.category: item for item in categories}
        scores: Dict[str, DiseaseRetrievalHit] = {}

        for assessment in categories:
            rule = _CATEGORY_RULES.get(assessment.category, {})
            for raw_name in rule.get("target_diseases") or []:
                name = self._resolve(raw_name)
                if not name:
                    continue
                self._add_score(
                    scores,
                    name,
                    0.42 + 0.35 * assessment.confidence,
                    assessment.category,
                    assessment.evidence_links + assessment.matched_terms,
                    {
                        "reason": "category_target",
                        "category_confidence": assessment.confidence,
                        "body_system": assessment.body_system,
                        "family": assessment.family,
                    },
                )

        for name, entry in getattr(self.knowledge, "entries", {}).items():
            if not self._allowed(name):
                continue
            support_score, links = self._entry_support(entry, bundle.observations)
            category_score = 0.0
            entry_category = str(entry.get("category") or "")
            entry_system = str(entry.get("body_system") or "")
            entry_family = str(entry.get("disease_family") or entry.get("family") or "")
            for assessment in categories:
                if entry_category and entry_category == assessment.category:
                    category_score = max(category_score, 0.22 * assessment.confidence)
                if entry_family and assessment.family and entry_family == assessment.family:
                    category_score = max(category_score, 0.25 * assessment.confidence)
                elif entry_system and assessment.body_system and entry_system == assessment.body_system:
                    category_score = max(category_score, 0.08 * assessment.confidence)

            if support_score <= 0 and category_score <= 0:
                continue

            specificity = float(entry.get("specificity", 0.5) or 0.5)
            score = min(1.0, support_score + category_score + 0.05 * max(0.0, specificity - 0.5))
            score = max(0.0, score - self._generic_conflict_penalty(name, categories))
            if self._is_cross_system_conflict(entry, categories):
                score = max(0.0, score - 0.16)
            if score <= 0:
                continue
            self._add_score(
                scores,
                name,
                score,
                entry_category,
                links,
                {
                    "reason": "evidence_profile",
                    "support_score": round(support_score, 4),
                    "body_system": entry_system,
                    "family": entry_family,
                },
            )

        hits = sorted(
            scores.values(),
            key=lambda item: (
                item.score,
                _scope_confidence(item, self.knowledge, categories),
                _specificity(self.knowledge, item.diagnosis),
                item.diagnosis,
            ),
            reverse=True,
        )
        return hits[: max(0, int(top_k or 20))], categories

    def _resolve(self, raw_name: Any) -> Optional[str]:
        resolution = self.resolver.resolve(raw_name)
        name = resolution.canonical_name
        if name and self._allowed(name):
            return name
        return None

    def _allowed(self, name: Any) -> bool:
        try:
            return bool(self.knowledge.is_allowed(name))
        except Exception:
            return bool(name)

    def _entry_support(
        self,
        entry: Dict[str, Any],
        observations: Sequence[Observation],
    ) -> Tuple[float, List[str]]:
        score = 0.0
        links: List[str] = []
        for spec in entry.get("supporting_evidence", []) or []:
            hits = [
                item for item in observations
                if item.polarity == "positive"
                and not getattr(item, "shadowed_by", "")
                and _observation_matches(spec, item)
            ]
            if not hits:
                continue
            weight = float(spec.get("weight", 0.2) or 0.2)
            confidence = max(item.confidence * _information_multiplier(item) for item in hits)
            score += weight * confidence
            links.extend(item.finding for item in hits)
        return min(0.78, score), list(dict.fromkeys(links))

    @staticmethod
    def _generic_conflict_penalty(name: str, categories: Sequence[DiseaseCategoryAssessment]) -> float:
        for assessment in categories:
            if assessment.confidence < 0.45:
                continue
            if name in _GENERIC_PENALTY_BY_CATEGORY.get(assessment.category, ()):
                if assessment.category in {
                    "acute_otologic_inflammation",
                    "post_traumatic_osteoarthritis",
                    "congenital_structural_heart",
                    "urinary_incontinence",
                }:
                    return 0.65
                return 0.28
        return 0.0

    @staticmethod
    def _is_cross_system_conflict(
        entry: Dict[str, Any],
        categories: Sequence[DiseaseCategoryAssessment],
    ) -> bool:
        entry_system = str(entry.get("body_system") or "")
        if not entry_system:
            return False
        for assessment in categories:
            if assessment.confidence < 0.55 or not assessment.body_system:
                continue
            if entry_system != assessment.body_system:
                return True
        return False

    @staticmethod
    def _add_score(
        scores: Dict[str, DiseaseRetrievalHit],
        name: str,
        score: float,
        category: str,
        evidence_links: Iterable[str],
        metadata: Dict[str, Any],
    ) -> None:
        current = scores.get(name)
        score = round(max(0.0, min(1.0, float(score or 0.0))), 4)
        if current is None:
            scores[name] = DiseaseRetrievalHit(
                diagnosis=name,
                score=score,
                category=category,
                evidence_links=list(dict.fromkeys(str(item) for item in evidence_links if str(item))),
                metadata=dict(metadata),
            )
            return
        if score > current.score:
            current.score = score
            current.category = category or current.category
        current.evidence_links = list(
            dict.fromkeys(current.evidence_links + [str(item) for item in evidence_links if str(item)])
        )
        current.metadata.update(metadata)


def _combined_text(observations: Sequence[Observation]) -> str:
    parts: List[str] = []
    for item in observations:
        parts.extend([item.finding, item.source, item.raw_text, item.field_path])
    return " ".join(str(item) for item in parts if str(item))


def _first_age(observations: Sequence[Observation]) -> Optional[float]:
    for item in observations:
        if item.finding == "field:age" and item.value is not None:
            return item.value
    return None


def _information_multiplier(item: Observation) -> float:
    try:
        value = float(getattr(item, "information_value", 0.0) or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0.0:
        return 1.0
    return max(0.35, min(1.45, 0.65 + value))


def _max_category_confidence(category: str, categories: Sequence[DiseaseCategoryAssessment]) -> float:
    if not category:
        return 0.0
    return max((item.confidence for item in categories if item.category == category), default=0.0)


def _scope_confidence(
    hit: DiseaseRetrievalHit,
    knowledge: Any,
    categories: Sequence[DiseaseCategoryAssessment],
) -> float:
    entry = {}
    try:
        entry = knowledge.get(hit.diagnosis)
    except Exception:
        entry = {}
    category = hit.category or str(entry.get("category") or "")
    body_system = str(hit.metadata.get("body_system") or entry.get("body_system") or "")
    family = str(
        hit.metadata.get("family")
        or entry.get("disease_family")
        or entry.get("family")
        or ""
    )
    best = _max_category_confidence(category, categories)
    for assessment in categories:
        if family and assessment.family and family == assessment.family:
            best = max(best, assessment.confidence + 0.05)
        elif body_system and assessment.body_system and body_system == assessment.body_system:
            best = max(best, assessment.confidence * 0.5)
    return best


def _specificity(knowledge: Any, name: str) -> float:
    try:
        return float(knowledge.get(name).get("specificity", 0.5) or 0.5)
    except Exception:
        return 0.5


def _observation_matches(spec: Dict[str, Any], item: Observation) -> bool:
    finding = str(spec.get("finding") or "")
    if finding and finding != item.finding:
        return False
    direction = str(spec.get("direction") or "")
    if direction and direction != item.direction:
        return False
    source_contains = str(spec.get("source_contains") or "")
    if source_contains and source_contains.lower() not in item.source.lower():
        return False
    terms = spec.get("terms") or []
    if isinstance(terms, str):
        terms = [terms]
    if terms and not any(str(term).lower() in item.raw_text.lower() for term in terms):
        return False
    if spec.get("min_value") is not None:
        if item.value is None or item.value < float(spec["min_value"]):
            return False
    if spec.get("max_value") is not None:
        if item.value is None or item.value > float(spec["max_value"]):
            return False
    return bool(finding or direction or source_contains or terms or spec.get("min_value") is not None or spec.get("max_value") is not None)
