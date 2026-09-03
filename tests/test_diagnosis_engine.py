import unittest

import yaml

from agent.clinical_evidence import ClinicalEvidenceNormalizer, EvidenceBundle, Observation
from agent.candidate_generator import CandidatePool
from agent.diagnosis_engine import DiagnosisDecisionEngine
from agent.replay import DiagnosticReplay


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class DiagnosisDecisionEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.normalizer = ClinicalEvidenceNormalizer("data/ref_data")
        cls.engine = DiagnosisDecisionEngine(cls.config, "data/ref_data")

    def decide(self, info, exams, llm=None):
        evidence = self.normalizer.normalize(info, exams)
        return evidence, self.engine.decide(llm or {}, [], evidence)

    def test_allowed_namespace_is_official_plus_controlled_extensions(self):
        self.assertEqual(len(self.engine.knowledge.official_names), 50)
        self.assertGreaterEqual(len(self.engine.knowledge.extension_names), 48)
        self.assertEqual(
            len(self.engine.knowledge.allowed_names),
            len(self.engine.knowledge.official_names) + len(self.engine.knowledge.extension_names),
        )
        self.assertIn("肺隐球菌病", self.engine.knowledge.extension_names)
        self.assertIn("霰粒肿", self.engine.knowledge.extension_names)
        self.assertIn("支原体肺炎", self.engine.knowledge.extension_names)
        self.assertIn("二度房室传导阻滞", self.engine.knowledge.extension_names)
        self.assertIn("克里格勒-纳贾尔综合征", self.engine.knowledge.extension_names)
        self.assertIn("慢性鼻咽炎", self.engine.knowledge.extension_names)
        self.assertIn("小耳畸形", self.engine.knowledge.extension_names)
        self.assertIn("急性细菌性前列腺炎", self.engine.knowledge.extension_names)
        self.assertIn("先天性心脏病", self.engine.knowledge.extension_names)
        self.assertIn("终末期肾病", self.engine.knowledge.extension_names)
        self.assertIn("卵巢过度刺激综合征", self.engine.knowledge.extension_names)
        self.assertIn("门静脉高压", self.engine.knowledge.extension_names)
        self.assertIn("创伤后骨关节炎", self.engine.knowledge.extension_names)
        self.assertIn("右位心", self.engine.knowledge.extension_names)
        self.assertIn("室间隔缺损（VSD）", self.engine.knowledge.extension_names)
        self.assertIn("压力性尿失禁", self.engine.knowledge.extension_names)
        self.assertIn("急性鼓膜炎", self.engine.knowledge.extension_names)
        self.assertEqual(self.engine.knowledge.knowledge_version, "2026-07-21-graph-v2")
        self.assertIn("acr_vasculitis_2021", self.engine.knowledge.source_registry)
        self.assertEqual(
            self.engine.knowledge.source_registry["esc_valvular_2025"]["role"],
            "excluded_from_software_transformation_without_license",
        )
        required_fields = {
            "diagnosis_type",
            "parent_diagnosis",
            "supporting_evidence",
            "required_groups",
            "contradictions",
            "discriminating_exams",
            "specificity",
            "treatment_protocol",
            "contraindications",
            "causes",
            "caused_by",
            "category",
            "generalization_suppressions",
            "sources",
            "source_version",
        }
        for entry in self.engine.knowledge.entries.values():
            self.assertTrue(required_fields <= set(entry), entry.get("name"))

    def test_low_magnesium_is_top_diagnosis(self):
        _, decision = self.decide(
            {"symptoms": ["手足抽筋", "心悸", "乏力"]},
            {
                "电解质": {
                    "status": "abnormal",
                    "result": {"血镁": "0.45 mmol/L［参考值：0.75-1.02 mmol/L］"},
                }
            },
        )
        self.assertEqual(decision.final_diagnoses[0], "低镁血症")
        self.assertIn("低镁血症", decision.trusted_diagnoses)

    def test_etiology_priority_keeps_low_magnesium_above_arrhythmia(self):
        _, decision = self.decide(
            {"symptoms": ["腹泻", "手足抽筋", "心悸"]},
            {
                "电解质": {
                    "status": "abnormal",
                    "result": {"血镁": "0.48 mmol/L［参考值：0.75-1.02 mmol/L］"},
                },
                "心电图": {
                    "status": "abnormal",
                    "result": {"结论": "QTc延长，频发房性期前收缩"},
                },
            },
            llm={
                "diagnosis_candidates": [
                    {"name": "心律失常", "confidence": 0.9},
                    {"name": "低镁血症", "confidence": 0.82},
                ]
            },
        )
        self.assertEqual(decision.final_diagnoses[0], "低镁血症")
        self.assertNotIn("心律失常", decision.final_diagnoses)
        self.assertIn("心律失常", [item.diagnosis for item in decision.candidates[:5]])

    def test_magnesium_depletion_satisfies_low_magnesium_required_evidence(self):
        evidence, decision = self.decide(
            {"symptoms": ["腹泻", "手足抽筋", "心悸"]},
            {
                "24小时尿电解质检测": {
                    "status": "abnormal",
                    "result": {"24小时尿镁": "1.2 mmol/24h［参考范围：3.0-5.0］"},
                },
                "镁负荷试验": {
                    "status": "abnormal",
                    "result": {"镁负荷保留率": "62%［参考范围：镁储备充足时通常＜20-30%］"},
                },
                "心电图": {
                    "status": "abnormal",
                    "result": {"结论": "QTc延长，频发房性期前收缩"},
                },
            },
            llm={
                "diagnosis_candidates": [
                    {"name": "心律失常", "confidence": 0.9},
                    {"name": "低镁血症", "confidence": 0.82},
                ]
            },
        )
        self.assertIn("magnesium_depletion", evidence.findings("positive"))
        self.assertIn("magnesium_load_retention_high", evidence.findings("positive"))
        low_mag = next(item for item in decision.candidates if item.diagnosis == "低镁血症")
        self.assertTrue(low_mag.required_met)
        self.assertEqual(decision.final_diagnoses[0], "低镁血症")

    def test_reasoning_exclusion_conflict_defers_and_blocks_low_magnesium_final(self):
        evidence, decision = self.decide(
            {"symptoms": ["腹泻", "手足抽筋", "心悸"]},
            {
                "镁负荷试验": {
                    "status": "abnormal",
                    "result": {
                        "镁负荷保留率": "62%［参考范围：镁储备充足时通常＜20-30%］",
                    },
                },
            },
            llm={
                "diagnosis_candidates": [
                    {"name": "低镁血症", "confidence": 0.88},
                ],
                "reasoning": "镁负荷试验排除低镁血症，暂考虑其他代谢性疾病。",
            },
        )
        self.assertIn("magnesium_load_retention_high", evidence.findings("positive"))
        low_mag = next(item for item in decision.candidates if item.diagnosis == "低镁血症")
        self.assertTrue(low_mag.unresolved_evidence_conflict)
        self.assertTrue(decision.evidence_conflicts)
        self.assertEqual(decision.judge_decision["primary_status"], "deferred")
        self.assertNotIn("低镁血症", decision.final_diagnoses)
        self.assertTrue(
            any(
                item.get("diagnosis") == "低镁血症"
                and item.get("eligibility_reason") == "ConflictNeedsAdjudication"
                for item in decision.judge_decision["blocked_diagnoses"]
            )
        )

    def test_unmet_etiology_candidate_uses_gap_state_not_required_gate(self):
        _, decision = self.decide(
            {"symptoms": ["腹泻", "手足抽筋", "心悸"]},
            {},
            llm={
                "diagnosis_candidates": [
                    {"name": "低镁血症", "confidence": 0.9},
                    {"name": "心律失常", "confidence": 0.85},
                ]
            },
        )
        low_mag = next(item for item in decision.candidates if item.diagnosis == "低镁血症")
        self.assertFalse(low_mag.required_met)
        self.assertFalse(low_mag.hard_contradiction)
        self.assertEqual(low_mag.required_gap_state, "actionable_gap")
        self.assertEqual(low_mag.eligibility_status, "Deferred")
        self.assertNotIn("低镁血症", decision.final_diagnoses)
        self.assertIn("低镁血症", decision.judge_decision["evidence_gap_targets"])
        self.assertEqual(decision.required_gap_authorized_diagnoses, [])
        self.assertTrue(
            any(
                item.get("diagnosis") == "低镁血症"
                and item.get("eligibility_status") == "Deferred"
                for item in decision.judge_decision["blocked_diagnoses"]
            )
        )

    def test_core_evidence_tiers_promote_specific_over_generic_pulmonary_candidate(self):
        tb = "\u80ba\u7ed3\u6838"
        bronchopneumonia = "\u652f\u6c14\u7ba1\u80ba\u708e"
        evidence = EvidenceBundle(
            observations=[
                Observation("cough", "\u95ee\u8bca", evidence_level="generic", information_value=0.18),
                Observation("fever", "\u95ee\u8bca", evidence_level="generic", information_value=0.18),
                Observation("hemoptysis", "\u95ee\u8bca", evidence_level="specific", information_value=0.84),
                Observation("tuberculosis_exposure", "\u95ee\u8bca", evidence_level="specific", information_value=0.92),
                Observation("night_sweats", "\u95ee\u8bca", evidence_level="specific", information_value=0.84),
            ]
        )
        pool = CandidatePool()
        pool.add(bronchopneumonia, bronchopneumonia, "test", prior=0.85)
        pool.add(tb, tb, "test", prior=0.70)
        decision = self.engine.rank(pool, evidence)
        self.assertEqual(decision.candidates[0].diagnosis, tb)
        tb_score = next(item for item in decision.candidates if item.diagnosis == tb)
        generic_score = next(
            item for item in decision.candidates if item.diagnosis == bronchopneumonia
        )
        self.assertGreater(tb_score.core_evidence_score, generic_score.core_evidence_score)
        self.assertGreater(
            generic_score.component_scores.get("specific_over_generic_penalty", 0.0),
            0.0,
        )

    def test_required_group_hits_count_as_core_evidence_for_urachal_cyst(self):
        urachal = "\u8110\u5c3f\u7ba1\u56ca\u80bf"
        urethral = "\u5c3f\u9053\u7efc\u5408\u5f81"
        evidence = EvidenceBundle(
            observations=[
                Observation("umbilical_discharge", "\u95ee\u8bca", evidence_level="specific", information_value=0.94),
                Observation("midline_suprapubic_pain", "\u95ee\u8bca", evidence_level="specific", information_value=0.88),
                Observation("urachal_remnant_pattern", "\u5f71\u50cf", evidence_level="diagnostic_pattern", information_value=0.96),
                Observation("dysuria", "\u95ee\u8bca", evidence_level="generic", information_value=0.20),
            ]
        )
        pool = CandidatePool()
        pool.add(urethral, urethral, "test", prior=0.86)
        pool.add(urachal, urachal, "test", prior=0.66)
        decision = self.engine.rank(pool, evidence)
        self.assertEqual(decision.candidates[0].diagnosis, urachal)
        urachal_score = next(item for item in decision.candidates if item.diagnosis == urachal)
        urethral_score = next(item for item in decision.candidates if item.diagnosis == urethral)
        self.assertIn("umbilical_discharge", urachal_score.core_matched_evidence)
        self.assertGreater(urachal_score.diagnostic_evidence_score, 0.0)
        self.assertGreater(
            urethral_score.component_scores.get("specific_over_generic_penalty", 0.0),
            0.0,
        )

    def test_pulmonary_renal_evidence_promotes_microscopic_polyangiitis(self):
        _, decision = self.decide(
            {"symptoms": ["咯血", "呼吸困难", "关节痛"]},
            {
                "尿常规": {
                    "status": "abnormal",
                    "result": {
                        "尿红细胞": "50 个/HPF［参考值：0-3］",
                        "尿蛋白": "尿蛋白阳性",
                    },
                },
                "肾功能": {
                    "status": "abnormal",
                    "result": {"肌酐": "220 umol/L［参考值：44-133］"},
                },
                "抗体检测": {
                    "status": "abnormal",
                    "result": {"MPO-ANCA": "MPO-ANCA阳性"},
                },
                "胸部CT": {
                    "status": "abnormal",
                    "result": {"结论": "弥漫性肺泡出血"},
                },
            },
        )
        self.assertEqual(decision.final_diagnoses[0], "显微镜下多血管炎")

    def test_high_explainability_unmet_vasculitis_is_gap_candidate_not_final(self):
        _, decision = self.decide(
            {"symptoms": ["咳血痰", "尿色变深", "全身酸痛", "脚踝水肿"]},
            {},
            llm={
                "diagnosis_candidates": [
                    {"name": "冠心病", "confidence": 0.82},
                    {"name": "显微镜下多血管炎", "confidence": 0.78},
                ]
            },
        )
        vasculitis = next(item for item in decision.candidates if item.diagnosis == "显微镜下多血管炎")
        cad = next(item for item in decision.candidates if item.diagnosis == "冠心病")
        self.assertFalse(vasculitis.required_met)
        self.assertFalse(vasculitis.hard_contradiction)
        self.assertGreater(vasculitis.coverage_score, cad.coverage_score)
        self.assertLess(vasculitis.residual_score, cad.residual_score)
        self.assertEqual(vasculitis.eligibility_status, "Deferred")
        self.assertNotIn("显微镜下多血管炎", decision.final_diagnoses)
        self.assertEqual(decision.required_gap_authorized_diagnoses, [])
        self.assertIn(
            "显微镜下多血管炎",
            decision.judge_decision["evidence_gap_targets"],
        )
        self.assertTrue(
            any(
                item.get("diagnosis") == "显微镜下多血管炎"
                and item.get("eligibility_status") == "Deferred"
                for item in decision.judge_decision["blocked_diagnoses"]
            )
        )

    def test_vitamin_d_biochemistry_and_bone_findings_promote_rickets(self):
        _, decision = self.decide(
            {"age": 4, "symptoms": ["O型腿", "步态异常", "腿痛"]},
            {
                "电解质": {
                    "status": "abnormal",
                    "result": {
                        "25羟维生素D": "8 ng/mL［参考值：20-50 ng/mL］",
                        "血钙": "1.9 mmol/L［参考值：2.1-2.6 mmol/L］",
                    },
                },
                "肝功能": {
                    "status": "abnormal",
                    "result": {"碱性磷酸酶": "560 U/L［参考值：45-125 U/L］"},
                },
            },
        )
        self.assertEqual(decision.final_diagnoses[0], "维生素D缺乏性佝偻病")

    def test_aspiration_imaging_suppresses_negative_asd(self):
        evidence, decision = self.decide(
            {"symptoms": ["呛咳", "发热", "咳嗽", "呼吸困难", "血氧下降"]},
            {
                "超声心动图": {
                    "status": "normal",
                    "result": {"结论": "未见先天性缺损（如 ASD、VSD）"},
                },
                "胸部CT": {
                    "status": "abnormal",
                    "result": {"结论": "右下肺肺不张并见支气管肺炎样斑片状阴影"},
                },
            },
        )
        self.assertNotIn("房间隔缺损", decision.final_diagnoses)
        self.assertTrue({"肺不张", "支气管肺炎"} & set(decision.final_diagnoses))
        self.assertNotIn("肺炎", decision.final_diagnoses)
        asd = next(item for item in decision.candidates if item.diagnosis == "房间隔缺损")
        self.assertTrue(asd.hard_contradiction)
        self.assertIn("diagnosis:房间隔缺损", evidence.findings("negative"))

    def test_aspiration_pneumonia_blocks_cross_system_prostatitis_primary(self):
        _, decision = self.decide(
            {"symptoms": ["呛咳", "发热", "咳嗽", "喘息", "呼吸困难"]},
            {
                "胸部CT扫描（Chest CT）": {
                    "status": "abnormal",
                    "result": {"结论": "右下叶实变伴容积减小，符合肺不张，支气管肺炎表现"},
                },
                "支气管镜检查": {
                    "status": "abnormal",
                    "result": {"结论": "右下叶支气管黏液栓阻塞"},
                },
                "痰培养": {
                    "status": "abnormal",
                    "result": {"结论": "肺炎链球菌高载量"},
                },
            },
            llm={
                "diagnosis_candidates": [
                    "急性细菌性前列腺炎",
                    "肺隐球菌病",
                    "支原体肺炎",
                    "肺不张",
                    "支气管肺炎",
                ]
            },
        )
        self.assertNotEqual(decision.final_diagnoses[0], "急性细菌性前列腺炎")
        self.assertNotIn("急性细菌性前列腺炎", decision.final_diagnoses)
        self.assertNotIn("肺隐球菌病", decision.final_diagnoses)
        self.assertNotIn("支原体肺炎", decision.final_diagnoses)
        self.assertTrue({"肺不张", "支气管肺炎"} <= set(decision.final_diagnoses))

    def test_structural_valve_disease_precedes_heart_failure(self):
        _, decision = self.decide(
            {"symptoms": ["活动后气短", "不能平卧", "双下肢水肿"]},
            {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"结论": "重度二尖瓣反流，伴心力衰竭表现和左心室扩大"},
                }
            },
        )
        self.assertEqual(decision.final_diagnoses[0], "二尖瓣反流")
        self.assertIn("心力衰竭", decision.final_diagnoses)

    def test_secondary_state_without_independent_evidence_is_not_final(self):
        _, decision = self.decide(
            {"symptoms": ["活动后气短", "双下肢水肿"]},
            {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"结论": "重度二尖瓣反流，左心房扩大"},
                }
            },
        )
        self.assertEqual(decision.final_diagnoses[0], "二尖瓣反流")
        self.assertNotIn("心力衰竭", decision.final_diagnoses)
        self.assertIn("心力衰竭", [item.diagnosis for item in decision.candidates[:8]])

    def test_mitral_regurgitation_drops_unrelated_high_residual_bone_candidate(self):
        _, decision = self.decide(
            {"age": 72, "symptoms": ["活动后气短", "双下肢水肿", "腰背痛"]},
            {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"结论": "重度二尖瓣反流，心力衰竭表现，左心室扩大"},
                },
                "骨密度": {
                    "status": "abnormal",
                    "result": {"结论": "低骨密度，骨质疏松风险"},
                },
            },
            llm={"diagnosis_candidates": ["骨质疏松症", "二尖瓣反流"]},
        )
        mitral = next(item for item in decision.candidates if item.diagnosis == "二尖瓣反流")
        bone = next(item for item in decision.candidates if item.diagnosis == "骨质疏松症")
        self.assertEqual(decision.final_diagnoses[0], "二尖瓣反流")
        self.assertNotIn("骨质疏松症", decision.final_diagnoses)
        self.assertLess(mitral.residual_score, bone.residual_score)

    def test_valve_gradient_and_regurgitation_promote_structural_causes(self):
        _, decision = self.decide(
            {"symptoms": ["呼吸困难", "心悸", "双下肢水肿"]},
            {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {
                        "结论": "重度三尖瓣反流，右心室扩大",
                        "肺动脉瓣峰值压差": "55 mmHg［参考值：0-20 mmHg］",
                    },
                }
            },
        )
        self.assertIn("三尖瓣反流", decision.final_diagnoses)
        self.assertIn("肺动脉瓣狭窄", decision.final_diagnoses)
        self.assertNotEqual(decision.final_diagnoses[0], "心力衰竭")

    def test_llm_candidate_with_required_gap_is_ranked_not_truncated(self):
        _, decision = self.decide(
            {"symptoms": ["活动后气短", "心悸", "口唇发绀"]},
            {
                "心电图（ECG）": {
                    "status": "abnormal",
                    "result": {"结论": "右心室肥厚"},
                }
            },
            llm={
                "diagnosis_candidates": [
                    {"name": "肺动脉瓣狭窄", "confidence": 0.92},
                    {"name": "肺不张", "confidence": 0.70},
                ]
            },
        )
        pulmonary = next(item for item in decision.candidates if item.diagnosis == "肺动脉瓣狭窄")
        self.assertFalse(pulmonary.required_met)
        self.assertTrue(pulmonary.required_gaps)
        self.assertGreater(pulmonary.score, self.engine.differential_threshold)
        self.assertGreater(pulmonary.score, next(item for item in decision.candidates if item.diagnosis == "肺不张").score)

    def test_congenital_heart_evidence_beats_atelectasis_when_age_and_findings_fit(self):
        _, decision = self.decide(
            {"age": 1, "symptoms": ["发绀", "喂养困难", "吃奶出汗", "呼吸急促"]},
            {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"结论": "大型室间隔缺损，右向左分流，肺动脉高压"},
                },
                "胸部CT扫描（Chest CT）": {
                    "status": "normal",
                    "result": {"结论": "未见肺不张或肺炎实变"},
                },
            },
            llm={"diagnosis": ["大型室间隔缺损伴艾森门格综合征早期表现"]},
        )
        self.assertEqual(decision.final_diagnoses[0], "室间隔缺损（VSD）")
        self.assertNotIn("先天性心脏病", decision.final_diagnoses)
        self.assertNotIn("肺不张", decision.final_diagnoses)

    def test_renal_failure_evidence_promotes_esrd_and_suppresses_bone_diagnosis(self):
        _, decision = self.decide(
            {"age": 70, "symptoms": ["少尿", "眼睑水肿", "皮肤瘙痒", "不能平卧"]},
            {
                "肾功能": {
                    "status": "abnormal",
                    "result": {
                        "肌酐": "720 umol/L［参考值：44-133］",
                        "eGFR": "6 ml/min/1.73m2［参考值：>90］",
                        "尿素氮": "32 mmol/L［参考值：3.2-7.1］",
                    },
                },
                "电解质": {
                    "status": "abnormal",
                    "result": {"血钾": "6.1 mmol/L［参考值：3.5-5.3］"},
                },
            },
            llm={"diagnosis_candidates": ["骨质疏松症", "终末期肾病"]},
        )
        self.assertEqual(decision.final_diagnoses[0], "终末期肾病")
        self.assertNotIn("骨质疏松症", decision.final_diagnoses)

    def test_ohss_evidence_promotes_controlled_extension(self):
        _, decision = self.decide(
            {"symptoms": ["腹胀", "下腹部不适", "呼吸困难"], "history": "促排卵后取卵"},
            {
                "盆腔超声": {
                    "status": "abnormal",
                    "result": {"结论": "双侧卵巢增大，多囊样改变，伴中等量腹水"},
                },
                "血常规": {
                    "status": "abnormal",
                    "result": {"红细胞压积": "49%［参考值：35-45%］"},
                },
                "肝功能检查（LFTs）": {
                    "status": "abnormal",
                    "result": {"白蛋白": "28 g/L［参考值：35-50］"},
                },
            },
        )
        self.assertEqual(decision.final_diagnoses[0], "卵巢过度刺激综合征")
        self.assertNotIn("上呼吸道感染", decision.final_diagnoses)

    def test_portal_hypertension_evidence_promotes_controlled_extension(self):
        _, decision = self.decide(
            {"symptoms": ["右上腹不适", "尿色加深", "皮肤瘙痒", "食欲减退"]},
            {
                "腹部超声多普勒": {
                    "status": "abnormal",
                    "result": {
                        "门静脉内径": "15 mm［参考值：<13 mm］",
                        "结论": "门静脉血流速度降低，脾大，少量腹水",
                    },
                },
                "全血细胞计数（CBC）": {
                    "status": "abnormal",
                    "result": {"血小板": "78 x10^9/L［参考值：100-300］"},
                },
            },
        )
        self.assertEqual(decision.final_diagnoses[0], "门静脉高压")
        self.assertNotIn("胆囊炎", decision.final_diagnoses)

    def test_negative_urine_markers_and_detrusor_activity_promote_urethral_syndrome(self):
        _, decision = self.decide(
            {"symptoms": ["尿急", "尿频", "排尿烧灼感"]},
            {
                "尿培养": {
                    "status": "normal",
                    "result": {"培养结果": "尿培养无生长"},
                },
                "尿常规": {
                    "status": "normal",
                    "result": {
                        "白细胞酯酶": "阴性",
                        "亚硝酸盐": "阴性",
                        "尿白细胞": "0-5 个/HPF，正常",
                    },
                },
                "尿动力学检查": {
                    "status": "abnormal",
                    "result": {"结论": "尿动力学提示逼尿肌过度活动，残余尿正常"},
                },
            },
        )
        self.assertEqual(decision.final_diagnoses[0], "尿道综合征")
        uti = next(item for item in decision.candidates if item.diagnosis == "泌尿系感染")
        self.assertGreater(uti.contradiction_penalty, 0)

    def test_empty_evidence_never_uses_catalog_first_item_as_fallback(self):
        _, decision = self.decide({}, {})
        self.assertEqual(decision.final_diagnoses, [])

    def test_diagnostic_replay_reports_metrics(self):
        evidence = self.normalizer.normalize(
            {},
            {
                "电解质": {
                    "status": "abnormal",
                    "result": {"血镁": "0.45 mmol/L［参考值：0.75-1.02 mmol/L］"},
                }
            },
        )
        report = DiagnosticReplay(self.engine, self.normalizer).evaluate(
            [{"patient_id": "case-low-mg", "expected": ["低镁血症"], "evidence": evidence.to_dict()}]
        )
        self.assertEqual(report["cases"], 1)
        self.assertEqual(report["candidate_recall_at_5"], 1.0)
        self.assertEqual(report["top1_accuracy"], 1.0)
        self.assertEqual(report["namespace_legal_rate"], 1.0)
        self.assertEqual(report["negation_false_positive_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
