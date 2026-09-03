import unittest

import yaml

from agent.clinical_evidence import ClinicalEvidenceNormalizer, EvidenceBundle, Observation
from agent.diagnosis_engine import DiagnosisDecisionEngine


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class DiseaseRetrievalUpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normalizer = ClinicalEvidenceNormalizer("data/ref_data")
        cls.engine = DiagnosisDecisionEngine(load_config(), "data/ref_data")

    def retrieve_names(self, info, exams=None):
        evidence = self.normalizer.normalize(info, exams or {})
        hits, categories = self.engine.candidate_generator.disease_retriever.retrieve(evidence, top_k=20)
        return evidence, [item.diagnosis for item in hits], [item.category for item in categories]

    def decide(self, info, exams=None, llm=None):
        evidence = self.normalizer.normalize(info, exams or {})
        return evidence, self.engine.decide(llm or {}, [], evidence)

    def test_conduction_path_retrieves_second_degree_av_block(self):
        evidence, names, categories = self.retrieve_names(
            {"symptoms": ["近晕厥", "头晕", "乏力"], "physical": "心率 40 次/分，心动过缓"},
            {"心电图": {"status": "abnormal", "result": {"结论": "二度房室传导阻滞，PR间期延长，间歇性漏搏"}}},
        )
        self.assertIn("cardiovascular_conduction", categories)
        self.assertIn("bradycardia", evidence.findings("positive"))
        self.assertIn("second_degree_av_block", evidence.findings("positive"))
        self.assertIn("二度房室传导阻滞", names[:5])

    def test_bilirubin_genetic_path_retrieves_crigler_najjar(self):
        evidence, names, categories = self.retrieve_names(
            {"age": 1, "symptoms": ["黄疸", "巩膜黄染", "嗜睡", "喂养差"]},
            {
                "肝功能检查（LFTs）": {
                    "status": "abnormal",
                    "result": {"间接胆红素": "320 umol/L［参考值：0-17］"},
                },
                "基因检测": {"status": "abnormal", "result": {"结论": "UGT1A1基因突变"}},
            },
        )
        self.assertIn("bilirubin_genetic", categories)
        self.assertIn("unconjugated_hyperbilirubinemia", evidence.findings("positive"))
        self.assertIn("ugt1a1_positive", evidence.findings("positive"))
        self.assertIn("克里格勒-纳贾尔综合征", names[:5])

    def test_chronic_ent_path_retrieves_chronic_nasopharyngitis(self):
        _, names, categories = self.retrieve_names(
            {"symptoms": ["咽部异物感", "咽干", "反复清嗓", "慢性鼻咽不适"]},
            {"鼻咽镜检查": {"status": "abnormal", "result": {"结论": "鼻咽黏膜充血，淋巴滤泡增生"}}},
        )
        self.assertIn("ent_chronic", categories)
        self.assertIn("慢性鼻咽炎", names[:5])

    def test_congenital_ear_path_retrieves_microtia(self):
        _, names, categories = self.retrieve_names(
            {"age": 6, "symptoms": ["出生即有小耳", "耳廓畸形", "听力下降"]},
            {"听性脑干反应（ABR）": {"status": "abnormal", "result": {"结论": "ABR异常"}}},
        )
        self.assertIn("congenital_ear", categories)
        self.assertIn("小耳畸形", names[:5])

    def test_acute_prostate_path_retrieves_acute_bacterial_prostatitis(self):
        _, names, categories = self.retrieve_names(
            {"symptoms": ["发热", "寒战", "尿频", "尿急", "尿痛", "会阴痛"]},
            {
                "直肠指检（DRE）": {"status": "abnormal", "result": {"结论": "前列腺压痛明显"}},
                "尿液分析（UA）": {"status": "abnormal", "result": {"尿白细胞": "80 个/HPF［参考值：0-5］"}},
            },
        )
        self.assertIn("acute_bacterial_prostate", categories)
        self.assertIn("急性细菌性前列腺炎", names[:5])

    def test_second_degree_av_block_beats_low_magnesium_anchor(self):
        _, decision = self.decide(
            {"symptoms": ["近晕厥", "头晕"], "physical": "心率 40 次/分"},
            {"心电图": {"status": "abnormal", "result": {"结论": "二度房室传导阻滞，PR间期延长，漏搏"}}},
            llm={"diagnosis_candidates": [{"name": "低镁血症", "confidence": 0.9}]},
        )
        self.assertEqual(decision.final_diagnoses[0], "二度房室传导阻滞")
        self.assertNotIn("低镁血症", decision.final_diagnoses)

    def test_specific_ent_diagnosis_beats_upper_respiratory_infection(self):
        _, decision = self.decide(
            {"symptoms": ["咽部异物感", "咽干", "反复清嗓", "慢性鼻咽不适"]},
            {"鼻咽镜检查": {"status": "abnormal", "result": {"结论": "鼻咽黏膜充血，慢性炎症改变"}}},
            llm={"diagnosis_candidates": [{"name": "上呼吸道感染", "confidence": 0.86}]},
        )
        self.assertEqual(decision.final_diagnoses[0], "慢性鼻咽炎")
        self.assertNotIn("上呼吸道感染", decision.final_diagnoses)

    def test_crigler_najjar_required_evidence_uses_ugt1a1_finding(self):
        _, decision = self.decide(
            {"age": 1, "symptoms": ["黄疸", "巩膜黄染", "嗜睡", "喂养差"]},
            {
                "肝功能检查（LFTs）": {
                    "status": "abnormal",
                    "result": {"非结合胆红素": "320 umol/L［参考值：0-17］"},
                },
                "基因检测": {
                    "status": "abnormal",
                    "result": {"结论": "UGT1A1 双等位致病变异"},
                },
            },
            llm={"diagnosis_candidates": [{"name": "肺炎", "confidence": 0.86}]},
        )
        cn = next(item for item in decision.candidates if item.diagnosis == "克里格勒-纳贾尔综合征")
        self.assertTrue(cn.required_met)
        self.assertIn("ugt1a1_positive", cn.matched_evidence)
        self.assertEqual(decision.final_diagnoses[0], "克里格勒-纳贾尔综合征")

    def test_chronic_nasopharyngitis_required_evidence_accepts_cytology(self):
        _, decision = self.decide(
            {"symptoms": ["咽部异物感", "咽干", "反复清嗓", "慢性鼻咽不适"]},
            {
                "脱落细胞学检查": {
                    "status": "abnormal",
                    "result": {"结论": "脱落细胞学提示慢性炎症"},
                }
            },
            llm={"diagnosis_candidates": [{"name": "上呼吸道感染", "confidence": 0.86}]},
        )
        ent = next(item for item in decision.candidates if item.diagnosis == "慢性鼻咽炎")
        self.assertTrue(ent.required_met)
        self.assertIn("cytology_chronic_inflammation", ent.matched_evidence)
        self.assertEqual(decision.final_diagnoses[0], "慢性鼻咽炎")

    def test_negative_urine_culture_defers_acute_bacterial_prostatitis(self):
        _, decision = self.decide(
            {"symptoms": ["发热", "寒战", "尿频", "尿急", "尿痛", "会阴痛"]},
            {
                "直肠指检（DRE）": {
                    "status": "abnormal",
                    "result": {"结论": "压痛明显"},
                },
                "尿液分析（UA）": {
                    "status": "abnormal",
                    "result": {"尿白细胞": "80 个/HPF［参考值：0-5］"},
                },
                "尿培养": {
                    "status": "normal",
                    "result": {"结论": "尿培养无生长"},
                },
            },
            llm={"diagnosis_candidates": [{"name": "前列腺增生", "confidence": 0.86}]},
        )
        prostatitis = next(item for item in decision.candidates if item.diagnosis == "急性细菌性前列腺炎")
        self.assertFalse(prostatitis.required_met)
        self.assertEqual(prostatitis.eligibility_status, "Deferred")
        self.assertFalse(prostatitis.hard_contradiction)
        self.assertIn("urine_culture_no_growth", prostatitis.soft_contradicted_evidence)
        self.assertNotIn("急性细菌性前列腺炎", decision.final_diagnoses)

    def test_pyuria_with_negative_urine_markers_is_not_final_prostatitis(self):
        _, decision = self.decide(
            {"symptoms": ["乏力", "低热"]},
            {
                "尿培养": {
                    "status": "normal",
                    "result": {"结论": "尿培养无生长"},
                },
                "尿液分析（UA）": {
                    "status": "abnormal",
                    "result": {
                        "尿白细胞": "80 个/HPF［参考值：0-5］",
                        "白细胞酯酶": "阴性",
                        "亚硝酸盐": "阴性",
                    },
                },
            },
            llm={"diagnosis_candidates": [{"name": "急性细菌性前列腺炎", "confidence": 0.86}]},
        )
        prostatitis = next(item for item in decision.candidates if item.diagnosis == "急性细菌性前列腺炎")
        self.assertEqual(prostatitis.eligibility_status, "DifferentialOnly")
        self.assertIn("urine_culture_no_growth", prostatitis.eligibility_blockers)
        self.assertTrue(
            any(
                item.get("role") == "negative_pattern"
                for item in prostatitis.evidence_pattern_matches
            )
        )
        self.assertNotIn("急性细菌性前列腺炎", decision.final_diagnoses)

    def test_microtia_beats_fracture_when_congenital_evidence_present(self):
        _, decision = self.decide(
            {"age": 5, "symptoms": ["出生即有小耳", "耳廓畸形", "听力下降"]},
            {"颞骨CT扫描（颞骨CT）": {"status": "abnormal", "result": {"结论": "外耳道闭锁，颞骨发育异常"}}},
            llm={"diagnosis_candidates": [{"name": "骨折", "confidence": 0.86}]},
        )
        self.assertEqual(decision.final_diagnoses[0], "小耳畸形")
        self.assertNotIn("骨折", decision.final_diagnoses)

    def test_microtia_positive_evidence_downgrades_spurious_negative_direct_mention(self):
        microtia = "\u5c0f\u8033\u7578\u5f62"
        fracture = "\u9aa8\u6298"
        evidence = EvidenceBundle(
            [
                Observation("microtia", "test", confidence=0.95),
                Observation("external_auditory_canal_atresia", "test", confidence=0.92),
                Observation("congenital_onset", "test", confidence=0.9),
                Observation(f"diagnosis:{microtia}", "test", confidence=0.98),
                Observation(
                    f"diagnosis:{microtia}",
                    "audit",
                    polarity="negative",
                    confidence=0.94,
                ),
            ]
        )
        decision = self.engine.decide(
            {"diagnosis_candidates": [{"name": fracture, "confidence": 0.86}]},
            [],
            evidence,
        )
        candidate = next(item for item in decision.candidates if item.diagnosis == microtia)
        self.assertFalse(candidate.hard_contradiction)
        self.assertIn(f"diagnosis:{microtia}", candidate.soft_contradicted_evidence)
        self.assertEqual(decision.final_diagnoses[0], microtia)
        self.assertNotIn(fracture, decision.final_diagnoses)

    def test_acute_bacterial_prostatitis_beats_bph(self):
        _, decision = self.decide(
            {"symptoms": ["发热", "寒战", "尿频", "尿急", "尿痛", "会阴痛"]},
            {
                "直肠指检（DRE）": {"status": "abnormal", "result": {"结论": "前列腺压痛明显"}},
                "尿液分析（UA）": {"status": "abnormal", "result": {"尿白细胞": "80 个/HPF［参考值：0-5］"}},
                "尿培养": {"status": "abnormal", "result": {"结论": "尿培养检出大肠埃希菌"}},
            },
            llm={"diagnosis_candidates": [{"name": "前列腺增生", "confidence": 0.86}]},
        )
        self.assertEqual(decision.final_diagnoses[0], "急性细菌性前列腺炎")
        self.assertNotIn("前列腺增生", decision.final_diagnoses)

    def test_graph_metadata_loaded_for_new_targets(self):
        targets = {
            "创伤后骨关节炎": ("musculoskeletal", "post_traumatic_degenerative_joint"),
            "右位心": ("cardiovascular", "congenital_structural_heart"),
            "室间隔缺损（VSD）": ("cardiovascular", "congenital_structural_heart"),
            "压力性尿失禁": ("genitourinary", "urinary_incontinence"),
            "急性鼓膜炎": ("ent", "acute_otologic_inflammation"),
            "三尖瓣反流": ("cardiovascular", "valvular_right_heart"),
        }
        for name, (system, family) in targets.items():
            with self.subTest(name=name):
                entry = self.engine.knowledge.get(name)
                self.assertEqual(entry.get("body_system"), system)
                self.assertEqual(entry.get("disease_family"), family)

    def test_post_traumatic_osteoarthritis_enters_top5_and_fracture_is_lower(self):
        _, names, categories = self.retrieve_names(
            {
                "symptoms": [
                    "外伤后膝关节疼痛",
                    "活动后疼痛",
                    "膝关节僵硬",
                ]
            },
            {
                "X线检查": {
                    "status": "abnormal",
                    "result": {"结论": "关节间隙变窄，骨赘形成，软骨下硬化"},
                }
            },
        )
        self.assertIn("post_traumatic_osteoarthritis", categories)
        self.assertIn("创伤后骨关节炎", names[:5])
        if "骨折" in names:
            self.assertLess(names.index("创伤后骨关节炎"), names.index("骨折"))
        _, decision = self.decide(
            {
                "symptoms": [
                    "步行距离增加后膝关节疼痛",
                    "爬楼梯后疼痛",
                    "膝关节僵硬",
                ]
            },
            {
                "X线检查": {
                    "status": "abnormal",
                    "result": {"结论": "关节间隙变窄，骨赘形成，软骨下硬化"},
                }
            },
            llm={"diagnosis_candidates": [{"name": "骨关节炎", "confidence": 0.98}]},
        )
        self.assertEqual(decision.final_diagnoses[0], "创伤后骨关节炎")
        self.assertNotIn("骨关节炎", decision.final_diagnoses)

    def test_dextrocardia_and_vsd_are_retrieved_and_authorized_together(self):
        info = {
            "age": 1,
            "symptoms": ["喂养困难", "多汗", "生长迟缓", "活动后气促"],
        }
        exams = {
            "胸部X线检查（CXR）": {
                "status": "abnormal",
                "result": {"结论": "右位心影，肺血增多"},
            },
            "心电图（ECG）": {
                "status": "abnormal",
                "result": {"结论": "镜像心电图"},
            },
            "超声心动图": {
                "status": "abnormal",
                "result": {"结论": "膜周部室间隔缺损，左向右分流"},
            },
        }
        _, names, categories = self.retrieve_names(info, exams)
        self.assertIn("congenital_structural_heart", categories)
        self.assertIn("右位心", names[:5])
        self.assertIn("室间隔缺损（VSD）", names[:5])
        _, decision = self.decide(
            info,
            exams,
            llm={
                "diagnosis_candidates": [
                    {"name": "慢性阻塞性肺疾病", "confidence": 0.9},
                    {"name": "房间隔缺损", "confidence": 0.82},
                ]
            },
        )
        self.assertIn("右位心", decision.final_diagnoses)
        self.assertIn("室间隔缺损（VSD）", decision.final_diagnoses)
        self.assertNotIn("慢性阻塞性肺疾病", decision.final_diagnoses)
        self.assertNotIn("房间隔缺损", decision.final_diagnoses)

    def test_stress_urinary_incontinence_beats_respiratory_anchor(self):
        _, names, categories = self.retrieve_names(
            {"symptoms": ["咳嗽漏尿", "运动漏尿", "压力性尿失禁"]},
            {
                "尿液分析（UA）": {
                    "status": "normal",
                    "result": {"白细胞酯酶": "阴性", "亚硝酸盐": "阴性"},
                },
                "尿培养": {
                    "status": "normal",
                    "result": {"结论": "尿培养无生长"},
                },
            },
        )
        self.assertIn("urinary_incontinence", categories)
        self.assertIn("压力性尿失禁", names[:5])
        self.assertNotIn("支气管炎", names[:3])

    def test_acute_tympanitis_beats_uri_and_microtia(self):
        _, names, categories = self.retrieve_names(
            {"symptoms": ["急性耳痛", "耳鸣", "听力下降"]},
            {
                "体格检查": {
                    "status": "abnormal",
                    "result": {"结论": "鼓膜充血，鼓膜疱疹，鼓膜炎症"},
                }
            },
        )
        self.assertIn("acute_otologic_inflammation", categories)
        self.assertIn("急性鼓膜炎", names[:5])
        self.assertNotIn("上呼吸道感染", names[:3])
        if "小耳畸形" in names:
            self.assertLess(names.index("急性鼓膜炎"), names.index("小耳畸形"))
        _, decision = self.decide(
            {"symptoms": ["急性耳痛", "耳鸣", "听力下降"]},
            {
                "耳镜检查": {
                    "status": "abnormal",
                    "result": {"结论": "鼓膜充血，鼓膜疱疹，鼓膜炎症"},
                }
            },
            llm={"diagnosis_candidates": [{"name": "中耳炎", "confidence": 0.98}]},
        )
        self.assertEqual(decision.final_diagnoses[0], "急性鼓膜炎")
        self.assertNotIn("中耳炎", decision.final_diagnoses)

    def test_acute_tympanitis_trigger_beats_parent_otitis(self):
        _, decision = self.decide(
            {
                "symptoms": ["急性耳痛", "耳鸣", "听力下降"],
                "history": {"triggers": "棉签掏耳后出现耳痛"},
            },
            {
                "耳镜检查": {
                    "status": "normal",
                    "result": {"鼓膜": "完整"},
                }
            },
            llm={"diagnosis_candidates": [{"name": "中耳炎", "confidence": 0.98}]},
        )
        acute = next(item for item in decision.candidates if item.diagnosis == "急性鼓膜炎")
        self.assertTrue(acute.required_met)
        self.assertEqual(decision.final_diagnoses[0], "急性鼓膜炎")
        self.assertNotIn("中耳炎", decision.final_diagnoses)

    def test_tricuspid_regurgitation_remains_top_with_right_heart_evidence(self):
        _, decision = self.decide(
            {"symptoms": ["双下肢水肿", "呼吸困难", "腹胀"]},
            {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"结论": "三尖瓣反流，右心房增大，右心室扩大"},
                }
            },
            llm={"diagnosis_candidates": [{"name": "卵巢过度刺激综合征", "confidence": 0.85}]},
        )
        self.assertEqual(decision.final_diagnoses[0], "三尖瓣反流")
        self.assertNotIn("卵巢过度刺激综合征", decision.final_diagnoses)

    def test_varicella_retrieval_beats_eczema_anchor(self):
        evidence, names, categories = self.retrieve_names(
            {
                "age": 4,
                "symptoms": [
                    "皮肤水疱伴瘙痒",
                    "低热",
                    "幼儿园同班小朋友有类似水疱",
                ],
            },
            {
                "体格检查": {
                    "status": "abnormal",
                    "result": {"皮肤": "躯干和面部成批小水疱，部分结痂"},
                }
            },
        )
        self.assertIn("vesicular_viral_exanthem", categories)
        self.assertIn("vesicular_rash", evidence.findings("positive"))
        self.assertIn("childcare_exposure", evidence.findings("positive"))
        self.assertIn("水痘", names[:5])
        _, decision = self.decide(
            {"symptoms": ["皮肤水疱伴瘙痒", "低热", "同班小朋友水痘"]},
            {"体格检查": {"status": "abnormal", "result": {"皮肤": "成批水疱"}}},
            llm={"diagnosis_candidates": [{"name": "湿疹", "confidence": 0.9}]},
        )
        self.assertEqual(decision.final_diagnoses[0], "水痘")
        self.assertNotIn("湿疹", decision.final_diagnoses)

    def test_tuberculous_pericarditis_retrieval_beats_generic_lung_infection(self):
        _, names, categories = self.retrieve_names(
            {
                "symptoms": ["胸痛平卧加重", "前倾缓解", "低热", "咳嗽", "呼吸困难"],
                "history": {"exposure": "近期接触肺结核确诊患者"},
            },
            {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"结论": "中量心包积液"},
                },
                "胸部CT扫描（Chest CT）": {
                    "status": "abnormal",
                    "result": {"结论": "心包增厚并可见结核感染线索"},
                },
            },
        )
        self.assertIn("tuberculous_pericardial_disease", categories)
        self.assertIn("结核性心包炎", names[:5])
        if "肺炎" in names:
            self.assertLess(names.index("结核性心包炎"), names.index("肺炎"))

    def test_lacrimal_gland_inflammation_retrieval_beats_fracture(self):
        _, names, categories = self.retrieve_names(
            {"symptoms": ["眼睑外上方肿胀", "泪腺区疼痛", "流泪", "畏光"]},
            {
                "体格检查": {
                    "status": "abnormal",
                    "result": {"眼部": "泪腺区肿胀，局部压痛"},
                }
            },
        )
        self.assertIn("lacrimal_gland_inflammation", categories)
        self.assertIn("泪腺炎", names[:5])
        self.assertNotIn("骨折", names[:3])

    def test_esophageal_ulcer_retrieval_uses_endoscopy_finding(self):
        evidence, names, categories = self.retrieve_names(
            {"symptoms": ["吞咽痛", "胸骨后烧灼样疼痛", "烧心", "反酸"]},
            {
                "胃镜": {
                    "status": "abnormal",
                    "result": {"结论": "食管黏膜溃疡伴糜烂"},
                }
            },
        )
        self.assertIn("esophageal_mucosal_injury", categories)
        self.assertIn("esophageal_ulcer", evidence.findings("positive"))
        self.assertIn("食管溃疡", names[:5])
        _, decision = self.decide(
            {"symptoms": ["吞咽痛", "胸骨后烧灼样疼痛", "烧心"]},
            {"胃镜": {"status": "abnormal", "result": {"结论": "食管溃疡"}}},
            llm={"diagnosis_candidates": [{"name": "骨折", "confidence": 0.88}]},
        )
        self.assertEqual(decision.final_diagnoses[0], "食管溃疡")
        self.assertNotIn("骨折", decision.final_diagnoses)

    def test_end_stage_renal_disease_retrieval_beats_tricuspid_anchor(self):
        evidence, names, categories = self.retrieve_names(
            {"symptoms": ["少尿", "皮肤瘙痒", "双下肢水肿", "乏力"]},
            {
                "肾功能检查（RFTs）": {
                    "status": "abnormal",
                    "result": {
                        "肌酐": "820 umol/L［参考范围：45-84］",
                        "eGFR": "5 ml/min/1.73m2［参考范围：90-120］",
                        "尿素氮": "32 mmol/L［参考范围：2.9-8.2］",
                    },
                },
                "血清电解质": {
                    "status": "abnormal",
                    "result": {"血钾": "6.2 mmol/L［参考范围：3.5-5.5］"},
                },
            },
        )
        self.assertIn("renal_failure", categories)
        self.assertIn("renal_impairment", evidence.findings("positive"))
        self.assertIn("egfr_low", evidence.findings("positive"))
        self.assertIn("终末期肾病", names[:5])
        if "三尖瓣反流" in names:
            self.assertLess(names.index("终末期肾病"), names.index("三尖瓣反流"))

    def test_upper_respiratory_infection_retrieval_beats_pulmonary_package(self):
        _, names, categories = self.retrieve_names(
            {"symptoms": ["流鼻涕", "鼻塞", "咳嗽", "低热", "受凉后2天"]},
            {
                "胸部X线检查（CXR）": {
                    "status": "normal",
                    "result": {"结论": "未见肺部实变或肺不张"},
                }
            },
        )
        self.assertIn("upper_respiratory_infection", categories)
        self.assertIn("上呼吸道感染", names[:5])
        self.assertNotIn("肺不张", names[:3])

    def test_yaws_rural_crusted_skin_path_enters_top5(self):
        _, names, categories = self.retrieve_names(
            {
                "age": 6,
                "symptoms": [
                    "农村亲戚家接触后皮损结痂流黄水",
                    "共用毛巾",
                    "其他儿童类似皮损",
                    "腹股沟淋巴结肿大",
                ],
            },
            {"体格检查": {"status": "abnormal", "result": {"皮肤": "乳头瘤样皮损，渗出结痂"}}},
        )
        self.assertIn("treponemal_skin_bone_infection", categories)
        self.assertIn("雅司病", names[:5])
        if "湿疹" in names:
            self.assertLess(names.index("雅司病"), names.index("湿疹"))

    def test_pulmonary_tuberculosis_contact_hemoptysis_beats_bronchopneumonia(self):
        _, names, categories = self.retrieve_names(
            {
                "symptoms": ["咳嗽10天", "低热", "盗汗", "痰中带血"],
                "history": {"exposure": "接触确诊肺结核患者"},
            },
            {
                "胸部CT扫描（Chest CT）": {
                    "status": "abnormal",
                    "result": {"结论": "右上肺空洞影，考虑结核感染"},
                }
            },
        )
        self.assertIn("pulmonary_tuberculosis", categories)
        self.assertIn("肺结核", names[:5])
        if "支气管肺炎" in names:
            self.assertLess(names.index("肺结核"), names.index("支气管肺炎"))

    def test_presbyopia_retrieval_beats_structural_eye_anchor(self):
        _, names, categories = self.retrieve_names(
            {"age": 52, "symptoms": ["看近模糊", "阅读困难", "填表困难", "远处视物尚可"]},
            {"屈光检查": {"status": "abnormal", "result": {"结论": "+1.50D阅读镜可改善近视力"}}},
        )
        self.assertIn("age_related_refractive_error", categories)
        self.assertIn("老视", names[:5])
        self.assertNotIn("晶状体脱位", names[:3])

    def test_ovotesticular_dsd_retrieval_beats_renal_stone_anchor(self):
        _, names, categories = self.retrieve_names(
            {"symptoms": ["外生殖器发育异常", "尿道下裂", "隐睾", "性别发育异常"]},
            {
                "染色体核型分析": {
                    "status": "abnormal",
                    "result": {"结论": "46,XX/46,XY嵌合"},
                },
                "组织病理学检查": {
                    "status": "abnormal",
                    "result": {"结论": "可见卵巢和睾丸组织"},
                },
            },
        )
        self.assertIn("sex_development_disorder", categories)
        self.assertIn("卵睾性别发育异常（Ovotesticular DSD）", names[:5])
        self.assertNotIn("肾结石", names[:3])

    def test_tmj_dislocation_retrieval_beats_zoster_anchor(self):
        _, names, categories = self.retrieve_names(
            {"symptoms": ["打哈欠后张口不能闭口", "嘴巴合不上", "耳前区疼痛", "咬合错乱"]},
            {"体格检查": {"status": "abnormal", "result": {"颌面部": "颞下颌关节脱位"}}},
        )
        self.assertIn("temporomandibular_joint_disorder", categories)
        self.assertIn("颞下颌关节脱位（TMJ）", names[:5])
        self.assertNotIn("带状疱疹", names[:3])

    def test_anogenital_hpv_and_trichomonas_retrieval(self):
        _, names, categories = self.retrieve_names(
            {"symptoms": ["外阴瘙痒", "黄绿色泡沫样白带", "肛周菜花样疣体"]},
            {
                "体格检查": {"status": "abnormal", "result": {"外阴": "菜花样疣体"}},
                "阴道分泌物湿片检查": {
                    "status": "abnormal",
                    "result": {"结论": "滴虫阳性，阴道pH>4.5"},
                },
            },
        )
        self.assertIn("anogenital_hpv_vaginitis", categories)
        self.assertIn("尖锐湿疣", names[:5])
        self.assertIn("滴虫性阴道炎", names[:5])
        self.assertNotIn("湿疹", names[:3])

    def test_urachal_cyst_retrieval_beats_fracture_anchor(self):
        _, names, categories = self.retrieve_names(
            {"symptoms": ["脐部流液", "脐孔流脓", "下腹正中疼痛"]},
            {
                "腹部B超": {
                    "status": "abnormal",
                    "result": {"结论": "膀胱顶部至脐部之间可见脐尿管囊肿"},
                }
            },
        )
        self.assertIn("urachal_remnant", categories)
        self.assertIn("脐尿管囊肿", names[:5])
        self.assertNotIn("骨折", names[:3])


if __name__ == "__main__":
    unittest.main()
