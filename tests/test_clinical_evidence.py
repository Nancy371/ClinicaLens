import unittest

from agent.clinical_evidence import (
    ClinicalEvidenceNormalizer,
    EvidenceAgent,
    HybridEvidenceCompiler,
    ReasoningEvidenceAdapter,
)


class ClinicalEvidenceNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = ClinicalEvidenceNormalizer("data/ref_data")

    def test_local_negation_does_not_leak_to_next_clause(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {
                        "结论": "未见先天性缺损（如 ASD、VSD），但见重度二尖瓣反流。"
                    },
                }
            },
        )
        asd = [item for item in bundle.observations if item.finding == "atrial_septal_defect"]
        asd_dx = [item for item in bundle.observations if item.finding == "diagnosis:房间隔缺损"]
        mitral = [item for item in bundle.observations if item.finding == "mitral_regurgitation"]
        self.assertTrue(asd and all(item.polarity == "negative" for item in asd))
        self.assertTrue(asd_dx and all(item.polarity == "negative" for item in asd_dx))
        self.assertTrue(mitral and any(item.polarity == "positive" for item in mitral))

    def test_numeric_value_and_reference_range_create_low_magnesium(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "电解质": {
                    "status": "abnormal",
                    "result": {"血镁": "0.45 mmol/L［参考值：0.75-1.02 mmol/L］"},
                }
            },
        )
        hits = [item for item in bundle.observations if item.finding == "low_magnesium"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].direction, "low")
        self.assertAlmostEqual(hits[0].value, 0.45)

    def test_weakness_is_not_treated_as_negation(self):
        bundle = self.normalizer.normalize(
            {"symptoms": ["全身无力", "手足抽筋", "心悸"]},
            {},
        )
        positives = {
            item.finding for item in bundle.observations if item.polarity == "positive"
        }
        negatives = {
            item.finding for item in bundle.observations if item.polarity == "negative"
        }
        self.assertIn("weakness", positives)
        self.assertIn("muscle_cramp", positives)
        self.assertIn("palpitation", positives)
        self.assertNotIn("weakness", negatives)

    def test_urine_magnesium_uses_leaf_value_not_field_number(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "24小时尿电解质检测": {
                    "status": "abnormal",
                    "result": {
                        "24小时尿镁": "1.2 mmol/24h［参考范围：3.0-5.0］",
                    },
                }
            },
        )
        field = next(item for item in bundle.observations if item.finding == "field:24小时尿镁")
        self.assertAlmostEqual(field.value, 1.2)
        self.assertEqual(field.direction, "low")
        self.assertEqual(field.unit, "mmol/24h")
        self.assertIn("low_urine_magnesium", bundle.findings("positive"))
        self.assertIn("magnesium_depletion", bundle.findings("positive"))
        self.assertNotIn("low_magnesium", bundle.findings("positive"))

    def test_magnesium_load_retention_with_textual_upper_range_is_high(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "镁负荷试验": {
                    "status": "abnormal",
                    "result": {
                        "镁负荷保留率": "62%［参考范围：镁储备充足时通常＜20-30%］",
                    },
                }
            },
        )
        field = next(item for item in bundle.observations if item.finding == "field:镁负荷保留率")
        self.assertAlmostEqual(field.value, 62.0)
        self.assertEqual(field.direction, "high")
        self.assertIn("magnesium_load_retention_high", bundle.findings("positive"))
        self.assertIn("magnesium_depletion", bundle.findings("positive"))

    def test_magnesium_load_retention_negated_conclusion_suppresses_positive_findings(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "镁负荷试验": {
                    "status": "abnormal",
                    "result": {
                        "镁负荷保留率": (
                            "62%［参考范围：镁储备充足时通常＜20-30%］；"
                            "结论：排除低镁血症。"
                        ),
                    },
                }
            },
        )
        self.assertNotIn("magnesium_load_retention_high", bundle.findings("positive"))
        self.assertNotIn("magnesium_depletion", bundle.findings("positive"))
        suppressed = self.normalizer.last_suppressed_structured_findings
        self.assertTrue(suppressed)
        self.assertEqual(suppressed[0]["affected_diagnosis"], "低镁血症")
        self.assertEqual(suppressed[0]["reason"], "same_segment_diagnosis_negation")

    def test_cannot_rule_out_low_magnesium_does_not_suppress_positive_findings(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "镁负荷试验": {
                    "status": "abnormal",
                    "result": {
                        "镁负荷保留率": (
                            "62%［参考范围：镁储备充足时通常＜20-30%］；"
                            "结论：不能排除低镁血症。"
                        ),
                    },
                }
            },
        )
        self.assertIn("magnesium_load_retention_high", bundle.findings("positive"))
        self.assertIn("magnesium_depletion", bundle.findings("positive"))
        self.assertEqual(self.normalizer.last_suppressed_structured_findings, [])

    def test_reference_only_magnesium_range_has_no_numeric_disease_evidence(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "报告模板": {
                    "status": "normal",
                    "result": {"备注": "血镁参考范围：0.75-1.02 mmol/L"},
                }
            },
        )
        field = next(item for item in bundle.observations if item.finding == "field:备注")
        self.assertIsNone(field.value)
        self.assertNotIn("low_magnesium", bundle.findings("positive"))
        self.assertNotIn("magnesium_depletion", bundle.findings("positive"))

    def test_reference_example_does_not_become_positive_diagnosis(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "报告模板": {
                    "status": "normal",
                    "result": {"备注": "参考范围：ASD 仅用于报告模板示例"},
                }
            },
        )
        positives = {
            item.finding for item in bundle.observations if item.polarity == "positive"
        }
        self.assertNotIn("atrial_septal_defect", positives)
        self.assertNotIn("diagnosis:房间隔缺损", positives)

    def test_cbc_and_smear_create_leukemia_atomic_evidence(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "\u5168\u8840\u7ec6\u80de\u8ba1\u6570\uff08CBC\uff09": {
                    "status": "abnormal",
                    "result": {
                        "WBC": "18.6 x 10^9/L\uff08\u53c2\u8003\u8303\u56f4\uff1a4.0-10.0\uff09",
                        "Hgb": "66 g/L\uff08\u53c2\u8003\u8303\u56f4\uff1a115-150\uff09",
                        "PLT": "8 x 10^9/L\uff08\u53c2\u8003\u8303\u56f4\uff1a100-300\uff09",
                    },
                },
                "\u5916\u5468\u8840\u6d82\u7247": {
                    "status": "abnormal",
                    "result": {
                        "\u7ed3\u8bba": "\u5916\u5468\u8840\u53ef\u89c1\u5927\u91cf\u5faa\u73af\u6bcd\u7ec6\u80de\u7ea625%",
                    },
                },
            },
        )
        positives = set(bundle.findings("positive"))
        self.assertIn("hemoglobin_low", positives)
        self.assertIn("platelet_low", positives)
        self.assertIn("white_blood_cell_abnormal", positives)
        self.assertIn("blast_present", positives)

    def test_vasculitis_serology_and_red_cell_casts_are_structured(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "抗中性粒细胞胞质抗体（ANCA）谱": {
                    "status": "abnormal",
                    "result": {"髓过氧化物酶抗体": "MPO抗体阳性"},
                },
                "尿液分析（UA）": {
                    "status": "abnormal",
                    "result": {"尿沉渣": "可见红细胞管型"},
                },
            },
        )
        findings = bundle.findings("positive")
        self.assertIn("mpo_anca_positive", findings)
        self.assertIn("microscopic_hematuria", findings)

    def test_bronchopneumonia_imaging_terms_are_structured(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "胸部CT扫描（Chest CT）": {
                    "status": "abnormal",
                    "result": {"结论": "右下叶实变并见空气支气管征"},
                },
                "支气管镜检查": {
                    "status": "abnormal",
                    "result": {"结论": "支气管内见脓性分泌物"},
                },
            },
        )
        findings = bundle.findings("positive")
        self.assertIn("pulmonary_consolidation", findings)
        self.assertNotIn("pneumonia_infiltrate", findings)
        self.assertIn("bronchopneumonia", findings)

    def test_imaging_infiltrate_with_pneumonia_impression_is_deetiologized(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "\u80f8\u90e8CT\u626b\u63cf\uff08Chest CT\uff09": {
                    "status": "abnormal",
                    "result": {
                        "\u7ed3\u8bba": "\u53f3\u80ba\u7247\u72b6\u6d78\u6da6\u5f71\uff0c\u8003\u8651\u652f\u6c14\u7ba1\u80ba\u708e"
                    },
                },
            },
        )
        observations = {item.finding: item for item in bundle.observations}
        self.assertIn("pulmonary_infiltrative_opacity", observations)
        self.assertEqual(observations["pulmonary_infiltrative_opacity"].semantic_level, "fact")
        self.assertIn("bronchopneumonia_suspected", observations)
        self.assertEqual(
            observations["bronchopneumonia_suspected"].semantic_level,
            "clinical_impression",
        )
        self.assertNotIn("pneumonia_infiltrate", observations)

    def test_radiotherapy_history_creates_typed_thoracic_treatment_fact(self):
        bundle = self.normalizer.normalize(
            {
                "history": "\u60a3\u80053\u4e2a\u6708\u524d\u56e0\u80ba\u764c\u63a5\u53d7\u80f8\u90e8\u653e\u7597\uff0c\u968f\u540e\u51fa\u73b0\u54b3\u55fd\u6c14\u4fc3"
            },
            {},
        )
        observations = {item.finding: item for item in bundle.observations}
        self.assertIn("history_of_radiotherapy", observations)
        self.assertIn("thoracic_radiotherapy", observations)
        radiotherapy = observations["thoracic_radiotherapy"]
        self.assertEqual(radiotherapy.observation_type, "treatment_history")
        self.assertEqual(radiotherapy.semantic_level, "fact")
        self.assertEqual(radiotherapy.anatomy, "thorax")
        self.assertIn("3", radiotherapy.temporality)

    def test_generic_or_invalid_radiotherapy_history_does_not_create_thoracic_exposure(self):
        generic = self.normalizer.normalize({"history": "\u65e2\u5f80\u63a5\u53d7\u8fc7\u653e\u7597"}, {})
        generic_findings = generic.findings("positive")
        self.assertIn("history_of_radiotherapy", generic_findings)
        self.assertNotIn("thoracic_radiotherapy", generic_findings)

        planned = self.normalizer.normalize({"history": "\u5efa\u8bae\u4e0b\u5468\u5f00\u59cb\u80f8\u90e8\u653e\u7597"}, {})
        self.assertNotIn("history_of_radiotherapy", planned.findings("positive"))
        self.assertNotIn("thoracic_radiotherapy", planned.findings("positive"))

        family = self.normalizer.normalize({"history": "\u6bcd\u4eb2\u65e2\u5f80\u63a5\u53d7\u8fc7\u80f8\u90e8\u653e\u7597"}, {})
        self.assertNotIn("history_of_radiotherapy", family.findings("positive"))
        self.assertNotIn("thoracic_radiotherapy", family.findings("positive"))

        negative = self.normalizer.normalize({"history": "\u5426\u8ba4\u63a5\u53d7\u8fc7\u653e\u7597"}, {})
        self.assertNotIn("history_of_radiotherapy", negative.findings("positive"))

    def test_ugt1a1_gene_result_becomes_standard_finding(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "基因检测": {
                    "status": "abnormal",
                    "result": {"结论": "UGT1A1 双等位致病变异"},
                }
            },
        )
        findings = bundle.findings("positive")
        self.assertIn("ugt1a1_positive", findings)
        self.assertIn("genetic_suspicion", findings)

    def test_nasopharyngeal_cytology_and_scope_become_standard_findings(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "鼻咽镜检查": {
                    "status": "abnormal",
                    "result": {"结论": "鼻咽黏膜充血，淋巴滤泡增生，慢性炎症改变"},
                },
                "脱落细胞学检查": {
                    "status": "abnormal",
                    "result": {"结论": "脱落细胞学提示慢性炎症"},
                },
            },
        )
        findings = bundle.findings("positive")
        self.assertIn("nasopharyngoscopy_abnormal", findings)
        self.assertIn("nasopharyngeal_chronic_inflammation", findings)
        self.assertIn("cytology_chronic_inflammation", findings)

    def test_otoscopy_abnormality_becomes_tympanitis_findings(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "耳镜检查": {
                    "status": "abnormal",
                    "result": {"结论": "鼓膜明显充血，鼓膜疱疹，局部鼓膜炎症"},
                }
            },
        )
        findings = bundle.findings("positive")
        self.assertIn("acute_tympanitis", findings)
        self.assertIn("tympanic_membrane_inflammation", findings)
        self.assertIn("tympanic_bulla", findings)

    def test_stage2_specialty_findings_are_structured(self):
        bundle = self.normalizer.normalize(
            {
                "symptoms": [
                    "皮肤水疱伴瘙痒，幼儿园同班小朋友有类似情况",
                    "平卧加重、前倾缓解的胸痛",
                    "阴道出血伴下腹痛，停经后早孕",
                ]
            },
            {
                "裂隙灯检查": {
                    "status": "abnormal",
                    "result": {"结论": "虹膜裂隙，钥匙孔样瞳孔"},
                },
                "胃镜": {
                    "status": "abnormal",
                    "result": {"结论": "食管黏膜溃疡"},
                },
                "血清β-hCG": {
                    "status": "abnormal",
                    "result": {"β-hCG": "1500 IU/L［参考值：<5］"},
                },
                "基因检测": {
                    "status": "abnormal",
                    "result": {"核型": "47,XXX"},
                },
            },
        )
        findings = bundle.findings("positive")
        self.assertIn("vesicular_rash", findings)
        self.assertIn("childcare_exposure", findings)
        self.assertIn("pericarditic_chest_pain", findings)
        self.assertIn("vaginal_bleeding", findings)
        self.assertIn("early_pregnancy", findings)
        self.assertIn("hcg_positive", findings)
        self.assertIn("iris_coloboma", findings)
        self.assertIn("esophageal_ulcer", findings)
        self.assertIn("triple_x_karyotype", findings)

    def test_stage2_seed53_findings_are_structured(self):
        bundle = self.normalizer.normalize(
            {
                "symptoms": [
                    "看近模糊，阅读困难，需要老花镜",
                    "张口后不能闭口，耳前区疼痛",
                    "外阴瘙痒伴黄绿色泡沫样分泌物",
                    "脐部流液伴下腹正中包块",
                    "农村接触后皮损结痂流黄水，腹股沟淋巴结肿大",
                ]
            },
            {
                "染色体核型分析": {
                    "status": "abnormal",
                    "result": {"结论": "46,XX/46,XY嵌合，提示性别发育异常"},
                },
                "组织病理学检查": {
                    "status": "abnormal",
                    "result": {"结论": "可见卵巢和睾丸组织"},
                },
                "体格检查": {
                    "status": "abnormal",
                    "result": {"外阴": "菜花样疣体，考虑HPV相关病变"},
                },
            },
        )
        findings = bundle.findings("positive")
        for finding in {
            "near_vision_difficulty",
            "age_related_near_blur",
            "jaw_locked_open",
            "preauricular_pain",
            "frothy_vaginal_discharge",
            "vaginal_pruritus",
            "umbilical_discharge",
            "umbilical_mass",
            "rural_child_contact",
            "crusted_exudative_skin_ulcer",
            "regional_lymphadenopathy",
            "sex_development_disorder",
            "karyotype_mosaic",
            "ovotesticular_tissue",
            "anogenital_warts",
            "cauliflower_lesions",
        }:
            self.assertIn(finding, findings)

    def test_interpreter_maps_patient_language_to_clinical_findings(self):
        bundle = self.normalizer.normalize(
            {
                "symptoms": [
                    "\u8d70\u4e24\u6b65\u5c31\u5598",
                    "\u996d\u540e\u6076\u5fc3",
                    "\u65e9\u6668\u8d77\u5e8a\u773c\u775b\u80bf",
                    "\u559d\u5f88\u591a\u6c34",
                    "\u70ed\u5e26\u5730\u533a\u751f\u6d3b\u540e\u51fa\u73b0\u6df1\u90e8\u6e83\u75a1\u5e76\u7ed3\u75c2",
                    "\u5c40\u90e8\u9aa8\u819c\u708e",
                ],
                "history": "\u519c\u6751\u73af\u5883\u66b4\u9732",
            },
            {},
        )
        findings = bundle.findings("positive")
        for finding in {
            "dyspnea_on_exertion",
            "exercise_intolerance",
            "postprandial_nausea",
            "periorbital_edema",
            "fluid_retention_pattern",
            "polydipsia",
            "tropical_exposure",
            "deep_skin_ulcer",
            "crusted_skin_lesion",
            "periostitis",
            "treponemal_skin_lesion",
            "cardiopulmonary_exertional_pattern",
        }:
            self.assertIn(finding, findings)

    def test_interpreter_v2_prioritizes_high_information_eye_findings(self):
        bundle = self.normalizer.normalize(
            {
                "age": 52,
                "symptoms": [
                    "\u770b\u8fd1\u6a21\u7cca\uff0c\u9605\u8bfb\u56f0\u96be\uff0c\u770b\u624b\u673a\u8d39\u52b2",
                    "\u89c6\u7269\u6a21\u7cca",
                ],
            },
            {
                "\u5c48\u5149\u68c0\u67e5": {
                    "status": "abnormal",
                    "result": {"\u7ed3\u8bba": "+1.50D\u9605\u8bfb\u955c\u53ef\u6539\u5584\u8fd1\u89c6\u529b"},
                }
            },
        )
        findings = bundle.findings("positive")
        self.assertIn("near_vision_difficulty", findings)
        self.assertIn("refractive_correction_improves_near_vision", findings)
        self.assertIn("presbyopia_pattern", findings)
        visual = [item for item in bundle.observations if item.finding == "visual_blurring"]
        self.assertTrue(visual)
        self.assertTrue(any(item.shadowed_by for item in visual))
        self.assertGreaterEqual(
            max(item.information_value for item in bundle.observations if item.finding == "near_vision_difficulty"),
            0.9,
        )

    def test_raw_case_text_enters_interpreter_without_losing_specific_findings(self):
        bundle = self.normalizer.normalize(
            {},
            {},
            raw_case_text="\u770b\u624b\u673a\u5c0f\u5b57\u8d39\u52b2\uff0c\u8fdc\u89c6\u529b\u5c1a\u53ef\uff0c\u9605\u8bfb\u955c\u53ef\u6539\u5584\u8fd1\u89c6\u529b\u3002",
        )
        findings = bundle.findings("positive")
        self.assertIn("near_vision_difficulty", findings)
        self.assertIn("refractive_correction_improves_near_vision", findings)
        self.assertIn("presbyopia_pattern", findings)
        self.assertTrue(
            all(
                item.source_text
                for item in bundle.observations
                if item.source == "raw_case_finding"
            )
        )
        visual = [item for item in bundle.observations if item.finding == "visual_blurring"]
        self.assertTrue(not visual or any(item.shadowed_by for item in visual))

    def test_raw_case_text_with_answer_leakage_is_blocked(self):
        bundle = self.normalizer.normalize(
            {},
            {},
            raw_case_text="expected diagnosis: \u8001\u89c6\u3002\u6807\u51c6\u7b54\u6848\uff1a\u8001\u89c6\u3002\u770b\u624b\u673a\u8d39\u52b2\u3002",
        )
        findings = bundle.findings("positive")
        self.assertNotIn("near_vision_difficulty", findings)
        self.assertTrue(self.normalizer.last_raw_case_audit["raw_case_blocked"])
        self.assertEqual(
            self.normalizer.last_raw_case_audit["raw_case_blocked_reason"],
            "raw_case_contains_answer_leakage",
        )

    def test_reasoning_adapter_adds_low_magnesium_soft_findings(self):
        adapter = ReasoningEvidenceAdapter()
        observations = adapter.adapt(
            {
                "reasoning": "\u8179\u6cfb\u5bfc\u81f4\u9541\u4e22\u5931\uff0cQTc\u5ef6\u957f\u652f\u6301\u4f4e\u9541\u8840\u75c7\u3002"
            }
        )
        findings = {item.finding for item in observations}
        self.assertIn("magnesium_depletion", findings)
        self.assertIn("low_magnesium_support", findings)
        self.assertTrue(all(item.source == "reasoning_inference" for item in observations))
        self.assertTrue(all(item.source_text for item in observations))
        self.assertTrue(all(item.confidence <= 0.78 for item in observations))

    def test_reasoning_adapter_blocks_differential_only_language(self):
        adapter = ReasoningEvidenceAdapter()
        observations = adapter.adapt({"reasoning": "\u80ba\u764c\u9700\u9274\u522b\u4f46\u8bc1\u636e\u4e0d\u8db3\u3002"})
        self.assertEqual(observations, [])
        self.assertEqual(adapter.last_audit["blocked_reasoning_inference_count"], 0)

    def test_reasoning_adapter_structures_pulmonary_renal_evidence(self):
        adapter = ReasoningEvidenceAdapter()
        observations = adapter.adapt(
            {
                "reasoning": "ANCA\u9633\u6027\u3001\u8840\u5c3f\u548c\u54b3\u8840\u652f\u6301\u80ba\u80be\u7efc\u5408\u5f81\u3002"
            }
        )
        findings = {item.finding for item in observations}
        for finding in {
            "anca_positive",
            "microscopic_hematuria",
            "pulmonary_hemorrhage",
            "pulmonary_renal_syndrome",
        }:
            self.assertIn(finding, findings)

    def test_hybrid_compiler_preserves_objective_evidence_priority(self):
        compiler = HybridEvidenceCompiler(normalizer=self.normalizer)
        bundle = compiler.compile(
            {"symptoms": ["\u8179\u6cfb"]},
            {
                "\u62a5\u544a\u6a21\u677f": {
                    "status": "normal",
                    "result": {"\u5907\u6ce8": "\u8840\u9541\u53c2\u8003\u8303\u56f4\uff1a0.75-1.02 mmol/L"},
                }
            },
            {"reasoning": "\u8179\u6cfb\u5bfc\u81f4\u9541\u4e22\u5931\uff0cQTc\u5ef6\u957f\u652f\u6301\u4f4e\u9541\u8840\u75c7\u3002"},
        )
        findings = bundle.findings("positive")
        self.assertIn("magnesium_depletion", findings)
        self.assertNotIn("low_magnesium", findings)
        self.assertEqual(
            compiler.last_audit["reasoning_inference_finding_count"],
            2,
        )

    def test_visual_blurring_alone_stays_low_information(self):
        bundle = self.normalizer.normalize({"symptoms": ["\u89c6\u7269\u6a21\u7cca"]}, {})
        findings = bundle.findings("positive")
        self.assertIn("visual_blurring", findings)
        self.assertNotIn("near_vision_difficulty", findings)
        self.assertNotIn("presbyopia_pattern", findings)
        visual = next(item for item in bundle.observations if item.finding == "visual_blurring")
        self.assertEqual(visual.evidence_level, "generic")
        self.assertLessEqual(visual.information_value, 0.2)

    def test_colloquial_near_vision_language_decomposes_into_high_value_findings(self):
        bundle = self.normalizer.normalize(
            {
                "symptoms": [
                    "\u6700\u8fd1\u770b\u624b\u673a\u603b\u8981\u62ff\u8fdc\u4e00\u70b9\uff0c\u5149\u7ebf\u6697\u7684\u65f6\u5019\u66f4\u660e\u663e\uff0c"
                    "\u4f46\u662f\u770b\u8fdc\u5904\u8fd8\u53ef\u4ee5\uff0c\u4e5f\u4e0d\u75bc\u4e0d\u7ea2\u3002"
                ]
            },
            {},
        )
        positives = bundle.findings("positive")
        negatives = bundle.findings("negative")
        for finding in {
            "near_vision_difficulty",
            "distance_vision_relatively_preserved",
            "worse_in_dim_light",
            "gradual_onset",
            "accommodation_failure_pattern",
        }:
            self.assertIn(finding, positives)
        self.assertIn("ocular_pain", negatives)
        self.assertIn("ocular_redness", negatives)
        near = next(item for item in bundle.observations if item.finding == "near_vision_difficulty")
        self.assertEqual(near.clinical_pattern, "accommodation_failure_pattern")
        self.assertIn("accommodation_failure", near.mechanism_ids)

    def test_eye_uncertain_and_unknown_are_not_merged_with_negative(self):
        bundle = self.normalizer.normalize(
            {
                "symptoms": [
                    "\u8bf4\u4e0d\u6e05\u662f\u5426\u773c\u75db\uff0c\u5c1a\u672a\u8be2\u95ee\u773c\u7ea2\uff0c\u89c6\u7269\u6a21\u7cca"
                ]
            },
            {},
        )
        polarities = {(item.finding, item.polarity) for item in bundle.observations}
        self.assertIn(("ocular_pain", "uncertain"), polarities)
        self.assertIn(("ocular_redness", "unknown"), polarities)
        self.assertNotIn(("ocular_pain", "negative"), polarities)
        self.assertNotIn(("ocular_redness", "negative"), polarities)

    def test_interpreter_v2_maps_night_vision_and_urachal_language(self):
        bundle = self.normalizer.normalize(
            {
                "symptoms": [
                    "\u591c\u76f2\uff0c\u6697\u9002\u5e94\u5dee",
                    "\u8110\u90e8\u5206\u6ccc\u7269\u4f34\u4e0b\u8179\u6b63\u4e2d\u75bc\u75db",
                ]
            },
            {},
        )
        findings = bundle.findings("positive")
        for finding in {
            "night_vision_decline",
            "nyctalopia_pattern",
            "umbilical_discharge",
            "midline_suprapubic_pain",
            "urachal_remnant_pattern",
        }:
            self.assertIn(finding, findings)

    def test_interpreter_v2_respects_simple_negation_window(self):
        bundle = self.normalizer.normalize(
            {"symptoms": ["\u5426\u8ba4\u770b\u8fd1\u56f0\u96be\uff0c\u65e0\u591c\u76f2\uff0c\u8679\u819c\u65e0\u7f3a\u635f"]},
            {},
        )
        findings = bundle.findings("positive")
        self.assertNotIn("near_vision_difficulty", findings)
        self.assertNotIn("night_vision_decline", findings)
        self.assertNotIn("iris_coloboma", findings)

    def test_normal_otoscopy_does_not_become_tympanitis(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "耳镜检查": {
                    "status": "normal",
                    "result": {"结论": "外耳道清洁，鼓膜完整且活动良好"},
                }
            },
        )
        findings = bundle.findings("positive")
        self.assertNotIn("acute_tympanitis", findings)
        self.assertNotIn("tympanic_membrane_inflammation", findings)
        self.assertNotIn("tympanic_bulla", findings)

    def test_dre_source_with_pressure_pain_becomes_prostate_tenderness(self):
        bundle = self.normalizer.normalize(
            {},
            {
                "直肠指检（DRE）": {
                    "status": "abnormal",
                    "result": {"结论": "压痛明显"},
                }
            },
        )
        self.assertIn("prostate_tenderness", bundle.findings("positive"))

    def test_short_mr_alias_does_not_match_mri(self):
        bundle = self.normalizer.normalize(
            {},
            {"影像": {"status": "normal", "result": {"结论": "MRI 未见异常"}}},
        )
        self.assertFalse(
            any(item.finding == "diagnosis:二尖瓣反流" for item in bundle.observations)
        )

    def test_evidence_agent_builds_graph_categories(self):
        graph = EvidenceAgent(normalizer=self.normalizer).build_graph(
            {"symptoms": ["腹胀", "下腹部不适"], "history": "近期促排卵后取卵"},
            {
                "盆腔超声": {
                    "status": "abnormal",
                    "result": {"卵巢": "双侧卵巢增大，伴腹水"},
                },
                "血常规": {
                    "status": "abnormal",
                    "result": {"红细胞压积": "48%［参考值：35-45%］"},
                },
            },
        )
        findings = {item["finding"] for item in graph.observations}
        self.assertIn("ohss_risk", findings)
        self.assertIn("ovarian_enlargement", findings)
        self.assertIn("ascites", findings)
        self.assertIn("hemoconcentration", findings)
        self.assertTrue(graph.symptoms)
        self.assertTrue(graph.imaging)
        self.assertTrue(graph.labs)
        self.assertTrue(graph.risk_factors)


if __name__ == "__main__":
    unittest.main()
