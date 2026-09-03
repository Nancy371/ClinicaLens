import unittest

import yaml

from agent.clinical_pattern_bridge import BRIDGE_REASON, CROSS_SYSTEM_SCOPE
from agent.clinical_reasoning_comparator import SWITCH_PRIMARY, UNLOCK_AND_DEFER
from agent.diagnosis_engine import CandidateScore, DiagnosisDecision, DiagnosisDecisionEngine


YAWS = "\u96c5\u53f8\u75c5"
ECZEMA = "\u6e7f\u75b9"
WATERPOX = "\u6c34\u75d8"
LEUKEMIA = "\u767d\u8840\u75c5"
PORTAL_HTN = "\u95e8\u9759\u8109\u9ad8\u538b"
TB = "\u80ba\u7ed3\u6838"
PNEUMONIA = "\u80ba\u708e"
BRONCHOPNEUMONIA = "\u652f\u6c14\u7ba1\u80ba\u708e"
LUNG_CANCER = "\u80ba\u764c"
MPA = "\u663e\u5fae\u955c\u4e0b\u591a\u8840\u7ba1\u708e"
URACHAL_CYST = "\u8110\u5c3f\u7ba1\u56ca\u80bf"
FRACTURE = "\u9aa8\u6298"
ZOSTER = "\u5e26\u72b6\u75b1\u75b9"
CONGENITAL_HEART = "\u5148\u5929\u6027\u5fc3\u810f\u75c5"
PULMONARY_STENOSIS = "\u80ba\u52a8\u8109\u74e3\u72ed\u7a84"
MITRAL_REGURGITATION = "\u4e8c\u5c16\u74e3\u53cd\u6d41"
HEART_FAILURE = "\u5fc3\u529b\u8870\u7aed"
OHSS = "\u5375\u5de2\u8fc7\u5ea6\u523a\u6fc0\u7efc\u5408\u5f81"
PANCREATITIS = "\u80f0\u817a\u708e"
URETHRAL_SYNDROME = "\u5c3f\u9053\u7efc\u5408\u5f81"
AV_BLOCK_2 = "\u4e8c\u5ea6\u623f\u5ba4\u4f20\u5bfc\u963b\u6ede"
ARRHYTHMIA = "\u5fc3\u5f8b\u5931\u5e38"
LOW_MAGNESIUM = "\u4f4e\u9541\u8840\u75c7"
RICKETS = "\u7ef4\u751f\u7d20D\u7f3a\u4e4f\u6027\u4f5d\u507b\u75c5"
REACTIVE_ARTHRITIS = "\u53cd\u5e94\u6027\u5173\u8282\u708e"
REITER = "\u8d56\u7279\u7efc\u5408\u5f81"
PAVM = "肺动静脉瘘"


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidate(
    name,
    score,
    *,
    required=True,
    diagnosis_type="disease",
    specificity=0.7,
    coverage=0.5,
    residual=0.4,
    core_coverage=None,
    residual_core=0,
    matched=None,
    gaps=None,
    parent="",
    core_score=0.0,
    diagnostic_score=0.0,
    generic_penalty=0.0,
):
    if core_coverage is None:
        core_coverage = coverage
    return CandidateScore(
        diagnosis=name,
        score=score,
        support_score=score,
        source_prior=0.5,
        explanation_score=coverage,
        coverage_score=coverage,
        residual_score=residual,
        explanatory_coverage=coverage,
        core_explanatory_coverage=core_coverage,
        residual_evidence_score=residual,
        residual_core_evidence_count=residual_core,
        explained_evidence=list(matched or ["symptom:signal"]),
        unexplained_core_evidence=[
            f"core_gap_{index + 1}" for index in range(max(0, int(residual_core)))
        ],
        contradiction_penalty=0.0,
        required_met=required,
        hard_contradiction=False,
        matched_evidence=list(matched or ["symptom:signal"]),
        core_matched_evidence=[
            item for item in list(matched or []) if item not in {"fever", "cough", "pain", "rash"}
        ],
        diagnostic_matched_evidence=[],
        core_evidence_score=core_score,
        diagnostic_evidence_score=diagnostic_score,
        generic_coverage_score=0.0,
        required_gaps=list(gaps or []),
        component_scores={
            "core_evidence_score": core_score,
            "diagnostic_evidence_score": diagnostic_score,
            "generic_parent_penalty": generic_penalty,
        },
        diagnosis_type=diagnosis_type,
        parent_diagnosis=parent,
        specificity=specificity,
    )


class DiagnosisJudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.engine = DiagnosisDecisionEngine(cls.config, "data/ref_data")

    def run_candidates(self, candidates):
        decision = DiagnosisDecision(
            final_diagnoses=[],
            trusted_diagnoses=[],
            candidates=candidates,
            unexplained_evidence=[],
            confidence=0.0,
            margin=0.0,
            low_confidence=False,
        )
        return self.engine.judge_and_submit(decision)

    def test_explanatory_power_beats_required_met_with_core_residual(self):
        yaws = candidate(
            YAWS,
            0.52,
            required=False,
            diagnosis_type="etiology",
            specificity=0.94,
            coverage=0.74,
            residual=0.18,
            core_coverage=0.78,
            residual_core=0,
            matched=[
                "rash",
                "treponemal_skin_lesion",
                "periostitis",
                "tropical_exposure",
            ],
            gaps=["treponema_positive"],
        )
        eczema = candidate(
            ECZEMA,
            0.62,
            required=True,
            specificity=0.48,
            coverage=0.30,
            residual=0.70,
            core_coverage=0.20,
            residual_core=3,
            matched=["rash", "pruritus"],
        )
        decision = self.run_candidates([eczema, yaws])
        payload = decision.judge_decision
        self.assertEqual(decision.final_diagnoses, [])
        self.assertEqual(payload["primary_status"], "deferred")
        self.assertIn(YAWS, payload["deferred_anchor_candidates"])
        self.assertIn(YAWS, payload["evidence_gap_targets"])
        self.assertEqual(decision.required_gap_authorized_diagnoses, [])
        self.assertEqual(payload["required_gap_state_by_candidate"][YAWS], "actionable_gap")
        self.assertEqual(payload["residual_core_evidence_count"], 0)

    def test_primary_lock_deferred_when_yaws_competes_with_waterpox(self):
        waterpox = candidate(
            WATERPOX,
            0.58,
            required=True,
            specificity=0.90,
            coverage=0.52,
            residual=0.38,
            matched=["vesicular_rash", "pruritus"],
        )
        yaws = candidate(
            YAWS,
            0.55,
            required=False,
            diagnosis_type="etiology",
            specificity=0.94,
            coverage=0.60,
            residual=0.30,
            matched=["treponemal_skin_lesion", "regional_lymphadenopathy"],
            gaps=["treponema_positive"],
        )
        leukemia = candidate(
            LEUKEMIA,
            0.44,
            required=True,
            specificity=0.78,
            coverage=0.30,
            residual=0.58,
            matched=["fever"],
        )
        decision = self.run_candidates([waterpox, yaws, leukemia])
        payload = decision.judge_decision
        self.assertEqual(payload["primary_status"], "deferred")
        self.assertTrue(payload["needs_discriminating_exams"])
        self.assertIn(YAWS, payload["differential_candidates"])
        self.assertIn("treponema_positive", payload["discriminating_findings"])
        self.assertIn(YAWS, payload["deferred_anchor_candidates"])
        self.assertIn(YAWS, payload["evidence_gap_targets"])
        self.assertEqual(decision.required_gap_authorized_diagnoses, [])

    def test_tb_is_not_locked_out_by_required_met_pneumonia(self):
        pneumonia = candidate(
            BRONCHOPNEUMONIA,
            0.58,
            required=True,
            specificity=0.72,
            coverage=0.46,
            residual=0.42,
            matched=["cough", "fever"],
        )
        tb = candidate(
            TB,
            0.54,
            required=False,
            diagnosis_type="etiology",
            specificity=0.92,
            coverage=0.68,
            residual=0.25,
            core_coverage=0.70,
            residual_core=0,
            matched=["hemoptysis", "night_sweats", "tuberculosis_exposure"],
            gaps=["afb_positive", "tb_naat_positive"],
        )
        lung_cancer = candidate(
            LUNG_CANCER,
            0.48,
            required=True,
            specificity=0.82,
            coverage=0.46,
            residual=0.44,
            matched=["hemoptysis"],
        )
        decision = self.run_candidates([pneumonia, tb, lung_cancer])
        payload = decision.judge_decision
        self.assertNotIn(TB, decision.final_diagnoses)
        self.assertIn(TB, payload["differential_candidates"])
        self.assertIn(TB, payload["deferred_anchor_candidates"])
        self.assertIn(TB, payload["evidence_gap_targets"])
        self.assertEqual(decision.required_gap_authorized_diagnoses, [])

    def test_urachal_cyst_core_evidence_beats_urethral_syndrome(self):
        urethral = candidate(
            URETHRAL_SYNDROME,
            0.64,
            required=True,
            diagnosis_type="syndrome",
            specificity=0.58,
            coverage=0.38,
            residual=0.62,
            core_coverage=0.12,
            residual_core=3,
            matched=["dysuria", "urinary_frequency"],
            generic_penalty=0.8,
        )
        urachal = candidate(
            URACHAL_CYST,
            0.50,
            required=False,
            diagnosis_type="structural",
            specificity=0.93,
            coverage=0.66,
            residual=0.18,
            core_coverage=0.78,
            residual_core=0,
            matched=[
                "umbilical_discharge",
                "midline_suprapubic_pain",
                "urachal_remnant_pattern",
            ],
            gaps=["urachal_cyst_imaging"],
            core_score=0.86,
        )
        decision = self.run_candidates([urethral, urachal])
        self.assertEqual(decision.final_diagnoses, [])
        self.assertEqual(decision.judge_decision["primary_status"], "deferred")
        self.assertIn(URACHAL_CYST, decision.judge_decision["deferred_anchor_candidates"])
        self.assertIn(URACHAL_CYST, decision.judge_decision["evidence_gap_targets"])
        self.assertEqual(decision.required_gap_authorized_diagnoses, [])

    def test_av_block_core_evidence_beats_arrhythmia_parent(self):
        arrhythmia = candidate(
            ARRHYTHMIA,
            0.68,
            required=True,
            specificity=0.50,
            coverage=0.40,
            residual=0.55,
            core_coverage=0.18,
            residual_core=2,
            matched=["palpitation", "dizziness"],
            generic_penalty=0.75,
        )
        av_block = candidate(
            AV_BLOCK_2,
            0.52,
            required=True,
            diagnosis_type="structural",
            specificity=0.94,
            coverage=0.70,
            residual=0.16,
            core_coverage=0.82,
            residual_core=0,
            matched=["second_degree_av_block", "bradycardia", "presyncope"],
            parent=ARRHYTHMIA,
            core_score=0.74,
            diagnostic_score=0.52,
        )
        decision = self.run_candidates([arrhythmia, av_block])
        self.assertEqual(decision.final_diagnoses[0], AV_BLOCK_2)
        self.assertNotIn(ARRHYTHMIA, decision.final_diagnoses)

    def test_tb_mpa_lung_cancer_tasks_prioritize_special_discriminators(self):
        lung_cancer = candidate(
            LUNG_CANCER,
            0.62,
            required=True,
            specificity=0.86,
            coverage=0.50,
            residual=0.36,
            matched=["hemoptysis"],
        )
        mpa = candidate(
            MPA,
            0.59,
            required=False,
            diagnosis_type="systemic",
            specificity=0.92,
            coverage=0.62,
            residual=0.26,
            core_coverage=0.64,
            residual_core=1,
            matched=["hemoptysis", "microscopic_hematuria"],
            gaps=["anca_positive", "renal_impairment"],
        )
        tb = candidate(
            TB,
            0.56,
            required=False,
            diagnosis_type="etiology",
            specificity=0.92,
            coverage=0.68,
            residual=0.22,
            core_coverage=0.70,
            residual_core=0,
            matched=["hemoptysis", "night_sweats", "tuberculosis_exposure"],
            gaps=["afb_positive", "tb_naat_positive"],
        )
        decision = self.run_candidates([lung_cancer, mpa, tb])
        payload = decision.judge_decision
        tasks = payload["discriminating_exam_tasks"]
        exams = [item["exam"] for item in tasks]
        self.assertLessEqual(len(exams), 6)
        self.assertLess(
            exams.index("抗酸杆菌染色（AFB）"),
            exams.index("全血细胞计数（CBC）") if "全血细胞计数（CBC）" in exams else len(exams),
        )
        self.assertIn("抗中性粒细胞胞质抗体（ANCA）谱", exams)
        self.assertTrue(
            all(item["target_candidates"] for item in tasks)
        )
        self.assertGreaterEqual(
            sum(1 for item in tasks if len(item["target_candidates"]) >= 2),
            4,
        )

    def test_differential_pool_filters_cross_system_noise_from_yaws(self):
        waterpox = candidate(WATERPOX, 0.60, required=True, specificity=0.90, matched=["vesicular_rash"])
        yaws = candidate(
            YAWS,
            0.57,
            required=False,
            diagnosis_type="etiology",
            specificity=0.94,
            coverage=0.62,
            residual=0.28,
            matched=["treponemal_skin_lesion", "regional_lymphadenopathy"],
            gaps=["treponema_positive"],
        )
        leukemia = candidate(LEUKEMIA, 0.52, required=True, matched=["fever", "bone_pain"])
        portal = candidate(PORTAL_HTN, 0.55, required=True, diagnosis_type="systemic", matched=["portal_flow_abnormal"])
        decision = self.run_candidates([waterpox, yaws, portal, leukemia])
        payload = decision.judge_decision
        self.assertIn(YAWS, payload["differential_candidates"])
        self.assertIn(LEUKEMIA, payload["differential_candidates"])
        self.assertNotIn(PORTAL_HTN, payload["differential_candidates"])

    def test_top20_tail_high_specificity_candidate_enters_differential_pool(self):
        head = [
            candidate(
                f"generic_{i}",
                0.70 - i * 0.01,
                required=True,
                specificity=0.55,
                coverage=0.42,
                residual=0.44,
                matched=[f"symptom:generic{i}"],
            )
            for i in range(6)
        ]
        urachal = candidate(
            URACHAL_CYST,
            0.24,
            required=False,
            diagnosis_type="structural",
            specificity=0.93,
            coverage=0.28,
            residual=0.50,
            matched=["umbilical_discharge", "midline_suprapubic_cyst"],
            gaps=["urachal_cyst_imaging"],
        )
        decision = self.run_candidates(head + [urachal])
        payload = decision.judge_decision
        self.assertIn(URACHAL_CYST, payload["differential_candidates"])
        self.assertEqual(payload["differential_pool_source"].get(URACHAL_CYST), "top20_priority_tail")
        self.assertIn(URACHAL_CYST, payload["required_gap_state_by_candidate"])

    def test_pattern_deferred_tail_candidate_forces_workup_pool(self):
        head = [
            candidate(
                f"generic_{i}",
                0.78 - i * 0.01,
                required=True,
                specificity=0.55,
                coverage=0.42,
                residual=0.44,
                matched=[f"symptom:generic{i}"],
            )
            for i in range(7)
        ]
        for item in head:
            item.eligibility_status = "PrimaryEligible"
            item.eligibility_reason = "AnchorsSatisfied"

        pavm = candidate(
            PAVM,
            0.20,
            required=False,
            diagnosis_type="structural",
            specificity=0.92,
            coverage=0.34,
            residual=0.48,
            core_coverage=0.32,
            matched=["hemoptysis", "right_to_left_shunt"],
            gaps=[
                "pulmonary_avm_confirmed_vascular_pattern:pulmonary_cta_positive|enhanced_ct_vascular_malformation|bubble_echo_right_to_left_shunt"
            ],
            core_score=0.58,
            diagnostic_score=0.35,
        )
        pavm.eligibility_status = "Deferred"
        pavm.eligibility_reason = "NeedsAnchor"
        pavm.evidence_pattern_matches = [
            {
                "pattern_id": "pulmonary_avm_initial_shunt_pattern",
                "pattern_type": "anchor_pattern",
                "matched": True,
                "matched_required_groups": [
                    {"matched_findings": ["hemoptysis"]},
                    {"matched_findings": ["right_to_left_shunt"]},
                ],
                "missing_required_groups": [],
                "effect": {"eligibility": "Deferred", "reason": "NeedsAnchor"},
            }
        ]

        decision = self.run_candidates(head + [pavm])
        payload = decision.judge_decision

        self.assertIn(PAVM, payload["differential_candidates"])
        self.assertEqual(payload["differential_pool_source"].get(PAVM), "pattern_deferred_workup")
        self.assertIn(PAVM, payload["evidence_gap_targets"])
        tasks = payload["discriminating_exam_tasks"]
        self.assertTrue(
            any(
                PAVM in task.get("target_candidates", [])
                and task.get("exam") in {"肺动脉CTA", "胸部增强CT", "超声心动图右心声学造影"}
                for task in tasks
            )
        )

    def test_bridge_protected_candidate_survives_cross_system_pool_filter(self):
        leukemia = candidate(
            LEUKEMIA,
            0.74,
            required=True,
            matched=["fever", "anemia", "platelet_low"],
            coverage=0.50,
            residual=0.36,
        )
        reactive = candidate(
            REACTIVE_ARTHRITIS,
            0.31,
            required=True,
            diagnosis_type="disease",
            matched=["arthralgia", "dysuria", "conjunctivitis"],
            coverage=0.42,
            residual=0.44,
        )
        reactive.bridge_protection_decisions = [
            {
                "candidate_id": "D100057",
                "source_assertion_id": "DPA-test-reactive-arthritis",
                "protection_scope": [CROSS_SYSTEM_SCOPE],
                "protection_status": "active",
                "reason_code": BRIDGE_REASON,
                "strength": "strong",
            }
        ]

        decision = self.run_candidates([leukemia, reactive])
        payload = decision.judge_decision

        self.assertIn(REACTIVE_ARTHRITIS, payload["differential_candidates"])
        self.assertEqual(
            payload["pool_filter_reasons"].get(REACTIVE_ARTHRITIS),
            BRIDGE_REASON,
        )
        self.assertFalse(
            any(
                item.get("diagnosis") == REACTIVE_ARTHRITIS
                and item.get("reason") == "cross_system_no_shared_core_evidence"
                for item in payload["excluded_from_pairwise"]
            )
        )
        self.assertTrue(
            any(
                item.get("allowed")
                and item.get("reason") == BRIDGE_REASON
                and REACTIVE_ARTHRITIS in {item.get("left"), item.get("right")}
                for item in payload["pairwise_allowed_matrix"]
            )
        )

    def test_protected_recall_candidate_gets_arbitration_consideration(self):
        primary = candidate(
            FRACTURE,
            0.78,
            required=True,
            matched=["osteophyte", "activity_related_joint_pain"],
            coverage=0.32,
            residual=0.68,
            residual_core=3,
        )
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"

        radiation = candidate(
            "\u653e\u5c04\u6027\u80ba\u708e",
            0.31,
            required=False,
            matched=[
                "thoracic_radiotherapy",
                "ground_glass_opacity",
                "pulmonary_consolidation",
                "lesion_within_prior_radiation_field",
            ],
            gaps=["post_radiotherapy_time_window"],
            coverage=0.74,
            residual=0.22,
            core_coverage=0.78,
            residual_core=0,
            core_score=0.66,
            diagnostic_score=0.42,
        )
        radiation.entity_id = "D100058"
        radiation.eligibility_status = "Deferred"
        radiation.eligibility_anchor_status = "PatternSupportedButUnconfirmed"
        radiation.candidate_sources = [
            {
                "source": "llm_pattern_hypothesis",
                "entity_id": "D100058",
                "metadata": {
                    "recall_mode": "protected_recall",
                    "protected_pool_slot": True,
                    "pattern_hypothesis_id": "PH_DET_radiation",
                    "pattern_recall_only": True,
                    "judge_evidence_weight": 0.0,
                    "eligibility_evidence_weight": 0.0,
                },
            }
        ]

        decision = self.run_candidates([primary, radiation])
        payload = decision.judge_decision

        self.assertIn("\u653e\u5c04\u6027\u80ba\u708e", payload["differential_candidates"])
        self.assertEqual(
            payload["pool_filter_reasons"].get("\u653e\u5c04\u6027\u80ba\u708e"),
            "protected_recall_arbitration",
        )
        self.assertTrue(payload["primary_arbitration_candidates"])
        self.assertEqual(
            payload["primary_arbitration_candidates"][0]["entered_by"],
            "protected_recall",
        )

    def test_anchor_primary_unlocks_when_high_value_residual_favors_protected_contender(self):
        primary = candidate(
            FRACTURE,
            0.86,
            required=True,
            matched=["osteophyte", "activity_related_joint_pain"],
            coverage=0.28,
            residual=0.72,
            residual_core=4,
        )
        primary.entity_id = "D100031"
        primary.body_system = "musculoskeletal"
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"
        primary.residual_evidence = [
            "thoracic_radiotherapy",
            "ground_glass_opacity",
            "pulmonary_consolidation",
            "lesion_within_prior_radiation_field",
        ]

        radiation = candidate(
            "\u653e\u5c04\u6027\u80ba\u708e",
            0.35,
            required=False,
            matched=[
                "thoracic_radiotherapy",
                "ground_glass_opacity",
                "pulmonary_consolidation",
                "lesion_within_prior_radiation_field",
            ],
            gaps=["radiation_field_lung_consistency"],
            coverage=0.80,
            residual=0.16,
            core_coverage=0.82,
            residual_core=0,
        )
        radiation.entity_id = "D100058"
        radiation.body_system = "respiratory"
        radiation.eligibility_status = "Deferred"
        radiation.eligibility_anchor_status = "PatternSupportedButUnconfirmed"
        radiation.candidate_sources = [
            {
                "source": "llm_pattern_hypothesis",
                "entity_id": "D100058",
                "metadata": {
                    "body_system": "respiratory",
                    "recall_mode": "protected_recall",
                    "protected_pool_slot": True,
                    "pattern_hypothesis_id": "PH_DET_radiation",
                    "pattern_recall_only": True,
                    "judge_evidence_weight": 0.0,
                    "eligibility_evidence_weight": 0.0,
                },
            }
        ]

        result = self.engine.judge._primary_arbitration(
            primary,
            [primary, radiation],
            [{"left": FRACTURE, "right": "\u653e\u5c04\u6027\u80ba\u708e"}],
        )

        self.assertIs(result["selected_candidate"], radiation)
        self.assertEqual(result["decision"]["action"], UNLOCK_AND_DEFER)
        reasons = set(result["decision"]["reason_codes"])
        self.assertIn("INCUMBENT_PRIMARY_PROTECTION_LOST", reasons)
        self.assertIn("NEW_MATERIAL_EVIDENCE_UNEXPLAINED", reasons)
        self.assertIn("PRIMARY_UNLOCKED_CONTENDER_DEFERRED", reasons)
        comparison = result["comparisons"][0]["clinical_explanatory_comparison"]
        incumbent = comparison["incumbent_profile"]
        self.assertEqual(incumbent["primary_protection_status"], "LOST")
        self.assertTrue(incumbent["primary_explanatory_mismatch"])
        self.assertEqual(primary.eligibility_anchor_status, "AnchorSatisfied")

    def test_comorbid_background_residual_does_not_challenge_good_primary(self):
        primary = candidate(
            PNEUMONIA,
            0.82,
            required=True,
            matched=["dyspnea", "ground_glass_opacity", "pulmonary_consolidation"],
            coverage=0.82,
            residual=0.12,
            core_coverage=0.86,
        )
        primary.body_system = "respiratory"
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"
        primary.residual_evidence = ["hypertension"]

        contender = candidate(
            HEART_FAILURE,
            0.48,
            required=False,
            matched=["dyspnea"],
            coverage=0.30,
            residual=0.66,
            core_coverage=0.25,
        )
        contender.body_system = "cardiovascular"
        contender.eligibility_status = "Deferred"
        contender.eligibility_anchor_status = "PatternSupportedButUnconfirmed"
        contender.candidate_sources = [
            {
                "source": "llm_pattern_hypothesis",
                "metadata": {
                    "recall_mode": "protected_recall",
                    "protected_pool_slot": True,
                    "pattern_hypothesis_id": "PH_background_control",
                },
            }
        ]

        result = self.engine.judge._primary_arbitration(
            primary,
            [primary, contender],
            [{"left": PNEUMONIA, "right": HEART_FAILURE}],
        )

        self.assertIs(result["selected_candidate"], primary)
        self.assertNotEqual(result["decision"]["action"], UNLOCK_AND_DEFER)
        incumbent = result["comparisons"][0]["clinical_explanatory_comparison"][
            "incumbent_profile"
        ]
        self.assertEqual(incumbent["primary_protection_status"], "PROTECTED")
        self.assertFalse(incumbent["high_value_residuals"])

    def test_cross_system_primary_not_downgraded_when_it_explains_core_case(self):
        primary = candidate(
            MPA,
            0.79,
            required=True,
            matched=[
                "dyspnea",
                "hematuria",
                "pulmonary_infiltrate",
                "renal_involvement",
            ],
            coverage=0.78,
            residual=0.18,
            core_coverage=0.80,
        )
        primary.body_system = "systemic_vasculitis"
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"

        contender = candidate(
            PNEUMONIA,
            0.58,
            required=False,
            matched=["dyspnea", "pulmonary_infiltrate"],
            coverage=0.45,
            residual=0.42,
            core_coverage=0.44,
        )
        contender.body_system = "respiratory"
        contender.eligibility_status = "Deferred"
        contender.eligibility_anchor_status = "PatternSupportedButUnconfirmed"
        contender.candidate_sources = [
            {
                "source": "llm_pattern_hypothesis",
                "metadata": {
                    "recall_mode": "protected_recall",
                    "protected_pool_slot": True,
                    "pattern_hypothesis_id": "PH_cross_system_control",
                },
            }
        ]

        result = self.engine.judge._primary_arbitration(
            primary,
            [primary, contender],
            [{"left": MPA, "right": PNEUMONIA}],
        )

        self.assertIs(result["selected_candidate"], primary)
        incumbent = result["comparisons"][0]["clinical_explanatory_comparison"][
            "incumbent_profile"
        ]
        self.assertEqual(incumbent["primary_protection_status"], "PROTECTED")
        self.assertNotIn(
            "PRIMARY_EXPLANATORY_MISMATCH",
            incumbent["profile_reason_codes"],
        )

    def test_protected_recall_signal_does_not_count_as_core_coverage(self):
        primary = candidate(
            PNEUMONIA,
            0.80,
            required=True,
            matched=["dyspnea", "pulmonary_infiltrate"],
            coverage=0.72,
            residual=0.20,
            core_coverage=0.75,
        )
        primary.body_system = "respiratory"
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"

        contender = candidate(
            "\u653e\u5c04\u6027\u80ba\u708e",
            0.34,
            required=False,
            matched=[],
            coverage=0.0,
            residual=0.80,
            core_coverage=0.0,
        )
        contender.entity_id = "D100058"
        contender.eligibility_status = "Deferred"
        contender.eligibility_anchor_status = "PatternSupportedButUnconfirmed"
        contender.candidate_sources = [
            {
                "source": "llm_pattern_hypothesis",
                "metadata": {
                    "recall_mode": "protected_recall",
                    "protected_pool_slot": True,
                    "pattern_hypothesis_id": "PH_DET_only_pattern",
                },
            }
        ]

        record = self.engine.judge.clinical_comparator.compare(primary, contender)

        contender_profile = record["clinical_explanatory_comparison"][
            "contender_profile"
        ]
        self.assertEqual(contender_profile["core_case_coverage"], 0.0)
        self.assertEqual(record["recommended_action"], "KEEP_CURRENT_AND_DEFER_CONTENDER")

    def test_primary_eligible_mr_contender_not_filtered_by_incumbent_anchor(self):
        primary = candidate(
            "\u7ec8\u672b\u671f\u80be\u75c5",
            0.78,
            required=True,
            diagnosis_type="state",
            coverage=0.52,
            residual=0.34,
            core_coverage=0.50,
            matched=["renal_failure", "edema"],
            core_score=0.38,
        )
        primary.entity_id = "D_ESRD"
        primary.body_system = "renal"
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"

        mitral = candidate(
            MITRAL_REGURGITATION,
            0.72,
            required=True,
            diagnosis_type="structural",
            specificity=0.90,
            coverage=0.74,
            residual=0.18,
            core_coverage=0.76,
            matched=[
                "mitral_regurgitation",
                "orthopnea",
                "pulmonary_edema",
            ],
            core_score=0.54,
            diagnostic_score=0.40,
        )
        mitral.entity_id = "D100012"
        mitral.body_system = "cardiovascular"
        mitral.eligibility_status = "PrimaryEligible"
        mitral.eligibility_anchor_status = "AnchorSatisfied"
        mitral.core_matched_evidence = ["orthopnea", "pulmonary_edema"]
        mitral.diagnostic_matched_evidence = ["mitral_regurgitation"]

        result = self.engine.judge._primary_arbitration(
            primary,
            [primary, mitral],
            [{"left": "\u7ec8\u672b\u671f\u80be\u75c5", "right": MITRAL_REGURGITATION}],
        )

        self.assertTrue(result["comparisons"])
        self.assertEqual(
            result["comparisons"][0]["candidate_b"],
            MITRAL_REGURGITATION,
        )
        audit = next(
            item
            for item in result["material_contender_filter"]
            if item["entity_id"] == "D100012"
        )
        self.assertTrue(audit["pairwise_allowed"])
        self.assertEqual(audit["candidate_anchor_status"], "AnchorSatisfied")
        self.assertEqual(audit["current_primary_anchor_status"], "AnchorSatisfied")
        self.assertTrue(audit["required_met"])
        self.assertTrue(audit["has_core_or_diagnostic_evidence"])
        self.assertTrue(audit["material_contender"])
        self.assertEqual(audit["filtered_reason"], "")
        dispositions = {
            item["entity_id"]: item for item in result["candidate_disposition_audit"]
        }
        self.assertTrue(dispositions["D100012"]["comparison_present"])
        self.assertNotEqual(
            result["decision"]["reason_codes"],
            ["NO_MATERIAL_ARBITRATION_CONTENDER"],
        )

    def test_topk_primary_eligible_candidate_without_disposition_is_deadlock(self):
        primary = candidate(
            PNEUMONIA,
            0.82,
            required=True,
            matched=["cough", "pulmonary_infiltrate"],
        )
        primary.entity_id = "D_PRIMARY"
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"

        mitral = candidate(
            MITRAL_REGURGITATION,
            0.70,
            required=True,
            matched=["mitral_regurgitation"],
            core_score=0.45,
            diagnostic_score=0.40,
        )
        mitral.entity_id = "D100012"
        mitral.eligibility_status = "PrimaryEligible"
        mitral.eligibility_anchor_status = "AnchorSatisfied"
        mitral.required_gaps = []
        mitral.actionable_gap_count = 0

        dispositions = self.engine.judge._candidate_disposition_audit(
            primary,
            [primary, mitral],
            [],
            {},
            {},
        )

        mitral_disposition = next(
            item for item in dispositions if item["entity_id"] == "D100012"
        )
        self.assertEqual(
            mitral_disposition["deadlock_code"],
            "ARBITRATION_DEADLOCK",
        )
        self.assertEqual(mitral_disposition["failure_stage"], "candidate_routing")
        self.assertEqual(
            mitral_disposition["lifecycle_state"],
            "READY_FOR_ARBITRATION",
        )
        self.assertIn(
            "PRIMARY_ELIGIBLE_NOT_IN_ARBITRATION_POOL",
            mitral_disposition["deadlock_codes"],
        )

    def test_primary_eligible_outside_differential_pool_enters_arbitration_pool(self):
        primary = candidate(
            FRACTURE,
            0.86,
            required=True,
            matched=["osteophyte", "joint_pain"],
            coverage=0.40,
            residual=0.55,
            residual_core=2,
        )
        primary.entity_id = "D100031"
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"

        distractors = []
        for index in range(6):
            item = candidate(
                f"候选{index}",
                0.80 - index * 0.02,
                required=False,
                matched=[f"generic_{index}"],
            )
            item.entity_id = f"D_NOISE_{index}"
            item.eligibility_status = "DifferentialOnly"
            item.differential_only = True
            distractors.append(item)

        radiation = candidate(
            "放射性肺炎",
            0.40,
            required=True,
            matched=[
                "thoracic_radiotherapy",
                "ground_glass_opacity",
                "lesion_within_prior_radiation_field",
            ],
            core_score=0.34,
            diagnostic_score=0.20,
            coverage=0.82,
            residual=0.12,
            core_coverage=0.84,
        )
        radiation.entity_id = "D100058"
        radiation.eligibility_status = "PrimaryEligible"
        radiation.eligibility_anchor_status = "AnchorSatisfied"

        result = self.engine.judge._primary_arbitration(
            primary,
            [primary] + distractors[:2],
            [],
            full_pool=[primary] + distractors + [radiation],
        )

        dispositions = {
            item["entity_id"]: item for item in result["candidate_disposition_audit"]
        }
        radiation_disposition = dispositions["D100058"]
        self.assertEqual(
            radiation_disposition["lifecycle_state"],
            "READY_FOR_ARBITRATION",
        )
        self.assertTrue(radiation_disposition["arbitration_pool_member"])
        self.assertEqual(
            radiation_disposition["arbitration_admission_reason"],
            "PRIMARY_ELIGIBLE",
        )
        self.assertTrue(radiation_disposition["comparison_present"])
        self.assertEqual(radiation_disposition["invariant_status"], "VALID")
        self.assertFalse(radiation_disposition["deadlock_codes"])

    def test_arbitration_pool_member_without_resolution_is_deadlock(self):
        primary = candidate(PNEUMONIA, 0.80, required=True)
        primary.entity_id = "D_PRIMARY"
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"

        contender = candidate(
            MITRAL_REGURGITATION,
            0.70,
            required=True,
            matched=["mitral_regurgitation"],
            core_score=0.35,
        )
        contender.entity_id = "D100012"
        contender.eligibility_status = "PrimaryEligible"
        contender.eligibility_anchor_status = "AnchorSatisfied"

        dispositions = self.engine.judge._candidate_disposition_audit(
            primary,
            [primary, contender],
            [
                {
                    "candidate": MITRAL_REGURGITATION,
                    "entity_id": "D100012",
                    "eligibility_status": "PrimaryEligible",
                    "candidate_anchor_status": "AnchorSatisfied",
                    "arbitration_pool_member": True,
                    "arbitration_admission_reason": "PRIMARY_ELIGIBLE",
                    "material_contender": True,
                }
            ],
            {},
            {},
            arbitration_pool_names={MITRAL_REGURGITATION},
            arbitration_admission_reason_by_name={
                MITRAL_REGURGITATION: "PRIMARY_ELIGIBLE"
            },
        )

        contender_disposition = next(
            item for item in dispositions if item["entity_id"] == "D100012"
        )
        self.assertIn(
            "ARBITRATION_MEMBER_NOT_RESOLVED",
            contender_disposition["deadlock_codes"],
        )
        self.assertEqual(
            contender_disposition["failure_stage"],
            "arbitration_resolution",
        )

    def test_disposition_scoped_by_diagnostic_state_version(self):
        primary = candidate(PNEUMONIA, 0.80, required=True)
        primary.entity_id = "D_PRIMARY"
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"

        contender = candidate(MITRAL_REGURGITATION, 0.70, required=False)
        contender.entity_id = "D100012"
        contender.eligibility_status = "Deferred"
        contender.eligibility_anchor_status = "PatternSupportedButUnconfirmed"
        contender.required_gaps = ["mitral_regurgitant_jet"]
        contender.diagnostic_state_version = 1
        v1 = self.engine.judge._candidate_disposition_audit(
            primary,
            [primary, contender],
            [],
            {},
            {},
        )
        contender.eligibility_status = "PrimaryEligible"
        contender.eligibility_anchor_status = "AnchorSatisfied"
        contender.required_gaps = []
        contender.required_met = True
        contender.diagnostic_state_version = 2
        v2 = self.engine.judge._candidate_disposition_audit(
            primary,
            [primary, contender],
            [
                {
                    "candidate": MITRAL_REGURGITATION,
                    "entity_id": "D100012",
                    "eligibility_status": "PrimaryEligible",
                    "candidate_anchor_status": "AnchorSatisfied",
                    "arbitration_pool_member": True,
                    "arbitration_admission_reason": "PRIMARY_ELIGIBLE",
                    "material_contender": True,
                }
            ],
            {MITRAL_REGURGITATION: "KEEP_CURRENT_PRIMARY"},
            {MITRAL_REGURGITATION: ["INCUMBENT_PREFERRED"]},
            arbitration_pool_names={MITRAL_REGURGITATION},
        )

        first = next(item for item in v1 if item["entity_id"] == "D100012")
        second = next(item for item in v2 if item["entity_id"] == "D100012")
        self.assertEqual(first["source_state_version"], 1)
        self.assertEqual(first["lifecycle_state"], "WORKUP_REQUIRED")
        self.assertEqual(second["source_state_version"], 2)
        self.assertEqual(second["lifecycle_state"], "READY_FOR_ARBITRATION")
        self.assertEqual(second["invariant_status"], "VALID")

    def test_deferred_with_pending_workup_without_new_gap_is_valid(self):
        primary = candidate(PNEUMONIA, 0.80, required=True)
        primary.entity_id = "D_PRIMARY"
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"

        contender = candidate("待确认疾病", 0.55, required=False)
        contender.entity_id = "D_PENDING"
        contender.eligibility_status = "Deferred"
        contender.eligibility_anchor_status = "PatternSupportedButUnconfirmed"
        contender.required_gaps = []
        contender.pending_exam_result = True

        dispositions = self.engine.judge._candidate_disposition_audit(
            primary,
            [primary, contender],
            [],
            {},
            {},
        )
        record = next(item for item in dispositions if item["entity_id"] == "D_PENDING")
        self.assertEqual(record["lifecycle_state"], "WORKUP_REQUIRED")
        self.assertEqual(record["invariant_status"], "VALID")
        self.assertNotIn("GAPLESS_DEFERRED_CANDIDATE", record["deadlock_codes"])

    def test_differential_only_remains_alive_without_deadlock(self):
        primary = candidate(PNEUMONIA, 0.80, required=True)
        primary.entity_id = "D_PRIMARY"
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"

        contender = candidate("低优先鉴别", 0.42, required=False)
        contender.entity_id = "D_DIFF"
        contender.eligibility_status = "DifferentialOnly"
        contender.differential_only = True

        dispositions = self.engine.judge._candidate_disposition_audit(
            primary,
            [primary, contender],
            [],
            {},
            {},
        )
        record = next(item for item in dispositions if item["entity_id"] == "D_DIFF")
        self.assertEqual(record["lifecycle_state"], "DIFFERENTIAL_ONLY")
        self.assertEqual(record["required_action"], "MONITOR")
        self.assertEqual(record["invariant_status"], "VALID")

    def test_lifecycle_recovery_does_not_mutate_score_or_state_versions(self):
        primary = candidate(PNEUMONIA, 0.80, required=True)
        primary.entity_id = "D_PRIMARY"
        primary.eligibility_status = "PrimaryEligible"
        primary.eligibility_anchor_status = "AnchorSatisfied"

        contender = candidate(
            MITRAL_REGURGITATION,
            0.61,
            required=True,
            matched=["mitral_regurgitation"],
            core_score=0.28,
        )
        contender.entity_id = "D100012"
        contender.eligibility_status = "PrimaryEligible"
        contender.eligibility_anchor_status = "AnchorSatisfied"
        contender.evidence_version = 7
        contender.claim_state_version = 3
        score_before = contender.score
        evidence_version_before = contender.evidence_version
        claim_state_version_before = contender.claim_state_version

        original = self.engine.judge.clinical_comparator.material_contender
        self.engine.judge.clinical_comparator.material_contender = lambda *_args, **_kwargs: False
        try:
            result = self.engine.judge._primary_arbitration(
                primary,
                [primary],
                [],
                full_pool=[primary, contender],
            )
        finally:
            self.engine.judge.clinical_comparator.material_contender = original

        self.assertEqual(contender.score, score_before)
        self.assertEqual(contender.evidence_version, evidence_version_before)
        self.assertEqual(contender.claim_state_version, claim_state_version_before)
        self.assertTrue(result["lifecycle_recoveries"])
        recovery = result["lifecycle_recoveries"][0]
        self.assertTrue(recovery["score_unchanged"])
        self.assertTrue(recovery["evidence_version_unchanged"])
        self.assertTrue(recovery["claim_state_unchanged"])

    def test_primary_arbitration_switches_when_score_primary_has_no_anchor(self):
        zoster = candidate(
            ZOSTER,
            0.91,
            required=True,
            matched=["pain", "ocular_redness"],
            coverage=0.46,
            residual=0.58,
            residual_core=3,
        )
        zoster.eligibility_status = "PrimaryEligible"
        zoster.eligibility_anchor_status = "NoValidAnchor"
        zoster.residual_evidence = ["arthralgia", "dysuria", "conjunctivitis"]

        reiter = candidate(
            REITER,
            0.52,
            required=True,
            diagnosis_type="disease",
            matched=["arthralgia", "dysuria", "conjunctivitis"],
            coverage=0.72,
            residual=0.18,
            core_coverage=0.76,
            residual_core=0,
            core_score=0.66,
            diagnostic_score=0.48,
        )
        reiter.entity_id = "D100057"
        reiter.eligibility_status = "PrimaryEligible"
        reiter.eligibility_anchor_status = "AnchorSatisfied"
        reiter.evidence_pattern_matches = [
            {
                "pattern_id": "reiter_triads_anchor_pattern",
                "pattern_type": "anchor_pattern",
                "matched": True,
                "effect": {"eligibility": "PrimaryEligible"},
            }
        ]
        reiter.clinical_pattern_matches = [
            {
                "pattern_id": "postinfectious_arthritis_uroocular_pattern",
                "verification_status": "verified",
                "supporting_findings": ["arthralgia", "dysuria", "conjunctivitis"],
            }
        ]
        reiter.derived_pattern_assertions = [
            {
                "assertion_id": "DPA-test-reiter",
                "canonical_pattern": "reactive_arthritis_bridge_pattern",
            }
        ]
        reiter.bridge_protection_decisions = [
            {
                "candidate_id": "D100057",
                "source_assertion_id": "DPA-test-reiter",
                "protection_scope": [CROSS_SYSTEM_SCOPE],
                "protection_status": "active",
                "reason_code": BRIDGE_REASON,
                "strength": "strong",
            }
        ]

        result = self.engine.judge._primary_arbitration(
            zoster,
            [zoster, reiter],
            [{"left": ZOSTER, "right": REITER}],
        )

        self.assertIs(result["selected_candidate"], reiter)
        self.assertEqual(result["decision"]["action"], SWITCH_PRIMARY)
        self.assertIn("CURRENT_PRIMARY_HAS_NO_VALID_ANCHOR", result["decision"]["reason_codes"])

    def test_primary_arbitration_unlocks_when_bridge_contender_needs_gap(self):
        zoster = candidate(
            ZOSTER,
            0.91,
            required=True,
            matched=["pain", "ocular_redness"],
            coverage=0.46,
            residual=0.58,
            residual_core=3,
        )
        zoster.eligibility_status = "PrimaryEligible"
        zoster.eligibility_anchor_status = "NoValidAnchor"
        zoster.residual_evidence = ["arthralgia", "dysuria", "conjunctivitis"]

        reiter = candidate(
            REITER,
            0.50,
            required=False,
            diagnosis_type="disease",
            matched=["arthralgia", "dysuria", "conjunctivitis"],
            gaps=["preceding_genitourinary_infection"],
            coverage=0.70,
            residual=0.20,
            core_coverage=0.72,
            residual_core=0,
        )
        reiter.entity_id = "D100057"
        reiter.eligibility_status = "Deferred"
        reiter.eligibility_anchor_status = "PatternSupportedButUnconfirmed"
        reiter.clinical_pattern_matches = [
            {
                "pattern_id": "postinfectious_arthritis_uroocular_pattern",
                "verification_status": "verified",
                "supporting_findings": ["arthralgia", "dysuria", "conjunctivitis"],
            }
        ]
        reiter.derived_pattern_assertions = [
            {
                "assertion_id": "DPA-test-reiter-deferred",
                "canonical_pattern": "reactive_arthritis_bridge_pattern",
            }
        ]
        reiter.bridge_protection_decisions = [
            {
                "candidate_id": "D100057",
                "source_assertion_id": "DPA-test-reiter-deferred",
                "protection_scope": [CROSS_SYSTEM_SCOPE],
                "protection_status": "active",
                "reason_code": BRIDGE_REASON,
                "strength": "strong",
            }
        ]

        result = self.engine.judge._primary_arbitration(
            zoster,
            [zoster, reiter],
            [{"left": ZOSTER, "right": REITER}],
        )

        self.assertIs(result["selected_candidate"], reiter)
        self.assertEqual(result["decision"]["action"], UNLOCK_AND_DEFER)
        self.assertTrue(result["defer_reason"])
        self.assertEqual(result["pairwise_discriminating_gaps"][0]["gap_type"], "pairwise_discrimination")

    def test_bridge_protection_does_not_override_hard_blocker(self):
        primary = candidate(LEUKEMIA, 0.72, required=True, matched=["blast_present"])
        reactive = candidate(
            REACTIVE_ARTHRITIS,
            0.44,
            required=True,
            matched=["arthralgia", "dysuria", "conjunctivitis"],
        )
        reactive.hard_contradiction = True
        reactive.bridge_protection_decisions = [
            {
                "candidate_id": "D100057",
                "source_assertion_id": "DPA-test-reactive-arthritis",
                "protection_scope": [CROSS_SYSTEM_SCOPE],
                "protection_status": "active",
                "reason_code": BRIDGE_REASON,
                "strength": "strong",
            }
        ]

        filtered = self.engine.judge.pool_filter.filter([primary, reactive])

        self.assertNotIn(
            REACTIVE_ARTHRITIS,
            [item.diagnosis for item in filtered.candidates],
        )
        self.assertTrue(
            any(
                item.get("diagnosis") == REACTIVE_ARTHRITIS
                and item.get("reason") == "negative_feature"
                for item in filtered.excluded
            )
        )

    def test_leukemia_confirmatory_gap_prefers_marrow_flow_and_molecular_exams(self):
        leukemia = candidate(
            LEUKEMIA,
            0.42,
            required=False,
            diagnosis_type="disease",
            specificity=0.86,
            coverage=0.46,
            residual=0.32,
            core_coverage=0.44,
            matched=[
                "anemia",
                "platelet_low",
                "white_blood_cell_abnormal",
                "bleeding_tendency",
            ],
            gaps=["bone_marrow_blast_confirmed"],
            core_score=0.46,
            diagnostic_score=0.28,
        )
        leukemia.entity_id = "D000025"
        leukemia.eligibility_status = "Deferred"
        leukemia.eligibility_reason = "NeedsAnchor"
        leukemia.eligibility_substatus = "DeferredNeedsConfirmatoryExam"
        generic = candidate(
            "generic_primary",
            0.60,
            required=True,
            matched=["symptom:fever"],
        )

        decision = self.run_candidates([generic, leukemia])
        payload = decision.judge_decision
        tasks = [
            item
            for item in payload["discriminating_exam_tasks"]
            if LEUKEMIA in item.get("target_candidates", [])
        ]
        exams = [item["exam"] for item in tasks]

        self.assertIn("\u9aa8\u9ad3\u7a7f\u523a\u548c\u6d3b\u68c0\uff08BMAB\uff09", exams)
        self.assertIn("\u6d41\u5f0f\u7ec6\u80de\u672f\u514d\u75ab\u5206\u578b", exams)
        self.assertLess(
            exams.index("\u9aa8\u9ad3\u7a7f\u523a\u548c\u6d3b\u68c0\uff08BMAB\uff09"),
            exams.index("\u5916\u5468\u8840\u6d82\u7247") if "\u5916\u5468\u8840\u6d82\u7247" in exams else len(exams),
        )
        marrow_task = next(
            item for item in tasks
            if item["exam"] == "\u9aa8\u9ad3\u7a7f\u523a\u548c\u6d3b\u68c0\uff08BMAB\uff09"
        )
        self.assertEqual(marrow_task["exam_source"], "deferred_gap_closure_exam")
        self.assertEqual(marrow_task["priority_bucket"], "high_value_deferred_gap_closure")
        self.assertEqual(marrow_task["gap_diagnostic_coverage"], 1.0)

    def test_hard_contradiction_still_blocks_gap_authorization(self):
        ohss = candidate(
            OHSS,
            0.70,
            required=False,
            diagnosis_type="systemic",
            specificity=0.94,
            coverage=0.8,
            residual=0.1,
            matched=["hemoconcentration"],
            gaps=["pelvic_ultrasound"],
        )
        ohss.hard_contradiction = True
        pancreatitis = candidate(PANCREATITIS, 0.52, required=True)
        decision = self.run_candidates([ohss, pancreatitis])
        self.assertEqual(decision.final_diagnoses, [PANCREATITIS])
        self.assertNotIn(OHSS, decision.required_gap_authorized_diagnoses)

    def test_reasoning_structured_conflict_defers_primary_and_orders_adjudication(self):
        low_magnesium = candidate(
            LOW_MAGNESIUM,
            0.72,
            required=True,
            diagnosis_type="metabolic",
            specificity=0.90,
            coverage=0.74,
            residual=0.16,
            core_coverage=0.78,
            residual_core=0,
            matched=[
                "magnesium_load_retention_high",
                "magnesium_depletion",
                "muscle_cramp",
            ],
            core_score=0.70,
            diagnostic_score=0.86,
        )
        conflict = {
            "conflict_type": "reasoning_structured_polarity_conflict",
            "affected_diagnosis": LOW_MAGNESIUM,
            "finding": "magnesium_load_retention_high",
            "reasoning_text": "镁负荷试验排除低镁血症",
            "structured_sources": [{"source": "镁负荷试验"}],
            "adjudication_exams": [
                "血清电解质",
                "24小时尿电解质检测",
                "镁负荷试验",
                "维生素D检测",
                "甲状旁腺激素检测（PTH）",
                "X线检查",
            ],
            "action": "defer_primary_and_order_discriminating_exams",
            "status": "unresolved",
        }
        low_magnesium.evidence_conflicts = [conflict]
        low_magnesium.unresolved_evidence_conflict = True
        low_magnesium.conflict_adjudication_exams = list(conflict["adjudication_exams"])
        rickets = candidate(
            RICKETS,
            0.60,
            required=True,
            diagnosis_type="metabolic",
            specificity=0.92,
            coverage=0.64,
            residual=0.24,
            core_coverage=0.68,
            residual_core=0,
            matched=["vitamin_d_low", "bone_deformity", "waddling_gait"],
            core_score=0.66,
            diagnostic_score=0.32,
        )
        decision = self.run_candidates([low_magnesium, rickets])
        payload = decision.judge_decision
        self.assertEqual(payload["primary_status"], "deferred")
        self.assertTrue(payload["needs_discriminating_exams"])
        self.assertIn(LOW_MAGNESIUM, payload["conflict_affected_diagnoses"])
        self.assertIn(RICKETS, payload["differential_candidates"])
        tasks = payload["discriminating_exam_tasks"]
        self.assertTrue(
            any(item.get("exam_source") == "conflict_adjudication_exam" for item in tasks)
        )
        exams = [item["exam"] for item in tasks]
        self.assertIn("血清电解质", exams)
        self.assertNotIn(LOW_MAGNESIUM, decision.final_diagnoses)
        self.assertEqual(low_magnesium.eligibility_status, "Deferred")
        self.assertEqual(low_magnesium.eligibility_reason, "ConflictNeedsAdjudication")

    def test_final_gate_blocks_conflict_candidate_but_keeps_clean_alternative(self):
        low_magnesium = candidate(
            LOW_MAGNESIUM,
            0.72,
            required=True,
            matched=["magnesium_load_retention_high", "magnesium_depletion"],
            diagnostic_score=0.86,
        )
        low_magnesium.unresolved_evidence_conflict = True
        low_magnesium.evidence_conflicts = [
            {
                "conflict_type": "reasoning_structured_polarity_conflict",
                "affected_diagnosis": LOW_MAGNESIUM,
                "finding": "magnesium_depletion",
                "status": "unresolved",
            }
        ]
        rickets = candidate(
            RICKETS,
            0.66,
            required=True,
            matched=["vitamin_d_low", "bone_deformity"],
            core_score=0.62,
            diagnostic_score=0.30,
        )
        decision = DiagnosisDecision(
            final_diagnoses=[LOW_MAGNESIUM, RICKETS],
            trusted_diagnoses=[LOW_MAGNESIUM, RICKETS],
            candidates=[low_magnesium, rickets],
            unexplained_evidence=[],
            confidence=0.0,
            margin=0.0,
            low_confidence=False,
        )
        self.engine.authorize_final_diagnoses(
            decision,
            [LOW_MAGNESIUM, RICKETS],
        )
        self.assertEqual(decision.final_diagnoses, [RICKETS])
        self.assertTrue(
            any(
                item.get("diagnosis") == LOW_MAGNESIUM
                and item.get("reason") == "unresolved reasoning-structured evidence conflict"
                for item in decision.blocked_diagnoses
            )
        )

    def test_root_cause_arbitration_promotes_metabolic_bone_cause(self):
        low_magnesium = candidate(
            LOW_MAGNESIUM,
            0.78,
            required=True,
            diagnosis_type="metabolic",
            specificity=0.90,
            coverage=0.82,
            residual=0.12,
            core_coverage=0.82,
            residual_core=0,
            matched=[
                "magnesium_load_retention_high",
                "magnesium_depletion",
                "muscle_cramp",
            ],
            core_score=0.68,
            diagnostic_score=0.86,
        )
        rickets = candidate(
            RICKETS,
            0.62,
            required=True,
            diagnosis_type="metabolic",
            specificity=0.92,
            coverage=0.70,
            residual=0.22,
            core_coverage=0.70,
            residual_core=0,
            matched=[
                "vitamin_d_low",
                "hypocalcemia",
                "alp_elevated",
                "bone_deformity",
                "waddling_gait",
            ],
            core_score=0.70,
            diagnostic_score=0.30,
        )
        decision = self.run_candidates([low_magnesium, rickets])
        payload = decision.judge_decision
        root_payload = payload["root_cause_arbitration"]
        self.assertTrue(root_payload["applied"])
        self.assertTrue(root_payload["primary_override"])
        self.assertEqual(decision.final_diagnoses, [RICKETS])
        self.assertEqual(payload["primary"], RICKETS)
        self.assertEqual(payload["root_cause_primary"], RICKETS)
        self.assertIn(LOW_MAGNESIUM, payload["root_cause_secondary"])
        self.assertEqual(low_magnesium.explained_by_root_cause, RICKETS)
        self.assertEqual(low_magnesium.root_cause_role, "secondary")
        self.assertFalse(low_magnesium.root_cause_submit_as_final)

    def test_root_cause_arbitration_does_not_move_pure_low_magnesium(self):
        low_magnesium = candidate(
            LOW_MAGNESIUM,
            0.78,
            required=True,
            diagnosis_type="metabolic",
            specificity=0.90,
            coverage=0.82,
            residual=0.12,
            core_coverage=0.82,
            residual_core=0,
            matched=[
                "magnesium_load_retention_high",
                "magnesium_depletion",
            ],
            core_score=0.68,
            diagnostic_score=0.86,
        )
        decision = self.run_candidates([low_magnesium])
        self.assertEqual(decision.final_diagnoses, [LOW_MAGNESIUM])
        self.assertFalse(decision.judge_decision["root_cause_arbitration"]["applied"])
        self.assertEqual(low_magnesium.root_cause_role, "")

    def test_root_cause_arbitration_keeps_deferred_root_cause_in_workup(self):
        low_magnesium = candidate(
            LOW_MAGNESIUM,
            0.67,
            required=True,
            diagnosis_type="metabolic",
            specificity=0.90,
            coverage=0.35,
            residual=0.65,
            core_coverage=0.40,
            residual_core=3,
            matched=["magnesium_load_retention_high", "magnesium_depletion"],
            diagnostic_score=1.0,
        )
        rickets = candidate(
            RICKETS,
            0.89,
            required=False,
            diagnosis_type="metabolic",
            specificity=0.92,
            coverage=0.52,
            residual=0.48,
            core_coverage=0.40,
            residual_core=3,
            matched=[
                "symptom:腿痛",
                "symptom:间歇性跛行",
                "alp_elevated",
                "hypocalcemia",
            ],
            diagnostic_score=1.0,
        )
        rickets.required_gaps = ["vitamin_d_low|bone_deformity"]
        rickets.required_gap_state = "actionable_gap"
        rickets.component_scores["required_gap_state"] = "actionable_gap"
        decision = self.run_candidates([low_magnesium, rickets])
        payload = decision.judge_decision
        self.assertEqual(decision.final_diagnoses, [LOW_MAGNESIUM])
        self.assertIn(RICKETS, payload["deferred_anchor_candidates"])
        self.assertIn(RICKETS, payload["evidence_gap_targets"])
        self.assertEqual(decision.required_gap_authorized_diagnoses, [])
        self.assertFalse(payload["root_cause_arbitration"]["applied"])

    def test_gap_value_is_emitted_separately_from_candidate_score(self):
        low_magnesium = candidate(
            LOW_MAGNESIUM,
            0.89,
            required=True,
            diagnosis_type="metabolic",
            specificity=0.90,
            coverage=0.78,
            residual=0.12,
            core_coverage=0.78,
            residual_core=0,
            matched=["magnesium_load_retention_high", "magnesium_depletion"],
            core_score=0.68,
            diagnostic_score=0.86,
        )
        rickets = candidate(
            RICKETS,
            0.73,
            required=False,
            diagnosis_type="metabolic",
            specificity=0.92,
            coverage=0.52,
            residual=0.48,
            core_coverage=0.42,
            residual_core=3,
            matched=[
                "symptom:leg_pain",
                "symptom:waddling_gait",
                "alp_elevated",
                "hypocalcemia",
            ],
            gaps=["vitamin_d_low|bone_deformity"],
            diagnostic_score=1.0,
        )
        rickets.required_gap_state = "actionable_gap"
        rickets.component_scores["required_gap_state"] = "actionable_gap"
        rickets.candidate_value = "high"

        decision = self.run_candidates([low_magnesium, rickets])
        payload = decision.judge_decision
        rickets_gaps = [
            gap
            for gap in payload["active_evidence_gaps"]
            if gap.get("candidate") == RICKETS
        ]

        self.assertTrue(rickets_gaps)
        best_gap = rickets_gaps[0]
        self.assertGreater(best_gap["gap_value"], 0.0)
        self.assertEqual(best_gap["candidate_score_at_decision"], 0.73)
        self.assertLess(best_gap["candidate_score_at_decision"], low_magnesium.score)
        self.assertTrue(best_gap["score_gap_decoupled"])
        self.assertIn("gap_value_components", best_gap)

        reviews = {
            item["diagnosis"]: item
            for item in payload["reviews"]
            if isinstance(item, dict)
        }
        self.assertEqual(reviews[RICKETS]["max_gap_value"], best_gap["gap_value"])
        self.assertEqual(reviews[RICKETS]["deferred_priority"], best_gap["gap_value"])
        self.assertGreater(reviews[RICKETS]["actionable_gap_count"], 0)

        rickets_tasks = [
            item
            for item in payload["discriminating_exam_tasks"]
            if RICKETS in item.get("target_candidates", [])
            and item.get("exam_source") == "deferred_gap_closure_exam"
        ]
        self.assertTrue(rickets_tasks)
        self.assertTrue(
            any(
                item.get("exam")
                in {
                    "\u7ef4\u751f\u7d20D\u68c0\u6d4b",
                    "\u7532\u72b6\u65c1\u817a\u6fc0\u7d20\u68c0\u6d4b\uff08PTH\uff09",
                    "X\u7ebf\u68c0\u67e5",
                }
                for item in rickets_tasks
            )
        )
        self.assertTrue(
            all(
                item.get("source_gap_value") == best_gap["gap_value"]
                and item.get("candidate_score_at_decision") == 0.73
                and item.get("score_gap_decoupled")
                for item in rickets_tasks
            )
        )

    def test_root_cause_arbitration_respects_upstream_contradiction(self):
        low_magnesium = candidate(
            LOW_MAGNESIUM,
            0.78,
            required=True,
            diagnosis_type="metabolic",
            specificity=0.90,
            coverage=0.82,
            residual=0.12,
            core_coverage=0.82,
            residual_core=0,
            matched=["magnesium_load_retention_high", "magnesium_depletion"],
            core_score=0.68,
            diagnostic_score=0.86,
        )
        rickets = candidate(
            RICKETS,
            0.66,
            required=True,
            diagnosis_type="metabolic",
            coverage=0.76,
            residual=0.18,
            core_coverage=0.76,
            matched=["vitamin_d_low", "hypocalcemia", "bone_deformity"],
            core_score=0.72,
            diagnostic_score=0.30,
        )
        rickets.hard_contradiction = True
        decision = self.run_candidates([low_magnesium, rickets])
        self.assertEqual(decision.final_diagnoses, [LOW_MAGNESIUM])
        self.assertFalse(decision.judge_decision["root_cause_arbitration"]["applied"])

    def test_root_cause_arbitration_uses_generic_structural_relation(self):
        heart_failure = candidate(
            HEART_FAILURE,
            0.78,
            required=True,
            diagnosis_type="state",
            specificity=0.72,
            coverage=0.82,
            residual=0.12,
            core_coverage=0.82,
            matched=["heart_failure_state", "fluid_retention_pattern"],
            core_score=0.56,
            diagnostic_score=0.0,
        )
        pulmonary_stenosis = candidate(
            PULMONARY_STENOSIS,
            0.60,
            required=True,
            diagnosis_type="structural",
            specificity=0.92,
            coverage=0.62,
            residual=0.28,
            core_coverage=0.62,
            matched=["pulmonary_valve_stenosis", "valve_gradient_high"],
            core_score=0.62,
            diagnostic_score=0.40,
        )
        decision = self.run_candidates([heart_failure, pulmonary_stenosis])
        self.assertEqual(decision.final_diagnoses[:2], [PULMONARY_STENOSIS, HEART_FAILURE])
        self.assertTrue(decision.judge_decision["root_cause_arbitration"]["applied"])
        self.assertEqual(heart_failure.explained_by_root_cause, PULMONARY_STENOSIS)
        self.assertTrue(heart_failure.root_cause_submit_as_final)

    def test_root_cause_arbitration_submits_heart_failure_with_state_pattern(self):
        mitral = candidate(
            MITRAL_REGURGITATION,
            0.82,
            required=True,
            diagnosis_type="structural",
            specificity=0.9,
            coverage=0.78,
            residual=0.18,
            core_coverage=0.78,
            matched=["mitral_regurgitation", "diagnosis:二尖瓣反流"],
            core_score=0.76,
            diagnostic_score=0.40,
        )
        heart_failure = candidate(
            HEART_FAILURE,
            0.52,
            required=True,
            diagnosis_type="state",
            specificity=0.50,
            coverage=0.62,
            residual=0.28,
            core_coverage=0.62,
            matched=[
                "fluid_retention_pattern",
                "leg_edema",
                "paroxysmal_nocturnal_dyspnea",
                "dyspnea_on_exertion",
            ],
            core_score=0.44,
            diagnostic_score=0.0,
        )
        decision = self.run_candidates([mitral, heart_failure])
        self.assertEqual(decision.final_diagnoses[:2], [MITRAL_REGURGITATION, HEART_FAILURE])
        self.assertTrue(decision.judge_decision["root_cause_arbitration"]["applied"])
        self.assertEqual(heart_failure.explained_by_root_cause, MITRAL_REGURGITATION)
        self.assertTrue(heart_failure.root_cause_submit_as_final)

    def test_direct_congenital_parent_beats_gap_child_even_without_parent_field(self):
        pulmonary = candidate(
            PULMONARY_STENOSIS,
            0.58,
            required=False,
            diagnosis_type="structural",
            specificity=0.92,
            coverage=0.64,
            residual=0.24,
            matched=["cyanosis", "right_ventricular_hypertrophy"],
            gaps=["pulmonary_valve_gradient"],
            parent=HEART_FAILURE,
        )
        congenital = candidate(
            CONGENITAL_HEART,
            0.47,
            required=True,
            diagnosis_type="structural",
            specificity=0.9,
            coverage=0.58,
            residual=0.30,
            matched=[f"diagnosis:{CONGENITAL_HEART}", "congenital_heart_defect"],
        )
        decision = self.run_candidates([pulmonary, congenital])
        self.assertEqual(decision.final_diagnoses, [CONGENITAL_HEART])

    def test_primary_unlock_reason_records_changed_preselection(self):
        previous = candidate(ECZEMA, 0.60, required=True, coverage=0.30, residual=0.70, residual_core=3)
        yaws = candidate(
            YAWS,
            0.56,
            required=True,
            diagnosis_type="etiology",
            specificity=0.94,
            coverage=0.74,
            residual=0.18,
            core_coverage=0.78,
            matched=["treponemal_skin_lesion", "periostitis"],
        )
        self.engine.eligibility_gate.evaluate_all([previous, yaws], None)
        judge_decision = self.engine.judge.judge([previous, yaws], preselected=[ECZEMA])
        self.assertEqual(judge_decision.primary, YAWS)
        self.assertTrue(judge_decision.primary_unlock_reason)


if __name__ == "__main__":
    unittest.main()
