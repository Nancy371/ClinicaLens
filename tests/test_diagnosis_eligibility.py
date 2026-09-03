import unittest
from types import SimpleNamespace

from agent.clinical_evidence import EvidenceBundle, Observation
from agent.diagnosis_eligibility import (
    ANCHORS_SATISFIED,
    DEFERRED,
    DIFFERENTIAL_ONLY,
    EXCLUDED,
    NEEDS_ANCHOR,
    NO_VALID_ANCHOR,
    PATTERN_CONTRADICTED,
    PRIMARY_ELIGIBLE,
    DiagnosisEligibilityGate,
)
from agent.diagnosis_engine import DiagnosticKnowledgeBase


def candidate(**overrides):
    data = {
        "diagnosis": "candidate",
        "diagnosis_type": "disease",
        "matched_evidence": ["symptom:signal"],
        "core_matched_evidence": [],
        "diagnostic_matched_evidence": [],
        "generic_matched_evidence": [],
        "required_gaps": [],
        "required_met": True,
        "hard_contradiction": False,
        "hard_contradicted_evidence": [],
        "unresolved_evidence_conflict": False,
        "differential_only": False,
        "differential_only_reason": "",
        "source_prior": 0.4,
        "coverage_score": 0.5,
        "core_explanatory_coverage": 0.5,
        "diagnostic_evidence_score": 0.0,
        "core_evidence_score": 0.0,
        "residual_core_evidence_count": 0,
        "component_scores": {},
    }
    data.update(overrides)
    return SimpleNamespace(**data)


class DiagnosisEligibilityGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = DiagnosisEligibilityGate()
        self.knowledge_gate = DiagnosisEligibilityGate(DiagnosticKnowledgeBase("data/ref_data"))

    def test_zoster_without_dermatomal_or_vesicular_anchor_is_not_primary_eligible(self):
        zoster = candidate(
            diagnosis="\u5e26\u72b6\u75b1\u75b9",
            diagnosis_type="disease",
            required_met=True,
            matched_evidence=["arthralgia", "dysuria", "ocular_redness"],
            core_matched_evidence=["ocular_redness"],
            required_gaps=[],
            source_prior=0.62,
            coverage_score=0.55,
            core_explanatory_coverage=0.42,
        )

        result = self.knowledge_gate.evaluate(zoster)

        self.assertEqual(result.status, DIFFERENTIAL_ONLY)
        self.assertEqual(result.reason, NO_VALID_ANCHOR)
        self.assertEqual(result.anchor_status, NO_VALID_ANCHOR)
        self.assertIn("no_valid_diagnostic_anchor", result.blockers)

    def test_zoster_dermatomal_pain_plus_vesicular_rash_satisfies_anchor_policy(self):
        zoster = candidate(
            diagnosis="\u5e26\u72b6\u75b1\u75b9",
            diagnosis_type="disease",
            required_met=True,
            matched_evidence=["dermatomal_pain", "vesicular_rash"],
            core_matched_evidence=["dermatomal_pain", "vesicular_rash"],
            diagnostic_matched_evidence=["vesicular_rash"],
            required_gaps=[],
            source_prior=0.62,
            coverage_score=0.72,
            core_explanatory_coverage=0.70,
            core_evidence_score=0.62,
            diagnostic_evidence_score=0.42,
        )

        result = self.knowledge_gate.evaluate(zoster)

        self.assertEqual(result.status, PRIMARY_ELIGIBLE)
        self.assertEqual(result.anchor_status, "AnchorSatisfied")
        self.assertIn(
            "disease_specific_anchor",
            result.anchor_policy_audit.get("matched_anchor_types", []),
        )

    def test_missing_vitamin_d_anchor_defers_rickets_for_workup(self):
        rickets = candidate(
            diagnosis="vitamin_d_deficiency_rickets",
            diagnosis_type="metabolic",
            required_met=False,
            matched_evidence=["bone_pain", "waddling_gait", "alp_elevated", "hypocalcemia"],
            core_matched_evidence=["bone_pain", "alp_elevated", "hypocalcemia"],
            required_gaps=["vitamin_d_low|bone_deformity"],
            source_prior=0.55,
            coverage_score=0.62,
            core_explanatory_coverage=0.58,
        )

        result = self.gate.evaluate(rickets)

        self.assertEqual(result.status, DEFERRED)
        self.assertEqual(result.reason, NEEDS_ANCHOR)
        self.assertIn("vitamin_d_low|bone_deformity", result.missing_required_anchors)

    def test_closed_vitamin_d_anchor_is_primary_eligible(self):
        rickets = candidate(
            diagnosis="vitamin_d_deficiency_rickets",
            diagnosis_type="metabolic",
            required_met=True,
            matched_evidence=["vitamin_d_low", "hypocalcemia", "alp_elevated", "bone_pain"],
            core_matched_evidence=["hypocalcemia", "alp_elevated", "bone_pain"],
            diagnostic_matched_evidence=["vitamin_d_low"],
            core_evidence_score=0.72,
            diagnostic_evidence_score=0.62,
        )

        result = self.gate.evaluate(rickets)

        self.assertEqual(result.status, PRIMARY_ELIGIBLE)
        self.assertEqual(result.reason, ANCHORS_SATISFIED)
        self.assertIn("vitamin_d_low", result.satisfied_required_anchors)

    def test_structural_chd_extension_requires_objective_structural_anchor(self):
        cor = candidate(
            diagnosis="三房心",
            diagnosis_type="structural",
            matched_evidence=["cyanosis", "dyspnea"],
            core_matched_evidence=["cyanosis"],
            diagnostic_matched_evidence=[],
            source_prior=0.72,
        )

        result = self.knowledge_gate.evaluate(
            cor,
            evidence=EvidenceBundle(
                [
                    Observation("cyanosis", "physical_exam"),
                    Observation("dyspnea", "patient_report"),
                ]
            ),
        )

        self.assertEqual(result.status, DEFERRED)
        self.assertIn("left_atrial_membrane", " ".join(result.missing_required_anchors))

    def test_structural_chd_extension_with_objective_anchor_is_primary_eligible(self):
        cor = candidate(
            diagnosis="三房心",
            diagnosis_type="structural",
            matched_evidence=["cyanosis", "left_atrial_membrane"],
            core_matched_evidence=["left_atrial_membrane"],
            diagnostic_matched_evidence=["left_atrial_membrane"],
            source_prior=0.72,
        )

        result = self.knowledge_gate.evaluate(
            cor,
            evidence=EvidenceBundle(
                [
                    Observation("cyanosis", "physical_exam"),
                    Observation("left_atrial_membrane", "exam_result"),
                ]
            ),
        )

        self.assertEqual(result.status, PRIMARY_ELIGIBLE)

    def test_missing_pericardial_anchor_does_not_exclude_candidate(self):
        tb_pericarditis = candidate(
            diagnosis="tuberculous_pericarditis",
            diagnosis_type="etiology",
            required_met=False,
            matched_evidence=["cough", "fever", "dyspnea"],
            required_gaps=["pericardial_effusion", "tb_microbiology_positive"],
            source_prior=0.2,
            coverage_score=0.18,
            core_explanatory_coverage=0.05,
        )

        result = self.gate.evaluate(tb_pericarditis)

        self.assertIn(result.status, {DEFERRED, DIFFERENTIAL_ONLY})
        self.assertNotEqual(result.status, PRIMARY_ELIGIBLE)
        self.assertNotEqual(result.status, EXCLUDED)

    def test_hard_contradiction_excludes_candidate(self):
        tb_pericarditis = candidate(
            diagnosis="tuberculous_pericarditis",
            required_met=False,
            hard_contradiction=True,
            hard_contradicted_evidence=["normal_pericardium"],
            matched_evidence=["cough", "fever"],
            required_gaps=["pericardial_effusion", "tb_microbiology_positive"],
        )

        result = self.gate.evaluate(tb_pericarditis)

        self.assertEqual(result.status, EXCLUDED)
        self.assertIn("normal_pericardium", result.blockers)

    def test_downstream_state_with_poor_global_explanation_is_differential_only(self):
        low_magnesium = candidate(
            diagnosis="low_magnesium",
            diagnosis_type="metabolic",
            required_met=True,
            matched_evidence=["magnesium_depletion"],
            diagnostic_matched_evidence=["magnesium_depletion"],
            coverage_score=0.25,
            core_explanatory_coverage=0.2,
            diagnostic_evidence_score=0.2,
            core_evidence_score=0.1,
            residual_core_evidence_count=4,
        )

        result = self.gate.evaluate(low_magnesium)

        self.assertEqual(result.status, DIFFERENTIAL_ONLY)

    def test_prostatitis_without_urinary_or_prostate_anchor_is_deferred(self):
        prostatitis = candidate(
            diagnosis="急性细菌性前列腺炎",
            diagnosis_type="etiology",
            required_met=True,
            matched_evidence=["fever", "acute_course", "cough", "dyspnea", "bronchopneumonia"],
            core_matched_evidence=["fever", "acute_course"],
            required_gaps=[],
            source_prior=0.55,
            coverage_score=0.50,
            core_explanatory_coverage=0.42,
            core_evidence_score=0.20,
            diagnostic_evidence_score=0.0,
        )

        result = self.knowledge_gate.evaluate(prostatitis)

        self.assertEqual(result.status, DEFERRED)
        self.assertEqual(result.reason, NEEDS_ANCHOR)
        self.assertTrue(
            any(
                item.startswith("acute_bacterial_prostatitis_confirmed_pattern:")
                for item in result.missing_required_anchors
            )
        )

    def test_pyuria_alone_is_not_prostatitis_anchor(self):
        prostatitis = candidate(
            diagnosis="急性细菌性前列腺炎",
            diagnosis_type="etiology",
            required_met=True,
            matched_evidence=["pyuria"],
            core_matched_evidence=["pyuria"],
            required_gaps=[],
            source_prior=0.55,
            coverage_score=0.45,
            core_explanatory_coverage=0.36,
        )

        result = self.knowledge_gate.evaluate(prostatitis)

        self.assertEqual(result.status, DEFERRED)
        self.assertEqual(result.reason, NEEDS_ANCHOR)
        self.assertTrue(
            any(
                item.startswith("acute_bacterial_prostatitis_confirmed_pattern:")
                for item in result.missing_required_anchors
            )
        )

    def test_negative_urine_pattern_downgrades_pyuria_support(self):
        prostatitis = candidate(
            diagnosis="急性细菌性前列腺炎",
            diagnosis_type="etiology",
            required_met=True,
            matched_evidence=[
                "pyuria",
                "urine_culture_no_growth",
                "leukocyte_esterase_negative",
                "nitrite_negative",
            ],
            core_matched_evidence=["pyuria"],
            required_gaps=[],
            source_prior=0.55,
            coverage_score=0.45,
            core_explanatory_coverage=0.36,
        )

        result = self.knowledge_gate.evaluate(prostatitis)

        self.assertEqual(result.status, DIFFERENTIAL_ONLY)
        self.assertEqual(result.reason, PATTERN_CONTRADICTED)
        self.assertIn("urine_culture_no_growth", result.blockers)
        self.assertEqual(result.evidence_pattern_matches[0]["pattern_type"], "negative_pattern")

    def test_confirmed_bacterial_prostatitis_pattern_is_primary_eligible(self):
        prostatitis = candidate(
            diagnosis="急性细菌性前列腺炎",
            diagnosis_type="etiology",
            required_met=True,
            matched_evidence=[
                "prostate_tenderness",
                "dysuria",
                "pyuria",
                "urine_culture_positive",
            ],
            core_matched_evidence=["prostate_tenderness", "dysuria"],
            diagnostic_matched_evidence=["urine_culture_positive"],
            required_gaps=[],
            core_evidence_score=0.62,
            diagnostic_evidence_score=0.55,
        )

        result = self.knowledge_gate.evaluate(prostatitis)

        self.assertEqual(result.status, PRIMARY_ELIGIBLE)
        self.assertEqual(result.reason, ANCHORS_SATISFIED)
        self.assertEqual(
            result.evidence_pattern_matches[0]["pattern_id"],
            "acute_bacterial_prostatitis_confirmed_pattern",
        )

    def test_reasoning_only_cannot_satisfy_objective_pavm_confirmed_pattern(self):
        pavm = candidate(
            diagnosis="肺动静脉瘘",
            diagnosis_type="structural",
            required_met=True,
            matched_evidence=[
                "hemoptysis",
                "right_to_left_shunt",
                "enhanced_ct_vascular_malformation",
            ],
            core_matched_evidence=["hemoptysis", "right_to_left_shunt"],
            diagnostic_matched_evidence=["enhanced_ct_vascular_malformation"],
            required_gaps=[],
            source_prior=0.75,
            coverage_score=0.7,
        )
        evidence = EvidenceBundle(
            [
                Observation("hemoptysis", "reasoning_inference", confidence=0.7),
                Observation("right_to_left_shunt", "reasoning_inference", confidence=0.7),
                Observation("enhanced_ct_vascular_malformation", "reasoning_inference", confidence=0.7),
            ]
        )

        result = self.knowledge_gate.evaluate(pavm, evidence=evidence)

        self.assertEqual(result.status, DEFERRED)
        self.assertEqual(result.reason, NEEDS_ANCHOR)
        confirmed = [
            item for item in result.evidence_pattern_matches
            if item["pattern_id"] == "pulmonary_avm_confirmed_vascular_pattern"
        ]
        self.assertEqual(len(confirmed), 1)
        self.assertFalse(confirmed[0]["matched"])
        self.assertFalse(confirmed[0]["objective_source_satisfied"])

    def test_pavm_initial_deferred_then_confirmed_with_objective_ct_anchor(self):
        initial = candidate(
            diagnosis="肺动静脉瘘",
            diagnosis_type="structural",
            required_met=True,
            matched_evidence=["hemoptysis", "right_to_left_shunt"],
            core_matched_evidence=["hemoptysis", "right_to_left_shunt"],
            required_gaps=[],
            source_prior=0.75,
            coverage_score=0.7,
        )

        initial_result = self.knowledge_gate.evaluate(initial)

        self.assertEqual(initial_result.status, DEFERRED)
        self.assertTrue(
            any(
                item["pattern_id"] == "pulmonary_avm_initial_shunt_pattern"
                for item in initial_result.evidence_pattern_matches
            )
        )

        confirmed = candidate(
            diagnosis="肺动静脉瘘",
            diagnosis_type="structural",
            required_met=True,
            matched_evidence=[
                "hemoptysis",
                "right_to_left_shunt",
                "enhanced_ct_vascular_malformation",
            ],
            core_matched_evidence=["hemoptysis", "right_to_left_shunt"],
            diagnostic_matched_evidence=["enhanced_ct_vascular_malformation"],
            required_gaps=[],
            source_prior=0.75,
            coverage_score=0.8,
            diagnostic_evidence_score=0.68,
            core_evidence_score=0.62,
        )
        evidence = EvidenceBundle(
            [
                Observation("hemoptysis", "问诊", confidence=0.86),
                Observation("right_to_left_shunt", "超声心动图右心声学造影", confidence=0.92),
                Observation("enhanced_ct_vascular_malformation", "胸部增强CT", confidence=0.96),
            ]
        )

        confirmed_result = self.knowledge_gate.evaluate(confirmed, evidence=evidence)

        self.assertEqual(confirmed_result.status, PRIMARY_ELIGIBLE)
        self.assertEqual(confirmed_result.reason, ANCHORS_SATISFIED)
        self.assertTrue(
            any(
                item["pattern_id"] == "pulmonary_avm_confirmed_vascular_pattern"
                for item in confirmed_result.evidence_pattern_matches
            )
        )

    def test_pulmonary_cryptococcosis_without_fungal_anchor_is_deferred(self):
        crypto = candidate(
            diagnosis="肺隐球菌病",
            diagnosis_type="disease",
            required_met=True,
            matched_evidence=["cough", "fever", "acute_course", "dyspnea"],
            core_matched_evidence=["cough", "fever"],
            required_gaps=[],
            source_prior=0.55,
            coverage_score=0.50,
            core_explanatory_coverage=0.42,
        )

        result = self.gate.evaluate(crypto)

        self.assertEqual(result.status, DEFERRED)
        self.assertEqual(result.reason, NEEDS_ANCHOR)

    def test_pulmonary_cryptococcosis_with_cryptococcal_anchor_is_primary_eligible(self):
        crypto = candidate(
            diagnosis="肺隐球菌病",
            diagnosis_type="disease",
            required_met=True,
            matched_evidence=["cryptococcal_antigen_positive", "pulmonary_nodule"],
            core_matched_evidence=["cryptococcal_antigen_positive"],
            diagnostic_matched_evidence=["cryptococcal_antigen_positive"],
            required_gaps=[],
            diagnostic_evidence_score=0.55,
            core_evidence_score=0.55,
        )

        result = self.gate.evaluate(crypto)

        self.assertEqual(result.status, PRIMARY_ELIGIBLE)
        self.assertEqual(result.reason, ANCHORS_SATISFIED)

    def test_mycoplasma_pneumonia_without_pathogen_anchor_is_deferred(self):
        mycoplasma = candidate(
            diagnosis="支原体肺炎",
            diagnosis_type="disease",
            required_met=True,
            matched_evidence=["cough", "fever", "dyspnea", "pneumonia_infiltrate"],
            core_matched_evidence=["cough", "fever"],
            required_gaps=[],
            source_prior=0.55,
            coverage_score=0.50,
            core_explanatory_coverage=0.42,
        )

        result = self.gate.evaluate(mycoplasma)

        self.assertEqual(result.status, DEFERRED)
        self.assertEqual(result.reason, NEEDS_ANCHOR)

    def test_mycoplasma_pneumonia_with_pathogen_anchor_is_primary_eligible(self):
        mycoplasma = candidate(
            diagnosis="支原体肺炎",
            diagnosis_type="disease",
            required_met=True,
            matched_evidence=["mycoplasma_naat_positive", "interstitial_infiltrate"],
            core_matched_evidence=["mycoplasma_naat_positive"],
            diagnostic_matched_evidence=["mycoplasma_naat_positive"],
            required_gaps=[],
            diagnostic_evidence_score=0.55,
            core_evidence_score=0.55,
        )

        result = self.gate.evaluate(mycoplasma)

        self.assertEqual(result.status, PRIMARY_ELIGIBLE)
        self.assertEqual(result.reason, ANCHORS_SATISFIED)


if __name__ == "__main__":
    unittest.main()
