import unittest

import yaml

from agent.case_board import (
    CaseBoard,
    CaseBoardPermissionError,
    ConsultationEvidencePipeline,
    EvidenceClaim,
    PatternCompiler,
    StaleJudgeDecisionError,
    TargetedEvidenceVerifier,
)
from agent.clinical_evidence import EvidenceBundle, Observation
from agent.diagnosis_eligibility import (
    DEFERRED,
    DEFERRED_NEEDS_CONFIRMATORY_EXAM,
    PRIMARY_ELIGIBLE,
)
from agent.candidate_generator import CandidatePool
from agent.diagnosis_engine import CandidateScore, DiagnosisDecision, DiagnosisDecisionEngine
from agent.diagnosis_judge import DiagnosisJudge, DiagnosisSubmitter, JudgeDecision
from agent.knowledge import KnowledgeBase


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidate(name="test disease", *, required_gap_authorized=False):
    return CandidateScore(
        diagnosis=name,
        score=0.9,
        support_score=0.9,
        source_prior=0.7,
        explanation_score=0.8,
        coverage_score=0.8,
        residual_score=0.1,
        contradiction_penalty=0.0,
        required_met=True,
        hard_contradiction=False,
        matched_evidence=[f"diagnosis:{name}"],
        eligibility_status=PRIMARY_ELIGIBLE,
        required_gap_authorized=required_gap_authorized,
    )


class CaseBoardTests(unittest.TestCase):
    def test_only_judge_can_write_official_decision_events(self):
        board = CaseBoard(case_id="case-1")

        with self.assertRaises(CaseBoardPermissionError):
            board.append_event(
                "candidate_decision",
                "reasoner",
                {"diagnosis": "A", "status": PRIMARY_ELIGIBLE},
            )

        board.append_event(
            "candidate_decision",
            "judge",
            {"diagnosis": "A", "status": PRIMARY_ELIGIBLE},
        )
        self.assertEqual(board.view()["candidate_decisions"][0]["diagnosis"], "A")

    def test_reasoning_claim_is_not_structured_evidence_without_verification(self):
        evidence = EvidenceBundle([Observation("fever", "问诊", raw_text="发热")])
        pipeline = ConsultationEvidencePipeline()

        board, enhanced = pipeline.run(
            evidence,
            llm_result={
                "diagnosis_candidates": [{"name": "白血病", "confidence": 0.9}],
                "reasoning": "需要排除白血病。",
            },
        )

        self.assertTrue(board.view()["evidence_claims"])
        self.assertNotIn("blast_present", enhanced.findings())
        self.assertEqual(pipeline.last_audit["unsupported_claim_admission_count"], 0)

    def test_targeted_verifier_requires_non_reasoning_source_span(self):
        verifier = TargetedEvidenceVerifier()
        evidence = EvidenceBundle(
            [
                Observation(
                    "field:smear",
                    "外周血涂片",
                    raw_text="外周血可见原始细胞约18%。",
                    source_text="外周血可见原始细胞约18%。",
                )
            ]
        )
        claim = EvidenceClaim(
            claim_id="claim_blast_present",
            target_evidence="blast_present",
            search_terms=["原始细胞"],
            importance="critical",
        )

        result = verifier.verify(claim, evidence)

        self.assertEqual(result.status, "Verified")
        self.assertEqual(result.evidence_id, "blast_present")
        self.assertIn("原始细胞", result.source_span)

    def test_reasoning_inference_cannot_verify_claim(self):
        verifier = TargetedEvidenceVerifier()
        evidence = EvidenceBundle(
            [Observation("blast_present", "reasoning_inference", raw_text="推测有原始细胞")]
        )
        claim = EvidenceClaim(
            claim_id="claim_blast_present",
            target_evidence="blast_present",
            search_terms=["原始细胞"],
            importance="critical",
        )

        result = verifier.verify(claim, evidence)

        self.assertEqual(result.status, "Unresolved")
        self.assertIn("reasoning_inference", result.reason)

    def test_pattern_compiler_derives_only_from_verified_atomic_evidence(self):
        compiler = PatternCompiler()
        evidence = EvidenceBundle(
            [
                Observation("anemia", "血常规", raw_text="血红蛋白降低"),
                Observation("thrombocytopenia", "血常规", raw_text="血小板减少"),
                Observation("leukocytosis", "血常规", raw_text="白细胞升高"),
            ]
        )

        derived = compiler.compile([], evidence)

        self.assertIn("multilineage_cytopenia", [item.finding for item in derived])

    def test_pattern_compiler_derives_acute_leukemia_pattern(self):
        compiler = PatternCompiler()
        evidence = EvidenceBundle(
            [
                Observation("blast_present", "targeted_evidence_verifier", raw_text="blast"),
                Observation("hemoglobin_low", "cbc", raw_text="hgb low"),
                Observation("platelet_low", "cbc", raw_text="plt low"),
                Observation("white_blood_cell_abnormal", "cbc", raw_text="wbc abnormal"),
            ]
        )

        derived = compiler.compile([], evidence)

        findings = [item.finding for item in derived]
        self.assertIn("multilineage_cytopenia", findings)
        self.assertIn("acute_leukemia_pattern", findings)

    def test_targeted_verifier_matches_blast_cell_synonym(self):
        verifier = TargetedEvidenceVerifier()
        evidence = EvidenceBundle(
            [
                Observation(
                    "field:smear",
                    "\u5916\u5468\u8840\u6d82\u7247",
                    raw_text="\u5916\u5468\u8840\u53ef\u89c1\u5927\u91cf\u5faa\u73af\u6bcd\u7ec6\u80de\u7ea625%",
                    source_text="\u5916\u5468\u8840\u53ef\u89c1\u5927\u91cf\u5faa\u73af\u6bcd\u7ec6\u80de\u7ea625%",
                )
            ]
        )
        claim = EvidenceClaim(
            claim_id="claim_blast_present",
            target_evidence="blast_present",
            search_terms=["\u6bcd\u7ec6\u80de"],
            importance="critical",
        )

        result = verifier.verify(claim, evidence)

        self.assertEqual(result.status, "Verified")
        self.assertEqual(result.evidence_id, "blast_present")

    def test_submitter_rejects_stale_judge_decision(self):
        item = candidate("白血病")
        decision = DiagnosisDecision(
            final_diagnoses=[item.diagnosis],
            trusted_diagnoses=[item.diagnosis],
            candidates=[item],
            unexplained_evidence=[],
            confidence=0.9,
            margin=0.9,
            low_confidence=False,
            evidence_snapshot_hash="sha256:new",
            case_version=2,
        )
        judge_decision = JudgeDecision(
            final_diagnoses=[item.diagnosis],
            evidence_snapshot_hash="sha256:old",
            case_version=1,
        )

        with self.assertRaises(StaleJudgeDecisionError):
            DiagnosisSubmitter().apply(decision, judge_decision)
        self.assertTrue(decision.stale_decision)

    def test_required_gap_authorized_is_audit_only_not_submission(self):
        engine = DiagnosisDecisionEngine(load_config(), "data/ref_data")
        item = candidate("白血病", required_gap_authorized=True)
        decision = DiagnosisDecision(
            final_diagnoses=[item.diagnosis],
            trusted_diagnoses=[item.diagnosis],
            candidates=[item],
            unexplained_evidence=[],
            confidence=0.9,
            margin=0.9,
            low_confidence=False,
        )

        engine.authorize_final_diagnoses(decision)

        self.assertEqual(decision.final_diagnoses, [])
        self.assertEqual(
            decision.blocked_diagnoses[0]["reason"],
            "required_gap_authorized is audit-only, not submission authorization",
        )

    def test_deferred_judge_decision_cannot_submit_final(self):
        engine = DiagnosisDecisionEngine(load_config(), "data/ref_data")
        item = candidate("test disease")
        decision = DiagnosisDecision(
            final_diagnoses=[item.diagnosis],
            trusted_diagnoses=[item.diagnosis],
            candidates=[item],
            unexplained_evidence=[],
            confidence=0.9,
            margin=0.9,
            low_confidence=False,
            judge_decision={
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
            },
        )

        engine.authorize_final_diagnoses(decision, [item.diagnosis])

        self.assertEqual(decision.final_diagnoses, [])
        self.assertEqual(
            decision.blocked_diagnoses[0]["reason"],
            "judge decision deferred pending discriminating exams",
        )

    def test_bph_direct_mention_without_luts_is_deferred(self):
        engine = DiagnosisDecisionEngine(load_config(), "data/ref_data")
        bph = "\u524d\u5217\u817a\u589e\u751f"
        item = candidate(bph)
        item.matched_evidence = [f"diagnosis:{bph}"]
        evidence = EvidenceBundle(
            [
                Observation(
                    f"diagnosis:{bph}",
                    "imaging",
                    raw_text="\u524d\u5217\u817a\u589e\u5927",
                )
            ]
        )

        engine.eligibility_gate.evaluate_all([item], evidence)

        self.assertEqual(item.eligibility_status, DEFERRED)
        self.assertTrue(
            any(
                "bph_luts_obstruction_pattern" in gap
                for gap in item.missing_required_anchors
            )
        )

    def test_unresolved_critical_claim_forces_workup_not_submission(self):
        engine = DiagnosisDecisionEngine(load_config(), "data/ref_data")
        leukemia = "\u767d\u8840\u75c5"
        pool = CandidatePool()
        pool.add(leukemia, leukemia, "llm", prior=0.75)
        evidence = EvidenceBundle(
            [Observation("fever", "history", raw_text="\u53d1\u70ed")]
        )

        decision = engine.rank(
            pool,
            evidence,
            llm_result={"diagnosis_candidates": [{"name": leukemia}]},
        )
        leukemia_score = next(
            item for item in decision.candidates if item.diagnosis == leukemia
        )

        self.assertEqual(leukemia_score.eligibility_status, DEFERRED)
        self.assertTrue(leukemia_score.unresolved_critical_evidence_claims)
        self.assertNotEqual(leukemia_score.eligibility_status, PRIMARY_ELIGIBLE)
        tasks = decision.judge_decision.get("discriminating_exam_tasks", [])
        self.assertTrue(
            any(
                task.get("exam_source")
                in {"evidence_claim_followup_exam", "deferred_gap_closure_exam"}
                and leukemia in task.get("target_candidates", [])
                for task in tasks
            )
        )

    def test_cross_system_claim_followup_does_not_block_locked_primary(self):
        judge = DiagnosisJudge(load_config())
        leukemia = candidate("\u767d\u8840\u75c5")
        leukemia.entity_id = "D000025"
        leukemia.core_matched_evidence = [
            "blast_present",
            "multilineage_cytopenia",
        ]
        leukemia.diagnostic_matched_evidence = ["acute_leukemia_pattern"]

        pavm = candidate("\u80ba\u52a8\u9759\u8109\u7618")
        pavm.entity_id = "D100055"
        pavm.eligibility_status = DEFERRED
        pavm.required_gaps = ["pulmonary_cta_positive"]
        pavm.unresolved_critical_evidence_claims = [
            {
                "claim_id": "claim_pavm_cta",
                "target_evidence": "pulmonary_cta_positive",
                "importance": "critical",
                "recommended_exam": "\u80ba\u52a8\u8109CTA",
                "confidence": 0.8,
            }
        ]
        pavm.claim_followup_exams = ["\u80ba\u52a8\u8109CTA"]
        pavm.core_matched_evidence = ["hemoptysis"]

        self.assertFalse(judge._high_value_unresolved_contender(leukemia, pavm))

    def test_shared_core_claim_followup_can_block_primary_lock(self):
        judge = DiagnosisJudge(load_config())
        coronary = candidate("\u51a0\u5fc3\u75c5")
        coronary.entity_id = "D000011"
        coronary.core_matched_evidence = ["hemoptysis"]

        pavm = candidate("\u80ba\u52a8\u9759\u8109\u7618")
        pavm.entity_id = "D100055"
        pavm.eligibility_status = DEFERRED
        pavm.required_gaps = ["pulmonary_cta_positive"]
        pavm.unresolved_critical_evidence_claims = [
            {
                "claim_id": "claim_pavm_cta",
                "target_evidence": "pulmonary_cta_positive",
                "importance": "critical",
                "recommended_exam": "\u80ba\u52a8\u8109CTA",
                "confidence": 0.8,
            }
        ]
        pavm.claim_followup_exams = ["\u80ba\u52a8\u8109CTA"]
        pavm.core_matched_evidence = ["hemoptysis"]

        self.assertTrue(judge._high_value_unresolved_contender(coronary, pavm))

    def test_high_value_deferred_gap_generates_priority_closure_exam(self):
        judge = DiagnosisJudge(load_config())
        lung_cancer = candidate("\u80ba\u764c")
        lung_cancer.entity_id = "D000006"
        lung_cancer.core_matched_evidence = ["hemoptysis"]
        lung_cancer.matched_evidence = ["hemoptysis", "pulmonary_nodule"]

        pavm = candidate("\u80ba\u52a8\u9759\u8109\u7618")
        pavm.entity_id = "D100055"
        pavm.eligibility_status = DEFERRED
        pavm.eligibility_substatus = DEFERRED_NEEDS_CONFIRMATORY_EXAM
        pavm.required_met = False
        pavm.required_gaps = ["pulmonary_cta_positive"]
        pavm.matched_evidence = ["hemoptysis"]
        pavm.core_matched_evidence = ["hemoptysis"]
        pavm.claim_followup_exams = ["\u80ba\u52a8\u8109CTA"]
        pavm.evidence_specificity_score = 0.8
        pavm.source_prior = 0.8

        decision = judge.judge([lung_cancer, pavm])

        self.assertEqual(decision.primary_status, "deferred")
        self.assertIn("\u80ba\u52a8\u9759\u8109\u7618", decision.high_value_gap_candidates)
        self.assertTrue(decision.exam_priority_overrides)
        self.assertTrue(decision.deferred_evidence_gaps)
        self.assertTrue(
            any(
                task.get("exam_source") == "deferred_gap_closure_exam"
                and task.get("priority_override")
                and "\u80ba\u52a8\u9759\u8109\u7618" in task.get("target_candidates", [])
                for task in decision.discriminating_exam_tasks
            )
        )

    def test_pavm_deferred_gap_tasks_prioritize_entity_confirmatory_bundle(self):
        judge = DiagnosisJudge(load_config(), knowledge=KnowledgeBase("data/ref_data"))
        pavm = candidate("\u80ba\u52a8\u9759\u8109\u7618")
        pavm.entity_id = "D100055"
        pavm.eligibility_status = DEFERRED
        pavm.eligibility_substatus = DEFERRED_NEEDS_CONFIRMATORY_EXAM
        pavm.required_met = False
        pavm.required_gaps = ["pulmonary_cta_positive"]
        pavm.matched_evidence = ["hemoptysis", "pulmonary_vascular_shunt"]
        pavm.core_matched_evidence = ["hemoptysis", "pulmonary_vascular_shunt"]
        pavm.evidence_specificity_score = 0.92
        pavm.source_prior = 0.85
        pavm.candidate_value = "high"

        judge._annotate_deferred_gap_priorities([pavm])
        tasks = judge._deferred_gap_closure_exam_tasks([pavm])

        self.assertTrue(tasks)
        self.assertLessEqual(len(tasks), 3)
        self.assertEqual(tasks[0]["exam"], "\u80ba\u52a8\u8109CTA")
        self.assertEqual(tasks[0]["priority_bucket"], "high_value_deferred_gap_closure")
        self.assertEqual(tasks[0]["closure_rank"], 1)
        self.assertEqual(tasks[0]["target_candidates"], ["\u80ba\u52a8\u9759\u8109\u7618"])
        self.assertTrue(
            any("\u58f0\u5b66\u9020\u5f71" in task["exam"] for task in tasks)
        )

    def test_low_value_deferred_gap_does_not_get_priority_override(self):
        judge = DiagnosisJudge(load_config())
        pulmonary_valve = candidate("\u80ba\u52a8\u8109\u74e3\u72ed\u7a84")
        pulmonary_valve.eligibility_status = DEFERRED
        pulmonary_valve.eligibility_substatus = DEFERRED_NEEDS_CONFIRMATORY_EXAM
        pulmonary_valve.required_met = False
        pulmonary_valve.required_gaps = ["pulmonary_valve_gradient"]
        pulmonary_valve.matched_evidence = ["diagnosis:\u80ba\u52a8\u8109\u74e3\u72ed\u7a84"]
        pulmonary_valve.core_matched_evidence = []
        pulmonary_valve.evidence_specificity_score = 0.2
        pulmonary_valve.diagnosis_type = "disease"

        judge._annotate_deferred_gap_priorities([pulmonary_valve])

        self.assertFalse(pulmonary_valve.exam_priority_override)
        self.assertTrue(pulmonary_valve.evidence_gaps)

    def test_broad_support_only_deferred_candidate_does_not_steal_exam_priority(self):
        judge = DiagnosisJudge(load_config())
        tb_pericarditis = candidate("\u7ed3\u6838\u6027\u5fc3\u5305\u708e")
        tb_pericarditis.eligibility_status = DEFERRED
        tb_pericarditis.eligibility_substatus = DEFERRED_NEEDS_CONFIRMATORY_EXAM
        tb_pericarditis.required_met = False
        tb_pericarditis.required_gaps = ["pericardial_effusion"]
        tb_pericarditis.matched_evidence = ["fever"]
        tb_pericarditis.core_matched_evidence = []
        tb_pericarditis.diagnostic_matched_evidence = []
        tb_pericarditis.claim_followup_exams = ["\u5fc3\u810f\u8d85\u58f0"]
        tb_pericarditis.evidence_specificity_score = 0.9

        judge._annotate_deferred_gap_priorities([tb_pericarditis])

        self.assertFalse(tb_pericarditis.exam_priority_override)
        self.assertTrue(tb_pericarditis.evidence_gaps)

    def test_deferred_cross_system_candidate_does_not_block_structural_primary(self):
        judge = DiagnosisJudge(load_config())
        mitral = candidate("\u4e8c\u5c16\u74e3\u53cd\u6d41")
        mitral.entity_id = "D100012"
        mitral.diagnosis_type = "structural"
        mitral.component_scores = {"objective_evidence": 1.0}
        mitral.core_matched_evidence = ["mitral_regurgitation"]
        mitral.diagnostic_matched_evidence = ["echo_mitral_regurgitation"]
        mitral.satisfied_required_anchors = ["echo_mitral_regurgitation"]

        esrd = candidate("\u7ec8\u672b\u671f\u80be\u75c5")
        esrd.entity_id = "D100004"
        esrd.eligibility_status = PRIMARY_ELIGIBLE
        esrd.differential_only = True
        esrd.score = mitral.score + 0.2
        esrd.required_gaps = []
        esrd.core_matched_evidence = ["heart_failure_state"]
        esrd.claim_followup_exams = ["\u80be\u529f\u80fd\u68c0\u67e5\uff08RFTs\uff09"]

        self.assertFalse(judge._high_value_unresolved_contender(mitral, esrd))

    def test_runtime_causal_manifestation_can_be_secondary(self):
        judge = DiagnosisJudge(load_config())
        mitral = candidate("\u4e8c\u5c16\u74e3\u53cd\u6d41")
        mitral.entity_id = "D100012"
        mitral.diagnosis_type = "structural"
        mitral.component_scores = {"objective_evidence": 1.0}
        mitral.core_matched_evidence = ["mitral_regurgitation"]
        mitral.diagnostic_matched_evidence = ["echo_mitral_regurgitation"]

        heart_failure = candidate("\u5fc3\u529b\u8870\u7aed")
        heart_failure.entity_id = "D000009"
        heart_failure.diagnosis_type = "state"
        heart_failure.causal_relation_to_selected = (
            "caused_by:\u4e8c\u5c16\u74e3\u53cd\u6d41"
        )
        heart_failure.matched_evidence = [
            "fluid_retention_pattern",
            "leg_edema",
            "paroxysmal_nocturnal_dyspnea",
        ]
        heart_failure.core_matched_evidence = [
            "fluid_retention_pattern",
            "leg_edema",
            "paroxysmal_nocturnal_dyspnea",
        ]

        secondary = judge._select_secondary(
            mitral,
            [mitral, heart_failure],
            max_final_diagnoses=3,
        )

        self.assertEqual([item.diagnosis for item in secondary], ["\u5fc3\u529b\u8870\u7aed"])


if __name__ == "__main__":
    unittest.main()
