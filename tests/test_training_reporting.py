import copy
import time
import unittest
from unittest.mock import Mock

import yaml

from agent.agent import MyDoctorAgent
from agent.knowledge import KnowledgeBase
from agent.qc import QualityAgent
from hospital_agent.base import summarize_training_results


class TrainingReportingTests(unittest.TestCase):
    def test_batch_summary_separates_evaluation_and_reflection_errors(self):
        rows = [
            {
                "status": "evaluated",
                "metrics": {
                    "diagnosis_accuracy": 1.0,
                    "examination_precision": 0.8,
                    "treatment_overall_score": 0.7,
                    "treatment_safety": 1.0,
                    "candidate_recall_at_20": True,
                    "candidate_recall_at_5": True,
                    "ranking_accuracy": True,
                    "submission_alignment": True,
                    "submission_override_count": 0,
                    "etiology_preference": True,
                    "decision_override_rate": False,
                    "judge_gap_authorization_rate": True,
                    "judge_primary_accuracy": True,
                    "differential_exam_precision": 0.5,
                    "discriminating_gap_closed_rate": 0.25,
                    "dynamic_rerank_changed_primary": True,
                    "pairwise_judge_accuracy": 1.0,
                    "unauthorized_exam_count": 2,
                    "required_evidence_coverage": 0.75,
                    "soft_contradiction_count": 1,
                    "hard_contradiction_count": 0,
                    "reasoning_inference_finding_count": 2,
                    "raw_case_finding_count": 3,
                    "reasoning_inference_used_by_primary": True,
                    "blocked_reasoning_inference_count": 1,
                    "required_gap_authorized_count": 1,
                    "explanatory_coverage": 0.82,
                    "core_explanatory_coverage": 0.76,
                    "residual_evidence_score": 0.18,
                    "residual_core_evidence_count": 1,
                    "discriminating_exam_recall": 0.5,
                    "exam_information_gain": 0.45,
                    "gap_value_exam_selection_rate": 0.9,
                    "reserved_highest_gap_survival_rate": 1.0,
                    "special_discriminator_rate": 0.75,
                    "multi_candidate_exam_rate": 0.8,
                    "generic_exam_suppression_count": 3,
                    "post_exam_primary_recomputed_rate": True,
                    "gap_closure_rate": 0.25,
                    "explanation_score_changed_ranking_rate": True,
                    "primary_unlock_rate": True,
                    "legacy_exam_package_contribution_rate": 0.2,
                    "differential_exam_contribution_rate": 0.8,
                    "gap_state_satisfied_count": 2,
                    "gap_state_actionable_count": 1,
                    "gap_state_nonblocking_count": 1,
                    "gap_state_unsupported_count": 3,
                    "gap_state_hard_blocked_count": 0,
                    "generic_primary_block_count": 2,
                    "specific_over_generic_preference_count": 1,
                    "core_evidence_primary_alignment": True,
                    "diagnostic_evidence_primary_alignment": False,
                    "residual_core_penalty_applied_count": 3,
                },
                "audit": {
                    "elapsed_seconds": 100,
                    "timed_out": False,
                    "critic_issues": ["narrow_margin:0.1"],
                    "critic_llm_used": True,
                },
                "evaluation_error": "",
                "reflection_error": "memory write failed",
            },
            {
                "status": "evaluation_failed",
                "metrics": {},
                "audit": {
                    "elapsed_seconds": 235,
                    "timed_out": True,
                    "critic_issues": [],
                    "critic_llm_used": False,
                },
                "evaluation_error": "backend 503",
                "reflection_error": "",
            },
        ]
        summary = summarize_training_results(rows)
        self.assertEqual(summary["diagnosis_accuracy"], 1.0)
        self.assertEqual(summary["candidate_recall_at_20"], 1.0)
        self.assertEqual(summary["candidate_recall_at_5"], 1.0)
        self.assertEqual(summary["ranking_accuracy"], 1.0)
        self.assertEqual(summary["submission_alignment"], 1.0)
        self.assertEqual(summary["submission_override_count"], 0.0)
        self.assertEqual(summary["etiology_preference"], 1.0)
        self.assertEqual(summary["decision_override_rate"], 0.0)
        self.assertEqual(summary["judge_gap_authorization_rate"], 1.0)
        self.assertEqual(summary["judge_primary_accuracy"], 1.0)
        self.assertEqual(summary["differential_exam_precision"], 0.5)
        self.assertEqual(summary["discriminating_gap_closed_rate"], 0.25)
        self.assertEqual(summary["dynamic_rerank_changed_primary"], 1.0)
        self.assertEqual(summary["pairwise_judge_accuracy"], 1.0)
        self.assertEqual(summary["unauthorized_exam_count"], 2.0)
        self.assertEqual(summary["required_evidence_coverage"], 0.75)
        self.assertEqual(summary["soft_contradiction_count"], 1.0)
        self.assertEqual(summary["hard_contradiction_count"], 0.0)
        self.assertEqual(summary["reasoning_inference_finding_count"], 2.0)
        self.assertEqual(summary["raw_case_finding_count"], 3.0)
        self.assertEqual(summary["reasoning_inference_used_by_primary"], 1.0)
        self.assertEqual(summary["blocked_reasoning_inference_count"], 1.0)
        self.assertEqual(summary["required_gap_authorized_count"], 1.0)
        self.assertEqual(summary["explanatory_coverage"], 0.82)
        self.assertEqual(summary["core_explanatory_coverage"], 0.76)
        self.assertEqual(summary["residual_evidence_score"], 0.18)
        self.assertEqual(summary["residual_core_evidence_count"], 1.0)
        self.assertEqual(summary["discriminating_exam_recall"], 0.5)
        self.assertEqual(summary["exam_information_gain"], 0.45)
        self.assertEqual(summary["gap_value_exam_selection_rate"], 0.9)
        self.assertEqual(summary["reserved_highest_gap_survival_rate"], 1.0)
        self.assertEqual(summary["special_discriminator_rate"], 0.75)
        self.assertEqual(summary["multi_candidate_exam_rate"], 0.8)
        self.assertEqual(summary["generic_exam_suppression_count"], 3.0)
        self.assertEqual(summary["post_exam_primary_recomputed_rate"], 1.0)
        self.assertEqual(summary["gap_closure_rate"], 0.25)
        self.assertEqual(summary["explanation_score_changed_ranking_rate"], 1.0)
        self.assertEqual(summary["primary_unlock_rate"], 1.0)
        self.assertEqual(summary["legacy_exam_package_contribution_rate"], 0.2)
        self.assertEqual(summary["differential_exam_contribution_rate"], 0.8)
        self.assertEqual(summary["gap_state_satisfied_count"], 2.0)
        self.assertEqual(summary["gap_state_actionable_count"], 1.0)
        self.assertEqual(summary["gap_state_nonblocking_count"], 1.0)
        self.assertEqual(summary["gap_state_unsupported_count"], 3.0)
        self.assertEqual(summary["gap_state_hard_blocked_count"], 0.0)
        self.assertEqual(summary["generic_primary_block_count"], 2.0)
        self.assertEqual(summary["specific_over_generic_preference_count"], 1.0)
        self.assertEqual(summary["core_evidence_primary_alignment"], 1.0)
        self.assertEqual(summary["diagnostic_evidence_primary_alignment"], 0.0)
        self.assertEqual(summary["residual_core_penalty_applied_count"], 3.0)
        self.assertEqual(summary["critic_issue_rate"], 0.5)
        self.assertEqual(summary["critic_llm_rate"], 0.5)
        self.assertEqual(summary["timeout_cases"], 1)
        self.assertEqual(summary["backend_error_cases"], 1)
        self.assertEqual(summary["reflection_error_cases"], 1)

    def test_exam_alias_feedback_stays_pending_when_auto_promotion_is_disabled(self):
        knowledge = KnowledgeBase(
            ref_dir="data/ref_data",
            allow_auto_alias_promotion=False,
        )
        knowledge._read_json_file = Mock(return_value={"candidates": []})
        knowledge._write_json_file = Mock()
        knowledge._infer_exam_alias_standard = Mock(return_value="尿液分析（UA）")
        knowledge._promote_pending_exam_aliases = Mock(return_value=1)
        report = {
            "diagnosisAccuracy": 1.0,
            "examinationPrecision": 1.0,
            "treatmentOverallScore": 1.0,
            "examinationDetail": {
                "expected": ["尿液分析（UA）"],
                "ordered": ["UA-new-alias"],
            },
        }
        result = knowledge.record_exam_alias_feedback(
            "case-alias", report, ["UA-new-alias"]
        )
        self.assertEqual(result, {"pending": 1, "promoted": 0})
        knowledge._promote_pending_exam_aliases.assert_not_called()

    def test_training_record_contains_replay_metrics_without_internal_payload(self):
        with open("config.yaml", "r", encoding="utf-8") as handle:
            config = copy.deepcopy(yaml.safe_load(handle))
        config["self_improve_enabled"] = False
        config["memory"]["json_path"] = "tests/_training_report_memory.json"
        config["memory"]["md_path"] = "tests/_training_report_memory.md"
        agent = MyDoctorAgent(config)
        agent._last_diagnosis_audit = {
            "diagnosis_decision": {
                "candidates": [
                    {
                        "diagnosis": "低镁血症",
                        "required_met": True,
                        "matched_evidence": ["low_magnesium"],
                        "required_gaps": [],
                        "soft_contradicted_evidence": ["urine_culture_no_growth"],
                        "hard_contradicted_evidence": [],
                    },
                    {"diagnosis": "心律失常"},
                ],
                "retriever_top1": "心律失常",
                "judge_primary": "低镁血症",
                "submitter_final": ["低镁血症"],
                "decision_override": True,
                "required_gap_authorized_diagnoses": [],
                "judge_decision": {
                    "pairwise_comparisons": [
                        {
                            "left": "低镁血症",
                            "right": "心律失常",
                            "preferred": "低镁血症",
                        }
                    ],
                    "discriminating_exams": ["综合代谢面板（CMP）"],
                    "discriminating_findings": ["low_magnesium"],
                    "dynamic_rerank_changed_primary": True,
                    "explanatory_coverage": 0.84,
                    "core_explanatory_coverage": 0.8,
                    "residual_evidence_score": 0.16,
                    "residual_core_evidence_count": 0,
                    "high_value_gap_candidates": [],
                },
                "pattern_recall_audit": {
                    "compiler_enabled": True,
                    "pattern_pipeline_audit": {
                        "proposal_count_by_source": {
                            "deterministic_relation": 1,
                        },
                        "candidate_admissions": [
                            {
                                "entity_id": "D100058",
                                "recall_mode": "recall_boost",
                            }
                        ],
                    },
                },
            },
            "evidence": {
                "observations": [
                    {
                        "finding": "low_magnesium",
                        "polarity": "positive",
                        "source": "电解质",
                    },
                    {
                        "finding": "field:血镁",
                        "polarity": "positive",
                        "source": "电解质",
                    },
                ]
            },
            "critic": {"issues": [], "llm_used": False},
            "elapsed_seconds": 88.5,
            "timed_out": False,
        }
        agent._last_exam_authorization = [
            {
                "strict_diagnosis_driven": True,
                "blocked_items": ["心脏MRI（CMR）"],
            }
        ]
        report = {
            "diagnosisAccuracy": 1.0,
            "examinationPrecision": 0.9,
            "treatmentOverallScore": 0.8,
            "treatmentSafety": 1.0,
            "diagnosisDetail": {
                "expected": ["低镁血症"],
                "submitted": ["低镁血症"],
            },
        }
        result = agent._build_training_result(
            "case-report",
            {
                "patient_id": "case-report",
                "diagnosis": ["低镁血症"],
                "treatment_plan": "补镁并监测。",
                "reasoning": "低血镁。",
                "_private": "must not leak",
            },
            report,
        )
        self.assertTrue(result["metrics"]["candidate_recall_at_5"])
        self.assertTrue(result["metrics"]["candidate_recall_at_20"])
        self.assertTrue(result["metrics"]["ranking_accuracy"])
        self.assertEqual(result["audit"]["elapsed_seconds"], 88.5)
        self.assertEqual(result["metrics"]["required_evidence_coverage"], 1.0)
        self.assertEqual(result["metrics"]["soft_contradiction_count"], 1)
        self.assertEqual(result["metrics"]["hard_contradiction_count"], 0)
        self.assertEqual(result["metrics"]["reasoning_structured_conflict_count"], 0)
        self.assertEqual(result["metrics"]["conflict_deferred_primary_count"], 0)
        self.assertEqual(result["metrics"]["conflict_blocked_final_count"], 0)
        self.assertEqual(result["metrics"]["root_cause_arbitration_count"], 0)
        self.assertEqual(result["metrics"]["root_cause_primary_override_count"], 0)
        self.assertEqual(result["metrics"]["root_cause_secondary_submission_count"], 0)
        self.assertEqual(result["metrics"]["root_cause_coverage"], 0.0)
        self.assertEqual(result["retriever_top1"], "心律失常")
        self.assertEqual(result["judge_primary"], "低镁血症")
        self.assertEqual(result["submitter_final"], ["低镁血症"])
        self.assertTrue(result["metrics"]["decision_override_rate"])
        self.assertTrue(result["metrics"]["judge_primary_accuracy"])
        self.assertEqual(result["metrics"]["pairwise_judge_accuracy"], 1.0)
        self.assertTrue(result["metrics"]["dynamic_rerank_changed_primary"])
        self.assertEqual(result["metrics"]["explanatory_coverage"], 0.84)
        self.assertEqual(result["metrics"]["core_explanatory_coverage"], 0.8)
        self.assertEqual(result["metrics"]["residual_evidence_score"], 0.16)
        self.assertEqual(result["metrics"]["residual_core_evidence_count"], 0)
        self.assertIsNone(result["metrics"]["discriminating_exam_recall"])
        self.assertIsNone(result["metrics"]["exam_information_gain"])
        self.assertEqual(result["metrics"]["gap_closure_rate"], 1.0)
        self.assertEqual(
            result["audit"]["exam_authorization_mode"],
            "strict_diagnosis_driven",
        )
        self.assertTrue(result["audit"]["pattern_recall_audit"]["compiler_enabled"])
        self.assertEqual(
            result["audit"]["pattern_pipeline_audit"]["proposal_count_by_source"][
                "deterministic_relation"
            ],
            1,
        )
        self.assertNotIn("pattern_pipeline_audit", result["final_result"])
        self.assertIn(
            "low_magnesium",
            result["audit"]["finding_extraction_summary"]["diagnostic_findings"],
        )
        self.assertNotIn("_private", result["final_result"])

    def test_quality_review_respects_authorization_lock(self):
        knowledge = KnowledgeBase("data/ref_data")
        quality = QualityAgent(
            knowledge,
            allowed_diagnoses=["克里格勒-纳贾尔综合征", "肺炎"],
        )
        fixed = quality.review_final_result(
            {
                "diagnosis": ["肺炎"],
                "_trusted_diagnoses": ["肺炎"],
                "_authorized_diagnoses": ["克里格勒-纳贾尔综合征"],
                "_authorization_locked": True,
                "treatment_plan": "按授权诊断制定治疗。",
                "reasoning": "授权诊断已确定。",
            }
        )
        self.assertEqual(fixed["diagnosis"], ["克里格勒-纳贾尔综合征"])


class TrainingBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_clinical_deadline_preserves_training_post_submit_budget(self):
        with open("config.yaml", "r", encoding="utf-8") as handle:
            config = copy.deepcopy(yaml.safe_load(handle))
        config["self_improve_enabled"] = False
        config["execution"]["case_timeout_seconds"] = 10
        config["execution"]["fallback_reserve_seconds"] = 2
        config["memory"]["json_path"] = "tests/_training_budget_memory.json"
        config["memory"]["md_path"] = "tests/_training_budget_memory.md"
        agent = MyDoctorAgent(config)

        async def completed_pipeline(_patient_id):
            return {"diagnosis": ["低镁血症"], "finished": True}

        agent._execute_with_planner = completed_pipeline
        before = time.monotonic()
        await agent._run_case_pipeline(
            "case-budget",
            post_submit_reserve_seconds=4,
        )
        self.assertGreaterEqual(agent._case_deadline, before + 9.9)
        self.assertAlmostEqual(
            agent._case_deadline - agent._case_clinical_deadline,
            4.0,
            places=2,
        )
        await agent._cleanup()


if __name__ == "__main__":
    unittest.main()
