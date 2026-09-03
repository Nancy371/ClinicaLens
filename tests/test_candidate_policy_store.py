import tempfile
import unittest

from agent.candidate_policy_store import (
    CandidatePolicyStore,
    RuleGeneralizer,
    normalize_policy_candidate,
    promotion_decision,
)
from agent.critic import DefectDetector
from agent.replay import DiagnosticReplay


class DefectAttributionTests(unittest.TestCase):
    def test_detector_outputs_failure_attribution_not_suggested_fix(self):
        detector = DefectDetector()
        defects = detector.detect(
            report={"diagnosisAccuracy": 0.0, "diagnosis": "wrong"},
            collected_info={"symptoms": ["fatigue"]},
            exam_results={},
        )
        self.assertTrue(defects)
        first = defects[0]
        self.assertIn(first["failure_stage"], {
            "candidate_recall",
            "eligibility",
            "ranking",
            "exam_selection",
            "submission",
            "evidence_mapping",
        })
        self.assertIn("failure_type", first)
        self.assertIn("root_cause", first)
        self.assertIn("generalizable_pattern", first)
        self.assertNotIn("suggested_fix", first)


class CandidatePolicyStoreTests(unittest.TestCase):
    def make_store(self):
        temp = tempfile.NamedTemporaryFile(delete=True)
        temp.close()
        return CandidatePolicyStore(temp.name)

    def test_single_failure_creates_candidate_without_promotion(self):
        store = self.make_store()
        generalizer = RuleGeneralizer()
        policies = generalizer.generalize(
            [
                {
                    "failure_stage": "eligibility",
                    "failure_type": "required_evidence_not_closed",
                    "affected_candidate": "low magnesium",
                    "root_cause": "dynamic test interpreted as strong evidence",
                    "generalizable_pattern": "ambiguous dynamic tests need anchor review",
                    "evidence_refs": ["magnesium_load_retention_high"],
                }
            ],
            source_case="Patient_08970",
        )
        result = store.upsert_many(policies)
        self.assertEqual(result["candidate"], 1)
        policy = store.policies[0]
        self.assertEqual(policy["status"], "candidate")
        self.assertFalse(policy["promotion_allowed"])
        self.assertEqual(policy["source_cases"], ["Patient_08970"])
        self.assertEqual(policy["target_layer"], "eligibility")
        self.assertNotIn("boost_disease", str(policy["action"]))

    def test_case_hotfix_is_temporary_and_source_scoped(self):
        store = self.make_store()
        policy = RuleGeneralizer().generalize(
            [
                {
                    "policy_type": "case_hotfix",
                    "failure_stage": "submission",
                    "failure_type": "temporary_backend_name_patch",
                    "source_cases": ["Patient_08970"],
                }
            ]
        )[0]
        saved = store.upsert_candidate(policy)
        self.assertEqual(saved["status"], "temporary")
        self.assertEqual(saved["source_cases"], ["Patient_08970"])
        self.assertEqual(saved["expires_after_days"], 30)

    def test_invalid_target_layer_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_policy_candidate(
                {
                    "policy_id": "POLICY_BAD",
                    "policy_type": "general_rule",
                    "target_layer": "disease_score",
                    "trigger_conditions": ["x"],
                    "action": {"boost_disease": "rickets"},
                    "applicable_scope": ["general"],
                    "excluded_scope": [],
                    "source_cases": ["case"],
                    "status": "candidate",
                }
            )

    def test_conflicting_policy_is_quarantined(self):
        store = self.make_store()
        base = {
            "policy_id": "POLICY_A",
            "policy_type": "general_rule",
            "target_layer": "submission",
            "trigger_conditions": ["missing anchor"],
            "action": {"block_final_when_ineligible": True},
            "applicable_scope": ["general"],
            "excluded_scope": ["single case"],
            "source_cases": ["case-a"],
            "status": "candidate",
        }
        conflict = {
            **base,
            "policy_id": "POLICY_B",
            "action": {"allow_final": True},
            "source_cases": ["case-b"],
        }
        store.upsert_candidate(base)
        saved = store.upsert_candidate(conflict)
        self.assertEqual(saved["status"], "quarantined")
        self.assertEqual(saved["conflict"]["conflict_type"], "overlapping_trigger_opposite_action")


class PromotionGateTests(unittest.TestCase):
    def test_five_gate_promotion_passes_only_when_all_thresholds_pass(self):
        passing = promotion_decision(
            {
                "target_fix_rate": 0.95,
                "neighboring_accuracy_delta": 0.0,
                "false_positive_increase": 0.0,
                "global_accuracy_delta": 0.0,
                "unsafe_submission_delta": 0.0,
            }
        )
        self.assertTrue(passing.promote_allowed)
        failing = promotion_decision(
            {
                "target_fix_rate": 1.0,
                "neighboring_accuracy_delta": 0.0,
                "false_positive_increase": 0.02,
                "global_accuracy_delta": 0.0,
                "unsafe_submission_delta": 0.0,
            }
        )
        self.assertFalse(failing.promote_allowed)
        self.assertIn("false_positive_increase", failing.failed_gates)

    def test_replay_bucket_summary_blocks_counterexample_regression(self):
        summary = DiagnosticReplay.policy_promotion_summary(
            [
                {"bucket": "target", "baseline_correct": False, "candidate_correct": True},
                {"bucket": "same_pattern_positive", "baseline_correct": True, "candidate_correct": True},
                {"bucket": "neighbor", "baseline_correct": True, "candidate_correct": True},
                {
                    "bucket": "counterexample",
                    "baseline_correct": True,
                    "candidate_correct": True,
                    "baseline_false_positives": 0,
                    "candidate_false_positives": 1,
                },
                {"bucket": "historical_stable", "baseline_correct": True, "candidate_correct": True},
            ]
        )
        self.assertFalse(summary["should_promote"])
        self.assertIn("false_positive_increase", summary["failed_gates"])


class LegacyPromotionTests(unittest.TestCase):
    def test_simple_gain_replay_no_longer_promotes_pending_rule(self):
        from agent.diagnostic_learning import DiagnosticLearningStore

        temp = tempfile.NamedTemporaryFile(delete=True)
        temp.close()
        store = DiagnosticLearningStore(temp.name)
        store._save(
            {
                "schema_version": 2,
                "candidates": [
                    {
                        "id": "candidate",
                        "status": "shadow",
                        "diagnosis": "x",
                        "support_cases": [],
                    }
                ],
            }
        )
        result = store.record_replay("candidate", {"a": 0.2, "b": 0.2, "c": 0.2})
        self.assertEqual(result["status"], "shadow")
        self.assertIn("target_fix_rate", result["promotion_decision"]["failed_gates"])
