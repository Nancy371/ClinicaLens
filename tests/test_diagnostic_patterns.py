import unittest
from types import SimpleNamespace

from agent.clinical_evidence import EvidenceBundle, Observation
from agent.diagnostic_patterns import DiagnosticPatternEvaluator


class _Knowledge:
    def __init__(self, entry):
        self.entry = entry

    def get(self, _name):
        return self.entry


def candidate(*findings):
    return SimpleNamespace(
        diagnosis="test disease",
        matched_evidence=list(findings),
        contradicted_evidence=[],
        soft_contradicted_evidence=[],
        hard_contradicted_evidence=[],
        evidence_contributions=[],
    )


class DiagnosticPatternEvaluatorTests(unittest.TestCase):
    def test_all_of_any_of_and_min_count_pattern_matches(self):
        evaluator = DiagnosticPatternEvaluator(
            _Knowledge(
                {
                    "diagnostic_patterns": [
                        {
                            "pattern_id": "combo_anchor",
                            "pattern_type": "anchor_pattern",
                            "logic": "all_of",
                            "required": [
                                {"any_of": ["a", "b"]},
                                {"min_count": 2, "of": ["c", "d", "e"]},
                            ],
                            "effect": {"eligibility": "PrimaryEligible"},
                        }
                    ]
                }
            )
        )

        result = evaluator.evaluate(candidate("a", "c", "d"))

        self.assertEqual(len(result["primary_eligible_matches"]), 1)
        match = result["primary_eligible_matches"][0]
        self.assertEqual(match["pattern_id"], "combo_anchor")
        self.assertEqual(len(match["matched_required_groups"]), 2)

    def test_not_any_of_prevents_pattern_match(self):
        evaluator = DiagnosticPatternEvaluator(
            _Knowledge(
                {
                    "diagnostic_patterns": [
                        {
                            "pattern_id": "blocked_anchor",
                            "pattern_type": "anchor_pattern",
                            "logic": "all_of",
                            "required": ["a", "b"],
                            "not_any_of": ["contradictor"],
                            "effect": {"eligibility": "PrimaryEligible"},
                        }
                    ]
                }
            )
        )

        result = evaluator.evaluate(candidate("a", "b", "contradictor"))

        self.assertEqual(result["primary_eligible_matches"], [])
        self.assertIn("contradictor", result["negative_hits"])

    def test_reasoning_inference_does_not_satisfy_objective_pattern(self):
        evaluator = DiagnosticPatternEvaluator(
            _Knowledge(
                {
                    "diagnostic_patterns": [
                        {
                            "pattern_id": "objective_anchor",
                            "pattern_type": "anchor_pattern",
                            "logic": "all_of",
                            "required": ["a", "b"],
                            "requires_objective_source": True,
                            "effect": {"eligibility": "PrimaryEligible"},
                        }
                    ]
                }
            )
        )
        evidence = EvidenceBundle(
            [
                Observation("a", "reasoning_inference", confidence=0.7),
                Observation("b", "reasoning_inference", confidence=0.7),
            ]
        )

        result = evaluator.evaluate(candidate("a", "b"), evidence=evidence)

        self.assertEqual(result["primary_eligible_matches"], [])
        self.assertFalse(result["missing_primary_patterns"][0]["objective_source_satisfied"])


if __name__ == "__main__":
    unittest.main()
