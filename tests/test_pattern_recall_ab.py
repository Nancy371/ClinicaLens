import unittest

from tools.compare_pattern_recall_ab import compare_runs


class PatternRecallABTests(unittest.TestCase):
    def test_compare_runs_marks_candidate_rescue(self):
        baseline = [
            {
                "patient_id": "Patient_03674",
                "evaluation_report": {
                    "diagnosisDetail": {
                        "expected": ["放射性肺炎"],
                        "submitted": ["支气管肺炎"],
                    }
                },
                "result": {
                    "_diagnosis_decision": {
                        "candidates": [
                            {"diagnosis": "支气管肺炎"},
                            {"diagnosis": "肺不张"},
                        ]
                    }
                },
            }
        ]
        experiment = [
            {
                "patient_id": "Patient_03674",
                "evaluation_report": {
                    "diagnosisDetail": {
                        "expected": ["放射性肺炎"],
                        "submitted": ["支气管肺炎"],
                    }
                },
                "result": {
                    "_diagnosis_decision": {
                        "candidates": [
                            {"diagnosis": "支气管肺炎"},
                            {"diagnosis": "放射性肺炎"},
                        ],
                        "pattern_recall_audit": {
                            "proposal_count": 1,
                            "verification_statuses": {"verified": 1},
                            "entity_link_count": 1,
                            "linked_entity_ids": ["D100058"],
                            "signal_count": 1,
                            "signal_modes": {"recall_boost": 1},
                            "signal_entity_ids": ["D100058"],
                        },
                        "pattern_candidate_admissions": [
                            {
                                "entity_id": "D100058",
                                "admitted_to_controlled_pool": True,
                            }
                        ],
                    }
                },
            }
        ]

        report = compare_runs(baseline, experiment)

        self.assertEqual(report["summary"]["candidate_rescue_count"], 1)
        case = report["cases"][0]
        self.assertTrue(case["candidate_rescue"])
        self.assertEqual(case["experiment"]["pattern_stages"]["proposal_count"], 1)
        self.assertEqual(case["experiment"]["pattern_stages"]["controlled_admission_count"], 1)

    def test_timeout_run_is_not_true_pattern_harm(self):
        baseline = [
            {
                "patient_id": "Patient_01640",
                "expected": ["二尖瓣反流"],
                "result": {
                    "_diagnosis_decision": {
                        "candidates": [{"diagnosis": "二尖瓣反流"}],
                    },
                    "diagnosis": ["二尖瓣反流"],
                },
                "thinking_success": True,
                "diagnosis_llm_success": True,
            }
        ]
        experiment = [
            {
                "patient_id": "Patient_01640",
                "expected": ["二尖瓣反流"],
                "result": {
                    "_diagnosis_decision": {
                        "candidates": [{"diagnosis": "肺不张"}],
                        "pattern_recall_audit": {
                            "proposal_count": 0,
                            "compiler_audit": {
                                "sources": {
                                    "structured_thinking": {
                                        "input_present": False,
                                        "generated": 0,
                                        "skip_reason": "thinking_timeout",
                                    }
                                }
                            },
                        },
                    },
                    "diagnosis": ["肺不张"],
                    "fallback_used": True,
                },
                "thinking_timeout": True,
                "fallback_used": True,
            }
        ]

        report = compare_runs(baseline, experiment)

        case = report["cases"][0]
        self.assertTrue(case["raw_outcome_harm"])
        self.assertFalse(case["pattern_harm"])
        self.assertEqual(case["harm_attribution"], "pipeline_timeout_harm")
        self.assertEqual(report["summary"]["pattern_harm_count"], 0)


if __name__ == "__main__":
    unittest.main()
