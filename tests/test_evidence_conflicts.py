import unittest
from types import SimpleNamespace

from agent.clinical_evidence import EvidenceBundle, Observation
from agent.evidence_conflicts import EvidenceConflictArbiter


class EvidenceConflictArbiterTests(unittest.TestCase):
    def setUp(self):
        self.arbiter = EvidenceConflictArbiter()

    @staticmethod
    def low_magnesium_candidate():
        return SimpleNamespace(
            diagnosis="低镁血症",
            score=0.82,
            matched_evidence=[
                "magnesium_load_retention_high",
                "magnesium_depletion",
            ],
            core_matched_evidence=["magnesium_depletion"],
            diagnostic_matched_evidence=["magnesium_load_retention_high"],
            specificity=0.9,
            hard_contradiction=False,
            evidence_conflicts=[],
            unresolved_evidence_conflict=False,
            conflict_adjudication_exams=[],
        )

    @staticmethod
    def magnesium_evidence():
        return EvidenceBundle(
            [
                Observation(
                    "magnesium_load_retention_high",
                    "镁负荷试验",
                    polarity="positive",
                    confidence=0.96,
                    raw_text="镁负荷保留率 62%，参考＜20-30%",
                    evidence_level="diagnostic_pattern",
                    information_value=0.96,
                ),
                Observation(
                    "magnesium_depletion",
                    "镁负荷试验",
                    polarity="positive",
                    confidence=0.95,
                    raw_text="镁负荷保留率 62%，参考＜20-30%",
                    evidence_level="diagnostic_pattern",
                    information_value=0.95,
                ),
            ]
        )

    def test_reasoning_excludes_candidate_with_structured_support_creates_conflict(self):
        conflicts = self.arbiter.detect(
            {"reasoning": "镁负荷试验排除低镁血症，考虑其他代谢性骨病。"},
            self.magnesium_evidence(),
            [self.low_magnesium_candidate()],
        )
        self.assertEqual(len(conflicts), 1)
        conflict = conflicts[0]
        self.assertEqual(
            conflict["conflict_type"],
            "reasoning_structured_polarity_conflict",
        )
        self.assertEqual(conflict["affected_diagnosis"], "低镁血症")
        self.assertEqual(conflict["action"], "defer_primary_and_order_discriminating_exams")
        self.assertIn(conflict["finding"], {"magnesium_load_retention_high", "magnesium_depletion"})
        self.assertIn("血清电解质", conflict["adjudication_exams"])
        self.assertTrue(conflict["structured_sources"])

    def test_supportive_reasoning_does_not_create_conflict(self):
        conflicts = self.arbiter.detect(
            {"reasoning": "镁负荷试验支持低镁血症，需复查电解质。"},
            self.magnesium_evidence(),
            [self.low_magnesium_candidate()],
        )
        self.assertEqual(conflicts, [])

    def test_other_candidates_lack_evidence_does_not_exclude_supported_candidate(self):
        conflicts = self.arbiter.detect(
            {
                "reasoning": (
                    "其他候选如甲状腺功能亢进、冠心病、贫血等虽有部分症状重叠，"
                    "但缺乏相应特异性证据，且所有症状和检查异常均可被低镁血症统一解释。"
                )
            },
            self.magnesium_evidence(),
            [self.low_magnesium_candidate()],
        )
        self.assertEqual(conflicts, [])

    def test_low_information_diagnosis_finding_is_not_blocking_support(self):
        candidate = self.low_magnesium_candidate()
        candidate.matched_evidence = ["diagnosis:心律失常"]
        candidate.core_matched_evidence = []
        candidate.diagnostic_matched_evidence = []
        evidence = EvidenceBundle(
            [
                Observation(
                    "diagnosis:心律失常",
                    "心电图（ECG）",
                    polarity="positive",
                    confidence=0.98,
                    raw_text="频发房性期前收缩",
                    evidence_level="supportive",
                    information_value=0.55,
                )
            ]
        )
        conflicts = self.arbiter.detect(
            {"reasoning": "镁负荷试验排除低镁血症，考虑其他病因。"},
            evidence,
            [candidate],
        )
        self.assertEqual(conflicts, [])

    def test_rule_out_uncertainty_does_not_create_blocking_conflict(self):
        conflicts = self.arbiter.detect(
            {"reasoning": "目前不能排除低镁血症，建议鉴别并复查。"},
            self.magnesium_evidence(),
            [self.low_magnesium_candidate()],
        )
        self.assertEqual(conflicts, [])


if __name__ == "__main__":
    unittest.main()
