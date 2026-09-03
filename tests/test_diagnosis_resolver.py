import unittest

import yaml

from agent.clinical_evidence import ClinicalEvidenceNormalizer
from agent.diagnosis_engine import DiagnosisDecisionEngine
from agent.prompt import DoctorPrompt


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class OpenWorldDiagnosisResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.engine = DiagnosisDecisionEngine(cls.config, "data/ref_data")
        cls.resolver = cls.engine.resolver
        cls.normalizer = ClinicalEvidenceNormalizer("data/ref_data")

    def test_qualified_alias_maps_to_controlled_specific_name(self):
        result = self.resolver.resolve("考虑：肺隐球菌感染")
        self.assertEqual(result.canonical_name, "肺隐球菌病")
        self.assertEqual(result.parent_name, "肺炎")
        self.assertEqual(result.method, "exact_or_alias")

    def test_specific_modifier_maps_to_official_parent(self):
        result = self.resolver.resolve("细菌性肺炎")
        self.assertEqual(result.canonical_name, "肺炎")
        self.assertEqual(result.method, "hierarchy")

    def test_complex_congenital_heart_phrase_maps_to_specific_vsd(self):
        result = self.resolver.resolve("大型室间隔缺损伴艾森门格综合征早期表现")
        self.assertEqual(result.canonical_name, "室间隔缺损（VSD）")
        self.assertEqual(result.parent_name, "先天性心脏病")
        self.assertEqual(result.method, "alias_contains")

    def test_fuzzy_typo_maps_only_when_unambiguous(self):
        result = self.resolver.resolve("慢性阻塞性肺病")
        self.assertEqual(result.canonical_name, "慢性阻塞性肺疾病")
        self.assertEqual(result.method, "fuzzy")
        self.assertGreaterEqual(result.confidence, 0.84)

    def test_result_extracts_open_candidates_and_confidence(self):
        resolutions = self.resolver.resolve_result(
            {
                "diagnosis": ["肺炎"],
                "diagnosis_candidates": [
                    {"name": "肺炎支原体肺炎", "confidence": 0.86},
                    {"name": "急性支气管炎", "confidence": 0.5},
                ],
            }
        )
        by_name = {item.canonical_name: item for item in resolutions if item.canonical_name}
        self.assertIn("支原体肺炎", by_name)
        self.assertEqual(by_name["支原体肺炎"].model_confidence, 0.86)
        self.assertIn("急性支气管炎", by_name)

    def test_engine_audits_resolved_name_not_raw_alias_without_evidence(self):
        evidence = self.normalizer.normalize({}, {})
        decision = self.engine.decide(
            {"diagnosis": ["考虑肺隐球菌感染"]},
            [],
            evidence,
        )
        self.assertEqual(decision.final_diagnoses, [])
        self.assertNotIn("考虑肺隐球菌感染", decision.final_diagnoses)
        self.assertEqual(decision.name_resolutions[0]["canonical_name"], "肺隐球菌病")
        candidate = next(item for item in decision.candidates if item.diagnosis == "肺隐球菌病")
        self.assertGreater(candidate.source_prior, 0)
        self.assertFalse(candidate.matched_evidence)

    def test_unknown_candidate_is_audited_but_never_submitted(self):
        evidence = self.normalizer.normalize({}, {})
        decision = self.engine.decide(
            {"diagnosis": ["不存在的星际肺病"]},
            [],
            evidence,
        )
        self.assertEqual(decision.final_diagnoses, [])
        self.assertEqual(decision.unresolved_candidates, ["不存在的星际肺病"])
        self.assertEqual(decision.open_world_candidates[0]["raw_name"], "不存在的星际肺病")
        self.assertFalse(decision.open_world_candidates[0]["submittable"])

    def test_prompt_allows_open_candidates_but_keeps_closed_submission(self):
        prompt = DoctorPrompt().build_diagnosis_prompt(
            collected_info={},
            exam_results={},
            chat_history=[],
            standard_diseases=["肺炎"],
        )
        self.assertIn("候选诊断可以使用比目录更具体", prompt)
        self.assertIn("无法可靠映射的候选只进入审计", prompt)
        self.assertIn("diagnosis_candidates", prompt)


if __name__ == "__main__":
    unittest.main()
