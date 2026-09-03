import unittest

import yaml

from agent.clinical_evidence import EvidenceBundle, Observation
from agent.diagnosis_eligibility import PRIMARY_ELIGIBLE
from agent.diagnosis_engine import CandidateScore, DiagnosisDecision, DiagnosisDecisionEngine
from agent.exam_strategy import ExamStrategyAgent
from agent.knowledge import KnowledgeBase


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class CanonicalDiseaseEntityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.engine = DiagnosisDecisionEngine(cls.config, "data/ref_data")

    def test_pavm_aliases_resolve_to_controlled_entity(self):
        for raw in ["PAVM", "肺动静脉畸形", "肺内右向左分流"]:
            resolution = self.engine.resolver.resolve(raw)
            self.assertEqual(resolution.entity_id, "D100055")
            self.assertEqual(resolution.canonical_name, "肺动静脉瘘")
            self.assertEqual(resolution.submission_name, "肺动静脉瘘")
            self.assertTrue(resolution.submittable)

    def test_aml_all_aliases_resolve_to_leukemia_entity(self):
        leukemia = self.engine.knowledge.resolve_entity("白血病")
        self.assertIsNotNone(leukemia)
        for raw in ["急性髓系白血病（AML）", "AML", "急性淋巴细胞白血病（ALL）", "ALL"]:
            resolution = self.engine.resolver.resolve(raw)
            self.assertEqual(resolution.entity_id, leukemia.entity_id)
            self.assertEqual(resolution.canonical_name, "白血病")
            self.assertEqual(resolution.submission_name, "白血病")
            self.assertTrue(resolution.submittable)

    def test_multi_source_candidates_merge_by_entity_id(self):
        evidence = EvidenceBundle(
            observations=[
                Observation("hemoptysis", "问诊", evidence_level="specific", information_value=0.84),
                Observation("cyanosis", "问诊", evidence_level="specific", information_value=0.88),
                Observation("hypoxemia", "动脉血气（ABG）", evidence_level="specific", information_value=0.92),
            ]
        )
        pool = self.engine.candidate_generator.generate(
            evidence=evidence,
            llm_result={"diagnosis_candidates": [{"name": "PAVM", "confidence": 0.92}]},
            rag_chunks=[
                {
                    "id": "external:pavm",
                    "type": "external_medical_knowledge",
                    "title": "肺动静脉畸形",
                    "score": 0.82,
                    "metadata": {"unreviewed_external": True},
                }
            ],
        )
        pavm_sources = [item for item in pool.items if item.entity_id == "D100055"]
        self.assertGreaterEqual(len(pavm_sources), 2)
        self.assertEqual({item.submission_name for item in pavm_sources}, {"肺动静脉瘘"})

        decision = self.engine.rank(pool, evidence)
        pavm = next(item for item in decision.candidates if item.entity_id == "D100055")
        self.assertEqual(pavm.diagnosis, "肺动静脉瘘")
        self.assertEqual({item["source"] for item in pavm.candidate_sources} & {"llm", "external_retrieval"}, {"llm", "external_retrieval"})

    def test_unsubmittable_entity_is_blocked_from_final(self):
        candidate = CandidateScore(
            diagnosis="未审核开放世界病",
            score=0.95,
            support_score=0.9,
            source_prior=0.9,
            explanation_score=0.8,
            coverage_score=0.8,
            residual_score=0.0,
            contradiction_penalty=0.0,
            required_met=True,
            hard_contradiction=False,
            matched_evidence=["diagnosis:未审核开放世界病"],
            eligibility_status=PRIMARY_ELIGIBLE,
            entity_id="D900001",
            canonical_name="未审核开放世界病",
            submission_name="未审核开放世界病",
            submittable=False,
        )
        decision = DiagnosisDecision(
            final_diagnoses=[candidate.diagnosis],
            trusted_diagnoses=[candidate.diagnosis],
            candidates=[candidate],
            unexplained_evidence=[],
            confidence=0.95,
            margin=0.95,
            low_confidence=False,
        )
        self.engine.authorize_final_diagnoses(decision)
        self.assertEqual(decision.final_diagnoses, [])
        self.assertEqual(decision.blocked_diagnoses[0]["reason"], "entity is not submittable")

    def test_exam_strategy_uses_entity_bundle_fallback_for_unavailable_special_exam(self):
        strategy = ExamStrategyAgent(KnowledgeBase("data/ref_data"), max_new_items=6)
        result = strategy.recommend(
            collected_info={"symptoms": ["活动后气短", "发绀", "咳血"]},
            candidate_diseases=["肺动静脉畸形", "肺癌"],
            proposed_items=[],
            existing_results={},
            judge_decision={
                "primary": "肺癌",
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
                "differential_candidates": ["肺动静脉畸形", "肺癌"],
                "discriminating_exam_tasks": [
                    {
                        "exam": "肺动脉CTA",
                        "target_candidates": ["肺动静脉畸形"],
                        "target_findings": ["pulmonary_vascular_shunt"],
                        "exam_type": "special_discriminator",
                    }
                ],
            },
        )
        self.assertTrue(result["differential_driven"])
        self.assertTrue({"胸部CT扫描（Chest CT）", "动脉血气（ABG）"} & set(result["items"]))
        self.assertTrue(
            any(
                detail.get("exam_source") == "entity_exam_bundle_fallback"
                for detail in result["exam_authorization_details"]
            )
        )


if __name__ == "__main__":
    unittest.main()
