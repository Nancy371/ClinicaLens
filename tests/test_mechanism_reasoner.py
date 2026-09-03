import unittest

import yaml

from agent.clinical_evidence import ClinicalEvidenceNormalizer
from agent.mechanism_reasoner import MechanismReasoner
from agent.rag_retriever import HybridRAGRetriever
from agent.knowledge import KnowledgeBase


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class MechanismReasonerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.normalizer = ClinicalEvidenceNormalizer("data/ref_data")
        cls.reasoner = MechanismReasoner()

    def test_urachal_language_forms_family_mechanism_before_disease_name(self):
        evidence = self.normalizer.normalize(
            {
                "symptoms": [
                    "\u8110\u90e8\u53cd\u590d\u6e17\u6db2\uff0c\u4e0b\u8179\u6b63\u4e2d\u75bc\u75db\uff0c\u6000\u7591\u4e2d\u7ebf\u5f02\u5e38\u5f00\u53e3"
                ]
            },
            {},
        )
        hypotheses = self.reasoner.evaluate(evidence)
        self.assertTrue(hypotheses)
        self.assertEqual(hypotheses[0].mechanism_id, "urachal_remnant_anomaly")
        self.assertIn("urachal_remnant", hypotheses[0].family_id)
        self.assertIn("\u8110\u5c3f\u7ba1\u56ca\u80bf", hypotheses[0].candidate_diseases)

    def test_missing_backend_diseases_are_open_world_mechanism_candidates(self):
        evidence = self.normalizer.normalize(
            {
                "symptoms": [
                    "\u52b3\u529b\u6027\u547c\u5438\u56f0\u96be\uff0c\u53d1\u7ec0\uff0c\u4f4e\u6c27\uff0c\u54af\u8840\uff0cCT\u63d0\u793a\u80ba\u7ed3\u8282"
                ]
            },
            {},
        )
        hypotheses = self.reasoner.evaluate(evidence)
        mechanism_ids = {item.mechanism_id for item in hypotheses}
        self.assertIn("pulmonary_vascular_shunt", mechanism_ids)
        pavm = next(item for item in hypotheses if item.mechanism_id == "pulmonary_vascular_shunt")
        self.assertIn("\u80ba\u52a8\u9759\u8109\u7618", pavm.candidate_diseases)

    def test_retrieval_views_feed_external_unreviewed_chunks(self):
        evidence = self.normalizer.normalize(
            {"symptoms": ["\u80f8\u9aa8\u51f9\u9677\uff0c\u6d3b\u52a8\u540e\u6c14\u77ed"]},
            {},
        )
        views = [item.to_dict() for item in self.reasoner.retrieval_views(evidence)]
        retriever = HybridRAGRetriever(self.config, KnowledgeBase("data/ref_data"))
        chunks = retriever.search(
            collected_info={"symptoms": ["\u80f8\u9aa8\u51f9\u9677"]},
            retrieval_views=views,
            top_k=8,
            score_threshold=0.0,
        )
        external = [item for item in chunks if item.get("type") == "external_medical_knowledge"]
        self.assertTrue(external)
        self.assertIn("\u6f0f\u6597\u80f8", {item.get("title") for item in external})
        self.assertTrue(all((item.get("metadata") or {}).get("unreviewed_external") for item in external))


if __name__ == "__main__":
    unittest.main()
