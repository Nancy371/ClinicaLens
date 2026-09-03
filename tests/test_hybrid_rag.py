import copy
import unittest

import yaml

from agent.knowledge import KnowledgeBase
from agent.memory import DoctorMemory
from agent.policy_store import PolicyStore
from agent.rag_retriever import HybridRAGRetriever


class HybridRAGTests(unittest.TestCase):
    def setUp(self):
        with open("config.yaml", "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.config = copy.deepcopy(self.config)
        self.config["memory"]["json_path"] = "tests/does-not-exist-memory.json"
        self.config["memory"]["md_path"] = "tests/does-not-exist-memory.md"
        self.knowledge = KnowledgeBase("data/ref_data")
        self.memory = DoctorMemory(self.config)
        self.memory.notes = [
            {
                "patient_id": "successful-coronary-case",
                "memory_kind": "success",
                "symptoms": ["胸闷", "胸痛"],
                "expected_diagnosis": ["冠心病"],
                "content": "胸闷伴活动诱发时优先完成心电图和心肌损伤评估。",
                "metrics": {"diagnosis_accuracy": 1.0, "quality_score": 0.9},
            }
        ]
        self.policy = PolicyStore("tests/does-not-exist-policies.json")
        self.policy.patches = [
            {
                "id": "active-chest",
                "type": "exam_mandatory",
                "trigger": {"symptoms_any": ["胸闷"]},
                "action": "胸痛高风险时优先完成心电图。",
                "items": ["心电图"],
                "stats": {"status": "active", "hits": 3},
                "source": {},
            }
        ]
        self.retriever = HybridRAGRetriever(
            self.config,
            self.knowledge,
            self.memory,
            self.policy,
        )

    def test_mqe_expands_colloquial_chest_symptoms(self):
        expanded = self.retriever.expand_query(
            "胸口闷，喘不上气",
            {"symptoms": ["胸口闷", "喘不上气"]},
            enable_mqe=True,
            mqe_expansions=2,
        )
        self.assertIn("胸闷", expanded)
        self.assertIn("呼吸困难", expanded)
        self.assertIn("冠心病", expanded)
        self.assertIn("肺炎", expanded)

    def test_unified_search_returns_all_four_chunk_families(self):
        chunks = self.retriever.search(
            collected_info={"symptoms": ["胸闷", "胸痛"]},
            query="胸口闷 胸痛",
            candidate_diseases=["冠心病", "心肌梗死"],
            top_k=8,
            include_policy_shadow=False,
        )
        kinds = {item["type"] for item in chunks}
        self.assertIn("disease_profile", kinds)
        self.assertIn("standard_exam", kinds)
        self.assertIn("case_experience", kinds)
        self.assertIn("policy_patch", kinds)

    def test_failure_chunk_exposes_lesson_not_wrong_submission(self):
        note = {
            "patient_id": "failed-case",
            "memory_kind": "failure_lesson",
            "content": "错误提交诊断：上呼吸道感染",
            "lesson": "参考诊断：低镁血症；避免忽略低血镁。",
            "expected_diagnosis": ["低镁血症"],
            "metrics": {"diagnosis_accuracy": 0.0, "quality_score": 0.2},
        }
        text = self.retriever._render_experience(note)
        self.assertIn("参考诊断：低镁血症", text)
        self.assertNotIn("错误提交诊断：上呼吸道感染", text)

    def test_removed_legacy_extension_cannot_enter_rag_namespace(self):
        chunks = self.retriever.search(
            collected_info={"symptoms": ["呼吸困难"]},
            query="三房心 左心房隔膜",
            candidate_diseases=["三房心"],
            top_k=8,
        )
        self.assertNotIn("三房心", {item.get("title") for item in chunks})


if __name__ == "__main__":
    unittest.main()
