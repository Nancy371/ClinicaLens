import unittest

import yaml

from agent.clinical_evidence import ClinicalEvidenceNormalizer
from agent.diagnosis_critic import DiagnosisCritic
from agent.diagnosis_engine import DiagnosisDecisionEngine
from agent.prompt import DoctorPrompt
from agent.qc import QualityAgent
from agent.policy_store import PolicyStore
from agent.replay import DiagnosticReplay
from agent.treatment_safety import TreatmentSafetyGate
from agent.treatment_strategy import TreatmentStrategyAgent


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class DiagnosisCriticTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_critic_cannot_create_out_of_namespace_diagnosis(self):
        config = load_config()
        normalizer = ClinicalEvidenceNormalizer("data/ref_data")
        engine = DiagnosisDecisionEngine(config, "data/ref_data")
        evidence = normalizer.normalize({"symptoms": ["头晕"]}, {})
        decision = engine.decide({}, [], evidence)
        calls = []

        async def fake_llm(messages, temperature=0.1):
            calls.append(messages)
            return {
                "selected_diagnoses": ["目录外疾病"],
                "recommended_exams": [],
                "reason": "尝试越界",
                "confidence": 0.9,
            }

        critic = DiagnosisCritic(config, engine.knowledge, fake_llm)
        reviewed = await critic.review(decision, evidence, remaining_seconds=100)
        self.assertEqual(len(calls), 1)
        self.assertTrue(reviewed.llm_used)
        self.assertNotIn("目录外疾病", reviewed.selected_diagnoses)

    def test_shadow_promotion_gate_requires_three_independent_cases(self):
        too_few = DiagnosticReplay.promotion_summary({"a": 0.3, "b": 0.3})
        enough = DiagnosticReplay.promotion_summary({"a": 0.2, "b": 0.1, "c": 0.2})
        self.assertFalse(too_few["should_promote"])
        self.assertTrue(enough["should_promote"])


class PolicyGateTests(unittest.TestCase):
    def test_unexecutable_zero_hit_shadow_is_retired(self):
        store = PolicyStore("tests/does-not-exist-policies.json")
        store._save = lambda: None
        store.patches = [
            {
                "id": "bad",
                "trigger": {"signal": "自由文本"},
                "stats": {"status": "shadow", "hits": 0},
                "source": {},
            },
            {
                "id": "good",
                "trigger": {"symptoms_any": "胸痛"},
                "stats": {"status": "shadow", "hits": 0},
                "source": {},
            },
        ]
        stats = store.sanitize_shadow_patches()
        self.assertEqual(stats["retired"], 1)
        self.assertEqual(store.patches[0]["stats"]["status"], "retired")
        self.assertEqual(store.patches[1]["trigger"]["symptoms_any"], ["胸痛"])

    def test_replay_needs_full_promotion_metrics_before_promoting(self):
        store = PolicyStore("tests/does-not-exist-policies.json")
        store._save = lambda: None
        store.patches = [
            {
                "id": "candidate",
                "trigger": {"symptoms_any": ["胸痛"]},
                "stats": {
                    "status": "shadow",
                    "hits": 0,
                    "diagnostic_replay": {
                        "independent_cases": 3,
                        "success_ratio": 0.6667,
                        "avg_diagnosis_gain": 0.12,
                    },
                },
                "source": {},
            }
        ]
        result = store.audit()
        self.assertEqual(result["promoted"], 0)
        self.assertEqual(store.patches[0]["stats"]["status"], "shadow")

        store.record_diagnostic_replay(
            "candidate",
            {"a": 0.2, "b": 0.2, "c": 0.2},
            promotion_metrics={
                "target_fix_rate": 0.95,
                "neighboring_accuracy_delta": 0.0,
                "false_positive_increase": 0.0,
                "global_accuracy_delta": 0.0,
                "unsafe_submission_delta": 0.0,
            },
        )
        result = store.audit()
        self.assertEqual(result["promoted"], 1)
        self.assertEqual(store.patches[0]["stats"]["status"], "active")

    def test_required_gap_authorized_legacy_patch_is_quarantined(self):
        store = PolicyStore("tests/does-not-exist-policies.json")
        store._save = lambda: None
        store.patches = [
            {
                "id": "unsafe",
                "type": "ranking",
                "trigger": {"always": True},
                "action": {"required_gap_authorized": True},
                "stats": {"status": "active"},
                "source": {},
            }
        ]
        changed = store._normalize_loaded_patches()
        self.assertTrue(changed)
        self.assertEqual(store.patches[0]["stats"]["status"], "quarantined")


class TreatmentAndMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config()
        engine = DiagnosisDecisionEngine(config, "data/ref_data")
        cls.gate = TreatmentSafetyGate(engine.knowledge)
        from agent.knowledge import KnowledgeBase
        knowledge = KnowledgeBase("data/ref_data")
        cls.qc = QualityAgent(knowledge, engine.knowledge.allowed_names)
        cls.strategy = TreatmentStrategyAgent(knowledge, engine.knowledge)

    def test_qc_never_invents_upper_respiratory_infection_for_empty_evidence(self):
        result = self.qc.review_final_result({}, collected_info={}, exam_results={})
        self.assertEqual(result["diagnosis"], [])
        self.assertNotIn("上呼吸道感染", result["diagnosis"])

    def test_explicit_allergy_removes_conflicting_treatment_segment(self):
        fixed = self.gate.review(
            {
                "diagnosis": ["冠心病"],
                "treatment_plan": "给予阿司匹林抗血小板治疗。补液并监测生命体征。",
            },
            {"allergies": "阿司匹林过敏"},
            {},
        )
        self.assertNotIn("给予阿司匹林", fixed["treatment_plan"])
        self.assertIn("补液并监测生命体征", fixed["treatment_plan"])
        self.assertFalse(fixed["_treatment_safety"]["safe"])

    def test_severe_renal_abnormality_filters_renal_risk_treatment(self):
        fixed = self.gate.review(
            {
                "diagnosis": ["骨关节炎"],
                "treatment_plan": "给予布洛芬止痛。采用物理治疗并复诊。",
            },
            {"age": 70},
            {"肾功能": {"status": "abnormal", "result": "严重肾功能不全"}},
        )
        self.assertNotIn("给予布洛芬", fixed["treatment_plan"])
        self.assertIn("采用物理治疗", fixed["treatment_plan"])

    def test_treatment_strategy_covers_authorized_diagnosis_protocols(self):
        fixed = self.strategy.review(
            {
                "diagnosis": ["\u4f4e\u9541\u8840\u75c7"],
                "treatment_plan": "\u5148\u7ed9\u4e88\u5bf9\u75c7\u5904\u7406\u3002",
                "reasoning": "\u5df2\u6388\u6743\u8be5\u4e3b\u8bca\u65ad\u3002",
            },
            {"age": 70},
            {"\u7535\u89e3\u8d28": {"result": "\u8840\u9541\u964d\u4f4e"}},
        )

        plan = fixed["treatment_plan"]
        audit = fixed["_treatment_strategy"]
        self.assertIn("\u6388\u6743\u8bca\u65ad\u5bf9\u5e94\u6cbb\u7597", plan)
        self.assertIn("\u6309\u75c7\u72b6\u4e25\u91cd\u5ea6\u548c\u80be\u529f\u80fd\u8865\u9541", plan)
        self.assertIn("\u76d1\u6d4b\u4e0e\u590d\u67e5", plan)
        self.assertIn("\u4e13\u79d1\u4e0e\u968f\u8bbf", plan)
        self.assertEqual(audit["covered_diagnoses"], ["\u4f4e\u9541\u8840\u75c7"])
        self.assertEqual(audit["uncovered_diagnoses"], [])
        self.assertEqual(audit["treatment_protocol_coverage_rate"], 1.0)
        self.assertIn("diagnosis_specific_protocol", audit["actionability_sections"])

    def test_failure_memory_renders_lesson_not_wrong_submission(self):
        section = DoctorPrompt()._build_experience_section(
            [
                {
                    "memory_kind": "failure_lesson",
                    "content": "错误提交诊断：上呼吸道感染",
                    "lesson": "参考诊断：低镁血症。错误类型：证据抽取失败。",
                    "metrics": {
                        "diagnosis_accuracy": 0,
                        "exam_precision": 0.8,
                        "treatment_score": 0.5,
                    },
                }
            ]
        )
        self.assertIn("参考诊断：低镁血症", section)
        self.assertNotIn("错误提交诊断：上呼吸道感染", section)


if __name__ == "__main__":
    unittest.main()
