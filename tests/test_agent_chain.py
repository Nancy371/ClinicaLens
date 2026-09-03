import copy
import time
import unittest
from contextlib import nullcontext

import yaml

from agent.agent import MyDoctorAgent
from agent.clinical_evidence import EvidenceBundle, Observation
from agent.diagnosis_critic import CriticDecision


class FakeActions:
    def __init__(self, events):
        self.events = events

    async def ask_patient(self, patient_id, input_data):
        self.events.append("ask_patient")
        return "手足抽筋并伴心悸。"

    async def order_examination(self, patient_id, items, reason=""):
        self.events.append("order_examination")
        return {
            "results": {
                "电解质": {
                    "status": "abnormal",
                    "result": {"血镁": "0.45 mmol/L［参考值：0.75-1.02 mmol/L］"},
                }
            }
        }

    async def prescribe_treatment(self, patient_id, diagnosis, treatment_plan, reasoning):
        self.events.append("prescribe_treatment")
        return {
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "treatment_plan": treatment_plan,
            "reasoning": reasoning,
            "finished": True,
        }


class RecordingExamActions:
    def __init__(self):
        self.ordered_items = []

    async def order_examination(self, patient_id, items, reason=""):
        self.ordered_items.append(list(items))
        return {
            "results": {
                item: {"status": "normal", "result": "test result"}
                for item in items
            }
        }


class AgentChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_actions_follow_evidence_first_submission_order(self):
        with open("config.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        with nullcontext("tests/_runtime_chain") as tmp:
            config = copy.deepcopy(config)
            config["memory"]["json_path"] = f"{tmp}/memory.json"
            config["memory"]["md_path"] = f"{tmp}/memory.md"
            config["memory"]["diagnostic_replay_path"] = f"{tmp}/replay.jsonl"
            config["policy_store_path"] = f"{tmp}/policies.json"
            config["self_improve_enabled"] = False
            events = []
            agent = MyDoctorAgent(config)
            agent.actions = FakeActions(events)
            agent._case_started_at = time.monotonic()
            agent._case_deadline = agent._case_started_at + 235

            answer = await agent.actions.ask_patient("case-chain", {"question": "症状？"})
            exam_response = await agent.actions.order_examination(
                "case-chain", ["电解质"], "确认电解质异常"
            )
            exam_results = exam_response["results"]

            async def fake_diagnosis(_messages):
                return {
                    "diagnosis": ["低镁血症"],
                    "treatment_plan": "结合肾功能补镁并复查电解质。",
                    "reasoning": "症状与低血镁一致。",
                }

            original_search = agent.memory_manager.search_rag
            original_decide = agent.diagnosis_engine.decide
            original_review = agent.diagnosis_critic.review
            original_safety = agent.treatment_safety.review

            def traced_search(*args, **kwargs):
                events.append("rag")
                return original_search(*args, **kwargs)

            def traced_decide(*args, **kwargs):
                events.append("evidence_score")
                return original_decide(*args, **kwargs)

            async def traced_critic(decision, evidence, remaining_seconds, allow_llm=True):
                events.append("critic")
                return CriticDecision(
                    selected_diagnoses=list(decision.final_diagnoses),
                    confidence=decision.confidence,
                    reason="测试确定性审查",
                )

            def traced_safety(*args, **kwargs):
                events.append("treatment_safety")
                return original_safety(*args, **kwargs)

            agent._llm_generate_diagnosis = fake_diagnosis
            agent.memory_manager.search_rag = traced_search
            agent.diagnosis_engine.decide = traced_decide
            agent.diagnosis_critic.review = traced_critic
            agent.treatment_safety.review = traced_safety

            result = await agent._prescribe(
                patient_id="case-chain",
                collected_info={"symptoms": [answer]},
                exam_results=exam_results,
                chat_history=[
                    {"from": "doctor", "text": "症状？"},
                    {"from": "patient", "text": answer},
                ],
            )

            expected_order = [
                "ask_patient",
                "order_examination",
                "rag",
                "evidence_score",
                "rag",
                "evidence_score",
                "critic",
                "treatment_safety",
                "prescribe_treatment",
            ]
            cursor = 0
            for event in events:
                if cursor < len(expected_order) and event == expected_order[cursor]:
                    cursor += 1
            self.assertEqual(cursor, len(expected_order), events)
            self.assertEqual(result["diagnosis"][0], "低镁血症")
            self.assertTrue(result["finished"])

    async def test_deferred_anchor_corrective_exam_can_exceed_regular_round_cap(self):
        with open("config.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config = copy.deepcopy(config)
        config["memory"]["json_path"] = "tests/_runtime_chain/memory.json"
        config["memory"]["md_path"] = "tests/_runtime_chain/memory.md"
        config["memory"]["diagnostic_replay_path"] = "tests/_runtime_chain/replay.jsonl"
        config["policy_store_path"] = "tests/_runtime_chain/policies.json"
        config["self_improve_enabled"] = False
        agent = MyDoctorAgent(config)
        actions = RecordingExamActions()
        agent.actions = actions
        planner = agent._get_planner()
        planner.exam_rounds = agent.max_exam_rounds

        rickets = "\u7ef4\u751f\u7d20D\u7f3a\u4e4f\u6027\u4f5d\u507b\u75c5"
        exams = [
            "\u7ef4\u751f\u7d20D\u68c0\u6d4b",
            "\u7532\u72b6\u65c1\u817a\u6fc0\u7d20\u68c0\u6d4b\uff08PTH\uff09",
        ]

        blocked = await agent._maybe_order_critic_exams(
            patient_id="case-deferred-anchor",
            recommended_exams=exams,
            exam_results={},
            collected_info={"symptoms": ["leg pain"]},
            candidate_diseases=[rickets],
            add_strong_verification=False,
        )
        self.assertEqual(blocked, {})
        self.assertEqual(actions.ordered_items, [])

        ordered = await agent._maybe_order_critic_exams(
            patient_id="case-deferred-anchor",
            recommended_exams=exams,
            exam_results={},
            collected_info={"symptoms": ["leg pain"]},
            candidate_diseases=[rickets],
            add_strong_verification=False,
            force_deferred_anchor_round=True,
        )
        self.assertIn("\u7ef4\u751f\u7d20D\u68c0\u6d4b", ordered)
        self.assertIn(
            "\u7532\u72b6\u65c1\u817a\u6fc0\u7d20\u68c0\u6d4b\uff08PTH\uff09",
            ordered,
        )
        self.assertTrue(actions.ordered_items)


class EvidenceGapExamRecommendationTests(unittest.TestCase):
    def make_agent(self):
        with open("config.yaml", "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config = copy.deepcopy(config)
        config["memory"]["json_path"] = "tests/_runtime_chain/memory.json"
        config["memory"]["md_path"] = "tests/_runtime_chain/memory.md"
        config["memory"]["diagnostic_replay_path"] = "tests/_runtime_chain/replay.jsonl"
        config["policy_store_path"] = "tests/_runtime_chain/policies.json"
        config["self_improve_enabled"] = False
        return MyDoctorAgent(config)

    def test_low_magnesium_gap_recommends_metabolic_confirmation(self):
        agent = self.make_agent()
        info = {"symptoms": ["腹泻", "手足抽筋", "心悸", "意识模糊"]}
        evidence = agent.clinical_normalizer.normalize(info, {})
        decision = agent.diagnosis_engine.decide(
            {
                "diagnosis_candidates": [
                    {"name": "心律失常", "confidence": 0.9},
                    {"name": "低镁血症", "confidence": 0.82},
                ]
            },
            [],
            evidence,
        )
        exams = agent._recommend_evidence_gap_exams(decision, info, {})
        self.assertLessEqual(len(exams), 4)
        self.assertIn("综合代谢面板（CMP）", exams)
        self.assertIn("24小时尿电解质检测", exams)

    def test_deferred_anchor_target_survives_discriminating_pool_for_exams(self):
        agent = self.make_agent()
        rickets = "维生素D缺乏性佝偻病"
        info = {"symptoms": ["腿痛", "间歇性跛行", "楼梯耐受下降", "乏力"]}
        evidence = EvidenceBundle(
            [
                Observation("bone_pain", "test", confidence=0.90),
                Observation("waddling_gait", "test", confidence=0.84),
                Observation("hypocalcemia", "test", confidence=0.92),
                Observation("alp_elevated", "test", confidence=0.90),
                Observation("magnesium_load_retention_high", "test", confidence=0.88),
            ]
        )
        decision = agent.diagnosis_engine.decide(
            {
                "diagnosis_candidates": [
                    {"name": "低镁血症", "confidence": 0.9},
                    {"name": rickets, "confidence": 0.84},
                    {"name": "肾上腺疾病", "confidence": 0.7},
                ]
            },
            [],
            evidence,
        )
        decision.judge_decision["needs_discriminating_exams"] = True
        decision.judge_decision["evidence_gap_targets"] = [rickets]
        decision.judge_decision["differential_candidates"] = ["低镁血症", "肾上腺疾病"]
        decision.judge_decision["discriminating_exams"] = ["血清电解质", "24小时尿电解质检测"]

        exams = agent._recommend_evidence_gap_exams(decision, info, {})

        self.assertIn("维生素D检测", exams)
        self.assertIn("甲状旁腺激素检测（PTH）", exams)

        decision.judge_decision["needs_discriminating_exams"] = True
        decision.judge_decision["evidence_gap_targets"] = [rickets, "房间隔缺损", "肺动脉瓣狭窄"]
        decision.judge_decision["differential_candidates"] = [
            "低镁血症",
            rickets,
            "房间隔缺损",
            "肺动脉瓣狭窄",
        ]
        decision.judge_decision["discriminating_exams"] = [
            "超声心动图",
            "胸部X线检查（CXR）",
            "心导管检查",
            "心脏MRI（CMR）",
        ]
        exams = agent._recommend_evidence_gap_exams(decision, info, {})
        self.assertIn("维生素D检测", exams)
        self.assertIn("甲状旁腺激素检测（PTH）", exams)

        decision.judge_decision["needs_discriminating_exams"] = False
        exams = agent._recommend_evidence_gap_exams(decision, info, {})
        self.assertIn("维生素D检测", exams)
        self.assertIn("甲状旁腺激素检测（PTH）", exams)

    def test_pulmonary_renal_gap_recommends_vasculitis_workup(self):
        agent = self.make_agent()
        info = {"symptoms": ["咳血痰", "尿色加深", "脚踝水肿", "气短"]}
        evidence = agent.clinical_normalizer.normalize(info, {})
        decision = agent.diagnosis_engine.decide(
            {
                "diagnosis_candidates": [
                    {"name": "冠心病", "confidence": 0.8},
                    {"name": "显微镜下多血管炎", "confidence": 0.76},
                ]
            },
            [],
            evidence,
        )
        exams = agent._recommend_evidence_gap_exams(decision, info, {})
        self.assertLessEqual(len(exams), 4)
        self.assertIn("抗中性粒细胞胞质抗体（ANCA）谱", exams)
        self.assertIn("MPO-ANCA", exams)
        self.assertIn("尿液分析（UA）", exams)
        self.assertIn("肾功能检查（RFTs）", exams)

    def test_strategy_order_items_preserves_reserved_pavm_closure_exam(self):
        agent = self.make_agent()
        cta = "\u80ba\u52a8\u8109CTA"
        anca = "\u6297\u4e2d\u6027\u7c92\u7ec6\u80de\u80de\u8d28\u6297\u4f53\uff08ANCA\uff09\u8c31"
        strategy = {
            "items": [cta, anca],
            "differential_driven": True,
            "exam_authorization_details": [
                {
                    "exam": cta,
                    "exam_source": "deferred_gap_closure_exam",
                    "priority_override": True,
                    "priority_bucket": "high_value_deferred_gap_closure",
                    "target_gaps": ["G-PAVF-01"],
                },
                {
                    "exam": anca,
                    "exam_source": "conflict_adjudication_exam",
                    "priority_bucket": "conflict_adjudication",
                },
            ],
        }

        items = agent._strategy_order_items(
            strategy,
            collected_info={"symptoms": ["\u54af\u8840", "\u4f4e\u6c27"]},
            candidate_diseases=["\u80ba\u52a8\u9759\u8109\u7618", "\u80ba\u764c"],
            existing_results={},
            max_items=1,
            add_strong_verification=False,
        )

        self.assertEqual(items, [cta])

    def test_strict_primary_av_block_stops_low_magnesium_gap_exams(self):
        agent = self.make_agent()
        av_block = "\u4e8c\u5ea6\u623f\u5ba4\u4f20\u5bfc\u963b\u6ede"
        low_magnesium = "\u4f4e\u9541\u8840\u75c7"
        evidence = EvidenceBundle(
            [
                Observation("second_degree_av_block", "test", confidence=0.98),
                Observation("av_block", "test", confidence=0.96),
                Observation("bradycardia", "test", confidence=0.9),
                Observation(f"diagnosis:{av_block}", "test", confidence=0.98),
                Observation("dizziness", "test", confidence=0.82),
                Observation("palpitation", "test", confidence=0.82),
            ]
        )
        decision = agent.diagnosis_engine.decide(
            {
                "diagnosis_candidates": [
                    {"name": av_block, "confidence": 0.96},
                    {"name": low_magnesium, "confidence": 0.9},
                ]
            },
            [],
            evidence,
        )
        self.assertEqual(decision.final_diagnoses[0], av_block)
        self.assertTrue(agent._strict_primary_exam_stop_active(decision))
        self.assertEqual(agent._evidence_gap_target_diagnoses(decision), [])
        self.assertEqual(agent._recommend_evidence_gap_exams(decision, {}, {}), [])

    def test_tricuspid_regurgitation_blocks_cross_system_ohss_gap_exams(self):
        agent = self.make_agent()
        primary = "三尖瓣反流"
        cross_system = "卵巢过度刺激综合征"
        evidence = EvidenceBundle(
            [
                Observation("tricuspid_regurgitation", "test", confidence=0.98),
                Observation("right_heart_enlargement", "test", confidence=0.92),
                Observation("leg_edema", "test", confidence=0.9),
                Observation("dyspnea", "test", confidence=0.88),
                Observation("abdominal_distension", "test", confidence=0.82),
                Observation(f"diagnosis:{primary}", "test", confidence=0.98),
                Observation("ascites", "test", confidence=0.76),
            ]
        )
        decision = agent.diagnosis_engine.decide(
            {
                "diagnosis_candidates": [
                    {"name": primary, "confidence": 0.95},
                    {"name": cross_system, "confidence": 0.9},
                ]
            },
            [],
            evidence,
        )
        self.assertEqual(decision.final_diagnoses[0], primary)
        self.assertNotIn(cross_system, agent._evidence_gap_target_diagnoses(decision))
        self.assertEqual(agent._recommend_evidence_gap_exams(decision, {}, {}), [])

    def test_final_name_filter_drops_generic_pneumonia_when_child_selected(self):
        agent = self.make_agent()
        filtered = agent._remove_suppressed_diagnosis_names(["支气管肺炎", "肺不张", "肺炎"])
        self.assertEqual(filtered, ["支气管肺炎", "肺不张"])

    def test_refilter_result_writes_back_suppressed_diagnosis_changes(self):
        agent = self.make_agent()
        info = {"symptoms": ["呛咳", "发热", "咳嗽", "呼吸困难"]}
        exams = {
            "胸部CT扫描（Chest CT）": {
                "status": "abnormal",
                "result": {"结论": "右下叶实变并见空气支气管征，肺不张"},
            },
            "支气管镜检查": {
                "status": "abnormal",
                "result": {"结论": "支气管内见脓性分泌物"},
            },
        }
        evidence = agent.clinical_normalizer.normalize(info, exams)
        decision = agent.diagnosis_engine.decide(
            {"diagnosis_candidates": ["支气管肺炎", "肺不张", "肺炎"]},
            [],
            evidence,
        )
        fixed = agent._refilter_diagnosis_result(
            {"diagnosis": ["支气管肺炎", "肺不张", "肺炎"], "reasoning": ""},
            decision,
            evidence,
        )
        self.assertNotIn("肺炎", fixed["diagnosis"])
        self.assertTrue({"支气管肺炎", "肺不张"} <= set(fixed["diagnosis"]))

    def test_critic_selection_keeps_weak_lung_cancer_as_differential_only(self):
        agent = self.make_agent()
        info = {"symptoms": ["咳血痰", "尿色加深", "全身酸痛", "脚踝水肿", "呼吸困难"]}
        exams = {
            "尿液分析（UA）": {
                "status": "abnormal",
                "result": {
                    "尿红细胞": "50 个/HPF［参考值：0-3］",
                    "尿蛋白": "尿蛋白阳性",
                },
            },
            "抗中性粒细胞胞质抗体（ANCA）谱": {
                "status": "abnormal",
                "result": {"MPO-ANCA": "MPO-ANCA阳性"},
            },
            "胸部CT扫描（Chest CT）": {
                "status": "abnormal",
                "result": {"结论": "弥漫性肺泡出血"},
            },
        }
        evidence = agent.clinical_normalizer.normalize(info, exams)
        decision = agent.diagnosis_engine.decide(
            {"diagnosis_candidates": ["显微镜下多血管炎", "肺癌"]},
            [],
            evidence,
        )

        agent._apply_critic_selection(
            decision,
            ["显微镜下多血管炎", "肺癌"],
            "显微镜下多血管炎最能解释肺肾综合征，肺癌需鉴别。",
        )
        agent._restore_legacy_candidate_submission(
            decision,
            {"diagnosis": ["显微镜下多血管炎"], "diagnosis_candidates": ["肺癌"]},
            CriticDecision(
                selected_diagnoses=["肺癌"],
                reason="肺癌作为鉴别诊断保留，但不作为最终诊断提交。",
            ),
        )
        fixed = agent.diagnosis_engine.apply_to_result(
            {"diagnosis": list(decision.final_diagnoses), "reasoning": ""},
            decision,
            evidence,
        )

        self.assertEqual(fixed["diagnosis"], ["显微镜下多血管炎"])
        differential_only = fixed["_diagnosis_decision"]["differential_only_diagnoses"]
        self.assertIn("肺癌", [item["diagnosis"] for item in differential_only])
        self.assertIn("仅鉴别", fixed["reasoning"])

    def test_critic_reason_text_is_not_recovered_as_final_diagnosis(self):
        agent = self.make_agent()
        critic = CriticDecision(
            selected_diagnoses=["显微镜下多血管炎"],
            reason="显微镜下多血管炎为主诊断，肺癌作为鉴别诊断保留。",
        )
        resolved = agent._resolved_critic_names(critic)
        self.assertEqual(resolved, ["显微镜下多血管炎"])

    def test_critic_cannot_add_weak_coronary_disease_to_valve_heart_failure(self):
        agent = self.make_agent()
        evidence = agent.clinical_normalizer.normalize(
            {
                "symptoms": ["活动后气短", "夜间阵发性呼吸困难", "不能平卧", "下肢水肿"],
                "physical_signs": "心尖部收缩期杂音",
            },
            {
                "超声心动图": {
                    "status": "abnormal",
                    "result": {"结论": "重度二尖瓣反流，左心室扩大，心力衰竭表现"},
                },
                "心肌酶谱": {
                    "status": "normal",
                    "result": {"肌钙蛋白": "正常"},
                },
            },
        )
        decision = agent.diagnosis_engine.decide(
            {
                "diagnosis_candidates": [
                    "二尖瓣反流",
                    "心力衰竭",
                    "冠心病",
                ]
            },
            [],
            evidence,
        )
        self.assertIn("二尖瓣反流", decision.final_diagnoses)
        self.assertIn("心力衰竭", decision.final_diagnoses)

        agent._apply_critic_selection(
            decision,
            ["二尖瓣反流", "冠心病", "心力衰竭"],
            "冠心病作为常见鉴别保留。",
        )

        self.assertIn("二尖瓣反流", decision.final_diagnoses)
        self.assertIn("心力衰竭", decision.final_diagnoses)
        self.assertNotIn("冠心病", decision.final_diagnoses)

    def test_reasoning_inference_promotes_low_magnesium_evidence(self):
        agent = self.make_agent()
        evidence = agent.clinical_normalizer.normalize(
            {"symptoms": ["腹泻", "手足抽筋", "心悸"]},
            {},
        )
        diagnosis_result = {
            "diagnosis_candidates": [
                {
                    "name": "低镁血症",
                    "supporting_evidence": [
                        "补查显示24小时尿镁降低，镁负荷保留率升高，提示镁储备不足。",
                    ],
                },
                {"name": "心律失常"},
            ],
            "reasoning": "强验证检查提示镁储备不足，低镁血症能统一解释抽筋、心悸与QT异常。",
        }
        augmented = agent._augment_evidence_from_reasoning(evidence, diagnosis_result)
        findings = augmented.findings("positive")
        self.assertIn("low_urine_magnesium", findings)
        self.assertIn("magnesium_load_retention_high", findings)
        self.assertIn("magnesium_depletion", findings)

        decision = agent.diagnosis_engine.decide(diagnosis_result, [], augmented)
        self.assertEqual(decision.final_diagnoses[0], "低镁血症")
        self.assertNotIn("心律失常", decision.final_diagnoses)

    def test_reasoning_inference_structures_pulmonary_renal_evidence(self):
        agent = self.make_agent()
        evidence = agent.clinical_normalizer.normalize(
            {"symptoms": ["咳血痰", "尿色变深", "全身酸痛", "脚踝水肿"]},
            {},
        )
        diagnosis_result = {
            "diagnosis_candidates": [
                {
                    "name": "显微镜下多血管炎",
                    "supporting_evidence": [
                        "MPO-ANCA阳性，尿色变深提示血尿/蛋白尿，胸部CT提示弥漫性肺泡出血。",
                    ],
                },
                {"name": "肺癌", "supporting_evidence": "肺癌需鉴别。"},
            ],
            "reasoning": "显微镜下多血管炎能更好解释肺肾综合征，肺癌仅作为鉴别保留。",
        }
        augmented = agent._augment_evidence_from_reasoning(evidence, diagnosis_result)
        findings = augmented.findings("positive")
        self.assertIn("mpo_anca_positive", findings)
        self.assertIn("microscopic_hematuria", findings)
        self.assertIn("proteinuria", findings)
        self.assertIn("pulmonary_hemorrhage", findings)

        decision = agent.diagnosis_engine.decide(diagnosis_result, [], augmented)
        self.assertEqual(decision.final_diagnoses, ["显微镜下多血管炎"])

    def test_legacy_restore_cannot_override_authorized_primary(self):
        agent = self.make_agent()
        evidence = agent.clinical_normalizer.normalize(
            {"symptoms": ["婴儿", "黄疸", "巩膜黄染", "嗜睡", "家族遗传病史"]},
            {
                "肝功能检查（LFTs）": {
                    "status": "abnormal",
                    "result": {
                        "总胆红素": "380 umol/L",
                        "间接胆红素": "360 umol/L",
                    },
                },
                "基因检测": {
                    "status": "abnormal",
                    "result": {"结论": "UGT1A1 双等位致病变异"},
                },
            },
        )
        decision = agent.diagnosis_engine.decide(
            {"diagnosis_candidates": ["肺炎", "克里格勒-纳贾尔综合征"]},
            [],
            evidence,
        )
        self.assertEqual(decision.final_diagnoses, ["克里格勒-纳贾尔综合征"])

        agent._restore_legacy_candidate_submission(
            decision,
            {"diagnosis": ["肺炎"], "diagnosis_candidates": ["肺炎"]},
            CriticDecision(selected_diagnoses=["肺炎"], reason="legacy primary"),
        )

        self.assertEqual(decision.final_diagnoses, ["克里格勒-纳贾尔综合征"])


if __name__ == "__main__":
    unittest.main()
