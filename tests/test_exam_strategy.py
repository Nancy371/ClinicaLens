import unittest

from agent.exam_strategy import ExamStrategyAgent
from agent.knowledge import KnowledgeBase


class ExamInformationGainTests(unittest.TestCase):
    def setUp(self):
        self.strategy = ExamStrategyAgent(KnowledgeBase("data/ref_data"), max_new_items=6)

    def test_top_three_candidates_drive_exam_ranking(self):
        result = self.strategy.recommend(
            collected_info={"symptoms": ["上腹痛", "恶心", "呕吐"]},
            candidate_diseases=["胃炎", "胆囊炎", "胰腺炎"],
            proposed_items=["超声心动图", "腹部B超", "血常规"],
            existing_results={},
        )
        self.assertLessEqual(len(result["items"]), 6)
        self.assertTrue(result["information_gain"])
        self.assertNotEqual(result["items"][0], "超声心动图")
        self.assertNotIn("心导管检查", result["items"])

    def test_differential_driven_plan_uses_judge_discriminating_exams(self):
        result = self.strategy.recommend(
            collected_info={"symptoms": ["皮疹", "关节痛", "乏力"]},
            candidate_diseases=["雅司病", "湿疹", "白血病"],
            proposed_items=["超声心动图", "心电图"],
            existing_results={},
            judge_decision={
                "primary": "雅司病",
                "differential_candidates": ["雅司病", "湿疹", "白血病"],
                "discriminating_exams": [
                    "全血细胞计数（CBC）",
                    "外周血涂片",
                    "梅毒螺旋体血清学试验",
                    "体格检查",
                    "组织病理学检查",
                ],
            },
        )
        self.assertTrue(result["differential_driven"])
        self.assertLessEqual(len(result["items"]), 6)
        self.assertTrue(result["exam_authorization_details"])
        self.assertTrue(
            all(
                item["exam_source"] == "judge_discriminating_exam"
                for item in result["exam_authorization_details"]
            )
        )
        self.assertIn("全血细胞计数（CBC）", result["items"])
        self.assertIn("梅毒血清学检查", result["items"])
        self.assertNotIn("超声心动图", result["items"])
        self.assertNotIn("心电图（ECG）", result["items"])

    def test_tb_differential_plan_uses_shared_discriminating_exams(self):
        result = self.strategy.recommend(
            collected_info={"symptoms": ["咳嗽", "咯血", "低热", "消瘦"]},
            candidate_diseases=["肺结核", "肺炎", "肺癌"],
            proposed_items=["泌尿道超声", "尿动力学"],
            existing_results={},
            judge_decision={
                "primary": "肺结核",
                "differential_candidates": ["肺结核", "肺炎", "肺癌"],
                "discriminating_exams": [
                    "胸部CT扫描（Chest CT）",
                    "痰培养",
                    "抗酸杆菌染色（AFB）",
                    "Xpert MTB/RIF",
                ],
            },
        )
        self.assertTrue(result["differential_driven"])
        self.assertEqual(result["items"][0], "胸部CT扫描（Chest CT）")
        self.assertIn("痰培养", result["items"])
        self.assertIn("抗酸杆菌染色（AFB）", result["items"])
        self.assertIn("核酸扩增检测（NAAT）", result["items"])
        self.assertNotIn("泌尿道超声", result["items"])
        self.assertNotIn("尿动力学检查（UDS）", result["items"])

    def test_deferred_gap_closure_task_beats_generic_exam_ranking(self):
        deferred_exam = "鑳搁儴CT鎵弿锛圕hest CT锛?"
        generic_exam = "鍏ㄨ缁嗚優璁℃暟锛圕BC锛?"

        ranked, scores = self.strategy._rank_by_information_gain(
            candidate_diseases=["鑲哄姩闈欒剦鐦?", "鑲虹檶"],
            symptoms=["鍜"],
            proposed_items=[deferred_exam, generic_exam],
            exam_tasks=[
                {
                    "exam": deferred_exam,
                    "target_candidates": ["鑲哄姩闈欒剦鐦?"],
                    "target_findings": ["pulmonary_vascular_malformation_confirmed"],
                    "target_gaps": ["G-PAVF-01"],
                    "exam_type": "deferred_gap_closure",
                    "exam_source": "deferred_gap_closure_exam",
                    "priority_override": True,
                    "information_gain_hint": 0.99,
                },
                {
                    "exam": generic_exam,
                    "target_candidates": ["鑲哄姩闈欒剦鐦?", "鑲虹檶"],
                    "target_findings": ["inflammation"],
                    "exam_type": "generic_inflammation",
                    "information_gain_hint": 0.5,
                },
            ],
        )

        normalized_deferred, _ = self.strategy.knowledge.normalize_examinations([deferred_exam])
        normalized_generic, _ = self.strategy.knowledge.normalize_examinations([generic_exam])
        expected_deferred = normalized_deferred[0] if normalized_deferred else deferred_exam
        expected_generic = normalized_generic[0] if normalized_generic else generic_exam

        self.assertEqual(ranked[0], expected_deferred)
        self.assertGreater(scores[expected_deferred], scores[expected_generic])

    def test_radiation_gap_task_carries_evidence_question_claim(self):
        chest_ct = "\u80f8\u90e8CT\u626b\u63cf\uff08Chest CT\uff09"
        result = self.strategy.recommend(
            collected_info={"symptoms": ["dyspnea", "cough"]},
            candidate_diseases=["\u653e\u5c04\u6027\u80ba\u708e"],
            proposed_items=[],
            existing_results={},
            judge_decision={
                "primary": "\u80ba\u4e0d\u5f20",
                "active_evidence_gaps": [
                    {
                        "gap_id": "G-D100058-derived_pattern_gap-1-post_radiotherapy_time_window",
                        "candidate": "\u653e\u5c04\u6027\u80ba\u708e",
                        "entity_id": "D100058",
                        "target_evidence": "post_radiotherapy_time_window",
                        "gap_value": 0.72,
                        "closure_exams": [chest_ct],
                        "expected_transition": {
                            "positive": "PrimaryEligible",
                            "negative": "DifferentialOnly",
                        },
                    }
                ],
            },
        )

        details = result["exam_authorization_details"]
        ct_detail = next(item for item in details if item["exam"] == chest_ct)
        self.assertIn("pulmonary_morphology", ct_detail["target_claims"])
        self.assertIn("radiation_field_lung_consistency", ct_detail["target_claims"])
        self.assertIn("post_radiotherapy_time_window", ct_detail["target_claims"])
        self.assertNotIn("ground_glass_opacity", ct_detail["target_claims"])
        self.assertIn("pulmonary_morphology", ct_detail["route_target_claims"])
        self.assertIn("radiation_field_lung_consistency", ct_detail["route_target_claims"])
        self.assertNotIn("post_radiotherapy_time_window", ct_detail["route_target_claims"])
        self.assertIn("ground_glass_opacity", ct_detail["target_findings"])
        self.assertIn("ground_glass_opacity", ct_detail["expected_evidence_concepts"])
        self.assertEqual(ct_detail["exam_role"], "target_claim_resolution")
        self.assertIn("radiation field", ct_detail["target_question"])
        self.assertEqual(
            ct_detail["expected_arbitration_effect"][
                "radiation_field_lung_consistency_supported"
            ],
            "favor_D100058",
        )

    def test_pairwise_gap_task_beats_generic_exam_ranking(self):
        pairwise_exam = "\u6ccc\u5c3f\u751f\u6b96\u9053\u75c5\u539f\u6838\u9178\u68c0\u6d4b"
        generic_exam = "\u5168\u8840\u7ec6\u80de\u8ba1\u6570\uff08CBC\uff09"

        ranked, scores = self.strategy._rank_by_information_gain(
            candidate_diseases=["\u5e26\u72b6\u75b1\u75b9", "\u8d56\u7279\u7efc\u5408\u5f81"],
            symptoms=["arthralgia", "dysuria", "conjunctivitis"],
            proposed_items=[generic_exam, pairwise_exam],
            exam_tasks=[
                {
                    "exam": pairwise_exam,
                    "target_candidates": ["\u5e26\u72b6\u75b1\u75b9", "\u8d56\u7279\u7efc\u5408\u5f81"],
                    "target_findings": ["preceding_genitourinary_infection"],
                    "exam_type": "pairwise_discrimination",
                    "exam_source": "pairwise_discrimination_exam",
                    "priority_bucket": "high_value_pairwise_gap_closure",
                    "source_gap_id": "PWG-test",
                    "target_pair": ["\u5e26\u72b6\u75b1\u75b9", "\u8d56\u7279\u7efc\u5408\u5f81"],
                    "target_question": "distinguish zoster from reactive arthritis",
                    "target_claim": "preceding_genitourinary_infection",
                    "exam_role": "trigger_evidence",
                    "information_gain_hint": 1.02,
                },
                {
                    "exam": generic_exam,
                    "target_candidates": ["\u5e26\u72b6\u75b1\u75b9", "\u8d56\u7279\u7efc\u5408\u5f81"],
                    "target_findings": ["inflammation"],
                    "exam_type": "generic_inflammation",
                    "information_gain_hint": 0.45,
                },
            ],
        )

        normalized_pairwise, _ = self.strategy.knowledge.normalize_examinations([pairwise_exam])
        normalized_generic, _ = self.strategy.knowledge.normalize_examinations([generic_exam])
        expected_pairwise = normalized_pairwise[0] if normalized_pairwise else pairwise_exam
        expected_generic = normalized_generic[0] if normalized_generic else generic_exam

        self.assertEqual(ranked[0], expected_pairwise)
        self.assertGreater(scores[expected_pairwise], scores[expected_generic])

    def test_high_value_pavm_gap_closure_is_reserved_over_conflict_and_generic(self):
        strategy = ExamStrategyAgent(
            KnowledgeBase("data/ref_data"),
            max_new_items=3,
            discriminating_exam_max_items=3,
        )
        pavm = "\u80ba\u52a8\u9759\u8109\u7618"
        mpa = "\u663e\u5fae\u955c\u4e0b\u591a\u8840\u7ba1\u708e"
        lung_cancer = "\u80ba\u764c"
        anca = "\u6297\u4e2d\u6027\u7c92\u7ec6\u80de\u80de\u8d28\u6297\u4f53\uff08ANCA\uff09\u8c31"
        cbc = "\u5168\u8840\u7ec6\u80de\u8ba1\u6570\uff08CBC\uff09"
        crp = "C\u53cd\u5e94\u86cb\u767d\uff08CRP\uff09"
        chest_ct = "\u80f8\u90e8CT\u626b\u63cf\uff08Chest CT\uff09"
        cta = "\u80ba\u52a8\u8109CTA"
        bubble_echo = "\u53f3\u5fc3\u58f0\u5b66\u9020\u5f71"
        enhanced_ct = "\u80f8\u90e8\u589e\u5f3aCT"

        result = strategy.recommend(
            collected_info={"symptoms": ["\u54af\u8840", "\u4f4e\u6c27", "\u53d1\u7ec0"]},
            candidate_diseases=[pavm, mpa, lung_cancer],
            proposed_items=[anca, cbc, crp, chest_ct],
            existing_results={},
            judge_decision={
                "primary": lung_cancer,
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
                "differential_candidates": [pavm, mpa, lung_cancer],
                "discriminating_exam_tasks": [
                    {
                        "exam": anca,
                        "target_candidates": [mpa],
                        "target_findings": ["anca_positive"],
                        "exam_type": "conflict_adjudication",
                        "exam_source": "conflict_adjudication_exam",
                        "information_gain_hint": 0.98,
                    },
                    {
                        "exam": cbc,
                        "target_candidates": [mpa, lung_cancer],
                        "target_findings": ["inflammation"],
                        "exam_type": "generic_inflammation",
                        "information_gain_hint": 0.70,
                    },
                    {
                        "exam": chest_ct,
                        "target_candidates": [pavm, lung_cancer],
                        "target_findings": ["pulmonary_nodule"],
                        "exam_type": "special_discriminator",
                        "information_gain_hint": 0.90,
                    },
                    {
                        "exam": cta,
                        "target_candidates": [pavm],
                        "target_findings": ["pulmonary_vascular_malformation_confirmed"],
                        "target_gap": "G-PAVF-01",
                        "target_gaps": ["G-PAVF-01"],
                        "exam_type": "deferred_gap_closure",
                        "exam_source": "deferred_gap_closure_exam",
                        "priority_override": True,
                        "priority_bucket": "high_value_deferred_gap_closure",
                        "closure_rank": 1,
                        "closure_priority": 100,
                        "diagnostic_coverage": 0.6,
                        "gap_diagnostic_coverage": 1.0,
                        "information_gain_hint": 0.99,
                    },
                    {
                        "exam": bubble_echo,
                        "target_candidates": [pavm],
                        "target_findings": ["bubble_echo_right_to_left_shunt"],
                        "target_gap": "G-PAVF-01",
                        "target_gaps": ["G-PAVF-01"],
                        "exam_type": "deferred_gap_closure",
                        "exam_source": "deferred_gap_closure_exam",
                        "priority_override": True,
                        "priority_bucket": "high_value_deferred_gap_closure",
                        "closure_rank": 2,
                        "closure_priority": 96,
                        "diagnostic_coverage": 0.55,
                        "gap_diagnostic_coverage": 0.96,
                        "information_gain_hint": 0.98,
                    },
                    {
                        "exam": enhanced_ct,
                        "target_candidates": [pavm],
                        "target_findings": ["enhanced_ct_vascular_malformation"],
                        "target_gap": "G-PAVF-01",
                        "target_gaps": ["G-PAVF-01"],
                        "exam_type": "deferred_gap_closure",
                        "exam_source": "deferred_gap_closure_exam",
                        "priority_override": True,
                        "priority_bucket": "high_value_deferred_gap_closure",
                        "closure_rank": 3,
                        "closure_priority": 92,
                        "diagnostic_coverage": 0.6,
                        "gap_diagnostic_coverage": 0.92,
                        "information_gain_hint": 0.97,
                    },
                ],
            },
        )

        self.assertTrue(result["differential_driven"])
        pavm_details = [
            detail
            for detail in result["exam_authorization_details"]
            if detail.get("exam_source") == "deferred_gap_closure_exam"
        ]
        self.assertTrue(pavm_details)
        self.assertEqual(pavm_details[0]["priority_bucket"], "high_value_deferred_gap_closure")
        self.assertEqual(pavm_details[0]["requested_exam"], cta)
        self.assertIn("G-PAVF-01", pavm_details[0]["target_gaps"])
        first_pavm_index = min(
            result["items"].index(detail["exam"])
            for detail in pavm_details
            if detail["exam"] in result["items"]
        )
        first_anca_index = result["items"].index(anca) if anca in result["items"] else 99
        self.assertLess(first_pavm_index, first_anca_index)
        self.assertNotIn(cbc, result["items"][:2])
        self.assertNotIn(crp, result["items"][:2])

    def test_deferred_gap_closure_tasks_are_consumed_without_discriminating_tasks(self):
        strategy = ExamStrategyAgent(
            KnowledgeBase("data/ref_data"),
            max_new_items=2,
            discriminating_exam_max_items=2,
        )
        pavm = "\u80ba\u52a8\u9759\u8109\u7618"
        anca = "\u6297\u4e2d\u6027\u7c92\u7ec6\u80de\u80de\u8d28\u6297\u4f53\uff08ANCA\uff09\u8c31"
        cta = "\u80ba\u52a8\u8109CTA"

        result = strategy.recommend(
            collected_info={"symptoms": ["\u54af\u8840", "\u4f4e\u6c27"]},
            candidate_diseases=[pavm, "\u80ba\u764c"],
            proposed_items=[anca],
            existing_results={},
            judge_decision={
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
                "differential_candidates": [pavm, "\u80ba\u764c"],
                "deferred_gap_closure_tasks": [
                    {
                        "exam": cta,
                        "target_candidates": [pavm],
                        "target_findings": ["pulmonary_vascular_malformation_confirmed"],
                        "target_gap": "G-PAVF-01",
                        "target_gaps": ["G-PAVF-01"],
                        "exam_type": "deferred_gap_closure",
                        "exam_source": "deferred_gap_closure_exam",
                        "priority_override": True,
                        "priority_bucket": "high_value_deferred_gap_closure",
                        "closure_rank": 1,
                        "closure_priority": 100,
                        "gap_diagnostic_coverage": 1.0,
                        "information_gain_hint": 0.99,
                    }
                ],
            },
        )

        self.assertTrue(result["differential_driven"])
        self.assertIn(cta, result["items"])
        self.assertEqual(result["items"][0], cta)
        self.assertTrue(
            any(
                item.get("exam_source") == "deferred_gap_closure_exam"
                for item in result["exam_authorization_details"]
            )
        )

    def test_reserved_deferred_gap_item_survives_max_item_truncation(self):
        strategy = ExamStrategyAgent(
            KnowledgeBase("data/ref_data"),
            max_new_items=1,
            discriminating_exam_max_items=1,
        )
        pavm = "\u80ba\u52a8\u9759\u8109\u7618"
        cta = "\u80ba\u52a8\u8109CTA"
        anca = "\u6297\u4e2d\u6027\u7c92\u7ec6\u80de\u80de\u8d28\u6297\u4f53\uff08ANCA\uff09\u8c31"
        chest_ct = "\u80f8\u90e8CT\u626b\u63cf\uff08Chest CT\uff09"

        result = strategy.recommend(
            collected_info={"symptoms": ["\u54af\u8840", "\u4f4e\u6c27"]},
            candidate_diseases=[pavm, "\u80ba\u764c"],
            proposed_items=[anca, chest_ct],
            existing_results={},
            judge_decision={
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
                "differential_candidates": [pavm, "\u80ba\u764c"],
                "discriminating_exam_tasks": [
                    {
                        "exam": anca,
                        "target_candidates": ["\u663e\u5fae\u955c\u4e0b\u591a\u8840\u7ba1\u708e"],
                        "exam_type": "conflict_adjudication",
                        "exam_source": "conflict_adjudication_exam",
                        "information_gain_hint": 0.99,
                    },
                    {
                        "exam": chest_ct,
                        "target_candidates": [pavm, "\u80ba\u764c"],
                        "exam_type": "special_discriminator",
                        "information_gain_hint": 0.95,
                    },
                    {
                        "exam": cta,
                        "target_candidates": [pavm],
                        "target_gap": "G-PAVF-01",
                        "target_gaps": ["G-PAVF-01"],
                        "exam_type": "deferred_gap_closure",
                        "exam_source": "deferred_gap_closure_exam",
                        "priority_override": True,
                        "priority_bucket": "high_value_deferred_gap_closure",
                        "closure_rank": 1,
                        "closure_priority": 100,
                        "gap_diagnostic_coverage": 1.0,
                        "information_gain_hint": 0.9,
                    },
                ],
            },
        )

        self.assertEqual(result["items"], [cta])

    def test_gap_value_beats_higher_candidate_score_for_next_exam(self):
        strategy = ExamStrategyAgent(
            KnowledgeBase("data/ref_data"),
            max_new_items=1,
            discriminating_exam_max_items=1,
        )
        low_magnesium = "\u4f4e\u9541\u8840\u75c7"
        rickets = "\u7ef4\u751f\u7d20D\u7f3a\u4e4f\u6027\u4f5d\u507b\u75c5"
        vitamin_d = "\u7ef4\u751f\u7d20D\u68c0\u6d4b"

        result = strategy.recommend(
            collected_info={"symptoms": ["\u817f\u75db", "\u8ddb\u884c", "\u4f4e\u9499"]},
            candidate_diseases=[low_magnesium, rickets],
            proposed_items=["\u9541\u8d1f\u8377\u8bd5\u9a8c", "\u5168\u8840\u7ec6\u80de\u8ba1\u6570\uff08CBC\uff09"],
            existing_results={},
            judge_decision={
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
                "differential_candidates": [low_magnesium, rickets],
                "active_evidence_gaps": [
                    {
                        "gap_id": "G-LOW-MAG-LOW-VALUE",
                        "candidate": low_magnesium,
                        "entity_id": "D100009",
                        "target_evidence": "magnesium_recheck",
                        "gap_type": "confirmation_gap",
                        "gap_value": 0.2,
                        "candidate_score_at_decision": 0.89,
                        "expected_transition": {
                            "positive": "PrimaryEligible",
                            "negative": "DifferentialOnly",
                        },
                        "closure_exams": ["\u9541\u8d1f\u8377\u8bd5\u9a8c"],
                    },
                    {
                        "gap_id": "G-RICKETS-HIGH-VALUE",
                        "candidate": rickets,
                        "entity_id": "D100010",
                        "target_evidence": "vitamin_d_low|bone_deformity",
                        "gap_type": "confirmation_gap",
                        "gap_value": 0.95,
                        "candidate_score_at_decision": 0.73,
                        "expected_transition": {
                            "positive": "PrimaryEligible",
                            "negative": "DifferentialOnly",
                        },
                        "closure_exams": [
                            vitamin_d,
                            "\u7532\u72b6\u65c1\u817a\u6fc0\u7d20\u68c0\u6d4b\uff08PTH\uff09",
                            "X\u7ebf\u68c0\u67e5",
                        ],
                    },
                ],
            },
        )

        self.assertEqual(result["items"], [vitamin_d])
        detail = result["exam_authorization_details"][0]
        self.assertEqual(detail["target_gap"], "G-RICKETS-HIGH-VALUE")
        self.assertEqual(detail["source_gap_value"], 0.95)
        self.assertEqual(detail["candidate_score_at_decision"], 0.73)
        self.assertTrue(detail["score_gap_decoupled"])

    def test_candidate_score_change_does_not_change_gap_value_order(self):
        strategy = ExamStrategyAgent(
            KnowledgeBase("data/ref_data"),
            max_new_items=1,
            discriminating_exam_max_items=1,
        )
        first = {
            "gap_id": "G-A",
            "candidate": "\u4f4e\u9541\u8840\u75c7",
            "entity_id": "D100009",
            "target_evidence": "magnesium_recheck",
            "gap_type": "confirmation_gap",
            "gap_value": 0.62,
            "candidate_score_at_decision": 0.99,
            "expected_transition": {"positive": "PrimaryEligible"},
            "closure_exams": ["\u9541\u8d1f\u8377\u8bd5\u9a8c"],
        }
        second = {
            "gap_id": "G-B",
            "candidate": "\u7ef4\u751f\u7d20D\u7f3a\u4e4f\u6027\u4f5d\u507b\u75c5",
            "entity_id": "D100010",
            "target_evidence": "vitamin_d_low",
            "gap_type": "confirmation_gap",
            "gap_value": 0.91,
            "candidate_score_at_decision": 0.41,
            "expected_transition": {"positive": "PrimaryEligible"},
            "closure_exams": ["\u7ef4\u751f\u7d20D\u68c0\u6d4b"],
        }

        def run(gaps):
            return strategy.recommend(
                collected_info={"symptoms": ["\u817f\u75db"]},
                candidate_diseases=[],
                proposed_items=[],
                existing_results={},
                judge_decision={
                    "primary_status": "deferred",
                    "needs_discriminating_exams": True,
                    "active_evidence_gaps": gaps,
                },
            )["items"]

        self.assertEqual(run([first, second]), run([
            {**first, "candidate_score_at_decision": 0.10},
            {**second, "candidate_score_at_decision": 0.99},
        ]))

    def test_leukemia_deferred_gap_closure_beats_urinary_and_generic_exams(self):
        strategy = ExamStrategyAgent(
            KnowledgeBase("data/ref_data"),
            max_new_items=2,
            discriminating_exam_max_items=2,
        )
        leukemia = "\u767d\u8840\u75c5"
        marrow = "\u9aa8\u9ad3\u7a7f\u523a\u548c\u6d3b\u68c0\uff08BMAB\uff09"
        flow = "\u6d41\u5f0f\u7ec6\u80de\u672f\u514d\u75ab\u5206\u578b"
        anca = "\u6297\u4e2d\u6027\u7c92\u7ec6\u80de\u80de\u8d28\u6297\u4f53\uff08ANCA\uff09\u8c31"
        ua = "\u5c3f\u6db2\u5206\u6790\uff08UA\uff09"
        dre = "\u76f4\u80a0\u6307\u68c0\uff08DRE\uff09"

        result = strategy.recommend(
            collected_info={
                "symptoms": ["\u53d1\u70ed", "\u4e4f\u529b", "\u76ae\u80a4\u7600\u9752"]
            },
            candidate_diseases=[leukemia, "\u6025\u6027\u7ec6\u83cc\u6027\u524d\u5217\u817a\u708e"],
            proposed_items=[anca, ua, dre],
            existing_results={},
            judge_decision={
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
                "differential_candidates": [
                    leukemia,
                    "\u6025\u6027\u7ec6\u83cc\u6027\u524d\u5217\u817a\u708e",
                ],
                "deferred_gap_closure_tasks": [
                    {
                        "exam": marrow,
                        "target_candidates": [leukemia],
                        "target_findings": ["bone_marrow_blast_confirmed"],
                        "target_gap": "G-LEUKEMIA-01",
                        "target_gaps": ["G-LEUKEMIA-01"],
                        "exam_type": "deferred_gap_closure",
                        "exam_source": "deferred_gap_closure_exam",
                        "priority_override": True,
                        "priority_bucket": "high_value_deferred_gap_closure",
                        "closure_rank": 1,
                        "closure_priority": 100,
                        "gap_diagnostic_coverage": 1.0,
                        "information_gain_hint": 0.99,
                    },
                    {
                        "exam": flow,
                        "target_candidates": [leukemia],
                        "target_findings": ["leukemia_lineage_identified"],
                        "target_gap": "G-LEUKEMIA-02",
                        "target_gaps": ["G-LEUKEMIA-02"],
                        "exam_type": "deferred_gap_closure",
                        "exam_source": "deferred_gap_closure_exam",
                        "priority_override": True,
                        "priority_bucket": "high_value_deferred_gap_closure",
                        "closure_rank": 2,
                        "closure_priority": 96,
                        "gap_diagnostic_coverage": 0.96,
                        "information_gain_hint": 0.98,
                    },
                    {
                        "exam": ua,
                        "target_candidates": ["\u6025\u6027\u7ec6\u83cc\u6027\u524d\u5217\u817a\u708e"],
                        "exam_type": "special_discriminator",
                        "information_gain_hint": 0.7,
                    },
                    {
                        "exam": dre,
                        "target_candidates": ["\u6025\u6027\u7ec6\u83cc\u6027\u524d\u5217\u817a\u708e"],
                        "exam_type": "special_discriminator",
                        "information_gain_hint": 0.7,
                    },
                    {
                        "exam": anca,
                        "target_candidates": ["\u663e\u5fae\u955c\u4e0b\u591a\u8840\u7ba1\u708e"],
                        "exam_type": "conflict_adjudication",
                        "exam_source": "conflict_adjudication_exam",
                        "information_gain_hint": 0.9,
                    },
                ],
            },
        )

        self.assertEqual(result["items"][:2], [marrow, flow])
        self.assertNotIn(ua, result["items"][:2])
        self.assertNotIn(dre, result["items"][:2])
        details = result["exam_authorization_details"]
        marrow_detail = next(item for item in details if item["exam"] == marrow)
        self.assertEqual(marrow_detail["priority_bucket"], "high_value_deferred_gap_closure")
        self.assertEqual(marrow_detail["requested_exam"], marrow)
        self.assertIn("G-LEUKEMIA-01", marrow_detail["target_gaps"])

    def test_leukemia_targeted_followup_prefers_marrow_over_esr_and_cbc(self):
        strategy = ExamStrategyAgent(
            KnowledgeBase("data/ref_data"),
            max_new_items=3,
            discriminating_exam_max_items=3,
        )
        leukemia = "\u767d\u8840\u75c5"
        marrow = "\u9aa8\u9ad3\u7a7f\u523a\u548c\u6d3b\u68c0\uff08BMAB\uff09"
        esr = "\u7ea2\u7ec6\u80de\u6c89\u964d\u7387\uff08ESR\uff09"
        cbc = "\u5168\u8840\u7ec6\u80de\u8ba1\u6570\uff08CBC\uff09"
        smear = "\u5916\u5468\u8840\u6d82\u7247"

        result = strategy.recommend(
            collected_info={
                "symptoms": ["\u53d1\u70ed", "\u4e4f\u529b", "\u76ae\u80a4\u7600\u9752"]
            },
            candidate_diseases=[leukemia],
            proposed_items=[],
            existing_results={},
            judge_decision={
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
                "differential_candidates": [leukemia],
                "discriminating_exam_tasks": [
                    {
                        "exam": esr,
                        "target_candidates": [leukemia],
                        "exam_type": "pattern_anchor_workup",
                        "exam_source": "pattern_anchor_workup_exam",
                        "information_gain_hint": 0.99,
                    },
                    {
                        "exam": cbc,
                        "target_candidates": [leukemia],
                        "exam_type": "pattern_anchor_workup",
                        "exam_source": "pattern_anchor_workup_exam",
                        "information_gain_hint": 0.99,
                    },
                    {
                        "exam": smear,
                        "target_candidates": [leukemia],
                        "exam_type": "evidence_claim_verification",
                        "exam_source": "evidence_claim_followup_exam",
                        "information_gain_hint": 0.99,
                    },
                    {
                        "exam": marrow,
                        "target_candidates": [leukemia],
                        "exam_type": "evidence_claim_verification",
                        "exam_source": "evidence_claim_followup_exam",
                        "information_gain_hint": 0.99,
                    },
                ],
            },
        )

        self.assertEqual(result["items"][0], marrow)
        self.assertIn(smear, result["items"][:3])
        self.assertNotEqual(result["items"][0], esr)

    def test_tb_mpa_lung_cancer_special_exams_beat_generic_inflammation(self):
        result = self.strategy.recommend(
            collected_info={"symptoms": ["咳嗽", "咯血", "夜汗", "尿色深"]},
            candidate_diseases=["肺结核", "显微镜下多血管炎", "肺癌"],
            proposed_items=["全血细胞计数（CBC）", "C反应蛋白（CRP）"],
            existing_results={},
            judge_decision={
                "primary": "肺癌",
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
                "differential_candidates": ["肺结核", "显微镜下多血管炎", "肺癌"],
                "discriminating_exam_tasks": [
                    {
                        "exam": "胸部CT扫描（Chest CT）",
                        "target_candidates": ["肺结核", "显微镜下多血管炎", "肺癌"],
                        "target_findings": ["cavitary_lesion", "pulmonary_hemorrhage", "lung_mass"],
                        "exam_type": "special_discriminator",
                    },
                    {
                        "exam": "抗酸杆菌染色（AFB）",
                        "target_candidates": ["肺结核", "肺癌"],
                        "target_findings": ["afb_positive"],
                        "exam_type": "special_discriminator",
                    },
                    {
                        "exam": "核酸扩增检测（NAAT）",
                        "target_candidates": ["肺结核", "肺癌"],
                        "target_findings": ["tb_naat_positive"],
                        "exam_type": "special_discriminator",
                    },
                    {
                        "exam": "抗中性粒细胞胞质抗体（ANCA）谱",
                        "target_candidates": ["显微镜下多血管炎", "肺癌"],
                        "target_findings": ["anca_positive"],
                        "exam_type": "special_discriminator",
                    },
                    {
                        "exam": "尿液分析（UA）",
                        "target_candidates": ["显微镜下多血管炎", "肺癌"],
                        "target_findings": ["microscopic_hematuria"],
                        "exam_type": "special_discriminator",
                    },
                    {
                        "exam": "全血细胞计数（CBC）",
                        "target_candidates": ["肺结核", "显微镜下多血管炎", "肺癌"],
                        "target_findings": ["anemia"],
                        "exam_type": "generic_inflammation",
                    },
                ],
            },
        )
        self.assertTrue(result["differential_driven"])
        self.assertLessEqual(len(result["items"]), 6)
        self.assertIn("抗酸杆菌染色（AFB）", result["items"][:4])
        self.assertIn("抗中性粒细胞胞质抗体（ANCA）谱", result["items"][:4])
        self.assertNotIn("C反应蛋白（CRP）", result["items"][:4])
        self.assertTrue(
            all(detail["target_candidates"] for detail in result["exam_authorization_details"])
        )
        self.assertGreaterEqual(
            sum(
                1
                for detail in result["exam_authorization_details"]
                if len(detail["target_candidates"]) >= 2
            ),
            4,
        )

    def test_deferred_judge_state_takes_priority_over_strict_primary(self):
        result = self.strategy.recommend(
            collected_info={"symptoms": ["皮肤瘙痒", "低热", "腹股沟淋巴结肿大"]},
            candidate_diseases=["水痘", "雅司病"],
            proposed_items=["体格检查", "血清学抗体检测", "胸部X线"],
            existing_results={},
            judge_decision={
                "primary": "水痘",
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
                "provisional_primary": "水痘",
                "differential_candidates": ["水痘", "雅司病", "白血病"],
                "discriminating_exams": [
                    "全血细胞计数（CBC）",
                    "外周血涂片",
                    "梅毒血清学检查",
                    "体格检查",
                ],
            },
        )
        self.assertTrue(result["differential_driven"])
        self.assertFalse(result["strict_diagnosis_driven"])
        self.assertTrue(
            all(
                item["allowed_reason"] == "needs_discriminating_exams"
                for item in result["exam_authorization_details"]
            )
        )
        self.assertIn("梅毒血清学检查", result["items"])
        self.assertNotIn("胸部X线检查（CXR）", result["items"])

    def test_judge_needing_discrimination_blocks_legacy_package_without_targets(self):
        result = self.strategy.recommend(
            collected_info={
                "symptoms": [
                    "\u70ed\u5e26\u5730\u533a\u751f\u6d3b\u540e\u6df1\u90e8\u6e83\u75a1\u7ed3\u75c2",
                    "\u5c40\u90e8\u9aa8\u819c\u708e",
                ]
            },
            candidate_diseases=["\u96c5\u53f8\u75c5", "\u6e7f\u75b9"],
            proposed_items=[
                "\u8179\u90e8\u8d85\u58f0",
                "\u809d\u529f\u80fd\u68c0\u67e5\uff08LFTs\uff09",
                "\u7efc\u5408\u4ee3\u8c22\u9762\u677f\uff08CMP\uff09",
                "\u80be\u529f\u80fd",
            ],
            existing_results={},
            judge_decision={
                "primary": "\u96c5\u53f8\u75c5",
                "primary_status": "deferred",
                "needs_discriminating_exams": True,
                "provisional_primary": "\u96c5\u53f8\u75c5",
                "differential_candidates": ["\u96c5\u53f8\u75c5", "\u6e7f\u75b9"],
                "discriminating_exams": [],
            },
        )
        self.assertEqual(result["items"], [])
        self.assertFalse(result["strict_diagnosis_driven"])
        self.assertFalse(result["differential_driven"])
        self.assertIn("\u8179\u90e8\u8d85\u58f0", result["blocked_items"])
        self.assertIn("\u809d\u529f\u80fd\u68c0\u67e5\uff08LFTs\uff09", result["blocked_items"])
        self.assertIn("\u7efc\u5408\u4ee3\u8c22\u9762\u677f\uff08CMP\uff09", result["blocked_items"])
        self.assertIn("\u80be\u529f\u80fd\u68c0\u67e5\uff08RFTs\uff09", result["blocked_items"])

    def test_completed_exam_is_not_reordered(self):
        result = self.strategy.recommend(
            collected_info={"symptoms": ["尿频", "尿急", "尿痛"]},
            candidate_diseases=["泌尿系感染", "肾结石"],
            proposed_items=["尿常规", "腹部B超"],
            existing_results={"尿常规": {"status": "normal"}},
        )
        self.assertNotIn("尿常规", result["items"])

    def test_electrolyte_crisis_path_prioritizes_metabolic_exams(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=8)
        result = strategy.recommend(
            collected_info={
                "symptoms": ["腹泻", "手足抽筋", "心悸", "意识模糊"],
                "chief_complaint": "腹泻后手足抽筋和心悸",
            },
            candidate_diseases=["心律失常", "低镁血症"],
            proposed_items=["经食管超声心动图", "心脏MRI"],
            existing_results={},
        )
        self.assertIn("综合代谢面板（CMP）", result["items"])
        self.assertIn("24小时尿电解质检测", result["items"])
        self.assertIn("镁负荷试验", result["items"])
        self.assertIn("心电图（ECG）", result["items"])
        self.assertIn("综合代谢面板（CMP）", result["strong_verification_items"])
        self.assertNotIn("电解质", result["items"])
        self.assertNotIn("肾功能检查（RFTs）", result["items"])
        self.assertNotIn("经食管超声心动图（TEE）", result["items"])
        self.assertNotIn("心脏MRI（CMR）", result["items"])

    def test_pulmonary_renal_path_suppresses_advanced_cardiac_package(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=8)
        result = strategy.recommend(
            collected_info={
                "symptoms": ["咳血痰", "尿色加深", "脚踝水肿", "气短"],
                "chief_complaint": "咳血痰、气短、尿色加深",
            },
            candidate_diseases=["冠心病", "显微镜下多血管炎"],
            proposed_items=["心肌酶谱", "经食管超声心动图", "三维超声心动图"],
            existing_results={},
        )
        self.assertIn("尿液分析（UA）", result["items"])
        self.assertIn("肾功能检查（RFTs）", result["items"])
        self.assertIn("胸部CT扫描（Chest CT）", result["items"])
        self.assertIn("抗中性粒细胞胞质抗体（ANCA）谱", result["items"])
        self.assertIn("MPO-ANCA", result["items"])
        self.assertIn("红细胞沉降率（ESR）", result["items"])
        self.assertIn("C反应蛋白（CRP）", result["items"])
        self.assertIn("凝血功能全套", result["items"])
        self.assertNotIn("经食管超声心动图（TEE）", result["items"])
        self.assertNotIn("三维超声心动图（3D Echo）", result["items"])

    def test_metabolic_bone_path_uses_rickets_workup(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=6)
        result = strategy.recommend(
            collected_info={"symptoms": ["腿痛", "跛行", "运动耐力下降"]},
            candidate_diseases=["维生素D缺乏性佝偻病"],
            proposed_items=["超声心动图", "胸部X线"],
            existing_results={},
        )
        self.assertEqual(
            result["items"][:6],
            [
                "维生素D检测",
                "血清电解质",
                "甲状旁腺激素检测（PTH）",
                "肝功能检查（LFTs）",
                "骨转换标志物（BTMs）",
                "X线检查",
            ],
        )
        self.assertNotIn("经食管超声心动图（TEE）", result["items"])

    def test_aspiration_path_uses_pulmonary_infection_and_atelectasis_workup(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=12)
        result = strategy.recommend(
            collected_info={"symptoms": ["呛咳", "咳嗽", "发热", "呼吸困难"]},
            candidate_diseases=["肺不张", "支气管肺炎"],
            proposed_items=["超声心动图", "心脏MRI"],
            existing_results={},
        )
        self.assertIn("脉搏血氧饱和度监测（SpO2）", result["items"])
        self.assertIn("动脉血气（ABG）", result["items"])
        self.assertIn("胸部X线检查（CXR）", result["items"])
        self.assertIn("全血细胞计数（CBC）", result["items"])
        self.assertIn("支气管镜检查", result["items"])
        self.assertIn("抗菌药物敏感性试验（AST）", result["items"])
        self.assertNotIn("心脏MRI（CMR）", result["items"])

    def test_advanced_cardiac_exams_remain_when_structural_signal_is_explicit(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=8)
        result = strategy.recommend(
            collected_info={
                "symptoms": ["活动后气短", "下肢水肿"],
                "physical_signs": "心尖部收缩期杂音，考虑二尖瓣反流",
            },
            candidate_diseases=["二尖瓣反流", "心力衰竭"],
            proposed_items=["经食管超声心动图", "心脏MRI"],
            existing_results={},
        )
        self.assertIn("经食管超声心动图（TEE）", result["items"])
        self.assertIn("心脏MRI（CMR）", result["items"])

    def test_conduction_path_uses_ecg_and_holter_not_low_magnesium_package_only(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=6)
        result = strategy.recommend(
            collected_info={"symptoms": ["近晕厥", "头晕", "心动过缓"]},
            candidate_diseases=["二度房室传导阻滞"],
            proposed_items=["电解质"],
            existing_results={},
        )
        self.assertIn("心电图（ECG）", result["items"])
        self.assertIn("动态心电图（Holter）", result["items"])
        self.assertIn("体格检查", result["items"])
        self.assertNotIn("血清电解质", result["items"])
        self.assertNotIn("肾功能检查（RFTs）", result["items"])
        self.assertTrue(result["strict_diagnosis_driven"])

    def test_acute_tympanitis_uses_otoscopy_not_microtia_package(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=6)
        result = strategy.recommend(
            collected_info={"symptoms": ["急性耳痛", "耳鸣", "听力下降"]},
            candidate_diseases=["急性鼓膜炎"],
            proposed_items=["听性脑干反应", "颞骨CT"],
            existing_results={},
        )
        self.assertIn("耳镜检查", result["items"])
        self.assertIn("体格检查", result["items"])
        self.assertIn("全血细胞计数（CBC）", result["items"])
        self.assertNotIn("听性脑干反应（ABR）", result["items"])
        self.assertNotIn("颞骨CT扫描（颞骨CT）", result["items"])
        self.assertTrue(result["strict_diagnosis_driven"])

    def test_crigler_najjar_path_uses_bilirubin_genetic_workup(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=6)
        result = strategy.recommend(
            collected_info={"symptoms": ["黄疸", "巩膜黄染", "嗜睡"]},
            candidate_diseases=["克里格勒-纳贾尔综合征"],
            proposed_items=["血常规"],
            existing_results={},
        )
        self.assertIn("肝功能检查（LFTs）", result["items"])
        self.assertIn("凝血功能全套", result["items"])
        self.assertIn("腹部超声", result["items"])
        self.assertIn("基因检测", result["items"])

    def test_chronic_nasopharyngitis_path_uses_ent_exams(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=4)
        result = strategy.recommend(
            collected_info={"symptoms": ["咽部异物感", "咽干", "反复清嗓"]},
            candidate_diseases=["慢性鼻咽炎"],
            proposed_items=["胸部X线"],
            existing_results={},
        )
        self.assertIn("鼻咽镜检查", result["items"])
        self.assertIn("脱落细胞学检查", result["items"])
        self.assertNotIn("胸部X线检查（CXR）", result["items"][:2])

    def test_microtia_path_uses_abr_temporal_ct_and_gene_test(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=5)
        result = strategy.recommend(
            collected_info={"symptoms": ["出生即有小耳", "耳廓畸形", "听力下降"]},
            candidate_diseases=["小耳畸形"],
            proposed_items=["胸部X线"],
            existing_results={},
        )
        self.assertIn("听性脑干反应（ABR）", result["items"])
        self.assertIn("颞骨CT扫描（颞骨CT）", result["items"])
        self.assertIn("基因检测", result["items"])
        self.assertNotIn("胸部X线检查（CXR）", result["items"][:3])

    def test_acute_bacterial_prostatitis_path_uses_dre_culture_ast(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=8)
        result = strategy.recommend(
            collected_info={"symptoms": ["发热", "寒战", "尿频", "尿急", "尿痛", "会阴痛"]},
            candidate_diseases=["急性细菌性前列腺炎"],
            proposed_items=["尿动力学"],
            existing_results={},
        )
        self.assertIn("直肠指检（DRE）", result["items"])
        self.assertIn("尿液分析（UA）", result["items"])
        self.assertIn("尿培养", result["items"])
        self.assertIn("抗菌药物敏感性试验（AST）", result["items"])
        self.assertIn("前列腺超声", result["items"])
        self.assertNotIn("尿动力学检查（UDS）", result["items"][:6])

    def test_corrective_exam_gate_blocks_advanced_cardiac_without_signal(self):
        strategy = ExamStrategyAgent(self.strategy.knowledge, max_new_items=8)
        items = strategy.prepare_order_items(
            ["经食管超声心动图", "心脏MRI", "三维超声心动图", "电解质"],
            collected_info={"symptoms": ["腿痛", "跛行", "运动耐量下降"]},
            candidate_diseases=["维生素D缺乏性佝偻病"],
            existing_results={},
            max_items=8,
        )
        self.assertNotIn("经食管超声心动图（TEE）", items)
        self.assertNotIn("心脏MRI（CMR）", items)
        self.assertNotIn("三维超声心动图（3D Echo）", items)
        self.assertIn("维生素D检测", items)

    def test_generic_electrolyte_does_not_globally_become_cmp(self):
        normalized, invalid = self.strategy.knowledge.normalize_examinations(["电解质"])
        self.assertEqual(normalized, ["电解质"])
        self.assertEqual(invalid, [])

    def test_new_backend_standard_aliases_normalize(self):
        normalized, invalid = self.strategy.knowledge.normalize_examinations(
            ["镁负荷", "24小时尿电解质", "PTH", "BTMs"]
        )
        self.assertEqual(
            normalized,
            [
                "镁负荷试验",
                "24小时尿电解质检测",
                "甲状旁腺激素检测（PTH）",
                "骨转换标志物（BTMs）",
            ],
        )
        self.assertEqual(invalid, [])


if __name__ == "__main__":
    unittest.main()
