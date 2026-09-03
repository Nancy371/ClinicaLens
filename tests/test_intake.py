import unittest

from agent.intake import (
    apply_consultation_correction,
    apply_consultation_turn,
    intake_answer,
    new_consultation_state,
)


class ConsultationStateTests(unittest.TestCase):
    def complete_state(self):
        state = new_consultation_state(session_id="test-session")
        state = apply_consultation_turn(state, "我最近胸闷和咳嗽，很担心", danger_signs=[])
        state = apply_consultation_turn(state, "3 天前慢慢开始，现在加重，最严重 7 分")
        state = apply_consultation_turn(state, "有血丝痰和活动后气短，没有发热、晕厥或腿肿")
        state = apply_consultation_turn(state, "有高血压，2012 年做过阑尾手术")
        state = apply_consultation_turn(state, "服用氨氯地平，没有已知药物、食物或造影剂过敏")
        state = apply_consultation_turn(state, "家里没有类似疾病，已经戒烟，偶尔接触装修粉尘")
        return state

    def test_grouped_questions_track_known_and_missing_information(self):
        state = new_consultation_state(session_id="test-session")
        state = apply_consultation_turn(state, "我胸口不舒服", danger_signs=[])
        self.assertTrue(state["safety_screened"])
        self.assertEqual(state["current_stage"], "SYMPTOM_CHARACTERIZATION")
        state = apply_consultation_turn(state, "3 天前慢慢开始")
        self.assertIn("severity", state["missing_required_fields"])
        self.assertEqual(state["pending_question"]["fields"], ["severity"])
        self.assertIn("最严重程度", state["pending_question"]["text"])

    def test_safety_screen_is_not_repeated_without_new_signal(self):
        state = new_consultation_state()
        state = apply_consultation_turn(state, "我胸闷", danger_signs=[])
        checked_at = state["safety_checked_at"]
        state = apply_consultation_turn(state, "2 天前突然开始，最严重 6 分")
        self.assertTrue(state["safety_screened"])
        self.assertEqual(state["safety_checked_at"], checked_at)
        self.assertNotEqual(state["pending_question"]["id"], "safety_screen")

    def test_new_related_symptom_requires_safety_recheck(self):
        state = new_consultation_state()
        state = apply_consultation_turn(state, "我咳嗽", danger_signs=[])
        state = apply_consultation_turn(state, "刚才开始咯血")
        self.assertFalse(state["safety_screened"])
        self.assertEqual(state["pending_question"]["id"], "safety_screen")
        self.assertIn("重新确认", state["pending_question"]["reason"])

    def test_strong_red_flag_interrupts_normal_intake(self):
        state = new_consultation_state()
        state = apply_consultation_turn(state, "我现在大量咯血，坐着也喘")
        answer = intake_answer(state)
        self.assertEqual(state["completion_status"], "emergency_interrupted")
        self.assertEqual(answer["urgency"], "emergency")
        self.assertIn("立即急诊", answer["direct_answer"])

    def test_summary_confirmation_is_required_and_correction_reopens_it(self):
        state = self.complete_state()
        self.assertEqual(state["completion_status"], "awaiting_summary_confirmation")
        self.assertFalse(state["summary_confirmed"])
        state = apply_consultation_turn(state, "", confirm_summary=True)
        self.assertEqual(state["completion_status"], "ready_for_assessment")
        state = apply_consultation_correction(state, "allergy", "青霉素过敏，曾出现皮疹", reason="刚刚想起")
        self.assertEqual(state["completion_status"], "awaiting_summary_confirmation")
        self.assertFalse(state["summary_confirmed"])
        self.assertIn("没有已知药物、食物或造影剂过敏", state["corrections"][-1]["old_value"])
        self.assertIn("青霉素", state["summary"]["用药与过敏"])


if __name__ == "__main__":
    unittest.main()
