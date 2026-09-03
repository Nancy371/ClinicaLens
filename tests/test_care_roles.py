import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.care import CareError, CareRuntime, LocalDocumentStore, SQLiteCareRepository
from agent.care_product import sample_clinical_history
from agent.care_roles import hydrate_journey_v3


class CareRoleDomainTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.env = patch.dict(
            os.environ,
            {
                "CARE_PUBLIC_DEPLOYMENT": "false",
                "CARE_AUTH_SECRET": "role-test-secret-that-is-long-enough",
                "CARE_DEV_OTP_CODE": "246810",
                "CARE_DEV_CLINICIAN_PHONES": "13900000000",
            },
            clear=False,
        )
        self.env.start()
        self.runtime = CareRuntime(
            repository=SQLiteCareRepository(root / "care.db"),
            document_store=LocalDocumentStore(root / "uploads", public_deployment=False),
        )
        await self.runtime.start()
        self.patient = await self.login("13800138000")
        self.clinician = await self.login("13900000000")
        journeys = await self.runtime.list_journeys(self.patient["id"])
        self.journey_id = journeys[0]["id"]

    async def asyncTearDown(self):
        await self.runtime.close()
        self.env.stop()
        self.tempdir.cleanup()

    async def login(self, phone):
        await self.runtime.request_otp(phone, f"client-{phone}")
        result = await self.runtime.verify_otp(phone, "246810")
        return result["user"]

    async def authorize(self):
        grant = await self.runtime.create_care_access_grant(self.patient["id"], self.journey_id)
        return await self.runtime.redeem_care_access_grant(self.clinician["id"], grant["code"])

    async def test_roles_are_server_assigned(self):
        self.assertEqual(self.patient["role"], "patient")
        self.assertEqual(self.clinician["role"], "clinician")
        self.assertEqual(await self.runtime.list_clinician_journeys(self.clinician["id"]), [])

    async def test_patient_projection_omits_clinician_reasoning(self):
        payload = await self.runtime.public_sample("patient")
        self.assertEqual(payload["projection"], "patient")
        self.assertNotIn("assessment_versions", payload)
        self.assertNotIn("exam_recommendations", payload)
        self.assertNotIn("treatment_recommendations", payload)
        self.assertNotIn("raw_case_document", payload)
        self.assertNotIn("consultation_case_documents", payload)
        self.assertNotIn("evidence", payload)
        self.assertGreater(len(payload["exam_reports"]), 0)
        self.assertGreater(len(payload["patient_explanations"]), 0)

    async def test_clinician_projection_requires_single_use_grant(self):
        grant = await self.runtime.create_care_access_grant(self.patient["id"], self.journey_id)
        link = await self.runtime.redeem_care_access_grant(self.clinician["id"], grant["code"])
        self.assertEqual(link["status"], "active")
        payload = await self.runtime.get_clinician_journey(self.clinician["id"], self.journey_id)
        self.assertEqual(payload["projection"], "clinician")
        self.assertIn("raw_case_document", payload)
        self.assertIn("consultation_case_documents", payload)
        self.assertIn("assessment_versions", payload)
        self.assertIn("exam_recommendations", payload)
        with self.assertRaises(CareError) as context:
            await self.runtime.redeem_care_access_grant(self.clinician["id"], grant["code"])
        self.assertEqual(context.exception.code, "access_grant_used")

    async def test_revocation_immediately_removes_clinician_access(self):
        link = await self.authorize()
        await self.runtime.get_clinician_journey(self.clinician["id"], self.journey_id)
        await self.runtime.revoke_care_team_link(self.patient["id"], str(link["id"]))
        with self.assertRaises(CareError) as context:
            await self.runtime.get_clinician_journey(self.clinician["id"], self.journey_id)
        self.assertEqual(context.exception.code, "clinician_access_denied")

    async def test_expired_grant_fails_closed(self):
        code = "ABCDEFGH"
        await self.runtime.repository.create_access_grant(
            self.patient["id"], self.journey_id,
            self.runtime._digest(f"care-access:{code}"), time.time() - 1,
        )
        with self.assertRaises(CareError) as context:
            await self.runtime.redeem_care_access_grant(self.clinician["id"], code)
        self.assertEqual(context.exception.code, "access_grant_expired")

    async def test_exam_report_rows_preserve_source_range_and_reviewed_explanation(self):
        payload = await self.runtime.public_sample("patient")
        observations = [item for report in payload["exam_reports"] for item in report["observations"]]
        self.assertTrue(observations)
        required = {"name", "value", "unit", "reference_range_display", "interpretation_status", "patient_explanation", "diagnostic_impact", "source_locator", "trend", "entered_assessment_version"}
        self.assertTrue(all(required.issubset(item) for item in observations))
        self.assertTrue(any(item["reference_range_display"] == "原报告未提供" for item in observations))
        self.assertTrue(all(item["patient_explanation"] for item in observations))

    async def test_patient_reasoning_graph_uses_public_node_and_edge_contract(self):
        payload = await self.runtime.public_sample("patient")
        explanation = payload["patient_explanations"][-1]
        graph = explanation["reasoning_graph"]
        self.assertEqual(graph["schema_version"], "patient-reasoning-graph.v1")
        node_ids = {item["id"] for item in graph["nodes"]}
        node_types = {item["type"] for item in graph["nodes"]}
        self.assertTrue({"symptom", "exam_result", "evidence", "hypothesis", "diagnosis"}.issubset(node_types))
        self.assertTrue(all(edge["source"] in node_ids and edge["target"] in node_ids for edge in graph["edges"]))
        self.assertIn("supports", {edge["relation"] for edge in graph["edges"]})
        serialized = json.dumps(graph, ensure_ascii=False)
        for internal_name in ("CandidateScore", "ClaimResolutionLedger", "AnchorSatisfied", "PrimaryEligible", "Comparator"):
            self.assertNotIn(internal_name, serialized)
        levels = explanation["language_levels"]
        self.assertIn("肺部异常和肾脏损伤", levels["level_1"])
        self.assertGreaterEqual(len(levels["level_2"]), 3)
        self.assertGreater(len(levels["level_3"]["terms"]), 0)

    async def test_legacy_string_result_migrates_without_invented_range(self):
        migrated = hydrate_journey_v3({
            "id": "legacy", "records": [{
                "id": "old-1", "title": "旧检查", "items": ["未结构化结果文本"],
                "verification_status": "imported", "source": {"type": "legacy", "locator": "旧记录第1行"},
            }], "synced_batches": [], "assessment_versions": [],
        })
        report = migrated["exam_reports"][0]
        observation = report["observations"][0]
        self.assertEqual(report["batch_key"], "legacy")
        self.assertEqual(observation["reference_range_display"], "原报告未提供")
        self.assertIn("暂无经过审核", observation["patient_explanation"])

    async def test_case_and_consultation_share_the_same_explanation_version(self):
        payload = await self.runtime.public_sample("patient")
        explanations = {item["assessment_version_id"]: item for item in payload["patient_explanations"]}
        updates = [item for item in payload["consultation"]["messages"] if item.get("kind") == "assessment_update"]
        self.assertEqual(set(explanations), {item["assessment_version_id"] for item in updates})
        for update in updates:
            self.assertEqual(update["patient_explanation"], explanations[update["assessment_version_id"]])

    async def test_disputed_hospital_report_withdraws_current_assessment(self):
        sample = await self.runtime.public_sample("legacy")
        sample["id"] = self.journey_id
        sample["owner_id"] = self.patient["id"]
        await self.runtime.repository.save_journey(self.patient["id"], sample)
        result = await self.runtime.dispute_exam_report(
            self.patient["id"], self.journey_id, "report-immunology", "医院原件数值不同"
        )
        self.assertTrue(result["assessment_withdrawn"])
        internal = await self.runtime.get_journey(self.patient["id"], self.journey_id)
        self.assertIsNone(internal["assessment"])
        self.assertEqual(next(item for item in internal["exam_reports"] if item["id"] == "report-immunology")["verification_status"], "disputed")
        self.assertEqual(next(item for item in internal["records"] if item["id"] == "record-anca")["verification_status"], "needs_correction")

    async def test_exam_decision_requires_reason_and_creates_only_sandbox_order(self):
        await self.authorize()
        with self.assertRaises(CareError) as context:
            await self.runtime.decide_exam_recommendation(
                self.clinician["id"], self.journey_id, "blood-safety", {"action": "confirmed", "rationale": ""}
            )
        self.assertEqual(context.exception.code, "recommendation_rationale_required")
        result = await self.runtime.decide_exam_recommendation(
            self.clinician["id"], self.journey_id, "blood-safety",
            {"action": "modified", "rationale": "结合当前出血风险", "edits": {"items": ["血常规", "血红蛋白动态"]}},
        )
        self.assertEqual(result["exam_order"]["status"], "sandbox_ordered")
        self.assertEqual(result["exam_order"]["source"]["type"], "clinician")
        self.assertIn("未向真实医院下单", result["exam_order"]["notice"])

    async def test_treatment_confirmation_creates_draft_but_not_patient_tasks(self):
        await self.authorize()
        clinician_journey = await self.runtime.get_clinician_journey(self.clinician["id"], self.journey_id)
        recommendation = clinician_journey["treatment_recommendations"][0]
        self.assertTrue(recommendation["goals"])
        self.assertTrue(recommendation["prerequisites"])
        before = clinician_journey["medications"]
        result = await self.runtime.decide_treatment_recommendation(
            self.clinician["id"], self.journey_id, "aav-guideline-path",
            {"action": "confirmed", "rationale": "路径方向可作为专科讨论基础"},
        )
        after = (await self.runtime.get_clinician_journey(self.clinician["id"], self.journey_id))["medications"]
        self.assertTrue(result["created_prescription_draft"])
        self.assertEqual(result["prescription_draft"]["status"], "draft")
        self.assertFalse(result["created_medication_task"])
        self.assertEqual(before, after)

    async def test_second_clinician_signature_creates_prescription_and_reminders(self):
        await self.runtime.connect_hospital(self.patient["id"], True)
        await self.runtime.update_clinical_history(
            self.patient["id"], self.journey_id, sample_clinical_history(confirmed=True)
        )
        await self.runtime.apply_doctor_document(
            self.patient["id"], self.journey_id,
            {"source_type": "sandbox_hospital", "diagnoses": ["显微镜下多血管炎（沙箱医生确认）"], "prescriptions": []},
        )
        await self.authorize()
        decision = await self.runtime.decide_treatment_recommendation(
            self.clinician["id"], self.journey_id, "aav-guideline-path",
            {"action": "confirmed", "rationale": "结合医生确诊选择指南路径"},
        )
        draft = decision["prescription_draft"]
        self.assertGreaterEqual(len(draft["items"]), 2)
        before = await self.runtime.get_journey(self.patient["id"], self.journey_id)
        self.assertEqual(before["medications"], [])
        signed = await self.runtime.sign_prescription_draft(
            self.clinician["id"], self.journey_id, draft["id"],
            {"rationale": "已核对诊断、过敏、筛查与剂量", "acknowledgements": ["diagnosis", "allergies", "screening", "dose"]},
        )
        self.assertEqual(signed["signed_prescription"]["status"], "signed")
        self.assertEqual(len(signed["medications"]), len(draft["items"]))
        self.assertTrue(all(item["source"]["type"] == "clinician_signed_ai_path" for item in signed["medications"]))
        self.assertEqual(len(signed["reminders"]), len(draft["items"]))
        patient = await self.runtime.get_patient_journey(self.patient["id"], self.journey_id)
        self.assertNotIn("prescription_drafts", patient)
        self.assertEqual(len(patient["medications"]), len(draft["items"]))

    async def test_prescription_signature_requires_doctor_confirmed_diagnosis(self):
        await self.authorize()
        decision = await self.runtime.decide_treatment_recommendation(
            self.clinician["id"], self.journey_id, "aav-guideline-path",
            {"action": "confirmed", "rationale": "先形成路径草稿"},
        )
        with self.assertRaises(CareError) as context:
            await self.runtime.sign_prescription_draft(
                self.clinician["id"], self.journey_id, decision["prescription_draft"]["id"],
                {"rationale": "尝试签署", "acknowledgements": ["diagnosis", "allergies", "screening", "dose"]},
            )
        self.assertEqual(context.exception.code, "doctor_diagnosis_required")

    async def test_consultation_case_document_is_traceable_and_excludes_exam_results(self):
        await self.runtime.send_consultation(
            self.patient["id"], self.journey_id, "最近三天有血丝痰，活动后气短。", []
        )
        document = await self.runtime.generate_consultation_case_document(self.patient["id"], self.journey_id)
        serialized = json.dumps(document, ensure_ascii=False)
        self.assertEqual(document["origin"], "consultation")
        self.assertTrue(document["generated_from_message_ids"])
        self.assertNotIn("MPO-ANCA 86", serialized)
        self.assertNotIn("肾活检示", serialized)
        confirmed = await self.runtime.confirm_consultation_case_document(
            self.patient["id"], self.journey_id, document["id"]
        )
        self.assertEqual(confirmed["status"], "confirmed")

    async def test_dangerous_conditions_link_to_specific_exam_recommendations(self):
        sample = await self.runtime.public_sample("clinician")
        matrix = sample["assessment_versions"][-1]["safety_matrix"]
        recommendation_ids = {item["id"] for item in sample["exam_recommendations"]}
        self.assertGreaterEqual(len(matrix), 5)
        for condition in matrix:
            self.assertTrue(condition["exam_items"])
            self.assertTrue(condition["exam_links"])
            self.assertTrue(all(link["recommendation_id"] in recommendation_ids for link in condition["exam_links"]))


if __name__ == "__main__":
    unittest.main()
