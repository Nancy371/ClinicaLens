import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.care import CareError, CareRuntime, LocalDocumentStore, SQLiteCareRepository


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
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(payload["projection"], "patient")
        self.assertNotIn("assessment_versions", payload)
        self.assertNotIn("exam_recommendations", payload)
        self.assertNotIn("treatment_recommendations", payload)
        self.assertNotIn('"evidence"', serialized)
        self.assertGreater(len(payload["exam_reports"]), 0)
        self.assertGreater(len(payload["patient_explanations"]), 0)

    async def test_clinician_projection_requires_single_use_grant(self):
        grant = await self.runtime.create_care_access_grant(self.patient["id"], self.journey_id)
        link = await self.runtime.redeem_care_access_grant(self.clinician["id"], grant["code"])
        self.assertEqual(link["status"], "active")
        payload = await self.runtime.get_clinician_journey(self.clinician["id"], self.journey_id)
        self.assertEqual(payload["projection"], "clinician")
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
        required = {"name", "value", "unit", "reference_range_display", "interpretation_status", "patient_explanation", "diagnostic_impact", "source_locator"}
        self.assertTrue(all(required.issubset(item) for item in observations))
        self.assertTrue(any(item["reference_range_display"] == "原报告未提供" for item in observations))
        self.assertTrue(all(item["patient_explanation"] for item in observations))

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

    async def test_treatment_decision_never_creates_medication_or_prescription(self):
        await self.authorize()
        before = (await self.runtime.get_clinician_journey(self.clinician["id"], self.journey_id))["medications"]
        result = await self.runtime.decide_treatment_recommendation(
            self.clinician["id"], self.journey_id, "aav-guideline-path",
            {"action": "confirmed", "rationale": "路径方向可作为专科讨论基础"},
        )
        after = (await self.runtime.get_clinician_journey(self.clinician["id"], self.journey_id))["medications"]
        self.assertFalse(result["created_prescription"])
        self.assertFalse(result["created_medication_task"])
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
