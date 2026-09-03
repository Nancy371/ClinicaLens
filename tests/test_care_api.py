import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import CookieJar, FormData
from aiohttp.test_utils import TestClient, TestServer

from agent.care import CareRuntime, LocalDocumentStore, SQLiteCareRepository
from agent.server import create_app


class CareApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.env = patch.dict(
            os.environ,
            {
                "CARE_PUBLIC_DEPLOYMENT": "false",
                "CARE_AUTH_SECRET": "test-secret-that-is-long-enough",
                "CARE_DEV_OTP_CODE": "246810",
                "CARE_DEV_CLINICIAN_PHONES": "13900000000",
                "CARE_APPOINTMENT_URL": "https://www.114yygh.com/",
            },
            clear=False,
        )
        self.env.start()
        repository = SQLiteCareRepository(root / "care.db")
        runtime = CareRuntime(
            repository=repository,
            document_store=LocalDocumentStore(root / "uploads", public_deployment=False),
        )
        self.runtime = runtime
        self.client = TestClient(
            TestServer(create_app(care_runtime=runtime)),
            cookie_jar=CookieJar(unsafe=True),
        )
        await self.client.start_server()
        self.csrf = ""

    async def asyncTearDown(self):
        await self.client.close()
        self.env.stop()
        self.tempdir.cleanup()

    async def login(self, phone="13800138000"):
        response = await self.client.post("/api/v1/auth/otp/request", json={"phone": phone})
        self.assertEqual(response.status, 202)
        otp = await response.json()
        response = await self.client.post(
            "/api/v1/auth/otp/verify",
            json={"phone": phone, "code": otp["development_code"]},
        )
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.csrf = payload["csrf_token"]
        return payload, response.cookies["clinicalens_session"].value

    def headers(self, csrf=None):
        return {"X-CSRF-Token": csrf or self.csrf}

    async def journey(self):
        response = await self.client.get("/api/v1/session")
        payload = await response.json()
        return payload["journeys"][0]

    async def sync_and_triage(self):
        await self.login()
        await self.client.post(
            "/api/v1/hospital-connections",
            json={"consent": True},
            headers=self.headers(),
        )
        response = await self.client.post(
            "/api/v1/records/sync",
            json={},
            headers=self.headers(),
        )
        synced = await response.json()
        journey = synced["journey"]
        self.assertTrue(journey["patient_profile"]["is_fictional"])
        self.assertEqual(journey["patient_profile"]["name"], "周予安")
        self.assertEqual(journey["patient_profile"]["hospital_record_no"], "SBX-20260902-001")
        response = await self.client.patch(
            f"/api/v1/journeys/{journey['id']}/clinical-history",
            json={
                **journey["clinical_history"],
                "field_statuses": {
                    "conditions": "confirmed",
                    "surgeries": "confirmed",
                    "current_medications": "confirmed",
                    "allergies": "confirmed",
                    "family_history": "confirmed",
                    "social_history": "confirmed",
                },
            },
            headers=self.headers(),
        )
        self.assertEqual(response.status, 200)
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/triage",
            json={"danger_signs": []},
            headers=self.headers(),
        )
        self.assertEqual(response.status, 200)
        return journey

    async def confirm_all(self, journey):
        for record in journey["records"]:
            response = await self.client.patch(
                f"/api/v1/journeys/{journey['id']}/records/{record['id']}",
                json={"confirmed": True},
                headers=self.headers(),
            )
            self.assertEqual(response.status, 200)

    async def test_otp_creates_private_session_and_seed_journey(self):
        payload, _ = await self.login()
        self.assertNotIn("13800138000", json.dumps(payload, ensure_ascii=False))
        response = await self.client.get("/api/v1/session")
        session = await response.json()
        self.assertTrue(session["authenticated"])
        self.assertEqual(session["user"]["phone_masked"], "138****8000")
        self.assertEqual(session["journeys"][0]["current_stage"], "consultation")
        self.assertEqual(session["user"]["role"], "patient")
        self.assertEqual(session["journeys"][0]["schema_version"], "patient-journey-dto.v1")
        self.assertNotIn("assessment_versions", session["journeys"][0])

    async def test_write_requires_csrf(self):
        await self.login()
        response = await self.client.post(
            "/api/v1/hospital-connections",
            json={"consent": True},
        )
        payload = await response.json()
        self.assertEqual(response.status, 403)
        self.assertEqual(payload["error"], "csrf_validation_failed")

    async def test_hospital_timeout_does_not_claim_success(self):
        await self.login()
        await self.client.post(
            "/api/v1/hospital-connections",
            json={"consent": True},
            headers=self.headers(),
        )
        response = await self.client.post(
            "/api/v1/records/sync",
            json={"simulate": "timeout"},
            headers=self.headers(),
        )
        payload = await response.json()
        self.assertEqual(response.status, 504)
        self.assertEqual(payload["error"], "hospital_sync_timeout")
        journey = await self.journey()
        self.assertEqual(journey["records"], [])

    async def test_danger_sign_stops_assessment(self):
        journey = await self.sync_and_triage()
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/triage",
            json={"danger_signs": ["active_hemoptysis"]},
            headers=self.headers(),
        )
        self.assertEqual((await response.json())["status"], "emergency")
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/assessments",
            json={},
            headers=self.headers(),
        )
        self.assertEqual(response.status, 409)
        self.assertEqual((await response.json())["error"], "emergency_path_active")

    async def test_disputed_records_cannot_enter_assessment(self):
        journey = await self.sync_and_triage()
        response = await self.client.patch(
            f"/api/v1/journeys/{journey['id']}/records/{journey['records'][0]['id']}",
            json={"confirmed": False, "correction": "与医院原件不一致"},
            headers=self.headers(),
        )
        self.assertEqual(response.status, 200)
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/assessments",
            json={},
            headers=self.headers(),
        )
        self.assertEqual(response.status, 409)
        self.assertEqual((await response.json())["error"], "records_not_confirmed")

    async def test_assessment_has_evidence_boundaries_without_treatment(self):
        journey = await self.sync_and_triage()
        await self.confirm_all(journey)
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/assessments",
            json={},
            headers=self.headers(),
        )
        accepted = await response.json()
        self.assertEqual(response.status, 202)
        response = await self.client.get(accepted["poll_url"])
        job = await response.json()
        serialized = json.dumps(job, ensure_ascii=False)
        self.assertNotIn("assessment", job["result"])
        self.assertIn("patient_explanation", job["result"])
        session = await (await self.client.get("/api/v1/session")).json()
        internal = await self.runtime.get_journey(session["user"]["id"], journey["id"])
        assessment = internal["assessment"]
        self.assertEqual(assessment["authority"], "decision_support")
        self.assertGreater(assessment["evidence_summary"]["contradicting_count"], 0)
        exams = assessment["care_navigation"]["exam_discussion_items"]
        self.assertEqual(len(exams), 6)
        self.assertTrue(all(item["status"] == "pending_doctor_confirmation" for item in exams))
        self.assertTrue(any("肾活检" in item["name"] for item in exams))
        self.assertNotIn("treatment_plan", serialized)

    async def test_ai_payload_cannot_create_doctor_plan(self):
        journey = await self.sync_and_triage()
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/doctor-documents",
            json={
                "source_type": "agent",
                "diagnoses": ["显微镜下多血管炎"],
                "prescriptions": [{"name": "药物", "dose": "1片", "frequency": "每日"}],
            },
            headers=self.headers(),
        )
        self.assertEqual(response.status, 409)
        self.assertEqual((await response.json())["error"], "invalid_doctor_source")
        self.assertIsNone((await self.journey())["doctor_plan"])

    async def test_doctor_source_creates_medication_and_event(self):
        journey = await self.sync_and_triage()
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/doctor-documents",
            json={
                "source_type": "sandbox_hospital",
                "diagnoses": ["显微镜下多血管炎（沙箱）"],
                "followup_at": "2026-09-16T09:00:00+08:00",
                "examination_orders": ["复查肾功能与电解质", "由肾内科评估肾活检必要性"],
                "prescriptions": [{"name": "碳酸钙D3片（虚构沙箱处方）", "dose": "1片", "frequency": "每日一次"}],
            },
            headers=self.headers(),
        )
        payload = await response.json()
        self.assertEqual(response.status, 201)
        self.assertEqual(payload["doctor_plan"]["authority"], "doctor_plan")
        self.assertEqual(len(payload["doctor_plan"]["examination_orders"]), 2)
        self.assertTrue(all(item["source"] == "doctor_plan" for item in payload["doctor_plan"]["examination_orders"]))
        medication_id = payload["medications"][0]["id"]
        response = await self.client.post(
            f"/api/v1/medications/{medication_id}/events",
            json={"type": "taken"},
            headers=self.headers(),
        )
        self.assertEqual(response.status, 201)
        self.assertEqual((await response.json())["type"], "taken")

    async def test_document_upload_validates_type_and_keeps_private_metadata(self):
        await self.login()
        form = FormData()
        form.add_field("document_kind", "doctor_visit")
        form.add_field("file", b"%PDF-1.4 test", filename="visit.pdf", content_type="application/pdf")
        response = await self.client.post(
            "/api/v1/record-imports",
            data=form,
            headers=self.headers(),
        )
        payload = await response.json()
        self.assertEqual(response.status, 202)
        self.assertEqual(payload["status"], "awaiting_user_confirmation")
        self.assertNotIn("%PDF", json.dumps(payload))

    async def test_sample_routes_are_read_only_aliases(self):
        response = await self.client.get("/api/sample/metrics")
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["cases"], 7)
        response = await self.client.get("/api/sample/cases/multi-organ-pattern")
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertNotIn("treatment_plan", json.dumps(payload, ensure_ascii=False))
        self.assertEqual(payload["conclusion"]["status"], "requires_doctor_confirmation")

    async def test_public_role_previews_use_distinct_projections(self):
        patient = await (await self.client.get("/api/sample/journey?audience=patient")).json()
        clinician = await (await self.client.get("/api/sample/journey?audience=clinician")).json()
        self.assertEqual(patient["projection"], "patient")
        self.assertEqual(clinician["projection"], "clinician")
        self.assertNotIn("assessment_versions", patient)
        self.assertNotIn("exam_recommendations", patient)
        self.assertNotIn("raw_case_document", patient)
        self.assertNotIn("consultation_case_documents", patient)
        self.assertIn("raw_case_document", clinician)
        self.assertEqual(len(clinician["assessment_versions"]), 4)
        self.assertGreaterEqual(len(clinician["exam_recommendations"]), 8)

    async def test_patient_role_cannot_call_clinician_api(self):
        await self.login()
        response = await self.client.get("/api/v1/clinician/journeys")
        self.assertEqual(response.status, 403)
        self.assertEqual((await response.json())["error"], "role_forbidden")

    async def test_exam_report_api_returns_report_style_rows(self):
        journey = await self.sync_and_triage()
        response = await self.client.get(f"/api/v1/journeys/{journey['id']}/exam-reports")
        payload = await response.json()
        self.assertEqual(response.status, 200)
        self.assertGreater(len(payload["exam_reports"]), 0)
        row = payload["exam_reports"][0]["observations"][0]
        self.assertIn("reference_range_display", row)
        self.assertIn("patient_explanation", row)
        self.assertIn("diagnostic_impact", row)
        self.assertIn("source_locator", row)

    async def test_complete_public_journey_is_browsable_without_login(self):
        response = await self.client.get("/api/sample/journey")
        payload = await response.json()
        self.assertEqual(response.status, 200)
        self.assertTrue(payload["patient_profile"]["is_fictional"])
        self.assertEqual(payload["patient_profile"]["name"], "周予安")
        self.assertEqual(len(payload["raw_case_document"]["sections"]), 5)
        self.assertEqual(payload["raw_case_document"]["origin"], "consultation")
        self.assertTrue(payload["raw_case_document"]["generated_from_message_ids"])
        self.assertNotIn("MPO-ANCA 86", payload["raw_case_document"]["full_text"])
        self.assertEqual(len(payload["assessment_versions"]), 4)
        self.assertEqual(payload["assessment_versions"][1]["primary_diagnosis"]["name"], "肺肾综合征（系统性小血管炎方向）")
        self.assertEqual(len(payload["medications"]), 4)
        self.assertTrue(all(item["source"]["type"] == "clinician_signed_ai_path" for item in payload["medications"]))

    async def test_consultation_runs_safety_rule_before_diagnostic_flow(self):
        await self.login()
        journey = await self.journey()
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/consultation/messages",
            json={"message": "我现在应该去急诊吗？", "danger_signs": ["low_oxygen"]},
            headers=self.headers(),
        )
        payload = await response.json()
        self.assertEqual(response.status, 201)
        self.assertEqual(payload["answer"]["urgency"], "emergency")
        self.assertEqual(payload["triage"]["status"], "emergency")
        self.assertIn("立即急诊", payload["answer"]["direct_answer"])

    async def test_history_must_be_confirmed_or_explicitly_unknown(self):
        await self.login()
        journey = await self.journey()
        response = await self.client.patch(
            f"/api/v1/journeys/{journey['id']}/clinical-history",
            json={"field_statuses": {key: "unknown" for key in ("conditions", "surgeries", "current_medications", "allergies", "family_history", "social_history")}},
            headers=self.headers(),
        )
        payload = await response.json()
        self.assertEqual(response.status, 200)
        self.assertEqual(payload["clinical_history"]["confirmation_status"], "confirmed")
        self.assertTrue(all(value == "unknown" for value in payload["clinical_history"]["field_statuses"].values()))

    async def test_explicitly_unknown_history_increases_assessment_uncertainty(self):
        journey = await self.sync_and_triage()
        history = (await self.journey())["clinical_history"]
        history["field_statuses"]["allergies"] = "unknown"
        await self.client.patch(
            f"/api/v1/journeys/{journey['id']}/clinical-history",
            json=history,
            headers=self.headers(),
        )
        await self.confirm_all(journey)
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/assessments",
            json={}, headers=self.headers(),
        )
        await (await self.client.get((await response.json())["poll_url"])).json()
        session = await (await self.client.get("/api/v1/session")).json()
        internal = await self.runtime.get_journey(session["user"]["id"], journey["id"])
        uncertainty = internal["assessment"]["uncertainty"]
        self.assertEqual(uncertainty["level"], "medium")
        self.assertTrue(any("过敏史" in item and "不了解" in item for item in uncertainty["gaps"]))

    async def test_record_batches_are_ordered_and_do_not_leak_future_results(self):
        await self.login()
        await self.client.post("/api/v1/hospital-connections", json={"consent": True}, headers=self.headers())
        journey = await self.journey()
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/record-batches/organ/sync",
            json={}, headers=self.headers(),
        )
        self.assertEqual(response.status, 409)
        self.assertEqual((await response.json())["error"], "previous_batch_required")
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/record-batches/baseline/sync",
            json={}, headers=self.headers(),
        )
        payload = await response.json()
        serialized = json.dumps(
            {
                "records": payload["journey"]["records"],
                "exam_reports": payload["journey"]["exam_reports"],
            },
            ensure_ascii=False,
        )
        self.assertEqual(response.status, 200)
        self.assertNotIn("MPO-ANCA 86", serialized)
        self.assertNotIn("肾活检示少免疫", serialized)

    async def test_ai_treatment_reference_cannot_create_medication(self):
        payload = await (await self.client.get("/api/sample/journey")).json()
        reference = payload["treatment_reference"]
        serialized = json.dumps(reference, ensure_ascii=False)
        self.assertEqual(reference["authority"], "decision_support")
        self.assertNotIn('"dose"', serialized)
        self.assertNotIn('"prescription"', serialized)
        self.assertEqual(len(payload["medications"]), 4)
        self.assertTrue(all(item["source"]["type"] == "clinician_signed_ai_path" for item in payload["medications"]))

    async def test_consultation_case_document_generation_and_confirmation_api(self):
        await self.login()
        journey = await self.journey()
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/consultation/messages",
            json={"message": "最近三天有血丝痰，活动后气短。", "danger_signs": []},
            headers=self.headers(),
        )
        self.assertEqual(response.status, 201)
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/consultation-case-documents",
            json={}, headers=self.headers(),
        )
        payload = await response.json()
        self.assertEqual(response.status, 201)
        document = payload["case_document"]
        self.assertEqual(document["status"], "draft")
        self.assertTrue(document["generated_from_message_ids"])
        response = await self.client.patch(
            f"/api/v1/journeys/{journey['id']}/consultation-case-documents/{document['id']}",
            json={"corrections": []}, headers=self.headers(),
        )
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["case_document"]["status"], "confirmed")

    async def test_clinician_two_step_prescription_api(self):
        journey = await self.sync_and_triage()
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/doctor-documents",
            json={"source_type": "sandbox_hospital", "diagnoses": ["显微镜下多血管炎（沙箱医生确认）"], "prescriptions": []},
            headers=self.headers(),
        )
        self.assertEqual(response.status, 201)
        response = await self.client.post(
            "/api/v1/care-access-grants", json={"journey_id": journey["id"]}, headers=self.headers()
        )
        grant = await response.json()
        self.assertEqual(response.status, 201)
        await self.login("13900000000")
        response = await self.client.post(
            "/api/v1/care-access-grants/redeem", json={"code": grant["code"]}, headers=self.headers()
        )
        self.assertEqual(response.status, 201)
        response = await self.client.post(
            f"/api/v1/clinician/journeys/{journey['id']}/treatment-recommendations/aav-guideline-path/decision",
            json={"action": "confirmed", "rationale": "采用指南路径并进入剂量核对"}, headers=self.headers(),
        )
        decision = await response.json()
        self.assertEqual(response.status, 201)
        self.assertTrue(decision["created_prescription_draft"])
        response = await self.client.post(
            f"/api/v1/clinician/journeys/{journey['id']}/prescription-drafts/{decision['prescription_draft']['id']}/sign",
            json={"rationale": "完成四项核对并签署", "acknowledgements": ["diagnosis", "allergies", "screening", "dose"]},
            headers=self.headers(),
        )
        signed = await response.json()
        self.assertEqual(response.status, 201)
        self.assertTrue(signed["medications"])
        self.assertTrue(signed["reminders"])

    async def test_allergy_conflict_blocks_doctor_prescription_confirmation(self):
        journey = await self.sync_and_triage()
        history = (await self.journey())["clinical_history"]
        history["allergies"] = [{"allergen": "磺胺", "reaction": "严重皮疹", "severity": "severe", "status": "confirmed"}]
        await self.client.patch(
            f"/api/v1/journeys/{journey['id']}/clinical-history",
            json=history,
            headers=self.headers(),
        )
        response = await self.client.post(
            f"/api/v1/journeys/{journey['id']}/doctor-documents",
            json={"source_type": "sandbox_hospital", "diagnoses": ["显微镜下多血管炎"], "prescriptions": [{"name": "复方磺胺甲噁唑", "dose": "400/80 mg", "frequency": "每日一次"}]},
            headers=self.headers(),
        )
        self.assertEqual(response.status, 409)
        self.assertEqual((await response.json())["error"], "medication_allergy_conflict")

    async def test_pwa_install_assets_are_available(self):
        for path in ("/", "/index.html", "/manifest.webmanifest", "/sw.js"):
            response = await self.client.get(path)
            self.assertEqual(response.status, 200, path)

    async def test_account_export_omits_credentials(self):
        await self.login()
        response = await self.client.get("/api/v1/account/export")
        payload = await response.json()
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(response.status, 200)
        self.assertNotIn("csrf_token", serialized)
        self.assertNotIn("token_hash", serialized)
        self.assertNotIn("13800138000", serialized)

    async def test_second_user_cannot_read_first_users_journey(self):
        await self.login("13800138000")
        first_journey = await self.journey()
        await self.login("13900139000")
        response = await self.client.get(f"/api/v1/journeys/{first_journey['id']}")
        self.assertEqual(response.status, 404)
        self.assertEqual((await response.json())["error"], "journey_not_found")

    async def test_correcting_source_record_withdraws_prior_assessment(self):
        journey = await self.sync_and_triage()
        await self.confirm_all(journey)
        await self.client.post(
            f"/api/v1/journeys/{journey['id']}/assessments",
            json={},
            headers=self.headers(),
        )
        response = await self.client.patch(
            f"/api/v1/journeys/{journey['id']}/records/{journey['records'][0]['id']}",
            json={"confirmed": False, "correction": "原文症状记录有误"},
            headers=self.headers(),
        )
        self.assertEqual(response.status, 200)
        updated = await self.journey()
        self.assertNotIn("assessment", updated)
        self.assertIsNone(updated["appointment_plan"])
        self.assertEqual(updated["current_stage"], "confirm_records")
        session = await (await self.client.get("/api/v1/session")).json()
        internal = await self.runtime.get_journey(session["user"]["id"], journey["id"])
        self.assertIsNone(internal["assessment"])

    async def test_public_auth_fails_closed_without_sms_provider(self):
        with patch.dict(
            os.environ,
            {
                "CARE_PUBLIC_DEPLOYMENT": "true",
                "CARE_AUTH_SECRET": "production-secret-that-is-long-enough",
                "CARE_SMS_PROVIDER_URL": "",
            },
            clear=False,
        ):
            runtime = CareRuntime(
                repository=SQLiteCareRepository(Path(self.tempdir.name) / "unused.db"),
                document_store=LocalDocumentStore(Path(self.tempdir.name) / "unused", public_deployment=True),
            )
            with self.assertRaisesRegex(Exception, "短信服务尚未配置"):
                await runtime.request_otp("13700137000", "client")


if __name__ == "__main__":
    unittest.main()
