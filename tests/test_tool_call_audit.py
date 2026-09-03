import unittest
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from hospital_agent.base import (
    Actions,
    BaseDoctorAgent,
    summarize_tool_call_audit,
    summarize_training_results,
)


def response(status_code=200, payload=None, text=None):
    request = httpx.Request("POST", "https://example.invalid/exam/results")
    if text is not None:
        return httpx.Response(status_code, text=text, request=request)
    return httpx.Response(status_code, json=payload or {}, request=request)


class SequenceClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.is_closed = False

    async def request(self, method, path, **kwargs):
        if not self.outcomes:
            raise AssertionError("no more fake outcomes")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeActions(Actions):
    def __init__(self, outcomes):
        super().__init__(
            base_url="https://example.invalid",
            token="token",
            team_id="team",
            model_api_key="model-key",
        )
        self._fake_client = SequenceClient(outcomes)

    async def _get_client(self):
        return self._fake_client


class FailingTrainingAgent(BaseDoctorAgent):
    def __init__(self, outcomes, output_dir):
        super().__init__(
            {
                "output_dir": str(output_dir),
                "service": {
                    "use_invoke": True,
                    "invoke_path": "/invoke",
                    "exam_results_path": "/exam/results",
                    "case_evaluation_path": "/evaluate/case",
                    "batch_evaluation_path": "/evaluate",
                },
                "train": {"patient_ids": ["Patient_Fail"], "patient_count": 1},
                "llm": {},
            }
        )
        self.actions = FakeActions(outcomes)

    async def train(self, patient_id):
        await self.actions.order_examination(patient_id, ["Chest CT"], "baseline")

    async def test(self, patient_id):
        return None


class ToolCallAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_recovery_groups_attempts_under_one_logical_call(self):
        actions = FakeActions(
            [
                httpx.ReadTimeout("slow"),
                response(200, {"request_id": "R1", "results": {}}),
            ]
        )
        actions.begin_case("Patient_A")

        with patch("hospital_agent.base.asyncio.sleep", new=AsyncMock()):
            result = await actions._request(
                "POST",
                actions.exam_results_path,
                json={"patient_id": "Patient_A"},
                _audit_context={
                    "action": "order_examination",
                    "patient_id": "Patient_A",
                    "items": ["Chest CT"],
                },
            )

        self.assertEqual(result["request_id"], "R1")
        records = actions.snapshot_tool_audit()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["logical_call_id"], records[1]["logical_call_id"])
        self.assertFalse(records[0]["success"])
        self.assertTrue(records[0]["will_retry"])
        self.assertEqual(records[0]["primary_failure_reason"], "read_timeout")
        self.assertTrue(records[1]["success"])
        summary = summarize_tool_call_audit(records)
        self.assertEqual(summary["total_logical_calls"], 1)
        self.assertEqual(summary["total_attempts"], 2)
        self.assertEqual(summary["retry_recovered_calls"], 1)
        self.assertEqual(summary["retry_exhausted_calls"], 0)

    async def test_retry_exhausted_preserves_root_cause(self):
        actions = FakeActions(
            [
                httpx.ReadTimeout("slow-1"),
                httpx.ReadTimeout("slow-2"),
                httpx.ReadTimeout("slow-3"),
            ]
        )
        actions.begin_case("Patient_Timeout")

        with patch("hospital_agent.base.asyncio.sleep", new=AsyncMock()):
            with self.assertRaises(httpx.ReadTimeout):
                await actions.order_examination(
                    "Patient_Timeout",
                    ["Chest CT", "CBC"],
                    "baseline",
                )

        records = actions.snapshot_tool_audit()
        self.assertEqual(len(records), 3)
        final = records[-1]
        self.assertEqual(final["action"], "order_examination")
        self.assertEqual(final["endpoint"], "/exam/results")
        self.assertEqual(final["items_count"], 2)
        self.assertEqual(final["primary_failure_reason"], "read_timeout")
        self.assertIn("retry_exhausted", final["failure_flags"])
        self.assertFalse(final["will_retry"])
        self.assertTrue(final["retry_exhausted"])

    async def test_empty_json_response_is_a_response_failure_not_success(self):
        actions = FakeActions([response(200, text="")])
        actions.begin_case("Patient_Empty")

        with self.assertRaises(ValueError):
            await actions._request(
                "POST",
                actions.exam_results_path,
                json={"patient_id": "Patient_Empty"},
                _audit_context={
                    "action": "order_examination",
                    "patient_id": "Patient_Empty",
                    "items": ["Chest CT"],
                },
            )

        record = actions.snapshot_tool_audit()[0]
        self.assertFalse(record["success"])
        self.assertEqual(record["http_status"], 200)
        self.assertEqual(record["primary_failure_reason"], "empty_response")
        self.assertIn("empty_response", record["failure_flags"])

    async def test_http_5xx_is_recorded_without_changing_retry_policy(self):
        actions = FakeActions([response(503, {"error": "busy"})])
        actions.begin_case("Patient_5xx")

        with self.assertRaises(httpx.HTTPStatusError):
            await actions._request(
                "POST",
                actions.exam_results_path,
                json={"patient_id": "Patient_5xx"},
                _audit_context={
                    "action": "order_examination",
                    "patient_id": "Patient_5xx",
                    "items": ["Chest CT"],
                },
            )

        records = actions.snapshot_tool_audit()
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["success"])
        self.assertEqual(records[0]["http_status"], 503)
        self.assertEqual(records[0]["primary_failure_reason"], "http_5xx")
        self.assertTrue(records[0]["retryable"])
        self.assertFalse(records[0]["will_retry"])

    async def test_begin_case_isolates_tool_audit_ledger(self):
        actions = FakeActions(
            [
                response(200, {"answer": "A"}),
                response(200, {"answer": "B"}),
            ]
        )

        actions.begin_case("Patient_A")
        await actions._request(
            "POST",
            "/invoke",
            json={},
            _audit_context={"action": "ask_patient", "patient_id": "Patient_A"},
        )
        self.assertEqual(actions.snapshot_tool_audit()[0]["patient_id"], "Patient_A")

        actions.begin_case("Patient_B")
        await actions._request(
            "POST",
            "/invoke",
            json={},
            _audit_context={"action": "ask_patient", "patient_id": "Patient_B"},
        )
        records = actions.snapshot_tool_audit()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["patient_id"], "Patient_B")
        self.assertEqual(records[0]["logical_call_id"], "TC0001")

    def test_training_summary_uses_logical_tool_failure_denominators(self):
        rows = [
            {
                "status": "failed",
                "metrics": {},
                "audit": {
                    "tool_call_audit": [
                        {
                            "logical_call_id": "TC0001",
                            "attempt_id": "TC0001-A1",
                            "attempt_index": 1,
                            "patient_id": "Patient_A",
                            "action": "order_examination",
                            "endpoint": "/exam/results",
                            "success": False,
                            "primary_failure_reason": "read_timeout",
                            "failure_flags": ["read_timeout"],
                            "will_retry": True,
                        },
                        {
                            "logical_call_id": "TC0001",
                            "attempt_id": "TC0001-A2",
                            "attempt_index": 2,
                            "patient_id": "Patient_A",
                            "action": "order_examination",
                            "endpoint": "/exam/results",
                            "success": False,
                            "primary_failure_reason": "read_timeout",
                            "failure_flags": ["read_timeout", "retry_exhausted"],
                            "retry_exhausted": True,
                        },
                    ]
                },
            },
            {
                "status": "evaluated",
                "metrics": {},
                "audit": {
                    "tool_call_audit": [
                        {
                            "logical_call_id": "TC0001",
                            "attempt_id": "TC0001-A1",
                            "attempt_index": 1,
                            "patient_id": "Patient_B",
                            "action": "ask_patient",
                            "endpoint": "/invoke",
                            "success": True,
                        }
                    ]
                },
            },
        ]

        summary = summarize_training_results(rows)

        self.assertEqual(summary["tool_logical_call_count"], 2)
        self.assertEqual(summary["tool_attempt_count"], 3)
        self.assertEqual(summary["tool_logical_failure_count_by_reason"], {"read_timeout": 1})
        self.assertEqual(summary["tool_retry_exhausted_count"], 1)
        self.assertEqual(summary["tool_failure_case_count"], 1)
        self.assertEqual(summary["tool_failure_case_rate"], 0.5)
        self.assertEqual(summary["exam_results_logical_call_count"], 1)
        self.assertEqual(summary["exam_results_logical_failure_count"], 1)
        self.assertEqual(summary["exam_results_failure_rate"], 1.0)

    async def test_failed_training_case_persists_tool_audit(self):
        output_dir = Path("tests/_tool_call_audit_train")
        shutil.rmtree(output_dir, ignore_errors=True)
        agent = FailingTrainingAgent(
            [
                httpx.ReadTimeout("slow-1"),
                httpx.ReadTimeout("slow-2"),
                httpx.ReadTimeout("slow-3"),
            ],
            output_dir,
        )

        try:
            with patch("hospital_agent.base.asyncio.sleep", new=AsyncMock()):
                result = await agent.run_train()
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

        row = result["results"][0]
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["audit"]["tool_call_audit"])
        summary = row["audit"]["tool_contract_summary"]
        self.assertEqual(summary["retry_exhausted_calls"], 1)
        self.assertEqual(
            summary["failure_count_by_endpoint"],
            {"/exam/results": 1},
        )


if __name__ == "__main__":
    unittest.main()
