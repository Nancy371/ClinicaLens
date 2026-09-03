import asyncio
import os
import unittest
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from agent.demo import DemoCatalog, DemoRuntime, LimitDecision
from agent.server import create_app


class FakeLimiter:
    configured = True
    global_limit = 10
    client_limit = 2

    def __init__(self, allowed=True):
        self.allowed = allowed
        self.identities = []

    async def status(self):
        return LimitDecision(True, remaining=9, reset_at=2_000_000_000)

    async def consume(self, identity):
        self.identities.append(identity)
        if self.allowed:
            return LimitDecision(True, remaining=8, reset_at=2_000_000_000)
        return LimitDecision(
            False,
            remaining=0,
            reset_at=2_000_000_000,
            reason="client_daily_limit",
        )


def _live_env():
    return {
        "DEMO_LIVE_ENABLED": "true",
        "DEMO_CASE_MAP_JSON": '{"electrolyte-signal":"private-patient-id"}',
        "SERVICE_BASE_URL": "https://service.invalid",
        "SERVICE_TRAIN_TOKEN": "secret-service-token",
        "MODEL_API_KEY": "secret-model-key",
        "TEAM_ID": "secret-team",
        "DEMO_RATE_LIMIT_SALT": "0123456789abcdef",
    }


class DemoApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clients = []

    async def asyncTearDown(self):
        for client in self.clients:
            await client.close()

    async def _client(self, runtime):
        client = TestClient(TestServer(create_app(runtime)))
        await client.start_server()
        self.clients.append(client)
        return client

    async def test_offline_catalog_metrics_and_case_are_public(self):
        runtime = DemoRuntime(catalog=DemoCatalog(), limiter=FakeLimiter())
        client = await self._client(runtime)

        cases_response = await client.get("/api/demo/cases")
        cases = await cases_response.json()
        self.assertEqual(cases_response.status, 200)
        self.assertEqual(len(cases["cases"]), 3)
        self.assertEqual(cases["default_case"], "multi-organ-pattern")

        metrics_response = await client.get("/api/demo/metrics")
        metrics = await metrics_response.json()
        self.assertEqual(metrics["cases"], 7)
        self.assertEqual(metrics["candidate_recall_at_5"], 1.0)

        case_response = await client.get("/api/demo/cases/electrolyte-signal")
        payload = await case_response.json()
        serialized = str(payload)
        self.assertEqual(case_response.status, 200)
        self.assertEqual(payload["mode"], "offline")
        self.assertNotIn("regression-low-magnesium", serialized)
        self.assertNotIn("patient_id", serialized)

        index_response = await client.get("/")
        index_html = await index_response.text()
        self.assertEqual(index_response.status, 200)
        self.assertIn("患者版", index_html)
        self.assertIn("医生版", index_html)
        self.assertIn("病例与检查结果", index_html)
        self.assertIn("医生第二次签署后生成处方和患者提醒", index_html)
        self.assertNotIn("金融", index_html)

    async def test_live_status_fails_closed_without_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = DemoRuntime(catalog=DemoCatalog(), limiter=FakeLimiter())
        client = await self._client(runtime)
        response = await client.get("/api/demo/status")
        payload = await response.json()
        self.assertFalse(payload["live_enabled"])
        self.assertEqual(payload["disabled_reason"], "live_mode_disabled")

    async def test_unknown_case_is_rejected_before_rate_limit(self):
        with patch.dict(os.environ, _live_env(), clear=True):
            limiter = FakeLimiter()
            runtime = DemoRuntime(catalog=DemoCatalog(), limiter=limiter)
        client = await self._client(runtime)
        response = await client.post("/api/demo/runs", json={"case_key": "not-a-case"})
        payload = await response.json()
        self.assertEqual(response.status, 404)
        self.assertEqual(payload["error"], "unknown_case")
        self.assertEqual(limiter.identities, [])

    async def test_rate_limited_response_contains_no_secret(self):
        with patch.dict(os.environ, _live_env(), clear=True):
            runtime = DemoRuntime(catalog=DemoCatalog(), limiter=FakeLimiter(False))
        client = await self._client(runtime)
        response = await client.post(
            "/api/demo/runs", json={"case_key": "electrolyte-signal"}
        )
        payload = await response.json()
        self.assertEqual(response.status, 429)
        self.assertEqual(payload["error"], "client_daily_limit")
        self.assertNotIn("secret", str(payload))
        self.assertNotIn("private-patient-id", str(payload))

    async def test_live_job_returns_sanitized_projection(self):
        async def runner(case_key, case_meta):
            await asyncio.sleep(0)
            return {
                "schema_version": "demo-result.v1",
                "case": case_meta,
                "mode": "live",
                "status": "completed",
                "timeline": [],
                "evidence": [],
                "candidates": [],
                "conclusion": {"diagnosis": ["低镁血症"]},
                "metrics": {},
                "runtime": {},
            }

        with patch.dict(os.environ, _live_env(), clear=True):
            limiter = FakeLimiter()
            runtime = DemoRuntime(
                catalog=DemoCatalog(), limiter=limiter, live_runner=runner
            )
        client = await self._client(runtime)
        submit = await client.post(
            "/api/demo/runs",
            json={"case_key": "electrolyte-signal"},
            headers={"X-Forwarded-For": "203.0.113.10"},
        )
        accepted = await submit.json()
        self.assertEqual(submit.status, 202)

        job = None
        for _ in range(20):
            response = await client.get(f"/api/demo/runs/{accepted['run_id']}")
            job = await response.json()
            if job["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        self.assertEqual(job["status"], "completed")
        serialized = str(job)
        self.assertNotIn("private-patient-id", serialized)
        self.assertNotIn("secret-model-key", serialized)
        self.assertEqual(job["result"]["conclusion"]["diagnosis"], ["低镁血症"])

    async def test_cors_only_allows_configured_origin(self):
        with patch.dict(
            os.environ,
            {"DEMO_ALLOWED_ORIGINS": "https://portfolio.example"},
            clear=True,
        ):
            runtime = DemoRuntime(catalog=DemoCatalog(), limiter=FakeLimiter())
            client = await self._client(runtime)
            allowed = await client.get(
                "/api/demo/cases", headers={"Origin": "https://portfolio.example"}
            )
            denied = await client.get(
                "/api/demo/cases", headers={"Origin": "https://attacker.example"}
            )
        self.assertEqual(
            allowed.headers.get("Access-Control-Allow-Origin"),
            "https://portfolio.example",
        )
        self.assertNotIn("Access-Control-Allow-Origin", denied.headers)

    async def test_public_deployment_blocks_legacy_test_without_admin_token(self):
        with patch.dict(
            os.environ,
            {
                "DEMO_PUBLIC_DEPLOYMENT": "true",
                "SERVER_TEST_ACCESS_TOKEN": "admin-only-token",
            },
            clear=True,
        ):
            runtime = DemoRuntime(catalog=DemoCatalog(), limiter=FakeLimiter())
            client = await self._client(runtime)
            response = await client.post("/test", json={"patient_id": "private"})
            payload = await response.json()

        self.assertEqual(response.status, 403)
        self.assertEqual(payload["error"], "legacy_endpoint_disabled_on_public_demo")
        self.assertNotIn("admin-only-token", str(payload))

    async def test_live_job_timeout_is_safely_reported(self):
        async def slow_runner(case_key, case_meta):
            await asyncio.sleep(1)
            return {}

        with patch.dict(os.environ, _live_env(), clear=True):
            runtime = DemoRuntime(
                catalog=DemoCatalog(),
                limiter=FakeLimiter(),
                live_runner=slow_runner,
                run_timeout_seconds=0.01,
            )
        client = await self._client(runtime)
        submit = await client.post(
            "/api/demo/runs", json={"case_key": "electrolyte-signal"}
        )
        accepted = await submit.json()

        job = None
        for _ in range(30):
            response = await client.get(f"/api/demo/runs/{accepted['run_id']}")
            job = await response.json()
            if job["status"] == "failed":
                break
            await asyncio.sleep(0.01)

        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error_code"], "run_timed_out")
        self.assertIn("离线回放", job["error"])

    async def test_backend_exception_never_reaches_public_job(self):
        async def broken_runner(case_key, case_meta):
            raise RuntimeError("secret-model-key private-patient-id internal/path")

        with patch.dict(os.environ, _live_env(), clear=True):
            runtime = DemoRuntime(
                catalog=DemoCatalog(),
                limiter=FakeLimiter(),
                live_runner=broken_runner,
            )
        client = await self._client(runtime)
        with self.assertLogs("agent.demo", level="ERROR") as captured:
            submit = await client.post(
                "/api/demo/runs", json={"case_key": "electrolyte-signal"}
            )
            accepted = await submit.json()

            job = None
            for _ in range(20):
                response = await client.get(f"/api/demo/runs/{accepted['run_id']}")
                job = await response.json()
                if job["status"] == "failed":
                    break
                await asyncio.sleep(0.01)

        serialized = str(job)
        self.assertEqual(job["status"], "failed")
        self.assertNotIn("secret-model-key", serialized)
        self.assertNotIn("private-patient-id", serialized)
        self.assertNotIn("internal/path", serialized)
        log_output = "\n".join(captured.output)
        self.assertNotIn("secret-model-key", log_output)
        self.assertNotIn("private-patient-id", log_output)
        self.assertNotIn("internal/path", log_output)

    async def test_queue_rejects_more_than_configured_waiting_jobs(self):
        gate = asyncio.Event()

        async def blocked_runner(case_key, case_meta):
            await gate.wait()
            return {"status": "completed"}

        env = {**_live_env(), "DEMO_MAX_QUEUE": "1"}
        with patch.dict(os.environ, env, clear=True):
            runtime = DemoRuntime(
                catalog=DemoCatalog(),
                limiter=FakeLimiter(),
                live_runner=blocked_runner,
            )
        client = await self._client(runtime)
        first = await client.post(
            "/api/demo/runs", json={"case_key": "electrolyte-signal"}
        )
        self.assertEqual(first.status, 202)
        await asyncio.sleep(0.02)
        second = await client.post(
            "/api/demo/runs", json={"case_key": "electrolyte-signal"}
        )
        third = await client.post(
            "/api/demo/runs", json={"case_key": "electrolyte-signal"}
        )
        third_payload = await third.json()
        gate.set()

        self.assertEqual(second.status, 202)
        self.assertEqual(third.status, 503)
        self.assertEqual(third_payload["error"], "queue_full")


if __name__ == "__main__":
    unittest.main()
