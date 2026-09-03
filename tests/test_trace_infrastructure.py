import asyncio
import json
import shutil
import uuid
import unittest
from contextlib import contextmanager
from pathlib import Path

from agent.trace import TraceCollector, TraceConfig, TraceValidator
from agent.trace.serializers import safe_serialize
from hospital_agent.base import Actions


@contextmanager
def trace_tempdir():
    root = Path.cwd() / ".trace_test_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


class TraceInfrastructureTest(unittest.TestCase):
    def test_collector_writes_valid_append_only_trace(self):
        with trace_tempdir() as tmp:
            collector = TraceCollector(
                TraceConfig(enabled=True, output_dir=tmp, fail_open=False)
            )
            trace_id = collector.start_trace("Patient_Trace", {"api_key": "secret"})
            span_id = collector.start_span("judge", "DiagnosisJudge", "judge_candidates")
            artifact = collector.create_artifact("diagnosis_decision", {"x": 1})
            collector.emit_decision(
                "diagnosis_decision",
                {"final_diagnoses": ["A"]},
                refs=[artifact],
            )
            collector.end_span(span_id)
            collector.emit_submission({"submitted_diagnoses": ["A"]})
            collector.complete_trace({"diagnosis": ["A"]})

            trace_dir = Path(tmp) / "Patient_Trace" / trace_id
            events = [
                json.loads(line)
                for line in (trace_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([event["sequence"] for event in events], list(range(1, len(events) + 1)))
            self.assertEqual(events[0]["event_type"], "trace.started")
            self.assertEqual(events[-1]["event_type"], "trace.completed")
            self.assertIn("trace_id", json.loads((trace_dir / "trace.json").read_text(encoding="utf-8")))
            report = TraceValidator().validate(trace_dir, write_report=False)
            self.assertTrue(report["schema_valid"], report)

            started_payload = events[0]["payload"]["metadata"]
            self.assertEqual(started_payload["api_key"], "[REDACTED]")

    def test_disabled_collector_creates_no_files(self):
        with trace_tempdir() as tmp:
            collector = TraceCollector(TraceConfig(enabled=False, output_dir=tmp))
            self.assertIsNone(collector.start_trace("Patient_Disabled", {}))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_fail_open_does_not_raise_on_bad_output_dir(self):
        with trace_tempdir() as tmp:
            bad_output = Path(tmp) / "not_a_directory"
            bad_output.write_text("file", encoding="utf-8")
            collector = TraceCollector(
                TraceConfig(enabled=True, output_dir=str(bad_output), fail_open=True)
            )
            self.assertIsNone(collector.start_trace("Patient_BadDir", {}))

    def test_serializer_does_not_mutate_business_object(self):
        original = {"token": "secret", "items": [{"name": "A"}]}
        serialized = safe_serialize(original)
        self.assertEqual(original["token"], "secret")
        self.assertEqual(serialized["token"], "[REDACTED]")


class FakeActions(Actions):
    async def _request(self, method, path, **kwargs):
        if path == self.exam_results_path:
            return {"results": {"血常规": {"status": "ok", "result": "正常"}}}
        if path == self.case_evaluation_path:
            return {"status": "evaluated", "diagnosisAccuracy": 1.0}
        return {"answer": "主诉发热"}


class TraceActionsTest(unittest.TestCase):
    def test_tool_events_are_paired_and_artifacted(self):
        async def run_case():
            with trace_tempdir() as tmp:
                collector = TraceCollector(
                    TraceConfig(enabled=True, output_dir=tmp, fail_open=False)
                )
                collector.start_trace("Patient_Tool", {})
                actions = FakeActions(
                    base_url="https://example.invalid",
                    token="token",
                    team_id="team",
                    model_api_key="model-key",
                )
                actions.trace_collector = collector
                await actions.order_examination("Patient_Tool", ["血常规"], "baseline")
                await actions.evaluation("Patient_Tool", {"diagnosis": ["A"]})
                collector.complete_trace({"diagnosis": ["A"]})
                trace_dir = next((Path(tmp) / "Patient_Tool").iterdir())
                events = [
                    json.loads(line)
                    for line in (trace_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                called = [
                    event for event in events if event["event_type"] == "tool.called"
                ]
                returned = [
                    event for event in events if event["event_type"] == "tool.returned"
                ]
                self.assertEqual(len(called), 2)
                self.assertEqual(len(returned), 2)
                self.assertEqual(
                    {event["payload"]["call_id"] for event in called},
                    {event["payload"]["call_id"] for event in returned},
                )
                self.assertTrue(all(event["output_refs"] for event in returned))
                report = TraceValidator().validate(trace_dir, write_report=False)
                self.assertTrue(report["schema_valid"], report)

        asyncio.run(run_case())


if __name__ == "__main__":
    unittest.main()
