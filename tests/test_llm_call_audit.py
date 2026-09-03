import copy
import unittest

import yaml

from agent.agent import MyDoctorAgent
from agent.context_compiler import StageContextCompiler
from hospital_agent.base import summarize_training_results


class FakeLLM:
    def __init__(self, result, metadata=None, *, raise_exc=None):
        self.result = result
        self.raise_exc = raise_exc
        self.last_call_metadata = metadata or {
            "model": "fake-model",
            "model_invoked": True,
            "attempt_index": 1,
            "http_status": 200,
            "latency_ms": 12.0,
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "finish_reason": "stop",
            "raw_response_present": True,
            "response_chars": 20,
            "exception_type": "",
        }
        self.model_name = "fake-model"

    async def chat_json(self, messages, temperature=None, **kwargs):
        if self.raise_exc:
            raise self.raise_exc
        self.last_chat_json_kwargs = dict(kwargs)
        return self.result

    async def chat(self, messages, temperature=None, **kwargs):
        if self.raise_exc:
            raise self.raise_exc
        return str(self.result)


class SequenceLLM(FakeLLM):
    def __init__(self, results):
        super().__init__({})
        self.results = list(results)
        self.calls = 0

    async def chat_json(self, messages, temperature=None, **kwargs):
        if not self.results:
            raise AssertionError("no more fake LLM responses")
        self.calls += 1
        self.last_chat_json_kwargs = dict(kwargs)
        self.last_call_metadata = {
            "model": "fake-model",
            "model_invoked": True,
            "attempt_index": self.calls,
            "http_status": 200,
            "latency_ms": 10.0,
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
            "finish_reason": "stop",
            "raw_response_present": True,
            "response_chars": 20,
            "exception_type": "",
        }
        return self.results.pop(0)


def make_agent():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    config["memory"]["json_path"] = "tests/_runtime_chain/memory.json"
    config["memory"]["md_path"] = "tests/_runtime_chain/memory.md"
    config["memory"]["diagnostic_replay_path"] = "tests/_runtime_chain/replay.jsonl"
    config["policy_store_path"] = "tests/_runtime_chain/policies.json"
    config["self_improve_enabled"] = False
    return MyDoctorAgent(config)


class LLMCallAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_planner_consumer_result_uses_agent_audit_callback(self):
        agent = make_agent()
        agent.llm = FakeLLM({"unexpected": "shape"})
        planner = agent._get_planner()
        planner.criticism_max_calls = 0

        result = await planner.plan({}, {}, [])

        self.assertIn("strategy", result)
        record = agent._llm_call_audit[-1]
        self.assertEqual(record["purpose"], "planning")
        self.assertFalse(record["consumer_accepted"])
        self.assertTrue(record["fallback_used"])
        self.assertEqual(record["fallback_trigger"], "schema_missing_fields")

    async def test_budget_skip_creates_audit_record(self):
        agent = make_agent()
        agent.max_llm_calls_per_case = 1
        agent._llm_call_count = 1

        result = await agent._llm_chat_json([], purpose="thinking")

        self.assertEqual(result, {})
        record = agent._llm_call_audit[-1]
        self.assertFalse(record["model_invoked"])
        self.assertEqual(record["primary_failure_reason"], "llm_budget_exhausted")
        self.assertTrue(record["fallback_used"])

    async def test_truncated_json_parse_failure_keeps_generation_root_cause(self):
        agent = make_agent()
        agent.llm = FakeLLM(
            {"raw_response": '{"diagnosis": ['},
            metadata={
                "model": "fake-model",
                "model_invoked": True,
                "attempt_index": 1,
                "http_status": 200,
                "latency_ms": 15.0,
                "input_tokens": 100,
                "output_tokens": 200,
                "total_tokens": 300,
                "finish_reason": "length",
                "raw_response_present": True,
                "response_chars": 15,
                "exception_type": "",
            },
        )

        await agent._llm_chat_json([], purpose="diagnosis")

        record = agent._llm_call_audit[-1]
        self.assertFalse(record["parse_success"])
        self.assertIn("generation_truncated", record["failure_flags"])
        self.assertIn("json_parse_failed", record["failure_flags"])
        self.assertEqual(record["primary_failure_reason"], "generation_truncated")

    async def test_schema_missing_fields_are_audited(self):
        agent = make_agent()
        agent.llm = FakeLLM({"not_diagnosis": []})

        await agent._llm_chat_json([], purpose="diagnosis")

        record = agent._llm_call_audit[-1]
        self.assertTrue(record["parse_success"])
        self.assertFalse(record["schema_success"])
        self.assertEqual(record["missing_fields"], ["diagnosis"])
        self.assertIn("schema_missing_fields", record["failure_flags"])

    async def test_contract_executor_repairs_missing_required_field_once(self):
        agent = make_agent()
        agent.llm = SequenceLLM(
            [
                {"not_diagnosis": ["x"]},
                {"diagnosis": ["A"], "treatment_plan": "observe", "reasoning": "fixed"},
            ]
        )

        result = await agent._llm_chat_json([], purpose="diagnosis")

        self.assertEqual(result["diagnosis"], ["A"])
        records = agent._llm_call_audit
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["logical_call_id"], records[1]["logical_call_id"])
        self.assertEqual(records[0]["attempt_type"], "generate")
        self.assertEqual(records[1]["attempt_type"], "repair")
        self.assertTrue(records[1]["contract_repair_succeeded"])
        self.assertEqual(agent._llm_call_by_kind["json"], 1)
        self.assertEqual(agent._llm_call_by_kind["json_repair"], 1)

    async def test_stage_contract_sets_output_budget(self):
        agent = make_agent()
        agent.llm = FakeLLM({"diagnosis": ["A"]})

        await agent._llm_chat_json([], purpose="diagnosis")

        self.assertEqual(agent.llm.last_chat_json_kwargs["max_tokens"], 2048)
        record = agent._llm_call_audit[-1]
        self.assertEqual(record["requested_max_tokens"], 2048)
        self.assertEqual(record["contract_version"], "diagnosis.v1")

    async def test_thinking_contract_normalizes_string_differential(self):
        agent = make_agent()
        agent.llm = FakeLLM({"differential_diagnosis": ["A", "B"]})

        result = await agent._llm_chat_json([], purpose="thinking")

        self.assertEqual(result["differential_diagnosis"][0], {"diagnosis": "A"})
        record = agent._llm_call_audit[-1]
        self.assertIn(
            "thinking.differential_diagnosis:canonical_object_list",
            record["deterministic_normalizations"],
        )
        self.assertTrue(record["schema_success"])

    async def test_field_level_repair_merges_missing_critical_field(self):
        agent = make_agent()
        agent.llm = SequenceLLM(
            [
                {"reasoning": "kept"},
                {"diagnosis": ["A"]},
            ]
        )

        result = await agent._llm_chat_json([], purpose="diagnosis")

        self.assertEqual(result["diagnosis"], ["A"])
        self.assertEqual(result["reasoning"], "kept")
        self.assertIn(
            "field_level_repair_merge",
            agent._llm_call_audit[-1]["deterministic_normalizations"],
        )

    async def test_contract_drift_is_audited_when_consumer_rejects_valid_schema(self):
        agent = make_agent()
        agent.llm = FakeLLM({"diagnosis": ["A"]})

        await agent._llm_chat_json([], purpose="diagnosis")
        agent._mark_last_llm_consumer_result(
            "diagnosis",
            False,
            fallback_used=True,
            fallback_trigger="consumer_rejected",
        )

        record = agent._llm_call_audit[-1]
        self.assertTrue(record["schema_success"])
        self.assertTrue(record["contract_drift_detected"])
        self.assertIn("contract_drift", record["failure_flags"])
        self.assertEqual(
            record["consumer_rejection_code"],
            "LEGACY_CONSUMER_CONTRACT_DRIFT",
        )

    async def test_contract_repair_respects_repair_budget(self):
        agent = make_agent()
        agent.max_llm_repair_calls_per_case = 0
        agent.llm = SequenceLLM([{"not_diagnosis": ["x"]}])

        result = await agent._llm_chat_json([], purpose="diagnosis")

        self.assertEqual(result, {"not_diagnosis": ["x"]})
        records = agent._llm_call_audit
        self.assertEqual(len(records), 2)
        self.assertTrue(records[1]["contract_repair_attempted"])
        self.assertFalse(records[1]["model_invoked"])
        self.assertEqual(records[1]["primary_failure_reason"], "llm_budget_exhausted")

    async def test_low_priority_repair_does_not_spend_reserved_final_budget(self):
        agent = make_agent()
        agent.max_llm_repair_calls_per_case = 1
        agent.llm = SequenceLLM([{"unexpected": []}])

        result = await agent._llm_chat_json([], purpose="planning_criticism")

        self.assertEqual(result, {"unexpected": []})
        records = agent._llm_call_audit
        self.assertEqual(len(records), 2)
        self.assertFalse(records[1]["model_invoked"])
        self.assertEqual(records[1]["primary_failure_reason"], "llm_budget_exhausted")
        self.assertEqual(agent._llm_call_by_kind.get("json_repair", 0), 0)

    async def test_contract_repair_does_not_consume_clinical_budget(self):
        agent = make_agent()
        agent.max_llm_calls_per_case = 2
        agent.max_llm_repair_calls_per_case = 1
        agent.llm = SequenceLLM(
            [
                {"not_diagnosis": ["x"]},
                {"diagnosis": ["A"]},
                {"diagnosis": ["B"]},
            ]
        )

        first = await agent._llm_chat_json([], purpose="diagnosis")
        second = await agent._llm_chat_json([], purpose="diagnosis")

        self.assertEqual(first["diagnosis"], ["A"])
        self.assertEqual(second["diagnosis"], ["B"])
        self.assertEqual(agent._llm_call_by_kind["json"], 2)
        self.assertEqual(agent._llm_call_by_kind["json_repair"], 1)

    def test_stage_context_compiler_trims_and_audits_context(self):
        agent = make_agent()
        chat_history = [{"from": "doctor", "text": str(i)} for i in range(20)]

        compiled = agent._compile_llm_context(
            "thinking",
            collected_info={"symptoms": ["cough"]},
            exam_results={},
            chat_history=chat_history,
        )

        self.assertEqual(len(compiled["chat_history"]), 8)
        self.assertEqual(compiled["chat_history"][0]["text"], "12")
        audit = agent._llm_context_audit[-1]
        self.assertEqual(audit["stage"], "thinking")
        self.assertGreater(audit["source_context_chars"], audit["compiled_context_chars"])
        self.assertIn("dropped_item_counts", audit)
        self.assertGreater(audit["estimated_input_tokens"], 0)

    def test_context_compiler_drops_oversized_single_audit_field(self):
        compiler = StageContextCompiler(
            {
                "stage_budget": {
                    "diagnosis": {"max_input_tokens": 3000, "fallback_max_chars": 12000}
                }
            }
        )
        huge_audit = "x" * 3_000_000

        compiled = compiler.compile(
            "diagnosis",
            collected_info={"chief_complaint": "cough"},
            candidate_table="放射性肺炎 | AnchorSatisfied | ground_glass_opacity",
            evidence_summary="thoracic_radiotherapy\nlesion_within_prior_radiation_field",
            top_candidates=[
                {
                    "name": "放射性肺炎",
                    "audit": huge_audit,
                    "anchor_status": "AnchorSatisfied",
                }
            ],
        )

        context_text = str(compiled["context"])
        audit = compiled["audit"]
        self.assertLess(audit["compiled_context_chars"], 12000)
        self.assertTrue(audit["audit_payload_detected"])
        self.assertNotIn(huge_audit[:100], context_text)
        self.assertGreater(audit["dropped_item_counts"].get("tier3_field", 0), 0)

    def test_context_compiler_retains_critical_evidence_under_extreme_budget(self):
        compiler = StageContextCompiler(
            {
                "stage_budget": {
                    "diagnosis": {"max_input_tokens": 800, "fallback_max_chars": 3200}
                }
            }
        )

        compiled = compiler.compile(
            "diagnosis",
            collected_info={
                "chief_complaint": "呼吸困难",
                "background": "low value " * 2000,
            },
            evidence_summary=(
                "new material evidence: lesion_within_prior_radiation_field SUPPORTED\n"
                "hard contradiction: infection evidence absent\n"
                + ("generic symptom line\n" * 1000)
            ),
            candidate_table=(
                "current_primary: 放射性肺炎\n"
                "protected contender: 肺不张 associated finding\n"
                + ("low rank candidate\n" * 1000)
            ),
            chat_history=[{"from": "doctor", "text": str(i)} for i in range(30)],
        )

        text = str(compiled["context"])
        audit = compiled["audit"]
        self.assertIn("chief_complaint", text)
        self.assertIn("lesion_within_prior_radiation_field", text)
        self.assertIn("hard contradiction", text)
        self.assertIn("current_primary", text)
        self.assertLessEqual(audit["compiled_context_chars"], 3200)
        self.assertTrue(audit["critical_evidence_retained"])

    def test_training_summary_aggregates_llm_contract_metrics(self):
        rows = [
            {
                "status": "evaluated",
                "audit": {
                    "llm_context_audit": [
                        {
                            "stage": "diagnosis",
                            "source_context_chars": 1000,
                            "source_estimated_tokens": 250,
                            "context_chars": 400,
                            "estimated_input_tokens": 100,
                            "compression_ratio": 0.4,
                            "budget_violation_after_packing": False,
                            "audit_payload_detected": True,
                            "recursive_payload_detected": False,
                        }
                    ],
                    "llm_call_audit": [
                        {
                            "purpose": "diagnosis",
                            "json_expected": True,
                            "model_invoked": True,
                            "raw_response_present": True,
                            "parse_success": False,
                            "schema_applicable": False,
                            "failure_flags": ["json_parse_failed"],
                            "primary_failure_reason": "json_parse_failed",
                            "fallback_used": True,
                        },
                        {
                            "purpose": "thinking",
                            "json_expected": True,
                            "model_invoked": True,
                            "raw_response_present": True,
                            "parse_success": True,
                            "schema_applicable": True,
                            "schema_success": False,
                            "failure_flags": ["schema_missing_fields"],
                            "primary_failure_reason": "schema_missing_fields",
                            "fallback_used": True,
                        },
                    ]
                },
            }
        ]

        summary = summarize_training_results(rows)

        self.assertEqual(summary["llm_call_count"], 2)
        self.assertEqual(summary["llm_call_count_by_purpose"]["diagnosis"], 1)
        self.assertEqual(summary["llm_failure_count_by_reason"]["json_parse_failed"], 1)
        self.assertEqual(summary["llm_parse_failure_rate"], 0.5)
        self.assertEqual(summary["llm_schema_failure_rate"], 1.0)
        self.assertEqual(summary["llm_fallback_case_rate"], 1.0)
        self.assertEqual(summary["llm_context_compile_count"], 1)
        self.assertEqual(summary["llm_context_compile_count_by_stage"], {"diagnosis": 1})
        self.assertEqual(summary["average_llm_context_estimated_input_tokens"], 100.0)
        self.assertEqual(summary["average_llm_source_context_chars"], 1000.0)
        self.assertEqual(summary["average_llm_context_compression_ratio"], 0.4)
        self.assertEqual(summary["llm_context_audit_payload_detected_count"], 1)

    def test_failure_attribution_marks_llm_budget_as_not_medically_evaluable(self):
        attribution = MyDoctorAgent._deterministic_failure_attribution(
            report={"diagnosisAccuracy": 0.0},
            runtime_audit={"tool_contract_summary": {}},
            llm_call_audit=[
                {
                    "purpose": "diagnosis",
                    "failure_flags": ["llm_budget_exhausted"],
                }
            ],
            expected=["A"],
            top_twenty=["A"],
            submitted=["B"],
        )

        self.assertEqual(attribution["primary_failure_domain"], "LLM_GENERATION")
        self.assertEqual(attribution["primary_failure_reason"], "LLM_BUDGET_EXHAUSTED")
        self.assertFalse(attribution["medical_failure_evaluable"])

    def test_failure_attribution_marks_wrong_submission_as_arbitration(self):
        attribution = MyDoctorAgent._deterministic_failure_attribution(
            report={"diagnosisAccuracy": 0.0},
            runtime_audit={"tool_contract_summary": {}},
            llm_call_audit=[],
            expected=["A"],
            top_twenty=["A", "B"],
            submitted=["B"],
        )

        self.assertEqual(attribution["primary_failure_domain"], "ARBITRATION")
        self.assertTrue(attribution["medical_failure_evaluable"])
