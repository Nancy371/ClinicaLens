import copy
import asyncio
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from agent.agent import MyDoctorAgent
from agent.clinical_evidence import EvidenceBundle, Observation
from agent.evidence_pattern_compiler import EvidencePatternCompiler
from agent.claim_resolution import claim_key
from agent.targeted_exam_result_parser import (
    ExamResultIntentBinding,
    TargetedExamResultParser,
)


class TargetedExamResultParserTests(unittest.TestCase):
    def test_gap_aware_ct_extracts_radiation_field_claim_support(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-RP-1",
            order_id="O-RP-1",
            requested_exam="胸部增强CT",
            resolved_exam="CT扫描（CT）",
            actual_result_exam="CT扫描（CT）",
            target_gap_ids=["G-D100058-03"],
            target_claims=[
                "pulmonary_morphology",
                "radiation_field_lung_consistency",
                "post_radiotherapy_time_window",
            ],
            route_target_claims=[
                "pulmonary_morphology",
                "radiation_field_lung_consistency",
            ],
            target_candidate="放射性肺炎",
            entity_id="D100058",
        )
        parsed = parser.parse(
            {
                "status": "abnormal",
                "result": {
                    "conclusion": (
                        "中下肺野可见斑片状磨玻璃影和实变，"
                        "局限于既往放疗照射野内，伴轻度容积减小。"
                    )
                },
            },
            binding,
        )

        findings = {item.finding for item in parsed.observations}
        self.assertEqual(parsed.status, "positive")
        self.assertEqual(parsed.gap_closure_assessment, "partial")
        self.assertEqual(parsed.gap_resolution_status, "PARTIALLY_CLOSED")
        self.assertIn("ground_glass_opacity", findings)
        self.assertIn("pulmonary_consolidation", findings)
        self.assertIn("pulmonary_volume_loss", findings)
        self.assertIn("lesion_within_prior_radiation_field", findings)
        self.assertNotIn("radiation_pneumonitis", findings)
        self.assertNotIn("D100058", findings)
        claim_by_id = {item["target_claim"]: item for item in parsed.claim_matches}
        self.assertEqual(
            claim_by_id["radiation_field_lung_consistency"]["claim_status"],
            "SUPPORTED",
        )
        self.assertEqual(
            claim_by_id["pulmonary_morphology"]["claim_status"],
            "SUPPORTED",
        )
        self.assertEqual(
            claim_by_id["post_radiotherapy_time_window"]["claim_status"],
            "NOT_APPLICABLE",
        )
        self.assertTrue(parsed.material_evidence_delta["material_evidence_changed"])

    def test_gap_aware_ct_can_contradict_radiation_field_claim(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-RP-2",
            order_id="O-RP-2",
            requested_exam="胸部CT",
            resolved_exam="CT扫描（CT）",
            actual_result_exam="CT扫描（CT）",
            target_gap_ids=["G-D100058-03"],
            target_claims=["radiation_field_lung_consistency"],
            target_candidate="放射性肺炎",
            entity_id="D100058",
        )
        parsed = parser.parse(
            {
                "status": "abnormal",
                "result": {
                    "conclusion": "双肺弥漫分布磨玻璃影，病变明显超出原照射区域。"
                },
            },
            binding,
        )

        findings = {item.finding for item in parsed.observations}
        self.assertEqual(parsed.status, "negative")
        self.assertEqual(parsed.gap_closure_assessment, "negative_closed")
        self.assertEqual(parsed.gap_resolution_status, "CONTRADICTED")
        self.assertIn("lesion_outside_prior_radiation_field", findings)
        claim = parsed.claim_matches[0]
        self.assertEqual(claim["target_claim"], "radiation_field_lung_consistency")
        self.assertEqual(claim["claim_status"], "CONTRADICTED")

    def test_gap_aware_ct_missing_field_relation_is_unresolved_not_negative(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-RP-3",
            order_id="O-RP-3",
            requested_exam="胸部CT",
            resolved_exam="CT扫描（CT）",
            actual_result_exam="CT扫描（CT）",
            target_gap_ids=["G-D100058-03"],
            target_claims=["radiation_field_lung_consistency"],
            target_candidate="放射性肺炎",
            entity_id="D100058",
        )
        parsed = parser.parse(
            {
                "status": "abnormal",
                "result": {"conclusion": "右肺可见磨玻璃影，未描述与既往放疗野关系。"},
            },
            binding,
        )

        self.assertEqual(parsed.status, "inconclusive")
        self.assertEqual(parsed.gap_resolution_status, "UNRESOLVED")
        self.assertEqual(parsed.claim_matches[0]["claim_status"], "NOT_ADDRESSED")
        self.assertNotEqual(parsed.gap_closure_assessment, "negative_closed")

    def test_ct_result_still_extracts_findings_when_route_claim_is_not_addressed(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-RP-3B",
            order_id="O-RP-3B",
            requested_exam="chest CT",
            resolved_exam="CT",
            actual_result_exam="CT",
            target_gap_ids=["G-D100058-03"],
            target_claims=["post_radiotherapy_time_window"],
            route_target_claims=["post_radiotherapy_time_window"],
            target_candidate="radiation pneumonitis",
            entity_id="D100058",
        )
        parsed = parser.parse(
            {
                "status": "abnormal",
                "result": {
                    "conclusion": (
                        "Chest CT shows ground-glass opacity and consolidation; "
                        "the radiotherapy timing window is not described."
                    )
                },
            },
            binding,
        )

        findings = {item.finding for item in parsed.observations}
        self.assertIn("ground_glass_opacity", findings)
        self.assertIn("pulmonary_consolidation", findings)
        self.assertEqual(parsed.claim_matches[0]["claim_status"], "NOT_APPLICABLE")
        self.assertNotEqual(parsed.observations, [])

    def test_gap_aware_ct_negation_does_not_create_positive_ground_glass(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-RP-4",
            order_id="O-RP-4",
            requested_exam="胸部CT",
            resolved_exam="CT扫描（CT）",
            actual_result_exam="CT扫描（CT）",
            target_gap_ids=["G-D100058-04"],
            target_claims=["ground_glass_opacity"],
            target_candidate="放射性肺炎",
            entity_id="D100058",
        )
        parsed = parser.parse(
            {"status": "normal", "result": {"conclusion": "双肺未见磨玻璃影。"}},
            binding,
        )

        by_finding = {item.finding: item for item in parsed.observations}
        self.assertEqual(by_finding["ground_glass_opacity"].polarity, "negative")
        self.assertEqual(parsed.claim_matches[0]["claim_status"], "CONTRADICTED")

    def test_enhanced_ct_bound_to_pavm_gap_recovers_vascular_anchor(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-PAVM-1",
            order_id="O-PAVM-1",
            requested_exam="胸部增强CT",
            resolved_exam="胸部增强CT",
            actual_result_exam="增强胸部CT扫描（Chest CECT）",
            target_gap_ids=["G-PAVF-01"],
            target_claims=["enhanced_ct_vascular_malformation"],
            target_candidate="肺动静脉瘘",
            entity_id="D100055",
        )
        parsed = parser.parse(
            {
                "status": "abnormal",
                "result": {
                    "结论": "右下叶强化迂曲血管性病变，可见供血肺动脉及早期引流肺静脉，考虑肺动静脉畸形。"
                },
            },
            binding,
        )

        findings = {item.finding for item in parsed.observations}
        self.assertEqual(parsed.status, "positive")
        self.assertEqual(parsed.gap_closure_assessment, "positive_closed")
        self.assertIn("feeding_pulmonary_artery_present", findings)
        self.assertIn("draining_pulmonary_vein_present", findings)
        self.assertIn("enhanced_ct_vascular_malformation", findings)
        self.assertTrue(all(item.target_gap_ids == ["G-PAVF-01"] for item in parsed.observations))

        compiler = EvidencePatternCompiler(ref_dir="data/ref_data")
        derived = compiler.compile(
            [],
            EvidenceBundle(
                [
                    Observation("hemoptysis", "问诊"),
                    *parsed.observations,
                ]
            ),
        )
        self.assertIn("pulmonary_av_fistula_pattern", {item.finding for item in derived})

    def test_standard_echo_normal_does_not_negate_bubble_echo_gap(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-PAVM-2",
            order_id="O-PAVM-2",
            requested_exam="右心声学造影",
            resolved_exam="超声心动图右心声学造影",
            actual_result_exam="超声心动图",
            target_gap_ids=["G-PAVF-02"],
            target_claims=["bubble_echo_right_to_left_shunt"],
            target_candidate="肺动静脉瘘",
            entity_id="D100055",
        )
        parsed = parser.parse(
            {"status": "normal", "result": {"结论": "心脏结构未见明显异常。"}},
            binding,
        )

        self.assertIn(parsed.status, {"unresolved", "inconclusive"})
        self.assertNotIn(
            "bubble_echo_right_to_left_shunt",
            {item.finding for item in parsed.observations},
        )
        self.assertNotEqual(parsed.gap_closure_assessment, "negative_closed")

    def test_plain_ct_cannot_create_pavm_confirmatory_pattern(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-PAVM-PLAIN",
            order_id="O-PAVM-PLAIN",
            requested_exam="肺动脉CTA",
            resolved_exam="肺动脉CTA",
            actual_result_exam="胸部CT扫描（Chest CT）",
            target_gap_ids=["G-PAVF-PLAIN"],
            target_claims=["pulmonary_cta_positive"],
            target_candidate="肺动静脉瘘",
            entity_id="D100055",
        )
        parsed = parser.parse(
            {
                "status": "abnormal",
                "result": {
                    "结论": "右下肺结节样血管影增粗，可疑供血动脉及引流静脉，建议增强检查。"
                },
            },
            binding,
        )

        findings = {item.finding for item in parsed.observations}
        self.assertEqual(parsed.gap_closure_assessment, "partial")
        self.assertIn("vascular_pulmonary_nodule_suspected", findings)
        self.assertNotIn("feeding_pulmonary_artery_present", findings)
        self.assertNotIn("draining_pulmonary_vein_present", findings)

        compiler = EvidencePatternCompiler(ref_dir="data/ref_data")
        derived = compiler.compile(
            [],
            EvidenceBundle(
                [
                    Observation("hemoptysis", "问诊"),
                    Observation("cyanosis", "体格检查"),
                    *parsed.observations,
                ]
            ),
        )
        self.assertNotIn("pulmonary_av_fistula_pattern", {item.finding for item in derived})

    def test_effective_cta_negative_can_close_gap_negative(self):
        parser = TargetedExamResultParser()
        binding = ExamResultIntentBinding(
            binding_id="B-PAVM-3",
            order_id="O-PAVM-3",
            requested_exam="肺动脉CTA",
            resolved_exam="肺动脉CTA",
            actual_result_exam="肺动脉CTA",
            target_gap_ids=["G-PAVF-03"],
            target_claims=["pulmonary_cta_positive"],
            target_candidate="肺动静脉瘘",
            entity_id="D100055",
        )
        parsed = parser.parse(
            {"status": "normal", "result": {"结论": "未见肺动静脉异常交通，未见肺血管畸形。"}},
            binding,
        )

        self.assertEqual(parsed.status, "negative")
        self.assertEqual(parsed.gap_closure_assessment, "negative_closed")
        self.assertTrue(any(item.polarity == "negative" for item in parsed.observations))


class AgentTargetedExamRecoveryTests(unittest.TestCase):
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

    def radiation_claim_detail(self):
        return {
            "exam": "chest CT",
            "requested_exam": "chest CT",
            "resolved_exam": "CT",
            "exam_source": "deferred_gap_closure_exam",
            "target_gaps": ["G-D100058"],
            "entity_id": "D100058",
            "target_candidates": ["D100058"],
            "target_claims": [
                "pulmonary_morphology",
                "radiation_field_lung_consistency",
                "post_radiotherapy_time_window",
            ],
            "route_target_claims": [
                "pulmonary_morphology",
                "radiation_field_lung_consistency",
            ],
            "claim_requirements": [
                {"claim_id": "pulmonary_morphology", "required_for_anchor": True},
                {
                    "claim_id": "radiation_field_lung_consistency",
                    "required_for_anchor": True,
                },
                {
                    "claim_id": "post_radiotherapy_time_window",
                    "required_for_anchor": True,
                },
            ],
            "closure_routes": [
                {
                    "route_id": "route_pulmonary_morphology_ct",
                    "route_type": "exam_result",
                    "exam": "chest CT",
                    "target_claims": ["pulmonary_morphology"],
                },
                {
                    "route_id": "route_radiation_field_ct",
                    "route_type": "exam_result",
                    "exam": "chest CT",
                    "target_claims": ["radiation_field_lung_consistency"],
                },
            ],
            "claim_closure_plan_version": "claim_closure_plan_v1",
            "source_evidence_version": 7,
        }

    def test_agent_binds_returned_cect_name_to_original_pavm_gap(self):
        agent = self.make_agent()
        strategy = {
            "exam_authorization_details": [
                {
                    "exam": "胸部增强CT",
                    "requested_exam": "胸部增强CT",
                    "resolved_exam": "胸部增强CT",
                    "exam_source": "deferred_gap_closure_exam",
                    "target_gaps": ["G-PAVF-01"],
                    "target_findings": ["enhanced_ct_vascular_malformation"],
                    "target_candidates": ["肺动静脉瘘"],
                    "source_gap_value": 0.91,
                }
            ]
        }
        new_results = {
            "增强胸部CT扫描（Chest CECT）": {
                "status": "abnormal",
                "result": {
                    "结论": "右下肺异常血管团，见供血肺动脉及早期引流肺静脉。"
                },
            }
        }

        agent._record_targeted_exam_result_recovery(
            patient_id="Patient_03998",
            stage="unit_test",
            ordered_items=["胸部增强CT"],
            new_results=new_results,
            strategy=strategy,
        )

        self.assertTrue(agent._exam_result_intent_bindings)
        binding = agent._exam_result_intent_bindings[0]
        self.assertEqual(binding["actual_result_exam"], "增强胸部CT扫描（Chest CECT）")
        findings = {item.finding for item in agent._targeted_exam_observations}
        self.assertIn("enhanced_ct_vascular_malformation", findings)
        evidence = agent._normalize_with_exam_recovery(
            {"symptoms": ["咯血", "低氧"]},
            new_results,
        )
        self.assertIn(
            "enhanced_ct_vascular_malformation",
            set(evidence.findings("positive")),
        )
        self.assertIn(
            "pulmonary_av_fistula_pattern",
            set(evidence.findings("positive")),
        )

    def test_exam_result_applicability_updates_radiation_claim_contract(self):
        agent = self.make_agent()
        radiation_detail = self.radiation_claim_detail()
        strategy = {
            "exam_authorization_details": [
                {
                    "exam": "chest CT",
                    "requested_exam": "chest CT",
                    "resolved_exam": "CT",
                    "exam_source": "judge_discriminating_exam",
                    "target_gaps": ["G-D100037"],
                    "entity_id": "D100037",
                    "target_candidates": ["D100037"],
                    "target_claims": ["tuberculosis_imaging_pattern"],
                    "route_target_claims": ["tuberculosis_imaging_pattern"],
                },
                radiation_detail,
            ]
        }

        agent._record_targeted_exam_result_recovery(
            patient_id="Patient_03674",
            stage="unit_test",
            ordered_items=["chest CT"],
            new_results={
                "CT": {
                    "status": "abnormal",
                    "result": {
                        "conclusion": (
                            "Chest CT shows ground-glass opacity and consolidation, "
                            "within prior radiation field."
                        )
                    },
                }
            },
            strategy=strategy,
        )

        morph_key = claim_key(
            entity_id="D100058",
            claim_id="pulmonary_morphology",
            contract_id="claim_anchor_contract:D100058",
            contract_version="claim_closure_plan_v1",
        )
        spatial_key = claim_key(
            entity_id="D100058",
            claim_id="radiation_field_lung_consistency",
            contract_id="claim_anchor_contract:D100058",
            contract_version="claim_closure_plan_v1",
        )
        self.assertEqual(
            agent._claim_resolution_ledger[morph_key]["resolution_status"],
            "SUPPORTED",
        )
        self.assertEqual(
            agent._claim_resolution_ledger[spatial_key]["resolution_status"],
            "SUPPORTED",
        )
        self.assertEqual(agent._claim_state_version, 1)
        self.assertEqual(agent._diagnostic_state_version, 1)
        radiation_payloads = [
            item
            for item in agent._targeted_exam_result_parses
            if item.get("entity_id") == "D100058"
        ]
        self.assertTrue(radiation_payloads)
        self.assertIn(
            radiation_payloads[0].get("binding_source"),
            {"SHARED_AUTHORIZATION", "RESULT_APPLICABILITY"},
        )
        self.assertEqual(
            len(
                [
                    item
                    for item in agent._targeted_exam_observations
                    if item.finding == "ground_glass_opacity"
                ]
            ),
            1,
        )
        transition_names = [
            item["name"] for item in agent._clinical_transition_trace
        ]
        for name in [
            "exam_result_received",
            "claim_contracts_bound",
            "observations_parsed",
            "claim_matches_generated",
            "claim_ledger_updated",
            "claim_state_transaction_committed",
        ]:
            self.assertIn(name, transition_names)
        self.assertLess(
            transition_names.index("exam_result_received"),
            transition_names.index("observations_parsed"),
        )
        self.assertLess(
            transition_names.index("observations_parsed"),
            transition_names.index("claim_matches_generated"),
        )

    def test_candidate_driven_claim_contract_updates_without_original_authorization(self):
        agent = self.make_agent()
        agent._last_diagnosis_decision_obj = SimpleNamespace(
            judge_primary="肺结核",
            bridge_protected_candidates=[],
            judge_decision={},
            candidates=[
                SimpleNamespace(
                    diagnosis="radiation pneumonitis",
                    entity_id="D100058",
                    eligibility_status="PrimaryEligible",
                    eligibility_anchor_status="AnchorSatisfied",
                    matched_evidence=["thoracic_radiotherapy"],
                    required_gaps=[],
                    evidence_gaps=[],
                    actionable_gap_count=0,
                )
            ],
        )
        strategy = {
            "exam_authorization_details": [
                {
                    "exam": "chest CT",
                    "requested_exam": "chest CT",
                    "resolved_exam": "CT",
                    "exam_source": "judge_discriminating_exam",
                    "target_gaps": ["G-D100037"],
                    "entity_id": "D100037",
                    "target_candidates": ["D100037"],
                    "target_claims": ["tuberculosis_imaging_pattern"],
                    "route_target_claims": ["tuberculosis_imaging_pattern"],
                }
            ]
        }

        agent._record_targeted_exam_result_recovery(
            patient_id="Patient_03674",
            stage="unit_test",
            ordered_items=["chest CT"],
            new_results={
                "CT": {
                    "status": "abnormal",
                    "result": {
                        "conclusion": (
                            "Chest CT shows ground-glass opacity and consolidation, "
                            "within prior radiation field."
                        )
                    },
                }
            },
            strategy=strategy,
        )

        morph_key = claim_key(
            entity_id="D100058",
            claim_id="pulmonary_morphology",
            contract_id="claim_anchor_contract:D100058",
            contract_version="claim_closure_plan_v1",
        )
        spatial_key = claim_key(
            entity_id="D100058",
            claim_id="radiation_field_lung_consistency",
            contract_id="claim_anchor_contract:D100058",
            contract_version="claim_closure_plan_v1",
        )
        temporal_key = claim_key(
            entity_id="D100058",
            claim_id="post_radiotherapy_time_window",
            contract_id="claim_anchor_contract:D100058",
            contract_version="claim_closure_plan_v1",
        )
        self.assertEqual(
            agent._claim_resolution_ledger[morph_key]["resolution_status"],
            "SUPPORTED",
        )
        self.assertEqual(
            agent._claim_resolution_ledger[spatial_key]["resolution_status"],
            "SUPPORTED",
        )
        self.assertEqual(
            agent._claim_resolution_ledger[temporal_key]["resolution_status"],
            "UNRESOLVED",
        )
        radiation_payloads = [
            item
            for item in agent._targeted_exam_result_parses
            if item.get("entity_id") == "D100058"
        ]
        self.assertTrue(radiation_payloads)
        self.assertEqual(
            radiation_payloads[0].get("applicability_reason"),
            "candidate_claim_contract_compatibility",
        )
        self.assertTrue(agent._candidate_claim_contract_views)
        self.assertTrue(
            any(
                item.get("entity_id") == "D100058"
                and item.get("clinical_admitted")
                for item in agent._clinical_admission_audit
            )
        )

    def test_late_admission_hydrates_existing_ct_observations(self):
        agent = self.make_agent()
        agent._targeted_exam_observations = [
            Observation(
                finding="ground_glass_opacity",
                source="unit",
                source_exam="CT",
                information_value=0.9,
            ),
            Observation(
                finding="pulmonary_consolidation",
                source="unit",
                source_exam="CT",
                information_value=0.9,
            ),
            Observation(
                finding="lesion_within_prior_radiation_field",
                source="unit",
                source_exam="CT",
                information_value=0.95,
            ),
        ]
        agent._last_diagnosis_decision_obj = SimpleNamespace(
            judge_primary="肺结核",
            bridge_protected_candidates=[],
            judge_decision={},
            candidates=[
                SimpleNamespace(
                    diagnosis="radiation pneumonitis",
                    entity_id="D100058",
                    eligibility_status="PrimaryEligible",
                    eligibility_anchor_status="AnchorSatisfied",
                    matched_evidence=["thoracic_radiotherapy"],
                    required_gaps=[],
                    evidence_gaps=[],
                    actionable_gap_count=0,
                )
            ],
        )

        views = agent._materialize_admitted_candidate_claim_states()
        agent._hydrate_claim_states_from_existing_exam_observations(
            views,
            stage="unit_late_hydration",
        )

        morph_key = claim_key(
            entity_id="D100058",
            claim_id="pulmonary_morphology",
            contract_id="claim_anchor_contract:D100058",
            contract_version="claim_closure_plan_v1",
        )
        spatial_key = claim_key(
            entity_id="D100058",
            claim_id="radiation_field_lung_consistency",
            contract_id="claim_anchor_contract:D100058",
            contract_version="claim_closure_plan_v1",
        )
        self.assertEqual(
            agent._claim_resolution_ledger[morph_key]["resolution_status"],
            "SUPPORTED",
        )
        self.assertEqual(
            agent._claim_resolution_ledger[spatial_key]["resolution_status"],
            "SUPPORTED",
        )
        self.assertEqual(agent._claim_state_version, 1)
        self.assertTrue(
            any(
                item.get("binding_source") == "HISTORICAL_RESULT_APPLICABILITY"
                for item in agent._targeted_exam_result_parses
            )
        )

    def test_failed_train_row_persists_d100058_clinical_runtime_audit(self):
        agent = self.make_agent()
        output_dir = Path("tests/_runtime_chain/clinical_runtime_audit")
        shutil.rmtree(output_dir, ignore_errors=True)
        agent.output_dir = str(output_dir)
        agent.config.setdefault("train", {})["patient_ids"] = ["Patient_03674"]
        agent.config["train"]["patient_count"] = 1
        strategy = {
            "exam_authorization_details": [
                {
                    "exam": "chest CT",
                    "requested_exam": "chest CT",
                    "resolved_exam": "CT",
                    "exam_source": "judge_discriminating_exam",
                    "target_gaps": ["G-D100037"],
                    "entity_id": "D100037",
                    "target_candidates": ["D100037"],
                    "target_claims": ["tuberculosis_imaging_pattern"],
                    "route_target_claims": ["tuberculosis_imaging_pattern"],
                },
                self.radiation_claim_detail(),
            ]
        }

        async def fail_after_claim_recovery(patient_id):
            agent._record_targeted_exam_result_recovery(
                patient_id=patient_id,
                stage="unit_test_failure",
                ordered_items=["chest CT"],
                new_results={
                    "CT": {
                        "status": "abnormal",
                        "result": {
                            "conclusion": (
                                "Chest CT shows ground-glass opacity and consolidation, "
                                "within prior radiation field."
                            )
                        },
                    }
                },
                strategy=strategy,
            )
            raise RuntimeError("boom-after-claim-ledger")

        agent.train = fail_after_claim_recovery
        try:
            result = asyncio.run(agent.run_train())
            row = result["results"][0]
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

        self.assertEqual(row["status"], "failed")
        audit = row["audit"]
        for field in [
            "targeted_exam_result_parses",
            "exam_result_applicability",
            "targeted_exam_observations",
            "claim_match_events",
            "claim_resolution_ledger",
            "claim_resolution_update_audit",
            "claim_state_version",
            "diagnostic_state_version",
            "gap_state",
            "last_completed_stage",
            "failure_stage",
            "last_successful_clinical_transition",
            "clinical_transition_trace",
        ]:
            self.assertIn(field, audit)
        self.assertGreater(audit["claim_state_version"], 0)
        self.assertGreater(audit["diagnostic_state_version"], 0)
        morph_key = claim_key(
            entity_id="D100058",
            claim_id="pulmonary_morphology",
            contract_id="claim_anchor_contract:D100058",
            contract_version="claim_closure_plan_v1",
        )
        spatial_key = claim_key(
            entity_id="D100058",
            claim_id="radiation_field_lung_consistency",
            contract_id="claim_anchor_contract:D100058",
            contract_version="claim_closure_plan_v1",
        )
        self.assertEqual(
            audit["claim_resolution_ledger"][morph_key]["resolution_status"],
            "SUPPORTED",
        )
        self.assertEqual(
            audit["claim_resolution_ledger"][spatial_key]["resolution_status"],
            "SUPPORTED",
        )
        applicability_entities = {
            item.get("entity_id") for item in audit["exam_result_applicability"]
        }
        self.assertIn("D100058", applicability_entities)
        transition_names = [
            item.get("name") for item in audit["clinical_transition_trace"]
        ]
        self.assertIn("claim_ledger_updated", transition_names)
        self.assertEqual(audit["failure_stage"], "post_result_re_evaluation")
        self.assertEqual(
            audit["last_successful_clinical_transition"]["name"],
            "claim_state_transaction_committed",
        )

    def test_pre_exam_judge_payload_injects_runtime_claim_state(self):
        agent = self.make_agent()
        ledger = {
            "D100058|pulmonary_morphology|claim_anchor_contract:D100058|claim_closure_plan_v1": {
                "entity_id": "D100058",
                "claim_id": "pulmonary_morphology",
                "contract_id": "claim_anchor_contract:D100058",
                "contract_version": "claim_closure_plan_v1",
                "resolution_status": "SUPPORTED",
            }
        }
        agent._claim_resolution_ledger = ledger
        agent._claim_state_version = 1
        agent._diagnostic_state_version = 1

        class CapturingEngine:
            def __init__(self):
                self.seen_payload = {}

            def decide(self, llm_result, rag_chunks, evidence):
                self.seen_payload = dict(llm_result)
                gap = {
                    "gap_id": "G-D100058",
                    "claim_resolutions": list(
                        llm_result.get("_claim_resolution_ledger", {}).values()
                    ),
                    "remaining_claims": ["post_radiotherapy_time_window"],
                }
                return SimpleNamespace(
                    judge_decision={"active_evidence_gaps": [gap]},
                    claim_state_version=int(llm_result.get("_claim_state_version") or 0),
                )

        engine = CapturingEngine()
        agent.diagnosis_engine = engine

        payload = agent._pre_exam_judge_payload(
            {"symptoms": ["dyspnea"]},
            {"chest CT": {"status": "abnormal"}},
            thinking={"differential_diagnosis": ["radiation pneumonitis"]},
        )

        self.assertEqual(engine.seen_payload["_claim_state_version"], 1)
        self.assertIn("_claim_resolution_ledger", engine.seen_payload)
        self.assertEqual(payload["pre_exam_engine_claim_state_version"], 1)
        self.assertFalse(payload["pre_exam_stale_claim_state_detected"])
        self.assertEqual(payload["pre_exam_hydrated_gap_count"], 1)

    def test_strategy_order_items_blocks_generic_completed_ct_duplicate(self):
        agent = self.make_agent()
        strategy = {
            "items": ["CT"],
            "differential_driven": True,
            "exam_authorization_details": [],
        }

        items = agent._strategy_order_items(
            strategy,
            collected_info={"symptoms": ["dyspnea"]},
            candidate_diseases=["radiation pneumonitis"],
            existing_results={"chest CT": {"status": "abnormal"}},
            max_items=None,
            add_strong_verification=False,
        )

        self.assertEqual(items, [])
        audit = strategy.get("exam_repeat_authorization_audit") or []
        self.assertTrue(audit)
        self.assertTrue(
            {
                "COMPLETED_EXAM_DUPLICATE",
                "GENERIC_WORKUP_DUPLICATE_BLOCKED",
            }
            & set(audit[0]["reason_codes"])
        )

    def test_completed_claim_route_blocks_repeat_ct_with_route_audit(self):
        agent = self.make_agent()
        strategy = {
            "items": ["chest CT"],
            "differential_driven": True,
            "exam_authorization_details": [
                {
                    "exam": "chest CT",
                    "exam_source": "deferred_gap_closure_exam",
                    "target_gaps": ["G-D100058"],
                    "target_claims": [
                        "pulmonary_morphology",
                        "radiation_field_lung_consistency",
                    ],
                    "route_target_claims": [
                        "pulmonary_morphology",
                        "radiation_field_lung_consistency",
                    ],
                    "closure_routes": [
                        {
                            "route_id": "route-ct-spatial",
                            "route_type": "exam_result",
                            "target_claims": [
                                "pulmonary_morphology",
                                "radiation_field_lung_consistency",
                            ],
                        }
                    ],
                    "source_evidence_version": 7,
                }
            ],
        }

        items = agent._strategy_order_items(
            strategy,
            collected_info={"symptoms": ["dyspnea"]},
            candidate_diseases=["radiation pneumonitis"],
            existing_results={"chest CT": {"status": "abnormal"}},
            max_items=None,
            add_strong_verification=False,
        )

        self.assertEqual(items, [])
        audit = strategy.get("exam_repeat_authorization_audit") or []
        self.assertTrue(audit)
        self.assertIn("CLAIM_ROUTE_ALREADY_RESOLVED", audit[0]["reason_codes"])
        self.assertEqual(audit[0]["route_target_claim_ids"], [
            "pulmonary_morphology",
            "radiation_field_lung_consistency",
        ])
        self.assertEqual(audit[0]["closure_route_ids"], ["route-ct-spatial"])
        self.assertEqual(audit[0]["source_evidence_version"], 7)
        self.assertEqual(audit[0]["prior_result_state"], "completed_same_exam")

    def test_repeat_exam_requires_explicit_route_authorization(self):
        agent = self.make_agent()
        strategy = {
            "items": ["chest CT"],
            "differential_driven": True,
            "exam_authorization_details": [
                {
                    "exam": "chest CT",
                    "exam_source": "deferred_gap_closure_exam",
                    "target_gaps": ["G-D100058-new"],
                    "target_claims": ["new_spatial_progression_claim"],
                    "route_target_claims": ["new_spatial_progression_claim"],
                    "repeat_requested": True,
                    "repeat_authorized": True,
                    "repeat_reason_codes": ["NEW_TARGET_CLAIM"],
                    "source_evidence_version": 8,
                }
            ],
        }

        items = agent._strategy_order_items(
            strategy,
            collected_info={"symptoms": ["dyspnea"]},
            candidate_diseases=["radiation pneumonitis"],
            existing_results={"chest CT": {"status": "abnormal"}},
            max_items=None,
            add_strong_verification=False,
        )

        self.assertEqual(items, ["chest CT"])
        audit = strategy.get("exam_repeat_authorization_audit") or []
        self.assertTrue(audit)
        self.assertFalse(audit[0]["blocked"])
        self.assertTrue(audit[0]["repeat_authorized"])
        self.assertIn("NEW_TARGET_CLAIM", audit[0]["reason_codes"])
        self.assertEqual(audit[0]["target_claim_ids"], ["new_spatial_progression_claim"])


if __name__ == "__main__":
    unittest.main()
