import unittest
from types import SimpleNamespace

from agent.claim_resolution import (
    ANCHOR_SATISFIED,
    CONFLICTED,
    CONTRADICTED,
    FULLY_CLOSED,
    NOT_APPLICABLE,
    PARTIALLY_CLOSED,
    PATTERN_SUPPORTED_BUT_UNCONFIRMED,
    SUPPORTED,
    AnchorEvaluator,
    ClaimResolutionUpdater,
    claim_key,
    materialize_candidate_claim_states,
    hydrate_gap_with_claim_state,
)
from agent.diagnosis_eligibility import DEFERRED, PRIMARY_ELIGIBLE, DiagnosisEligibilityGate
from agent.diagnosis_engine import DiagnosticKnowledgeBase
from agent.exam_strategy import ExamStrategyAgent
from agent.targeted_exam_result_parser import ExamResultIntentBinding, TargetedExamResultParser


def radiation_gap(gap_id="G-D100058-derived_pattern_gap-1-post_radiotherapy_time_window"):
    return {
        "gap_id": gap_id,
        "entity_id": "D100058",
        "contract_id": "claim_anchor_contract:D100058",
        "contract_version": "claim_closure_plan_v1",
        "claim_closure_plan_version": "claim_closure_plan_v1",
        "claim_requirements": [
            {"claim_id": "pulmonary_morphology", "required_for_anchor": True},
            {"claim_id": "radiation_field_lung_consistency", "required_for_anchor": True},
            {"claim_id": "post_radiotherapy_time_window", "required_for_anchor": True},
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
            {
                "route_id": "route_radiotherapy_timing_history",
                "route_type": "history_inquiry",
                "target_claims": ["post_radiotherapy_time_window"],
            },
        ],
    }


def radiation_binding():
    return ExamResultIntentBinding(
        binding_id="B-RP-CLAIM",
        order_id="O-RP-CLAIM",
        requested_exam="chest CT",
        resolved_exam="CT",
        actual_result_exam="CT",
        result_id="result-ct-1",
        target_gap_ids=["G-D100058-derived_pattern_gap-1-post_radiotherapy_time_window"],
        target_claims=[
            "pulmonary_morphology",
            "radiation_field_lung_consistency",
            "post_radiotherapy_time_window",
        ],
        route_target_claims=[
            "pulmonary_morphology",
            "radiation_field_lung_consistency",
        ],
        target_candidate="radiation pneumonitis",
        entity_id="D100058",
        source_evidence_version=7,
    )


class ClaimResolutionTests(unittest.TestCase):
    def test_candidate_claim_state_materialization_deactivates_and_reactivates(self):
        contract = {
            "contract_id": "claim_anchor_contract:D100058",
            "contract_version": "claim_closure_plan_v1",
            "required_claims": [
                "pulmonary_morphology",
                "radiation_field_lung_consistency",
            ],
        }
        ledger, audit = materialize_candidate_claim_states(
            ledger={},
            contract_views=[
                {
                    "entity_id": "D100058",
                    "candidate": "radiation pneumonitis",
                    "claim_anchor_contract": contract,
                    "clinical_admission_reasons": ["PRIMARY_ELIGIBLE"],
                }
            ],
            active_entity_ids=["D100058"],
        )
        self.assertEqual(audit["materialized_claim_state_count"], 2)
        morph_key = claim_key(
            entity_id="D100058",
            claim_id="pulmonary_morphology",
            contract_id="claim_anchor_contract:D100058",
            contract_version="claim_closure_plan_v1",
        )
        self.assertEqual(ledger[morph_key]["lifecycle_status"], "ACTIVE")

        ledger, audit = materialize_candidate_claim_states(
            ledger=ledger,
            contract_views=[],
            active_entity_ids=["D100037"],
        )
        self.assertEqual(audit["inactivated_claim_state_count"], 2)
        self.assertEqual(ledger[morph_key]["lifecycle_status"], "INACTIVE")

        ledger, audit = materialize_candidate_claim_states(
            ledger=ledger,
            contract_views=[
                {
                    "entity_id": "D100058",
                    "candidate": "radiation pneumonitis",
                    "claim_anchor_contract": contract,
                    "clinical_admission_reasons": ["ARBITRATION_MEMBER"],
                }
            ],
            active_entity_ids=["D100058"],
        )
        self.assertEqual(audit["reactivated_claim_state_count"], 2)
        self.assertEqual(audit["materialized_claim_state_count"], 0)
        self.assertEqual(ledger[morph_key]["lifecycle_status"], "ACTIVE")

    def test_ct_claim_matches_persist_to_ledger_and_hydrate_rebuilt_gap(self):
        parsed = TargetedExamResultParser().parse(
            {
                "status": "abnormal",
                "result": {
                    "conclusion": (
                        "Chest CT shows ground-glass opacity and consolidation, "
                        "within prior radiation field."
                    )
                },
            },
            radiation_binding(),
        )
        updater = ClaimResolutionUpdater()
        updated = updater.update_from_parse(
            ledger={},
            parsed_result=parsed.to_dict(),
            intent_binding=radiation_binding().to_dict(),
            gap_contract=radiation_gap(),
        )

        ledger = updated["ledger"]
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
        self.assertEqual(ledger[morph_key]["resolution_status"], SUPPORTED)
        self.assertEqual(ledger[spatial_key]["resolution_status"], SUPPORTED)
        self.assertEqual(ledger[temporal_key]["resolution_status"], "UNRESOLVED")
        self.assertEqual(ledger[temporal_key]["last_attempt_status"], NOT_APPLICABLE)
        self.assertEqual(
            updated["gap_closure_evaluation"]["gap_closure_level"],
            PARTIALLY_CLOSED,
        )

        rebuilt = hydrate_gap_with_claim_state(
            radiation_gap("G-D100058-rebuilt-gap-id"),
            ledger,
        )
        self.assertEqual(rebuilt["gap_closure_level"], PARTIALLY_CLOSED)
        self.assertIn("post_radiotherapy_time_window", rebuilt["remaining_claims"])
        self.assertEqual(len(rebuilt["claim_resolutions"]), 3)

    def test_same_claim_event_is_idempotent(self):
        parsed = TargetedExamResultParser().parse(
            {
                "status": "abnormal",
                "result": {"conclusion": "Ground-glass opacity within prior radiation field."},
            },
            radiation_binding(),
        )
        updater = ClaimResolutionUpdater()
        first = updater.update_from_parse(
            ledger={},
            parsed_result=parsed.to_dict(),
            intent_binding=radiation_binding().to_dict(),
            gap_contract=radiation_gap(),
        )
        second = updater.update_from_parse(
            ledger=first["ledger"],
            parsed_result=parsed.to_dict(),
            intent_binding=radiation_binding().to_dict(),
            gap_contract=radiation_gap(),
        )
        self.assertEqual(second["persisted_claim_resolution_delta_count"], 0)
        self.assertTrue(
            any(
                item["merge_decision"] == "idempotent_replay"
                for item in second["claim_resolution_update_audit"]
            )
        )

    def test_supported_then_contradicted_claim_becomes_conflicted(self):
        updater = ClaimResolutionUpdater()
        supported = {
            "binding_id": "B",
            "target_gap_ids": ["G"],
            "actual_result_exam": "CT",
            "claim_matches": [
                {
                    "target_claim": "radiation_field_lung_consistency",
                    "claim_status": SUPPORTED,
                    "supporting_observations": ["lesion_within_prior_radiation_field"],
                    "source_type": "exam_result",
                }
            ],
        }
        contradicted = {
            "binding_id": "B2",
            "target_gap_ids": ["G"],
            "actual_result_exam": "CT",
            "claim_matches": [
                {
                    "target_claim": "radiation_field_lung_consistency",
                    "claim_status": CONTRADICTED,
                    "contradicting_observations": ["lesion_outside_prior_radiation_field"],
                    "source_type": "exam_result",
                }
            ],
        }
        binding = radiation_binding().to_dict()
        first = updater.update_from_parse(
            ledger={},
            parsed_result=supported,
            intent_binding=binding,
            gap_contract=radiation_gap(),
        )
        second_binding = dict(binding)
        second_binding["result_id"] = "result-ct-2"
        second = updater.update_from_parse(
            ledger=first["ledger"],
            parsed_result=contradicted,
            intent_binding=second_binding,
            gap_contract=radiation_gap(),
        )
        key = claim_key(
            entity_id="D100058",
            claim_id="radiation_field_lung_consistency",
            contract_id="claim_anchor_contract:D100058",
            contract_version="claim_closure_plan_v1",
        )
        self.assertEqual(second["ledger"][key]["resolution_status"], CONFLICTED)

    def test_anchor_evaluator_all_required_and_optional_claims(self):
        evaluator = AnchorEvaluator()
        contract = {
            "contract_id": "claim_anchor_contract:X",
            "contract_version": "1",
            "required_claims": ["A", "B"],
            "optional_claims": ["C"],
        }
        ledger = {
            claim_key(
                entity_id="X",
                claim_id="A",
                contract_id="claim_anchor_contract:X",
                contract_version="1",
            ): {
                "entity_id": "X",
                "claim_id": "A",
                "contract_id": "claim_anchor_contract:X",
                "contract_version": "1",
                "resolution_status": SUPPORTED,
            },
            claim_key(
                entity_id="X",
                claim_id="B",
                contract_id="claim_anchor_contract:X",
                contract_version="1",
            ): {
                "entity_id": "X",
                "claim_id": "B",
                "contract_id": "claim_anchor_contract:X",
                "contract_version": "1",
                "resolution_status": SUPPORTED,
            },
        }
        self.assertEqual(
            evaluator.evaluate(entity_id="X", anchor_contract=contract, ledger=ledger)[
                "anchor_status_after"
            ],
            ANCHOR_SATISFIED,
        )
        partial = dict(ledger)
        partial.pop(
            claim_key(
                entity_id="X",
                claim_id="B",
                contract_id="claim_anchor_contract:X",
                contract_version="1",
            )
        )
        self.assertEqual(
            evaluator.evaluate(entity_id="X", anchor_contract=contract, ledger=partial)[
                "anchor_status_after"
            ],
            PATTERN_SUPPORTED_BUT_UNCONFIRMED,
        )

    def test_eligibility_reads_claim_anchor_contract_without_score_bonus(self):
        knowledge = DiagnosticKnowledgeBase("data/ref_data")
        gate = DiagnosisEligibilityGate(knowledge)
        candidate = SimpleNamespace(
            diagnosis="放射性肺炎",
            entity_id="D100058",
            matched_evidence=["thoracic_radiotherapy", "ground_glass_opacity"],
            required_gaps=[],
            required_met=True,
            hard_contradiction=False,
            differential_only=False,
            source_prior=0.7,
            coverage_score=0.7,
            core_explanatory_coverage=0.5,
            diagnostic_evidence_score=0.0,
            core_evidence_score=0.0,
            claim_anchor_evaluation={
                "anchor_status_after": PATTERN_SUPPORTED_BUT_UNCONFIRMED,
                "unresolved_claims": ["post_radiotherapy_time_window"],
                "contradicted_claims": [],
                "conflicted_claims": [],
            },
        )
        partial = gate.evaluate(candidate)
        self.assertEqual(partial.status, DEFERRED)

        candidate.claim_anchor_evaluation = {
            "anchor_status_after": ANCHOR_SATISFIED,
            "unresolved_claims": [],
            "contradicted_claims": [],
            "conflicted_claims": [],
        }
        confirmed = gate.evaluate(candidate)
        self.assertEqual(confirmed.status, PRIMARY_ELIGIBLE)

    def test_exam_strategy_skips_completed_ct_claim_routes(self):
        gap = radiation_gap()
        gap["closure_exams"] = ["chest CT"]
        gap["remaining_claims"] = ["post_radiotherapy_time_window"]
        gap["resolved_claims"] = [
            "pulmonary_morphology",
            "radiation_field_lung_consistency",
        ]
        self.assertFalse(ExamStrategyAgent._gap_has_exam_closure_route(gap))


if __name__ == "__main__":
    unittest.main()
