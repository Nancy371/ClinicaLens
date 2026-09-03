import unittest

import yaml

from agent.diagnosis_engine import CandidateScore, DiagnosisDecision, DiagnosisDecisionEngine
from agent.diagnosis_eligibility import PRIMARY_ELIGIBLE
from agent.submission_authorization import (
    AUTH_AUTHORIZED,
    AUTH_NOT_AUTHORIZED,
    ROLE_ASSOCIATED_FINDING,
    ROLE_COMPLICATION,
    ROLE_PRIMARY,
)


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidate(engine, name, *, score=0.8, matched=None, diagnosis_type="disease"):
    return CandidateScore(
        diagnosis=name,
        score=score,
        support_score=score,
        source_prior=0.5,
        explanation_score=0.75,
        coverage_score=0.75,
        residual_score=0.15,
        contradiction_penalty=0.0,
        required_met=True,
        hard_contradiction=False,
        matched_evidence=list(matched or ["diagnosis:" + name]),
        explained_evidence=list(matched or ["diagnosis:" + name]),
        core_matched_evidence=list(matched or []),
        diagnostic_matched_evidence=list(matched or []),
        component_scores={"objective_evidence": 1.0},
        eligibility_status=PRIMARY_ELIGIBLE,
        eligibility_anchor_status="AnchorSatisfied",
        diagnosis_type=diagnosis_type,
        entity_id=engine.knowledge.entity_id_for(name),
        canonical_name=name,
        submission_name=name,
    )


class SubmissionAuthorizationLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = DiagnosisDecisionEngine(load_config(), "data/ref_data")

    def decision(self, candidates, requested):
        return DiagnosisDecision(
            final_diagnoses=list(requested),
            trusted_diagnoses=list(requested),
            candidates=list(candidates),
            unexplained_evidence=[],
            confidence=0.0,
            margin=0.0,
            low_confidence=False,
            judge_primary=requested[0] if requested else "",
            judge_decision={
                "primary_status": "locked",
                "needs_discriminating_exams": False,
                "judge_primary": requested[0] if requested else "",
            },
            evidence_snapshot_hash="sha256:test",
            diagnostic_state_version=7,
        )

    def test_radiation_primary_blocks_atelectasis_as_associated_finding(self):
        radiation = candidate(
            self.engine,
            "放射性肺炎",
            matched=[
                "thoracic_radiotherapy",
                "ground_glass_opacity",
                "pulmonary_consolidation",
                "lesion_within_prior_radiation_field",
            ],
        )
        atelectasis = candidate(
            self.engine,
            "肺不张",
            score=0.82,
            matched=["atelectasis", "lung_volume_loss"],
            diagnosis_type="structural",
        )
        decision = self.decision([radiation, atelectasis], ["放射性肺炎", "肺不张"])

        self.engine.authorize_final_diagnoses(decision, ["放射性肺炎", "肺不张"])

        self.assertEqual(decision.final_diagnoses, ["放射性肺炎"])
        self.assertEqual(radiation.submission_role, ROLE_PRIMARY)
        self.assertEqual(radiation.submission_authorization, AUTH_AUTHORIZED)
        self.assertEqual(atelectasis.submission_role, ROLE_ASSOCIATED_FINDING)
        self.assertEqual(atelectasis.submission_authorization, AUTH_NOT_AUTHORIZED)
        self.assertFalse(atelectasis.submission_authorized)
        self.assertEqual(decision.associated_finding_block_count, 1)
        self.assertEqual(decision.submission_authorization_bypass_count, 0)
        self.assertTrue(
            any(
                item.get("diagnosis_name") == "肺不张"
                and item.get("submission_role") == ROLE_ASSOCIATED_FINDING
                for item in decision.submission_authorization_records
            )
        )
        self.assertTrue(
            any(
                item.get("source_entity_id") == atelectasis.entity_id
                and item.get("target_entity_id") == radiation.entity_id
                for item in decision.submission_dependency_edges
            )
        )

    def test_material_complication_can_still_be_authorized(self):
        mitral = candidate(
            self.engine,
            "二尖瓣反流",
            matched=["mitral_regurgitation", "diagnosis:二尖瓣反流"],
            diagnosis_type="structural",
        )
        heart_failure = candidate(
            self.engine,
            "心力衰竭",
            score=0.62,
            matched=[
                "fluid_retention_pattern",
                "leg_edema",
                "paroxysmal_nocturnal_dyspnea",
                "dyspnea_on_exertion",
            ],
            diagnosis_type="state",
        )
        heart_failure.explained_by_root_cause = "二尖瓣反流"
        heart_failure.root_cause_submit_as_final = True
        decision = self.decision([mitral, heart_failure], ["二尖瓣反流", "心力衰竭"])

        self.engine.authorize_final_diagnoses(decision, ["二尖瓣反流", "心力衰竭"])

        self.assertEqual(decision.final_diagnoses, ["二尖瓣反流", "心力衰竭"])
        self.assertEqual(heart_failure.submission_role, ROLE_COMPLICATION)
        self.assertEqual(heart_failure.submission_authorization, AUTH_AUTHORIZED)
        self.assertEqual(decision.authorized_secondary_count, 1)
        current_records = [
            item
            for item in decision.submission_authorization_records
            if item.get("submission_authorization") == AUTH_AUTHORIZED
            and item.get("diagnostic_state_version") == decision.diagnostic_state_version
        ]
        self.assertEqual(
            {item.get("diagnosis_name") for item in current_records},
            set(decision.final_diagnoses),
        )


if __name__ == "__main__":
    unittest.main()
