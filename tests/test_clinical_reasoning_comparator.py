import unittest
from types import SimpleNamespace

from agent.clinical_reasoning_comparator import (
    KEEP_CURRENT_AND_DEFER_CONTENDER,
    NO_MATERIAL_DIFFERENCE,
    SWITCH_PRIMARY,
    UNLOCK_AND_DEFER,
    ClinicalReasoningComparator,
)


def candidate(
    name,
    *,
    entity_id="",
    anchor="NoValidAnchor",
    eligibility="DifferentialOnly",
    body_system="",
    matched=None,
    residual=None,
    required_met=False,
    hard_contradiction=False,
    patterns=None,
    bridges=None,
):
    return SimpleNamespace(
        diagnosis=name,
        entity_id=entity_id,
        eligibility_anchor_status=anchor,
        eligibility_status=eligibility,
        body_system=body_system,
        matched_evidence=list(matched or []),
        core_matched_evidence=list(matched or []),
        diagnostic_matched_evidence=[],
        residual_evidence=list(residual or []),
        unexplained_core_evidence=[],
        hard_contradiction=hard_contradiction,
        required_met=required_met,
        score=0.5,
        evidence_pattern_matches=list(patterns or []),
        clinical_pattern_matches=list(patterns or []),
        derived_pattern_assertions=list(bridges or []),
        bridge_protection_decisions=[],
        required_gaps=[],
        candidate_sources=[],
    )


class ClinicalReasoningComparatorTests(unittest.TestCase):
    def setUp(self):
        self.comparator = ClinicalReasoningComparator()

    def high_value_evidence(self):
        return [
            "ground_glass_opacity",
            "thoracic_radiotherapy",
            "pulmonary_consolidation",
            "pulmonary_volume_loss",
            "wheeze",
        ]

    def invalid_musculoskeletal_primary(self):
        return candidate(
            "post traumatic osteoarthritis",
            entity_id="D100031",
            anchor="AnchorSatisfied",
            eligibility="PrimaryEligible",
            body_system="musculoskeletal",
            matched=["osteophyte"],
            residual=self.high_value_evidence(),
            required_met=True,
        )

    def established_radiation_contender(self):
        return candidate(
            "radiation pneumonitis",
            entity_id="D100058",
            anchor="AnchorSatisfied",
            eligibility="PrimaryEligible",
            body_system="respiratory",
            matched=[
                "ground_glass_opacity",
                "thoracic_radiotherapy",
                "pulmonary_consolidation",
                "pulmonary_volume_loss",
            ],
            residual=[],
            required_met=True,
        )

    def test_claim_anchor_switches_without_matched_pattern(self):
        record = self.comparator.compare(
            self.invalid_musculoskeletal_primary(),
            self.established_radiation_contender(),
            high_value_evidence=self.high_value_evidence(),
        )

        self.assertEqual(record["recommended_action"], SWITCH_PRIMARY)
        self.assertEqual(record["contender_establishment_status"], "ESTABLISHED")
        self.assertTrue(record["contender_clinically_established"])
        self.assertEqual(record["incumbent_validity_status"], "INVALIDATED")
        self.assertNotEqual(record["material_difference_status"], "NONE")
        reasons = set(record["decision_reason_codes"])
        self.assertIn("CLINICALLY_ESTABLISHED_BY_CLAIM_ANCHOR", reasons)
        self.assertIn("CLAIM_ANCHOR_ESTABLISHED", reasons)
        self.assertIn("INCUMBENT_PRIMARY_PROTECTION_LOST", reasons)
        self.assertIn("SWITCH_PRIMARY_AUTHORIZED", reasons)

    def test_primary_eligible_cannot_self_establish_without_anchor(self):
        contender = self.established_radiation_contender()
        contender.eligibility_anchor_status = "NoValidAnchor"
        contender.eligibility_status = "PrimaryEligible"

        record = self.comparator.compare(
            self.invalid_musculoskeletal_primary(),
            contender,
            high_value_evidence=self.high_value_evidence(),
        )

        self.assertNotEqual(record["recommended_action"], SWITCH_PRIMARY)
        self.assertEqual(record["contender_establishment_status"], "NOT_ESTABLISHED")
        reasons = set(record["decision_reason_codes"])
        self.assertIn("ELIGIBILITY_WITHOUT_CLINICAL_ESTABLISHMENT", reasons)
        self.assertIn("CONTENDER_CLINICAL_ESTABLISHMENT_GATE_NOT_MET", reasons)

    def test_pattern_only_contender_is_provisional_and_does_not_switch(self):
        contender = self.established_radiation_contender()
        contender.eligibility_anchor_status = "PatternSupportedButUnconfirmed"
        contender.eligibility_status = "Deferred"
        contender.required_met = False
        contender.clinical_pattern_matches = [
            {
                "pattern_id": "radiation_pattern",
                "verification_status": "verified",
                "supporting_findings": ["ground_glass_opacity"],
            }
        ]

        record = self.comparator.compare(
            self.invalid_musculoskeletal_primary(),
            contender,
            high_value_evidence=self.high_value_evidence(),
        )

        self.assertEqual(record["contender_establishment_status"], "PROVISIONAL")
        self.assertEqual(record["recommended_action"], UNLOCK_AND_DEFER)
        self.assertIn(
            "MATERIAL_GAIN_BUT_CONTENDER_UNCONFIRMED",
            record["decision_reason_codes"],
        )

    def test_protected_incumbent_with_weak_contender_has_no_material_difference(self):
        primary = candidate(
            "pneumonia",
            entity_id="D_PRIMARY",
            anchor="AnchorSatisfied",
            eligibility="PrimaryEligible",
            body_system="respiratory",
            matched=["ground_glass_opacity", "pulmonary_consolidation"],
            residual=[],
            required_met=True,
        )
        contender = candidate(
            "weak contender",
            entity_id="D_WEAK",
            anchor="NoValidAnchor",
            eligibility="DifferentialOnly",
            body_system="musculoskeletal",
            matched=["pain"],
            residual=["ground_glass_opacity"],
        )

        record = self.comparator.compare(
            primary,
            contender,
            high_value_evidence=["ground_glass_opacity", "pulmonary_consolidation"],
        )

        self.assertEqual(record["recommended_action"], NO_MATERIAL_DIFFERENCE)
        self.assertEqual(record["material_difference_status"], "NONE")

    def test_established_challenger_can_beat_still_valid_incumbent(self):
        primary = candidate(
            "partial respiratory primary",
            entity_id="D_PRIMARY",
            anchor="AnchorSatisfied",
            eligibility="PrimaryEligible",
            body_system="respiratory",
            matched=["ground_glass_opacity"],
            residual=["pulmonary_consolidation", "pulmonary_volume_loss"],
            required_met=True,
        )
        contender = self.established_radiation_contender()

        record = self.comparator.compare(
            primary,
            contender,
            high_value_evidence=[
                "ground_glass_opacity",
                "thoracic_radiotherapy",
                "pulmonary_consolidation",
                "pulmonary_volume_loss",
            ],
        )

        self.assertEqual(record["recommended_action"], SWITCH_PRIMARY)
        self.assertEqual(record["contender_establishment_status"], "ESTABLISHED")

    def test_hard_contradiction_overrides_anchor_satisfied(self):
        contender = self.established_radiation_contender()
        contender.hard_contradiction = True

        record = self.comparator.compare(
            self.invalid_musculoskeletal_primary(),
            contender,
            high_value_evidence=self.high_value_evidence(),
        )

        self.assertEqual(record["recommended_action"], "REJECT_CONTENDER")
        self.assertIn("CONTENDER_HARD_BLOCKED", record["decision_reason_codes"])


if __name__ == "__main__":
    unittest.main()
