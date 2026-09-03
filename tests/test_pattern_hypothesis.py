import unittest

import yaml

from agent.clinical_evidence import EvidenceBundle, Observation
from agent.diagnosis_engine import DiagnosisDecisionEngine
from agent.pattern_hypothesis import (
    EvidenceRefResolver,
    EvidenceRelationBinder,
    PatternHypothesisVerifier,
    ThinkingSnapshot,
    _observation_ref,
)


RAD_PNEUMONITIS = "\u653e\u5c04\u6027\u80ba\u708e"
MITRAL_REGURGITATION = "\u4e8c\u5c16\u74e3\u53cd\u6d41"
PAVM_ENTITY_ID = "D100055"


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def radiation_evidence() -> EvidenceBundle:
    return EvidenceBundle(
        [
            Observation(
                "thoracic_radiotherapy",
                "patient_reported_observation",
                confidence=0.94,
                information_value=0.9,
            ),
            Observation(
                "post_radiotherapy_time_window",
                "disease_agnostic_deterministic_relation",
                confidence=0.9,
                information_value=0.86,
            ),
            Observation(
                "ground_glass_opacity",
                "imaging_result",
                confidence=0.92,
                information_value=0.92,
            ),
            Observation(
                "dyspnea",
                "patient_reported_observation",
                confidence=0.88,
                information_value=0.62,
            ),
        ]
    )


def typed_radiation_relation_evidence() -> EvidenceBundle:
    return EvidenceBundle(
        [
            Observation(
                "thoracic_radiotherapy",
                "patient_reported_observation",
                anatomy="thorax",
                temporality="3\u4e2a\u6708\u524d",
                observation_type="treatment_history",
                semantic_level="fact",
                confidence=0.94,
                information_value=0.9,
            ),
            Observation(
                "dyspnea",
                "patient_reported_observation",
                observation_type="symptom",
                semantic_level="fact",
                confidence=0.88,
                information_value=0.62,
            ),
            Observation(
                "pulmonary_infiltrate",
                "imaging_result",
                anatomy="lung",
                observation_type="imaging_finding",
                semantic_level="fact",
                confidence=0.92,
                information_value=0.92,
            ),
        ]
    )


def radiation_hypothesis():
    return {
        "pattern_hypothesis_id": "PH_RAD_001",
        "pattern_name": "post_thoracic_radiotherapy_lung_injury_pattern",
        "pattern_type": "temporal_causal_multievidence",
        "evidence_bindings": [
            {
                "evidence_id": "thoracic_radiotherapy",
                "role": "support",
                "expected_polarity": "positive",
                "relation_slot": "exposure",
            },
            {
                "evidence_id": "post_radiotherapy_time_window",
                "role": "support",
                "expected_polarity": "positive",
                "relation_slot": "temporal_relation",
            },
            {
                "evidence_id": "ground_glass_opacity",
                "role": "support",
                "expected_polarity": "positive",
                "relation_slot": "imaging_or_objective_finding",
            },
            {
                "evidence_id": "dyspnea",
                "role": "support",
                "expected_polarity": "positive",
                "relation_slot": "organ_manifestation",
            },
        ],
        "relations": [
            {
                "type": "temporal_after",
                "from": "thoracic_radiotherapy",
                "to": "dyspnea",
            }
        ],
        "suggested_diseases": [
            {
                "name": RAD_PNEUMONITIS,
                "canonical_id": "radiation_pneumonitis",
                "hypothesis_confidence": 0.86,
            }
        ],
        "missing_evidence_requests": [
            {"target_evidence": "infection_exclusion", "importance": "supportive"}
        ],
        "model_confidence": 0.86,
    }


class PatternHypothesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.engine = DiagnosisDecisionEngine(cls.config, "data/ref_data")
        cls.verifier = PatternHypothesisVerifier(cls.engine.knowledge, config=cls.config)

    def test_missing_source_evidence_is_rejected(self):
        evidence = radiation_evidence()
        payload = radiation_hypothesis()
        payload["evidence_bindings"][0]["evidence_id"] = "missing_evidence"
        context = self.engine.build_pattern_recall_context(
            {"clinical_pattern_hypotheses": [payload]},
            evidence,
        )
        rejected = context["pattern_verification_results"][0]
        self.assertEqual(rejected["verification_status"], "rejected")
        self.assertIn("unsupported_source_evidence", rejected["rejection_reasons"])
        self.assertTrue(
            any(
                item["entity_id"] == "D100058"
                and item["pattern_hypothesis_id"].startswith("PH_DET")
                for item in context["pattern_recall_signals"]
            )
        )

    def test_typed_radiation_relation_activates_without_temporal_observation(self):
        evidence = typed_radiation_relation_evidence()
        binding = EvidenceRelationBinder(evidence.observations, self.verifier.evidence_ontology).bind(
            "exposure_temporal_organ_injury"
        )
        audit = binding["audit"]
        self.assertEqual(audit["activation_status"], "activated")
        self.assertEqual(audit["missing_slots"], [])
        self.assertIn("exposure", audit["bound_slots"])
        self.assertIn("organ_manifestation", audit["bound_slots"])
        self.assertIn("imaging_or_objective_finding", audit["bound_slots"])
        constraints = {
            item["constraint_type"]: item["status"]
            for item in audit["constraint_results"]
        }
        self.assertEqual(constraints["temporal_after"], "satisfied")
        self.assertEqual(constraints["anatomical_consistency"], "satisfied")

        context = self.engine.build_pattern_recall_context(
            {},
            evidence,
            case_id="Patient_03674",
            evidence_snapshot_id="ES_TYPED_RAD",
        )
        self.assertTrue(context["pattern_hypotheses"])
        self.assertTrue(context["pattern_recall_signals"])
        self.assertTrue(
            any(
                item["entity_id"] == "D100058"
                for item in context["pattern_recall_signals"]
            )
        )

    def test_generic_radiotherapy_history_is_partial_not_activated(self):
        evidence = EvidenceBundle(
            [
                Observation(
                    "history_of_radiotherapy",
                    "patient_reported_observation",
                    observation_type="treatment_history",
                    semantic_level="fact",
                    confidence=0.86,
                ),
                Observation(
                    "dyspnea",
                    "patient_reported_observation",
                    observation_type="symptom",
                    semantic_level="fact",
                    confidence=0.88,
                ),
                Observation(
                    "pulmonary_infiltrate",
                    "imaging_result",
                    anatomy="lung",
                    observation_type="imaging_finding",
                    semantic_level="fact",
                    confidence=0.92,
                ),
            ]
        )
        audit = EvidenceRelationBinder(evidence.observations, self.verifier.evidence_ontology).bind(
            "exposure_temporal_organ_injury"
        )["audit"]
        self.assertEqual(audit["activation_status"], "partial")
        self.assertIn("exposure", audit["missing_slots"])

    def test_objective_injury_prefers_canonical_imaging_over_field_finding(self):
        evidence = EvidenceBundle(
            [
                Observation(
                    "thoracic_radiotherapy",
                    "patient_reported_observation",
                    anatomy="thorax",
                    temporality="3\u4e2a\u6708\u524d",
                    observation_type="treatment_history",
                    semantic_level="fact",
                    confidence=0.94,
                ),
                Observation(
                    "dyspnea",
                    "patient_reported_observation",
                    observation_type="symptom",
                    semantic_level="fact",
                    confidence=0.88,
                ),
                Observation(
                    "field:0",
                    "imaging_result",
                    anatomy="lung",
                    observation_type="imaging_finding",
                    semantic_level="fact",
                    confidence=0.95,
                    information_value=0.08,
                ),
                Observation(
                    "pulmonary_consolidation",
                    "imaging_result",
                    anatomy="lung",
                    observation_type="imaging_finding",
                    semantic_level="fact",
                    confidence=0.9,
                    information_value=0.9,
                ),
            ]
        )
        audit = EvidenceRelationBinder(evidence.observations, self.verifier.evidence_ontology).bind(
            "exposure_temporal_organ_injury"
        )["audit"]
        objective = audit["supporting_evidence"]["imaging_or_objective_finding"]
        self.assertEqual(objective["canonical_concept"], "pulmonary_consolidation")

    def test_non_pulmonary_imaging_does_not_fill_pulmonary_objective_slot(self):
        evidence = EvidenceBundle(
            [
                Observation(
                    "thoracic_radiotherapy",
                    "patient_reported_observation",
                    anatomy="thorax",
                    temporality="3\u4e2a\u6708\u524d",
                    observation_type="treatment_history",
                    semantic_level="fact",
                    confidence=0.94,
                ),
                Observation(
                    "dyspnea",
                    "patient_reported_observation",
                    observation_type="symptom",
                    semantic_level="fact",
                    confidence=0.88,
                ),
                Observation(
                    "osteophyte",
                    "imaging_result",
                    anatomy="spine",
                    observation_type="imaging_finding",
                    semantic_level="fact",
                    confidence=0.95,
                    information_value=0.8,
                ),
            ]
        )
        audit = EvidenceRelationBinder(evidence.observations, self.verifier.evidence_ontology).bind(
            "exposure_temporal_organ_injury"
        )["audit"]
        self.assertIn("imaging_or_objective_finding", audit["missing_slots"])

    def test_pelvic_radiotherapy_does_not_activate_pulmonary_injury_relation(self):
        evidence = EvidenceBundle(
            [
                Observation(
                    "thoracic_radiotherapy",
                    "patient_reported_observation",
                    anatomy="pelvis",
                    temporality="3\u4e2a\u6708\u524d",
                    observation_type="treatment_history",
                    semantic_level="fact",
                    confidence=0.94,
                ),
                Observation(
                    "dyspnea",
                    "patient_reported_observation",
                    observation_type="symptom",
                    semantic_level="fact",
                    confidence=0.88,
                ),
                Observation(
                    "pulmonary_infiltrate",
                    "imaging_result",
                    anatomy="lung",
                    observation_type="imaging_finding",
                    semantic_level="fact",
                    confidence=0.92,
                ),
            ]
        )
        audit = EvidenceRelationBinder(evidence.observations, self.verifier.evidence_ontology).bind(
            "exposure_temporal_organ_injury"
        )["audit"]
        self.assertEqual(audit["activation_status"], "partial")
        constraints = {
            item["constraint_type"]: item["status"]
            for item in audit["constraint_results"]
        }
        self.assertEqual(constraints["anatomical_consistency"], "unresolved")

    def test_reasoning_inference_source_is_rejected(self):
        evidence = radiation_evidence()
        evidence.observations[0].source = "reasoning_inference"
        context = self.engine.build_pattern_recall_context(
            {"clinical_pattern_hypotheses": [radiation_hypothesis()]},
            evidence,
        )
        rejected = context["pattern_verification_results"][0]
        self.assertEqual(rejected["verification_status"], "rejected")
        self.assertIn("reasoning_inference_source", rejected["rejection_reasons"])
        self.assertEqual(context["unverified_pattern_leakage_count"], 0)

    def test_radiation_pattern_recall_adds_candidate_without_evidence_leakage(self):
        evidence = radiation_evidence()
        llm = {"clinical_pattern_hypotheses": [radiation_hypothesis()]}
        decision = self.engine.decide(llm, [], evidence)
        top20 = [item.diagnosis for item in decision.candidates[:20]]
        self.assertIn(RAD_PNEUMONITIS, top20)
        candidate = next(item for item in decision.candidates if item.diagnosis == RAD_PNEUMONITIS)
        self.assertTrue(
            any(
                source.get("source") == "llm_pattern_hypothesis"
                and source.get("metadata", {}).get("judge_evidence_weight") == 0.0
                and source.get("metadata", {}).get("eligibility_evidence_weight") == 0.0
                and source.get("metadata", {}).get("active_gap_write_permission") == "none"
                for source in candidate.candidate_sources
            )
        )
        self.assertNotIn("post_thoracic_radiotherapy_lung_injury_pattern", candidate.matched_evidence)
        self.assertEqual(decision.unverified_pattern_leakage_count, 0)
        self.assertEqual(decision.pattern_generated_active_gaps, 0)
        self.assertLessEqual(decision.pattern_expansion_round_count, 1)
        self.assertTrue(decision.pattern_gap_suggestions)

    def test_generic_radiotherapy_history_cannot_be_protected_recall(self):
        evidence = EvidenceBundle(
            [
                Observation(
                    "history_of_radiotherapy",
                    "patient_reported_observation",
                    confidence=0.9,
                    information_value=0.5,
                ),
                Observation(
                    "post_radiotherapy_time_window",
                    "disease_agnostic_deterministic_relation",
                    confidence=0.9,
                    information_value=0.86,
                ),
                Observation(
                    "ground_glass_opacity",
                    "imaging_result",
                    confidence=0.92,
                    information_value=0.92,
                ),
                Observation(
                    "dyspnea",
                    "patient_reported_observation",
                    confidence=0.88,
                    information_value=0.62,
                ),
            ]
        )
        payload = radiation_hypothesis()
        payload["evidence_bindings"][0]["evidence_id"] = "history_of_radiotherapy"
        context = self.engine.build_pattern_recall_context(
            {"clinical_pattern_hypotheses": [payload]},
            evidence,
        )
        result = context["pattern_verification_results"][0]
        self.assertNotEqual(result["verification_status"], "verified")
        self.assertEqual(context["pattern_protected_candidate_recall"], [])

    def test_evidence_ref_resolver_uses_ontology_parent_without_fabrication(self):
        observations = [
            Observation("pneumonia_infiltrate", "imaging_result", confidence=0.9),
        ]
        resolver = EvidenceRefResolver(
            observations,
            {
                "pulmonary_abnormality": {
                    "children": ["pneumonia_infiltrate", "ground_glass_opacity"],
                    "aliases": [],
                }
            },
        )
        resolved = resolver.resolve("pulmonary_abnormality")
        self.assertEqual(resolved.binding_status, "resolved")
        self.assertEqual(resolved.binding_method, "ontology_parent")
        self.assertEqual(resolved.canonical_concept, "pulmonary_abnormality")
        self.assertEqual(
            resolved.candidate_matches[0]["canonical_concept"],
            "pneumonia_infiltrate",
        )
        self.assertNotEqual(resolved.candidate_matches[0]["canonical_concept"], "ground_glass_opacity")

    def test_evidence_ref_resolver_reports_ambiguous_parent(self):
        observations = [
            Observation("pneumonia_infiltrate", "imaging_result", field_path="exam[1]"),
            Observation("ground_glass_opacity", "imaging_result", field_path="exam[2]"),
        ]
        resolver = EvidenceRefResolver(
            observations,
            {
                "pulmonary_abnormality": {
                    "children": ["pneumonia_infiltrate", "ground_glass_opacity"],
                    "aliases": [],
                }
            },
        )
        resolved = resolver.resolve("pulmonary_abnormality")
        self.assertEqual(resolved.binding_status, "ambiguous")
        self.assertEqual(resolved.failure_reason, "ambiguous_evidence_binding")

    def test_legacy_pneumonia_infiltrate_does_not_fill_radiation_objective_slot(self):
        observations = [
            Observation(
                "thoracic_radiotherapy",
                "patient_reported_observation",
                confidence=0.9,
                observation_type="treatment_history",
                semantic_level="fact",
                anatomy="thorax",
                temporality="3_months_ago",
            ),
            Observation("dyspnea", "patient_reported_observation", observation_type="symptom"),
            Observation(
                "pneumonia_infiltrate",
                "imaging_result",
                confidence=0.9,
                observation_type="imaging_finding",
                semantic_level="fact",
                anatomy="lung",
            ),
        ]
        binder = EvidenceRelationBinder(observations, {})
        result = binder.bind("exposure_temporal_organ_injury")
        audit = result["audit"]
        self.assertNotIn("imaging_or_objective_finding", audit["bound_slots"])
        self.assertIn("imaging_or_objective_finding", audit["missing_slots"])

    def test_observation_ref_is_stable_under_bundle_order(self):
        first = Observation("dyspnea", "patient_reported_observation", field_path="symptom[1]")
        second = Observation("cough", "patient_reported_observation", field_path="symptom[2]")
        refs_a = [_observation_ref(item) for item in [first, second]]
        refs_b = [_observation_ref(item) for item in [second, first]]
        self.assertEqual(refs_a[0], refs_b[1])
        self.assertEqual(refs_a[1], refs_b[0])
        self.assertNotEqual(refs_a[0], refs_a[1])

    def test_relation_binder_does_not_turn_mvp_into_regurgitation(self):
        evidence = EvidenceBundle(
            [
                Observation("mitral_valve_prolapse", "imaging_result", confidence=0.95),
                Observation("orthopnea", "patient_reported_observation", confidence=0.9),
                Observation("pink_frothy_sputum", "patient_reported_observation", confidence=0.9),
            ]
        )
        binder = EvidenceRelationBinder(evidence.observations, {})
        binding = binder.bind("structural_function_abnormality")
        audit = binding["audit"]
        self.assertEqual(audit["activation_status"], "activated")
        self.assertIn("structure_or_credible_sign", audit["bound_slots"])
        self.assertIn("function_impairment", audit["bound_slots"])
        self.assertNotIn("regurgitation_specific", audit["bound_slots"])

    def test_deterministic_relation_audit_and_gap_suggestion_are_recall_only(self):
        evidence = radiation_evidence()
        context = self.engine.build_pattern_recall_context({}, evidence)
        self.assertTrue(
            any(
                item["generator_source"] == "deterministic_relation"
                for item in context["pattern_hypotheses"]
            )
        )
        verified = [
            item
            for item in context["pattern_verification_results"]
            if item["verification_status"] == "verified"
        ]
        self.assertTrue(verified)
        audit = verified[0]["relation_activation_audit"]
        self.assertEqual(audit["activation_status"], "activated")
        self.assertIn("exposure", audit["bound_slots"])
        self.assertIn("organ_manifestation", audit["bound_slots"])
        self.assertIn("imaging_or_objective_finding", audit["bound_slots"])
        self.assertTrue(context["pattern_gap_suggestions"])
        self.assertTrue(
            all(
                item.get("active_gap_write_permission") == "none"
                for item in context["pattern_gap_suggestions"]
            )
        )

    def test_valvular_family_expansion_keeps_family_specificity(self):
        evidence = EvidenceBundle(
            [
                Observation("mitral_valve_prolapse", "imaging_result", confidence=0.96, information_value=0.9),
                Observation("orthopnea", "patient_reported_observation", confidence=0.9, information_value=0.7),
                Observation("pink_frothy_sputum", "patient_reported_observation", confidence=0.9, information_value=0.8),
            ]
        )
        context = self.engine.build_pattern_recall_context({}, evidence)
        signals = [item for item in context["pattern_recall_signals"] if item["entity_id"] == "D100012"]
        self.assertTrue(signals)
        self.assertEqual(signals[0]["admission_level"], "family_expansion")
        self.assertEqual(signals[0]["verified_specificity"], "family")

    def test_pattern_recall_merges_existing_mitral_regurgitation_entity(self):
        evidence = EvidenceBundle(
            [
                Observation("cardiac_murmur", "clinician_observed_finding", confidence=0.96, information_value=0.96),
                Observation("left_heart_enlargement", "imaging_result", confidence=0.96, information_value=0.96),
                Observation("dyspnea", "patient_reported_observation", confidence=0.92, information_value=0.9),
            ]
        )
        llm = {
            "clinical_pattern_hypotheses": [
                {
                    "pattern_hypothesis_id": "PH_MR_001",
                    "pattern_name": "left_sided_valvular_regurgitation_pattern",
                    "pattern_type": "mechanism_multievidence",
                    "evidence_bindings": [
                        {
                            "evidence_id": "cardiac_murmur",
                            "role": "support",
                            "expected_polarity": "positive",
                            "relation_slot": "organ_manifestation",
                        },
                        {
                            "evidence_id": "left_heart_enlargement",
                            "role": "support",
                            "expected_polarity": "positive",
                            "relation_slot": "imaging_or_objective_finding",
                        },
                        {
                            "evidence_id": "dyspnea",
                            "role": "support",
                            "expected_polarity": "positive",
                            "relation_slot": "support",
                        },
                    ],
                    "relations": [
                        {
                            "type": "anatomical_consistency",
                            "from": "cardiac_murmur",
                            "to": "left_heart_enlargement",
                        }
                    ],
                    "suggested_diseases": [
                        {"name": MITRAL_REGURGITATION, "canonical_id": "D100012"}
                    ],
                    "missing_evidence_requests": [
                        {"target_evidence": "echo_regurgitant_jet"}
                    ],
                }
            ]
        }
        decision = self.engine.decide(llm, [], evidence)
        candidate = next(item for item in decision.candidates if item.entity_id == "D100012")
        self.assertEqual(candidate.diagnosis, MITRAL_REGURGITATION)
        self.assertTrue(
            any(source.get("source") == "llm_pattern_hypothesis" for source in candidate.candidate_sources)
        )
        self.assertEqual(decision.pattern_generated_active_gaps, 0)

    def test_thinking_structured_pattern_can_drive_recall(self):
        evidence = radiation_evidence()
        thinking = {
            "differential_diagnosis": [
                {"diagnosis": "\u652f\u6c14\u7ba1\u80ba\u708e", "likelihood": 0.5}
            ],
            "clinical_pattern_proposals": [radiation_hypothesis()],
            "key_unknowns": ["infection_exclusion"],
            "is_sufficient": False,
        }
        snapshot = ThinkingSnapshot.from_thinking(
            thinking,
            case_id="Patient_03674",
            patient_id="Patient_03674",
            phase="examination",
            evidence_snapshot_id="ES_TEST",
        )
        context = self.engine.build_pattern_recall_context(
            {},
            evidence,
            case_id="Patient_03674",
            evidence_snapshot_id="ES_TEST",
            thinking_snapshots=[snapshot.to_dict()],
        )
        self.assertEqual(context["thinking_snapshot_count"], 1)
        self.assertGreaterEqual(len(context["pattern_hypotheses"]), 1)
        self.assertTrue(context["pattern_recall_signals"])
        audit = context["pattern_recall_audit"]
        self.assertGreaterEqual(audit["proposal_count"], 1)
        self.assertGreaterEqual(audit["verification_statuses"]["verified"], 1)
        self.assertIn("D100058", audit["signal_entity_ids"])
        decision = self.engine.decide({}, [], evidence, pattern_recall_context=context)
        top20 = [item.diagnosis for item in decision.candidates[:20]]
        self.assertIn(RAD_PNEUMONITIS, top20)
        self.assertGreaterEqual(decision.pattern_recall_audit["proposal_count"], 1)
        self.assertTrue(
            any(
                item.get("entity_id") == "D100058"
                and item.get("admitted_to_controlled_pool")
                for item in decision.pattern_candidate_admissions
            )
        )

    def test_thinking_disease_name_only_does_not_strong_recall_from_thinking(self):
        evidence = radiation_evidence()
        snapshot = ThinkingSnapshot.from_thinking(
            {
                "differential_diagnosis": [
                    {"diagnosis": RAD_PNEUMONITIS, "likelihood": 0.8}
                ],
                "action_reasoning": "\u8003\u8651\u653e\u5c04\u6027\u80ba\u708e",
            },
            case_id="Patient_03674",
            patient_id="Patient_03674",
            phase="examination",
            evidence_snapshot_id="ES_TEST",
        )
        context = self.engine.build_pattern_recall_context(
            {},
            evidence,
            case_id="Patient_03674",
            evidence_snapshot_id="ES_TEST",
            thinking_snapshots=[snapshot.to_dict()],
        )
        self.assertTrue(context["pattern_hypotheses"])
        self.assertTrue(
            all(
                item["generator_source"] == "deterministic_relation"
                for item in context["pattern_hypotheses"]
            )
        )
        self.assertTrue(context["pattern_recall_signals"])
        self.assertEqual(
            context["pattern_recall_audit"]["compiler_audit"]["sources"]["reasoning_adapter"]["generated"],
            0,
        )

    def test_thinking_differential_with_refs_is_not_protected_without_relations(self):
        evidence = radiation_evidence()
        snapshot = ThinkingSnapshot.from_thinking(
            {
                "differential_diagnosis": [
                    {
                        "diagnosis": RAD_PNEUMONITIS,
                        "likelihood": 0.8,
                        "supporting_evidence_refs": [
                            "thoracic_radiotherapy",
                            "ground_glass_opacity",
                        ],
                    }
                ],
            },
            case_id="Patient_03674",
            patient_id="Patient_03674",
            phase="examination",
            evidence_snapshot_id="ES_TEST",
        )
        context = self.engine.build_pattern_recall_context(
            {},
            evidence,
            case_id="Patient_03674",
            evidence_snapshot_id="ES_TEST",
            thinking_snapshots=[snapshot.to_dict()],
        )
        self.assertTrue(context["pattern_hypotheses"])
        self.assertTrue(context["pattern_recall_signals"])
        self.assertTrue(
            any(
                item["pattern_hypothesis_id"].startswith("PH_DET")
                for item in context["pattern_recall_signals"]
            )
        )
        self.assertEqual(
            context["pattern_recall_audit"]["compiler_audit"]["sources"]["reasoning_adapter"]["generated"],
            0,
        )

    def test_thinking_and_diagnosis_draft_duplicate_pattern_is_deduped(self):
        evidence = radiation_evidence()
        payload = radiation_hypothesis()
        snapshot = ThinkingSnapshot.from_thinking(
            {
                "clinical_pattern_proposals": [payload],
                "differential_diagnosis": [],
            },
            case_id="Patient_03674",
            patient_id="Patient_03674",
            phase="examination",
            evidence_snapshot_id="ES_TEST",
        )
        context = self.engine.build_pattern_recall_context(
            {"clinical_pattern_hypotheses": [payload]},
            evidence,
            case_id="Patient_03674",
            evidence_snapshot_id="ES_TEST",
            thinking_snapshots=[snapshot.to_dict()],
        )
        sources = [item["generator_source"] for item in context["pattern_hypotheses"]]
        self.assertEqual(sources.count("thinking_structured"), 1)
        self.assertEqual(sources.count("deterministic_relation"), 1)

    def test_family_relation_can_recall_entity_without_disease_name(self):
        evidence = EvidenceBundle(
            [
                Observation("cyanosis", "patient_reported_observation", confidence=0.9, information_value=0.86),
                Observation("hypoxemia", "laboratory_result", confidence=0.9, information_value=0.9),
                Observation(
                    "pulmonary_vascular_abnormality",
                    "imaging_result",
                    confidence=0.86,
                    information_value=0.88,
                ),
            ]
        )
        snapshot = ThinkingSnapshot.from_thinking(
            {
                "clinical_pattern_proposals": [
                    {
                        "pattern_hypothesis_id": "PH_PVASC_001",
                        "pattern_name": "pulmonary_vascular_shunt_or_malformation_pattern",
                        "pattern_type": "vascular_shunt",
                        "suggested_family": "pulmonary_vascular_shunt",
                        "evidence_bindings": [
                            {
                                "evidence_id": "cyanosis",
                                "role": "support",
                                "expected_polarity": "positive",
                                "relation_slot": "organ_manifestation",
                            },
                            {
                                "evidence_id": "hypoxemia",
                                "role": "support",
                                "expected_polarity": "positive",
                                "relation_slot": "imaging_or_objective_finding",
                            },
                            {
                                "evidence_id": "pulmonary_vascular_abnormality",
                                "role": "support",
                                "expected_polarity": "positive",
                                "relation_slot": "support",
                            },
                        ],
                        "relations": [
                            {
                                "type": "anatomical_consistency",
                                "from_evidence_ref": "pulmonary_vascular_abnormality",
                                "to_evidence_ref": "hypoxemia",
                            }
                        ],
                        "suggested_diseases": [],
                    }
                ]
            },
            case_id="Patient_03998",
            patient_id="Patient_03998",
            phase="examination",
            evidence_snapshot_id="ES_TEST",
        )
        context = self.engine.build_pattern_recall_context(
            {},
            evidence,
            case_id="Patient_03998",
            evidence_snapshot_id="ES_TEST",
            thinking_snapshots=[snapshot.to_dict()],
        )
        signals = context["pattern_recall_signals"]
        self.assertTrue(any(item["entity_id"] == PAVM_ENTITY_ID for item in signals))
        self.assertEqual(context["pattern_protected_candidate_recall"], [])


if __name__ == "__main__":
    unittest.main()
