import unittest

from agent.candidate_generator import CandidatePool
from agent.case_board import ConsultationEvidencePipeline
from agent.clinical_evidence import EvidenceBundle, Observation
from agent.evidence_hypothesis import EvidenceHypothesisGenerator
from agent.evidence_pattern_compiler import EvidencePatternCompiler
from agent.evidence_query_planner import EvidenceQueryPlanner
from agent.evidence_registry import EvidenceDefinitionRegistry
from agent.targeted_evidence_verifier import (
    UNSUPPORTED,
    VERIFIED_NEGATIVE,
    DeterministicEvidenceVerifier,
)


class LlmGuidedEvidencePipelineTests(unittest.TestCase):
    def test_hypothesis_uses_registry_not_llm_search_terms(self):
        registry = EvidenceDefinitionRegistry("data/ref_data")
        generator = EvidenceHypothesisGenerator(registry)
        planner = EvidenceQueryPlanner(registry)

        hypotheses = generator.generate(
            {"diagnosis_candidates": [{"name": "\u767d\u8840\u75c5"}]}
        )
        blast = next(
            item for item in hypotheses if item.target_evidence_id == "blast_present"
        )

        self.assertEqual(blast.to_dict()["search_terms"], [])
        tasks = planner.plan_all([blast])
        self.assertIn("\u539f\u59cb\u7ec6\u80de", tasks[0].aliases)
        self.assertEqual(tasks[0].strategy, "targeted_span_search")

    def test_verifier_distinguishes_unsupported_from_verified_negative(self):
        registry = EvidenceDefinitionRegistry("data/ref_data")
        generator = EvidenceHypothesisGenerator(registry)
        planner = EvidenceQueryPlanner(registry)
        verifier = DeterministicEvidenceVerifier(registry)
        hypothesis = next(
            item
            for item in generator.generate(
                {"diagnosis_candidates": [{"name": "\u767d\u8840\u75c5"}]}
            )
            if item.target_evidence_id == "blast_present"
        )
        task = planner.plan_all([hypothesis])[0]

        unsupported = verifier.verify(
            task,
            EvidenceBundle([Observation("fever", "history", raw_text="\u53d1\u70ed")]),
        )
        self.assertEqual(unsupported.verification_status, UNSUPPORTED)

        negative = verifier.verify(
            task,
            EvidenceBundle(
                [
                    Observation(
                        "field:smear",
                        "\u5916\u5468\u8840\u6d82\u7247",
                        raw_text="\u5916\u5468\u8840\u6d82\u7247\u672a\u89c1\u539f\u59cb\u7ec6\u80de",
                        source_text="\u5916\u5468\u8840\u6d82\u7247\u672a\u89c1\u539f\u59cb\u7ec6\u80de",
                    )
                ]
            ),
        )
        self.assertEqual(negative.verification_status, VERIFIED_NEGATIVE)
        observations = verifier.observations_from_results(
            [negative],
            EvidenceBundle([]),
        )
        self.assertEqual(observations[0].finding, "blast_present")
        self.assertEqual(observations[0].polarity, "negative")

    def test_missing_source_span_does_not_enter_observed_evidence(self):
        registry = EvidenceDefinitionRegistry("data/ref_data")
        generator = EvidenceHypothesisGenerator(registry)
        planner = EvidenceQueryPlanner(registry)
        verifier = DeterministicEvidenceVerifier(registry)
        task = planner.plan_all(
            generator.generate({"diagnosis_candidates": [{"name": "\u767d\u8840\u75c5"}]})
        )[0]
        result = verifier.verify(
            task,
            EvidenceBundle([Observation("blast_present", "\u5916\u5468\u8840\u6d82\u7247")]),
        )

        self.assertNotEqual(result.verification_status, "verified_positive")
        self.assertEqual(
            verifier.observations_from_results([result], EvidenceBundle([])),
            [],
        )

    def test_pattern_compiler_min_count_uses_verified_facts_only(self):
        compiler = EvidencePatternCompiler(ref_dir="data/ref_data")
        evidence = EvidenceBundle(
            [
                Observation("hemoglobin_low", "CBC", raw_text="Hb low"),
                Observation("platelet_low", "CBC", raw_text="PLT low"),
                Observation("blast_present", "reasoning_inference", raw_text="LLM guessed blast"),
            ]
        )

        derived = compiler.compile([], evidence)
        findings = [item.finding for item in derived]

        self.assertIn("multilineage_cytopenia", findings)
        self.assertNotIn("acute_leukemia_pattern", findings)

    def test_pipeline_keeps_hypothesis_verification_and_fact_layers_separate(self):
        pool = CandidatePool()
        pool.add("\u767d\u8840\u75c5", "\u767d\u8840\u75c5", "llm", prior=0.8)
        evidence = EvidenceBundle(
            [
                Observation(
                    "field:smear",
                    "\u5916\u5468\u8840\u6d82\u7247",
                    raw_text="\u5916\u5468\u8840\u53ef\u89c1\u539f\u59cb\u7ec6\u80de\u7ea618%",
                    source_text="\u5916\u5468\u8840\u53ef\u89c1\u539f\u59cb\u7ec6\u80de\u7ea618%",
                ),
                Observation("hemoglobin_low", "CBC", raw_text="Hb low"),
                Observation("platelet_low", "CBC", raw_text="PLT low"),
            ]
        )

        pipeline = ConsultationEvidencePipeline()
        board, enhanced = pipeline.run(
            evidence,
            llm_result={"diagnosis_candidates": [{"name": "\u767d\u8840\u75c5"}]},
            candidate_pool=pool,
        )
        view = board.view()

        self.assertTrue(view["evidence_hypotheses"])
        self.assertTrue(view["verification_results"])
        self.assertIn("blast_present", enhanced.findings("positive"))
        self.assertIn("acute_leukemia_pattern", enhanced.findings("positive"))
        self.assertTrue(view["evidence_store"]["observed_evidence"])
        self.assertTrue(view["evidence_store"]["derived_evidence"])
        self.assertEqual(pipeline.last_audit["unverified_evidence_leakage"], 0)

    def test_unresolved_claim_creates_protection_audit_but_not_fact(self):
        pool = CandidatePool()
        pool.add("\u767d\u8840\u75c5", "\u767d\u8840\u75c5", "llm", prior=0.8)
        pipeline = ConsultationEvidencePipeline()

        board, enhanced = pipeline.run(
            EvidenceBundle([Observation("fever", "history", raw_text="\u53d1\u70ed")]),
            llm_result={"diagnosis_candidates": [{"name": "\u767d\u8840\u75c5"}]},
            candidate_pool=pool,
        )

        self.assertNotIn("blast_present", enhanced.findings("positive"))
        self.assertTrue(board.view()["candidate_protections"])
        self.assertEqual(pipeline.last_audit["unsupported_claim_admission_count"], 0)


if __name__ == "__main__":
    unittest.main()
