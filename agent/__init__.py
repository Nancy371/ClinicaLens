"""Agent package public API exports.

Lazy loading avoids importing heavy modules before the package entrypoint runs.
"""

__all__ = [
    "MyDoctorAgent",
    "DefectDetector",
    "PolicyStore",
    "CandidatePolicyStore",
    "RuleGeneralizer",
    "FailureAttribution",
    "PromotionDecision",
    "ExamStrategyAgent",
    "InquiryStrategyAgent",
    "QualityAgent",
    "TreatmentStrategyAgent",
    "StructuralDiagnosisAgent",
    "EvidenceDiagnosisEngine",
    "ClinicalEvidenceNormalizer",
    "EvidenceAgent",
    "EvidenceGraph",
    "EvidenceBundle",
    "Observation",
    "ClinicalPattern",
    "ClinicalPatternCompiler",
    "ClinicalPatternHypothesis",
    "ThinkingSnapshot",
    "PatternProposalAdapter",
    "PatternProposalCompiler",
    "PatternHypothesisVerifier",
    "PatternRecallSignal",
    "PatternVerificationResult",
    "CandidateGenerator",
    "CandidatePool",
    "CandidateSource",
    "DiagnosisDecisionEngine",
    "DiagnosticKnowledgeBase",
    "DiagnosisDecision",
    "DiseaseEntityRegistry",
    "DiseaseEntity",
    "DiagnosticPatternEvaluator",
    "CaseBoard",
    "CaseBoardEvent",
    "CaseBoardPermissionError",
    "ConsultationEvidencePipeline",
    "EvidenceClaim",
    "EvidenceClaimGenerator",
    "EvidenceConflictAuditor",
    "EvidenceDefinition",
    "EvidenceDefinitionRegistry",
    "EvidenceHypothesis",
    "EvidenceHypothesisGenerator",
    "EvidencePatternCompiler",
    "EvidencePatternMatch",
    "EvidenceQueryPlanner",
    "EvidenceQueryTask",
    "PatternCompiler",
    "StaleJudgeDecisionError",
    "DeterministicEvidenceVerifier",
    "ExamResultIntentBinding",
    "TargetedExamParseResult",
    "TargetedExamResultParser",
    "TargetedEvidenceVerifier",
    "VerificationResult",
    "evidence_snapshot_hash",
    "ExamResolver",
    "ExamResolution",
    "DiagnosisEligibilityGate",
    "EligibilityResult",
    "PRIMARY_ELIGIBLE",
    "DEFERRED",
    "DIFFERENTIAL_ONLY",
    "EXCLUDED",
    "DiagnosisJudge",
    "JudgeDecision",
    "JudgeCandidateReview",
    "DiagnosisSubmitter",
    "OpenWorldDiagnosisResolver",
    "DiagnosisResolution",
    "DiagnosisCritic",
    "CriticDecision",
    "TreatmentSafetyGate",
    "DiagnosticLearningStore",
    "DoctorAgentMemory",
    "WorkingCaseMemory",
    "MemoryItem",
    "MemoryConfig",
    "HybridRAGRetriever",
    "HybridRAGConfig",
    "RagChunk",
    "MechanismReasoner",
    "MechanismHypothesis",
    "RetrievalView",
    "RootCauseArbiter",
    "RootCauseArbitrationResult",
    "ExternalMedicalKnowledgeRetriever",
    "ExternalMedicalResult",
    "ShadowReplay",
    "DiagnosticReplay",
    "heuristic_plan_score",
]


def __getattr__(name: str):
    """Import public symbols lazily on first access."""
    if name == "MyDoctorAgent":
        from .agent import MyDoctorAgent
        return MyDoctorAgent
    if name == "DefectDetector":
        from .critic import DefectDetector
        return DefectDetector
    if name == "PolicyStore":
        from .policy_store import PolicyStore
        return PolicyStore
    if name in (
        "CandidatePolicyStore",
        "RuleGeneralizer",
        "FailureAttribution",
        "PromotionDecision",
    ):
        from . import candidate_policy_store as _candidate_policy_store
        return getattr(_candidate_policy_store, name)
    if name == "ExamStrategyAgent":
        from .exam_strategy import ExamStrategyAgent
        return ExamStrategyAgent
    if name == "InquiryStrategyAgent":
        from .inquiry_strategy import InquiryStrategyAgent
        return InquiryStrategyAgent
    if name == "QualityAgent":
        from .qc import QualityAgent
        return QualityAgent
    if name == "TreatmentStrategyAgent":
        from .treatment_strategy import TreatmentStrategyAgent
        return TreatmentStrategyAgent
    if name == "StructuralDiagnosisAgent":
        from .structural_diagnosis import StructuralDiagnosisAgent
        return StructuralDiagnosisAgent
    if name == "EvidenceDiagnosisEngine":
        from .evidence_engine import EvidenceDiagnosisEngine
        return EvidenceDiagnosisEngine
    if name in ("ClinicalEvidenceNormalizer", "EvidenceAgent", "EvidenceGraph", "EvidenceBundle", "Observation"):
        from . import clinical_evidence as _clinical_evidence
        return getattr(_clinical_evidence, name)
    if name in ("ClinicalPattern", "ClinicalPatternCompiler"):
        from . import clinical_pattern_compiler as _clinical_pattern_compiler
        return getattr(_clinical_pattern_compiler, name)
    if name in (
        "ClinicalPatternHypothesis",
        "ThinkingSnapshot",
        "PatternProposalAdapter",
        "PatternProposalCompiler",
        "PatternHypothesisVerifier",
        "PatternRecallSignal",
        "PatternVerificationResult",
    ):
        from . import pattern_hypothesis as _pattern_hypothesis
        return getattr(_pattern_hypothesis, name)
    if name in ("CandidateGenerator", "CandidatePool", "CandidateSource"):
        from . import candidate_generator as _candidate_generator
        return getattr(_candidate_generator, name)
    if name in ("DiagnosisDecisionEngine", "DiagnosticKnowledgeBase", "DiagnosisDecision"):
        from . import diagnosis_engine as _diagnosis_engine
        return getattr(_diagnosis_engine, name)
    if name in ("DiseaseEntityRegistry", "DiseaseEntity"):
        from . import disease_entity as _disease_entity
        return getattr(_disease_entity, name)
    if name == "DiagnosticPatternEvaluator":
        from .diagnostic_patterns import DiagnosticPatternEvaluator
        return DiagnosticPatternEvaluator
    if name in ("EvidenceDefinition", "EvidenceDefinitionRegistry"):
        from . import evidence_registry as _evidence_registry
        return getattr(_evidence_registry, name)
    if name in ("EvidenceHypothesis", "EvidenceHypothesisGenerator"):
        from . import evidence_hypothesis as _evidence_hypothesis
        return getattr(_evidence_hypothesis, name)
    if name in ("EvidenceQueryPlanner", "EvidenceQueryTask"):
        from . import evidence_query_planner as _evidence_query_planner
        return getattr(_evidence_query_planner, name)
    if name in ("DeterministicEvidenceVerifier", "VerificationResult"):
        from . import targeted_evidence_verifier as _targeted_evidence_verifier
        return getattr(_targeted_evidence_verifier, name)
    if name in (
        "ExamResultIntentBinding",
        "TargetedExamParseResult",
        "TargetedExamResultParser",
    ):
        from . import targeted_exam_result_parser as _targeted_exam_result_parser
        return getattr(_targeted_exam_result_parser, name)
    if name in ("EvidencePatternCompiler", "EvidencePatternMatch"):
        from . import evidence_pattern_compiler as _evidence_pattern_compiler
        return getattr(_evidence_pattern_compiler, name)
    if name == "EvidenceConflictAuditor":
        from .evidence_conflict_auditor import EvidenceConflictAuditor
        return EvidenceConflictAuditor
    if name in (
        "CaseBoard",
        "CaseBoardEvent",
        "CaseBoardPermissionError",
        "ConsultationEvidencePipeline",
        "EvidenceClaim",
        "EvidenceClaimGenerator",
        "PatternCompiler",
        "StaleJudgeDecisionError",
        "TargetedEvidenceVerifier",
        "evidence_snapshot_hash",
    ):
        from . import case_board as _case_board
        return getattr(_case_board, name)
    if name in ("ExamResolver", "ExamResolution"):
        from . import exam_resolver as _exam_resolver
        return getattr(_exam_resolver, name)
    if name in (
        "DiagnosisEligibilityGate",
        "EligibilityResult",
        "PRIMARY_ELIGIBLE",
        "DEFERRED",
        "DIFFERENTIAL_ONLY",
        "EXCLUDED",
    ):
        from . import diagnosis_eligibility as _diagnosis_eligibility
        return getattr(_diagnosis_eligibility, name)
    if name in ("DiagnosisJudge", "JudgeDecision", "JudgeCandidateReview", "DiagnosisSubmitter"):
        from . import diagnosis_judge as _diagnosis_judge
        return getattr(_diagnosis_judge, name)
    if name in ("OpenWorldDiagnosisResolver", "DiagnosisResolution"):
        from . import diagnosis_resolver as _diagnosis_resolver
        return getattr(_diagnosis_resolver, name)
    if name in ("DiagnosisCritic", "CriticDecision"):
        from . import diagnosis_critic as _diagnosis_critic
        return getattr(_diagnosis_critic, name)
    if name == "TreatmentSafetyGate":
        from .treatment_safety import TreatmentSafetyGate
        return TreatmentSafetyGate
    if name == "DiagnosticLearningStore":
        from .diagnostic_learning import DiagnosticLearningStore
        return DiagnosticLearningStore
    if name in ("DoctorAgentMemory", "WorkingCaseMemory", "MemoryItem", "MemoryConfig"):
        from . import memory_system as _memory_system
        return getattr(_memory_system, name)
    if name in ("HybridRAGRetriever", "HybridRAGConfig", "RagChunk"):
        from . import rag_retriever as _rag_retriever
        return getattr(_rag_retriever, name)
    if name in ("MechanismReasoner", "MechanismHypothesis", "RetrievalView"):
        from . import mechanism_reasoner as _mechanism_reasoner
        return getattr(_mechanism_reasoner, name)
    if name in ("RootCauseArbiter", "RootCauseArbitrationResult"):
        from . import root_cause_arbitration as _root_cause_arbitration
        return getattr(_root_cause_arbitration, name)
    if name in ("ExternalMedicalKnowledgeRetriever", "ExternalMedicalResult"):
        from . import medical_retrieval as _medical_retrieval
        return getattr(_medical_retrieval, name)
    if name in ("ShadowReplay", "DiagnosticReplay", "heuristic_plan_score"):
        from . import replay as _replay
        return getattr(_replay, name)
    raise AttributeError(f"module 'agent' has no attribute {name!r}")
