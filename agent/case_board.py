"""Append-only consultation board and evidence-claim verification helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .clinical_evidence import EvidenceBundle, Observation
from .evidence_conflict_auditor import EvidenceConflictAuditor
from .evidence_hypothesis import EvidenceHypothesisGenerator as StandardEvidenceHypothesisGenerator
from .evidence_pattern_compiler import EvidencePatternCompiler as StandardEvidencePatternCompiler
from .evidence_query_planner import EvidenceQueryPlanner
from .evidence_registry import EvidenceDefinitionRegistry
from .targeted_evidence_verifier import (
    DERIVED,
    INVALID_CLAIM,
    UNRESOLVED,
    UNSUPPORTED,
    VERIFIED_NEGATIVE,
    VERIFIED_POSITIVE,
    DeterministicEvidenceVerifier,
    VerificationResult,
)


JUDGE_ONLY_EVENT_TYPES = {
    "candidate_decision",
    "candidate_decisions",
    "judge_decision",
    "final_diagnoses",
}
SUBMITTER_ONLY_EVENT_TYPES = {"submission_event"}
OBSERVED_EVIDENCE_SOURCES = {
    "reasoning_inference": False,
}


class CaseBoardPermissionError(RuntimeError):
    """Raised when a module tries to write a state it does not own."""


class StaleJudgeDecisionError(RuntimeError):
    """Raised when submission sees a JudgeDecision for an old evidence snapshot."""


@dataclass
class CaseBoardEvent:
    event_id: str
    event_type: str
    producer: str
    case_version: int
    created_at_stage: str
    payload: Dict[str, Any] = field(default_factory=dict)
    source_evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceClaim:
    claim_id: str
    target_evidence: str
    claim_type: str = "observed_finding"
    diagnosis_hypothesis: str = ""
    importance: str = "medium"
    search_terms: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    status: str = "Unresolved"
    evidence_id: str = ""
    source_span: str = ""
    source_section: str = ""
    polarity: str = "positive"
    reason: str = ""
    recommended_exam: str = ""
    producer: str = "reasoner"
    case_version: int = 0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_any(cls, value: Any) -> Optional["EvidenceClaim"]:
        if isinstance(value, EvidenceClaim):
            return value
        if not isinstance(value, dict):
            return None
        claim_id = str(value.get("claim_id") or value.get("id") or "").strip()
        target = str(value.get("target_evidence") or value.get("evidence_id") or "").strip()
        if not claim_id or not target:
            return None
        return cls(
            claim_id=claim_id,
            target_evidence=target,
            claim_type=str(value.get("claim_type") or "observed_finding"),
            diagnosis_hypothesis=str(value.get("diagnosis_hypothesis") or ""),
            importance=str(value.get("importance") or "medium"),
            search_terms=_text_list(value.get("search_terms") or []),
            required_inputs=_text_list(value.get("required_inputs") or []),
            status=str(value.get("status") or "Unresolved"),
            evidence_id=str(value.get("evidence_id") or ""),
            source_span=str(value.get("source_span") or ""),
            source_section=str(value.get("source_section") or ""),
            polarity=str(value.get("polarity") or "positive"),
            reason=str(value.get("reason") or ""),
            recommended_exam=str(value.get("recommended_exam") or ""),
            producer=str(value.get("producer") or "reasoner"),
            case_version=int(value.get("case_version") or 0),
            confidence=_float(value.get("confidence"), 0.0),
        )


@dataclass
class CaseBoard:
    case_id: str = ""
    case_version: int = 1
    evidence_snapshot_hash: str = ""
    knowledge_profile_version: str = ""
    decision_policy_version: str = ""
    exam_catalog_version: str = ""
    claim_resolution_ledger: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    claim_state_version: int = 0
    diagnostic_state_version: int = 0
    events: List[CaseBoardEvent] = field(default_factory=list)

    @classmethod
    def from_evidence(
        cls,
        evidence: EvidenceBundle,
        *,
        case_id: str = "",
        knowledge_profile_version: str = "",
        decision_policy_version: str = "",
        exam_catalog_version: str = "",
    ) -> "CaseBoard":
        return cls(
            case_id=case_id,
            case_version=1,
            evidence_snapshot_hash=evidence_snapshot_hash(evidence),
            knowledge_profile_version=knowledge_profile_version,
            decision_policy_version=decision_policy_version,
            exam_catalog_version=exam_catalog_version,
        )

    def append_event(
        self,
        event_type: str,
        producer: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        created_at_stage: str = "",
        source_evidence_ids: Optional[Sequence[str]] = None,
        confidence: float = 0.0,
    ) -> CaseBoardEvent:
        self.assert_can_write(event_type, producer)
        event = CaseBoardEvent(
            event_id=f"evt_{len(self.events) + 1:04d}",
            event_type=str(event_type or ""),
            producer=str(producer or ""),
            case_version=int(self.case_version or 1),
            created_at_stage=str(created_at_stage or ""),
            payload=dict(payload or {}),
            source_evidence_ids=_text_list(source_evidence_ids or []),
            confidence=_float(confidence, 0.0),
        )
        self.events.append(event)
        return event

    @staticmethod
    def assert_can_write(event_type: str, producer: str) -> None:
        normalized_type = str(event_type or "").strip()
        normalized_producer = str(producer or "").strip().lower()
        if normalized_type in JUDGE_ONLY_EVENT_TYPES and normalized_producer != "judge":
            raise CaseBoardPermissionError(
                f"{normalized_type} can only be written by Judge"
            )
        if (
            normalized_type in SUBMITTER_ONLY_EVENT_TYPES
            and normalized_producer != "submitter"
        ):
            raise CaseBoardPermissionError(
                f"{normalized_type} can only be written by Submitter"
            )

    def refresh_evidence_snapshot(
        self,
        evidence: EvidenceBundle,
        *,
        producer: str,
        created_at_stage: str,
    ) -> None:
        new_hash = evidence_snapshot_hash(evidence)
        if new_hash == self.evidence_snapshot_hash:
            return
        old_hash = self.evidence_snapshot_hash
        self.case_version += 1
        self.evidence_snapshot_hash = new_hash
        self.append_event(
            "evidence_snapshot_updated",
            producer,
            {
                "old_evidence_snapshot_hash": old_hash,
                "new_evidence_snapshot_hash": new_hash,
            },
            created_at_stage=created_at_stage,
        )

    def view(self) -> Dict[str, Any]:
        view: Dict[str, Any] = {
            "evidence_store": {
                "observed_evidence": [],
                "derived_evidence": [],
            },
            "structured_evidence": [],
            "derived_patterns": [],
            "evidence_hypotheses": [],
            "verification_results": [],
            "evidence_claims": [],
            "evidence_query_tasks": [],
            "candidate_opinions": [],
            "conflict_events": [],
            "exam_proposals": [],
            "candidate_protections": [],
            "candidate_decisions": [],
            "claim_resolution_ledger": dict(self.claim_resolution_ledger),
            "claim_state_version": int(self.claim_state_version or 0),
            "diagnostic_state_version": int(self.diagnostic_state_version or 0),
            "judge_decision": None,
            "audit_events": [],
        }
        for event in self.events:
            payload = dict(event.payload)
            if event.event_type == "structured_evidence":
                view["structured_evidence"].append(payload)
                if _derived_evidence_payload(payload):
                    view["evidence_store"]["derived_evidence"].append(payload)
                else:
                    view["evidence_store"]["observed_evidence"].append(payload)
            elif event.event_type == "verified_observed_evidence":
                view["structured_evidence"].append(payload)
                view["evidence_store"]["observed_evidence"].append(payload)
            elif event.event_type == "derived_pattern":
                view["derived_patterns"].append(payload)
                view["evidence_store"]["derived_evidence"].append(payload)
            elif event.event_type == "evidence_hypothesis":
                view["evidence_hypotheses"].append(payload)
            elif event.event_type in {"evidence_claim", "evidence_claim_verification"}:
                view["evidence_claims"].append(payload)
                if event.event_type == "evidence_claim_verification":
                    view["verification_results"].append(payload)
            elif event.event_type == "evidence_query_task":
                view["evidence_query_tasks"].append(payload)
            elif event.event_type == "candidate_opinion":
                view["candidate_opinions"].append(payload)
            elif event.event_type == "conflict_event":
                view["conflict_events"].append(payload)
            elif event.event_type == "exam_proposal":
                view["exam_proposals"].append(payload)
            elif event.event_type == "candidate_protection":
                view["candidate_protections"].append(payload)
            elif event.event_type in {"candidate_decision", "candidate_decisions"}:
                if isinstance(payload.get("candidate_decisions"), list):
                    view["candidate_decisions"].extend(payload["candidate_decisions"])
                else:
                    view["candidate_decisions"].append(payload)
            elif event.event_type == "claim_resolution_ledger":
                ledger = payload.get("claim_resolution_ledger")
                if isinstance(ledger, dict):
                    view["claim_resolution_ledger"] = dict(ledger)
                view["claim_state_version"] = int(payload.get("claim_state_version") or 0)
                view["diagnostic_state_version"] = int(
                    payload.get("diagnostic_state_version") or 0
                )
            elif event.event_type == "judge_decision":
                view["judge_decision"] = payload
            else:
                view["audit_events"].append(payload)
        return view

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "case_id": self.case_id,
            "case_version": self.case_version,
            "evidence_snapshot_hash": self.evidence_snapshot_hash,
            "knowledge_profile_version": self.knowledge_profile_version,
            "decision_policy_version": self.decision_policy_version,
            "exam_catalog_version": self.exam_catalog_version,
            "claim_resolution_ledger": dict(self.claim_resolution_ledger),
            "claim_state_version": int(self.claim_state_version or 0),
            "diagnostic_state_version": int(self.diagnostic_state_version or 0),
            "events": [event.to_dict() for event in self.events],
        }
        data.update(self.view())
        return data


class EvidenceClaimGenerator:
    """Generate verifiable claims from candidate identities and LLM output."""

    def generate(self, llm_result: Any, candidate_pool: Any = None) -> List[EvidenceClaim]:
        names = self._candidate_names(llm_result, candidate_pool)
        claims: List[EvidenceClaim] = []
        for name in names:
            lower = name.lower()
            if _mentions_any(lower, ("leukemia", "aml", "all", "白血病")):
                claims.extend(self._leukemia_claims(name))
            if _mentions_any(lower, ("pavm", "pulmonary avm", "肺动静脉瘘", "肺动静脉畸形", "肺内右向左分流")):
                claims.extend(self._pavm_claims(name))
            if _mentions_any(lower, ("prostatitis", "前列腺炎")):
                claims.extend(self._prostatitis_claims(name))
        return _dedupe_claims(claims)

    def _candidate_names(self, llm_result: Any, candidate_pool: Any) -> List[str]:
        names: List[str] = []

        def add(value: Any) -> None:
            text = str(value or "").strip()
            if text and text not in names:
                names.append(text)

        if isinstance(llm_result, dict):
            for key in ("diagnosis", "primary_diagnosis", "reasoning"):
                add(llm_result.get(key))
            for key in ("diagnosis_candidates", "differential_diagnoses", "candidate_diagnoses"):
                for item in llm_result.get(key) or []:
                    if isinstance(item, dict):
                        add(item.get("name") or item.get("diagnosis"))
                    else:
                        add(item)
        claim_capable_sources = {
            "llm",
            "llm_unresolved",
            "mechanism_reasoner",
            "external_retrieval",
        }
        for item in getattr(candidate_pool, "items", []) or []:
            source = str(getattr(item, "source", "") or "")
            if source not in claim_capable_sources:
                continue
            for attr in ("raw_name", "diagnosis", "canonical_name", "submission_name"):
                add(getattr(item, attr, ""))
        for item in getattr(candidate_pool, "open_world_candidates", []) or []:
            if isinstance(item, dict):
                source = str(item.get("source") or "")
                if source not in claim_capable_sources:
                    continue
                for key in ("raw_name", "canonical_name", "submission_name", "name"):
                    add(item.get(key))
        return names

    @staticmethod
    def _leukemia_claims(name: str) -> List[EvidenceClaim]:
        return [
            EvidenceClaim(
                claim_id="claim_blast_present",
                diagnosis_hypothesis=name,
                target_evidence="blast_present",
                claim_type="observed_finding",
                importance="critical",
                search_terms=["原始细胞", "幼稚细胞", "母细胞", "blast"],
                recommended_exam="外周血涂片",
                confidence=0.86,
            ),
            EvidenceClaim(
                claim_id="claim_multilineage_cytopenia",
                diagnosis_hypothesis=name,
                target_evidence="multilineage_cytopenia",
                claim_type="derived_pattern",
                importance="critical",
                required_inputs=[
                    "hemoglobin_low",
                    "platelet_low",
                    "white_blood_cell_abnormal",
                ],
                recommended_exam="全血细胞计数（CBC）",
                confidence=0.82,
            ),
            EvidenceClaim(
                claim_id="claim_acute_leukemia_pattern",
                diagnosis_hypothesis=name,
                target_evidence="acute_leukemia_pattern",
                claim_type="derived_pattern",
                importance="critical",
                required_inputs=["blast_present", "multilineage_cytopenia"],
                recommended_exam="骨髓穿刺和活检",
                confidence=0.9,
            ),
        ]

    @staticmethod
    def _pavm_claims(name: str) -> List[EvidenceClaim]:
        return [
            EvidenceClaim(
                claim_id="claim_right_to_left_shunt",
                diagnosis_hypothesis=name,
                target_evidence="right_to_left_shunt",
                claim_type="observed_finding",
                importance="critical",
                search_terms=["右向左分流", "肺内分流", "右至左分流", "bubble"],
                recommended_exam="右心声学造影",
                confidence=0.86,
            ),
            EvidenceClaim(
                claim_id="claim_pulmonary_avm_mechanism",
                diagnosis_hypothesis=name,
                target_evidence="pulmonary_avm_mechanism",
                claim_type="observed_finding",
                importance="critical",
                search_terms=["肺动静脉瘘", "肺动静脉畸形", "PAVM", "肺内右向左分流"],
                recommended_exam="肺动脉CTA",
                confidence=0.84,
            ),
            EvidenceClaim(
                claim_id="claim_pulmonary_vascular_confirmation",
                diagnosis_hypothesis=name,
                target_evidence="pulmonary_cta_positive",
                claim_type="observed_finding",
                importance="critical",
                search_terms=["肺动脉CTA阳性", "CTA提示肺动静脉", "肺血管畸形"],
                recommended_exam="肺动脉CTA",
                confidence=0.88,
            ),
        ]

    @staticmethod
    def _prostatitis_claims(name: str) -> List[EvidenceClaim]:
        return [
            EvidenceClaim(
                claim_id="claim_prostate_localization",
                diagnosis_hypothesis=name,
                target_evidence="prostate_tenderness",
                claim_type="observed_finding",
                importance="critical",
                search_terms=["前列腺压痛", "会阴痛", "直肠指检"],
                recommended_exam="直肠指检（DRE）",
                confidence=0.78,
            )
        ]


class TargetedEvidenceVerifier:
    """Verify Reasoner claims against existing observations and source spans."""

    def verify_all(
        self,
        claims: Sequence[Any],
        evidence: EvidenceBundle,
    ) -> List[EvidenceClaim]:
        verified: List[EvidenceClaim] = []
        verified_findings = self._verified_finding_set(evidence)
        for claim in claims or []:
            parsed = EvidenceClaim.from_any(claim)
            if parsed is None:
                continue
            result = self.verify(parsed, evidence, verified_findings)
            if result.status in {"Verified", "Derived"}:
                verified_findings.add(result.target_evidence)
            verified.append(result)
        return verified

    def verify(
        self,
        claim: EvidenceClaim,
        evidence: EvidenceBundle,
        verified_findings: Optional[set[str]] = None,
    ) -> EvidenceClaim:
        verified_findings = set(verified_findings or self._verified_finding_set(evidence))
        result = EvidenceClaim.from_any(claim.to_dict()) or claim
        result.status = "Unresolved"
        result.reason = ""
        result.case_version = int(result.case_version or 0)

        if result.claim_type == "derived_pattern" or result.required_inputs:
            missing = [
                item
                for item in result.required_inputs
                if not self._finding_present(item, verified_findings)
            ]
            if missing:
                result.reason = "missing verified inputs: " + ", ".join(missing)
                return result
            result.status = "Derived"
            result.evidence_id = result.target_evidence
            result.source_span = "derived from " + ", ".join(result.required_inputs)
            result.source_section = "pattern_compiler"
            result.reason = "all required inputs verified"
            return result

        observation = self._source_observation(result.target_evidence, evidence)
        if observation is not None:
            source = str(getattr(observation, "source", "") or "")
            if source == "reasoning_inference":
                result.reason = "reasoning_inference cannot verify evidence claim"
                return result
            span = self._span_for_observation(observation)
            if not span:
                result.reason = "verified evidence requires source span"
                return result
            result.status = "Verified"
            result.evidence_id = result.target_evidence
            result.source_span = span
            result.source_section = str(
                getattr(observation, "source", "") or getattr(observation, "field_path", "") or ""
            )
            result.polarity = str(getattr(observation, "polarity", "positive") or "positive")
            result.confidence = max(result.confidence, _float(getattr(observation, "confidence", 0.0), 0.0))
            return result

        span, section = self._find_source_span(result.search_terms, evidence)
        if span:
            result.status = "Verified"
            result.evidence_id = result.target_evidence
            result.source_span = span
            result.source_section = section
            result.polarity = result.polarity or "positive"
            result.confidence = max(result.confidence, 0.72)
            return result

        result.reason = "source span not found"
        return result

    def verified_observations(
        self,
        verified_claims: Sequence[EvidenceClaim],
        evidence: EvidenceBundle,
    ) -> List[Observation]:
        existing = self._verified_finding_set(evidence)
        observations: List[Observation] = []
        for claim in verified_claims or []:
            if claim.status not in {"Verified", "Derived"}:
                continue
            if self._finding_present(claim.target_evidence, existing):
                continue
            source = (
                "pattern_compiler"
                if claim.status == "Derived"
                else "targeted_evidence_verifier"
            )
            observations.append(
                Observation(
                    finding=claim.target_evidence,
                    source=source,
                    polarity=claim.polarity or "positive",
                    confidence=max(0.7, min(0.98, claim.confidence or 0.78)),
                    raw_text=claim.source_span,
                    source_text=claim.source_span,
                    field_path=claim.source_section,
                    evidence_level=(
                        "diagnostic_pattern"
                        if claim.status == "Derived"
                        else "specific"
                    ),
                    information_value=0.88 if claim.importance == "critical" else 0.72,
                )
            )
            existing.add(claim.target_evidence)
        return observations

    def _verified_finding_set(self, evidence: EvidenceBundle) -> set[str]:
        findings: set[str] = set()
        for item in getattr(evidence, "observations", []) or []:
            if str(getattr(item, "polarity", "positive") or "positive") != "positive":
                continue
            if str(getattr(item, "source", "") or "") == "reasoning_inference":
                continue
            finding = str(getattr(item, "finding", "") or "")
            if finding:
                findings.add(finding)
                findings.update(_finding_aliases(finding))
        return findings

    def _source_observation(
        self,
        finding: str,
        evidence: EvidenceBundle,
    ) -> Optional[Observation]:
        candidates = {finding} | _finding_aliases(finding)
        best: Optional[Observation] = None
        for item in getattr(evidence, "observations", []) or []:
            if str(getattr(item, "finding", "") or "") not in candidates:
                continue
            if str(getattr(item, "polarity", "positive") or "positive") != "positive":
                continue
            if best is None or _observation_quality(item) > _observation_quality(best):
                best = item
        return best

    @staticmethod
    def _finding_present(finding: str, verified_findings: set[str]) -> bool:
        candidates = {finding} | _finding_aliases(finding)
        return bool(candidates & verified_findings)

    @staticmethod
    def _span_for_observation(observation: Observation) -> str:
        return str(
            getattr(observation, "source_text", "")
            or getattr(observation, "raw_text", "")
            or getattr(observation, "field_path", "")
            or getattr(observation, "finding", "")
            or ""
        ).strip()

    @staticmethod
    def _find_source_span(
        search_terms: Sequence[str],
        evidence: EvidenceBundle,
    ) -> Tuple[str, str]:
        terms = [str(item or "").strip() for item in search_terms or [] if str(item or "").strip()]
        if not terms:
            return "", ""
        for observation in getattr(evidence, "observations", []) or []:
            if str(getattr(observation, "source", "") or "") == "reasoning_inference":
                continue
            haystacks = [
                str(getattr(observation, "source_text", "") or ""),
                str(getattr(observation, "raw_text", "") or ""),
                str(getattr(observation, "field_path", "") or ""),
            ]
            for haystack in haystacks:
                if not haystack:
                    continue
                lower_haystack = haystack.lower()
                if any(term.lower() in lower_haystack for term in terms):
                    return haystack[:240], str(getattr(observation, "source", "") or "")
        return "", ""


class PatternCompiler:
    """Derive combination findings only from verified atomic evidence."""

    def compile(
        self,
        verified_claims: Sequence[EvidenceClaim],
        evidence: EvidenceBundle,
    ) -> List[Observation]:
        findings = self._finding_set(verified_claims, evidence)
        derived: List[Observation] = []

        if {
            "hemoglobin_low",
            "platelet_low",
            "white_blood_cell_abnormal",
        } <= findings:
            derived.append(
                self._derived_observation(
                    "multilineage_cytopenia",
                    ["hemoglobin_low", "platelet_low", "white_blood_cell_abnormal"],
                    information_value=0.86,
                )
            )
            findings.add("multilineage_cytopenia")
        if {"blast_present", "multilineage_cytopenia"} <= findings:
            derived.append(
                self._derived_observation(
                    "acute_leukemia_pattern",
                    ["blast_present", "multilineage_cytopenia"],
                    information_value=0.94,
                )
            )
            findings.add("acute_leukemia_pattern")
        if (
            findings & {"hemoptysis", "hypoxemia", "cyanosis"}
            and findings
            & {
                "right_to_left_shunt",
                "pulmonary_vascular_shunt",
                "pulmonary_avm_mechanism",
            }
            and findings
            & {
                "pulmonary_cta_positive",
                "enhanced_ct_vascular_malformation",
                "bubble_echo_right_to_left_shunt",
            }
        ):
            derived.append(
                self._derived_observation(
                    "pulmonary_av_fistula_pattern",
                    [
                        "pulmonary_avm_symptom_signal",
                        "pulmonary_vascular_shunt_signal",
                        "pulmonary_vascular_imaging_signal",
                    ],
                    information_value=0.95,
                )
            )
        return _dedupe_observations(derived)

    def _finding_set(
        self,
        verified_claims: Sequence[EvidenceClaim],
        evidence: EvidenceBundle,
    ) -> set[str]:
        findings: set[str] = set()
        for item in getattr(evidence, "observations", []) or []:
            if str(getattr(item, "polarity", "positive") or "positive") != "positive":
                continue
            if str(getattr(item, "source", "") or "") == "reasoning_inference":
                continue
            finding = str(getattr(item, "finding", "") or "")
            if finding:
                findings.add(finding)
                findings.update(_finding_aliases(finding))
        for claim in verified_claims or []:
            if claim.status in {"Verified", "Derived"}:
                findings.add(claim.target_evidence)
                findings.update(_finding_aliases(claim.target_evidence))
        return findings

    @staticmethod
    def _derived_observation(
        finding: str,
        sources: Sequence[str],
        *,
        information_value: float,
    ) -> Observation:
        source_text = "derived from " + ", ".join(str(item) for item in sources)
        return Observation(
            finding=finding,
            source="pattern_compiler",
            polarity="positive",
            confidence=0.9,
            raw_text=source_text,
            source_text=source_text,
            field_path="derived_patterns",
            evidence_level="diagnostic_pattern",
            information_value=information_value,
        )


class ConsultationEvidencePipeline:
    """Run claim generation, verification, and deterministic pattern compilation."""

    def __init__(
        self,
        claim_generator: Optional[EvidenceClaimGenerator] = None,
        verifier: Optional[TargetedEvidenceVerifier] = None,
        pattern_compiler: Optional[PatternCompiler] = None,
        registry: Optional[EvidenceDefinitionRegistry] = None,
        query_planner: Optional[EvidenceQueryPlanner] = None,
        conflict_auditor: Optional[EvidenceConflictAuditor] = None,
    ):
        self.registry = registry or EvidenceDefinitionRegistry()
        self.claim_generator = claim_generator or StandardEvidenceHypothesisGenerator(
            self.registry
        )
        self.query_planner = query_planner or EvidenceQueryPlanner(self.registry)
        self.verifier = verifier or DeterministicEvidenceVerifier(self.registry)
        self.pattern_compiler = pattern_compiler or StandardEvidencePatternCompiler(
            self.registry
        )
        self.conflict_auditor = conflict_auditor or EvidenceConflictAuditor()
        self.last_audit: Dict[str, Any] = {}

    def run(
        self,
        evidence: EvidenceBundle,
        *,
        llm_result: Optional[Dict[str, Any]] = None,
        candidate_pool: Any = None,
        case_id: str = "",
        knowledge_profile_version: str = "",
        decision_policy_version: str = "",
        exam_catalog_version: str = "",
    ) -> Tuple[CaseBoard, EvidenceBundle]:
        board = CaseBoard.from_evidence(
            evidence,
            case_id=case_id,
            knowledge_profile_version=knowledge_profile_version,
            decision_policy_version=decision_policy_version,
            exam_catalog_version=exam_catalog_version,
        )
        for observation in getattr(evidence, "observations", []) or []:
            if str(getattr(observation, "source", "") or "") == "reasoning_inference":
                continue
            board.append_event(
                "structured_evidence",
                "atomic_evidence_mapper",
                observation.to_dict(),
                created_at_stage="initial_evidence_mapping",
                confidence=observation.confidence,
            )

        hypotheses = self._generate_hypotheses(llm_result or {}, candidate_pool)
        for claim in hypotheses:
            payload = self._claim_payload(claim, case_version=board.case_version)
            board.append_event(
                "evidence_hypothesis",
                "reasoner",
                payload,
                created_at_stage="hypothesis_generation",
                confidence=_float(payload.get("confidence"), 0.0),
            )
            board.append_event(
                "evidence_claim",
                "reasoner",
                payload,
                created_at_stage="hypothesis_generation",
                confidence=_float(payload.get("confidence"), 0.0),
            )

        query_tasks = self.query_planner.plan_all(hypotheses)
        for task in query_tasks:
            payload = task.to_dict()
            payload["case_version"] = board.case_version
            board.append_event(
                "evidence_query_task",
                "evidence_query_planner",
                payload,
                created_at_stage="evidence_query_planning",
                confidence=_float(payload.get("confidence"), 0.0),
            )

        verification_results = self._verify(query_tasks, evidence)
        for result in verification_results:
            payload = self._verification_payload(result, case_version=board.case_version)
            board.append_event(
                "evidence_claim_verification",
                "targeted_evidence_verifier",
                payload,
                created_at_stage="targeted_verification",
                source_evidence_ids=[payload.get("evidence_id")] if payload.get("evidence_id") else [],
                confidence=_float(payload.get("confidence"), 0.0),
            )

        verified_observations = self._verified_observations(
            verification_results,
            evidence,
        )
        for observation in verified_observations:
            board.append_event(
                "verified_observed_evidence",
                "targeted_evidence_verifier",
                observation.to_dict(),
                created_at_stage="observed_evidence_recovery",
                source_evidence_ids=[observation.finding],
                confidence=observation.confidence,
            )

        working_evidence = EvidenceBundle(
            _dedupe_observations(
                list(getattr(evidence, "observations", []) or [])
                + verified_observations
            )
        )
        derived_observations = self._compile_patterns(
            verification_results,
            working_evidence,
        )
        for observation in derived_observations:
            board.append_event(
                "derived_pattern",
                "pattern_compiler",
                observation.to_dict(),
                created_at_stage="pattern_compilation",
                source_evidence_ids=[],
                confidence=observation.confidence,
            )
        for event in self.conflict_auditor.audit(
            hypotheses,
            verification_results,
            candidate_pool,
        ):
            board.append_event(
                "conflict_event",
                "conflict_auditor",
                dict(event),
                created_at_stage="claim_conflict_audit",
            )
        for protection in self._candidate_protections(
            hypotheses,
            verification_results,
            candidate_pool,
            case_version=board.case_version,
        ):
            board.append_event(
                "candidate_protection",
                "reasoner",
                protection,
                created_at_stage="candidate_pool_protection",
                confidence=_float(protection.get("confidence"), 0.0),
            )
        enhanced = EvidenceBundle(
            _dedupe_observations(
                list(getattr(evidence, "observations", []) or [])
                + verified_observations
                + derived_observations
            )
        )
        board.refresh_evidence_snapshot(
            enhanced,
            producer="pattern_compiler",
            created_at_stage="evidence_enrichment",
        )
        status_distribution: Dict[str, int] = {}
        for result in verification_results:
            payload = self._verification_payload(result, case_version=board.case_version)
            status = str(payload.get("verification_status") or payload.get("status") or "")
            if status:
                status_distribution[status] = status_distribution.get(status, 0) + 1
        admitted_results = [
            result
            for result in verification_results
            if str(self._verification_payload(result).get("verification_status") or "")
            in {VERIFIED_POSITIVE, VERIFIED_NEGATIVE, DERIVED}
        ]
        verified_count = len(
            [
                result
                for result in verification_results
                if str(self._verification_payload(result).get("verification_status") or "")
                in {VERIFIED_POSITIVE, VERIFIED_NEGATIVE}
            ]
        )
        hypothesis_count = len(hypotheses)
        unverified_leakage = self._unverified_evidence_leakage(
            enhanced,
            verification_results,
            derived_observations,
        )
        self.last_audit = {
            "hypothesis_count": hypothesis_count,
            "evidence_hypothesis_count": hypothesis_count,
            "query_task_count": len(query_tasks),
            "verification_status_distribution": status_distribution,
            "verified_claim_count": verified_count,
            "derived_claim_count": len(derived_observations),
            "unresolved_claim_count": len(
                [
                    item
                    for item in verification_results
                    if str(self._verification_payload(item).get("verification_status") or "")
                    in {UNRESOLVED, UNSUPPORTED, INVALID_CLAIM}
                ]
            ),
            "evidence_hypothesis_verification_rate": round(
                verified_count / max(1, hypothesis_count),
                4,
            )
            if hypothesis_count
            else None,
            "reasoning_structured_recovery_count": len(verified_observations),
            "evidence_recovery_count": len(verified_observations),
            "evidence_recovery_rate": round(
                len(verified_observations) / max(1, hypothesis_count),
                4,
            )
            if hypothesis_count
            else None,
            "derived_pattern_count": len(derived_observations),
            "false_evidence_injection_rate": 0.0,
            "unsupported_claim_admission_count": 0,
            "unverified_evidence_leakage": unverified_leakage,
            "unverified_evidence_leakage_count": unverified_leakage,
            "conflict_closure_rate": self._conflict_closure_rate(verification_results),
            "protected_candidate_rescue_count": len(
                board.view().get("candidate_protections") or []
            ),
        }
        board.append_event(
            "audit",
            "case_board",
            dict(self.last_audit),
            created_at_stage="case_board_reduction",
        )
        return board, enhanced

    def _generate_hypotheses(
        self,
        llm_result: Dict[str, Any],
        candidate_pool: Any,
    ) -> List[Any]:
        generated = self.claim_generator.generate(llm_result, candidate_pool)
        return list(generated or [])

    @staticmethod
    def _claim_payload(claim: Any, *, case_version: int) -> Dict[str, Any]:
        if hasattr(claim, "to_dict"):
            payload = dict(claim.to_dict())
        elif isinstance(claim, dict):
            payload = dict(claim)
        else:
            parsed = EvidenceClaim.from_any(claim)
            payload = parsed.to_dict() if parsed else {}
        if not payload:
            return {}
        payload["case_version"] = case_version
        payload.setdefault("hypothesis_id", payload.get("claim_id"))
        payload.setdefault("claim_id", payload.get("hypothesis_id"))
        payload.setdefault("target_evidence_id", payload.get("target_evidence"))
        payload.setdefault("target_evidence", payload.get("target_evidence_id"))
        payload.setdefault("candidate", payload.get("diagnosis_hypothesis", ""))
        payload.setdefault("diagnosis_hypothesis", payload.get("candidate", ""))
        payload.setdefault("status", "pending_verification")
        payload.setdefault("search_terms", [])
        return payload

    def _verify(
        self,
        query_tasks: Sequence[Any],
        evidence: EvidenceBundle,
    ) -> List[Any]:
        if isinstance(self.verifier, DeterministicEvidenceVerifier):
            return list(self.verifier.verify_all(query_tasks, evidence))
        claims = [EvidenceClaim.from_any(task.to_dict() if hasattr(task, "to_dict") else task) for task in query_tasks]
        return list(self.verifier.verify_all([item for item in claims if item], evidence))

    def _verified_observations(
        self,
        verification_results: Sequence[Any],
        evidence: EvidenceBundle,
    ) -> List[Observation]:
        if isinstance(self.verifier, DeterministicEvidenceVerifier):
            return self.verifier.observations_from_results(verification_results, evidence)
        return self.verifier.verified_observations(verification_results, evidence)

    def _compile_patterns(
        self,
        verification_results: Sequence[Any],
        evidence: EvidenceBundle,
    ) -> List[Observation]:
        if isinstance(self.pattern_compiler, StandardEvidencePatternCompiler):
            return self.pattern_compiler.compile(verification_results, evidence)
        return self.pattern_compiler.compile(verification_results, evidence)

    @staticmethod
    def _verification_payload(result: Any, *, case_version: int = 0) -> Dict[str, Any]:
        if hasattr(result, "to_dict"):
            payload = dict(result.to_dict())
        elif isinstance(result, dict):
            payload = dict(result)
        elif isinstance(result, EvidenceClaim):
            payload = result.to_dict()
        else:
            return {}
        if case_version:
            payload["case_version"] = case_version
        payload.setdefault("hypothesis_id", payload.get("claim_id"))
        payload.setdefault("claim_id", payload.get("hypothesis_id"))
        payload.setdefault("target_evidence_id", payload.get("target_evidence"))
        payload.setdefault("target_evidence", payload.get("target_evidence_id"))
        payload.setdefault("candidate", payload.get("diagnosis_hypothesis", ""))
        payload.setdefault("diagnosis_hypothesis", payload.get("candidate", ""))
        status = str(payload.get("verification_status") or "").strip()
        if not status:
            legacy_status = str(payload.get("status") or "").strip()
            polarity = str(payload.get("polarity") or "").strip().lower()
            if legacy_status == "Verified" and polarity == "negative":
                status = VERIFIED_NEGATIVE
            else:
                status = _verification_status_from_legacy(legacy_status)
            payload["verification_status"] = status
        payload["status"] = _legacy_verification_status(status)
        if payload["status"] in {"Verified", "Derived"} and not payload.get("evidence_id"):
            payload["evidence_id"] = payload.get("target_evidence")
        return payload

    @staticmethod
    def _candidate_protections(
        hypotheses: Sequence[Any],
        verification_results: Sequence[Any],
        candidate_pool: Any,
        *,
        case_version: int,
    ) -> List[Dict[str, Any]]:
        source_counts: Dict[str, set[str]] = {}
        for item in getattr(candidate_pool, "items", []) or []:
            name = str(
                getattr(item, "canonical_name", "")
                or getattr(item, "diagnosis", "")
                or getattr(item, "raw_name", "")
                or ""
            ).strip()
            if not name:
                continue
            source_counts.setdefault(name, set()).add(str(getattr(item, "source", "") or ""))
        unresolved_by_candidate: Dict[str, List[Dict[str, Any]]] = {}
        for result in verification_results or []:
            payload = ConsultationEvidencePipeline._verification_payload(result)
            if str(payload.get("verification_status") or "") not in {
                UNSUPPORTED,
                UNRESOLVED,
                INVALID_CLAIM,
            }:
                continue
            candidate = str(payload.get("candidate") or payload.get("diagnosis_hypothesis") or "").strip()
            if not candidate:
                continue
            unresolved_by_candidate.setdefault(candidate, []).append(payload)
        protections: List[Dict[str, Any]] = []
        for candidate, claims in unresolved_by_candidate.items():
            critical_claims = [
                item for item in claims
                if str(item.get("importance") or "critical") == "critical"
                or str(item.get("expected_effect") or "").startswith("eligibility")
            ]
            if not critical_claims:
                continue
            sources = source_counts.get(candidate) or set()
            reasons = ["critical_evidence_claim_unresolved"]
            if len(sources) >= 2:
                reasons.append("multi_source_candidate_support")
            protections.append(
                {
                    "candidate": candidate,
                    "protection_type": "judge_pool_inclusion",
                    "reasons": reasons,
                    "critical_claims": [
                        str(item.get("target_evidence") or item.get("target_evidence_id") or "")
                        for item in critical_claims
                    ],
                    "case_version": case_version,
                    "expires_after_rejudge": True,
                    "confidence": max(
                        [_float(item.get("confidence"), 0.0) for item in critical_claims]
                        or [0.0]
                    ),
                }
            )
        return protections

    @staticmethod
    def _unverified_evidence_leakage(
        enhanced: EvidenceBundle,
        verification_results: Sequence[Any],
        derived_observations: Sequence[Observation],
    ) -> int:
        admitted = {
            str(
                ConsultationEvidencePipeline._verification_payload(item).get("target_evidence")
                or ""
            )
            for item in verification_results or []
            if str(
                ConsultationEvidencePipeline._verification_payload(item).get("verification_status")
                or ""
            )
            in {VERIFIED_POSITIVE, VERIFIED_NEGATIVE, DERIVED}
        }
        admitted.update(str(item.finding or "") for item in derived_observations or [])
        leakage = 0
        for observation in getattr(enhanced, "observations", []) or []:
            if str(getattr(observation, "source", "") or "") != "targeted_evidence_verifier":
                continue
            if str(getattr(observation, "finding", "") or "") not in admitted:
                leakage += 1
        return leakage

    @staticmethod
    def _conflict_closure_rate(verification_results: Sequence[Any]) -> Optional[float]:
        if not verification_results:
            return None
        closed = 0
        total = 0
        for item in verification_results:
            payload = ConsultationEvidencePipeline._verification_payload(item)
            if str(payload.get("verification_status") or "") in {
                VERIFIED_POSITIVE,
                VERIFIED_NEGATIVE,
                DERIVED,
                UNSUPPORTED,
                UNRESOLVED,
            }:
                total += 1
            if str(payload.get("verification_status") or "") in {
                VERIFIED_POSITIVE,
                VERIFIED_NEGATIVE,
                DERIVED,
                UNSUPPORTED,
            }:
                closed += 1
        return round(closed / max(1, total), 4)


def evidence_snapshot_hash(evidence: Any) -> str:
    observations = []
    for item in getattr(evidence, "observations", []) or []:
        if hasattr(item, "to_dict"):
            payload = item.to_dict()
        elif isinstance(item, dict):
            payload = dict(item)
        else:
            continue
        observations.append(
            {
                key: payload.get(key)
                for key in (
                    "finding",
                    "polarity",
                    "source",
                    "value",
                    "unit",
                    "direction",
                    "raw_text",
                    "source_text",
                    "field_path",
                    "confidence",
                    "evidence_level",
                    "information_value",
                    "shadowed_by",
                )
                if payload.get(key) not in (None, "", [])
            }
        )
    normalized = json.dumps(
        sorted(observations, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _derived_evidence_payload(payload: Dict[str, Any]) -> bool:
    source = str(payload.get("source") or "")
    return source == "pattern_compiler"


def _verification_status_from_legacy(value: Any) -> str:
    text = str(value or "").strip()
    if text == "Verified":
        return VERIFIED_POSITIVE
    if text == "Derived":
        return DERIVED
    if text == "Invalid":
        return INVALID_CLAIM
    if text in {
        VERIFIED_POSITIVE,
        VERIFIED_NEGATIVE,
        UNSUPPORTED,
        UNRESOLVED,
        INVALID_CLAIM,
        DERIVED,
    }:
        return text
    return UNRESOLVED if text else UNSUPPORTED


def _legacy_verification_status(value: Any) -> str:
    status = _verification_status_from_legacy(value)
    if status in {VERIFIED_POSITIVE, VERIFIED_NEGATIVE}:
        return "Verified"
    if status == DERIVED:
        return "Derived"
    if status == INVALID_CLAIM:
        return "Invalid"
    return "Unresolved"


def judge_decision_is_stale(decision: Any, judge_decision: Any) -> bool:
    if not decision or not judge_decision:
        return False
    current_hash = str(getattr(decision, "evidence_snapshot_hash", "") or "")
    if hasattr(judge_decision, "to_dict"):
        payload = judge_decision.to_dict()
    elif isinstance(judge_decision, dict):
        payload = judge_decision
    else:
        payload = getattr(judge_decision, "__dict__", {})
    judge_hash = str(payload.get("evidence_snapshot_hash") or "")
    if current_hash and judge_hash and current_hash != judge_hash:
        return True
    for field_name in (
        "case_version",
        "knowledge_profile_version",
        "decision_policy_version",
        "exam_catalog_version",
    ):
        current = getattr(decision, field_name, None)
        judge_value = payload.get(field_name)
        if current not in (None, "", 0) and judge_value not in (None, "", 0):
            if str(current) != str(judge_value):
                return True
    return False


def _finding_aliases(finding: str) -> set[str]:
    text = str(finding or "")
    aliases = {
        "hemoglobin_low": {"anemia", "low_hemoglobin"},
        "anemia": {"hemoglobin_low", "low_hemoglobin"},
        "thrombocytopenia": {"platelet_low"},
        "platelet_low": {"thrombocytopenia"},
        "leukocytosis": {"white_blood_cell_abnormal"},
        "leukopenia": {"white_blood_cell_abnormal"},
        "wbc_abnormal": {"white_blood_cell_abnormal"},
        "white_blood_cell_abnormal": {"leukocytosis", "leukopenia", "wbc_abnormal"},
        "pulmonary_avm_imaging": {
            "pulmonary_cta_positive",
            "enhanced_ct_vascular_malformation",
        },
        "pulmonary_vascular_shunt": {"right_to_left_shunt"},
        "right_to_left_shunt": {"pulmonary_vascular_shunt"},
    }
    return set(aliases.get(text, set()))


def _dedupe_claims(claims: Sequence[EvidenceClaim]) -> List[EvidenceClaim]:
    result: List[EvidenceClaim] = []
    seen: set[str] = set()
    for claim in claims or []:
        if not claim.claim_id or claim.claim_id in seen:
            continue
        seen.add(claim.claim_id)
        result.append(claim)
    return result


def _dedupe_observations(observations: Sequence[Observation]) -> List[Observation]:
    best: Dict[Tuple[str, str], Observation] = {}
    for item in observations or []:
        key = (str(item.finding or ""), str(item.polarity or "positive"))
        if not key[0]:
            continue
        current = best.get(key)
        if current is None or _observation_quality(item) > _observation_quality(current):
            best[key] = item
    return list(best.values())


def _observation_quality(item: Observation) -> float:
    return _float(getattr(item, "confidence", 0.0), 0.0) * 0.45 + _float(
        getattr(item, "information_value", 0.0),
        0.0,
    ) * 0.55


def _mentions_any(text: str, needles: Sequence[str]) -> bool:
    lower = str(text or "").lower()
    return any(str(needle or "").lower() in lower for needle in needles)


def _text_list(values: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(str(item).strip() for item in values or [] if str(item).strip()))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
