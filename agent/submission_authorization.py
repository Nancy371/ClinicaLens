"""Submission authorization records and policy orchestration.

This layer sits after Judge / primary arbitration. It does not rank candidates
or decide whether disease evidence is true; it only decides which established
diagnoses may enter the final submission payload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set


ROLE_PRIMARY = "PRIMARY"
ROLE_SECONDARY_INDEPENDENT = "SECONDARY_INDEPENDENT"
ROLE_COMPLICATION = "COMPLICATION"
ROLE_ASSOCIATED_FINDING = "ASSOCIATED_FINDING"
ROLE_DIFFERENTIAL_ONLY = "DIFFERENTIAL_ONLY"
ROLE_UNCONFIRMED = "UNCONFIRMED"

AUTH_AUTHORIZED = "AUTHORIZED"
AUTH_NOT_AUTHORIZED = "NOT_AUTHORIZED"
AUTH_DEFERRED = "DEFERRED"

DEP_MANIFESTATION_OF = "MANIFESTATION_OF"
DEP_COMPLICATION_OF = "COMPLICATION_OF"
DEP_DOWNSTREAM_OF = "DOWNSTREAM_OF"
DEP_OVERLAPPING_EXPLANATION = "OVERLAPPING_EXPLANATION"
DEP_COEXISTING_INDEPENDENT = "COEXISTING_INDEPENDENT"
DEP_UNCERTAIN_DEPENDENCY = "UNCERTAIN_DEPENDENCY"


@dataclass
class SubmissionDependencyEdge:
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    source_diagnosis: str = ""
    target_diagnosis: str = ""
    supporting_evidence_refs: List[str] = field(default_factory=list)
    shared_evidence_refs: List[str] = field(default_factory=list)
    independent_evidence_refs: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "submission_authorization"
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubmissionAuthorizationRecord:
    entity_id: str
    diagnosis_name: str
    submission_role: str
    submission_authorization: str
    anchor_status: str = ""
    eligibility_status: str = ""
    primary_arbitration_role: str = ""
    independent_evidence_refs: List[str] = field(default_factory=list)
    dependency_edges: List[Dict[str, Any]] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    decision_id: str = ""
    arbitration_id: str = ""
    diagnostic_state_version: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubmissionAuthorizationResult:
    pre_authorization_diagnoses: List[str]
    authorized_diagnoses: List[str]
    blocked_diagnoses: List[Dict[str, Any]]
    records: List[SubmissionAuthorizationRecord]
    dependency_edges: List[SubmissionDependencyEdge]
    authorized_candidates: List[Any]
    primary_candidate: Optional[Any]
    submission_authorization_bypass_count: int = 0
    associated_finding_block_count: int = 0
    authorized_primary_count: int = 0
    authorized_secondary_count: int = 0

    def record_dicts(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.records]

    def edge_dicts(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.dependency_edges]


class SubmissionAuthorizationLayer:
    """Authorize final submission roles from already-ranked candidates."""

    _PULMONARY_MORPHOLOGY_TOKENS = {
        "atelectasis",
        "lung_volume_loss",
        "pulmonary_volume_loss",
        "ground_glass_opacity",
        "pulmonary_consolidation",
        "patchy_pulmonary_opacity",
        "pulmonary_opacity",
        "pulmonary_infiltrative_opacity",
        "lesion_within_prior_radiation_field",
        "radiation_field_lung_consistency",
    }

    def __init__(self, knowledge: Any, max_final_diagnoses: int = 3):
        self.knowledge = knowledge
        self.max_final_diagnoses = int(max_final_diagnoses or 3)

    def authorize(
        self,
        decision: Any,
        requested_names: Sequence[str],
        *,
        policy: Any,
        respect_differential_only: bool = True,
    ) -> SubmissionAuthorizationResult:
        pre_names = [
            str(item).strip()
            for item in dict.fromkeys(requested_names or [])
            if str(item).strip()
        ]
        score_by_name = {item.diagnosis: item for item in decision.candidates or []}
        score_by_entity = {
            item.entity_id: item
            for item in decision.candidates or []
            if getattr(item, "entity_id", "")
        }
        eligible: List[Any] = []
        blocked: List[Dict[str, Any]] = []
        records: List[SubmissionAuthorizationRecord] = []
        edges: List[SubmissionDependencyEdge] = []

        for name in pre_names:
            candidate = self._candidate_for_name(score_by_name, score_by_entity, name)
            reason = policy._authorization_ineligible_reason(
                candidate,
                respect_differential_only=respect_differential_only,
                decision=decision,
            )
            if reason:
                blocked.append(policy._authorization_block_record(name, candidate, reason))
                records.append(
                    self._record_for_candidate(
                        candidate,
                        diagnosis_name=name,
                        role=self._blocked_role(candidate),
                        authorization=(
                            AUTH_DEFERRED
                            if str(getattr(candidate, "eligibility_status", "")) == "Deferred"
                            else AUTH_NOT_AUTHORIZED
                        ),
                        reason_codes=[self._reason_code(reason)],
                        decision=decision,
                    )
                )
                continue
            if candidate and candidate not in eligible:
                eligible.append(candidate)

        if not eligible:
            return SubmissionAuthorizationResult(
                pre_authorization_diagnoses=pre_names,
                authorized_diagnoses=[],
                blocked_diagnoses=blocked,
                records=records,
                dependency_edges=edges,
                authorized_candidates=[],
                primary_candidate=None,
            )

        primary = policy._choose_authorized_primary(eligible, decision)
        authorized: List[Any] = [primary]
        records.append(
            self._record_for_candidate(
                primary,
                role=ROLE_PRIMARY,
                authorization=AUTH_AUTHORIZED,
                reason_codes=["PRIMARY_ARBITRATION_WINNER_AUTHORIZED"],
                decision=decision,
                primary_arbitration_role="winner",
            )
        )

        associated_blocks = 0
        for candidate in eligible:
            if candidate.diagnosis == primary.diagnosis:
                continue
            dependency_edges = self._dependency_edges(primary, candidate, policy)
            edges.extend(dependency_edges)
            role, role_reason = self._secondary_role(primary, candidate, dependency_edges, policy)
            reason = ""
            if role == ROLE_ASSOCIATED_FINDING:
                reason = role_reason or "associated finding is explained by primary diagnosis"
                associated_blocks += 1
            else:
                reason = policy._secondary_authorization_block_reason(
                    candidate,
                    primary,
                    authorized,
                )
            if reason:
                blocked.append(policy._authorization_block_record(candidate.diagnosis, candidate, reason))
                records.append(
                    self._record_for_candidate(
                        candidate,
                        role=role if role else self._blocked_role(candidate),
                        authorization=(
                            AUTH_DEFERRED if role == ROLE_UNCONFIRMED else AUTH_NOT_AUTHORIZED
                        ),
                        reason_codes=[self._reason_code(reason)],
                        decision=decision,
                        dependency_edges=dependency_edges,
                        primary_arbitration_role="non_winner",
                    )
                )
                continue
            authorized.append(candidate)
            records.append(
                self._record_for_candidate(
                    candidate,
                    role=role or ROLE_SECONDARY_INDEPENDENT,
                    authorization=AUTH_AUTHORIZED,
                    reason_codes=["SECONDARY_INCLUSION_AUTHORIZED"],
                    decision=decision,
                    dependency_edges=dependency_edges,
                    primary_arbitration_role="secondary_included",
                )
            )
            if len(authorized) >= self.max_final_diagnoses:
                break

        authorized_names = [item.diagnosis for item in authorized]
        authorized_set = set(authorized_names)
        bypass_count = 0
        for name in pre_names:
            if name in authorized_set:
                continue
            if any(item.get("diagnosis") == name for item in blocked):
                continue
            candidate = self._candidate_for_name(score_by_name, score_by_entity, name)
            reason = "not selected by final diagnosis authorization gate"
            blocked.append(policy._authorization_block_record(name, candidate, reason))
            records.append(
                self._record_for_candidate(
                    candidate,
                    diagnosis_name=name,
                    role=self._blocked_role(candidate),
                    authorization=AUTH_NOT_AUTHORIZED,
                    reason_codes=[self._reason_code(reason)],
                    decision=decision,
                )
            )

        authorized_records = {
            item.diagnosis_name
            for item in records
            if item.submission_authorization == AUTH_AUTHORIZED
        }
        for name in authorized_names:
            if name not in authorized_records:
                bypass_count += 1

        return SubmissionAuthorizationResult(
            pre_authorization_diagnoses=pre_names,
            authorized_diagnoses=authorized_names,
            blocked_diagnoses=blocked,
            records=records,
            dependency_edges=edges,
            authorized_candidates=authorized,
            primary_candidate=primary,
            submission_authorization_bypass_count=bypass_count,
            associated_finding_block_count=associated_blocks,
            authorized_primary_count=1 if primary else 0,
            authorized_secondary_count=max(0, len(authorized) - 1),
        )

    def _candidate_for_name(
        self,
        score_by_name: Dict[str, Any],
        score_by_entity: Dict[str, Any],
        name: str,
    ) -> Optional[Any]:
        entity_id = self.knowledge.entity_id_for(name) if self.knowledge else ""
        return (score_by_entity.get(entity_id) if entity_id else None) or score_by_name.get(name)

    def _record_for_candidate(
        self,
        candidate: Optional[Any],
        *,
        diagnosis_name: str = "",
        role: str,
        authorization: str,
        reason_codes: Sequence[str],
        decision: Any,
        dependency_edges: Sequence[SubmissionDependencyEdge] = (),
        primary_arbitration_role: str = "",
    ) -> SubmissionAuthorizationRecord:
        name = diagnosis_name or str(getattr(candidate, "diagnosis", "") or "")
        return SubmissionAuthorizationRecord(
            entity_id=str(getattr(candidate, "entity_id", "") or ""),
            diagnosis_name=name,
            submission_role=role,
            submission_authorization=authorization,
            anchor_status=str(getattr(candidate, "eligibility_anchor_status", "") or ""),
            eligibility_status=str(getattr(candidate, "eligibility_status", "") or ""),
            primary_arbitration_role=primary_arbitration_role,
            independent_evidence_refs=self._independent_evidence_refs(candidate, dependency_edges),
            dependency_edges=[item.to_dict() for item in dependency_edges],
            reason_codes=list(dict.fromkeys(reason_codes or [])),
            decision_id=str(getattr(decision, "evidence_snapshot_hash", "") or ""),
            arbitration_id=str((getattr(decision, "judge_decision", {}) or {}).get("arbitration_id") or ""),
            diagnostic_state_version=int(getattr(decision, "diagnostic_state_version", 0) or 0),
        )

    @staticmethod
    def _blocked_role(candidate: Optional[Any]) -> str:
        status = str(getattr(candidate, "eligibility_status", "") or "")
        if status == "Deferred":
            return ROLE_UNCONFIRMED
        if bool(getattr(candidate, "differential_only", False)):
            return ROLE_DIFFERENTIAL_ONLY
        return ROLE_UNCONFIRMED

    def _secondary_role(
        self,
        primary: Any,
        candidate: Any,
        dependency_edges: Sequence[SubmissionDependencyEdge],
        policy: Any,
    ) -> tuple[str, str]:
        if self._policy_marks_associated(primary, candidate):
            return ROLE_ASSOCIATED_FINDING, "associated finding is explained by primary diagnosis"
        if self._primary_explains_pulmonary_morphology(primary, candidate):
            return ROLE_ASSOCIATED_FINDING, "pulmonary morphology finding is explained by primary diagnosis"
        if any(item.relation_type == DEP_DOWNSTREAM_OF for item in dependency_edges):
            if bool(getattr(candidate, "root_cause_submit_as_final", False)):
                return ROLE_COMPLICATION, ""
            return ROLE_ASSOCIATED_FINDING, "downstream finding is explained by primary diagnosis"
        if any(item.relation_type == DEP_COMPLICATION_OF for item in dependency_edges):
            return ROLE_COMPLICATION, ""
        if str(getattr(candidate, "eligibility_status", "") or "") == "Deferred":
            return ROLE_UNCONFIRMED, "candidate deferred pending required anchor evidence"
        return ROLE_SECONDARY_INDEPENDENT, ""

    def _dependency_edges(
        self,
        primary: Any,
        candidate: Any,
        policy: Any,
    ) -> List[SubmissionDependencyEdge]:
        edges: List[SubmissionDependencyEdge] = []
        shared = sorted(set(primary.matched_evidence or []) & set(candidate.matched_evidence or []))
        independent = self._independent_evidence_refs(candidate, [])
        if self._policy_marks_associated(primary, candidate):
            edges.append(
                self._edge(
                    primary,
                    candidate,
                    DEP_MANIFESTATION_OF,
                    shared,
                    independent,
                    ["CONTROLLED_PRIMARY_ASSOCIATED_FINDING_POLICY"],
                    confidence=0.86,
                )
            )
        if policy._diagnosis_causes(primary.diagnosis, candidate.diagnosis):
            edges.append(
                self._edge(
                    primary,
                    candidate,
                    DEP_DOWNSTREAM_OF,
                    shared,
                    independent,
                    ["KNOWLEDGE_CAUSAL_DOWNSTREAM"],
                    confidence=0.82,
                )
            )
        if str(getattr(candidate, "explained_by_root_cause", "") or "") == primary.diagnosis:
            relation = (
                DEP_COMPLICATION_OF
                if bool(getattr(candidate, "root_cause_submit_as_final", False))
                else DEP_DOWNSTREAM_OF
            )
            edges.append(
                self._edge(
                    primary,
                    candidate,
                    relation,
                    shared,
                    independent,
                    ["ROOT_CAUSE_ARBITRATION_EDGE"],
                    confidence=0.8,
                )
            )
        if self._primary_explains_pulmonary_morphology(primary, candidate):
            edges.append(
                self._edge(
                    primary,
                    candidate,
                    DEP_OVERLAPPING_EXPLANATION,
                    shared,
                    independent,
                    ["PRIMARY_EXPLAINS_PULMONARY_MORPHOLOGY"],
                    confidence=0.74,
                )
            )
        return edges

    def _edge(
        self,
        primary: Any,
        candidate: Any,
        relation_type: str,
        shared: Sequence[str],
        independent: Sequence[str],
        reason_codes: Sequence[str],
        *,
        confidence: float,
    ) -> SubmissionDependencyEdge:
        return SubmissionDependencyEdge(
            source_entity_id=str(getattr(candidate, "entity_id", "") or ""),
            target_entity_id=str(getattr(primary, "entity_id", "") or ""),
            source_diagnosis=str(getattr(candidate, "diagnosis", "") or ""),
            target_diagnosis=str(getattr(primary, "diagnosis", "") or ""),
            relation_type=relation_type,
            supporting_evidence_refs=list(getattr(candidate, "matched_evidence", []) or [])[:8],
            shared_evidence_refs=list(shared)[:8],
            independent_evidence_refs=list(independent)[:8],
            confidence=float(confidence),
            reason_codes=list(dict.fromkeys(reason_codes or [])),
        )

    def _policy_marks_associated(self, primary: Any, candidate: Any) -> bool:
        entry = self.knowledge.get(getattr(primary, "diagnosis", "")) if self.knowledge else {}
        policy = entry.get("submission_dependency_policy") or {}
        names = {
            str(item)
            for item in policy.get("associated_findings", []) or []
        }
        entity_ids = {
            str(item)
            for item in policy.get("associated_entity_ids", []) or []
        }
        candidate_name = str(getattr(candidate, "diagnosis", "") or "")
        candidate_entity = str(getattr(candidate, "entity_id", "") or "")
        return bool(candidate_name in names or candidate_entity in entity_ids)

    def _primary_explains_pulmonary_morphology(self, primary: Any, candidate: Any) -> bool:
        primary_entry = self.knowledge.get(getattr(primary, "diagnosis", "")) if self.knowledge else {}
        primary_family = str(primary_entry.get("family") or primary_entry.get("disease_family") or "")
        primary_system = str(primary_entry.get("body_system") or "")
        if primary_system != "pulmonary" and "lung_injury" not in primary_family:
            return False
        candidate_entry = self.knowledge.get(getattr(candidate, "diagnosis", "")) if self.knowledge else {}
        candidate_type = str(getattr(candidate, "diagnosis_type", "") or candidate_entry.get("diagnosis_type") or "")
        if candidate_type not in {"structural", "state", "complication"}:
            return False
        evidence = set(str(item) for item in getattr(candidate, "matched_evidence", []) or [])
        if not evidence:
            return False
        morphology = evidence & self._PULMONARY_MORPHOLOGY_TOKENS
        if not morphology:
            return False
        primary_evidence = set(str(item) for item in getattr(primary, "matched_evidence", []) or [])
        primary_has_context = bool(
            primary_evidence
            & {
                "thoracic_radiotherapy",
                "radiation_field_lung_consistency",
                "lesion_within_prior_radiation_field",
                "ground_glass_opacity",
                "pulmonary_consolidation",
            }
        )
        if not primary_has_context:
            return False
        independent = set(self._independent_evidence_refs(candidate, []))
        return not bool(independent - morphology)

    def _independent_evidence_refs(
        self,
        candidate: Optional[Any],
        dependency_edges: Sequence[SubmissionDependencyEdge],
    ) -> List[str]:
        if not candidate:
            return []
        evidence = list(dict.fromkeys(str(item) for item in getattr(candidate, "matched_evidence", []) or []))
        shared: Set[str] = set()
        for edge in dependency_edges or []:
            shared.update(edge.shared_evidence_refs)
        return [
            item
            for item in evidence
            if item
            and item not in shared
            and not item.startswith("symptom:")
            and item not in {"cough", "dyspnea", "fever", "pain", "symptom:signal"}
        ][:8]

    @staticmethod
    def _reason_code(reason: str) -> str:
        text = str(reason or "").strip()
        if not text:
            return "UNSPECIFIED"
        code = "".join(ch if ch.isalnum() else "_" for ch in text.upper())
        code = "_".join(part for part in code.split("_") if part)
        return code[:96] or "UNSPECIFIED"
