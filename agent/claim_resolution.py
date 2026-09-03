"""Claim resolution persistence for gap-aware diagnostic contracts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SUPPORTED = "SUPPORTED"
CONTRADICTED = "CONTRADICTED"
INCONCLUSIVE = "INCONCLUSIVE"
NOT_ADDRESSED = "NOT_ADDRESSED"
NOT_APPLICABLE = "NOT_APPLICABLE"

UNRESOLVED = "UNRESOLVED"
CONFLICTED = "CONFLICTED"
CLAIM_ACTIVE = "ACTIVE"
CLAIM_INACTIVE = "INACTIVE"

OPEN = "OPEN"
PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
FULLY_CLOSED = "FULLY_CLOSED"
BLOCKED_BY_CONTRADICTION = "BLOCKED_BY_CONTRADICTION"

PATTERN_SUPPORTED_BUT_UNCONFIRMED = "PatternSupportedButUnconfirmed"
ANCHOR_SATISFIED = "AnchorSatisfied"
ANCHOR_CONTRADICTED = "AnchorContradicted"
ANCHOR_CONFLICTED = "AnchorConflicted"
ANCHOR_UNSATISFIED = "AnchorUnsatisfied"

CLAIM_LEDGER_SCHEMA_VERSION = "claim_resolution_ledger_v1"
CLAIM_REDUCER_VERSION = "claim_resolution_reducer_v1"
GAP_CLOSURE_EVALUATOR_VERSION = "gap_closure_evaluator_v1"
ANCHOR_EVALUATOR_VERSION = "anchor_evaluator_v1"


@dataclass(frozen=True)
class ClaimMatchEvent:
    event_id: str
    candidate_id: str
    entity_id: str
    claim_id: str
    contract_id: str
    contract_version: str
    match_status: str
    supporting_evidence_refs: List[str] = field(default_factory=list)
    contradicting_evidence_refs: List[str] = field(default_factory=list)
    source_type: str = ""
    source_route_id: str = ""
    source_exam: str = ""
    evidence_version: int = 0
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimResolutionState:
    entity_id: str
    claim_id: str
    contract_id: str
    contract_version: str
    resolution_status: str = UNRESOLVED
    supporting_evidence_refs: List[str] = field(default_factory=list)
    contradicting_evidence_refs: List[str] = field(default_factory=list)
    last_attempt_status: str = ""
    last_route_id: str = ""
    first_resolved_evidence_version: int = 0
    last_updated_evidence_version: int = 0
    update_count: int = 0
    event_ids: List[str] = field(default_factory=list)
    lifecycle_status: str = CLAIM_ACTIVE
    claim_revision: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_any(cls, value: Any) -> Optional["ClaimResolutionState"]:
        if isinstance(value, ClaimResolutionState):
            return cls(**value.to_dict())
        if not isinstance(value, Mapping):
            return None
        entity_id = str(value.get("entity_id") or "").strip()
        claim_id = str(value.get("claim_id") or "").strip()
        contract_id = str(value.get("contract_id") or "").strip()
        if not entity_id or not claim_id or not contract_id:
            return None
        return cls(
            entity_id=entity_id,
            claim_id=claim_id,
            contract_id=contract_id,
            contract_version=str(value.get("contract_version") or "1"),
            resolution_status=str(value.get("resolution_status") or UNRESOLVED),
            supporting_evidence_refs=_text_list(value.get("supporting_evidence_refs") or []),
            contradicting_evidence_refs=_text_list(value.get("contradicting_evidence_refs") or []),
            last_attempt_status=str(value.get("last_attempt_status") or ""),
            last_route_id=str(value.get("last_route_id") or ""),
            first_resolved_evidence_version=_int(
                value.get("first_resolved_evidence_version"), 0
            ),
            last_updated_evidence_version=_int(
                value.get("last_updated_evidence_version"), 0
            ),
            update_count=_int(value.get("update_count"), 0),
            event_ids=_text_list(value.get("event_ids") or []),
            lifecycle_status=str(value.get("lifecycle_status") or CLAIM_ACTIVE),
            claim_revision=_int(value.get("claim_revision"), 0),
        )


def claim_key(
    *,
    entity_id: str,
    claim_id: str,
    contract_id: str,
    contract_version: str = "1",
) -> str:
    return "|".join(
        [
            str(entity_id or "").strip(),
            str(claim_id or "").strip(),
            str(contract_id or "").strip(),
            str(contract_version or "1").strip(),
        ]
    )


def normalize_ledger(value: Any) -> Dict[str, Dict[str, Any]]:
    if not value:
        return {}
    items: Iterable[Any]
    if isinstance(value, Mapping):
        items = value.values()
    else:
        items = value
    result: Dict[str, Dict[str, Any]] = {}
    for item in items or []:
        state = ClaimResolutionState.from_any(item)
        if not state:
            continue
        key = claim_key(
            entity_id=state.entity_id,
            claim_id=state.claim_id,
            contract_id=state.contract_id,
            contract_version=state.contract_version,
        )
        result[key] = state.to_dict()
    return result


class ClaimResolutionReducer:
    """Merge immutable claim match events into a case-level claim ledger."""

    def apply_event(
        self,
        ledger: Mapping[str, Any],
        event: ClaimMatchEvent,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        normalized = normalize_ledger(ledger)
        key = claim_key(
            entity_id=event.entity_id,
            claim_id=event.claim_id,
            contract_id=event.contract_id,
            contract_version=event.contract_version,
        )
        before = ClaimResolutionState.from_any(normalized.get(key)) or ClaimResolutionState(
            entity_id=event.entity_id,
            claim_id=event.claim_id,
            contract_id=event.contract_id,
            contract_version=event.contract_version,
        )
        after = ClaimResolutionState.from_any(before.to_dict())
        assert after is not None
        if event.event_id in set(after.event_ids):
            audit = self._audit(
                event,
                before,
                after,
                merge_decision="idempotent_replay",
                idempotent=True,
                conflict_created=False,
                version_delta=0,
            )
            return normalized, audit

        status = str(event.match_status or "").strip() or UNRESOLVED
        after.last_attempt_status = status
        after.last_route_id = str(event.source_route_id or "")
        after.event_ids = list(dict.fromkeys(list(after.event_ids) + [event.event_id]))
        after.update_count += 1

        merge_decision = "attempt_recorded"
        conflict_created = False
        changed = False
        resolution_delta = False
        if status == SUPPORTED:
            after.supporting_evidence_refs = _dedupe(
                list(after.supporting_evidence_refs)
                + list(event.supporting_evidence_refs or [])
            )
            if before.resolution_status == CONTRADICTED:
                after.resolution_status = CONFLICTED
                merge_decision = "conflict_created"
                conflict_created = True
            elif before.resolution_status != CONFLICTED:
                after.resolution_status = SUPPORTED
                merge_decision = "supported"
            changed = after.to_dict() != before.to_dict()
            resolution_delta = changed
        elif status == CONTRADICTED:
            after.contradicting_evidence_refs = _dedupe(
                list(after.contradicting_evidence_refs)
                + list(event.contradicting_evidence_refs or [])
            )
            if before.resolution_status == SUPPORTED:
                after.resolution_status = CONFLICTED
                merge_decision = "conflict_created"
                conflict_created = True
            elif before.resolution_status != CONFLICTED:
                after.resolution_status = CONTRADICTED
                merge_decision = "contradicted"
            changed = after.to_dict() != before.to_dict()
            resolution_delta = changed
        elif status in {NOT_ADDRESSED, NOT_APPLICABLE, INCONCLUSIVE}:
            after.resolution_status = before.resolution_status
            merge_decision = "route_attempt_only"
            changed = after.to_dict() != before.to_dict()
        else:
            after.resolution_status = before.resolution_status
            merge_decision = "unresolved_attempt_only"
            changed = after.to_dict() != before.to_dict()

        if after.resolution_status in {SUPPORTED, CONTRADICTED, CONFLICTED}:
            if not after.first_resolved_evidence_version:
                after.first_resolved_evidence_version = int(event.evidence_version or 0)
        if changed:
            after.last_updated_evidence_version = int(event.evidence_version or 0)
            after.claim_revision = int(after.claim_revision or 0) + 1
            normalized[key] = after.to_dict()
        audit = self._audit(
            event,
            before,
            after,
            merge_decision=merge_decision,
            idempotent=False,
            conflict_created=conflict_created,
            version_delta=1 if resolution_delta else 0,
            route_attempt_delta=1 if changed and not resolution_delta else 0,
        )
        return normalized, audit

    @staticmethod
    def _audit(
        event: ClaimMatchEvent,
        before: ClaimResolutionState,
        after: ClaimResolutionState,
        *,
        merge_decision: str,
        idempotent: bool,
        conflict_created: bool,
        version_delta: int,
        route_attempt_delta: int = 0,
    ) -> Dict[str, Any]:
        return {
            "event_id": event.event_id,
            "claim_key": claim_key(
                entity_id=event.entity_id,
                claim_id=event.claim_id,
                contract_id=event.contract_id,
                contract_version=event.contract_version,
            ),
            "source_route": event.source_route_id,
            "incoming_match_status": event.match_status,
            "resolution_before": before.to_dict(),
            "resolution_after": after.to_dict(),
            "supporting_refs_added": [
                ref for ref in after.supporting_evidence_refs
                if ref not in set(before.supporting_evidence_refs)
            ],
            "contradicting_refs_added": [
                ref for ref in after.contradicting_evidence_refs
                if ref not in set(before.contradicting_evidence_refs)
            ],
            "merge_decision": merge_decision,
            "idempotent_replay": bool(idempotent),
            "conflict_created": bool(conflict_created),
            "claim_state_version_delta": int(version_delta),
            "route_attempt_state_delta": int(route_attempt_delta),
            "reducer_version": CLAIM_REDUCER_VERSION,
        }


def materialize_candidate_claim_states(
    *,
    ledger: Mapping[str, Any],
    contract_views: Sequence[Mapping[str, Any]],
    active_entity_ids: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Ensure admitted candidates with claim contracts have case-level states."""

    normalized = normalize_ledger(ledger)
    active_set = {
        str(item or "").strip()
        for item in (active_entity_ids or [])
        if str(item or "").strip()
    }
    created: List[str] = []
    reactivated: List[str] = []
    inactivated: List[str] = []
    missing_contracts: List[Dict[str, Any]] = []
    view_audit: List[Dict[str, Any]] = []

    for view in contract_views or []:
        if not isinstance(view, Mapping):
            continue
        entity_id = str(view.get("entity_id") or "").strip()
        if not entity_id:
            continue
        contract = dict(view.get("claim_anchor_contract") or view)
        contract_id = str(
            contract.get("contract_id")
            or contract.get("anchor_contract_id")
            or f"claim_anchor_contract:{entity_id}"
        )
        contract_version = str(
            contract.get("contract_version")
            or contract.get("claim_closure_plan_version")
            or "1"
        )
        requirements = claim_requirements_from_contract(contract)
        if not requirements:
            missing_contracts.append(
                {
                    "entity_id": entity_id,
                    "candidate": view.get("candidate") or view.get("diagnosis") or "",
                    "reason": "NO_CLAIM_SCHEMA_AVAILABLE",
                    "clinical_admission_reasons": list(
                        view.get("clinical_admission_reasons") or []
                    ),
                }
            )
            continue
        for requirement in requirements:
            claim_id = str(requirement.get("claim_id") or "").strip()
            if not claim_id:
                continue
            key = claim_key(
                entity_id=entity_id,
                claim_id=claim_id,
                contract_id=contract_id,
                contract_version=contract_version,
            )
            before = ClaimResolutionState.from_any(normalized.get(key))
            if before is None:
                state = ClaimResolutionState(
                    entity_id=entity_id,
                    claim_id=claim_id,
                    contract_id=contract_id,
                    contract_version=contract_version,
                    lifecycle_status=CLAIM_ACTIVE,
                )
                normalized[key] = state.to_dict()
                created.append(key)
            elif before.lifecycle_status != CLAIM_ACTIVE:
                before.lifecycle_status = CLAIM_ACTIVE
                normalized[key] = before.to_dict()
                reactivated.append(key)
        view_audit.append(
            {
                "entity_id": entity_id,
                "candidate": view.get("candidate") or view.get("diagnosis") or "",
                "contract_id": contract_id,
                "contract_version": contract_version,
                "claim_ids": [
                    str(item.get("claim_id") or "")
                    for item in requirements
                    if str(item.get("claim_id") or "")
                ],
                "clinical_admission_reasons": list(
                    view.get("clinical_admission_reasons") or []
                ),
                "materialization_status": "materialized",
            }
        )

    if active_set:
        for key, raw in list(normalized.items()):
            state = ClaimResolutionState.from_any(raw)
            if not state or state.entity_id in active_set:
                continue
            if state.lifecycle_status == CLAIM_ACTIVE:
                state.lifecycle_status = CLAIM_INACTIVE
                normalized[key] = state.to_dict()
                inactivated.append(key)

    audit = {
        "contract_view_count": len(view_audit),
        "materialized_claim_state_count": len(created),
        "reactivated_claim_state_count": len(reactivated),
        "inactivated_claim_state_count": len(inactivated),
        "created_claim_keys": created,
        "reactivated_claim_keys": reactivated,
        "inactivated_claim_keys": inactivated,
        "missing_claim_contracts": missing_contracts,
        "candidate_claim_contract_views": view_audit,
        "claim_state_materialization_delta_count": (
            len(created) + len(reactivated) + len(inactivated)
        ),
    }
    return normalized, audit


def claim_requirements_from_contract(contract: Mapping[str, Any]) -> List[Dict[str, Any]]:
    requirements = [
        dict(item)
        for item in contract.get("claim_requirements", []) or []
        if isinstance(item, Mapping) and str(item.get("claim_id") or "").strip()
    ]
    if requirements:
        return requirements
    result: List[Dict[str, Any]] = []
    for claim_id in contract.get("required_claims", []) or []:
        text = str(claim_id or "").strip()
        if text:
            result.append({"claim_id": text, "required_for_anchor": True})
    for claim_id in contract.get("optional_claims", []) or []:
        text = str(claim_id or "").strip()
        if text:
            result.append({"claim_id": text, "required_for_anchor": False})
    return result


class GapClosureEvaluator:
    def evaluate(
        self,
        gap: Mapping[str, Any],
        ledger: Mapping[str, Any],
    ) -> Dict[str, Any]:
        requirements = _claim_requirements(gap)
        states_by_claim = _states_by_claim(ledger)
        resolved: List[str] = []
        remaining: List[str] = []
        contradicted: List[str] = []
        conflicted: List[str] = []
        claim_resolutions: List[Dict[str, Any]] = []
        required_claim_ids = {
            str(item.get("claim_id") or "")
            for item in requirements
            if _bool(item.get("required_for_anchor"), True)
        }
        for requirement in requirements:
            claim_id = str(requirement.get("claim_id") or "").strip()
            if not claim_id:
                continue
            state = states_by_claim.get(
                (
                    str(gap.get("entity_id") or gap.get("candidate_id") or ""),
                    claim_id,
                    _contract_id(gap),
                    _contract_version(gap),
                )
            )
            if state:
                payload = dict(state)
                claim_resolutions.append(payload)
                status = str(state.get("resolution_status") or UNRESOLVED)
            else:
                status = UNRESOLVED
            if status == SUPPORTED:
                resolved.append(claim_id)
            elif status == CONTRADICTED:
                contradicted.append(claim_id)
            elif status == CONFLICTED:
                conflicted.append(claim_id)
            elif claim_id in required_claim_ids:
                remaining.append(claim_id)

        if contradicted or conflicted:
            closure = BLOCKED_BY_CONTRADICTION
        elif required_claim_ids and required_claim_ids.issubset(set(resolved)):
            closure = FULLY_CLOSED
        elif resolved:
            closure = PARTIALLY_CLOSED
        else:
            closure = OPEN
        return {
            "gap_id": str(gap.get("gap_id") or ""),
            "entity_id": str(gap.get("entity_id") or ""),
            "contract_id": _contract_id(gap),
            "contract_version": _contract_version(gap),
            "gap_closure_level": closure,
            "resolved_claims": resolved,
            "remaining_claims": remaining,
            "contradicted_claims": contradicted,
            "conflicted_claims": conflicted,
            "claim_resolutions": claim_resolutions,
            "evaluator_version": GAP_CLOSURE_EVALUATOR_VERSION,
        }


class AnchorEvaluator:
    def evaluate(
        self,
        *,
        entity_id: str,
        anchor_contract: Mapping[str, Any],
        ledger: Mapping[str, Any],
        previous_status: str = "",
    ) -> Dict[str, Any]:
        contract_id = str(
            anchor_contract.get("contract_id")
            or anchor_contract.get("anchor_contract_id")
            or f"claim_anchor_contract:{entity_id}"
        )
        contract_version = str(anchor_contract.get("contract_version") or "1")
        required = [
            str(item or "").strip()
            for item in anchor_contract.get("required_claims", []) or []
            if str(item or "").strip()
        ]
        optional = [
            str(item or "").strip()
            for item in anchor_contract.get("optional_claims", []) or []
            if str(item or "").strip()
        ]
        states_by_claim = _states_by_claim(ledger)
        satisfied: List[str] = []
        unresolved: List[str] = []
        contradicted: List[str] = []
        conflicted: List[str] = []
        for claim_id in required + optional:
            state = states_by_claim.get((entity_id, claim_id, contract_id, contract_version))
            status = str((state or {}).get("resolution_status") or UNRESOLVED)
            if status == SUPPORTED:
                satisfied.append(claim_id)
            elif status == CONTRADICTED:
                contradicted.append(claim_id)
            elif status == CONFLICTED:
                conflicted.append(claim_id)
            elif claim_id in required:
                unresolved.append(claim_id)
        if conflicted:
            status = ANCHOR_CONFLICTED
        elif contradicted:
            status = ANCHOR_CONTRADICTED
        elif set(required).issubset(set(satisfied)):
            status = ANCHOR_SATISFIED
        elif satisfied:
            status = PATTERN_SUPPORTED_BUT_UNCONFIRMED
        else:
            status = ANCHOR_UNSATISFIED
        reasons: List[str] = []
        if status == ANCHOR_SATISFIED:
            reasons.append("ALL_REQUIRED_CLAIMS_SUPPORTED")
        if unresolved:
            reasons.append("REQUIRED_ANCHOR_CLAIMS_UNRESOLVED")
        if contradicted:
            reasons.append("REQUIRED_ANCHOR_CLAIMS_CONTRADICTED")
        if conflicted:
            reasons.append("REQUIRED_ANCHOR_CLAIMS_CONFLICTED")
        return {
            "entity_id": entity_id,
            "anchor_contract_id": contract_id,
            "contract_version": contract_version,
            "required_claims": required,
            "optional_claims": optional,
            "satisfied_claims": satisfied,
            "unresolved_claims": unresolved,
            "contradicted_claims": contradicted,
            "conflicted_claims": conflicted,
            "anchor_status_before": previous_status,
            "anchor_status_after": status,
            "reason_codes": reasons,
            "evaluator_version": ANCHOR_EVALUATOR_VERSION,
        }


class ClaimResolutionUpdater:
    def __init__(self) -> None:
        self.reducer = ClaimResolutionReducer()
        self.gap_evaluator = GapClosureEvaluator()

    def update_from_parse(
        self,
        *,
        ledger: Mapping[str, Any],
        parsed_result: Mapping[str, Any],
        intent_binding: Mapping[str, Any],
        gap_contract: Mapping[str, Any],
    ) -> Dict[str, Any]:
        current = normalize_ledger(ledger)
        events = claim_match_events_from_parse(
            parsed_result=parsed_result,
            intent_binding=intent_binding,
            gap_contract=gap_contract,
        )
        audits: List[Dict[str, Any]] = []
        resolvable = 0
        deltas = 0
        for event in events:
            if event.match_status in {SUPPORTED, CONTRADICTED}:
                resolvable += 1
            current, audit = self.reducer.apply_event(current, event)
            audits.append(audit)
            deltas += int(audit.get("claim_state_version_delta") or 0)
        gap_eval = self.gap_evaluator.evaluate(gap_contract, current)
        return {
            "ledger": current,
            "claim_match_events": [event.to_dict() for event in events],
            "claim_resolution_update_audit": audits,
            "gap_closure_evaluation": gap_eval,
            "claim_match_event_count": len(events),
            "resolvable_claim_match_count": resolvable,
            "persisted_claim_resolution_delta_count": deltas,
            "claim_resolution_writeback_missing": bool(resolvable and deltas == 0),
        }


def claim_match_events_from_parse(
    *,
    parsed_result: Mapping[str, Any],
    intent_binding: Mapping[str, Any],
    gap_contract: Mapping[str, Any],
) -> List[ClaimMatchEvent]:
    entity_id = str(
        intent_binding.get("entity_id")
        or gap_contract.get("entity_id")
        or gap_contract.get("candidate_id")
        or ""
    ).strip()
    contract_id = _contract_id(gap_contract, entity_id=entity_id)
    contract_version = _contract_version(gap_contract)
    source_route_ids = _route_by_claim(gap_contract)
    evidence_version = _int(intent_binding.get("source_evidence_version"), 0)
    source_exam = str(
        intent_binding.get("actual_result_exam")
        or intent_binding.get("resolved_exam")
        or intent_binding.get("requested_exam")
        or parsed_result.get("actual_result_exam")
        or ""
    )
    result_id = str(intent_binding.get("result_id") or "")
    binding_id = str(intent_binding.get("binding_id") or parsed_result.get("binding_id") or "")
    created_at = datetime.now(timezone.utc).isoformat()
    events: List[ClaimMatchEvent] = []
    for item in parsed_result.get("claim_matches", []) or []:
        if not isinstance(item, Mapping):
            continue
        claim_id = str(item.get("target_claim") or item.get("claim_id") or "").strip()
        if not claim_id:
            continue
        status = str(item.get("claim_status") or "").strip() or UNRESOLVED
        support = _text_list(
            item.get("supporting_evidence_refs")
            or item.get("supporting_observations")
            or []
        )
        contradict = _text_list(
            item.get("contradicting_evidence_refs")
            or item.get("contradicting_observations")
            or []
        )
        event_seed = {
            "result_id": result_id,
            "binding_id": binding_id,
            "entity_id": entity_id,
            "claim_id": claim_id,
            "contract_id": contract_id,
            "contract_version": contract_version,
            "status": status,
            "support": support,
            "contradict": contradict,
        }
        events.append(
            ClaimMatchEvent(
                event_id=_stable_id("claim_evt", event_seed),
                candidate_id=str(intent_binding.get("target_candidate") or ""),
                entity_id=entity_id,
                claim_id=claim_id,
                contract_id=contract_id,
                contract_version=contract_version,
                match_status=status,
                supporting_evidence_refs=support,
                contradicting_evidence_refs=contradict,
                source_type=str(item.get("source_type") or "exam_result"),
                source_route_id=(
                    source_route_ids.get(claim_id, "")
                    if str(item.get("source_type") or "") != "not_addressed_by_route"
                    else ""
                ),
                source_exam=source_exam,
                evidence_version=evidence_version,
                created_at=created_at,
            )
        )
    return events


def hydrate_gap_with_claim_state(
    gap: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = dict(gap or {})
    evaluation = GapClosureEvaluator().evaluate(payload, normalize_ledger(ledger))
    payload["claim_resolutions"] = list(evaluation.get("claim_resolutions") or [])
    payload["gap_closure_level"] = evaluation.get("gap_closure_level")
    payload["resolved_claims"] = list(evaluation.get("resolved_claims") or [])
    payload["remaining_claims"] = list(evaluation.get("remaining_claims") or [])
    payload["contradicted_claims"] = list(evaluation.get("contradicted_claims") or [])
    payload["conflicted_claims"] = list(evaluation.get("conflicted_claims") or [])
    payload["gap_closure_evaluation"] = evaluation
    return payload


def _claim_requirements(gap: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in gap.get("claim_requirements", []) or []
        if isinstance(item, Mapping) and str(item.get("claim_id") or "").strip()
    ]


def _states_by_claim(ledger: Mapping[str, Any]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    result: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for state in normalize_ledger(ledger).values():
        result[
            (
                str(state.get("entity_id") or ""),
                str(state.get("claim_id") or ""),
                str(state.get("contract_id") or ""),
                str(state.get("contract_version") or "1"),
            )
        ] = state
    return result


def _route_by_claim(gap: Mapping[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for route in gap.get("closure_routes", []) or []:
        if not isinstance(route, Mapping):
            continue
        route_id = str(route.get("route_id") or "")
        for claim_id in route.get("target_claims", []) or []:
            text = str(claim_id or "").strip()
            if text and text not in result:
                result[text] = route_id
    return result


def _contract_id(gap: Mapping[str, Any], *, entity_id: str = "") -> str:
    explicit = str(
        gap.get("contract_id")
        or gap.get("claim_contract_id")
        or gap.get("anchor_contract_id")
        or ""
    ).strip()
    if explicit:
        return explicit
    entity = str(entity_id or gap.get("entity_id") or gap.get("candidate_id") or "").strip()
    if entity:
        return f"claim_anchor_contract:{entity}"
    return str(gap.get("claim_closure_plan_version") or "claim_anchor_contract:unknown")


def _contract_version(gap: Mapping[str, Any]) -> str:
    return str(
        gap.get("contract_version")
        or gap.get("claim_closure_plan_version")
        or "1"
    )


def _stable_id(prefix: str, value: Any) -> str:
    data = value
    try:
        data = {
            key: value[key]
            for key in sorted(value)
        } if isinstance(value, Mapping) else value
        raw = repr(data).encode("utf-8")
    except Exception:
        raw = str(value).encode("utf-8", errors="ignore")
    return f"{prefix}_{hashlib.sha1(raw).hexdigest()[:16]}"


def _text_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        text = str(values).strip()
        return [text] if text else []
    result: List[str] = []
    try:
        iterator = iter(values)
    except TypeError:
        text = str(values).strip()
        return [text] if text else []
    for value in iterator:
        text = str(value or "").strip()
        if text:
            result.append(text)
    return _dedupe(result)


def _dedupe(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(str(item) for item in values if str(item or "").strip()))


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
