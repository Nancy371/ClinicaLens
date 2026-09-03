"""Candidate policy storage and generalization for runtime self-improvement.

This module keeps failed-case lessons out of the active PolicyStore until they
are generalized, validated by replay, and explicitly promoted.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TARGET_LAYERS = {
    "evidence_mapping",
    "candidate_recall",
    "eligibility",
    "ranking",
    "exam_selection",
    "submission",
}
POLICY_TYPES = {"general_rule", "case_hotfix"}
POLICY_STATUSES = {
    "candidate",
    "active",
    "quarantined",
    "rejected",
    "deprecated",
    "temporary",
}

PROMOTION_THRESHOLDS = {
    "target_fix_rate": 0.90,
    "neighboring_accuracy_delta": 0.0,
    "false_positive_increase": 0.01,
    "global_accuracy_delta": -0.002,
    "unsafe_submission_delta": 0.0,
}

PRIORITY_ORDER = {
    "safety_hard_constraint": 500,
    "evidence_qualification": 400,
    "differential": 300,
    "ranking": 200,
    "historical_preference": 100,
}


@dataclass
class FailureAttribution:
    failure_stage: str
    failure_type: str
    affected_candidate: str = ""
    root_cause: str = ""
    generalizable_pattern: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    source_case: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionDecision:
    promote_allowed: bool
    failed_gates: List[str]
    target_fix_rate: float = 0.0
    neighboring_accuracy_delta: float = 0.0
    false_positive_increase: float = 0.0
    global_accuracy_delta: float = 0.0
    unsafe_submission_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RuleGeneralizer:
    """Convert failure attribution into scoped policy candidates."""

    def generalize(
        self,
        attributions: Sequence[Dict[str, Any]],
        source_case: str = "",
    ) -> List[Dict[str, Any]]:
        policies: List[Dict[str, Any]] = []
        for attribution in attributions or []:
            if not isinstance(attribution, dict):
                continue
            policies.append(self._generalize_one(attribution, source_case=source_case))
        return policies

    def _generalize_one(
        self,
        attribution: Dict[str, Any],
        source_case: str = "",
    ) -> Dict[str, Any]:
        try:
            stage = _normalize_layer(
                attribution.get("failure_stage") or attribution.get("subsystem")
            )
        except ValueError:
            stage = "ranking"
        failure_type = str(attribution.get("failure_type") or attribution.get("signal") or "").strip()
        affected = str(attribution.get("affected_candidate") or "").strip()
        pattern = str(attribution.get("generalizable_pattern") or attribution.get("root_cause") or failure_type).strip()
        source_cases = _unique_texts(
            [source_case, attribution.get("source_case")]
            + _as_list(attribution.get("source_cases"))
        )
        if not source_cases:
            source_cases = ["legacy_runtime_patch"]
        policy_type = "case_hotfix" if attribution.get("policy_type") == "case_hotfix" else "general_rule"
        status = "temporary" if policy_type == "case_hotfix" else "candidate"
        action = self._action_for(stage, failure_type, affected, attribution)
        priority_class = _priority_class_for(stage, action)
        policy = {
            "policy_id": _mk_policy_id(stage, failure_type, pattern),
            "policy_type": policy_type,
            "target_layer": stage,
            "trigger_conditions": self._trigger_conditions(attribution, pattern, affected),
            "action": action,
            "applicable_scope": self._scope(attribution, affected),
            "excluded_scope": self._excluded_scope(attribution),
            "source_cases": source_cases,
            "validation_cases": [],
            "validation_metrics": {},
            "status": status,
            "promotion_allowed": False,
            "priority_class": priority_class,
            "priority": PRIORITY_ORDER[priority_class],
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "version": 1,
            "rollback_version": 0,
            "failure_attribution": dict(attribution),
        }
        if policy_type == "case_hotfix":
            policy["expires_after_days"] = int(attribution.get("expires_after_days") or 30)
        return normalize_policy_candidate(policy)

    @staticmethod
    def _trigger_conditions(
        attribution: Dict[str, Any],
        pattern: str,
        affected: str,
    ) -> List[str]:
        raw = _as_list(attribution.get("trigger_conditions"))
        if raw:
            return _unique_texts(raw)
        conditions = []
        if affected:
            conditions.append(f"affected_candidate:{affected}")
        if pattern:
            conditions.append(pattern)
        failure_type = str(attribution.get("failure_type") or attribution.get("signal") or "").strip()
        if failure_type:
            conditions.append(f"failure_type:{failure_type}")
        return _unique_texts(conditions)

    @staticmethod
    def _scope(attribution: Dict[str, Any], affected: str) -> List[str]:
        raw = _as_list(attribution.get("applicable_scope"))
        if raw:
            return _unique_texts(raw)
        stage = str(attribution.get("failure_stage") or attribution.get("subsystem") or "").strip()
        scope = [stage] if stage else []
        if affected:
            scope.append(f"candidate_family:{affected}")
        return _unique_texts(scope or ["general"])

    @staticmethod
    def _excluded_scope(attribution: Dict[str, Any]) -> List[str]:
        raw = _as_list(attribution.get("excluded_scope"))
        if raw:
            return _unique_texts(raw)
        return [
            "single-case answer memorization",
            "disease-name score boost without required evidence",
        ]

    @staticmethod
    def _action_for(
        stage: str,
        failure_type: str,
        affected: str,
        attribution: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw = attribution.get("action")
        if isinstance(raw, dict) and raw:
            return raw
        if stage == "eligibility":
            return {
                "set_status": "Deferred",
                "require_evidence_review": True,
                "do_not_final_submit_until_primary_eligible": True,
            }
        if stage == "submission":
            return {
                "block_final_when_ineligible": True,
                "require_primary_eligible": True,
            }
        if stage == "exam_selection":
            return {
                "generate_adjudication_exams": True,
                "prioritize_missing_anchors": True,
            }
        if stage == "evidence_mapping":
            return {
                "require_source_text_review": True,
                "do_not_use_reasoning_as_required_anchor": True,
            }
        if stage == "candidate_recall":
            return {
                "broaden_recall_for_pattern": True,
                "keep_as_differential_until_eligible": True,
            }
        return {
            "adjust_ranking_only_after_eligibility": True,
            "do_not_boost_disease_name": True,
        }


class CandidatePolicyStore:
    """Runtime candidate-policy store.

    Candidates are auditable lessons. They are not injected into normal agent
    behavior unless promoted into the active PolicyStore or explicitly replayed.
    """

    def __init__(self, path: str = "outputs/runtime_state/candidate_policies.json"):
        self.path = path
        self.policies: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, TypeError, ValueError):
            self.policies = []
            return
        raw = data.get("policies") if isinstance(data, dict) else data
        self.policies = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            try:
                self.policies.append(normalize_policy_candidate(item))
            except ValueError:
                continue

    def _save(self) -> None:
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        temp_path = f"{self.path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        data = {"schema_version": 1, "policies": self.policies}
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temp_path, self.path)
        except OSError:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def upsert_candidate(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        candidate = normalize_policy_candidate(policy)
        conflict = self.find_conflict(candidate)
        if conflict:
            candidate["status"] = "quarantined"
            candidate["promotion_allowed"] = False
            candidate["conflict"] = conflict
        existing = self._find_similar(candidate)
        now = _now_iso()
        if existing:
            existing["source_cases"] = _unique_texts(
                list(existing.get("source_cases") or []) + list(candidate.get("source_cases") or [])
            )
            existing["trigger_conditions"] = _unique_texts(
                list(existing.get("trigger_conditions") or [])
                + list(candidate.get("trigger_conditions") or [])
            )
            existing["updated_at"] = now
            existing["version"] = int(existing.get("version", 1) or 1) + 1
            if candidate.get("status") == "quarantined":
                existing["status"] = "quarantined"
                existing["conflict"] = candidate.get("conflict")
            self._save()
            return existing
        self.policies.append(candidate)
        self._save()
        return candidate

    def upsert_many(self, policies: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        added = 0
        updated = 0
        quarantined = 0
        ids: List[str] = []
        before = {item.get("policy_id"): int(item.get("version", 1) or 1) for item in self.policies}
        for policy in policies or []:
            item = self.upsert_candidate(policy)
            ids.append(str(item.get("policy_id") or ""))
            if item.get("status") == "quarantined":
                quarantined += 1
            old_version = before.get(item.get("policy_id"))
            if old_version is None:
                added += 1
            elif int(item.get("version", 1) or 1) > old_version:
                updated += 1
        return {
            "candidate": added,
            "updated": updated,
            "quarantined": quarantined,
            "policy_ids": [item for item in ids if item],
        }

    def find_conflict(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        for existing in self.policies:
            if existing.get("policy_id") == policy.get("policy_id"):
                continue
            if existing.get("target_layer") != policy.get("target_layer"):
                continue
            if not _overlaps(existing.get("trigger_conditions"), policy.get("trigger_conditions")):
                continue
            if _actions_conflict(existing.get("action"), policy.get("action")):
                return {
                    "conflict_type": "overlapping_trigger_opposite_action",
                    "existing_policy_id": existing.get("policy_id"),
                    "existing_status": existing.get("status"),
                }
            if int(existing.get("priority", 0) or 0) == int(policy.get("priority", 0) or 0):
                return {
                    "conflict_type": "overlapping_trigger_same_priority",
                    "existing_policy_id": existing.get("policy_id"),
                    "existing_status": existing.get("status"),
                }
        return {}

    def record_validation(
        self,
        policy_id: str,
        metrics: Dict[str, Any],
        validation_cases: Optional[Sequence[str]] = None,
    ) -> PromotionDecision:
        policy = self.get(policy_id)
        if policy is None:
            raise ValueError(f"policy candidate {policy_id!r} does not exist")
        decision = promotion_decision(metrics)
        policy["validation_metrics"] = dict(metrics or {})
        policy["validation_cases"] = _unique_texts(
            list(policy.get("validation_cases") or []) + list(validation_cases or [])
        )
        policy["promotion_allowed"] = decision.promote_allowed
        policy["updated_at"] = _now_iso()
        if not decision.promote_allowed and policy.get("status") == "active":
            policy["status"] = "quarantined"
        self._save()
        return decision

    def promote(self, policy_id: str, active_store: Any = None) -> Dict[str, Any]:
        policy = self.get(policy_id)
        if policy is None:
            raise ValueError(f"policy candidate {policy_id!r} does not exist")
        metrics = dict(policy.get("validation_metrics") or {})
        decision = promotion_decision(metrics)
        if not decision.promote_allowed:
            policy["promotion_allowed"] = False
            policy["status"] = "quarantined"
            policy["promotion_decision"] = decision.to_dict()
            self._save()
            return policy
        conflict = self.find_conflict(policy)
        if conflict:
            policy["promotion_allowed"] = False
            policy["status"] = "quarantined"
            policy["conflict"] = conflict
            self._save()
            return policy
        policy["status"] = "active"
        policy["promotion_allowed"] = True
        policy["promoted_at"] = _now_iso()
        policy["promotion_decision"] = decision.to_dict()
        if active_store is not None and hasattr(active_store, "upsert_policy_candidate"):
            active_store.upsert_policy_candidate(policy)
        self._save()
        return policy

    def quarantine(self, policy_id: str, reason: str = "") -> bool:
        policy = self.get(policy_id)
        if policy is None:
            return False
        policy["status"] = "quarantined"
        policy["quarantine_reason"] = reason
        policy["promotion_allowed"] = False
        policy["updated_at"] = _now_iso()
        self._save()
        return True

    def rollback(self, policy_id: str) -> bool:
        policy = self.get(policy_id)
        if policy is None:
            return False
        policy["status"] = "quarantined"
        policy["version"] = int(policy.get("rollback_version", 0) or 0) or max(
            1,
            int(policy.get("version", 1) or 1) - 1,
        )
        policy["updated_at"] = _now_iso()
        self._save()
        return True

    def deprecate_inactive(self, max_age_days: int = 90) -> Dict[str, int]:
        now = time.time()
        deprecated = 0
        for policy in self.policies:
            if policy.get("status") not in {"candidate", "quarantined"}:
                continue
            last = _parse_iso_seconds(policy.get("updated_at") or policy.get("created_at"))
            if last and (now - last) >= max_age_days * 86400:
                policy["status"] = "deprecated"
                policy["promotion_allowed"] = False
                deprecated += 1
        if deprecated:
            self._save()
        return {"deprecated": deprecated}

    def get(self, policy_id: str) -> Optional[Dict[str, Any]]:
        for item in self.policies:
            if item.get("policy_id") == policy_id:
                return item
        return None

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        conflicts = 0
        stages: Dict[str, int] = {}
        for policy in self.policies:
            status = str(policy.get("status") or "candidate")
            counts[status] = counts.get(status, 0) + 1
            layer = str(policy.get("target_layer") or "unknown")
            stages[layer] = stages.get(layer, 0) + 1
            if policy.get("conflict"):
                conflicts += 1
        return {
            "candidate_policy_count": counts.get("candidate", 0),
            "policy_promotion_count": counts.get("active", 0),
            "policy_quarantine_count": counts.get("quarantined", 0),
            "policy_rejected_count": counts.get("rejected", 0),
            "policy_conflict_count": conflicts,
            "failure_stage_distribution": stages,
            "status_distribution": counts,
        }

    def ingest_legacy_patches(self, patches: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        converted = 0
        quarantined = 0
        generalizer = RuleGeneralizer()
        for patch in patches or []:
            if not isinstance(patch, dict):
                continue
            stats = patch.get("stats") or {}
            status = str(stats.get("status") or patch.get("status") or "shadow")
            source = patch.get("source") or {}
            action_text = json.dumps(patch.get("action") or "", ensure_ascii=False)
            unsafe_gap = "required_gap_authorized" in action_text or "required_gap_authorized" in json.dumps(source, ensure_ascii=False)
            attribution = {
                "failure_stage": _legacy_layer(patch.get("type")),
                "failure_type": str(source.get("signal") or patch.get("type") or "legacy_patch"),
                "root_cause": str(patch.get("action") or source.get("signal") or "legacy runtime patch"),
                "generalizable_pattern": str(patch.get("action") or source.get("signal") or "legacy runtime patch"),
                "source_cases": _as_list(source.get("source_cases")),
            }
            candidate = generalizer.generalize([attribution])[0]
            candidate["legacy_patch_id"] = patch.get("id")
            if status == "active" and not unsafe_gap:
                candidate["status"] = "active"
                candidate["promotion_allowed"] = True
            elif unsafe_gap:
                candidate["status"] = "quarantined"
                candidate["promotion_allowed"] = False
                candidate["quarantine_reason"] = "required_gap_authorized policies cannot be promoted"
                quarantined += 1
            else:
                candidate["status"] = "candidate"
                candidate["promotion_allowed"] = False
            self.upsert_candidate(candidate)
            converted += 1
        return {"converted": converted, "quarantined": quarantined}

    def _find_similar(self, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = _policy_key(candidate)
        for existing in self.policies:
            if _policy_key(existing) == key:
                return existing
        return None


def normalize_policy_candidate(policy: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        raise ValueError("policy candidate must be a dict")
    out = dict(policy)
    out["policy_type"] = str(out.get("policy_type") or "general_rule")
    if out["policy_type"] not in POLICY_TYPES:
        raise ValueError(f"invalid policy_type {out['policy_type']!r}")
    out["target_layer"] = _normalize_layer(out.get("target_layer"))
    out["status"] = str(out.get("status") or "candidate")
    if out["status"] not in POLICY_STATUSES:
        raise ValueError(f"invalid status {out['status']!r}")
    out["trigger_conditions"] = _unique_texts(_as_list(out.get("trigger_conditions")))
    out["applicable_scope"] = _unique_texts(_as_list(out.get("applicable_scope"))) or ["general"]
    out["excluded_scope"] = _unique_texts(_as_list(out.get("excluded_scope")))
    out["source_cases"] = _unique_texts(_as_list(out.get("source_cases")))
    out["validation_cases"] = _unique_texts(_as_list(out.get("validation_cases")))
    if not isinstance(out.get("action"), dict) or not out.get("action"):
        raise ValueError("policy candidate requires structured action")
    if not out["trigger_conditions"]:
        raise ValueError("policy candidate requires trigger_conditions")
    if not out["excluded_scope"]:
        raise ValueError("policy candidate requires excluded_scope")
    if not out["source_cases"]:
        raise ValueError("policy candidate requires source_cases")
    if not out.get("policy_id"):
        out["policy_id"] = _mk_policy_id(
            out["target_layer"],
            ",".join(out["trigger_conditions"]),
            json.dumps(out["action"], sort_keys=True, ensure_ascii=False),
        )
    out.setdefault("validation_metrics", {})
    out.setdefault("promotion_allowed", False)
    out.setdefault("created_at", _now_iso())
    out["updated_at"] = out.get("updated_at") or out["created_at"]
    out["version"] = int(out.get("version", 1) or 1)
    out["rollback_version"] = int(out.get("rollback_version", max(0, out["version"] - 1)) or 0)
    out["priority_class"] = str(out.get("priority_class") or _priority_class_for(out["target_layer"], out["action"]))
    out["priority"] = int(out.get("priority", PRIORITY_ORDER.get(out["priority_class"], 100)) or 100)
    if out["policy_type"] == "case_hotfix":
        out["expires_after_days"] = int(out.get("expires_after_days") or 30)
        if out["status"] == "candidate":
            out["status"] = "temporary"
    return out


def promotion_decision(metrics: Dict[str, Any]) -> PromotionDecision:
    metrics = metrics or {}
    target = _float(metrics.get("target_fix_rate"))
    neighboring = _float(metrics.get("neighboring_accuracy_delta"))
    false_positive = _float(metrics.get("false_positive_increase"))
    global_delta = _float(metrics.get("global_accuracy_delta"))
    unsafe = _float(metrics.get("unsafe_submission_delta"))
    failed: List[str] = []
    if target < PROMOTION_THRESHOLDS["target_fix_rate"]:
        failed.append("target_fix_rate")
    if neighboring < PROMOTION_THRESHOLDS["neighboring_accuracy_delta"]:
        failed.append("neighboring_accuracy_delta")
    if false_positive > PROMOTION_THRESHOLDS["false_positive_increase"]:
        failed.append("false_positive_increase")
    if global_delta < PROMOTION_THRESHOLDS["global_accuracy_delta"]:
        failed.append("global_accuracy_delta")
    if unsafe > PROMOTION_THRESHOLDS["unsafe_submission_delta"]:
        failed.append("unsafe_submission_delta")
    return PromotionDecision(
        promote_allowed=not failed,
        failed_gates=failed,
        target_fix_rate=target,
        neighboring_accuracy_delta=neighboring,
        false_positive_increase=false_positive,
        global_accuracy_delta=global_delta,
        unsafe_submission_delta=unsafe,
    )


def _normalize_layer(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "inquiry": "candidate_recall",
        "reasoning": "ranking",
        "examination": "exam_selection",
        "boundary": "submission",
        "treatment": "submission",
        "diagnosis": "ranking",
    }
    text = aliases.get(text, text)
    if text not in TARGET_LAYERS:
        raise ValueError(f"invalid target_layer {text!r}")
    return text


def _legacy_layer(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"exam_mandatory", "exam_prune"}:
        return "exam_selection"
    if text == "inquiry_deepen":
        return "candidate_recall"
    if text == "treatment_personalize":
        return "submission"
    return "ranking"


def _priority_class_for(layer: str, action: Dict[str, Any]) -> str:
    if action.get("block_final_when_ineligible") or action.get("require_primary_eligible"):
        return "safety_hard_constraint"
    if layer in {"eligibility", "evidence_mapping", "submission"}:
        return "evidence_qualification"
    if layer in {"candidate_recall", "exam_selection"}:
        return "differential"
    if layer == "ranking":
        return "ranking"
    return "historical_preference"


def _policy_key(policy: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(policy.get("policy_type") or ""),
        str(policy.get("target_layer") or ""),
        "|".join(sorted(str(item) for item in policy.get("trigger_conditions") or [])),
        json.dumps(policy.get("action") or {}, sort_keys=True, ensure_ascii=False),
    )


def _overlaps(left: Any, right: Any) -> bool:
    left_set = set(_unique_texts(_as_list(left)))
    right_set = set(_unique_texts(_as_list(right)))
    if not left_set or not right_set:
        return False
    return bool(left_set & right_set)


def _actions_conflict(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    left_text = json.dumps(left, sort_keys=True, ensure_ascii=False)
    right_text = json.dumps(right, sort_keys=True, ensure_ascii=False)
    opposite_pairs = (
        ("block_final", "allow_final"),
        ("Deferred", "PrimaryEligible"),
        ("require_primary_eligible", "required_gap_authorized"),
        ("do_not_boost_disease_name", "boost_disease"),
    )
    return any(a in left_text and b in right_text or b in left_text and a in right_text for a, b in opposite_pairs)


def _mk_policy_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts if str(part).strip())
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"POLICY_{digest}"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _parse_iso_seconds(value: Any) -> float:
    text = str(value or "")
    if not text:
        return 0.0
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(text[:19], fmt))
        except (TypeError, ValueError):
            continue
    return 0.0


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        return [value] if value else []
    return [value]


def _unique_texts(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
