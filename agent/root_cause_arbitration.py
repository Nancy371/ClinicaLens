"""Root-cause arbitration for primary/secondary diagnosis ordering."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .diagnosis_eligibility import PRIMARY_ELIGIBLE


_OBJECTIVE_FINDINGS = {
    "afb_positive",
    "alp_elevated",
    "anca_positive",
    "av_block",
    "bone_deformity",
    "echo_vsd",
    "egfr_low",
    "heart_failure_state",
    "hypocalcemia",
    "low_magnesium",
    "low_urine_magnesium",
    "magnesium_depletion",
    "magnesium_load_retention_high",
    "mitral_regurgitation",
    "mpo_anca_positive",
    "pulmonary_valve_gradient",
    "pulmonary_valve_stenosis",
    "renal_impairment",
    "second_degree_av_block",
    "tb_naat_positive",
    "tricuspid_regurgitation",
    "valve_gradient_high",
    "ventricular_septal_defect",
    "vitamin_d_low",
}


@dataclass
class RootCauseArbitrationResult:
    applied: bool = False
    primary_before: str = ""
    primary_after: str = ""
    primary_override: bool = False
    root_cause_primary: str = ""
    root_cause_secondary: List[str] = field(default_factory=list)
    root_cause_final_secondary: List[str] = field(default_factory=list)
    candidate_explanation_edges: List[Dict[str, Any]] = field(default_factory=list)
    root_cause_coverage: float = 0.0
    reason: str = ""
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied": bool(self.applied),
            "primary_before": self.primary_before,
            "primary_after": self.primary_after,
            "primary_override": bool(self.primary_override),
            "root_cause_primary": self.root_cause_primary,
            "root_cause_secondary": list(self.root_cause_secondary),
            "root_cause_final_secondary": list(self.root_cause_final_secondary),
            "candidate_explanation_edges": list(self.candidate_explanation_edges),
            "root_cause_coverage": round(float(self.root_cause_coverage or 0.0), 4),
            "reason": self.reason,
            "audit": dict(self.audit),
        }


class RootCauseArbiter:
    """Choose the upstream diagnosis that best explains downstream candidates."""

    def __init__(self, knowledge: Any, ref_dir: str = "data/ref_data"):
        self.knowledge = knowledge
        self.ref_dir = ref_dir
        self.path = os.path.join(ref_dir, "root_cause_relations.json")
        self.relations = self._load_relations()

    def arbitrate(
        self,
        judge_decision: Any,
        candidates: Sequence[Any],
        *,
        mechanism_hypotheses: Optional[Sequence[Dict[str, Any]]] = None,
        max_final_diagnoses: int = 3,
    ) -> RootCauseArbitrationResult:
        primary_before = str(
            getattr(judge_decision, "primary", "")
            or getattr(judge_decision, "judge_primary", "")
            or ""
        )
        result = RootCauseArbitrationResult(
            primary_before=primary_before,
            primary_after=primary_before,
        )
        if not primary_before or not candidates or not self.relations:
            return result

        by_name = {self._name(item): item for item in candidates if self._name(item)}
        current_primary = by_name.get(primary_before)
        edges = self._candidate_edges(candidates)
        if not edges:
            result.audit = {"reason": "no_root_cause_edges"}
            self._write_empty_candidate_roles(candidates)
            return result

        current_coverage = self._root_candidate_coverage(current_primary, [])
        mechanism_ids = {
            str(item.get("mechanism_id") or "")
            for item in mechanism_hypotheses or []
            if isinstance(item, dict) and str(item.get("mechanism_id") or "")
        }
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for edge in edges:
            grouped.setdefault(str(edge.get("source") or ""), []).append(edge)

        contenders: List[Tuple[float, Any, List[Dict[str, Any]]]] = []
        for source_name, source_edges in grouped.items():
            source = by_name.get(source_name)
            if source is None or source_name == "":
                continue
            explains_current_primary = any(
                str(edge.get("target") or "") == primary_before for edge in source_edges
            )
            min_explained = min(
                int(edge.get("min_explained_candidates", 1) or 1)
                for edge in source_edges
            )
            is_current_primary = source_name == primary_before
            required_explained = min_explained if is_current_primary else max(2, min_explained)
            if not explains_current_primary and len(source_edges) < required_explained:
                continue
            coverage = self._root_candidate_coverage(source, source_edges)
            if coverage + 0.04 < current_coverage and explains_current_primary:
                continue
            if coverage < 0.62:
                continue
            if not self._mechanism_supported(source_edges, mechanism_ids):
                continue
            contenders.append((coverage, source, source_edges))

        if not contenders:
            result.candidate_explanation_edges = edges
            result.root_cause_coverage = round(current_coverage, 4)
            result.audit = {
                "reason": "no_eligible_root_cause_contender",
                "current_primary_coverage": round(current_coverage, 4),
            }
            self._write_candidate_roles(candidates, "", [], edges, current_coverage)
            return result

        contenders.sort(
            key=lambda item: (
                item[0],
                len(item[2]),
                float(getattr(item[1], "core_explanatory_coverage", 0.0) or 0.0),
                float(getattr(item[1], "score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        coverage, root, root_edges = contenders[0]
        root_name = self._name(root)
        audit_secondary = self._secondary_from_edges(
            root_edges,
            by_name,
            max_final_diagnoses,
            submit_only=False,
        )
        final_secondary = self._secondary_from_edges(
            root_edges,
            by_name,
            max_final_diagnoses,
            submit_only=True,
        )
        if not audit_secondary and root_name != primary_before:
            result.candidate_explanation_edges = edges
            result.root_cause_coverage = round(coverage, 4)
            result.audit = {"reason": "root_cause_has_no_explained_downstream"}
            self._write_candidate_roles(candidates, "", [], edges, coverage)
            return result

        blocked_secondary = set(audit_secondary) - set(final_secondary)
        existing_final = [
            name
            for name in list(getattr(judge_decision, "final_diagnoses", []) or [])
            if name not in blocked_secondary
        ]
        final_names = self._final_names(
            root_name,
            final_secondary,
            existing_final,
            max_final_diagnoses,
        )
        final_secondary = [name for name in final_names if name != root_name]
        self._write_candidate_roles(
            candidates,
            root_name,
            audit_secondary,
            edges,
            coverage,
        )

        result.applied = bool(root_name and final_names)
        result.primary_after = root_name or primary_before
        result.primary_override = bool(root_name and root_name != primary_before)
        result.root_cause_primary = root_name
        result.root_cause_secondary = list(audit_secondary)
        result.root_cause_final_secondary = list(final_secondary)
        result.candidate_explanation_edges = root_edges
        result.root_cause_coverage = round(coverage, 4)
        result.reason = (
            "root cause explains downstream candidate evidence"
            if result.applied
            else ""
        )
        result.audit = {
            "current_primary_coverage": round(current_coverage, 4),
            "all_candidate_explanation_edges": edges,
        }
        return result

    def apply_to_judge_decision(
        self,
        judge_decision: Any,
        result: RootCauseArbitrationResult,
        *,
        max_final_diagnoses: int = 3,
    ) -> Any:
        if not judge_decision or not result:
            return judge_decision
        payload = result.to_dict()
        setattr(judge_decision, "root_cause_arbitration", payload)
        setattr(judge_decision, "root_cause_primary", result.root_cause_primary)
        setattr(judge_decision, "root_cause_secondary", list(result.root_cause_secondary))
        setattr(
            judge_decision,
            "candidate_explanation_edges",
            list(result.candidate_explanation_edges),
        )
        if not result.applied:
            return judge_decision

        primary = result.root_cause_primary
        blocked_secondary = set(result.root_cause_secondary) - set(
            result.root_cause_final_secondary
        )
        existing_final = [
            name
            for name in list(getattr(judge_decision, "final_diagnoses", []) or [])
            if name not in blocked_secondary
        ]
        final = self._final_names(
            primary,
            result.root_cause_final_secondary,
            existing_final,
            max_final_diagnoses,
        )
        setattr(judge_decision, "primary", primary)
        setattr(judge_decision, "judge_primary", primary)
        setattr(judge_decision, "locked_primary", primary)
        setattr(judge_decision, "provisional_primary", "")
        setattr(judge_decision, "primary_status", "locked")
        setattr(judge_decision, "needs_discriminating_exams", False)
        setattr(judge_decision, "defer_reason", "")
        setattr(judge_decision, "secondary", [name for name in final if name != primary])
        setattr(judge_decision, "final_diagnoses", final)
        setattr(judge_decision, "root_cause_primary_override", result.primary_override)
        setattr(judge_decision, "primary_override_source", "root_cause_arbitration")
        setattr(judge_decision, "root_cause_coverage", result.root_cause_coverage)
        setattr(judge_decision, "required_gap_authorized_diagnoses", [])
        trace = list(getattr(judge_decision, "dynamic_rerank_trace", []) or [])
        trace.append(
            {
                "stage": "root_cause_arbitration",
                "primary_before": result.primary_before,
                "primary_after": result.primary_after,
                "primary_override": bool(result.primary_override),
                "root_cause_secondary": list(result.root_cause_secondary),
                "root_cause_final_secondary": list(result.root_cause_final_secondary),
                "root_cause_coverage": result.root_cause_coverage,
                "candidate_explanation_edges": list(result.candidate_explanation_edges),
            }
        )
        setattr(judge_decision, "dynamic_rerank_trace", trace)
        return judge_decision

    def _candidate_edges(self, candidates: Sequence[Any]) -> List[Dict[str, Any]]:
        edges: List[Dict[str, Any]] = []
        for relation in self.relations:
            upstream_selector = relation.get("upstream_selector") or {}
            downstream_selector = relation.get("downstream_selector") or {}
            for source in candidates:
                if not self._root_ready(source, upstream_selector):
                    continue
                if not self._matches_selector(source, upstream_selector):
                    continue
                for target in candidates:
                    if source is target:
                        continue
                    if not self._downstream_ready(target, downstream_selector):
                        continue
                    if not self._matches_selector(target, downstream_selector):
                        continue
                    edge = self._edge(source, target, relation)
                    if edge:
                        edges.append(edge)
        return self._dedupe_edges(edges)

    def _edge(
        self,
        source: Any,
        target: Any,
        relation: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_selector = relation.get("upstream_selector") or {}
        target_selector = relation.get("downstream_selector") or {}
        source_hits = self._selector_hits(source, source_selector)
        target_hits = self._selector_hits(target, target_selector)
        relation_weight = float(relation.get("relation_weight", 0.24) or 0.24)
        target_coverage = self._candidate_signal(target)
        edge_coverage = min(1.0, target_coverage + relation_weight)
        return {
            "relation_id": str(relation.get("relation_id") or ""),
            "mechanism_id": str(relation.get("mechanism_id") or ""),
            "source": self._name(source),
            "target": self._name(target),
            "target_role": str(relation.get("target_role") or "secondary"),
            "matched_upstream_findings": source_hits,
            "matched_downstream_findings": target_hits,
            "downstream_coverage": round(target_coverage, 4),
            "edge_coverage": round(edge_coverage, 4),
            "authorize_required_gap": bool(
                relation.get("authorize_required_gap")
                or (relation.get("upstream_selector") or {}).get("authorize_required_gap")
            ),
            "min_explained_candidates": int(
                relation.get("min_explained_candidates", 1) or 1
            ),
            "secondary_policy": str(
                relation.get("secondary_policy") or "submit_if_objective"
            ),
            "submit_as_final": str(
                relation.get("secondary_policy") or "submit_if_objective"
            )
            != "audit_only",
            "audit_source": str(relation.get("audit_source") or ""),
        }

    def _root_ready(self, candidate: Any, selector: Dict[str, Any]) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        status = str(getattr(candidate, "eligibility_status", "") or "")
        if status and status != PRIMARY_ELIGIBLE:
            return False
        if getattr(candidate, "unresolved_evidence_conflict", False):
            return False
        if not getattr(candidate, "matched_evidence", None):
            return False
        required_gaps = list(getattr(candidate, "required_gaps", []) or [])
        if required_gaps:
            return False
        elif not bool(getattr(candidate, "required_met", False)):
            return False
        return self._objective_or_core(candidate, selector)

    def _downstream_ready(self, candidate: Any, selector: Dict[str, Any]) -> bool:
        if not candidate or getattr(candidate, "hard_contradiction", False):
            return False
        if getattr(candidate, "unresolved_evidence_conflict", False):
            return False
        if not getattr(candidate, "matched_evidence", None):
            return False
        return self._objective_or_core(candidate, selector)

    def _objective_or_core(self, candidate: Any, selector: Dict[str, Any]) -> bool:
        components = getattr(candidate, "component_scores", {}) or {}
        if float(components.get("objective_evidence", 0.0) or 0.0) >= 1.0:
            return True
        if float(getattr(candidate, "diagnostic_evidence_score", 0.0) or 0.0) >= 0.30:
            return True
        if float(getattr(candidate, "core_evidence_score", 0.0) or 0.0) >= 0.30:
            return True
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        if matched & _OBJECTIVE_FINDINGS:
            return True
        required = set(str(item) for item in selector.get("required_any_findings", []) or [])
        if required and matched & required:
            return True
        return len(set(getattr(candidate, "core_matched_evidence", []) or [])) >= 2

    def _matches_selector(self, candidate: Any, selector: Dict[str, Any]) -> bool:
        if not selector:
            return True
        name = self._name(candidate)
        entry = self._entry(candidate)
        diagnosis_names = self._normalized_names(
            selector.get("diagnosis_names") or selector.get("names") or []
        )
        if diagnosis_names and name not in diagnosis_names:
            return False
        diagnosis_types = {
            str(item).strip().lower()
            for item in selector.get("diagnosis_types", []) or []
            if str(item).strip()
        }
        dtype = str(
            getattr(candidate, "diagnosis_type", "")
            or entry.get("diagnosis_type")
            or ""
        ).lower()
        if diagnosis_types and dtype not in diagnosis_types:
            return False
        body_systems = {
            str(item).strip()
            for item in selector.get("body_systems", []) or []
            if str(item).strip()
        }
        body = str(entry.get("body_system") or "")
        if body_systems and body not in body_systems:
            return False
        families = {
            str(item).strip()
            for item in (
                selector.get("families")
                or selector.get("family_ids")
                or selector.get("disease_families")
                or []
            )
            if str(item).strip()
        }
        family = str(entry.get("disease_family") or entry.get("family") or "")
        if families and family not in families:
            return False
        categories = {
            str(item).strip()
            for item in selector.get("categories", []) or []
            if str(item).strip()
        }
        category = str(entry.get("category") or "")
        if categories and category not in categories:
            return False
        required_any = {
            str(item).strip()
            for item in selector.get("required_any_findings", []) or []
            if str(item).strip()
        }
        if required_any and not (self._matched(candidate) & required_any):
            return False
        required_all = {
            str(item).strip()
            for item in selector.get("required_all_findings", []) or []
            if str(item).strip()
        }
        if required_all and not required_all.issubset(self._matched(candidate)):
            return False
        return self._passes_numeric_selector(candidate, selector)

    def _passes_numeric_selector(self, candidate: Any, selector: Dict[str, Any]) -> bool:
        checks = [
            ("min_core_coverage", "core_explanatory_coverage"),
            ("min_coverage", "coverage_score"),
            ("min_core_evidence_score", "core_evidence_score"),
            ("min_diagnostic_evidence_score", "diagnostic_evidence_score"),
        ]
        for selector_key, candidate_key in checks:
            if selector.get(selector_key) is None:
                continue
            try:
                expected = float(selector.get(selector_key) or 0.0)
                value = float(getattr(candidate, candidate_key, 0.0) or 0.0)
            except (TypeError, ValueError):
                return False
            if value < expected:
                return False
        min_matched = int(selector.get("min_matched_findings", 0) or 0)
        return not min_matched or len(self._matched(candidate)) >= min_matched

    def _selector_hits(self, candidate: Any, selector: Dict[str, Any]) -> List[str]:
        watched = list(selector.get("required_any_findings") or []) + list(
            selector.get("required_all_findings") or []
        )
        matched = self._matched(candidate)
        hits = [str(item) for item in watched if str(item) in matched]
        if hits:
            return list(dict.fromkeys(hits))[:8]
        return list(dict.fromkeys(list(matched)))[:8]

    def _secondary_from_edges(
        self,
        edges: Sequence[Dict[str, Any]],
        by_name: Dict[str, Any],
        max_final_diagnoses: int,
        *,
        submit_only: bool,
    ) -> List[str]:
        limit = max(0, int(max_final_diagnoses or 1) - 1)
        scored: List[Tuple[float, str]] = []
        for edge in edges:
            if str(edge.get("target_role") or "secondary") != "secondary":
                continue
            name = str(edge.get("target") or "")
            target = by_name.get(name)
            if not target or not self._downstream_ready(target, {}):
                continue
            if submit_only:
                policy = str(edge.get("secondary_policy") or "submit_if_objective")
                if policy == "audit_only":
                    continue
                if policy == "submit_if_objective" and not self._secondary_submit_ready(target):
                    continue
            scored.append((float(edge.get("edge_coverage", 0.0) or 0.0), name))
        scored.sort(reverse=True)
        result: List[str] = []
        for _, name in scored:
            if name not in result:
                result.append(name)
            if len(result) >= limit:
                break
        return result

    def _secondary_submit_ready(self, candidate: Any) -> bool:
        name = self._name(candidate)
        matched = self._matched(candidate)
        if f"diagnosis:{name}" in matched:
            return True
        if name == "心力衰竭":
            return self._heart_failure_state_evidence(matched)
        components = getattr(candidate, "component_scores", {}) or {}
        if float(components.get("objective_evidence", 0.0) or 0.0) >= 1.0:
            return True
        return bool(matched & _OBJECTIVE_FINDINGS)

    @staticmethod
    def _heart_failure_state_evidence(matched: set[str]) -> bool:
        if "heart_failure_state" in matched:
            return True
        congestion = bool(
            matched
            & {
                "fluid_retention_pattern",
                "leg_edema",
                "symptom:下肢水肿",
                "symptom:脚踝水肿",
            }
        )
        positional_dyspnea = bool(
            matched
            & {
                "orthopnea",
                "paroxysmal_nocturnal_dyspnea",
                "symptom:端坐呼吸",
                "symptom:夜间阵发性呼吸困难",
            }
        )
        return congestion and positional_dyspnea

    def _final_names(
        self,
        primary: str,
        secondary: Sequence[str],
        existing: Sequence[str],
        max_final_diagnoses: int,
    ) -> List[str]:
        names: List[str] = []
        for name in [primary] + list(secondary) + list(existing or []):
            text = str(name or "").strip()
            if text and text not in names:
                names.append(text)
            if len(names) >= max(1, int(max_final_diagnoses or 1)):
                break
        return names

    def _root_candidate_coverage(
        self,
        candidate: Any,
        edges: Sequence[Dict[str, Any]],
    ) -> float:
        if candidate is None:
            return 0.0
        own = self._candidate_signal(candidate)
        downstream = sum(float(edge.get("edge_coverage", 0.0) or 0.0) for edge in edges)
        downstream = min(0.42, downstream * 0.34)
        return round(min(1.0, own + downstream), 4)

    def _candidate_signal(self, candidate: Any) -> float:
        if candidate is None:
            return 0.0
        coverage = float(getattr(candidate, "coverage_score", 0.0) or 0.0)
        core = float(getattr(candidate, "core_explanatory_coverage", 0.0) or 0.0)
        diagnostic = float(getattr(candidate, "diagnostic_evidence_score", 0.0) or 0.0)
        objective = 1.0 if self._objective_or_core(candidate, {}) else 0.0
        required = 1.0 if bool(getattr(candidate, "required_met", False)) else 0.0
        residual_core = int(getattr(candidate, "residual_core_evidence_count", 0) or 0)
        signal = (
            0.30 * coverage
            + 0.32 * core
            + 0.16 * min(1.0, diagnostic)
            + 0.12 * objective
            + 0.10 * required
            - min(0.16, 0.04 * max(0, residual_core))
        )
        return max(0.0, min(1.0, signal))

    def _write_empty_candidate_roles(self, candidates: Sequence[Any]) -> None:
        for candidate in candidates:
            if not candidate:
                continue
            setattr(candidate, "root_cause_coverage", 0.0)
            setattr(candidate, "explains_candidates", [])
            setattr(candidate, "explained_by_root_cause", "")
            setattr(candidate, "root_cause_role", "")
            setattr(candidate, "root_cause_submit_as_final", False)

    def _write_candidate_roles(
        self,
        candidates: Sequence[Any],
        primary: str,
        secondary: Sequence[str],
        edges: Sequence[Dict[str, Any]],
        coverage: float,
    ) -> None:
        self._write_empty_candidate_roles(candidates)
        by_name = {self._name(item): item for item in candidates if self._name(item)}
        edge_targets: Dict[str, List[str]] = {}
        for edge in edges:
            edge_targets.setdefault(str(edge.get("source") or ""), []).append(
                str(edge.get("target") or "")
            )
        for name, target_names in edge_targets.items():
            candidate = by_name.get(name)
            if candidate:
                setattr(candidate, "explains_candidates", list(dict.fromkeys(target_names)))
        root = by_name.get(primary)
        if root:
            setattr(root, "root_cause_role", "primary")
            setattr(root, "root_cause_coverage", round(float(coverage or 0.0), 4))
        for name in secondary:
            candidate = by_name.get(name)
            if candidate:
                setattr(candidate, "root_cause_role", "secondary")
                setattr(candidate, "explained_by_root_cause", primary)
                submit_as_final = any(
                    str(edge.get("source") or "") == primary
                    and str(edge.get("target") or "") == name
                    and bool(edge.get("submit_as_final"))
                    for edge in edges
                )
                setattr(candidate, "root_cause_submit_as_final", submit_as_final)

    def _mechanism_supported(
        self,
        edges: Sequence[Dict[str, Any]],
        mechanism_ids: Iterable[str],
    ) -> bool:
        # Explicit relation matches are sufficient. Mechanism hypotheses only add audit support.
        return bool(edges)

    def _entry(self, candidate: Any) -> Dict[str, Any]:
        if not self.knowledge:
            return {}
        try:
            return dict(self.knowledge.get(self._name(candidate)) or {})
        except Exception:
            return {}

    def _matched(self, candidate: Any) -> set:
        return set(str(item) for item in getattr(candidate, "matched_evidence", []) or [])

    def _name(self, candidate: Any) -> str:
        return str(getattr(candidate, "diagnosis", "") or "").strip()

    def _normalized_names(self, names: Sequence[Any]) -> set:
        result = set()
        for item in names or []:
            text = str(item or "").strip()
            if not text:
                continue
            normalized = None
            if self.knowledge:
                try:
                    normalized = self.knowledge.normalize_name(text)
                except Exception:
                    normalized = None
            result.add(str(normalized or text))
        return result

    def _load_relations(self) -> List[Dict[str, Any]]:
        payload = self._read_json(self.path, {})
        relations = payload.get("relations", []) if isinstance(payload, dict) else []
        return [dict(item) for item in relations if isinstance(item, dict)]

    @staticmethod
    def _read_json(path: str, default: Any) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return default

    @staticmethod
    def _dedupe_edges(edges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        seen = set()
        for edge in edges:
            key = (
                edge.get("relation_id"),
                edge.get("source"),
                edge.get("target"),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(dict(edge))
        return result
