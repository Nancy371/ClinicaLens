"""Compile derived evidence patterns from verified atomic evidence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .clinical_evidence import EvidenceBundle, Observation
from .evidence_registry import EvidenceDefinitionRegistry
from .targeted_evidence_verifier import DERIVED, VERIFIED_POSITIVE, VerificationResult


@dataclass
class EvidencePatternMatch:
    pattern_id: str
    output_evidence_id: str
    matched: bool
    matched_inputs: List[str] = field(default_factory=list)
    missing_inputs: List[str] = field(default_factory=list)
    rule_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "output_evidence_id": self.output_evidence_id,
            "matched": bool(self.matched),
            "matched_inputs": list(self.matched_inputs),
            "missing_inputs": list(self.missing_inputs),
            "rule_id": self.rule_id,
        }


class EvidencePatternCompiler:
    """Derive pattern observations from verified observed/derived facts."""

    def __init__(
        self,
        registry: Optional[EvidenceDefinitionRegistry] = None,
        ref_dir: str = "data/ref_data",
    ):
        self.registry = registry or EvidenceDefinitionRegistry(ref_dir)
        self.ref_dir = ref_dir
        self.path = os.path.join(ref_dir, "evidence_patterns.json")
        self.patterns = self._load_patterns()
        self.last_matches: List[Dict[str, Any]] = []

    def compile(
        self,
        verification_results: Sequence[Any],
        evidence: EvidenceBundle,
    ) -> List[Observation]:
        facts = self._facts(verification_results, evidence)
        derived: List[Observation] = []
        matches: List[Dict[str, Any]] = []
        changed = True
        while changed:
            changed = False
            for pattern in self.patterns:
                output = str(
                    pattern.get("output", {}).get("evidence_id")
                    or pattern.get("output_evidence_id")
                    or pattern.get("pattern_id")
                    or ""
                ).strip()
                if not output or output in facts.positive:
                    continue
                match = self._evaluate_pattern(pattern, facts)
                matches.append(match.to_dict())
                if not match.matched:
                    continue
                observation = self._observation(pattern, match)
                derived.append(observation)
                facts.positive.add(output)
                facts.sources[output] = {"pattern_compiler"}
                changed = True
        self.last_matches = _dedupe_dicts(matches)
        return _dedupe_observations(derived)

    def _evaluate_pattern(self, pattern: Dict[str, Any], facts: "_FactContext") -> EvidencePatternMatch:
        logic = pattern.get("logic") or {}
        if isinstance(logic, str):
            logic = {logic: pattern.get("required") or []}
        output = str(
            pattern.get("output", {}).get("evidence_id")
            or pattern.get("output_evidence_id")
            or pattern.get("pattern_id")
            or ""
        )
        matched_inputs: List[str] = []
        missing_inputs: List[str] = []
        matched = self._evaluate_logic(logic, facts, matched_inputs, missing_inputs)
        not_any_of = pattern.get("not_any_of") or []
        negative_hits = [item for item in _flatten_labels(not_any_of) if facts.has_positive(item)]
        if negative_hits:
            matched = False
            missing_inputs.extend([f"not_any_of:{item}" for item in negative_hits])
        return EvidencePatternMatch(
            pattern_id=str(pattern.get("pattern_id") or output),
            output_evidence_id=output,
            matched=bool(matched),
            matched_inputs=list(dict.fromkeys(matched_inputs)),
            missing_inputs=list(dict.fromkeys(missing_inputs)),
            rule_id=str(pattern.get("rule_id") or pattern.get("pattern_id") or output),
        )

    def _evaluate_logic(
        self,
        logic: Any,
        facts: "_FactContext",
        matched_inputs: List[str],
        missing_inputs: List[str],
    ) -> bool:
        if isinstance(logic, str):
            if facts.has_positive(logic):
                matched_inputs.append(logic)
                return True
            missing_inputs.append(logic)
            return False
        if isinstance(logic, list):
            return self._all_of(logic, facts, matched_inputs, missing_inputs)
        if not isinstance(logic, dict):
            missing_inputs.append(str(logic))
            return False
        if "all_of" in logic:
            return self._all_of(logic.get("all_of") or [], facts, matched_inputs, missing_inputs)
        if "any_of" in logic:
            return self._any_of(logic.get("any_of") or [], facts, matched_inputs, missing_inputs)
        if "min_count" in logic:
            spec = logic.get("min_count")
            if isinstance(spec, dict):
                count = _int(spec.get("count"), 0)
                items = spec.get("of") or []
            else:
                count = _int(spec, 0)
                items = logic.get("of") or []
            return self._min_count(count, items, facts, matched_inputs, missing_inputs)
        if "numeric_threshold" in logic:
            return self._numeric_threshold(logic.get("numeric_threshold") or {}, facts, matched_inputs, missing_inputs)
        return False

    def _all_of(
        self,
        items: Sequence[Any],
        facts: "_FactContext",
        matched_inputs: List[str],
        missing_inputs: List[str],
    ) -> bool:
        local_matched: List[str] = []
        local_missing: List[str] = []
        ok = True
        for item in items or []:
            if self._evaluate_logic(item, facts, local_matched, local_missing):
                continue
            ok = False
        matched_inputs.extend(local_matched)
        missing_inputs.extend(local_missing)
        return ok

    def _any_of(
        self,
        items: Sequence[Any],
        facts: "_FactContext",
        matched_inputs: List[str],
        missing_inputs: List[str],
    ) -> bool:
        local_missing: List[str] = []
        for item in items or []:
            local_matched: List[str] = []
            nested_missing: List[str] = []
            if self._evaluate_logic(item, facts, local_matched, nested_missing):
                matched_inputs.extend(local_matched)
                return True
            local_missing.extend(nested_missing)
        missing_inputs.extend(local_missing)
        return False

    def _min_count(
        self,
        count: int,
        items: Sequence[Any],
        facts: "_FactContext",
        matched_inputs: List[str],
        missing_inputs: List[str],
    ) -> bool:
        local_matched: List[str] = []
        local_missing: List[str] = []
        for item in items or []:
            nested_matched: List[str] = []
            nested_missing: List[str] = []
            if self._evaluate_logic(item, facts, nested_matched, nested_missing):
                local_matched.extend(nested_matched)
            else:
                local_missing.extend(nested_missing)
        matched_inputs.extend(local_matched)
        if len(local_matched) < max(0, count):
            missing_inputs.extend(local_missing)
        return len(local_matched) >= max(0, count)

    @staticmethod
    def _numeric_threshold(
        spec: Dict[str, Any],
        facts: "_FactContext",
        matched_inputs: List[str],
        missing_inputs: List[str],
    ) -> bool:
        evidence_id = str(spec.get("evidence_id") or "")
        if not evidence_id or evidence_id not in facts.values:
            missing_inputs.append(evidence_id or "numeric_threshold")
            return False
        value = facts.values[evidence_id]
        if spec.get("min_value") is not None and value < float(spec["min_value"]):
            missing_inputs.append(evidence_id)
            return False
        if spec.get("max_value") is not None and value > float(spec["max_value"]):
            missing_inputs.append(evidence_id)
            return False
        matched_inputs.append(evidence_id)
        return True

    def _observation(self, pattern: Dict[str, Any], match: EvidencePatternMatch) -> Observation:
        output = str(pattern.get("output", {}).get("evidence_id") or match.output_evidence_id)
        definition = self.registry.require(output)
        output_payload = dict(pattern.get("output") or {})
        source_text = "derived from " + ", ".join(match.matched_inputs)
        return Observation(
            finding=output,
            source="pattern_compiler",
            polarity=str(output_payload.get("polarity") or "positive"),
            confidence=_float(output_payload.get("confidence"), 0.9),
            raw_text=source_text,
            source_text=source_text,
            field_path="derived_patterns",
            evidence_level=str(output_payload.get("evidence_level") or definition.evidence_level or "diagnostic_pattern"),
            information_value=_float(output_payload.get("information_value"), definition.information_value),
        )

    def _facts(
        self,
        verification_results: Sequence[Any],
        evidence: EvidenceBundle,
    ) -> "_FactContext":
        facts = _FactContext()
        for item in getattr(evidence, "observations", []) or []:
            source = str(getattr(item, "source", "") or "")
            if source == "reasoning_inference":
                continue
            finding = self.registry.normalize_evidence_id(getattr(item, "finding", ""))
            if not finding:
                continue
            if str(getattr(item, "polarity", "positive") or "positive") == "positive":
                facts.positive.add(finding)
            else:
                facts.negative.add(finding)
            facts.sources.setdefault(finding, set()).add(source)
            if getattr(item, "value", None) is not None:
                facts.values[finding] = float(getattr(item, "value"))
        for value in verification_results or []:
            result = VerificationResult.from_any(value)
            if result is None:
                continue
            finding = self.registry.normalize_evidence_id(result.target_evidence_id)
            if result.verification_status in {VERIFIED_POSITIVE, DERIVED}:
                facts.positive.add(finding)
            elif result.verification_status == "verified_negative":
                facts.negative.add(finding)
            if result.value is not None:
                facts.values[finding] = result.value
            if result.source_section:
                facts.sources.setdefault(finding, set()).add(result.source_section)
        return facts

    def _load_patterns(self) -> List[Dict[str, Any]]:
        payload = _read_json(self.path, {})
        raw = payload.get("patterns") if isinstance(payload, dict) else None
        patterns = [dict(item) for item in _BUILTIN_PATTERNS]
        for item in raw or []:
            if isinstance(item, dict):
                patterns.append(dict(item))
        return _dedupe_dicts(patterns)


class _FactContext:
    def __init__(self) -> None:
        self.positive: Set[str] = set()
        self.negative: Set[str] = set()
        self.values: Dict[str, float] = {}
        self.sources: Dict[str, Set[str]] = {}

    def has_positive(self, evidence_id: str) -> bool:
        return evidence_id in self.positive


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return default


def _flatten_labels(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: List[str] = []
        for item in value:
            result.extend(_flatten_labels(item))
        return result
    if isinstance(value, dict):
        result = []
        for key in ("all_of", "any_of", "of", "not_any_of"):
            result.extend(_flatten_labels(value.get(key) or []))
        return result
    return []


def _dedupe_dicts(values: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for value in values or []:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _dedupe_observations(values: Iterable[Observation]) -> List[Observation]:
    result: Dict[tuple[str, str], Observation] = {}
    for item in values or []:
        key = (str(item.finding or ""), str(item.polarity or "positive"))
        if key[0] and key not in result:
            result[key] = item
    return list(result.values())


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_BUILTIN_PATTERNS: List[Dict[str, Any]] = [
    {
        "pattern_id": "multilineage_cytopenia",
        "rule_id": "PATTERN-HEM-001",
        "logic": {
            "min_count": {
                "count": 2,
                "of": [
                    "hemoglobin_low",
                    "platelet_low",
                    "neutrophil_low",
                    "white_blood_cell_abnormal",
                ],
            }
        },
        "output": {
            "evidence_id": "multilineage_cytopenia",
            "evidence_level": "diagnostic_pattern",
            "polarity": "positive",
            "confidence": 0.9,
            "information_value": 0.88,
        },
    },
    {
        "pattern_id": "acute_leukemia_pattern",
        "rule_id": "PATTERN-LEUKEMIA-001",
        "logic": {
            "all_of": [
                "blast_present",
                {"any_of": ["multilineage_cytopenia", "extreme_white_blood_cell_abnormality"]},
            ]
        },
        "output": {
            "evidence_id": "acute_leukemia_pattern",
            "evidence_level": "diagnostic_pattern",
            "polarity": "positive",
            "confidence": 0.92,
            "information_value": 0.96,
        },
    },
    {
        "pattern_id": "bacterial_prostatitis_negative_pattern",
        "rule_id": "PATTERN-URO-NEG-001",
        "logic": {
            "all_of": [
                {
                    "min_count": {
                        "count": 2,
                        "of": [
                            "urine_culture_no_growth",
                            "nitrite_negative",
                            "leukocyte_esterase_negative",
                            "urine_wbc_normal",
                        ],
                    }
                }
            ]
        },
        "not_any_of": [
            "prostate_tenderness",
            "perineal_pain",
            "dysuria",
            "urinary_frequency",
            "urinary_urgency",
            "urine_culture_positive",
        ],
        "output": {
            "evidence_id": "bacterial_prostatitis_negative_pattern",
            "evidence_level": "diagnostic_pattern",
            "polarity": "positive",
            "confidence": 0.88,
            "information_value": 0.84,
        },
    },
    {
        "pattern_id": "pavm_ct_vascular_connection_pattern",
        "rule_id": "PATTERN-PAVM-CT-001",
        "logic": {
            "any_of": [
                "abnormal_pulmonary_av_connection_described",
                {"all_of": ["feeding_pulmonary_artery_present", "draining_pulmonary_vein_present"]},
                {"all_of": ["abnormal_pulmonary_vascular_cluster", "early_pulmonary_venous_enhancement"]},
            ]
        },
        "output": {
            "evidence_id": "enhanced_ct_vascular_malformation",
            "evidence_level": "diagnostic_pattern",
            "polarity": "positive",
            "confidence": 0.94,
            "information_value": 0.97,
        },
    },
    {
        "pattern_id": "pavm_bubble_echo_shunt_pattern",
        "rule_id": "PATTERN-PAVM-ECHO-001",
        "logic": {
            "any_of": [
                "bubble_echo_right_to_left_shunt",
                {"all_of": ["delayed_bubbles_in_left_heart", "intrapulmonary_right_to_left_shunt_observed"]},
            ]
        },
        "output": {
            "evidence_id": "right_to_left_shunt",
            "evidence_level": "diagnostic_pattern",
            "polarity": "positive",
            "confidence": 0.92,
            "information_value": 0.96,
        },
    },
    {
        "pattern_id": "pulmonary_av_fistula_pattern",
        "rule_id": "PATTERN-PAVM-001",
        "logic": {
            "all_of": [
                {"any_of": ["hemoptysis", "hypoxemia", "cyanosis"]},
                {"any_of": ["right_to_left_shunt", "pulmonary_vascular_shunt", "pulmonary_avm_mechanism", "intrapulmonary_right_to_left_shunt_observed"]},
                {"any_of": ["pulmonary_cta_positive", "enhanced_ct_vascular_malformation", "bubble_echo_right_to_left_shunt", "abnormal_pulmonary_av_connection_described"]},
            ]
        },
        "output": {
            "evidence_id": "pulmonary_av_fistula_pattern",
            "evidence_level": "diagnostic_pattern",
            "polarity": "positive",
            "confidence": 0.92,
            "information_value": 0.95,
        },
    },
]
