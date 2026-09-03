"""Clinical pattern compilation from verified observations.

This module deliberately stops one layer before diagnosis. It turns reusable
observation combinations into clinical patterns that can improve recall and
exam planning, while disease-level eligibility remains with the Judge pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .clinical_evidence import EvidenceBundle, Observation


@dataclass
class ClinicalPattern:
    pattern_id: str
    pattern_type: str
    pattern_version: str = ""
    supporting_findings: List[str] = field(default_factory=list)
    supporting_observation_ids: List[str] = field(default_factory=list)
    contradicting_findings: List[str] = field(default_factory=list)
    missing_required_groups: List[List[str]] = field(default_factory=list)
    matched_domains: List[str] = field(default_factory=list)
    body_system: str = ""
    temporal_pattern: str = ""
    temporal_consistency: str = "unknown"
    polarity_consistency: str = "consistent"
    confidence: float = 0.0
    information_value: float = 0.0
    mechanism_ids: List[str] = field(default_factory=list)
    family_ids: List[str] = field(default_factory=list)
    source_level: str = "deterministic_rule"
    verified: bool = True
    derivation_trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClinicalPatternCompiler:
    """Compile formal clinical patterns from structured observations."""

    def __init__(self, ref_dir: str = "data/ref_data"):
        self.ref_dir = ref_dir
        self.path = os.path.join(ref_dir, "clinical_patterns.json")
        payload = self._load_payload()
        self.version = str(payload.get("version") or "") if isinstance(payload, dict) else ""
        self.rules = self._load_rules(payload)

    def compile(self, evidence: Optional[EvidenceBundle]) -> List[ClinicalPattern]:
        bundle = evidence or EvidenceBundle()
        observations = [
            item
            for item in list(bundle.observations or [])
            if not getattr(item, "shadowed_by", "")
            and str(getattr(item, "source", "") or "") != "reasoning_inference"
        ]
        positives = _best_by_finding(
            item for item in observations if item.polarity == "positive"
        )
        negatives = _best_by_finding(
            item for item in observations if item.polarity == "negative"
        )
        compiled: List[ClinicalPattern] = []
        for rule in self.rules:
            pattern = self._match_rule(rule, positives, negatives)
            if pattern is not None:
                compiled.append(pattern)
        compiled.sort(
            key=lambda item: (item.information_value, item.confidence, item.pattern_id),
            reverse=True,
        )
        return compiled

    def candidate_rules(self, pattern_id: str) -> List[Dict[str, Any]]:
        for rule in self.rules:
            if str(rule.get("pattern_id") or "") == str(pattern_id or ""):
                return [
                    dict(item)
                    for item in list(rule.get("candidate_entities") or [])
                    if isinstance(item, dict)
                ]
        return []

    def retrieval_views(self, evidence: Optional[EvidenceBundle]) -> List[Dict[str, Any]]:
        patterns = self.compile(evidence)
        views: List[Dict[str, Any]] = []
        for pattern in patterns[:6]:
            terms = _dedupe(
                [
                    pattern.pattern_id,
                    *pattern.supporting_findings,
                    *pattern.mechanism_ids,
                    *pattern.family_ids,
                    *list(pattern.derivation_trace.get("retrieval_terms") or []),
                ]
            )
            if not terms:
                continue
            views.append(
                {
                    "view_type": "clinical_pattern",
                    "query": " ".join(terms),
                    "terms": terms,
                    "weight": 1.25,
                    "metadata": pattern.to_dict(),
                }
            )
        return views

    def _match_rule(
        self,
        rule: Dict[str, Any],
        positives: Dict[str, Observation],
        negatives: Dict[str, Observation],
    ) -> Optional[ClinicalPattern]:
        required_groups = [
            group
            for group in list(rule.get("required_groups") or [])
            if isinstance(group, dict)
        ]
        if not required_groups:
            return None

        matched_required = []
        missing_required: List[List[str]] = []
        matched_domains: List[str] = []
        supporting: List[str] = []
        observation_ids: List[str] = []
        finding_domains = {
            str(key): str(value)
            for key, value in dict(rule.get("finding_domains") or {}).items()
            if str(key) and str(value)
        }
        for group in required_groups:
            findings = [str(item) for item in group.get("findings") or [] if str(item)]
            min_match = max(1, int(group.get("min_match", 1) or 1))
            hits = [finding for finding in findings if finding in positives]
            if len(hits) >= min_match:
                domain = str(group.get("domain") or "").strip()
                matched_required.append(
                    {
                        "findings": findings,
                        "min_match": min_match,
                        "matched": hits,
                        "domain": domain,
                    }
                )
                supporting.extend(hits)
                observation_ids.extend(_observation_ids(positives[finding]) for finding in hits)
                if domain:
                    matched_domains.append(domain)
                matched_domains.extend(
                    finding_domains.get(finding, "") for finding in hits
                )
            else:
                missing_required.append(findings)

        contradicting = [
            finding
            for finding in list(rule.get("contradicting_findings") or [])
            if str(finding) in positives or str(finding) in negatives
        ]
        if missing_required:
            return None
        if contradicting and bool(rule.get("block_on_contradiction", True)):
            return None

        optional = [str(item) for item in rule.get("optional_findings") or [] if str(item)]
        optional_hits = [finding for finding in optional if finding in positives]
        supporting.extend(optional_hits)
        observation_ids.extend(_observation_ids(positives[finding]) for finding in optional_hits)
        matched_domains.extend(finding_domains.get(finding, "") for finding in optional_hits)

        required_coverage = len(matched_required) / max(1, len(required_groups))
        optional_support = len(optional_hits) / max(1, len(optional)) if optional else 0.0
        source_reliability = max(
            (_source_reliability(positives[finding]) for finding in set(supporting) if finding in positives),
            default=0.68,
        )
        specificity = float(rule.get("information_value", 0.75) or 0.75)
        contradiction_penalty = 0.22 * len(contradicting)
        confidence = (
            0.40 * required_coverage
            + 0.15 * optional_support
            + 0.15 * float(rule.get("temporal_fit", 1.0) or 1.0)
            + 0.15 * source_reliability
            + 0.15 * specificity
            - contradiction_penalty
        )
        min_confidence = float(rule.get("min_confidence", 0.62) or 0.62)
        confidence = max(0.0, min(0.98, confidence))
        if confidence < min_confidence:
            return None

        return ClinicalPattern(
            pattern_id=str(rule.get("pattern_id") or ""),
            pattern_type=str(rule.get("pattern_type") or "clinical_pattern"),
            pattern_version=self.version,
            supporting_findings=_dedupe(supporting),
            supporting_observation_ids=_dedupe(observation_ids),
            contradicting_findings=_dedupe(contradicting),
            missing_required_groups=[],
            matched_domains=_dedupe(matched_domains),
            body_system=str(rule.get("body_system") or ""),
            temporal_pattern=str(rule.get("temporal_pattern") or ""),
            temporal_consistency=str(rule.get("temporal_consistency") or "supported"),
            polarity_consistency="consistent" if not contradicting else "conflicted",
            confidence=round(confidence, 4),
            information_value=round(max(0.0, min(1.0, specificity)), 4),
            mechanism_ids=_dedupe(rule.get("mechanism_ids") or []),
            family_ids=_dedupe(rule.get("family_ids") or []),
            source_level=str(rule.get("source_level") or "deterministic_rule"),
            verified=True,
            derivation_trace={
                "rule_id": str(rule.get("pattern_id") or ""),
                "matched_required_groups": matched_required,
                "optional_hits": optional_hits,
                "retrieval_terms": list(rule.get("retrieval_terms") or []),
                "candidate_entities": list(rule.get("candidate_entities") or []),
            },
        )

    def _load_payload(self) -> Dict[str, Any]:
        payload = _read_json(self.path, {})
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _load_rules(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        rules = payload.get("patterns", []) if isinstance(payload, dict) else []
        return [dict(item) for item in rules if isinstance(item, dict)]


def _best_by_finding(observations: Iterable[Observation]) -> Dict[str, Observation]:
    result: Dict[str, Observation] = {}
    for item in observations:
        finding = str(getattr(item, "finding", "") or "")
        if not finding:
            continue
        current = result.get(finding)
        if current is None or _observation_quality(item) > _observation_quality(current):
            result[finding] = item
    return result


def _observation_quality(item: Observation) -> float:
    try:
        confidence = float(getattr(item, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    try:
        info = float(getattr(item, "information_value", 0.0) or 0.0)
    except (TypeError, ValueError):
        info = 0.0
    return confidence * 0.45 + info * 0.55


def _observation_ids(item: Observation) -> str:
    field_path = str(getattr(item, "field_path", "") or "")
    source = str(getattr(item, "source", "") or "")
    finding = str(getattr(item, "finding", "") or "")
    return "|".join(part for part in (source, field_path, finding) if part)


def _source_reliability(item: Observation) -> float:
    source = str(getattr(item, "source", "") or "").lower()
    if source == "raw_case_finding":
        return 0.58
    if "reasoning" in source:
        return 0.22
    if any(token in source for token in ("ct", "mri", "x线", "影像", "超声", "心电图", "ecg")):
        return 0.88
    if any(token in source for token in ("血", "尿", "实验", "检查", "培养", "naat", "pcr")):
        return 0.86
    if any(token in source for token in ("体格", "查体", "医生观察")):
        return 0.74
    if any(token in source for token in ("问诊", "主诉", "patient", "history")):
        return 0.62
    return 0.68


def _dedupe(items: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default
