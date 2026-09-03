"""Bridge verified clinical patterns into Judge-pool protection signals.

Clinical patterns improve recall, but they are not diagnoses. This module only
turns high-quality multi-system syndrome patterns into a scoped permission to
survive one pool-filter rule. Eligibility and submission remain Judge-owned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


CROSS_SYSTEM_SCOPE = "cross_system_no_shared_core_evidence"
BRIDGE_REASON = "verified_multi_system_syndrome_bridge"
BRIDGE_PATTERN_TYPE = "multi_system_syndrome_bridge_pattern"

_STRENGTH_RANK = {"weak": 0, "probable": 1, "strong": 2}


@dataclass
class ClinicalPatternMatch:
    match_id: str
    pattern_id: str
    pattern_version: str = ""
    verification_status: str = "verified"
    source_observation_ids: List[str] = field(default_factory=list)
    matched_domains: List[str] = field(default_factory=list)
    temporal_consistency: str = "unknown"
    polarity_consistency: str = "consistent"
    confidence: float = 0.0
    supporting_findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DerivedPatternAssertion:
    assertion_id: str
    assertion_type: str
    canonical_pattern: str
    source_match_id: str
    derived: bool = True
    submittable: bool = False
    strength: str = "weak"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BridgeProtectionDecision:
    candidate_id: str
    source_assertion_id: str
    protection_scope: List[str] = field(default_factory=lambda: [CROSS_SYSTEM_SCOPE])
    protection_status: str = "active"
    reason_code: str = BRIDGE_REASON
    allowed_pairwise_targets: List[str] = field(
        default_factory=lambda: ["current_primary", "top_conflicting_candidate"]
    )
    expires_at_case_version: int = 0
    strength: str = "weak"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BridgeValidationResult:
    candidate_id: str
    pattern_id: str
    validation_status: str
    strength: str = "weak"
    reason_code: str = ""
    clinical_pattern_match: Optional[ClinicalPatternMatch] = None
    derived_assertion: Optional[DerivedPatternAssertion] = None
    protection_decision: Optional[BridgeProtectionDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.clinical_pattern_match:
            data["clinical_pattern_match"] = self.clinical_pattern_match.to_dict()
        if self.derived_assertion:
            data["derived_assertion"] = self.derived_assertion.to_dict()
        if self.protection_decision:
            data["protection_decision"] = self.protection_decision.to_dict()
        return data


class BridgePatternValidator:
    """Validate clinical-pattern bridge configs declared by disease metadata."""

    def validate_all(
        self,
        candidates: Sequence[Any],
        clinical_patterns: Sequence[Dict[str, Any]],
        knowledge: Any = None,
        *,
        case_version: int = 0,
    ) -> Dict[str, Any]:
        pattern_by_id = {
            str(item.get("pattern_id") or ""): dict(item)
            for item in clinical_patterns or []
            if isinstance(item, dict) and str(item.get("pattern_id") or "")
        }
        results: List[Dict[str, Any]] = []
        active_candidates: List[str] = []
        protected_count = 0
        for candidate in candidates or []:
            candidate_results = self.validate_candidate(
                candidate,
                pattern_by_id,
                knowledge,
                case_version=case_version,
            )
            if not candidate_results:
                continue
            self.apply_candidate_results(candidate, candidate_results)
            results.extend(item.to_dict() for item in candidate_results)
            if any(item.protection_decision for item in candidate_results):
                protected_count += 1
                name = str(getattr(candidate, "diagnosis", "") or "")
                if name:
                    active_candidates.append(name)
        return {
            "bridge_validation_results": results,
            "bridge_protected_candidates": list(dict.fromkeys(active_candidates)),
            "bridge_protection_activation_count": protected_count,
        }

    def validate_candidate(
        self,
        candidate: Any,
        pattern_by_id: Dict[str, Dict[str, Any]],
        knowledge: Any = None,
        *,
        case_version: int = 0,
    ) -> List[BridgeValidationResult]:
        if not candidate or bool(getattr(candidate, "hard_contradiction", False)):
            return []
        entry = self._entry(candidate, knowledge)
        configs = [
            dict(item)
            for item in entry.get("accepted_bridge_patterns", []) or []
            if isinstance(item, dict)
        ]
        if not configs:
            return []
        result: List[BridgeValidationResult] = []
        source_patterns = self._candidate_source_patterns(candidate, pattern_by_id)
        candidate_id = str(
            getattr(candidate, "entity_id", "")
            or getattr(candidate, "diagnosis", "")
            or ""
        )
        for config in configs:
            pattern_id = str(config.get("pattern_id") or "").strip()
            if not pattern_id:
                continue
            pattern = source_patterns.get(pattern_id) or pattern_by_id.get(pattern_id)
            if not pattern:
                continue
            result.append(
                self._validate_config(
                    candidate_id,
                    pattern,
                    config,
                    case_version=case_version,
                )
            )
        return result

    @staticmethod
    def apply_candidate_results(
        candidate: Any,
        results: Sequence[BridgeValidationResult],
    ) -> None:
        matches = list(getattr(candidate, "clinical_pattern_matches", []) or [])
        assertions = list(getattr(candidate, "derived_pattern_assertions", []) or [])
        validations = list(getattr(candidate, "bridge_validation_results", []) or [])
        protections = list(getattr(candidate, "bridge_protection_decisions", []) or [])
        pattern_matches = list(getattr(candidate, "evidence_pattern_matches", []) or [])
        seen_pattern_ids = {
            str(item.get("pattern_id") or "")
            for item in pattern_matches
            if isinstance(item, dict)
        }
        for result in results or []:
            validations.append(result.to_dict())
            if result.clinical_pattern_match:
                matches.append(result.clinical_pattern_match.to_dict())
            if result.derived_assertion:
                assertion = result.derived_assertion.to_dict()
                assertions.append(assertion)
                pattern_id = assertion["canonical_pattern"]
                if pattern_id not in seen_pattern_ids:
                    pattern_matches.append(
                        {
                            "pattern_id": pattern_id,
                            "pattern_type": BRIDGE_PATTERN_TYPE,
                            "matched": result.protection_decision is not None,
                            "source_pattern_id": result.pattern_id,
                            "source_assertion_id": assertion["assertion_id"],
                            "strength": result.strength,
                            "effect": {
                                "pool_protection": [CROSS_SYSTEM_SCOPE],
                                "submittable": False,
                            },
                            "verification_status": result.validation_status,
                            "reason_code": result.reason_code,
                        }
                    )
                    seen_pattern_ids.add(pattern_id)
            if result.protection_decision:
                protections.append(result.protection_decision.to_dict())
        setattr(candidate, "clinical_pattern_matches", _dedupe_dicts(matches, "match_id"))
        setattr(candidate, "derived_pattern_assertions", _dedupe_dicts(assertions, "assertion_id"))
        setattr(candidate, "bridge_validation_results", validations)
        setattr(candidate, "bridge_protection_decisions", _dedupe_dicts(protections, "source_assertion_id"))
        setattr(candidate, "evidence_pattern_matches", pattern_matches)

    def _validate_config(
        self,
        candidate_id: str,
        pattern: Dict[str, Any],
        config: Dict[str, Any],
        *,
        case_version: int = 0,
    ) -> BridgeValidationResult:
        pattern_id = str(pattern.get("pattern_id") or "")
        match = ClinicalPatternMatch(
            match_id=f"CPM-{candidate_id or 'candidate'}-{pattern_id}",
            pattern_id=pattern_id,
            pattern_version=str(pattern.get("pattern_version") or ""),
            verification_status=self._verification_status(pattern),
            source_observation_ids=_texts(pattern.get("supporting_observation_ids") or []),
            matched_domains=_texts(pattern.get("matched_domains") or []),
            temporal_consistency=str(pattern.get("temporal_consistency") or "unknown"),
            polarity_consistency=str(pattern.get("polarity_consistency") or "consistent"),
            confidence=_float(pattern.get("confidence"), 0.0),
            supporting_findings=_texts(pattern.get("supporting_findings") or []),
        )
        canonical = str(
            config.get("canonical_pattern")
            or config.get("bridge_pattern")
            or pattern_id
        )
        assertion = DerivedPatternAssertion(
            assertion_id=f"DPA-{candidate_id or 'candidate'}-{canonical}",
            assertion_type=str(config.get("assertion_type") or BRIDGE_PATTERN_TYPE),
            canonical_pattern=canonical,
            source_match_id=match.match_id,
            strength="weak",
        )
        verification_blocker = self._verification_blocker(match, pattern)
        if verification_blocker:
            return BridgeValidationResult(
                candidate_id=candidate_id,
                pattern_id=pattern_id,
                validation_status="rejected",
                reason_code=verification_blocker,
                clinical_pattern_match=match,
                derived_assertion=assertion,
            )
        strength = self._strength(match, config)
        assertion.strength = strength
        minimum = str(config.get("minimum_strength") or "probable").lower()
        if _STRENGTH_RANK.get(strength, 0) < _STRENGTH_RANK.get(minimum, 1):
            return BridgeValidationResult(
                candidate_id=candidate_id,
                pattern_id=pattern_id,
                validation_status="weak",
                strength=strength,
                reason_code="below_bridge_protection_strength",
                clinical_pattern_match=match,
                derived_assertion=assertion,
            )
        protection = BridgeProtectionDecision(
            candidate_id=candidate_id,
            source_assertion_id=assertion.assertion_id,
            protection_scope=_texts(config.get("protection_scope") or [CROSS_SYSTEM_SCOPE]),
            expires_at_case_version=case_version + 1 if case_version else 0,
            strength=strength,
        )
        return BridgeValidationResult(
            candidate_id=candidate_id,
            pattern_id=pattern_id,
            validation_status="active",
            strength=strength,
            reason_code=BRIDGE_REASON,
            clinical_pattern_match=match,
            derived_assertion=assertion,
            protection_decision=protection,
        )

    @staticmethod
    def _strength(match: ClinicalPatternMatch, config: Dict[str, Any]) -> str:
        domains = set(match.matched_domains)
        strong_domains = set(_texts(config.get("strong_required_domains") or []))
        probable_domains = set(_texts(config.get("probable_required_domains") or []))
        strong_min = _float(config.get("strong_min_confidence"), 0.82)
        probable_min = _float(config.get("probable_min_confidence"), 0.62)
        if strong_domains and strong_domains.issubset(domains) and match.confidence >= strong_min:
            if match.temporal_consistency in {"supported", "unknown", ""}:
                return "strong"
        if probable_domains and probable_domains.issubset(domains) and match.confidence >= probable_min:
            return "probable"
        return "weak"

    @staticmethod
    def _verification_status(pattern: Dict[str, Any]) -> str:
        if bool(pattern.get("verified", False)):
            return "verified"
        return str(pattern.get("verification_status") or "unverified")

    @staticmethod
    def _verification_blocker(match: ClinicalPatternMatch, pattern: Dict[str, Any]) -> str:
        if match.verification_status != "verified":
            return "pattern_member_unverified"
        if match.polarity_consistency == "conflicted":
            return "pattern_polarity_conflicted"
        if any("reasoning" in str(item).lower() for item in match.source_observation_ids):
            return "pattern_member_from_reasoning"
        if str(pattern.get("source_level") or "") == "llm_hypothesis":
            return "pattern_member_unverified"
        return ""

    @staticmethod
    def _entry(candidate: Any, knowledge: Any) -> Dict[str, Any]:
        if not knowledge:
            return {}
        try:
            return dict(knowledge.get(str(getattr(candidate, "diagnosis", "") or "")) or {})
        except Exception:
            return {}

    @staticmethod
    def _candidate_source_patterns(
        candidate: Any,
        pattern_by_id: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        result = dict(pattern_by_id)
        for source in getattr(candidate, "candidate_sources", []) or []:
            if not isinstance(source, dict) or source.get("source") != "clinical_pattern":
                continue
            metadata = dict(source.get("metadata") or {})
            pattern = dict(metadata.get("clinical_pattern") or {})
            pattern_id = str(pattern.get("pattern_id") or metadata.get("pattern_id") or "")
            if not pattern_id:
                continue
            if not pattern:
                pattern = dict(pattern_by_id.get(pattern_id) or {})
            if not pattern:
                pattern = {
                    "pattern_id": pattern_id,
                    "verified": bool(metadata.get("verified_pattern")),
                    "confidence": metadata.get("pattern_confidence", 0.0),
                    "information_value": metadata.get("pattern_information_value", 0.0),
                    "source_level": metadata.get("source_level", ""),
                    "supporting_findings": metadata.get("supporting_findings", []),
                    "supporting_observation_ids": metadata.get("supporting_observation_ids", []),
                    "matched_domains": metadata.get("matched_domains", []),
                    "temporal_consistency": metadata.get("temporal_consistency", "unknown"),
                    "polarity_consistency": metadata.get("polarity_consistency", "consistent"),
                }
            result[pattern_id] = pattern
        return result


def has_active_bridge_protection(candidate: Any, scope: str = CROSS_SYSTEM_SCOPE) -> bool:
    if not candidate or bool(getattr(candidate, "hard_contradiction", False)):
        return False
    for item in getattr(candidate, "bridge_protection_decisions", []) or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("protection_status") or "") != "active":
            continue
        if scope in set(_texts(item.get("protection_scope") or [])):
            return True
    return False


def _texts(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dedupe_dicts(values: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for value in values or []:
        if not isinstance(value, dict):
            continue
        marker = str(value.get(key) or value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(dict(value))
    return result
