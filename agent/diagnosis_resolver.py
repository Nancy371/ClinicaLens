"""Open-world diagnosis proposal normalization with closed-world submission."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class DiagnosisResolution:
    raw_name: str
    canonical_name: Optional[str]
    parent_name: Optional[str]
    method: str
    confidence: float
    model_confidence: float = 1.0
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    entity_id: str = ""
    submission_name: str = ""
    submittable: bool = False

    @property
    def resolved(self) -> bool:
        return bool(self.canonical_name or self.entity_id)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OpenWorldDiagnosisResolver:
    """Map free-form LLM diagnoses into the controlled submission namespace."""

    _LEADING_QUALIFIERS = re.compile(
        r"^(?:初步诊断|考虑诊断|诊断考虑|考虑|疑似|可能为|可能|拟诊|诊断为)\s*[:：]?\s*"
    )
    _TRAILING_QUALIFIERS = re.compile(
        r"\s*(?:待排除|待排|待查|可能|疑似|\?|？)\s*$"
    )
    _PHASE_SUFFIXES = (
        "急性发作",
        "急性加重期",
        "急性加重",
        "活动期",
        "稳定期",
        "恢复期",
    )
    _COMPOUND_MARKERS = ("合并", "伴有", "伴", "以及", "和", "及", "/", "、")

    def __init__(self, knowledge: Any, config: Optional[Dict[str, Any]] = None):
        section = (config or {}).get("diagnosis", {}) or {}
        self.knowledge = knowledge
        self.enabled = bool(section.get("open_world_candidates", True))
        self.fuzzy_threshold = float(section.get("name_match_threshold", 0.84) or 0.84)
        self.fuzzy_margin = float(section.get("name_match_margin", 0.08) or 0.08)
        self.max_raw_candidates = int(section.get("max_open_world_candidates", 10) or 10)
        self._search_terms = self._build_search_terms()

    def resolve_result(self, result: Any) -> List[DiagnosisResolution]:
        raw_candidates = self.extract_candidates(result)
        resolutions: List[DiagnosisResolution] = []
        seen: set[Tuple[str, Optional[str]]] = set()
        for raw_name, model_confidence in raw_candidates[: self.max_raw_candidates]:
            resolution = self.resolve(raw_name, model_confidence=model_confidence)
            key = (resolution.raw_name, resolution.canonical_name)
            if key not in seen:
                seen.add(key)
                resolutions.append(resolution)
        return resolutions

    def resolve(
        self,
        value: Any,
        model_confidence: float = 1.0,
    ) -> DiagnosisResolution:
        raw = str(value or "").strip()
        clean = self.clean_name(raw)
        if not clean:
            return DiagnosisResolution(raw, None, None, "empty", 0.0, model_confidence)

        entity = self._resolve_entity(clean)
        if entity:
            return self._resolved_entity(raw, entity, "exact_or_alias", 1.0, model_confidence)

        exact = self.knowledge.normalize_name(clean)
        if exact:
            return self._resolved(raw, exact, "exact_or_alias", 1.0, model_confidence)
        if not self.enabled:
            return DiagnosisResolution(raw, None, None, "unresolved", 0.0, model_confidence)

        hierarchy = self._hierarchy_match(clean)
        if hierarchy:
            return self._resolved(raw, hierarchy, "hierarchy", 0.88, model_confidence)

        alias_contains = self._alias_contains_match(clean)
        if alias_contains:
            canonical, confidence = alias_contains
            return self._resolved(raw, canonical, "alias_contains", confidence, model_confidence)

        ranked = self._rank_terms(clean)
        alternatives = [
            {"name": name, "score": round(score, 4), "term": term}
            for score, name, term in ranked[:3]
        ]
        if ranked:
            best_score, best_name, _ = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0
            unambiguous = best_score >= 0.94 or best_score - second_score >= self.fuzzy_margin
            if best_score >= self.fuzzy_threshold and unambiguous:
                resolved = self._resolved(
                    raw,
                    best_name,
                    "fuzzy",
                    best_score,
                    model_confidence,
                )
                resolved.alternatives = alternatives
                return resolved
        return DiagnosisResolution(
            raw_name=raw,
            canonical_name=None,
            parent_name=None,
            method="unresolved",
            confidence=0.0,
            model_confidence=model_confidence,
            alternatives=alternatives,
        )

    def raw_candidate_names(self, result: Any) -> List[str]:
        return [name for name, _ in self.extract_candidates(result)]

    def extract_candidates(self, result: Any) -> List[Tuple[str, float]]:
        if not isinstance(result, dict):
            return []
        values: List[Any] = []
        for key in (
            "diagnosis",
            "diagnoses",
            "diagnosis_candidates",
            "open_diagnosis_candidates",
            "candidate_diagnoses",
            "differential_diagnoses",
        ):
            current = result.get(key)
            if current:
                values.extend(current if isinstance(current, list) else [current])

        candidates: List[Tuple[str, float]] = []
        seen: set[str] = set()
        for item in values:
            confidence = 1.0
            if isinstance(item, dict):
                name = item.get("name") or item.get("disease") or item.get("diagnosis")
                try:
                    confidence = float(item.get("confidence", 1.0) or 1.0)
                except (TypeError, ValueError):
                    confidence = 1.0
            else:
                name = item
            text = str(name or "").strip()
            if text and text not in seen:
                seen.add(text)
                candidates.append((text, max(0.0, min(1.0, confidence))))
        return candidates[: self.max_raw_candidates]

    @classmethod
    def clean_name(cls, value: Any) -> str:
        text = str(value or "").strip().strip("，,。；;：:[]【】\"'")
        previous = None
        while text and text != previous:
            previous = text
            text = cls._LEADING_QUALIFIERS.sub("", text)
            text = cls._TRAILING_QUALIFIERS.sub("", text)
        if text.startswith("（") and text.endswith("）"):
            text = text[1:-1].strip()
        return re.sub(r"\s+", "", text)

    def _build_search_terms(self) -> List[Tuple[str, str]]:
        terms: List[Tuple[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        registry = getattr(self.knowledge, "entity_registry", None)
        if registry is not None:
            for entity in getattr(registry, "entities_by_id", {}).values():
                if not getattr(entity, "submittable", False):
                    continue
                canonical = getattr(entity, "display_name", "") or getattr(entity, "canonical_name", "")
                for alias in [entity.canonical_name, entity.submission_name] + list(entity.aliases or []):
                    clean = self.clean_name(alias).lower()
                    key = (clean, canonical)
                    if clean and key not in seen:
                        seen.add(key)
                        terms.append(key)
        for alias, canonical in self.knowledge.aliases.items():
            if canonical not in set(self.knowledge.allowed_names):
                continue
            clean = self.clean_name(alias).lower()
            key = (clean, canonical)
            if clean and key not in seen:
                seen.add(key)
                terms.append(key)
        return terms

    def _hierarchy_match(self, clean: str) -> Optional[str]:
        allowed = sorted(self.knowledge.allowed_names, key=len, reverse=True)
        for name in allowed:
            if len(name) < 2 or clean == name:
                continue
            if clean.endswith(name):
                prefix = clean[: -len(name)]
                if prefix and len(prefix) <= 8 and not any(marker in prefix for marker in self._COMPOUND_MARKERS):
                    return name
            if clean.startswith(name):
                suffix = clean[len(name):]
                if suffix in self._PHASE_SUFFIXES:
                    return name
        return None

    def _alias_contains_match(self, clean: str) -> Optional[Tuple[str, float]]:
        query = clean.lower()
        matches: List[Tuple[int, str, str]] = []
        for term, canonical in self._search_terms:
            if len(term) < 4:
                continue
            if term == query:
                continue
            if term in query:
                matches.append((len(term), canonical, term))
        if not matches:
            return None
        matches.sort(reverse=True)
        longest = matches[0][0]
        best = [item for item in matches if item[0] == longest]
        canonicals = {item[1] for item in best}
        if len(canonicals) > 1:
            specific = self._choose_specific_alias_match([item[1] for item in best])
            if not specific:
                return None
            return specific, min(0.93, 0.70 + longest / max(1, len(query)) * 0.35)
        confidence = min(0.93, 0.70 + longest / max(1, len(query)) * 0.35)
        return best[0][1], confidence

    def _choose_specific_alias_match(self, names: Sequence[str]) -> Optional[str]:
        unique = list(dict.fromkeys(name for name in names if name))
        if len(unique) <= 1:
            return unique[0] if unique else None
        entries = {name: self.knowledge.get(name) for name in unique}
        for name, entry in entries.items():
            parent = str(entry.get("parent_diagnosis") or "")
            if parent and parent in unique:
                return name
        ranked = sorted(
            unique,
            key=lambda item: float(entries[item].get("specificity", 0.5) or 0.5),
            reverse=True,
        )
        best_specificity = float(entries[ranked[0]].get("specificity", 0.5) or 0.5)
        second_specificity = float(entries[ranked[1]].get("specificity", 0.5) or 0.5)
        if best_specificity >= second_specificity + 0.04:
            return ranked[0]
        return None

    def _rank_terms(self, clean: str) -> List[Tuple[float, str, str]]:
        query = clean.lower()
        by_name: Dict[str, Tuple[float, str]] = {}
        for term, canonical in self._search_terms:
            score = max(
                SequenceMatcher(None, query, term).ratio(),
                _bigram_dice(query, term),
            )
            current = by_name.get(canonical)
            if current is None or score > current[0]:
                by_name[canonical] = (score, term)
        ranked = [(score, name, term) for name, (score, term) in by_name.items()]
        ranked.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
        return ranked

    def _resolve_entity(self, value: Any) -> Any:
        resolver = getattr(self.knowledge, "resolve_entity", None)
        if not callable(resolver):
            return None
        return resolver(value)

    def _resolved_entity(
        self,
        raw: str,
        entity: Any,
        method: str,
        confidence: float,
        model_confidence: float,
    ) -> DiagnosisResolution:
        canonical = str(
            getattr(entity, "submission_name", "")
            or getattr(entity, "canonical_name", "")
            or ""
        )
        resolution = self._resolved(raw, canonical, method, confidence, model_confidence)
        resolution.entity_id = str(getattr(entity, "entity_id", "") or "")
        resolution.canonical_name = canonical or str(getattr(entity, "canonical_name", "") or "")
        resolution.submission_name = str(
            getattr(entity, "submission_name", "")
            or getattr(entity, "canonical_name", "")
            or resolution.canonical_name
            or ""
        )
        resolution.submittable = bool(getattr(entity, "submittable", False))
        if not resolution.parent_name:
            resolution.parent_name = str(getattr(entity, "parent_name", "") or "") or None
        return resolution

    def _resolved(
        self,
        raw: str,
        canonical: str,
        method: str,
        confidence: float,
        model_confidence: float,
    ) -> DiagnosisResolution:
        entry = self.knowledge.get(canonical)
        parent = str(entry.get("parent_diagnosis") or "").strip() or None
        entity = self._resolve_entity(canonical)
        return DiagnosisResolution(
            raw_name=raw,
            canonical_name=canonical,
            parent_name=parent,
            method=method,
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            model_confidence=round(max(0.0, min(1.0, model_confidence)), 4),
            entity_id=str(getattr(entity, "entity_id", "") or entry.get("entity_id") or ""),
            submission_name=str(
                getattr(entity, "submission_name", "")
                or entry.get("submission_name")
                or canonical
            ),
            submittable=bool(
                getattr(entity, "submittable", entry.get("submittable", True))
            ),
        )


def _bigram_dice(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if len(left) == 1 or len(right) == 1:
        return 0.0
    left_pairs = {left[index:index + 2] for index in range(len(left) - 1)}
    right_pairs = {right[index:index + 2] for index in range(len(right) - 1)}
    if not left_pairs or not right_pairs:
        return 0.0
    return 2.0 * len(left_pairs & right_pairs) / (len(left_pairs) + len(right_pairs))
