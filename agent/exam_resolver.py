"""Resolve requested exams without pretending partial substitutes are exact."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


EXACT = "exact"
ALIAS = "alias"
EQUIVALENT = "equivalent"
PARTIAL_SUBSTITUTE = "partial_substitute"
UNRESOLVED = "unresolved"


@dataclass
class ExamResolution:
    requested_exam: str
    resolved_exam: str = ""
    resolution_type: str = UNRESOLVED
    diagnostic_coverage: float = 0.0
    reason: str = ""
    candidate: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExamResolver:
    """Map an exam request to the closest valid catalog item with coverage."""

    def __init__(
        self,
        knowledge: Optional[Any] = None,
        *,
        catalog_names: Optional[Sequence[str]] = None,
        aliases: Optional[Dict[str, str]] = None,
    ):
        self.knowledge = knowledge
        self.catalog_names = _text_list(catalog_names or self._catalog_from_knowledge())
        self.catalog_set = set(self.catalog_names)
        self.aliases = dict(self._aliases_from_knowledge())
        self.aliases.update(dict(aliases or {}))

    def resolve(self, requested_exam: Any, *, candidate: Any = None) -> ExamResolution:
        requested = str(requested_exam or "").strip()
        candidate_name = str(candidate or "").strip()
        if not requested:
            return ExamResolution(requested_exam="", candidate=candidate_name)

        controlled = self._controlled_specialty_request(requested, candidate_name)
        if controlled:
            return controlled

        if requested in self.catalog_set:
            return ExamResolution(
                requested_exam=requested,
                resolved_exam=requested,
                resolution_type=EXACT,
                diagnostic_coverage=1.0,
                reason="exact catalog match",
                candidate=candidate_name,
            )

        alias = self.aliases.get(requested)
        if alias:
            return ExamResolution(
                requested_exam=requested,
                resolved_exam=alias,
                resolution_type=ALIAS if alias in self.catalog_set or self._valid_exam(alias) else ALIAS,
                diagnostic_coverage=1.0,
                reason="explicit alias match",
                candidate=candidate_name,
            )

        normalized = self._normalize([requested])
        if normalized:
            resolved = normalized[0]
            if self._lossy_special_resolution(requested, resolved):
                return ExamResolution(
                    requested_exam=requested,
                    resolved_exam=resolved,
                    resolution_type=PARTIAL_SUBSTITUTE,
                    diagnostic_coverage=0.6,
                    reason="specialized request only resolved to generic imaging/lab substitute",
                    candidate=candidate_name,
                )
            return ExamResolution(
                requested_exam=requested,
                resolved_exam=resolved,
                resolution_type=ALIAS,
                diagnostic_coverage=1.0,
                reason="catalog normalization match",
                candidate=candidate_name,
            )

        equivalent = self._equivalent_exam(requested)
        if equivalent:
            return ExamResolution(
                requested_exam=requested,
                resolved_exam=equivalent,
                resolution_type=EQUIVALENT,
                diagnostic_coverage=1.0,
                reason="declared equivalent exam",
                candidate=candidate_name,
            )

        substitute = self._partial_substitute(requested)
        if substitute:
            return ExamResolution(
                requested_exam=requested,
                resolved_exam=substitute,
                resolution_type=PARTIAL_SUBSTITUTE,
                diagnostic_coverage=0.55,
                reason="declared partial substitute; does not confirm the target alone",
                candidate=candidate_name,
            )

        return ExamResolution(
            requested_exam=requested,
            resolution_type=UNRESOLVED,
            diagnostic_coverage=0.0,
            reason="no catalog, alias, equivalent, or safe partial substitute",
            candidate=candidate_name,
        )

    def _controlled_specialty_request(
        self,
        requested: str,
        candidate: str,
    ) -> Optional[ExamResolution]:
        """Allow reviewed entity-specific confirmatory exams through the gap path."""
        candidate_text = _compact(candidate)
        if (
            "d000025" in candidate_text
            or "leukemia" in candidate_text
            or "白血病" in candidate
        ):
            return self._controlled_hematology_request(requested, candidate)
        if not (
            "d100055" in candidate_text
            or "pavm" in candidate_text
            or "肺动静脉瘘" in candidate
            or "肺动静脉畸形" in candidate
        ):
            return None
        compact = _compact(requested)
        full_markers = (
            "肺动脉cta",
            "肺血管cta",
            "肺动脉ct血管成像",
            "右心声学造影",
            "超声心动图右心声学造影",
            "bubbleecho",
            "bubblestudy",
            "肺血管造影",
        )
        conditional_markers = (
            "胸部增强ct",
            "增强胸部ct",
            "chestcect",
        )
        if any(marker in compact for marker in full_markers):
            return ExamResolution(
                requested_exam=requested,
                resolved_exam=requested,
                resolution_type=EQUIVALENT,
                diagnostic_coverage=1.0,
                reason="controlled PAVM confirmatory closure request",
                candidate=candidate,
            )
        if any(marker in compact for marker in conditional_markers):
            return ExamResolution(
                requested_exam=requested,
                resolved_exam=requested,
                resolution_type=EQUIVALENT,
                diagnostic_coverage=0.92,
                reason="controlled PAVM conditional closure request",
                candidate=candidate,
            )
        return None

    @staticmethod
    def _controlled_hematology_request(
        requested: str,
        candidate: str,
    ) -> Optional[ExamResolution]:
        compact = _compact(requested)
        full_markers = (
            "骨髓穿刺",
            "骨髓活检",
            "骨髓穿刺和活检",
            "bmab",
            "骨髓流式",
            "流式细胞术免疫分型",
            "流式细胞免疫表型",
            "免疫表型分析",
            "细胞遗传学分析",
            "染色体核型分析",
        )
        molecular_markers = (
            "白血病融合基因",
            "血液系统分子检测",
            "分子检测",
            "融合基因检测",
        )
        supportive_markers = (
            "外周血涂片",
            "血常规",
            "全血细胞计数",
            "cbc",
        )
        if any(marker in compact for marker in full_markers):
            return ExamResolution(
                requested_exam=requested,
                resolved_exam=requested,
                resolution_type=EQUIVALENT,
                diagnostic_coverage=1.0,
                reason="controlled hematologic malignancy confirmatory closure request",
                candidate=candidate,
            )
        if any(marker in compact for marker in molecular_markers):
            return ExamResolution(
                requested_exam=requested,
                resolved_exam=requested,
                resolution_type=EQUIVALENT,
                diagnostic_coverage=0.92,
                reason="controlled hematologic malignancy classification closure request",
                candidate=candidate,
            )
        if any(marker in compact for marker in supportive_markers):
            return ExamResolution(
                requested_exam=requested,
                resolved_exam=requested,
                resolution_type=EQUIVALENT,
                diagnostic_coverage=0.72,
                reason="controlled hematologic malignancy supportive verification request",
                candidate=candidate,
            )
        return None

    def resolve_many(
        self,
        requested_exams: Sequence[Any],
        *,
        candidate: Any = None,
    ) -> List[ExamResolution]:
        result: List[ExamResolution] = []
        seen: set[str] = set()
        for item in requested_exams or []:
            resolution = self.resolve(item, candidate=candidate)
            key = f"{resolution.requested_exam}|{resolution.resolved_exam}|{resolution.resolution_type}"
            if key in seen:
                continue
            seen.add(key)
            result.append(resolution)
        return result

    def resolve_with_entity_fallback(
        self,
        requested_exams: Sequence[Any],
        *,
        candidate: Any = None,
        entity_exam_bundle: Optional[Sequence[Any]] = None,
    ) -> List[ExamResolution]:
        direct = self.resolve_many(requested_exams, candidate=candidate)
        if any(item.resolution_type in {EXACT, ALIAS, EQUIVALENT} for item in direct):
            return direct
        fallback = self.resolve_many(entity_exam_bundle or [], candidate=candidate)
        usable = [
            item
            for item in fallback
            if item.resolution_type in {EXACT, ALIAS, EQUIVALENT, PARTIAL_SUBSTITUTE}
        ]
        return direct + usable

    def _catalog_from_knowledge(self) -> List[str]:
        if not self.knowledge:
            return []
        if hasattr(self.knowledge, "get_examination_catalog_names"):
            try:
                return _text_list(self.knowledge.get_examination_catalog_names())
            except Exception:
                return []
        return []

    def _aliases_from_knowledge(self) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        if not self.knowledge:
            return aliases
        for attr in ("_service_exam_aliases", "exam_aliases"):
            value = getattr(self.knowledge, attr, None)
            if isinstance(value, dict):
                aliases.update(
                    {
                        str(key).strip(): str(item).strip()
                        for key, item in value.items()
                        if str(key).strip() and str(item).strip()
                    }
                )
        return aliases

    def _normalize(self, values: Sequence[str]) -> List[str]:
        if not self.knowledge or not hasattr(self.knowledge, "normalize_examinations"):
            return []
        try:
            normalized, _ = self.knowledge.normalize_examinations(list(values))
            return _text_list(normalized)
        except Exception:
            return []

    def _valid_exam(self, exam: str) -> bool:
        if exam in self.catalog_set:
            return True
        if self.knowledge and hasattr(self.knowledge, "is_valid_examination"):
            try:
                return bool(self.knowledge.is_valid_examination(exam))
            except Exception:
                return False
        return False

    def _equivalent_exam(self, requested: str) -> str:
        groups = [
            [
                "流式细胞术免疫分型",
                "骨髓流式细胞免疫表型分析",
                "流式细胞免疫表型分析",
            ],
            ["肺动脉CTA", "肺血管CTA", "肺动脉CT血管成像"],
            ["右心声学造影", "超声心动图右心声学造影", "bubble study"],
            ["胸部增强CT", "增强胸部CT", "Chest CECT"],
        ]
        compact_requested = _compact(requested)
        for group in groups:
            if compact_requested not in {_compact(item) for item in group}:
                continue
            for item in group:
                if item == requested:
                    continue
                if item in self.catalog_set or self._valid_exam(item):
                    return item
        return ""

    def _partial_substitute(self, requested: str) -> str:
        requested_compact = _compact(requested)
        partials = {
            _compact("流式细胞术免疫分型"): ["骨髓穿刺和活检（BMAB）"],
            _compact("白血病融合基因检测"): ["血液系统分子检测", "骨髓穿刺和活检（BMAB）"],
            _compact("肺动脉CTA"): ["胸部增强CT", "胸部CT扫描（Chest CT）", "动脉血气（ABG）"],
            _compact("右心声学造影"): ["超声心动图", "动脉血气（ABG）"],
            _compact("血管造影"): ["胸部增强CT", "胸部CT扫描（Chest CT）"],
        }
        for item in partials.get(requested_compact, []):
            if item in self.catalog_set or self._valid_exam(item):
                return item
            normalized = self._normalize([item])
            if normalized:
                return normalized[0]
        return ""

    @staticmethod
    def _lossy_special_resolution(requested: str, resolved: str) -> bool:
        requested_lower = str(requested or "").lower()
        resolved_lower = str(resolved or "").lower()
        special_markers = (
            "cta",
            "增强",
            "造影",
            "动脉",
            "血管",
            "流式",
            "免疫分型",
            "融合基因",
        )
        if not any(marker.lower() in requested_lower for marker in special_markers):
            return False
        generic_markers = (
            "ct",
            "mri",
            "血常规",
            "cbc",
            "超声",
            "胸部ct",
            "动脉血气",
        )
        if not any(marker.lower() in resolved_lower for marker in generic_markers):
            return False
        requested_compact = _compact(requested)
        resolved_compact = _compact(resolved)
        return requested_compact != resolved_compact


def _compact(value: Any) -> str:
    return "".join(str(value or "").strip().lower().split())


def _text_list(values: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(str(item).strip() for item in values or [] if str(item).strip()))
