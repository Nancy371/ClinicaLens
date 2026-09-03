"""Detect conflicts between LLM reasoning and structured clinical evidence."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .clinical_evidence import EvidenceBundle, Observation, ReasoningEvidenceAdapter


_EXCLUSION_RE = re.compile(
    r"(?:排除|除外|不支持|未提示|未见|未发现|无证据|证据不足|阴性|不能解释|难以解释|缺乏)"
)
_EXCLUSION_GUARD_RE = re.compile(
    r"(?:不能排除|不能除外|不排除|不除外|难以排除|尚不能排除|未能排除|待排|待除外|待鉴别|鉴别诊断|需要鉴别|需鉴别)"
)
_SUPPORT_RE = re.compile(
    r"(?:支持|符合|提示|高度提示|强烈提示|最符合|首要诊断|可解释|统一解释|归因|由于|导致|继发)"
)
_CLAUSE_SPLIT_RE = re.compile(r"[。；;！!？?\n]")

_CONFLICT_ACTION = "defer_primary_and_order_discriminating_exams"
_CONFLICT_TYPE = "reasoning_structured_polarity_conflict"
_REASONING_SOURCE = "reasoning_inference"

_LOW_MAGNESIUM_ADJUDICATION_EXAMS = (
    "血清电解质",
    "24小时尿电解质检测",
    "镁负荷试验",
    "维生素D检测",
    "甲状旁腺激素检测（PTH）",
    "X线检查",
)


@dataclass
class EvidenceConflict:
    conflict_type: str
    affected_diagnosis: str
    finding: str
    reasoning_text: str
    structured_sources: List[Dict[str, Any]] = field(default_factory=list)
    adjudication_exams: List[str] = field(default_factory=list)
    action: str = _CONFLICT_ACTION
    reasoning_polarity: str = "negative"
    structured_polarity: str = "positive"
    status: str = "unresolved"
    severity: str = "blocking"
    reason: str = (
        "reasoning excludes diagnosis while non-reasoning structured evidence "
        "strongly supports it"
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceConflictArbiter:
    """Raise blocking audit events when reasoning contradicts hard evidence."""

    def __init__(self, knowledge: Optional[Any] = None):
        self.knowledge = knowledge
        self.reasoning_adapter = ReasoningEvidenceAdapter()

    def detect(
        self,
        llm_result: Optional[Dict[str, Any]],
        evidence: EvidenceBundle,
        candidates: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        if not isinstance(llm_result, dict) or not candidates:
            return []
        reasoning_texts = self.reasoning_adapter.reasoning_texts(llm_result)
        if not reasoning_texts:
            return []

        observations = list(getattr(evidence, "observations", []) or [])
        conflicts: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates:
            diagnosis = str(getattr(candidate, "diagnosis", "") or "").strip()
            if not diagnosis:
                continue
            reasoning_clause = self._exclusion_clause_for_candidate(
                reasoning_texts,
                candidate,
            )
            if not reasoning_clause:
                continue
            support_by_finding = self._strong_structured_support(
                candidate,
                observations,
            )
            if not support_by_finding:
                continue
            finding = self._best_supported_finding(candidate, support_by_finding)
            if not finding:
                continue
            key = (diagnosis, finding)
            if key in seen:
                continue
            seen.add(key)
            conflict = EvidenceConflict(
                conflict_type=_CONFLICT_TYPE,
                affected_diagnosis=diagnosis,
                finding=finding,
                reasoning_text=reasoning_clause,
                structured_sources=support_by_finding[finding],
                adjudication_exams=self._adjudication_exams(candidate, candidates),
            )
            conflicts.append(conflict.to_dict())
        return conflicts

    def _exclusion_clause_for_candidate(
        self,
        reasoning_texts: Sequence[str],
        candidate: Any,
    ) -> str:
        aliases = self._candidate_aliases(candidate)
        for text in reasoning_texts:
            for clause in self._clauses(text):
                if not self._has_alias(clause, aliases):
                    continue
                if _EXCLUSION_GUARD_RE.search(clause):
                    continue
                if not self._directly_excludes_alias(clause, aliases):
                    continue
                if self._supports_alias(clause, aliases):
                    continue
                if _EXCLUSION_RE.search(clause):
                    return clause[:500]
        return ""

    @staticmethod
    def _directly_excludes_alias(text: str, aliases: Iterable[str]) -> bool:
        for alias in aliases:
            value = str(alias or "").strip()
            if not value:
                continue
            escaped = re.escape(value)
            patterns = (
                rf"(?:排除|除外|不支持|未提示|未见|未发现|无证据|证据不足|不能解释|难以解释|缺乏)[^。；;！!？?\n]{{0,24}}{escaped}",
                rf"{escaped}[^。；;！!？?\n]{{0,24}}(?:被)?(?:排除|除外)",
                rf"{escaped}[^。；;！!？?\n]{{0,24}}(?:不支持|未提示|未见|未发现|无证据|证据不足|不能解释|难以解释|缺乏)",
            )
            if any(re.search(pattern, text) for pattern in patterns):
                return True
        return False

    @staticmethod
    def _supports_alias(text: str, aliases: Iterable[str]) -> bool:
        if not _SUPPORT_RE.search(text):
            return False
        for alias in aliases:
            value = str(alias or "").strip()
            if not value:
                continue
            escaped = re.escape(value)
            patterns = (
                rf"(?:支持|符合|提示|高度提示|强烈提示|最符合|首要诊断)[^。；;！!？?\n]{{0,24}}{escaped}",
                rf"{escaped}[^。；;！!？?\n]{{0,24}}(?:支持|符合|提示|可解释|统一解释)",
                rf"(?:可被|可由|均可由|均可被|全部可由|全部可被)[^。；;！!？?\n]{{0,18}}{escaped}[^。；;！!？?\n]{{0,18}}(?:解释|导致|引起)",
            )
            if any(re.search(pattern, text) for pattern in patterns):
                return True
        return False

    def _candidate_aliases(self, candidate: Any) -> List[str]:
        diagnosis = str(getattr(candidate, "diagnosis", "") or "").strip()
        aliases: List[str] = []
        if diagnosis:
            aliases.append(diagnosis)
        entry: Dict[str, Any] = {}
        if self.knowledge is not None and diagnosis:
            try:
                entry = self.knowledge.get(diagnosis) or {}
            except Exception:
                entry = {}
        for alias in entry.get("aliases", []) or []:
            text = str(alias or "").strip()
            if text:
                aliases.append(text)
        normalized = None
        if self.knowledge is not None and diagnosis:
            try:
                normalized = self.knowledge.normalize_name(diagnosis)
            except Exception:
                normalized = None
        if normalized:
            aliases.append(str(normalized))
        return list(dict.fromkeys(aliases))

    @staticmethod
    def _clauses(text: str) -> List[str]:
        raw = " ".join(str(text or "").split())
        if not raw:
            return []
        clauses = [item.strip() for item in _CLAUSE_SPLIT_RE.split(raw) if item.strip()]
        if raw not in clauses:
            clauses.append(raw)
        return clauses

    @staticmethod
    def _has_alias(text: str, aliases: Iterable[str]) -> bool:
        lowered = str(text or "").lower()
        for alias in aliases:
            value = str(alias or "").strip()
            if not value:
                continue
            if value.lower() in lowered:
                return True
        return False

    def _strong_structured_support(
        self,
        candidate: Any,
        observations: Sequence[Observation],
    ) -> Dict[str, List[Dict[str, Any]]]:
        matched = set(getattr(candidate, "matched_evidence", []) or [])
        if not matched:
            return {}
        core = set(getattr(candidate, "core_matched_evidence", []) or [])
        diagnostic = set(getattr(candidate, "diagnostic_matched_evidence", []) or [])
        support: Dict[str, List[Dict[str, Any]]] = {}
        for item in observations:
            finding = str(getattr(item, "finding", "") or "")
            if (
                not finding
                or finding not in matched
                or getattr(item, "polarity", "") != "positive"
                or getattr(item, "source", "") == _REASONING_SOURCE
                or getattr(item, "shadowed_by", "")
            ):
                continue
            if not self._is_strong_structured_observation(item, core, diagnostic):
                continue
            support.setdefault(finding, []).append(self._observation_source(item))
        return support

    @staticmethod
    def _is_strong_structured_observation(
        item: Observation,
        core: set[str],
        diagnostic: set[str],
    ) -> bool:
        finding = str(getattr(item, "finding", "") or "")
        evidence_level = str(getattr(item, "evidence_level", "") or "")
        try:
            info = float(getattr(item, "information_value", 0.0) or 0.0)
        except (TypeError, ValueError):
            info = 0.0
        try:
            confidence = float(getattr(item, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if finding.startswith(("field:", "symptom:")):
            return False
        if finding.startswith("diagnosis:") and finding not in diagnostic:
            return bool(info >= 0.82 and confidence >= 0.9)
        if finding in diagnostic:
            return True
        if evidence_level == "diagnostic_pattern":
            return True
        if finding in core and (info >= 0.55 or evidence_level in {"specific", "supportive"}):
            return True
        if (
            finding in core
            and (getattr(item, "value", None) is not None or getattr(item, "direction", ""))
        ):
            return True
        if info >= 0.85 and evidence_level in {"specific", "diagnostic_pattern"}:
            return True
        return bool(evidence_level == "specific" and confidence >= 0.9)

    @staticmethod
    def _observation_source(item: Observation) -> Dict[str, Any]:
        return {
            "finding": str(getattr(item, "finding", "") or ""),
            "source": str(getattr(item, "source", "") or ""),
            "source_text": str(
                getattr(item, "source_text", "")
                or getattr(item, "raw_text", "")
                or ""
            )[:500],
            "raw_text": str(getattr(item, "raw_text", "") or "")[:500],
            "confidence": float(getattr(item, "confidence", 0.0) or 0.0),
            "evidence_level": str(getattr(item, "evidence_level", "") or ""),
            "information_value": float(
                getattr(item, "information_value", 0.0) or 0.0
            ),
            "field_path": str(getattr(item, "field_path", "") or ""),
        }

    def _best_supported_finding(
        self,
        candidate: Any,
        support_by_finding: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        diagnostic = set(getattr(candidate, "diagnostic_matched_evidence", []) or [])
        core = set(getattr(candidate, "core_matched_evidence", []) or [])

        def key(finding: str) -> tuple[float, float, float, str]:
            source_values = support_by_finding.get(finding) or []
            best_info = max(
                (float(item.get("information_value", 0.0) or 0.0) for item in source_values),
                default=0.0,
            )
            best_conf = max(
                (float(item.get("confidence", 0.0) or 0.0) for item in source_values),
                default=0.0,
            )
            return (
                1.0 if finding in diagnostic or finding.startswith("diagnosis:") else 0.0,
                1.0 if finding in core else 0.0,
                best_info + best_conf * 0.1,
                finding,
            )

        return max(support_by_finding, key=key) if support_by_finding else ""

    def _adjudication_exams(
        self,
        candidate: Any,
        candidates: Sequence[Any],
    ) -> List[str]:
        exams: List[str] = []

        def add_many(items: Sequence[Any]) -> None:
            for item in items or []:
                text = str(item or "").strip()
                if text and text not in exams:
                    exams.append(text)

        add_many(self._profile_exams(candidate))
        if str(getattr(candidate, "diagnosis", "") or "") == "低镁血症":
            add_many(_LOW_MAGNESIUM_ADJUDICATION_EXAMS)

        affected_score = self._candidate_score(candidate)
        close_cutoff = affected_score - 0.26
        contenders = [
            item
            for item in candidates[:8]
            if item is not candidate
            and not getattr(item, "hard_contradiction", False)
            and self._candidate_score(item) >= close_cutoff
            and (
                getattr(item, "diagnostic_matched_evidence", None)
                or getattr(item, "core_matched_evidence", None)
                or float(getattr(item, "specificity", 0.0) or 0.0) >= 0.85
            )
        ]
        for contender in contenders[:3]:
            add_many(self._profile_exams(contender))
        return exams[:10]

    def _profile_exams(self, candidate: Any) -> List[str]:
        diagnosis = str(getattr(candidate, "diagnosis", "") or "")
        entry: Dict[str, Any] = {}
        if self.knowledge is not None and diagnosis:
            try:
                entry = self.knowledge.get(diagnosis) or {}
            except Exception:
                entry = {}
        exams: List[str] = []
        for key in (
            "discriminating_exams",
            "strong_verification_exams",
            "required_exams",
        ):
            for item in entry.get(key, []) or []:
                text = str(item or "").strip()
                if text and text not in exams:
                    exams.append(text)
        return exams

    @staticmethod
    def _candidate_score(candidate: Any) -> float:
        try:
            return float(getattr(candidate, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
