from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class ExternalMedicalResult:
    title: str
    score: float
    summary: str = ""
    candidate_diseases: List[str] = field(default_factory=list)
    recommended_exams: List[str] = field(default_factory=list)
    source_label: str = "offline_unreviewed_seed"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExternalMedicalKnowledgeRetriever:
    """Pluggable external medical retrieval boundary.

    The default provider is intentionally an offline unreviewed seed file so
    tests stay deterministic. A network provider can implement the same search
    method and must still return unreviewed results unless a human promotes them
    into the formal knowledge layer.
    """

    def __init__(self, config: Dict[str, Any], ref_dir: str = "data/ref_data"):
        raw = (config or {}).get("external_medical_retrieval", {}) or {}
        self.enabled = bool(raw.get("enabled", True))
        self.provider = str(raw.get("provider") or "offline_seed")
        self.top_k = _positive_int(raw.get("top_k"), 5)
        self.ref_dir = str(ref_dir or (config or {}).get("ref_data_dir") or "data/ref_data")
        self.seed_path = str(
            raw.get("seed_path")
            or os.path.join(self.ref_dir, "medical_knowledge", "external_retrieval_seeds.json")
        )
        self._records: Optional[List[Dict[str, Any]]] = None

    def search(
        self,
        retrieval_views: Optional[Sequence[Any]] = None,
        query_terms: Optional[Iterable[Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[ExternalMedicalResult]:
        if not self.enabled:
            return []
        terms = _dedupe(list(query_terms or []) + _terms_from_views(retrieval_views or []))
        if not terms:
            return []
        records = self._load_records()
        results: List[ExternalMedicalResult] = []
        for record in records:
            score = self._score_record(record, terms)
            if score <= 0:
                continue
            results.append(
                ExternalMedicalResult(
                    title=str(record.get("title") or ""),
                    score=round(score, 4),
                    summary=str(record.get("summary") or ""),
                    candidate_diseases=[
                        str(item) for item in record.get("candidate_diseases", []) or [] if str(item)
                    ],
                    recommended_exams=[
                        str(item) for item in record.get("recommended_exams", []) or [] if str(item)
                    ],
                    source_label=str(record.get("source_label") or self.provider),
                    metadata={
                        "unreviewed_external": True,
                        "provider": self.provider,
                        "mechanism_ids": list(record.get("mechanism_ids", []) or []),
                        "family_id": str(record.get("family_id") or ""),
                        "body_system": str(record.get("body_system") or ""),
                    },
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[: _positive_int(top_k, self.top_k)]

    def _score_record(self, record: Dict[str, Any], terms: Sequence[str]) -> float:
        identity = " ".join(
            str(item)
            for item in [
                record.get("title", ""),
                record.get("family_id", ""),
                record.get("body_system", ""),
                " ".join(record.get("mechanism_ids", []) or []),
                " ".join(record.get("aliases", []) or []),
                " ".join(record.get("candidate_diseases", []) or []),
                " ".join(record.get("terms", []) or []),
            ]
        ).lower()
        summary = str(record.get("summary") or "").lower()
        score = 0.0
        for term in terms:
            needle = str(term or "").strip().lower()
            if not needle:
                continue
            if needle in identity:
                score += 0.34 if len(needle) >= 4 else 0.14
            elif needle in summary:
                score += 0.06
            else:
                pieces = [part for part in needle.replace("_", " ").split() if len(part) >= 4]
                if pieces and any(piece in identity for piece in pieces):
                    score += 0.08
        if score <= 0:
            return 0.0
        return min(0.99, 0.28 + score)

    def _load_records(self) -> List[Dict[str, Any]]:
        if self._records is not None:
            return self._records
        try:
            with open(self.seed_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, TypeError, ValueError):
            payload = {}
        records = payload.get("records", []) if isinstance(payload, dict) else []
        self._records = [dict(item) for item in records if isinstance(item, dict)]
        return self._records


def _terms_from_views(views: Sequence[Any]) -> List[str]:
    terms: List[str] = []
    for view in views:
        if isinstance(view, dict):
            terms.extend(str(item) for item in view.get("terms", []) or [] if str(item))
            if view.get("query"):
                terms.append(str(view.get("query")))
            continue
        terms.extend(str(item) for item in getattr(view, "terms", []) or [] if str(item))
        query = getattr(view, "query", "")
        if query:
            terms.append(str(query))
    return _dedupe(terms)


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


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default
