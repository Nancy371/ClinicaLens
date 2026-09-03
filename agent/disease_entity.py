"""Canonical disease entity registry.

The diagnosis pipeline still keeps human-readable disease names for prompts,
logs, and the backend submission payload. Internally, this registry provides a
stable entity identity so aliases from LLM, RAG, and mechanism reasoning merge
before ranking and submission.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class DiseaseEntity:
    entity_id: str
    canonical_name: str
    submission_name: str = ""
    aliases: List[str] = field(default_factory=list)
    submittable: bool = False
    source_kind: str = ""
    parent_entity_id: str = ""
    parent_name: str = ""
    icd10: str = ""
    department: str = ""
    diagnosis_type: str = "disease"
    body_system: str = ""
    disease_family: str = ""
    family: str = ""
    exam_bundle: List[str] = field(default_factory=list)
    discriminating_exam_bundle: List[str] = field(default_factory=list)
    treatment_bundle: List[str] = field(default_factory=list)
    evidence_profile: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.submission_name or self.canonical_name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiseaseEntityRegistry:
    """Resolve disease strings to stable entity objects."""

    def __init__(self, ref_dir: str = "data/ref_data"):
        self.ref_dir = ref_dir
        self.path = os.path.join(ref_dir, "disease_entities.json")
        self.entities_by_id: Dict[str, DiseaseEntity] = {}
        self._key_to_id: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        payload = _read_json(self.path, {})
        for item in payload.get("entities", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("entity_id") or "").strip()
            canonical_name = str(item.get("canonical_name") or "").strip()
            if not entity_id or not canonical_name:
                continue
            entity = DiseaseEntity(
                entity_id=entity_id,
                canonical_name=canonical_name,
                submission_name=str(item.get("submission_name") or canonical_name).strip(),
                aliases=_dedupe_texts(item.get("aliases") or []),
                submittable=bool(item.get("submittable", False)),
                source_kind=str(item.get("source_kind") or ""),
                parent_entity_id=str(item.get("parent_entity_id") or ""),
                parent_name=str(item.get("parent_name") or ""),
                icd10=str(item.get("icd10") or ""),
                department=str(item.get("department") or ""),
                diagnosis_type=str(item.get("diagnosis_type") or "disease"),
                body_system=str(item.get("body_system") or ""),
                disease_family=str(item.get("disease_family") or item.get("family") or ""),
                family=str(item.get("family") or item.get("disease_family") or ""),
                exam_bundle=_dedupe_texts(item.get("exam_bundle") or []),
                discriminating_exam_bundle=_dedupe_texts(
                    item.get("discriminating_exam_bundle") or []
                ),
                treatment_bundle=_dedupe_texts(item.get("treatment_bundle") or []),
                evidence_profile=dict(item.get("evidence_profile") or {}),
                metadata=dict(item.get("metadata") or {}),
            )
            if not entity.family:
                entity.family = entity.disease_family
            self.entities_by_id[entity.entity_id] = entity
            self._index(entity)

    def _index(self, entity: DiseaseEntity) -> None:
        for value in [entity.entity_id, entity.canonical_name, entity.submission_name]:
            self._add_key(value, entity.entity_id, override=True)
        for alias in entity.aliases:
            self._add_key(alias, entity.entity_id)

    def _add_key(self, value: Any, entity_id: str, *, override: bool = False) -> None:
        for key in _candidate_keys(value):
            if override:
                self._key_to_id[key] = entity_id
            else:
                self._key_to_id.setdefault(key, entity_id)

    def get(self, entity_id: Any) -> Optional[DiseaseEntity]:
        text = str(entity_id or "").strip()
        if not text:
            return None
        if text in self.entities_by_id:
            return self.entities_by_id[text]
        return self.resolve(text)

    def resolve(self, value: Any) -> Optional[DiseaseEntity]:
        for key in _candidate_keys(value):
            entity_id = self._key_to_id.get(key)
            if entity_id:
                return self.entities_by_id.get(entity_id)
        return None

    def entity_id_for(self, value: Any) -> str:
        entity = self.get(value)
        return entity.entity_id if entity else ""

    def canonical_name_for(self, value: Any) -> str:
        entity = self.get(value)
        return entity.canonical_name if entity else ""

    def submission_name_for(self, value: Any) -> str:
        entity = self.get(value)
        return entity.display_name if entity else ""

    def is_submittable(self, value: Any) -> bool:
        entity = self.get(value)
        return bool(entity and entity.submittable)

    def aliases_for(self, value: Any) -> List[str]:
        entity = self.get(value)
        return list(entity.aliases) if entity else []

    def all_names(self, *, submittable_only: bool = False) -> List[str]:
        names: List[str] = []
        for entity in self.entities_by_id.values():
            if submittable_only and not entity.submittable:
                continue
            for value in [entity.canonical_name, entity.submission_name] + list(entity.aliases):
                text = str(value or "").strip()
                if text and text not in names:
                    names.append(text)
        return names

    def exam_bundle_for(self, value: Any) -> List[str]:
        entity = self.get(value)
        return list(entity.exam_bundle) if entity else []

    def discriminating_exam_bundle_for(self, value: Any) -> List[str]:
        entity = self.get(value)
        if not entity:
            return []
        return list(entity.discriminating_exam_bundle or entity.exam_bundle)

    def resolution_record(self, raw_name: Any) -> Dict[str, Any]:
        entity = self.resolve(raw_name)
        if not entity:
            return {"raw_name": str(raw_name or ""), "entity_id": "", "resolved": False}
        return {
            "raw_name": str(raw_name or ""),
            "entity_id": entity.entity_id,
            "canonical_name": entity.canonical_name,
            "submission_name": entity.display_name,
            "submittable": bool(entity.submittable),
            "source_kind": entity.source_kind,
            "resolved": True,
        }


def _candidate_keys(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    compact = _compact(text)
    keys = [text, text.lower(), compact, compact.lower()]
    return _dedupe_texts(keys)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip())


def _dedupe_texts(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
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
    except Exception:
        return default
