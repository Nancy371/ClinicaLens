"""Candidate pool generation for evidence-first diagnosis ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .clinical_evidence import EvidenceBundle, EvidenceGraph, Observation
from .clinical_pattern_compiler import ClinicalPattern, ClinicalPatternCompiler
from .disease_retrieval import DiseaseRetriever
from .mechanism_reasoner import MechanismHypothesis, MechanismReasoner
from .pattern_hypothesis import (
    RECALL_BOOST,
    RECALL_PROTECTED,
    PatternRecallSignal,
)


@dataclass
class CandidateSource:
    raw_name: str
    canonical_name: str
    source: str
    entity_id: str = ""
    submission_name: str = ""
    submittable: bool = True
    prior: float = 0.0
    evidence_links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePool:
    items: List[CandidateSource] = field(default_factory=list)
    name_resolutions: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_candidates: List[str] = field(default_factory=list)
    disease_categories: List[Dict[str, Any]] = field(default_factory=list)
    open_world_candidates: List[Dict[str, Any]] = field(default_factory=list)
    mechanism_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    clinical_patterns: List[Dict[str, Any]] = field(default_factory=list)
    pattern_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    pattern_verification_results: List[Dict[str, Any]] = field(default_factory=list)
    pattern_recall_signals: List[Dict[str, Any]] = field(default_factory=list)
    pattern_candidate_admissions: List[Dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        raw_name: Any,
        canonical_name: Any,
        source: str,
        prior: float = 0.0,
        evidence_links: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entity_id: str = "",
        submission_name: str = "",
        submittable: bool = True,
    ) -> None:
        canonical = str(canonical_name or "").strip()
        raw = str(raw_name or canonical).strip()
        if not canonical:
            return
        self.items.append(
            CandidateSource(
                raw_name=raw,
                canonical_name=canonical,
                source=str(source or "unknown"),
                entity_id=str(entity_id or "").strip(),
                submission_name=str(submission_name or canonical).strip(),
                submittable=bool(submittable),
                prior=max(0.0, min(1.0, float(prior or 0.0))),
                evidence_links=list(dict.fromkeys(str(item) for item in (evidence_links or []) if str(item))),
                metadata=dict(metadata or {}),
            )
        )

    def priors(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for item in self.items:
            key = item.entity_id or item.canonical_name
            result[key] = max(result.get(key, 0.0), item.prior)
            result[item.canonical_name] = max(result.get(item.canonical_name, 0.0), item.prior)
        return result

    def add_open_world(
        self,
        raw_name: Any,
        source: str,
        prior: float = 0.0,
        evidence_links: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entity_id: str = "",
        canonical_name: str = "",
        submission_name: str = "",
        submittable: bool = False,
    ) -> None:
        raw = str(raw_name or "").strip()
        if not raw:
            return
        record = {
            "raw_name": raw,
            "entity_id": str(entity_id or "").strip(),
            "canonical_name": str(canonical_name or "").strip(),
            "submission_name": str(submission_name or canonical_name or raw).strip(),
            "source": str(source or "unknown"),
            "prior": max(0.0, min(1.0, float(prior or 0.0))),
            "submittable": bool(submittable),
            "evidence_links": list(dict.fromkeys(str(item) for item in (evidence_links or []) if str(item))),
            "metadata": dict(metadata or {}),
        }
        key = (record.get("entity_id") or record["raw_name"], record["source"])
        existing_keys = {
            (item.get("entity_id") or item.get("raw_name"), item.get("source"))
            for item in self.open_world_candidates
        }
        if key in existing_keys:
            return
        self.open_world_candidates.append(record)

    def sources_by_name(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.items:
            payload = item.to_dict()
            key = item.entity_id or item.canonical_name
            grouped.setdefault(key, []).append(payload)
            if key != item.canonical_name:
                grouped.setdefault(item.canonical_name, []).append(payload)
        return grouped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "name_resolutions": list(self.name_resolutions),
            "unresolved_candidates": list(self.unresolved_candidates),
            "disease_categories": list(self.disease_categories),
            "open_world_candidates": list(self.open_world_candidates),
            "mechanism_hypotheses": list(self.mechanism_hypotheses),
            "clinical_patterns": list(self.clinical_patterns),
            "pattern_hypotheses": list(self.pattern_hypotheses),
            "pattern_verification_results": list(self.pattern_verification_results),
            "pattern_recall_signals": list(self.pattern_recall_signals),
            "pattern_candidate_admissions": list(self.pattern_candidate_admissions),
        }


class CandidateGenerator:
    """Union candidates from symptoms, structured evidence, LLM, RAG, and memory."""

    def __init__(self, knowledge: Any, resolver: Any):
        self.knowledge = knowledge
        self.resolver = resolver
        self.disease_retriever = DiseaseRetriever(knowledge, resolver)
        self.mechanism_reasoner = MechanismReasoner()
        ref_dir = str(getattr(knowledge, "ref_dir", "data/ref_data") or "data/ref_data")
        self.clinical_pattern_compiler = ClinicalPatternCompiler(ref_dir)

    def generate(
        self,
        evidence_graph: Optional[EvidenceGraph] = None,
        llm_result: Optional[Dict[str, Any]] = None,
        rag_chunks: Optional[Sequence[Dict[str, Any]]] = None,
        memory_hits: Optional[Sequence[Dict[str, Any]]] = None,
        evidence: Optional[EvidenceBundle] = None,
        pattern_recall_signals: Optional[Sequence[Any]] = None,
        pattern_recall_context: Optional[Dict[str, Any]] = None,
    ) -> CandidatePool:
        bundle = evidence or _bundle_from_graph(evidence_graph)
        pool = CandidatePool()
        if pattern_recall_context:
            pool.pattern_hypotheses = list(pattern_recall_context.get("pattern_hypotheses") or [])
            pool.pattern_verification_results = list(
                pattern_recall_context.get("pattern_verification_results") or []
            )
            pool.pattern_recall_signals = list(
                pattern_recall_context.get("pattern_recall_signals") or []
            )
        explicit_signals = list(pattern_recall_signals or pool.pattern_recall_signals or [])
        clinical_patterns = self.clinical_pattern_compiler.compile(bundle)
        pool.clinical_patterns = [item.to_dict() for item in clinical_patterns]
        mechanisms = self.mechanism_reasoner.evaluate(bundle)
        pool.mechanism_hypotheses = [item.to_dict() for item in mechanisms]
        self._from_clinical_patterns(pool, clinical_patterns)
        self._from_mechanisms(pool, mechanisms)
        self._from_pattern_recall_signals(pool, explicit_signals)
        self._from_llm(pool, llm_result or {})
        self._from_rag(pool, rag_chunks or [])
        self._from_memory(pool, memory_hits or [])
        self._from_disease_retriever(pool, bundle)
        self._from_evidence(pool, bundle)
        return pool

    def _from_pattern_recall_signals(
        self,
        pool: CandidatePool,
        signals: Sequence[Any],
    ) -> None:
        for raw_signal in signals or []:
            signal = _coerce_pattern_signal(raw_signal)
            if not signal:
                continue
            metadata = {
                "pattern_hypothesis_id": signal.pattern_hypothesis_id,
                "recall_mode": signal.recall_mode,
                "recall_strength": signal.recall_strength,
                "protected_pool_slot": signal.protected_pool_slot,
                "admission_level": signal.admission_level,
                "verified_specificity": signal.verified_specificity,
                "source_evidence_ids": list(signal.source_evidence_ids),
                "missing_evidence_requests": list(signal.missing_evidence_requests),
                "judge_evidence_weight": 0.0,
                "eligibility_evidence_weight": 0.0,
                "pattern_recall_only": True,
                "gap_suggestion_only": True,
                "active_gap_write_permission": "none",
            }
            admission = {
                "pattern_hypothesis_id": signal.pattern_hypothesis_id,
                "entity_id": signal.entity_id,
                "canonical_name": signal.canonical_name,
                "raw_name": signal.raw_name,
                "recall_mode": signal.recall_mode,
                "recall_strength": signal.recall_strength,
                "protected_pool_slot": signal.protected_pool_slot,
                "admission_level": signal.admission_level,
                "verified_specificity": signal.verified_specificity,
                "source_evidence_ids": list(signal.source_evidence_ids),
                "admitted_to_controlled_pool": False,
                "admitted_to_open_world": False,
                "admission_source": "",
                "admission_reason": "",
                "resolver_status": "",
            }
            if signal.recall_mode not in {RECALL_BOOST, RECALL_PROTECTED}:
                pool.add_open_world(
                    signal.raw_name or signal.canonical_name or signal.entity_id,
                    "llm_pattern_hypothesis_query",
                    prior=0.0,
                    evidence_links=signal.source_evidence_ids,
                    entity_id=signal.entity_id,
                    canonical_name=signal.canonical_name,
                    submission_name=signal.submission_name,
                    submittable=False,
                    metadata=metadata,
                )
                admission.update(
                    {
                        "admitted_to_open_world": True,
                        "admission_source": "llm_pattern_hypothesis_query",
                        "admission_reason": "query_expansion_only",
                        "resolver_status": "not_required_for_query_expansion",
                    }
                )
                pool.pattern_candidate_admissions.append(admission)
                continue
            raw_name = signal.raw_name or signal.entity_id or signal.canonical_name
            resolution = self.resolver.resolve(signal.entity_id or signal.canonical_name or raw_name)
            admission["resolver_status"] = "resolved" if resolution.canonical_name else "unresolved"
            prior = min(
                0.86,
                max(
                    0.35,
                    0.22 + 0.52 * float(signal.recall_strength or 0.0),
                ),
            )
            if signal.protected_pool_slot:
                prior = max(prior, 0.68)
            if resolution.canonical_name:
                pool.add(
                    raw_name,
                    resolution.canonical_name,
                    "llm_pattern_hypothesis",
                    prior=prior,
                    evidence_links=signal.source_evidence_ids,
                    metadata=metadata,
                    entity_id=getattr(resolution, "entity_id", "") or signal.entity_id,
                    submission_name=getattr(resolution, "submission_name", "") or resolution.canonical_name,
                    submittable=bool(getattr(resolution, "submittable", True)),
                )
                admission.update(
                    {
                        "admitted_to_controlled_pool": True,
                        "canonical_name": resolution.canonical_name,
                        "entity_id": getattr(resolution, "entity_id", "") or signal.entity_id,
                        "submission_name": getattr(resolution, "submission_name", "") or resolution.canonical_name,
                        "submittable": bool(getattr(resolution, "submittable", True)),
                        "admission_source": "llm_pattern_hypothesis",
                        "admission_reason": "verified_pattern_recall_signal",
                        "prior": prior,
                    }
                )
                pool.pattern_candidate_admissions.append(admission)
                continue
            pool.add_open_world(
                raw_name,
                "llm_pattern_hypothesis_unresolved",
                prior=0.0,
                evidence_links=signal.source_evidence_ids,
                entity_id=signal.entity_id,
                canonical_name=signal.canonical_name,
                submission_name=signal.submission_name,
                submittable=False,
                metadata=dict(metadata, submittable=False),
            )
            admission.update(
                {
                    "admitted_to_open_world": True,
                    "admission_source": "llm_pattern_hypothesis_unresolved",
                    "admission_reason": "resolver_unresolved_after_signal",
                    "submittable": False,
                    "prior": 0.0,
                }
            )
            pool.pattern_candidate_admissions.append(admission)

    def _from_clinical_patterns(
        self,
        pool: CandidatePool,
        patterns: Sequence[ClinicalPattern],
    ) -> None:
        for pattern in patterns or []:
            candidates = self.clinical_pattern_compiler.candidate_rules(pattern.pattern_id)
            if not candidates:
                continue
            evidence_links = [pattern.pattern_id] + list(pattern.supporting_findings or [])
            for item in candidates:
                raw_name = str(item.get("entity_id") or item.get("name") or "").strip()
                if not raw_name:
                    continue
                resolution = self.resolver.resolve(raw_name)
                try:
                    priority = float(item.get("priority", 0.8) or 0.8)
                except (TypeError, ValueError):
                    priority = 0.8
                prior = min(
                    0.92,
                    max(0.35, 0.28 + 0.48 * pattern.confidence + 0.18 * priority),
                )
                metadata = {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "pattern_version": pattern.pattern_version,
                    "body_system": pattern.body_system,
                    "mechanism_ids": list(pattern.mechanism_ids),
                    "family_ids": list(pattern.family_ids),
                    "candidate_role": str(item.get("role") or "pattern_candidate"),
                    "pattern_confidence": pattern.confidence,
                    "pattern_information_value": pattern.information_value,
                    "source_level": pattern.source_level,
                    "verified_pattern": bool(pattern.verified),
                    "supporting_findings": list(pattern.supporting_findings),
                    "supporting_observation_ids": list(pattern.supporting_observation_ids),
                    "matched_domains": list(pattern.matched_domains),
                    "temporal_consistency": pattern.temporal_consistency,
                    "polarity_consistency": pattern.polarity_consistency,
                    "clinical_pattern": pattern.to_dict(),
                }
                if resolution.canonical_name:
                    pool.add(
                        item.get("name") or resolution.canonical_name,
                        resolution.canonical_name,
                        "clinical_pattern",
                        prior=prior,
                        evidence_links=evidence_links,
                        metadata=metadata,
                        entity_id=getattr(resolution, "entity_id", ""),
                        submission_name=getattr(resolution, "submission_name", "") or resolution.canonical_name,
                        submittable=bool(getattr(resolution, "submittable", True)),
                    )
                    continue
                pool.add_open_world(
                    item.get("name") or raw_name,
                    "clinical_pattern",
                    prior=prior,
                    evidence_links=evidence_links,
                    metadata=dict(metadata, submittable=False),
                )

    def _from_llm(self, pool: CandidatePool, llm_result: Dict[str, Any]) -> None:
        resolutions = self.resolver.resolve_result(llm_result or {})
        pool.name_resolutions = [item.to_dict() for item in resolutions]
        pool.unresolved_candidates = [item.raw_name for item in resolutions if not item.resolved]
        for index, item in enumerate(resolutions):
            rank_prior = max(0.65, 1.0 - index * 0.10)
            prior = rank_prior * item.confidence * item.model_confidence
            if not item.canonical_name:
                pool.add_open_world(
                    item.raw_name,
                    "llm_unresolved",
                    prior=prior,
                    metadata={
                        "method": item.method,
                        "model_confidence": item.model_confidence,
                        "submittable": False,
                    },
                )
                continue
            pool.add(
                item.raw_name,
                item.canonical_name,
                "llm",
                prior=prior,
                metadata={"method": item.method, "parent_name": item.parent_name},
                entity_id=getattr(item, "entity_id", ""),
                submission_name=getattr(item, "submission_name", "") or item.canonical_name,
                submittable=bool(getattr(item, "submittable", True)),
            )

    def _from_rag(self, pool: CandidatePool, rag_chunks: Sequence[Dict[str, Any]]) -> None:
        for chunk in rag_chunks:
            chunk_type = chunk.get("type")
            if chunk_type not in {"disease_profile", "external_medical_knowledge"}:
                continue
            raw_names = [chunk.get("title")]
            metadata = dict(chunk.get("metadata") or {})
            raw_names.extend(metadata.get("candidate_diseases") or [])
            for raw_name in raw_names:
                if not raw_name:
                    continue
                resolution = self.resolver.resolve(raw_name)
                try:
                    prior = float(chunk.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    prior = 0.0
                if not resolution.canonical_name:
                    pool.add_open_world(
                        raw_name,
                        "external_retrieval" if chunk_type == "external_medical_knowledge" else "rag_unresolved",
                        prior=prior,
                        metadata={
                            "chunk_id": chunk.get("id"),
                            "chunk_type": chunk_type,
                            "unreviewed_external": bool(metadata.get("unreviewed_external")),
                            "submittable": False,
                        },
                    )
                    continue
                if chunk_type == "external_medical_knowledge":
                    if getattr(resolution, "submittable", False):
                        pool.add(
                            raw_name,
                            resolution.canonical_name,
                            "external_retrieval",
                            prior=prior,
                            metadata={
                                "chunk_id": chunk.get("id"),
                                "chunk_type": chunk_type,
                                "unreviewed_external": bool(metadata.get("unreviewed_external")),
                                "submittable": True,
                            },
                            entity_id=getattr(resolution, "entity_id", ""),
                            submission_name=getattr(resolution, "submission_name", "") or resolution.canonical_name,
                            submittable=True,
                        )
                        pool.add_open_world(
                            raw_name,
                            "external_retrieval_audit",
                            prior=prior,
                            entity_id=getattr(resolution, "entity_id", ""),
                            canonical_name=resolution.canonical_name,
                            submission_name=getattr(resolution, "submission_name", "") or resolution.canonical_name,
                            submittable=True,
                            metadata={
                                "chunk_id": chunk.get("id"),
                                "chunk_type": chunk_type,
                                "controlled_entity": True,
                                "unreviewed_external": bool(metadata.get("unreviewed_external")),
                            },
                        )
                        continue
                    pool.add_open_world(
                        raw_name,
                        "external_retrieval",
                        prior=prior,
                        entity_id=getattr(resolution, "entity_id", ""),
                        canonical_name=resolution.canonical_name,
                        submission_name=getattr(resolution, "submission_name", "") or resolution.canonical_name,
                        submittable=False,
                        metadata={
                            "chunk_id": chunk.get("id"),
                            "chunk_type": chunk_type,
                            "canonical_name": resolution.canonical_name,
                            "unreviewed_external": True,
                            "submittable": False,
                        },
                    )
                    continue
                pool.add(
                    raw_name,
                    resolution.canonical_name,
                    "rag",
                    prior=prior,
                    metadata={"chunk_id": chunk.get("id"), "chunk_type": chunk.get("type")},
                    entity_id=getattr(resolution, "entity_id", ""),
                    submission_name=getattr(resolution, "submission_name", "") or resolution.canonical_name,
                    submittable=bool(getattr(resolution, "submittable", True)),
                )

    def _from_mechanisms(
        self,
        pool: CandidatePool,
        mechanisms: Sequence[MechanismHypothesis],
    ) -> None:
        for hypothesis in mechanisms:
            prior = min(0.86, 0.35 + 0.45 * float(hypothesis.confidence or 0.0))
            evidence_links = list(hypothesis.supporting_findings or [])
            for raw_name in list(hypothesis.candidate_diseases or []):
                resolution = self.resolver.resolve(raw_name)
                if resolution.canonical_name:
                    pool.add(
                        raw_name,
                        resolution.canonical_name,
                        "mechanism_reasoner",
                        prior=prior,
                        evidence_links=evidence_links,
                        metadata={
                            "mechanism_id": hypothesis.mechanism_id,
                            "family_id": hypothesis.family_id,
                            "body_system": hypothesis.body_system,
                        },
                        entity_id=getattr(resolution, "entity_id", ""),
                        submission_name=getattr(resolution, "submission_name", "") or resolution.canonical_name,
                        submittable=bool(getattr(resolution, "submittable", True)),
                    )
                    continue
                pool.add_open_world(
                    raw_name,
                    "mechanism_reasoner",
                    prior=prior,
                    evidence_links=evidence_links,
                    metadata={
                        "mechanism_id": hypothesis.mechanism_id,
                        "family_id": hypothesis.family_id,
                        "body_system": hypothesis.body_system,
                        "submittable": False,
                    },
                )
            for raw_name in list(hypothesis.open_world_candidates or []):
                resolution = self.resolver.resolve(raw_name)
                if resolution.canonical_name and getattr(resolution, "submittable", False):
                    pool.add(
                        raw_name,
                        resolution.canonical_name,
                        "mechanism_reasoner",
                        prior=max(0.20, prior - 0.08),
                        evidence_links=evidence_links,
                        metadata={
                            "mechanism_id": hypothesis.mechanism_id,
                            "family_id": hypothesis.family_id,
                            "body_system": hypothesis.body_system,
                            "promoted_open_world_entity": True,
                        },
                        entity_id=getattr(resolution, "entity_id", ""),
                        submission_name=getattr(resolution, "submission_name", "") or resolution.canonical_name,
                        submittable=True,
                    )
                    continue
                pool.add_open_world(
                    raw_name,
                    "mechanism_reasoner",
                    prior=max(0.20, prior - 0.08),
                    evidence_links=evidence_links,
                    metadata={
                        "mechanism_id": hypothesis.mechanism_id,
                        "family_id": hypothesis.family_id,
                        "body_system": hypothesis.body_system,
                        "submittable": False,
                    },
                )

    def _from_memory(self, pool: CandidatePool, memory_hits: Sequence[Dict[str, Any]]) -> None:
        for hit in memory_hits:
            names = (
                hit.get("expected_diagnoses")
                or hit.get("diagnosis")
                or hit.get("diagnoses")
                or hit.get("submitted_diagnoses")
                or []
            )
            if isinstance(names, str):
                names = [names]
            try:
                prior = float(hit.get("score", hit.get("similarity", 0.45)) or 0.45)
            except (TypeError, ValueError):
                prior = 0.45
            for raw in names:
                resolution = self.resolver.resolve(raw)
                if resolution.canonical_name:
                    pool.add(
                        raw,
                        resolution.canonical_name,
                        "memory",
                        prior=min(0.75, prior),
                        entity_id=getattr(resolution, "entity_id", ""),
                        submission_name=getattr(resolution, "submission_name", "") or resolution.canonical_name,
                        submittable=bool(getattr(resolution, "submittable", True)),
                    )

    def _from_disease_retriever(self, pool: CandidatePool, evidence: EvidenceBundle) -> None:
        hits, categories = self.disease_retriever.retrieve(evidence, top_k=20)
        pool.disease_categories = [item.to_dict() for item in categories]
        for hit in hits:
            pool.add(
                hit.diagnosis,
                hit.diagnosis,
                "disease_retriever",
                prior=hit.score,
                evidence_links=hit.evidence_links,
                metadata={
                    "category": hit.category,
                    **dict(hit.metadata or {}),
                },
                entity_id=self.knowledge.entity_id_for(hit.diagnosis) if hasattr(self.knowledge, "entity_id_for") else "",
                submission_name=self.knowledge.submission_name_for(hit.diagnosis) if hasattr(self.knowledge, "submission_name_for") else hit.diagnosis,
                submittable=self.knowledge.is_submittable_entity(hit.diagnosis) if hasattr(self.knowledge, "is_submittable_entity") else True,
            )

    def _from_evidence(self, pool: CandidatePool, evidence: EvidenceBundle) -> None:
        observations = evidence.observations if evidence else []
        for name, entry in self.knowledge.entries.items():
            matched = []
            weight = 0.0
            for spec in entry.get("supporting_evidence", []) or []:
                hits = _matching_observations(spec, observations)
                if not hits:
                    continue
                spec_weight = float(spec.get("weight", 0.2) or 0.2)
                confidence = max(
                    item.confidence * _information_multiplier(item)
                    for item in hits
                )
                weight += spec_weight * confidence
                matched.extend(item.finding for item in hits)
            if matched:
                pool.add(
                    name,
                    name,
                    "evidence",
                    prior=min(0.85, max(0.20, weight / 1.5)),
                    evidence_links=list(dict.fromkeys(matched)),
                    entity_id=str(entry.get("entity_id") or ""),
                    submission_name=str(entry.get("submission_name") or name),
                    submittable=bool(entry.get("submittable", True)),
                )


def _bundle_from_graph(graph: Optional[EvidenceGraph]) -> EvidenceBundle:
    if graph is None:
        return EvidenceBundle()
    if getattr(graph, "bundle", None):
        return graph.bundle
    observations: List[Observation] = []
    for item in getattr(graph, "observations", []) or []:
        if isinstance(item, Observation):
            observations.append(item)
        elif isinstance(item, dict):
            try:
                observations.append(Observation(**{k: v for k, v in item.items() if k in Observation.__dataclass_fields__}))
            except (TypeError, ValueError):
                continue
    return EvidenceBundle(observations)


def _coerce_pattern_signal(value: Any) -> Optional[PatternRecallSignal]:
    if isinstance(value, PatternRecallSignal):
        return value
    if not isinstance(value, dict):
        return None
    try:
        return PatternRecallSignal(
            pattern_hypothesis_id=str(value.get("pattern_hypothesis_id") or ""),
            entity_id=str(value.get("entity_id") or ""),
            entity_link_confidence=float(value.get("entity_link_confidence") or 0.0),
            recall_mode=str(value.get("recall_mode") or ""),
            recall_strength=float(value.get("recall_strength") or 0.0),
            protected_pool_slot=bool(value.get("protected_pool_slot", False)),
            source_evidence_ids=[
                str(item)
                for item in value.get("source_evidence_ids") or []
                if str(item)
            ],
            missing_evidence_requests=[
                dict(item) if isinstance(item, dict) else {"target_evidence": str(item)}
                for item in value.get("missing_evidence_requests") or []
                if item
            ],
            canonical_name=str(value.get("canonical_name") or ""),
            submission_name=str(value.get("submission_name") or ""),
            raw_name=str(value.get("raw_name") or ""),
            judge_evidence_weight=0.0,
            eligibility_evidence_weight=0.0,
            gap_suggestion_only=True,
            active_gap_write_permission="none",
            admission_level=str(value.get("admission_level") or "family_expansion"),
            verified_specificity=str(value.get("verified_specificity") or "family"),
        )
    except (TypeError, ValueError):
        return None


def _matching_observations(spec: Dict[str, Any], observations: Sequence[Observation]) -> List[Observation]:
    return [
        item for item in observations
        if item.polarity == "positive"
        and not getattr(item, "shadowed_by", "")
        and _observation_matches(spec, item)
    ]


def _observation_matches(spec: Dict[str, Any], item: Observation) -> bool:
    finding = str(spec.get("finding") or "")
    if finding and finding != item.finding:
        return False
    direction = str(spec.get("direction") or "")
    if direction and direction != item.direction:
        return False
    source_contains = str(spec.get("source_contains") or "")
    if source_contains and source_contains.lower() not in item.source.lower():
        return False
    terms = spec.get("terms") or []
    if isinstance(terms, str):
        terms = [terms]
    if terms and not any(str(term).lower() in item.raw_text.lower() for term in terms):
        return False
    if spec.get("min_value") is not None:
        if item.value is None or item.value < float(spec["min_value"]):
            return False
    if spec.get("max_value") is not None:
        if item.value is None or item.value > float(spec["max_value"]):
            return False
    return bool(finding or direction or source_contains or terms or spec.get("min_value") is not None or spec.get("max_value") is not None)


def _information_multiplier(item: Observation) -> float:
    try:
        value = float(getattr(item, "information_value", 0.0) or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0.0:
        return 1.0
    return max(0.35, min(1.45, 0.65 + value))
