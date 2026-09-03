from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .clinical_evidence import EvidenceBundle, Observation


@dataclass
class MechanismHypothesis:
    mechanism_id: str
    family_id: str
    body_system: str
    confidence: float
    diagnostic_value: float
    supporting_findings: List[str] = field(default_factory=list)
    contradicting_findings: List[str] = field(default_factory=list)
    missing_key_evidence: List[str] = field(default_factory=list)
    query_terms: List[str] = field(default_factory=list)
    candidate_diseases: List[str] = field(default_factory=list)
    open_world_candidates: List[str] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalView:
    view_type: str
    query: str
    terms: List[str] = field(default_factory=list)
    mechanism_id: str = ""
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_MECHANISM_RULES: Tuple[Dict[str, Any], ...] = (
    {
        "mechanism_id": "urachal_remnant_anomaly",
        "family_id": "urachal_remnant",
        "body_system": "genitourinary_midline_embryologic",
        "candidate_diseases": ["\u8110\u5c3f\u7ba1\u56ca\u80bf"],
        "open_world_candidates": ["\u8110\u5c3f\u7ba1\u7a66", "\u8110\u5c3f\u7ba1\u7aa6", "\u8110\u5c3f\u7ba1\u5f02\u5e38"],
        "finding_weights": {
            "umbilical_discharge": 0.52,
            "umbilical_mass": 0.34,
            "midline_suprapubic_pain": 0.28,
            "midline_suprapubic_cyst": 0.36,
            "urachal_cyst_imaging": 0.54,
            "urachal_remnant_pattern": 0.62,
        },
        "term_weights": {
            "\u8110\u90e8\u53cd\u590d\u6e17\u6db2": 0.45,
            "\u8110\u90e8\u6d41\u6db2": 0.38,
            "\u8110\u5b54\u6d41\u6db2": 0.38,
            "\u4e2d\u7ebf\u5f02\u5e38\u5f00\u53e3": 0.50,
            "\u80da\u80ce\u6b8b\u4f59": 0.46,
            "\u8110\u5c3f\u7ba1": 0.58,
        },
        "required_any": ["umbilical_discharge", "urachal_remnant_pattern", "urachal_cyst_imaging"],
        "missing_key_evidence": ["midline umbilicus-to-bladder tract imaging", "urachal remnant ultrasound or CT"],
        "query_terms": ["umbilical discharge", "midline opening", "embryologic remnant", "urachal remnant"],
        "threshold": 0.38,
    },
    {
        "mechanism_id": "pulmonary_vascular_shunt",
        "family_id": "pulmonary_arteriovenous_malformation_family",
        "body_system": "pulmonary_vascular",
        "candidate_diseases": ["\u80ba\u52a8\u9759\u8109\u7618"],
        "open_world_candidates": ["\u80ba\u52a8\u9759\u8109\u7578\u5f62", "\u80ba\u5185\u53f3\u5411\u5de6\u5206\u6d41"],
        "finding_weights": {
            "hypoxemia": 0.26,
            "cyanosis": 0.24,
            "hemoptysis": 0.22,
            "dyspnea_on_exertion": 0.18,
            "exercise_intolerance": 0.14,
            "right_to_left_shunt": 0.46,
            "pulmonary_hypertension": 0.12,
            "cardiopulmonary_exertional_pattern": 0.12,
        },
        "term_weights": {
            "\u80ba\u52a8\u9759\u8109\u7618": 0.72,
            "\u80ba\u52a8\u9759\u8109\u7578\u5f62": 0.60,
            "\u52a8\u9759\u8109\u7578\u5f62": 0.46,
            "\u53f3\u5411\u5de6\u5206\u6d41": 0.44,
            "\u53d1\u7ec0": 0.22,
            "\u4f4e\u6c27": 0.22,
            "\u54af\u8840": 0.20,
            "\u80ba\u7ed3\u8282": 0.14,
            "\u5b64\u7acb\u6027\u80ba": 0.14,
        },
        "required_any": ["right_to_left_shunt", "hypoxemia", "cyanosis"],
        "missing_key_evidence": ["contrast echocardiography for shunt", "pulmonary CTA or angiography"],
        "query_terms": ["pulmonary vascular shunt", "PAVM", "hemoptysis cyanosis hypoxemia", "right-to-left shunt"],
        "threshold": 0.36,
    },
    {
        "mechanism_id": "chest_wall_structural_deformity",
        "family_id": "pectus_deformity",
        "body_system": "musculoskeletal_thoracic",
        "candidate_diseases": ["\u6f0f\u6597\u80f8"],
        "open_world_candidates": ["\u80f8\u5eca\u7578\u5f62", "\u80f8\u9aa8\u51f9\u9677"],
        "finding_weights": {
            "dyspnea_on_exertion": 0.12,
            "exercise_intolerance": 0.12,
            "cardiopulmonary_exertional_pattern": 0.08,
        },
        "term_weights": {
            "\u6f0f\u6597\u80f8": 0.74,
            "\u80f8\u9aa8\u51f9\u9677": 0.62,
            "\u524d\u80f8\u51f9\u9677": 0.58,
            "\u80f8\u5eca\u7578\u5f62": 0.54,
            "Haller": 0.50,
            "haller": 0.50,
        },
        "required_any": [],
        "missing_key_evidence": ["chest CT Haller index", "focused chest wall physical examination"],
        "query_terms": ["pectus excavatum", "sternal depression", "chest wall deformity", "Haller index"],
        "threshold": 0.34,
    },
    {
        "mechanism_id": "mucocutaneous_ocular_vasculitis",
        "family_id": "behcet_spectrum",
        "body_system": "systemic_vasculitis",
        "candidate_diseases": ["\u767d\u585e\u7efc\u5408\u5f81"],
        "open_world_candidates": ["Behcet disease", "\u53e3-\u773c-\u751f\u6b96\u5668\u6e83\u75a1\u7efc\u5408\u5f81"],
        "finding_weights": {
            "arthralgia": 0.10,
            "conjunctivitis": 0.18,
            "oral_ulcer": 0.34,
            "genital_ulcer": 0.38,
            "ocular_inflammation": 0.30,
        },
        "term_weights": {
            "\u767d\u585e": 0.76,
            "\u53e3\u8154\u6e83\u75a1": 0.34,
            "\u590d\u53d1\u53e3\u8154\u6e83\u75a1": 0.42,
            "\u5916\u9634\u6e83\u75a1": 0.42,
            "\u751f\u6b96\u5668\u6e83\u75a1": 0.42,
            "\u7ed3\u819c\u708e": 0.22,
            "\u8461\u8404\u819c\u708e": 0.34,
            "\u9488\u523a\u53cd\u5e94": 0.36,
        },
        "required_any": ["genital_ulcer", "oral_ulcer"],
        "missing_key_evidence": ["recurrent oral ulcers", "genital ulcers", "ocular inflammation exam", "pathergy test"],
        "query_terms": ["Behcet syndrome", "recurrent oral genital ulcers", "ocular inflammation vasculitis"],
        "threshold": 0.38,
    },
    {
        "mechanism_id": "accommodation_failure",
        "family_id": "age_related_refractive_accommodation",
        "body_system": "ophthalmology_refraction",
        "candidate_diseases": ["\u8001\u89c6"],
        "open_world_candidates": ["presbyopia", "accommodation insufficiency"],
        "finding_weights": {
            "near_vision_difficulty": 0.42,
            "distance_vision_relatively_preserved": 0.24,
            "worse_in_dim_light": 0.18,
            "refractive_correction_improves_near_vision": 0.44,
            "presbyopia_pattern": 0.50,
            "accommodation_failure_pattern": 0.42,
        },
        "term_weights": {
            "\u624b\u673a\u8981\u62ff\u8fdc": 0.34,
            "\u770b\u8fdc\u8fd8\u53ef\u4ee5": 0.22,
            "\u5149\u7ebf\u6697": 0.16,
            "\u9605\u8bfb\u955c": 0.36,
            "\u8001\u89c6": 0.52,
            "\u8001\u82b1": 0.52,
        },
        "required_any": ["near_vision_difficulty", "refractive_correction_improves_near_vision"],
        "missing_key_evidence": ["near visual acuity", "refraction or reading lens response"],
        "query_terms": ["near vision difficulty", "distance vision preserved", "dim light worse", "accommodation failure"],
        "threshold": 0.36,
    },
)


class MechanismReasoner:
    """Form disease-family and mechanism hypotheses before closed-world submission."""

    def evaluate(self, evidence: Optional[EvidenceBundle]) -> List[MechanismHypothesis]:
        bundle = evidence or EvidenceBundle()
        observations = list(bundle.observations or [])
        positive = {
            item.finding
            for item in observations
            if item.polarity == "positive" and not getattr(item, "shadowed_by", "")
        }
        negative = {
            item.finding
            for item in observations
            if item.polarity == "negative" and not getattr(item, "shadowed_by", "")
        }
        text = _combined_text(observations)
        hypotheses: List[MechanismHypothesis] = []
        for rule in _MECHANISM_RULES:
            score = 0.0
            support: List[str] = []
            for finding, weight in (rule.get("finding_weights") or {}).items():
                if finding in positive:
                    score += float(weight)
                    support.append(finding)
            for term, weight in (rule.get("term_weights") or {}).items():
                if str(term) and str(term).lower() in text.lower():
                    score += float(weight)
                    support.append(str(term))
            contradictions = [
                finding
                for finding in (rule.get("negative_findings") or [])
                if finding in negative
            ]
            score -= 0.18 * len(contradictions)
            threshold = float(rule.get("threshold", 0.35) or 0.35)
            if score < threshold:
                continue
            required_any = [str(item) for item in rule.get("required_any") or [] if str(item)]
            missing = list(rule.get("missing_key_evidence") or [])
            if required_any and not any(item in positive for item in required_any):
                missing = list(dict.fromkeys(required_any + missing))
            confidence = max(0.0, min(0.98, score))
            query_terms = list(
                dict.fromkeys(
                    list(rule.get("query_terms") or [])
                    + support
                    + [rule.get("family_id"), rule.get("body_system")]
                )
            )
            hypotheses.append(
                MechanismHypothesis(
                    mechanism_id=str(rule.get("mechanism_id") or ""),
                    family_id=str(rule.get("family_id") or ""),
                    body_system=str(rule.get("body_system") or ""),
                    confidence=round(confidence, 4),
                    diagnostic_value=round(min(0.98, confidence + 0.08), 4),
                    supporting_findings=list(dict.fromkeys(support)),
                    contradicting_findings=contradictions,
                    missing_key_evidence=missing,
                    query_terms=[str(item) for item in query_terms if str(item)],
                    candidate_diseases=list(rule.get("candidate_diseases") or []),
                    open_world_candidates=list(rule.get("open_world_candidates") or []),
                    audit={"matched_score": round(score, 4), "threshold": threshold},
                )
            )
        hypotheses.sort(key=lambda item: (item.confidence, item.diagnostic_value), reverse=True)
        return hypotheses

    def retrieval_views(
        self,
        evidence: Optional[EvidenceBundle],
        mechanisms: Optional[Sequence[MechanismHypothesis]] = None,
    ) -> List[RetrievalView]:
        bundle = evidence or EvidenceBundle()
        hypotheses = list(mechanisms if mechanisms is not None else self.evaluate(bundle))
        views: List[RetrievalView] = []
        positive = [
            item for item in bundle.observations
            if item.polarity == "positive" and not getattr(item, "shadowed_by", "")
        ]
        ranked_positive = sorted(
            positive,
            key=lambda item: (item.information_value, item.confidence),
            reverse=True,
        )
        evidence_terms = _dedupe(
            [item.finding for item in ranked_positive[:12]]
            + [item.source_text or item.raw_text for item in ranked_positive[:8]]
        )
        if evidence_terms:
            views.append(
                RetrievalView(
                    view_type="standardized_evidence",
                    query=" ".join(evidence_terms),
                    terms=evidence_terms,
                    weight=1.0,
                )
            )
        patterns = _dedupe(
            [item.clinical_pattern for item in positive if getattr(item, "clinical_pattern", "")]
            + [item.finding for item in positive if str(item.evidence_level) == "diagnostic_pattern"]
        )
        if patterns:
            views.append(
                RetrievalView(
                    view_type="clinical_pattern",
                    query=" ".join(patterns),
                    terms=patterns,
                    weight=1.08,
                )
            )
        for hypothesis in hypotheses[:6]:
            if hypothesis.query_terms:
                views.append(
                    RetrievalView(
                        view_type="mechanism",
                        query=" ".join(hypothesis.query_terms),
                        terms=list(hypothesis.query_terms),
                        mechanism_id=hypothesis.mechanism_id,
                        weight=1.15,
                        metadata={
                            "family_id": hypothesis.family_id,
                            "body_system": hypothesis.body_system,
                            "confidence": hypothesis.confidence,
                        },
                    )
                )
            views.append(
                RetrievalView(
                    view_type="disease_family",
                    query=" ".join(
                        _dedupe(
                            [
                                hypothesis.family_id,
                                hypothesis.body_system,
                            ]
                            + hypothesis.candidate_diseases
                            + hypothesis.open_world_candidates
                        )
                    ),
                    terms=_dedupe(
                        [
                            hypothesis.family_id,
                            hypothesis.body_system,
                        ]
                        + hypothesis.candidate_diseases
                        + hypothesis.open_world_candidates
                    ),
                    mechanism_id=hypothesis.mechanism_id,
                    weight=1.05,
                    metadata={"unreviewed_candidates": list(hypothesis.open_world_candidates)},
                )
            )
        negatives = _dedupe(
            [
                item.finding
                for item in bundle.observations
                if item.polarity == "negative" and not getattr(item, "shadowed_by", "")
            ][:12]
        )
        if negatives:
            views.append(
                RetrievalView(
                    view_type="key_negative",
                    query=" ".join(negatives),
                    terms=negatives,
                    weight=0.75,
                )
            )
        return [view for view in views if view.query or view.terms]


def _combined_text(observations: Sequence[Observation]) -> str:
    parts: List[str] = []
    for item in observations:
        parts.extend(
            [
                item.finding,
                item.raw_text,
                item.source_text,
                item.field_path,
                item.clinical_pattern,
                " ".join(getattr(item, "mechanism_ids", []) or []),
            ]
        )
    return " ".join(str(item) for item in parts if str(item))


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
