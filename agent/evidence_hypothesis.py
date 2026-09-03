"""LLM-guided but verifier-owned evidence hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .evidence_registry import EvidenceDefinitionRegistry


OBSERVED_FINDING = "observed_finding"
DERIVED_PATTERN = "derived_pattern"
COUNTEREVIDENCE = "counterevidence"
MISSING_CONFIRMATION = "missing_confirmation"


@dataclass
class EvidenceHypothesis:
    hypothesis_id: str
    candidate: str
    target_evidence_id: str
    claim_type: str = OBSERVED_FINDING
    importance: str = "medium"
    expected_effect: str = "ranking"
    reason: str = ""
    status: str = "pending_verification"
    required_inputs: List[str] = field(default_factory=list)
    recommended_exam: str = ""
    producer: str = "reasoner"
    case_version: int = 0
    confidence: float = 0.0
    entity_id: str = ""

    @property
    def claim_id(self) -> str:
        return self.hypothesis_id

    @property
    def target_evidence(self) -> str:
        return self.target_evidence_id

    @property
    def diagnosis_hypothesis(self) -> str:
        return self.candidate

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Compatibility keys used by the existing Judge audit path.
        payload["claim_id"] = self.hypothesis_id
        payload["target_evidence"] = self.target_evidence_id
        payload["diagnosis_hypothesis"] = self.candidate
        # Search terms intentionally stay outside LLM output. The Query Planner
        # expands these from EvidenceDefinitionRegistry.
        payload["search_terms"] = []
        return payload

    @classmethod
    def from_any(cls, value: Any) -> Optional["EvidenceHypothesis"]:
        if isinstance(value, EvidenceHypothesis):
            return value
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if not isinstance(value, dict):
            return None
        target = str(
            value.get("target_evidence_id")
            or value.get("target_evidence")
            or value.get("evidence_id")
            or ""
        ).strip()
        hypothesis_id = str(
            value.get("hypothesis_id")
            or value.get("claim_id")
            or value.get("id")
            or ""
        ).strip()
        if not target or not hypothesis_id:
            return None
        return cls(
            hypothesis_id=hypothesis_id,
            candidate=str(
                value.get("candidate")
                or value.get("diagnosis_hypothesis")
                or value.get("diagnosis")
                or ""
            ),
            target_evidence_id=target,
            claim_type=str(value.get("claim_type") or OBSERVED_FINDING),
            importance=str(value.get("importance") or "medium"),
            expected_effect=str(value.get("expected_effect") or "ranking"),
            reason=str(value.get("reason") or ""),
            status=str(value.get("status") or "pending_verification"),
            required_inputs=_text_list(value.get("required_inputs") or []),
            recommended_exam=str(value.get("recommended_exam") or ""),
            producer=str(value.get("producer") or "reasoner"),
            case_version=_int(value.get("case_version"), 0),
            confidence=_float(value.get("confidence"), 0.0),
            entity_id=str(value.get("entity_id") or ""),
        )


class EvidenceHypothesisGenerator:
    """Convert LLM/candidate hints into canonical evidence targets.

    This class does not generate aliases or thresholds. It only says which
    canonical evidence ids should be checked.
    """

    def __init__(self, registry: Optional[EvidenceDefinitionRegistry] = None):
        self.registry = registry or EvidenceDefinitionRegistry()

    def generate(self, llm_result: Any, candidate_pool: Any = None) -> List[EvidenceHypothesis]:
        candidates = self._candidate_records(llm_result, candidate_pool)
        result: List[EvidenceHypothesis] = []
        for record in candidates:
            name = record["name"]
            entity_id = record.get("entity_id", "")
            text = f"{name} {entity_id}".lower()
            if self._is_leukemia(text):
                result.extend(self._leukemia(name, entity_id))
            if self._is_pavm(text):
                result.extend(self._pavm(name, entity_id))
            if self._is_prostatitis(text):
                result.extend(self._prostatitis(name, entity_id))
        return _dedupe_hypotheses(result)

    def _hypothesis(
        self,
        *,
        hypothesis_id: str,
        candidate: str,
        target: str,
        claim_type: str,
        importance: str,
        expected_effect: str,
        reason: str,
        entity_id: str = "",
        required_inputs: Optional[Sequence[str]] = None,
        confidence: float = 0.8,
    ) -> EvidenceHypothesis:
        return EvidenceHypothesis(
            hypothesis_id=hypothesis_id,
            candidate=candidate,
            target_evidence_id=target,
            claim_type=claim_type,
            importance=importance,
            expected_effect=expected_effect,
            reason=reason,
            required_inputs=list(required_inputs or []),
            recommended_exam=self.registry.followup_exam_for(target),
            confidence=confidence,
            entity_id=entity_id,
        )

    def _leukemia(self, candidate: str, entity_id: str) -> List[EvidenceHypothesis]:
        return [
            self._hypothesis(
                hypothesis_id="H-LEUKEMIA-blast_present",
                candidate=candidate,
                entity_id=entity_id,
                target="blast_present",
                claim_type=OBSERVED_FINDING,
                importance="critical",
                expected_effect="eligibility",
                reason="Blast evidence can close the objective anchor for acute leukemia.",
                confidence=0.88,
            ),
            self._hypothesis(
                hypothesis_id="H-LEUKEMIA-multilineage_cytopenia",
                candidate=candidate,
                entity_id=entity_id,
                target="multilineage_cytopenia",
                claim_type=DERIVED_PATTERN,
                importance="critical",
                expected_effect="eligibility",
                reason="Multiple cytopenias are a high-value hematologic pattern.",
                required_inputs=[
                    "hemoglobin_low",
                    "platelet_low",
                    "white_blood_cell_abnormal",
                ],
                confidence=0.82,
            ),
            self._hypothesis(
                hypothesis_id="H-LEUKEMIA-acute_pattern",
                candidate=candidate,
                entity_id=entity_id,
                target="acute_leukemia_pattern",
                claim_type=DERIVED_PATTERN,
                importance="critical",
                expected_effect="eligibility",
                reason="Blast evidence plus cytopenia pattern can establish a leukemia anchor.",
                required_inputs=["blast_present", "multilineage_cytopenia"],
                confidence=0.9,
            ),
        ]

    def _pavm(self, candidate: str, entity_id: str) -> List[EvidenceHypothesis]:
        return [
            self._hypothesis(
                hypothesis_id="H-PAVM-right_to_left_shunt",
                candidate=candidate,
                entity_id=entity_id,
                target="right_to_left_shunt",
                claim_type=OBSERVED_FINDING,
                importance="critical",
                expected_effect="eligibility",
                reason="Right-to-left shunt is a central mechanism claim for pulmonary AV fistula.",
                confidence=0.86,
            ),
            self._hypothesis(
                hypothesis_id="H-PAVM-mechanism",
                candidate=candidate,
                entity_id=entity_id,
                target="pulmonary_avm_mechanism",
                claim_type=OBSERVED_FINDING,
                importance="critical",
                expected_effect="eligibility",
                reason="A pulmonary vascular malformation mechanism should be verified from source text or imaging.",
                confidence=0.84,
            ),
            self._hypothesis(
                hypothesis_id="H-PAVM-confirmation",
                candidate=candidate,
                entity_id=entity_id,
                target="pulmonary_cta_positive",
                claim_type=MISSING_CONFIRMATION,
                importance="critical",
                expected_effect="eligibility",
                reason="CTA or equivalent vascular imaging is needed before final confirmation.",
                confidence=0.88,
            ),
        ]

    def _prostatitis(self, candidate: str, entity_id: str) -> List[EvidenceHypothesis]:
        return [
            self._hypothesis(
                hypothesis_id="H-PROSTATITIS-localization",
                candidate=candidate,
                entity_id=entity_id,
                target="prostate_tenderness",
                claim_type=OBSERVED_FINDING,
                importance="critical",
                expected_effect="eligibility",
                reason="Pyuria alone is nonspecific; prostatitis needs localization evidence.",
                confidence=0.78,
            ),
            self._hypothesis(
                hypothesis_id="H-PROSTATITIS-negative_urine_pattern",
                candidate=candidate,
                entity_id=entity_id,
                target="bacterial_prostatitis_negative_pattern",
                claim_type=COUNTEREVIDENCE,
                importance="critical",
                expected_effect="eligibility_blocker",
                reason="Culture and urinalysis negatives can downgrade bacterial prostatitis.",
                required_inputs=[
                    "urine_culture_no_growth",
                    "nitrite_negative",
                    "leukocyte_esterase_negative",
                    "urine_wbc_normal",
                ],
                confidence=0.86,
            ),
        ]

    def _candidate_records(self, llm_result: Any, candidate_pool: Any) -> List[Dict[str, str]]:
        records: List[Dict[str, str]] = []

        def add(name: Any, entity_id: Any = "") -> None:
            text = str(name or "").strip()
            if not text:
                return
            payload = {"name": text, "entity_id": str(entity_id or "").strip()}
            if payload not in records:
                records.append(payload)

        if isinstance(llm_result, dict):
            for key in ("diagnosis", "primary_diagnosis"):
                add(llm_result.get(key))
            for key in ("diagnosis_candidates", "differential_diagnoses", "candidate_diagnoses"):
                for item in llm_result.get(key) or []:
                    if isinstance(item, dict):
                        add(item.get("name") or item.get("diagnosis"), item.get("entity_id"))
                    else:
                        add(item)
            reasoning = str(llm_result.get("reasoning") or "")
            if reasoning:
                for marker, name in (
                    ("白血病", "白血病"),
                    ("leukemia", "leukemia"),
                    ("AML", "AML"),
                    ("ALL", "ALL"),
                    ("肺动静脉瘘", "肺动静脉瘘"),
                    ("肺动静脉畸形", "肺动静脉瘘"),
                    ("PAVM", "PAVM"),
                    ("前列腺炎", "急性细菌性前列腺炎"),
                ):
                    if marker.lower() in reasoning.lower():
                        add(name)

        claim_capable_sources = {"llm", "llm_unresolved", "mechanism_reasoner", "external_retrieval"}
        for item in getattr(candidate_pool, "items", []) or []:
            source = str(getattr(item, "source", "") or "")
            if source not in claim_capable_sources:
                continue
            entity_id = str(getattr(item, "entity_id", "") or "")
            for attr in ("raw_name", "diagnosis", "canonical_name", "submission_name"):
                add(getattr(item, attr, ""), entity_id)
        for item in getattr(candidate_pool, "open_world_candidates", []) or []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "")
            if source not in claim_capable_sources:
                continue
            for key in ("raw_name", "canonical_name", "submission_name", "name"):
                add(item.get(key), item.get("entity_id"))
        return records

    @staticmethod
    def _is_leukemia(text: str) -> bool:
        return any(token in text for token in ("白血病", "leukemia", " aml", " all", "d000025"))

    @staticmethod
    def _is_pavm(text: str) -> bool:
        return any(
            token in text
            for token in (
                "肺动静脉瘘",
                "肺动静脉畸形",
                "肺内右向左分流",
                "pavm",
                "pulmonary avm",
                "d100055",
            )
        )

    @staticmethod
    def _is_prostatitis(text: str) -> bool:
        return any(token in text for token in ("前列腺炎", "prostatitis"))


def _dedupe_hypotheses(items: Sequence[EvidenceHypothesis]) -> List[EvidenceHypothesis]:
    result: List[EvidenceHypothesis] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items or []:
        key = (item.candidate, item.target_evidence_id, item.claim_type)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _text_list(values: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(str(item).strip() for item in values or [] if str(item).strip()))


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
