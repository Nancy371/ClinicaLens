"""LLM JSON contract validation, normalization, and bounded repair helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


JsonDict = Dict[str, Any]


@dataclass
class StageContract:
    """Single source of truth for one structured LLM stage."""

    purpose: str
    version: str
    required_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, Tuple[type, ...]] = field(default_factory=dict)
    critical_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    max_output_tokens: int = 1024
    repair_max_output_tokens: int = 1024
    repair_priority: str = "medium"

    def schema_payload(self) -> JsonDict:
        return {
            "purpose": self.purpose,
            "version": self.version,
            "required_fields": list(self.required_fields),
            "critical_fields": list(self.critical_fields),
            "optional_fields": list(self.optional_fields),
            "field_types": {
                key: [_type_name(item) for item in value]
                for key, value in self.field_types.items()
            },
        }


@dataclass
class NormalizationResult:
    value: Any
    normalizations: List[str] = field(default_factory=list)


@dataclass
class LLMContractValidation:
    purpose: str
    contract_version: str
    applicable: bool
    schema_success: Optional[bool]
    semantic_success: Optional[bool]
    missing_fields: List[str] = field(default_factory=list)
    type_errors: List[str] = field(default_factory=list)
    semantic_errors: List[str] = field(default_factory=list)
    critical_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    consumer_acceptance_reason: str = ""
    consumer_rejection_code: str = ""

    @property
    def accepted(self) -> bool:
        if not self.applicable:
            return True
        return bool(self.schema_success) and self.semantic_success is not False

    @property
    def invalid_fields(self) -> List[str]:
        fields = list(self.missing_fields)
        for item in self.type_errors:
            field = str(item).split(":", 1)[0]
            if field and field not in fields:
                fields.append(field)
        return fields

    def to_audit(self) -> JsonDict:
        return {
            "purpose": self.purpose,
            "contract_version": self.contract_version,
            "applicable": self.applicable,
            "schema_success": self.schema_success,
            "semantic_success": self.semantic_success,
            "missing_fields": list(self.missing_fields),
            "type_errors": list(self.type_errors),
            "semantic_errors": list(self.semantic_errors),
            "critical_fields": list(self.critical_fields),
            "optional_fields": list(self.optional_fields),
            "consumer_acceptance_reason": self.consumer_acceptance_reason,
            "consumer_rejection_code": self.consumer_rejection_code,
            "accepted": self.accepted,
        }


class LLMContractExecutor:
    """Validate structured LLM outputs without making clinical decisions."""

    def __init__(self, *, enabled: bool = True, repair_enabled: bool = True):
        self.enabled = bool(enabled)
        self.repair_enabled = bool(repair_enabled)
        self.contracts: Dict[str, StageContract] = {
            "planning": StageContract(
                purpose="planning",
                version="planning.v1",
                required_fields=["strategy"],
                critical_fields=["strategy"],
                optional_fields=[
                    "primary_hypothesis",
                    "hypothesis_confidence",
                    "differential_diagnoses",
                ],
                field_types={"strategy": (dict,)},
                max_output_tokens=1536,
                repair_max_output_tokens=1024,
                repair_priority="medium",
            ),
            "planning_criticism": StageContract(
                purpose="planning_criticism",
                version="planning_criticism.v1",
                required_fields=["criticisms"],
                critical_fields=["criticisms"],
                optional_fields=["overall_assessment", "confidence_in_plan"],
                field_types={"criticisms": (list,)},
                max_output_tokens=1024,
                repair_max_output_tokens=768,
                repair_priority="low",
            ),
            "thinking": StageContract(
                purpose="thinking",
                version="thinking.v1",
                required_fields=["differential_diagnosis"],
                critical_fields=["differential_diagnosis"],
                optional_fields=[
                    "key_unknowns",
                    "is_sufficient",
                    "next_action",
                    "clinical_pattern_proposals",
                ],
                field_types={"differential_diagnosis": (list,)},
                max_output_tokens=2048,
                repair_max_output_tokens=1024,
                repair_priority="high",
            ),
            "diagnosis": StageContract(
                purpose="diagnosis",
                version="diagnosis.v1",
                required_fields=["diagnosis"],
                critical_fields=["diagnosis"],
                optional_fields=[
                    "candidate_id",
                    "decision",
                    "confidence",
                    "evidence_refs",
                    "treatment_plan",
                    "reasoning",
                ],
                field_types={"diagnosis": (list, str)},
                max_output_tokens=2048,
                repair_max_output_tokens=1024,
                repair_priority="critical",
            ),
        }

    def contract_for(self, purpose: str) -> Optional[StageContract]:
        if not self.enabled:
            return None
        return self.contracts.get(str(purpose or ""))

    # Backward-compatible alias used by older code/tests.
    schema_for = contract_for

    def required_fields_for(self, purpose: str) -> List[str]:
        contract = self.contract_for(purpose)
        return list(contract.required_fields) if contract else []

    def output_tokens_for(self, purpose: str) -> Optional[int]:
        contract = self.contract_for(purpose)
        if contract and contract.max_output_tokens > 0:
            return int(contract.max_output_tokens)
        return None

    def repair_output_tokens_for(self, purpose: str) -> Optional[int]:
        contract = self.contract_for(purpose)
        if contract and contract.repair_max_output_tokens > 0:
            return int(contract.repair_max_output_tokens)
        return None

    def repair_priority_for(self, purpose: str) -> str:
        contract = self.contract_for(purpose)
        return str(contract.repair_priority if contract else "medium")

    def normalize(self, value: Any, purpose: str) -> NormalizationResult:
        contract = self.contract_for(purpose)
        if contract is None:
            return NormalizationResult(value=value, normalizations=[])
        if not isinstance(value, dict):
            return NormalizationResult(value=value, normalizations=[])
        if set(value.keys()) == {"raw_response"} and isinstance(value.get("raw_response"), str):
            return NormalizationResult(value=value, normalizations=[])

        normalized = dict(value)
        normalizations: List[str] = []

        if contract.purpose == "planning":
            self._normalize_planning(normalized, normalizations)
        elif contract.purpose == "planning_criticism":
            self._normalize_planning_criticism(normalized, normalizations)
        elif contract.purpose == "thinking":
            self._normalize_thinking(normalized, normalizations)
        elif contract.purpose == "diagnosis":
            self._normalize_diagnosis(normalized, normalizations)

        return NormalizationResult(value=normalized, normalizations=normalizations)

    def validate(self, value: Any, purpose: str) -> LLMContractValidation:
        contract = self.contract_for(purpose)
        if contract is None:
            return LLMContractValidation(
                purpose=str(purpose or "unclassified"),
                contract_version="",
                applicable=False,
                schema_success=None,
                semantic_success=None,
            )
        if not isinstance(value, dict):
            return LLMContractValidation(
                purpose=contract.purpose,
                contract_version=contract.version,
                applicable=True,
                schema_success=False,
                semantic_success=None,
                type_errors=["root_not_object"],
                critical_fields=list(contract.critical_fields),
                optional_fields=list(contract.optional_fields),
                consumer_rejection_code="TYPE_INCOMPATIBLE",
            )
        if set(value.keys()) == {"raw_response"} and isinstance(value.get("raw_response"), str):
            return LLMContractValidation(
                purpose=contract.purpose,
                contract_version=contract.version,
                applicable=True,
                schema_success=False,
                semantic_success=None,
                type_errors=["unparsed_raw_response"],
                critical_fields=list(contract.critical_fields),
                optional_fields=list(contract.optional_fields),
                consumer_rejection_code="TYPE_INCOMPATIBLE",
            )

        missing = [
            field_name
            for field_name in contract.required_fields
            if field_name not in value
        ]
        type_errors = []
        for field_name, expected in contract.field_types.items():
            if field_name in value and not isinstance(value.get(field_name), expected):
                type_errors.append(f"{field_name}:expected_{_type_names(expected)}")
        semantic_errors = self._semantic_errors(value, contract)
        schema_success = not missing and not type_errors
        semantic_success: Optional[bool] = None
        if schema_success:
            semantic_success = not semantic_errors
        rejection_code = ""
        if missing:
            rejection_code = "MISSING_REQUIRED_FIELD"
        elif type_errors:
            rejection_code = "TYPE_INCOMPATIBLE"
        elif semantic_errors:
            rejection_code = "SEMANTIC_CONSTRAINT_FAILED"
        return LLMContractValidation(
            purpose=contract.purpose,
            contract_version=contract.version,
            applicable=True,
            schema_success=schema_success,
            semantic_success=semantic_success,
            missing_fields=missing,
            type_errors=type_errors,
            semantic_errors=semantic_errors,
            critical_fields=list(contract.critical_fields),
            optional_fields=list(contract.optional_fields),
            consumer_acceptance_reason="ACCEPTED" if schema_success and not semantic_errors else "",
            consumer_rejection_code=rejection_code,
        )

    def should_repair(self, validation: LLMContractValidation) -> bool:
        return bool(
            self.enabled
            and self.repair_enabled
            and validation.applicable
            and not validation.accepted
        )

    def build_repair_messages(
        self,
        *,
        original_messages: List[Dict[str, str]],
        previous_value: Any,
        validation: LLMContractValidation,
    ) -> List[Dict[str, str]]:
        contract = self.contract_for(validation.purpose)
        invalid_fields = validation.invalid_fields or list(validation.critical_fields)
        invalid_fields = [
            field_name
            for field_name in invalid_fields
            if not contract or field_name in set(contract.required_fields + contract.critical_fields)
        ]
        if not invalid_fields and contract:
            invalid_fields = list(contract.critical_fields or contract.required_fields)
        previous_payload = _safe_json(previous_value)
        return [
            {
                "role": "system",
                "content": (
                    "You repair a previous structured JSON response so it matches the "
                    "requested schema. Return only one JSON object. Do not redo clinical "
                    "reasoning, do not add prose, and do not invent evidence."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "purpose": validation.purpose,
                        "contract": contract.schema_payload() if contract else {},
                        "repair_fields": invalid_fields,
                        "validation_errors": validation.to_audit(),
                        "previous_response": previous_payload,
                        "instruction": (
                            "Return only the fields listed in repair_fields when possible. "
                            "Correct representation only and preserve clinical content."
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    def merge_repair(
        self,
        *,
        previous_value: Any,
        repair_value: Any,
        validation: LLMContractValidation,
        purpose: str,
    ) -> NormalizationResult:
        previous = self.normalize(previous_value, purpose).value
        repaired = self.normalize(repair_value, purpose).value
        if not isinstance(previous, dict):
            previous = {}
        if not isinstance(repaired, dict):
            return NormalizationResult(value=previous, normalizations=["repair_not_object"])
        merged = dict(previous)
        fields = validation.invalid_fields
        if not fields:
            fields = list(repaired.keys())
        copied: List[str] = []
        for field_name in fields:
            if field_name in repaired:
                merged[field_name] = repaired[field_name]
                copied.append(field_name)
        if not copied and repaired:
            # Some providers ignore the field-level instruction and return a full
            # corrected object. Accept that representation while keeping the merge
            # deterministic.
            merged.update(repaired)
            copied = list(repaired.keys())
        normalized = self.normalize(merged, purpose)
        normalizations = ["field_level_repair_merge"] + [
            f"repair_field:{field_name}" for field_name in copied
        ] + normalized.normalizations
        return NormalizationResult(value=normalized.value, normalizations=normalizations)

    @staticmethod
    def _normalize_planning(value: JsonDict, normalizations: List[str]) -> None:
        strategy = value.get("strategy")
        if isinstance(strategy, str):
            value["strategy"] = {"summary": strategy}
            normalizations.append("planning.strategy:string_to_object")

    @staticmethod
    def _normalize_planning_criticism(value: JsonDict, normalizations: List[str]) -> None:
        criticisms = value.get("criticisms")
        if isinstance(criticisms, str):
            value["criticisms"] = [{"issue": criticisms}]
            normalizations.append("planning_criticism.criticisms:string_to_object_list")
        elif isinstance(criticisms, list):
            normalized = []
            changed = False
            for item in criticisms:
                if isinstance(item, str):
                    normalized.append({"issue": item})
                    changed = True
                else:
                    normalized.append(item)
            if changed:
                value["criticisms"] = normalized
                normalizations.append("planning_criticism.criticisms:list_string_to_object")

    @staticmethod
    def _normalize_thinking(value: JsonDict, normalizations: List[str]) -> None:
        differential = value.get("differential_diagnosis")
        if isinstance(differential, dict):
            differential = list(differential.values())
            value["differential_diagnosis"] = differential
            normalizations.append("thinking.differential_diagnosis:object_to_list")
        if isinstance(differential, list):
            normalized = []
            changed = False
            for item in differential:
                if isinstance(item, str):
                    normalized.append({"diagnosis": item})
                    changed = True
                elif isinstance(item, dict):
                    entry = dict(item)
                    if "diagnosis" not in entry:
                        for alias in ("disease", "name", "entity", "candidate"):
                            if entry.get(alias):
                                entry["diagnosis"] = entry.get(alias)
                                changed = True
                                break
                    normalized.append(entry)
                else:
                    normalized.append(item)
            if changed:
                value["differential_diagnosis"] = normalized
                normalizations.append("thinking.differential_diagnosis:canonical_object_list")

    @staticmethod
    def _normalize_diagnosis(value: JsonDict, normalizations: List[str]) -> None:
        diagnosis = value.get("diagnosis")
        if isinstance(diagnosis, dict):
            for alias in ("diagnosis", "name", "disease", "entity", "candidate"):
                candidate = diagnosis.get(alias)
                if candidate:
                    value["diagnosis"] = [str(candidate)]
                    normalizations.append("diagnosis:object_to_string_list")
                    return
        if isinstance(diagnosis, list):
            normalized = []
            changed = False
            for item in diagnosis:
                if isinstance(item, dict):
                    for alias in ("diagnosis", "name", "disease", "entity", "candidate"):
                        candidate = item.get(alias)
                        if candidate:
                            normalized.append(str(candidate))
                            changed = True
                            break
                    else:
                        normalized.append(str(item))
                        changed = True
                else:
                    normalized.append(item)
            if changed:
                value["diagnosis"] = normalized
                normalizations.append("diagnosis:list_object_to_string_list")

    @staticmethod
    def _semantic_errors(value: Mapping[str, Any], contract: StageContract) -> List[str]:
        errors: List[str] = []
        if contract.purpose == "diagnosis":
            diagnosis = value.get("diagnosis")
            if isinstance(diagnosis, list) and not [
                item for item in diagnosis if str(item).strip()
            ]:
                errors.append("diagnosis_empty")
            elif isinstance(diagnosis, str) and not diagnosis.strip():
                errors.append("diagnosis_empty")
        if contract.purpose == "thinking":
            differential = value.get("differential_diagnosis")
            if isinstance(differential, list) and differential:
                if not any(
                    isinstance(item, dict) and str(item.get("diagnosis") or "").strip()
                    for item in differential
                ):
                    errors.append("differential_diagnosis_missing_entity")
        return errors


def _type_names(types: Tuple[type, ...]) -> str:
    return "_or_".join(_type_name(item) for item in types)


def _type_name(item: type) -> str:
    return getattr(item, "__name__", str(item))


def _safe_json(value: Any, *, limit: int = 4000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text
