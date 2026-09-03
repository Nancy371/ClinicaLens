"""Stage-specific LLM context compilation and audit.

The compiler receives authoritative runtime state and produces a smaller LLM
view. It never mutates medical state and never treats compression output as
evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


class StageContextCompiler:
    """Compile runtime state into stage-specific, budgeted LLM context views."""

    VERSION = "stage_context_v2"

    DEFAULT_STAGE_BUDGETS = {
        "planning": {"max_input_tokens": 4000, "fallback_max_chars": 16000},
        "planning_criticism": {"max_input_tokens": 4000, "fallback_max_chars": 16000},
        "thinking": {"max_input_tokens": 6000, "fallback_max_chars": 24000},
        "diagnosis": {"max_input_tokens": 7000, "fallback_max_chars": 28000},
        "repair": {"max_input_tokens": 2000, "fallback_max_chars": 8000},
    }

    STAGE_ALLOWLIST = {
        "planning": {
            "collected_info",
            "exam_results",
            "chat_history",
            "phase",
            "relevant_experience",
            "previous_plan",
            "current_plan",
        },
        "planning_criticism": {
            "current_plan",
            "collected_info",
            "exam_results",
            "thinking_result",
            "action_history",
        },
        "thinking": {
            "collected_info",
            "exam_results",
            "chat_history",
            "phase",
            "relevant_experience",
            "knowledge_context",
            "evidence_summary",
        },
        "diagnosis": {
            "collected_info",
            "exam_results",
            "chat_history",
            "relevant_experience",
            "standard_diseases",
            "rag_context",
            "evidence_summary",
            "candidate_table",
            "critic_feedback",
            "current_primary",
            "top_candidates",
            "protected_candidates",
            "active_gaps",
            "new_material_evidence",
        },
        "repair": {
            "invalid_output",
            "validation_errors",
            "canonical_schema",
            "repair_instructions",
        },
    }

    ROOT_ITEM_LIMITS = {
        "planning": {"chat_history": 8, "exam_results": 12, "relevant_experience": 2},
        "planning_criticism": {"action_history": 8, "exam_results": 12},
        "thinking": {"chat_history": 8, "exam_results": 16, "relevant_experience": 2},
        "diagnosis": {
            "chat_history": 8,
            "exam_results": 18,
            "relevant_experience": 2,
            "standard_diseases": 60,
        },
        "repair": {},
    }

    ROOT_STRING_LIMITS = {
        "planning": {
            "knowledge_context": 0,
            "relevant_experience": 1800,
        },
        "thinking": {
            "knowledge_context": 3500,
            "evidence_summary": 9000,
            "relevant_experience": 1800,
        },
        "diagnosis": {
            "rag_context": 3500,
            "evidence_summary": 9000,
            "candidate_table": 9000,
            "critic_feedback": 1500,
            "relevant_experience": 1800,
        },
        "planning_criticism": {},
        "repair": {
            "invalid_output": 3500,
            "validation_errors": 2000,
            "canonical_schema": 1800,
            "repair_instructions": 1000,
        },
    }

    GENERIC_STRING_LIMIT = 1200
    GENERIC_LIST_LIMIT = 8
    GENERIC_DICT_LIMIT = 24
    TOP_PROFILE_FIELDS = 20

    DROP_PRIORITY_BY_KEYWORD = {
        "audit": "tier3",
        "trace": "tier3",
        "raw_prompt": "tier3",
        "raw_response": "tier3",
        "full_response": "tier3",
        "debug": "tier3",
        "reasoning_history": "tier3",
        "historical_rejected": "tier3",
        "disease_profile": "tier3",
        "profile": "tier3",
    }

    CRITICAL_KEYWORDS = {
        "chief_complaint",
        "target_claim",
        "target_claims",
        "hard_contradiction",
        "contradiction",
        "current_primary",
        "primary",
        "protected",
        "active_gap",
        "new_material",
        "material_evidence",
    }

    HIGH_VALUE_TEXT_MARKERS = (
        "chief",
        "主诉",
        "new material",
        "material evidence",
        "target claim",
        "hard contradiction",
        "contradiction",
        "primary",
        "protected",
        "gap",
        "objective",
        "high-value",
        "高价值",
        "客观",
        "反证",
        "矛盾",
        "放疗",
        "影像",
        "CT",
        "supported",
        "contradicted",
        "SUPPORTED",
        "CONTRADICTED",
    )

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.compact_enabled = bool(self.config.get("compact_enabled", True))
        self.stage_budgets = _merge_mapping(
            self.DEFAULT_STAGE_BUDGETS,
            self.config.get("stage_budget") or self.config.get("stage_budgets") or {},
        )

    def compile(self, stage: str, **state: Any) -> Dict[str, Any]:
        stage_name = str(stage or "unclassified")
        source = dict(state)
        source_profile = _profile_payload(source, top_n=self.TOP_PROFILE_FIELDS)
        if not self.enabled:
            audit = self._audit(
                stage_name,
                source,
                source,
                source_profile=source_profile,
                included=list(source.keys()),
                dropped=[],
                retained_counts={},
            )
            audit["omitted_sections"] = ["compiler_disabled"]
            return {"context": source, "audit": audit}

        view = self._project_stage(stage_name, source)
        retained_counts: Dict[str, int] = {}
        dropped: List[Dict[str, Any]] = []
        if self.compact_enabled:
            view = self._pack_mapping(
                stage_name,
                view,
                path=(),
                dropped=dropped,
                retained_counts=retained_counts,
                critical=False,
            )
            view = self._enforce_budget(stage_name, view, dropped)

        included = [key for key, value in view.items() if value not in (None, "", [], {})]
        audit = self._audit(
            stage_name,
            source,
            view,
            source_profile=source_profile,
            included=included,
            dropped=dropped,
            retained_counts=retained_counts,
        )
        return {"context": view, "audit": audit}

    def _project_stage(self, stage: str, source: Mapping[str, Any]) -> Dict[str, Any]:
        allow = self.STAGE_ALLOWLIST.get(stage)
        if not allow:
            allow = set(source.keys())
        return {
            key: copy.deepcopy(value)
            for key, value in source.items()
            if key in allow and value not in (None, "", [], {})
        }

    def _pack_mapping(
        self,
        stage: str,
        payload: Mapping[str, Any],
        *,
        path: Tuple[str, ...],
        dropped: List[Dict[str, Any]],
        retained_counts: MutableMapping[str, int],
        critical: bool,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        items = list(payload.items())
        limit = self._dict_limit(stage, path)
        if len(items) > limit:
            for key, value in items[limit:]:
                self._record_drop(dropped, path + (str(key),), value, "dict_item_limit")
            items = items[:limit]
        for key, value in items:
            key_text = str(key)
            child_path = path + (key_text,)
            if self._should_drop_key(child_path):
                self._record_drop(dropped, child_path, value, "tier3_field")
                continue
            child_critical = critical or self._is_critical_path(child_path)
            packed = self._pack_value(
                stage,
                value,
                path=child_path,
                dropped=dropped,
                retained_counts=retained_counts,
                critical=child_critical,
            )
            if packed not in (None, "", [], {}):
                result[key_text] = packed
        retained_counts["dict_items"] = int(retained_counts.get("dict_items", 0)) + len(result)
        return result

    def _pack_sequence(
        self,
        stage: str,
        payload: Sequence[Any],
        *,
        path: Tuple[str, ...],
        dropped: List[Dict[str, Any]],
        retained_counts: MutableMapping[str, int],
        critical: bool,
    ) -> List[Any]:
        limit = self._list_limit(stage, path)
        payload_list = list(payload)
        if len(payload_list) > limit:
            kept = payload_list[-limit:] if self._prefer_recent(path) else payload_list[:limit]
            dropped_items = (
                payload_list[:-limit] if self._prefer_recent(path) else payload_list[limit:]
            )
            for index, item in enumerate(dropped_items):
                self._record_drop(dropped, path + (f"dropped_{index}",), item, "list_item_limit")
            payload_list = kept
        result = [
            self._pack_value(
                stage,
                item,
                path=path + (str(index),),
                dropped=dropped,
                retained_counts=retained_counts,
                critical=critical or self._is_critical_payload(item),
            )
            for index, item in enumerate(payload_list)
        ]
        result = [item for item in result if item not in (None, "", [], {})]
        retained_counts["list_items"] = int(retained_counts.get("list_items", 0)) + len(result)
        return result

    def _pack_value(
        self,
        stage: str,
        value: Any,
        *,
        path: Tuple[str, ...],
        dropped: List[Dict[str, Any]],
        retained_counts: MutableMapping[str, int],
        critical: bool,
    ) -> Any:
        if isinstance(value, str):
            limit = self._string_limit(stage, path, critical)
            return self._pack_text(value, limit=limit, path=path, dropped=dropped)
        if isinstance(value, Mapping):
            return self._pack_mapping(
                stage,
                value,
                path=path,
                dropped=dropped,
                retained_counts=retained_counts,
                critical=critical,
            )
        if isinstance(value, (list, tuple)):
            return self._pack_sequence(
                stage,
                value,
                path=path,
                dropped=dropped,
                retained_counts=retained_counts,
                critical=critical,
            )
        return value

    def _pack_text(
        self,
        text: str,
        *,
        limit: int,
        path: Tuple[str, ...],
        dropped: List[Dict[str, Any]],
    ) -> str:
        if limit <= 0:
            self._record_drop(dropped, path, text, "string_section_disabled")
            return ""
        if len(text) <= limit:
            return text
        source_hash = _hash_text(text)
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            kept = text[:limit]
            omitted = len(text) - len(kept)
            self._record_drop(dropped, path, text[limit:], "string_char_limit")
            return f"{kept}\n[context_compiler: omitted_chars={omitted}; source_hash={source_hash}]"

        selected: List[str] = []
        selected_ids = set()
        for idx, line in enumerate(lines):
            if _line_has_marker(line, self.HIGH_VALUE_TEXT_MARKERS):
                selected.append(line)
                selected_ids.add(idx)
        for idx, line in enumerate(lines):
            if idx in selected_ids:
                continue
            selected.append(line)
        kept_lines: List[str] = []
        used = 0
        footer = f"[context_compiler: omitted_chars={{omitted}}; source_hash={source_hash}]"
        budget = max(0, limit - len(footer) - 2)
        for line in selected:
            projected = used + len(line) + (1 if kept_lines else 0)
            if projected > budget:
                continue
            kept_lines.append(line)
            used = projected
        omitted = max(0, len(text) - len("\n".join(kept_lines)))
        self._record_drop(dropped, path, text[limit:], "string_char_limit")
        if not kept_lines:
            kept_lines = [text[: max(0, budget)]]
            omitted = len(text) - len(kept_lines[0])
        return "\n".join(kept_lines + [footer.format(omitted=omitted)])

    def _enforce_budget(
        self,
        stage: str,
        view: Dict[str, Any],
        dropped: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        budget = self._stage_budget(stage)
        max_chars = int(budget.get("fallback_max_chars") or budget.get("max_input_tokens", 0) * 4)
        if max_chars <= 0:
            return view
        if len(_json_text(view)) <= max_chars:
            return view
        result = copy.deepcopy(view)
        optional_order = [
            "relevant_experience",
            "chat_history",
            "action_history",
            "knowledge_context",
            "rag_context",
            "standard_diseases",
            "exam_results",
        ]
        for key in optional_order:
            if len(_json_text(result)) <= max_chars:
                break
            if key not in result or self._is_critical_path((key,)):
                continue
            value = result.get(key)
            self._record_drop(dropped, (key,), value, "budget_drop")
            result.pop(key, None)
        if len(_json_text(result)) <= max_chars:
            return result

        # Final safety valve: shrink non-critical strings field by field. This is
        # not blind whole-context truncation; critical fields remain represented.
        for key in list(result.keys()):
            if len(_json_text(result)) <= max_chars:
                break
            if isinstance(result.get(key), str) and not self._is_critical_path((key,)):
                result[key] = self._pack_text(
                    result[key],
                    limit=max(400, max_chars // max(4, len(result))),
                    path=(key,),
                    dropped=dropped,
                )
        return result

    def _stage_budget(self, stage: str) -> Dict[str, int]:
        return dict(self.stage_budgets.get(stage) or {})

    def _string_limit(self, stage: str, path: Tuple[str, ...], critical: bool) -> int:
        root = path[0] if path else ""
        stage_limits = self.ROOT_STRING_LIMITS.get(stage, {})
        if root in stage_limits:
            return int(stage_limits[root])
        if critical:
            return max(self.GENERIC_STRING_LIMIT, 1800)
        return self.GENERIC_STRING_LIMIT

    def _list_limit(self, stage: str, path: Tuple[str, ...]) -> int:
        root = path[0] if path else ""
        return int(
            self.ROOT_ITEM_LIMITS.get(stage, {}).get(root)
            or self.GENERIC_LIST_LIMIT
        )

    def _dict_limit(self, stage: str, path: Tuple[str, ...]) -> int:
        root = path[0] if path else ""
        return int(
            self.ROOT_ITEM_LIMITS.get(stage, {}).get(root)
            or self.GENERIC_DICT_LIMIT
        )

    @staticmethod
    def _prefer_recent(path: Tuple[str, ...]) -> bool:
        joined = ".".join(path).lower()
        return any(key in joined for key in ("history", "action", "exam_results"))

    def _should_drop_key(self, path: Tuple[str, ...]) -> bool:
        joined = ".".join(path).lower()
        return any(keyword in joined for keyword in self.DROP_PRIORITY_BY_KEYWORD)

    def _is_critical_path(self, path: Tuple[str, ...]) -> bool:
        joined = ".".join(path).lower()
        return any(keyword in joined for keyword in self.CRITICAL_KEYWORDS)

    def _is_critical_payload(self, value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(self._is_critical_path((str(key),)) for key in value.keys())
        if isinstance(value, str):
            return _line_has_marker(value, self.HIGH_VALUE_TEXT_MARKERS)
        return False

    @staticmethod
    def _record_drop(
        dropped: List[Dict[str, Any]],
        path: Tuple[str, ...],
        value: Any,
        reason: str,
    ) -> None:
        dropped.append(
            {
                "path": ".".join(path),
                "reason": reason,
                "chars": len(_json_text(value)),
                "priority": _priority_for_path(path),
                "hash": _hash_text(_json_text(value)),
            }
        )

    def _audit(
        self,
        stage: str,
        source: Mapping[str, Any],
        compiled: Mapping[str, Any],
        *,
        source_profile: Dict[str, Any],
        included: List[str],
        dropped: List[Dict[str, Any]],
        retained_counts: Mapping[str, int],
    ) -> Dict[str, Any]:
        source_text = _json_text(source)
        compiled_text = _json_text(compiled)
        section_char_counts = {
            key: len(_json_text(value))
            for key, value in compiled.items()
        }
        budget = self._stage_budget(stage)
        max_input_tokens = int(budget.get("max_input_tokens") or 0)
        fallback_max_chars = int(budget.get("fallback_max_chars") or 0)
        source_tokens = _estimate_tokens(source_text)
        compiled_tokens = _estimate_tokens(compiled_text)
        dropped_priority_distribution: Dict[str, int] = {}
        for item in dropped:
            priority = str(item.get("priority") or "unknown")
            dropped_priority_distribution[priority] = (
                dropped_priority_distribution.get(priority, 0) + 1
            )
        audit_payload_detected = bool(source_profile.get("audit_payload_detected"))
        recursive_payload_detected = bool(source_profile.get("recursive_payload_detected"))
        critical_retained = self._critical_evidence_retained(compiled)
        return {
            "context_version": self.VERSION,
            "stage": stage,
            "source_case_version": str(source.get("case_version") or ""),
            "source_context_chars": len(source_text),
            "source_estimated_tokens": source_tokens,
            "compiled_context_chars": len(compiled_text),
            "compiled_estimated_tokens": compiled_tokens,
            # Backward-compatible names.
            "context_chars": len(compiled_text),
            "estimated_input_tokens": compiled_tokens,
            "compression_ratio": round(len(compiled_text) / max(1, len(source_text)), 6),
            "section_char_counts": section_char_counts,
            "section_token_counts": {
                key: _estimate_tokens(_json_text(value))
                for key, value in compiled.items()
            },
            "field_char_counts": source_profile.get("field_char_counts", {}),
            "largest_fields": source_profile.get("largest_fields", []),
            "retained_item_counts": dict(retained_counts),
            "dropped_item_count": len(dropped),
            "dropped_item_counts": _count_by_reason(dropped),
            "dropped_priority_distribution": dropped_priority_distribution,
            "dropped_item_preview": dropped[:12],
            "budget": {
                "max_input_tokens": max_input_tokens,
                "fallback_max_chars": fallback_max_chars,
            },
            "budget_exceeded_before_packing": (
                source_tokens > max_input_tokens if max_input_tokens else False
            ),
            "budget_violation_after_packing": (
                compiled_tokens > max_input_tokens if max_input_tokens else False
            ),
            "critical_evidence_retained": critical_retained,
            "audit_payload_detected": audit_payload_detected,
            "recursive_payload_detected": recursive_payload_detected,
            "included_sections": list(included),
            "omitted_sections": [
                item["path"]
                for item in dropped[:20]
            ],
        }

    def _critical_evidence_retained(self, compiled: Mapping[str, Any]) -> bool:
        text = _json_text(compiled).lower()
        # A context with no explicit critical markers should not be marked as a
        # failure; it simply had no critical marker to retain.
        has_known_marker = any(marker in text for marker in self.CRITICAL_KEYWORDS)
        if has_known_marker:
            return True
        return bool(compiled.get("collected_info") or compiled.get("evidence_summary"))


def _merge_mapping(
    base: Mapping[str, Mapping[str, Any]],
    override: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    result = {key: dict(value) for key, value in base.items()}
    for key, value in dict(override or {}).items():
        merged = dict(result.get(key) or {})
        merged.update(dict(value or {}))
        result[str(key)] = merged
    return result


def _profile_payload(value: Any, *, top_n: int) -> Dict[str, Any]:
    field_counts: Dict[str, int] = {}
    seen: set[int] = set()
    recursive = False
    audit_payload = False

    def walk(item: Any, path: Tuple[str, ...]) -> int:
        nonlocal recursive, audit_payload
        if isinstance(item, (Mapping, list, tuple)):
            item_id = id(item)
            if item_id in seen:
                recursive = True
                return 0
            seen.add(item_id)
        if any("audit" in part.lower() or "trace" in part.lower() for part in path):
            audit_payload = True
        if isinstance(item, Mapping):
            total = 2
            for key, child in item.items():
                total += walk(child, path + (str(key),))
            field_counts[".".join(path) or "<root>"] = total
            return total
        if isinstance(item, (list, tuple)):
            total = 2
            for index, child in enumerate(item):
                total += walk(child, path + (str(index),))
            field_counts[".".join(path) or "<root>"] = total
            return total
        chars = len(_json_text(item))
        field_counts[".".join(path) or "<root>"] = chars
        return chars

    walk(value, ())
    largest = sorted(field_counts.items(), key=lambda pair: pair[1], reverse=True)[:top_n]
    return {
        "field_char_counts": dict(largest),
        "largest_fields": [
            {"path": path, "chars": chars}
            for path, chars in largest
        ],
        "recursive_payload_detected": recursive,
        "audit_payload_detected": audit_payload,
    }


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _line_has_marker(line: str, markers: Iterable[str]) -> bool:
    lower = line.lower()
    return any(marker.lower() in lower for marker in markers)


def _priority_for_path(path: Tuple[str, ...]) -> str:
    joined = ".".join(path).lower()
    if any(key in joined for key in ("chief", "target_claim", "hard_contradiction", "primary")):
        return "tier0"
    if any(key in joined for key in ("evidence", "gap", "candidate", "required")):
        return "tier1"
    if any(key in joined for key in ("history", "experience", "older")):
        return "tier2"
    return "tier3"


def _count_by_reason(dropped: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in dropped:
        reason = str(item.get("reason") or "unknown")
        result[reason] = result.get(reason, 0) + 1
    return result
