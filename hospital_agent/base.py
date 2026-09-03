"""
Hospital Agent SDK 基类实现。

提供 BaseDoctorAgent 基类和 Actions 接口，
通过 HTTP 调用比赛服务 API 实现问诊、检查、诊疗等能力。
"""

import asyncio
import json
import logging
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


def _mean_training_value(values: List[Any]) -> Optional[float]:
    numbers: List[float] = []
    for value in values:
        try:
            if value is not None:
                numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    return round(sum(numbers) / len(numbers), 4) if numbers else None


def summarize_tool_call_audit(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate attempt-level tool audit records into logical-call metrics."""
    attempts = [record for record in records or [] if isinstance(record, dict)]
    logical: Dict[str, List[Dict[str, Any]]] = {}
    for record in attempts:
        logical_id = str(record.get("logical_call_id") or record.get("attempt_id") or "")
        logical_id = f"{record.get('patient_id') or ''}|{logical_id}"
        if not logical_id:
            continue
        logical.setdefault(logical_id, []).append(record)

    def _dist_attempt(key: str) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for record in attempts:
            value = record.get(key)
            values = value if isinstance(value, list) else [value]
            for item in values:
                text = str(item or "")
                if text:
                    result[text] = result.get(text, 0) + 1
        return result

    def _dist_logical(key: str, *, failed_only: bool = False) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for group in logical.values():
            ordered = sorted(group, key=lambda item: int(item.get("attempt_index") or 0))
            final = ordered[-1] if ordered else {}
            if failed_only and final.get("success") is not False:
                continue
            value = final.get(key)
            text = str(value or "")
            if text:
                result[text] = result.get(text, 0) + 1
        return result

    retried = 0
    recovered = 0
    exhausted = 0
    terminal_failures = 0
    for group in logical.values():
        ordered = sorted(group, key=lambda item: int(item.get("attempt_index") or 0))
        if len(ordered) > 1:
            retried += 1
        final = ordered[-1] if ordered else {}
        if len(ordered) > 1 and final.get("success") is True:
            recovered += 1
        if final.get("retry_exhausted"):
            exhausted += 1
        if final.get("success") is False:
            terminal_failures += 1

    return {
        "total_logical_calls": len(logical),
        "total_attempts": len(attempts),
        "retried_logical_calls": retried,
        "retry_recovered_calls": recovered,
        "retry_exhausted_calls": exhausted,
        "terminal_failure_logical_calls": terminal_failures,
        "failure_count_by_reason": _dist_logical(
            "primary_failure_reason", failed_only=True
        ),
        "failure_count_by_action": _dist_logical("action", failed_only=True),
        "failure_count_by_endpoint": _dist_logical("endpoint", failed_only=True),
        "attempt_failure_count_by_reason": _dist_attempt("primary_failure_reason"),
    }


def summarize_training_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-case training records without exposing credentials."""
    evaluated = [item for item in results if item.get("status") == "evaluated"]

    def metric(name: str) -> Optional[float]:
        return _mean_training_value(
            [(item.get("metrics") or {}).get(name) for item in evaluated]
        )

    def merged_distribution(name: str) -> Dict[str, int]:
        merged: Dict[str, int] = {}
        for item in evaluated:
            value = (item.get("metrics") or {}).get(name) or {}
            if not isinstance(value, dict):
                continue
            for key, count in value.items():
                try:
                    merged[str(key)] = merged.get(str(key), 0) + int(count or 0)
                except (TypeError, ValueError):
                    continue
        return merged

    audits = [item.get("audit") or {} for item in results]
    recall_values = [
        (item.get("metrics") or {}).get("candidate_recall_at_5")
        for item in evaluated
        if (item.get("metrics") or {}).get("candidate_recall_at_5") is not None
    ]
    recall20_values = [
        (item.get("metrics") or {}).get("candidate_recall_at_20")
        for item in evaluated
        if (item.get("metrics") or {}).get("candidate_recall_at_20") is not None
    ]
    critic_issue_count = sum(1 for audit in audits if audit.get("critic_issues"))
    critic_llm_count = sum(1 for audit in audits if audit.get("critic_llm_used"))
    llm_call_audits = [
        record
        for audit in audits
        for record in (audit.get("llm_call_audit") or [])
        if isinstance(record, dict)
    ]
    tool_call_audits = [
        record
        for audit in audits
        for record in (audit.get("tool_call_audit") or [])
        if isinstance(record, dict)
    ]
    llm_context_audits = [
        record
        for audit in audits
        for record in (audit.get("llm_context_audit") or [])
        if isinstance(record, dict)
    ]
    failure_attributions = [
        audit.get("failure_attribution")
        for audit in audits
        if isinstance(audit.get("failure_attribution"), dict)
    ]

    def llm_distribution(key: str) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for record in llm_call_audits:
            value = record.get(key)
            if isinstance(value, list):
                for item in value:
                    text = str(item or "")
                    if text:
                        result[text] = result.get(text, 0) + 1
            else:
                text = str(value or "")
                if text:
                    result[text] = result.get(text, 0) + 1
        return result

    def llm_context_distribution(key: str) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for record in llm_context_audits:
            text = str(record.get(key) or "")
            if text:
                result[text] = result.get(text, 0) + 1
        return result

    def _distribution_for_records(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for record in records:
            text = str(record.get(key) or "")
            if text:
                result[text] = result.get(text, 0) + 1
        return result

    llm_failure_by_purpose: Dict[str, int] = {}
    for record in llm_call_audits:
        if not record.get("primary_failure_reason") and not record.get("fallback_used"):
            continue
        purpose = str(record.get("purpose") or "unclassified")
        llm_failure_by_purpose[purpose] = llm_failure_by_purpose.get(purpose, 0) + 1

    json_response_calls = [
        record
        for record in llm_call_audits
        if record.get("json_expected")
        and record.get("model_invoked")
        and record.get("raw_response_present")
    ]
    schema_applicable_calls = [
        record for record in llm_call_audits if record.get("schema_applicable")
    ]
    repair_calls = [
        record
        for record in llm_call_audits
        if str(record.get("attempt_type") or "") == "repair"
        or record.get("contract_repair_attempted")
    ]
    diagnosis_repair_calls = [
        record for record in repair_calls if str(record.get("purpose") or "") == "diagnosis"
    ]
    contract_drift_calls = [
        record
        for record in llm_call_audits
        if record.get("contract_drift_detected")
        or "contract_drift" in (record.get("failure_flags") or [])
    ]
    fallback_call_count = sum(1 for record in llm_call_audits if record.get("fallback_used"))
    fallback_case_count = sum(
        1
        for audit in audits
        if any(
            isinstance(record, dict) and record.get("fallback_used")
            for record in (audit.get("llm_call_audit") or [])
        )
    )
    tool_logical_groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in tool_call_audits:
        logical_id = str(record.get("logical_call_id") or record.get("attempt_id") or "")
        logical_id = f"{record.get('patient_id') or ''}|{logical_id}"
        if logical_id:
            tool_logical_groups.setdefault(logical_id, []).append(record)
    tool_logical_finals = []
    for group in tool_logical_groups.values():
        ordered = sorted(group, key=lambda item: int(item.get("attempt_index") or 0))
        if ordered:
            tool_logical_finals.append(ordered[-1])

    def tool_logical_distribution(key: str, *, failed_only: bool = False) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for record in tool_logical_finals:
            if failed_only and record.get("success") is not False:
                continue
            text = str(record.get(key) or "")
            if text:
                result[text] = result.get(text, 0) + 1
        return result

    def tool_attempt_distribution(key: str) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for record in tool_call_audits:
            value = record.get(key)
            values = value if isinstance(value, list) else [value]
            for item in values:
                text = str(item or "")
                if text:
                    result[text] = result.get(text, 0) + 1
        return result

    tool_failure_case_count = 0
    for audit in audits:
        case_groups: Dict[str, List[Dict[str, Any]]] = {}
        for record in (audit.get("tool_call_audit") or []):
            if not isinstance(record, dict):
                continue
            logical_id = str(record.get("logical_call_id") or record.get("attempt_id") or "")
            logical_id = f"{record.get('patient_id') or ''}|{logical_id}"
            if logical_id:
                case_groups.setdefault(logical_id, []).append(record)
        if any(
            sorted(group, key=lambda item: int(item.get("attempt_index") or 0))[-1].get(
                "success"
            )
            is False
            for group in case_groups.values()
            if group
        ):
            tool_failure_case_count += 1
    exam_result_calls = [
        record
        for record in tool_logical_finals
        if str(record.get("endpoint") or "") == "/exam/results"
    ]
    exam_result_failed_calls = [
        record for record in exam_result_calls if record.get("success") is False
    ]
    total = len(results)
    return {
        "cases": total,
        "evaluated_cases": len(evaluated),
        "diagnosis_accuracy": metric("diagnosis_accuracy"),
        "examination_precision": metric("examination_precision"),
        "treatment_overall_score": metric("treatment_overall_score"),
        "treatment_safety": metric("treatment_safety"),
        "treatment_protocol_coverage_rate": metric(
            "treatment_protocol_coverage_rate"
        ),
        "treatment_uncovered_diagnosis_count": metric(
            "treatment_uncovered_diagnosis_count"
        ),
        "treatment_actionability_section_count": metric(
            "treatment_actionability_section_count"
        ),
        "candidate_recall_at_20": _mean_training_value(recall20_values),
        "candidate_recall_at_5": _mean_training_value(recall_values),
        "ranking_accuracy": metric("ranking_accuracy"),
        "submission_alignment": metric("submission_alignment"),
        "submission_override_count": metric("submission_override_count"),
        "etiology_preference": metric("etiology_preference"),
        "decision_override_rate": metric("decision_override_rate"),
        "judge_gap_authorization_rate": metric("judge_gap_authorization_rate"),
        "required_gap_authorized_count": metric("required_gap_authorized_count"),
        "primary_eligible_count": metric("primary_eligible_count"),
        "deferred_needs_anchor_count": metric("deferred_needs_anchor_count"),
        "differential_only_count": metric("differential_only_count"),
        "excluded_count": metric("excluded_count"),
        "judge_primary_accuracy": metric("judge_primary_accuracy"),
        "explanatory_coverage": metric("explanatory_coverage"),
        "core_explanatory_coverage": metric("core_explanatory_coverage"),
        "residual_evidence_score": metric("residual_evidence_score"),
        "residual_core_evidence_count": metric("residual_core_evidence_count"),
        "differential_exam_precision": metric("differential_exam_precision"),
        "discriminating_exam_recall": metric("discriminating_exam_recall"),
        "exam_information_gain": metric("exam_information_gain"),
        "deferred_gap_closure_rate": metric("deferred_gap_closure_rate"),
        "deferred_exam_coverage": metric("deferred_exam_coverage"),
        "gap_value_exam_selection_rate": metric("gap_value_exam_selection_rate"),
        "reserved_highest_gap_survival_rate": metric(
            "reserved_highest_gap_survival_rate"
        ),
        "exam_priority_alignment": metric("exam_priority_alignment"),
        "wrong_primary_exam_drift": metric("wrong_primary_exam_drift"),
        "deferred_gap_count": metric("deferred_gap_count"),
        "exam_priority_override_count": metric("exam_priority_override_count"),
        "special_discriminator_rate": metric("special_discriminator_rate"),
        "multi_candidate_exam_rate": metric("multi_candidate_exam_rate"),
        "generic_exam_suppression_count": metric("generic_exam_suppression_count"),
        "exam_route_authorization_blocked_count": metric(
            "exam_route_authorization_blocked_count"
        ),
        "exam_route_repeat_authorized_count": metric(
            "exam_route_repeat_authorized_count"
        ),
        "exam_route_claim_resolved_block_count": metric(
            "exam_route_claim_resolved_block_count"
        ),
        "exam_route_generic_duplicate_block_count": metric(
            "exam_route_generic_duplicate_block_count"
        ),
        "post_exam_primary_recomputed_rate": metric("post_exam_primary_recomputed_rate"),
        "discriminating_gap_closed_rate": metric("discriminating_gap_closed_rate"),
        "gap_closure_rate": metric("gap_closure_rate"),
        "dynamic_rerank_changed_primary": metric("dynamic_rerank_changed_primary"),
        "explanation_score_changed_ranking_rate": metric(
            "explanation_score_changed_ranking_rate"
        ),
        "primary_unlock_rate": metric("primary_unlock_rate"),
        "legacy_exam_package_contribution_rate": metric(
            "legacy_exam_package_contribution_rate"
        ),
        "differential_exam_contribution_rate": metric(
            "differential_exam_contribution_rate"
        ),
        "gap_state_satisfied_count": metric("gap_state_satisfied_count"),
        "gap_state_actionable_count": metric("gap_state_actionable_count"),
        "gap_state_nonblocking_count": metric("gap_state_nonblocking_count"),
        "gap_state_unsupported_count": metric("gap_state_unsupported_count"),
        "gap_state_hard_blocked_count": metric("gap_state_hard_blocked_count"),
        "gap_state_partially_satisfied_count": metric(
            "gap_state_partially_satisfied_count"
        ),
        "fallback_to_pre_discrimination_primary": metric(
            "fallback_to_pre_discrimination_primary"
        ),
        "pairwise_judge_accuracy": metric("pairwise_judge_accuracy"),
        "differential_pool_precision": metric("differential_pool_precision"),
        "differential_pool_expected_included": metric(
            "differential_pool_expected_included"
        ),
        "generic_primary_block_count": metric("generic_primary_block_count"),
        "specific_over_generic_preference_count": metric(
            "specific_over_generic_preference_count"
        ),
        "core_evidence_primary_alignment": metric("core_evidence_primary_alignment"),
        "diagnostic_evidence_primary_alignment": metric(
            "diagnostic_evidence_primary_alignment"
        ),
        "residual_core_penalty_applied_count": metric(
            "residual_core_penalty_applied_count"
        ),
        "pairwise_noise_rejection_count": metric("pairwise_noise_rejection_count"),
        "cluster_gate_rejection_count": metric("cluster_gate_rejection_count"),
        "core_evidence_coverage": metric("core_evidence_coverage"),
        "judge_deferred_primary": metric("judge_deferred_primary"),
        "unauthorized_exam_count": metric("unauthorized_exam_count"),
        "required_evidence_coverage": metric("required_evidence_coverage"),
        "soft_contradiction_count": metric("soft_contradiction_count"),
        "hard_contradiction_count": metric("hard_contradiction_count"),
        "high_information_finding_count": metric("high_information_finding_count"),
        "generic_finding_shadowed_count": metric("generic_finding_shadowed_count"),
        "reasoning_inference_finding_count": metric("reasoning_inference_finding_count"),
        "raw_case_finding_count": metric("raw_case_finding_count"),
        "reasoning_inference_used_by_primary": metric(
            "reasoning_inference_used_by_primary"
        ),
        "blocked_reasoning_inference_count": metric(
            "blocked_reasoning_inference_count"
        ),
        "reasoning_structured_conflict_count": metric(
            "reasoning_structured_conflict_count"
        ),
        "conflict_deferred_primary_count": metric("conflict_deferred_primary_count"),
        "conflict_blocked_final_count": metric("conflict_blocked_final_count"),
        "root_cause_arbitration_count": metric("root_cause_arbitration_count"),
        "root_cause_primary_override_count": metric("root_cause_primary_override_count"),
        "root_cause_secondary_submission_count": metric(
            "root_cause_secondary_submission_count"
        ),
        "root_cause_coverage": metric("root_cause_coverage"),
        "candidate_policy_count": metric("candidate_policy_count"),
        "policy_promotion_count": metric("policy_promotion_count"),
        "policy_quarantine_count": metric("policy_quarantine_count"),
        "policy_rejected_count": metric("policy_rejected_count"),
        "policy_conflict_count": metric("policy_conflict_count"),
        "failure_stage_distribution": merged_distribution("failure_stage_distribution"),
        "evidence_hypothesis_count": metric("evidence_hypothesis_count"),
        "evidence_query_task_count": metric("evidence_query_task_count"),
        "evidence_hypothesis_verification_rate": metric(
            "evidence_hypothesis_verification_rate"
        ),
        "evidence_recovery_count": metric("evidence_recovery_count"),
        "evidence_recovery_rate": metric("evidence_recovery_rate"),
        "false_evidence_injection_rate": metric("false_evidence_injection_rate"),
        "unverified_evidence_leakage": metric("unverified_evidence_leakage"),
        "conflict_closure_rate": metric("conflict_closure_rate"),
        "protected_candidate_rescue_count": metric("protected_candidate_rescue_count"),
        "derived_pattern_count": metric("derived_pattern_count"),
        "generic_only_candidate_count": metric("generic_only_candidate_count"),
        "evidence_information_value_mean": metric("evidence_information_value_mean"),
        "critic_issue_rate": round(critic_issue_count / total, 4) if total else 0.0,
        "critic_llm_rate": round(critic_llm_count / total, 4) if total else 0.0,
        "llm_call_count": len(llm_call_audits),
        "llm_call_count_by_purpose": llm_distribution("purpose"),
        "llm_model_invocation_count": sum(
            1 for record in llm_call_audits if record.get("model_invoked")
        ),
        "llm_failure_count_by_reason": llm_distribution("primary_failure_reason"),
        "llm_failure_count_by_purpose": llm_failure_by_purpose,
        "llm_parse_failure_rate": (
            round(
                sum(1 for record in json_response_calls if record.get("parse_success") is False)
                / len(json_response_calls),
                4,
            )
            if json_response_calls
            else 0.0
        ),
        "llm_schema_failure_rate": (
            round(
                sum(
                    1
                    for record in schema_applicable_calls
                    if record.get("schema_success") is False
                )
                / len(schema_applicable_calls),
                4,
            )
            if schema_applicable_calls
            else 0.0
        ),
        "llm_timeout_rate": (
            round(
                sum(
                    1
                    for record in llm_call_audits
                    if "timeout" in (record.get("failure_flags") or [])
                )
                / len(llm_call_audits),
                4,
            )
            if llm_call_audits
            else 0.0
        ),
        "llm_truncation_rate": (
            round(
                sum(
                    1
                    for record in llm_call_audits
                    if "generation_truncated" in (record.get("failure_flags") or [])
                )
                / len(llm_call_audits),
                4,
            )
            if llm_call_audits
            else 0.0
        ),
        "llm_generation_truncation_count_by_purpose": {
            purpose: count
            for purpose, count in _distribution_for_records(
                [
                    record
                    for record in llm_call_audits
                    if "generation_truncated" in (record.get("failure_flags") or [])
                ],
                "purpose",
            ).items()
        },
        "llm_schema_type_mismatch_count": sum(
            1
            for record in llm_call_audits
            if "schema_type_mismatch" in (record.get("failure_flags") or [])
        ),
        "llm_contract_drift_count": len(contract_drift_calls),
        "llm_contract_drift_rate": (
            round(len(contract_drift_calls) / len(llm_call_audits), 4)
            if llm_call_audits
            else 0.0
        ),
        "llm_deterministic_normalization_count": sum(
            len(record.get("deterministic_normalizations") or [])
            for record in llm_call_audits
        ),
        "llm_repair_attempt_count": len(repair_calls),
        "llm_repair_success_count": sum(
            1 for record in repair_calls if record.get("contract_repair_succeeded") is True
        ),
        "llm_repair_success_rate": (
            round(
                sum(
                    1
                    for record in repair_calls
                    if record.get("contract_repair_succeeded") is True
                )
                / len(repair_calls),
                4,
            )
            if repair_calls
            else 0.0
        ),
        "llm_repair_budget_skip_count": sum(
            1
            for record in repair_calls
            if record.get("model_invoked") is False
            and record.get("primary_failure_reason") == "llm_budget_exhausted"
        ),
        "llm_diagnosis_repair_budget_skip_count": sum(
            1
            for record in diagnosis_repair_calls
            if record.get("model_invoked") is False
            and record.get("primary_failure_reason") == "llm_budget_exhausted"
        ),
        "llm_consumer_rejection_rate": (
            round(
                sum(1 for record in llm_call_audits if record.get("consumer_accepted") is False)
                / len(llm_call_audits),
                4,
            )
            if llm_call_audits
            else 0.0
        ),
        "llm_fallback_call_rate": (
            round(fallback_call_count / len(llm_call_audits), 4)
            if llm_call_audits
            else 0.0
        ),
        "llm_fallback_case_rate": round(fallback_case_count / total, 4) if total else 0.0,
        "failure_attribution_domain_distribution": _distribution_for_records(
            failure_attributions,
            "primary_failure_domain",
        ),
        "failure_attribution_reason_distribution": _distribution_for_records(
            failure_attributions,
            "primary_failure_reason",
        ),
        "medical_failure_evaluable_rate": (
            round(
                sum(1 for item in failure_attributions if item.get("medical_failure_evaluable"))
                / len(failure_attributions),
                4,
            )
            if failure_attributions
            else 0.0
        ),
        "llm_context_compile_count": len(llm_context_audits),
        "llm_context_compile_count_by_stage": llm_context_distribution("stage"),
        "average_llm_context_chars": _mean_training_value(
            [record.get("context_chars") for record in llm_context_audits]
        ),
        "average_llm_context_estimated_input_tokens": _mean_training_value(
            [record.get("estimated_input_tokens") for record in llm_context_audits]
        ),
        "average_llm_source_context_chars": _mean_training_value(
            [record.get("source_context_chars") for record in llm_context_audits]
        ),
        "average_llm_source_estimated_tokens": _mean_training_value(
            [record.get("source_estimated_tokens") for record in llm_context_audits]
        ),
        "average_llm_context_compression_ratio": _mean_training_value(
            [record.get("compression_ratio") for record in llm_context_audits]
        ),
        "llm_context_budget_violation_count": sum(
            1 for record in llm_context_audits if record.get("budget_violation_after_packing")
        ),
        "llm_context_audit_payload_detected_count": sum(
            1 for record in llm_context_audits if record.get("audit_payload_detected")
        ),
        "llm_context_recursive_payload_detected_count": sum(
            1 for record in llm_context_audits if record.get("recursive_payload_detected")
        ),
        "tool_logical_call_count": len(tool_logical_finals),
        "tool_attempt_count": len(tool_call_audits),
        "tool_logical_call_count_by_action": tool_logical_distribution("action"),
        "tool_logical_call_count_by_endpoint": tool_logical_distribution("endpoint"),
        "tool_attempt_failure_count_by_reason": tool_attempt_distribution(
            "primary_failure_reason"
        ),
        "tool_logical_failure_count_by_reason": tool_logical_distribution(
            "primary_failure_reason", failed_only=True
        ),
        "tool_retried_logical_call_count": sum(
            1 for group in tool_logical_groups.values() if len(group) > 1
        ),
        "tool_retry_recovery_count": sum(
            1
            for group in tool_logical_groups.values()
            if len(group) > 1
            and sorted(group, key=lambda item: int(item.get("attempt_index") or 0))[-1].get(
                "success"
            )
            is True
        ),
        "tool_retry_exhausted_count": sum(
            1 for record in tool_logical_finals if record.get("retry_exhausted")
        ),
        "tool_failure_case_count": tool_failure_case_count,
        "tool_failure_case_rate": round(tool_failure_case_count / total, 4)
        if total
        else 0.0,
        "exam_results_logical_call_count": len(exam_result_calls),
        "exam_results_logical_failure_count": len(exam_result_failed_calls),
        "exam_results_failure_rate": (
            round(len(exam_result_failed_calls) / len(exam_result_calls), 4)
            if exam_result_calls
            else 0.0
        ),
        "exam_results_retry_exhausted_count": sum(
            1 for record in exam_result_failed_calls if record.get("retry_exhausted")
        ),
        "average_elapsed_seconds": _mean_training_value(
            [audit.get("elapsed_seconds") for audit in audits]
        ),
        "timeout_cases": sum(1 for audit in audits if audit.get("timed_out")),
        "backend_error_cases": sum(
            1 for item in results if item.get("evaluation_error")
        ),
        "reflection_error_cases": sum(
            1 for item in results if item.get("reflection_error")
        ),
    }


class Actions:
    """比赛能力接口，封装与服务端的 HTTP 交互。

    提供 5 个核心 Action：
    - ask_patient：询问患者
    - order_examination：申请检查
    - prescribe_treatment：提交诊断和治疗方案
    - evaluation：训练阶段获取评测结果
    - batch_evaluation：批量评估
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        team_id: str,
        endpoint_prefixes: Optional[List[str]] = None,
        model_api_key: str = "",
        gateway_token: str = "",
        use_invoke: bool = True,
        invoke_path: str = "/invoke",
        exam_results_path: str = "/exam/results",
        case_evaluation_path: str = "/evaluate/case",
        batch_evaluation_path: str = "/evaluate",
    ):
        """初始化 Actions。

        Args:
            base_url: 服务基础 URL
            token: 认证 token
            team_id: 团队 ID
        """
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.team_id = team_id
        self.endpoint_prefixes = self._normalize_endpoint_prefixes(endpoint_prefixes or [])
        self.model_api_key = model_api_key
        # Optional hosting-gateway credential (for example ModelScope).  This is
        # deliberately separate from the hospital service token because the
        # gateway and the application authenticate at different layers.
        self.gateway_token = gateway_token
        self.use_invoke = bool(use_invoke)
        self.invoke_path = "/" + str(invoke_path or "/invoke").lstrip("/")
        self.exam_results_path = "/" + str(exam_results_path or "/exam/results").lstrip("/")
        self.case_evaluation_path = "/" + str(case_evaluation_path or "/evaluate/case").lstrip("/")
        self.batch_evaluation_path = "/" + str(batch_evaluation_path or "/evaluate").lstrip("/")
        self._conversation_rounds: Dict[str, int] = {}
        self._ordered_examinations: Dict[str, List[str]] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self.trace_collector = None
        self.tool_call_audit: List[Dict[str, Any]] = []
        self._tool_logical_call_index = 0

    def begin_case(self, patient_id: str) -> None:
        """Reset per-case tool audit state."""
        self.tool_call_audit = []
        self._tool_logical_call_index = 0

    def snapshot_tool_audit(self) -> List[Dict[str, Any]]:
        return [dict(record) for record in self.tool_call_audit]

    def tool_contract_summary(self) -> Dict[str, Any]:
        return summarize_tool_call_audit(self.snapshot_tool_audit())

    def _next_tool_logical_call_id(self) -> str:
        self._tool_logical_call_index += 1
        return f"TC{self._tool_logical_call_index:04d}"

    @staticmethod
    def _response_shape(value: Any) -> str:
        if isinstance(value, dict):
            return "dict"
        if isinstance(value, list):
            return "list"
        if value is None:
            return "null"
        return type(value).__name__

    @staticmethod
    def _classify_tool_exception(error: BaseException) -> Dict[str, Any]:
        if isinstance(error, httpx.ReadTimeout):
            return {"flags": ["timeout", "read_timeout"], "reason": "read_timeout"}
        if isinstance(error, httpx.ConnectTimeout):
            return {"flags": ["timeout", "connect_timeout"], "reason": "connect_timeout"}
        if isinstance(error, httpx.ConnectError):
            return {"flags": ["connect_error"], "reason": "connect_error"}
        return {"flags": ["unknown_exception"], "reason": "unknown_exception"}

    @staticmethod
    def _classify_http_status(status_code: int) -> Dict[str, Any]:
        if status_code == 429:
            return {"flags": ["http_429"], "reason": "http_429", "retryable": True}
        if 500 <= status_code:
            return {"flags": ["http_5xx"], "reason": "http_5xx", "retryable": True}
        if 400 <= status_code:
            return {"flags": ["http_4xx"], "reason": "http_4xx", "retryable": False}
        return {"flags": [], "reason": "", "retryable": False}

    def _append_tool_attempt_record(self, record: Dict[str, Any]) -> None:
        self.tool_call_audit.append(record)

    def _trace_tool_called(
        self,
        tool_name: str,
        patient_id: str = "",
        arguments: Optional[Dict[str, Any]] = None,
        *,
        target_gap_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        collector = getattr(self, "trace_collector", None)
        if not collector or not getattr(collector, "enabled", False):
            return None
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        try:
            collector.emit_event(
                "tool.called",
                payload={
                    "tool_name": tool_name,
                    "tool_version": "hospital_agent.actions.v1",
                    "call_id": call_id,
                    "patient_id": patient_id,
                    "arguments": arguments or {},
                    "attempt": 1,
                    "timeout_ms": 60000,
                    "target_gap_ids": target_gap_ids or [],
                },
                stage="tool",
                component="Actions",
                action=tool_name,
            )
        except Exception:
            if not getattr(getattr(collector, "config", None), "fail_open", True):
                raise
        return call_id

    def _trace_tool_returned(
        self,
        tool_name: str,
        call_id: Optional[str],
        raw_result: Any,
        normalized_result: Any = None,
    ) -> None:
        collector = getattr(self, "trace_collector", None)
        if not call_id or not collector or not getattr(collector, "enabled", False):
            return
        try:
            output_refs = []
            if getattr(getattr(collector, "config", None), "capture_raw_tool_result", True):
                raw_ref = collector.create_artifact("tool_result_raw", raw_result)
                if raw_ref:
                    output_refs.append(raw_ref)
            normalized_ref = collector.create_artifact(
                "tool_result_normalized",
                normalized_result if normalized_result is not None else raw_result,
            )
            if normalized_ref:
                output_refs.append(normalized_ref)
            collector.emit_event(
                "tool.returned",
                payload={
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "result_status": "success",
                    "backend_request_id": (
                        raw_result.get("request_id")
                        if isinstance(raw_result, dict)
                        else None
                    ),
                    "retry_count": 0,
                },
                output_refs=output_refs,
                stage="tool",
                component="Actions",
                action=tool_name,
            )
        except Exception:
            if not getattr(getattr(collector, "config", None), "fail_open", True):
                raise

    def _trace_tool_failed(
        self,
        tool_name: str,
        call_id: Optional[str],
        error: BaseException,
        *,
        retryable: bool = False,
        will_retry: bool = False,
    ) -> None:
        collector = getattr(self, "trace_collector", None)
        if not call_id or not collector or not getattr(collector, "enabled", False):
            return
        try:
            collector.emit_event(
                "tool.failed",
                payload={
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "error_type": type(error).__name__,
                    "error_code": getattr(error, "errno", None),
                    "message": str(error),
                    "retryable": bool(retryable),
                    "attempt": 1,
                    "will_retry": bool(will_retry),
                },
                status="failed",
                stage="tool",
                component="Actions",
                action=tool_name,
            )
        except Exception:
            if not getattr(getattr(collector, "config", None), "fail_open", True):
                raise

    @staticmethod
    def _normalize_endpoint_prefixes(prefixes: List[str]) -> List[str]:
        normalized = [""]
        for prefix in prefixes or []:
            text = str(prefix or "").strip()
            if not text:
                continue
            if not text.startswith("/"):
                text = "/" + text
            text = text.rstrip("/")
            if text not in normalized:
                normalized.append(text)
        return normalized

    def _candidate_paths(self, path: str) -> List[str]:
        clean_path = "/" + str(path or "").lstrip("/")
        paths: List[str] = []
        for prefix in self.endpoint_prefixes:
            candidate = f"{prefix}{clean_path}" if prefix else clean_path
            if candidate not in paths:
                paths.append(candidate)
        return paths

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建异步 HTTP 客户端。"""
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("SERVICE_BASE_URL must include http:// or https://")
        if not self.token:
            raise ValueError("SERVICE_TRAIN_TOKEN is required")
        if not self.team_id:
            raise ValueError("TEAM_ID is required")
        if self._client is None or self._client.is_closed:
            headers = {
                "X-Hospital-Service-Token": self.token,
                "Content-Type": "application/json",
                "X-Team-ID": self.team_id,
            }
            if self.gateway_token:
                headers["Authorization"] = f"Bearer {self.gateway_token}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=60.0,
            )
        return self._client

    async def _request_legacy(self, method: str, path: str, **kwargs) -> Any:
        """发送 HTTP 请求。

        Args:
            method: HTTP 方法
            path: API 路径
            **kwargs: 传递给 httpx 的额外参数

        Returns:
            响应 JSON 数据

        Raises:
            httpx.HTTPError: HTTP 请求失败
        """
        client = await self._get_client()
        last_error: Optional[httpx.HTTPStatusError] = None
        candidate_paths = self._candidate_paths(path)
        tried_paths: List[str] = []
        max_retries = 3
        for candidate_path in candidate_paths:
            tried_paths.append(candidate_path)
            # 针对 asyncio 事件循环下偶发 DNS/连接抖动加入短延迟指数退避重试
            response = None
            for attempt in range(max_retries):
                try:
                    response = await client.request(method, candidate_path, **kwargs)
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as net_err:
                    if attempt == max_retries - 1:
                        logger.error(
                            "[Action] %s %s 网络错误(已重试 %d 次): %s",
                            method, candidate_path, max_retries, net_err,
                        )
                        raise
                    delay = 0.5 * (2 ** attempt)
                    logger.warning(
                        "[Action] %s %s 网络错误(第 %d 次): %s, %.1fs 后重试",
                        method, candidate_path, attempt + 1, net_err, delay,
                    )
                    await asyncio.sleep(delay)
            try:
                response.raise_for_status()
                if candidate_path != path:
                    logger.info(
                        "[Action] endpoint fallback 命中: %s -> %s",
                        path,
                        candidate_path,
                    )
                return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 404 and candidate_path != candidate_paths[-1]:
                    logger.warning(
                        "[Action] %s %s 返回 404，尝试下一个 endpoint 前缀",
                        method,
                        candidate_path,
                    )
                    continue
                break

        if last_error is not None:
            response = last_error.response
            detail = response.text[:300] if response is not None else ""
            if response is not None and response.status_code == 404:
                logger.error(
                    "[Action] 404 Not Found: base_url=%s, tried_paths=%s, "
                    "可能原因：病例 ID 在当前 token/team 下不可访问，或 SERVICE_BASE_URL/endpoint 前缀不匹配。"
                    "response=%s",
                    self.base_url,
                    tried_paths,
                    detail,
                )
            raise last_error
        raise RuntimeError(f"HTTP request failed before sending: {method} {path}")

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        """Send an HTTP request and record attempt-level tool audit."""
        audit_context = dict(kwargs.pop("_audit_context", {}) or {})
        client = await self._get_client()
        last_error: Optional[httpx.HTTPStatusError] = None
        candidate_paths = self._candidate_paths(path)
        tried_paths: List[str] = []
        max_retries = 3
        logical_call_id = self._next_tool_logical_call_id()
        attempt_counter = 0
        action = str(audit_context.get("action") or "http_request")
        patient_id = str(audit_context.get("patient_id") or "")
        items = audit_context.get("items") or []
        if not isinstance(items, list):
            items = [items]
        item_names = [str(item) for item in items if str(item).strip()]

        def _base_record(
            candidate_path: str,
            attempt_index: int,
            started: float,
            started_at: str,
        ) -> Dict[str, Any]:
            return {
                "attempt_id": f"{logical_call_id}-A{attempt_index}",
                "logical_call_id": logical_call_id,
                "attempt_index": attempt_index,
                "patient_id": patient_id,
                "action": action,
                "endpoint": candidate_path,
                "method": method.upper(),
                "items_count": len(item_names),
                "item_names_preview": item_names[:10],
                "started_at": started_at,
                "latency_ms": round((time.monotonic() - started) * 1000, 2),
                "http_status": None,
                "response_present": False,
                "response_chars": 0,
                "content_type": "",
                "json_decode_success": None,
                "response_shape": "",
                "success": False,
                "failure_flags": [],
                "primary_failure_reason": "",
                "retryable": False,
                "will_retry": False,
                "retry_exhausted": False,
                "exception_type": "",
                "backend_request_id": "",
            }

        for candidate_path in candidate_paths:
            tried_paths.append(candidate_path)
            response = None
            for attempt in range(max_retries):
                attempt_counter += 1
                started_at = (
                    datetime.now(timezone.utc)
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z")
                )
                started = time.monotonic()
                try:
                    response = await client.request(method, candidate_path, **kwargs)
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as net_err:
                    classification = self._classify_tool_exception(net_err)
                    will_retry = attempt < max_retries - 1
                    record = _base_record(
                        candidate_path, attempt_counter, started, started_at
                    )
                    record.update(
                        {
                            "failure_flags": list(classification["flags"])
                            + ([] if will_retry else ["retry_exhausted"]),
                            "primary_failure_reason": classification["reason"],
                            "retryable": True,
                            "will_retry": will_retry,
                            "retry_exhausted": not will_retry,
                            "exception_type": type(net_err).__name__,
                        }
                    )
                    self._append_tool_attempt_record(record)
                    if not will_retry:
                        logger.error(
                            "[Action] %s %s 网络错误(已重试 %d 次): %s",
                            method, candidate_path, max_retries, net_err,
                        )
                        raise
                    delay = 0.5 * (2 ** attempt)
                    logger.warning(
                        "[Action] %s %s 网络错误(第 %d 次): %s, %.1fs 后重试",
                        method, candidate_path, attempt + 1, net_err, delay,
                    )
                    await asyncio.sleep(delay)
            try:
                response.raise_for_status()
                try:
                    payload = response.json()
                except ValueError as json_err:
                    response_text = response.text or ""
                    if response_text:
                        failure_reason = "invalid_json_response"
                        failure_flags = ["invalid_json_response"]
                    else:
                        failure_reason = "empty_response"
                        failure_flags = ["empty_response"]
                    record = _base_record(
                        candidate_path, attempt_counter, started, started_at
                    )
                    record.update(
                        {
                            "http_status": response.status_code,
                            "response_present": True,
                            "response_chars": len(response_text),
                            "content_type": response.headers.get("content-type", ""),
                            "json_decode_success": False,
                            "failure_flags": failure_flags,
                            "primary_failure_reason": failure_reason,
                            "exception_type": type(json_err).__name__,
                        }
                    )
                    self._append_tool_attempt_record(record)
                    raise
                record = _base_record(
                    candidate_path, attempt_counter, started, started_at
                )
                record.update(
                    {
                        "http_status": response.status_code,
                        "response_present": True,
                        "response_chars": len(response.text or ""),
                        "content_type": response.headers.get("content-type", ""),
                        "json_decode_success": True,
                        "response_shape": self._response_shape(payload),
                        "success": True,
                        "backend_request_id": str(payload.get("request_id") or "")
                        if isinstance(payload, dict)
                        else "",
                    }
                )
                self._append_tool_attempt_record(record)
                if candidate_path != path:
                    logger.info(
                        "[Action] endpoint fallback 命中: %s -> %s",
                        path,
                        candidate_path,
                    )
                return payload
            except httpx.HTTPStatusError as e:
                last_error = e
                response = e.response
                status_code = response.status_code if response is not None else 0
                classification = self._classify_http_status(status_code)
                will_retry = status_code == 404 and candidate_path != candidate_paths[-1]
                record = _base_record(
                    candidate_path, attempt_counter, started, started_at
                )
                record.update(
                    {
                        "http_status": status_code or None,
                        "response_present": response is not None,
                        "response_chars": len(response.text or "")
                        if response is not None
                        else 0,
                        "content_type": response.headers.get("content-type", "")
                        if response is not None
                        else "",
                        "json_decode_success": False,
                        "failure_flags": classification["flags"],
                        "primary_failure_reason": classification["reason"],
                        "retryable": bool(classification.get("retryable")),
                        "will_retry": will_retry,
                        "exception_type": type(e).__name__,
                    }
                )
                self._append_tool_attempt_record(record)
                if will_retry:
                    logger.warning(
                        "[Action] %s %s 返回 404，尝试下一个 endpoint 前缀",
                        method,
                        candidate_path,
                    )
                    continue
                break

        if last_error is not None:
            response = last_error.response
            detail = response.text[:300] if response is not None else ""
            if response is not None and response.status_code == 404:
                logger.error(
                    "[Action] 404 Not Found: base_url=%s, tried_paths=%s, "
                    "可能原因：病例 ID 在当前 token/team 下不可访问，或 SERVICE_BASE_URL/endpoint 前缀不匹配。"
                    "response=%s",
                    self.base_url,
                    tried_paths,
                    detail,
                )
            raise last_error
        raise RuntimeError(f"HTTP request failed before sending: {method} {path}")

    async def _invoke_action(
        self,
        patient_id: str,
        action: str,
        input_data: Dict[str, Any],
        audit_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Call the unified /invoke action endpoint used by the current service."""
        if not self.model_api_key:
            raise ValueError("MODEL_API_KEY is required when service.use_invoke is enabled")
        payload = {
            "team_id": self.team_id,
            "api_key": self.model_api_key,
            "patient_id": patient_id,
            "input": {
                "action": action,
                "input_data": input_data,
            },
        }
        return await self._request(
            "POST",
            self.invoke_path,
            json=payload,
            _audit_context=audit_context
            or {"action": action, "patient_id": patient_id},
        )

    async def _invoke_or_direct(
        self,
        patient_id: str,
        action: str,
        invoke_input: Dict[str, Any],
        direct_path: str,
        direct_payload: Dict[str, Any],
        audit_context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Prefer /invoke, with direct action endpoints as a compatibility fallback."""
        if self.use_invoke:
            try:
                return await self._invoke_action(
                    patient_id,
                    action,
                    invoke_input,
                    audit_context=audit_context,
                )
            except httpx.HTTPStatusError as e:
                if e.response is None or e.response.status_code != 404:
                    raise
                logger.warning(
                    "[Action] %s %s returned 404, falling back to %s",
                    "POST",
                    self.invoke_path,
                    direct_path,
                )
        return await self._request(
            "POST",
            direct_path,
            json=direct_payload,
            _audit_context=audit_context
            or {"action": action, "patient_id": patient_id},
        )

    def _remember_ordered_examinations(
        self,
        patient_id: str,
        response: Dict[str, Any],
        requested_items: Optional[List[str]] = None,
    ) -> None:
        requested = [
            str(item).strip()
            for item in (requested_items or [])
            if str(item).strip()
        ]
        results = response.get("results") if isinstance(response, dict) else None
        ordered = self._ordered_examinations.setdefault(patient_id, [])

        invalid_items = set()
        response_items: List[str] = []
        if isinstance(results, dict):
            for item, detail in results.items():
                item_name = str(item).strip()
                if not item_name:
                    continue
                status = detail.get("status") if isinstance(detail, dict) else None
                result_text = detail.get("result") if isinstance(detail, dict) else None
                if status == "invalid" or result_text == "无效检查":
                    invalid_items.add(item_name)
                    continue
                response_items.append(item_name)

        candidates = [item for item in requested if item not in invalid_items]
        for item in response_items:
            if item not in candidates:
                candidates.append(item)

        for item in candidates:
            if item not in ordered:
                ordered.append(item)

    async def ask_patient(
        self, patient_id: str, input_data: Dict[str, Any]
    ) -> str:
        """询问患者。

        Args:
            patient_id: 患者 ID
            input_data: 输入数据，包含 question 和 chat_history

        Returns:
            患者的回复文本
        """
        logger.info(f"[Action] ask_patient: patient_id={patient_id}")
        call_id = self._trace_tool_called(
            "ask_patient",
            patient_id,
            {"input_data": input_data},
        )
        try:
            result = await self._invoke_or_direct(
                patient_id=patient_id,
                action="ask_patient",
                invoke_input=input_data,
                direct_path="/ask_patient",
                direct_payload={
                    "patient_id": patient_id,
                    "input_data": input_data,
                    "team_id": self.team_id,
                },
                audit_context={
                    "action": "ask_patient",
                    "patient_id": patient_id,
                },
            )
            answer = result.get("answer", result.get("response", ""))
            if isinstance(answer, dict):
                answer = json.dumps(answer, ensure_ascii=False)
            self._conversation_rounds[patient_id] = (
                self._conversation_rounds.get(patient_id, 0) + 1
            )
            self._trace_tool_returned("ask_patient", call_id, result, {"answer": str(answer)})
            return str(answer)
        except Exception as e:
            self._trace_tool_failed("ask_patient", call_id, e, retryable=True)
            logger.error(f"[Action] ask_patient 失败: {e}")
            raise

    async def order_examination(
        self,
        patient_id: str,
        items: List[str],
        reason: str = "",
    ) -> Dict[str, Any]:
        """申请检查。

        Args:
            patient_id: 患者 ID
            items: 检查项目列表
            reason: 申请原因

        Returns:
            检查结果字典，包含 "results" 键
        """
        logger.info(
            f"[Action] order_examination: patient_id={patient_id}, items={items}"
        )
        call_id = self._trace_tool_called(
            "order_examination",
            patient_id,
            {"items": items, "reason": reason},
        )
        try:
            result = await self._request(
                "POST",
                self.exam_results_path,
                json={
                    "patient_id": patient_id,
                    "items": items,
                    "reason": reason,
                    "team_id": self.team_id,
                },
                _audit_context={
                    "action": "order_examination",
                    "patient_id": patient_id,
                    "items": items,
                },
            )
            self._remember_ordered_examinations(patient_id, result, items)
            self._trace_tool_returned(
                "order_examination",
                call_id,
                result,
                {
                    "ordered_items": self._ordered_examinations.get(patient_id, []),
                    "result_names": list((result.get("results") or {}).keys())
                    if isinstance(result, dict)
                    else [],
                },
            )
            return result
        except Exception as e:
            self._trace_tool_failed("order_examination", call_id, e, retryable=True)
            logger.error(f"[Action] order_examination 失败: {e}")
            raise

    async def prescribe_treatment(
        self,
        patient_id: str,
        diagnosis: List[str],
        treatment_plan: str,
        reasoning: str = "",
    ) -> Dict[str, Any]:
        """提交诊断和治疗方案。

        Args:
            patient_id: 患者 ID
            diagnosis: 诊断列表
            treatment_plan: 治疗方案
            reasoning: 诊断推理

        Returns:
            提交结果
        """
        logger.info(
            f"[Action] prescribe_treatment: patient_id={patient_id}, diagnosis={diagnosis}"
        )
        call_id = self._trace_tool_called(
            "prescribe_treatment",
            patient_id,
            {
                "diagnosis": diagnosis,
                "treatment_plan": treatment_plan,
                "reasoning": reasoning,
            },
        )
        result = {
            "patient_id": patient_id,
            "team_id": self.team_id,
            "diagnosis": diagnosis,
            "treatment_plan": treatment_plan,
            "reasoning": reasoning,
            "ordered_examinations": self._ordered_examinations.get(patient_id, []),
            "conversation_rounds": self._conversation_rounds.get(patient_id, 0),
            "finished": True,
        }
        self._trace_tool_returned(
            "prescribe_treatment",
            call_id,
            result,
            {"diagnosis": diagnosis, "finished": True},
        )
        return result

    async def evaluation(
        self,
        patient_id: str,
        final_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """训练阶段获取评测结果。

        Args:
            patient_id: 患者 ID
            final_result: 最终诊疗结果

        Returns:
            评测报告字典
        """
        logger.info(f"[Action] evaluation: patient_id={patient_id}")
        call_id = self._trace_tool_called(
            "evaluation",
            patient_id,
            {"final_result": final_result},
        )
        try:
            if not self.model_api_key:
                raise ValueError("MODEL_API_KEY is required for evaluation")
            result = await self._request(
                "POST",
                self.case_evaluation_path,
                json={
                    "patient_id": patient_id,
                    "api_key": self.model_api_key,
                    "final_result": final_result,
                    "team_id": self.team_id,
                },
                _audit_context={
                    "action": "evaluation",
                    "patient_id": patient_id,
                },
            )
            self._trace_tool_returned("evaluation", call_id, result, result)
            return result
        except Exception as e:
            self._trace_tool_failed("evaluation", call_id, e, retryable=True)
            logger.error(f"[Action] evaluation 失败: {e}")
            raise

    async def batch_evaluation(self, test_dir: str) -> Dict[str, Any]:
        """批量评估。

        Args:
            test_dir: 测试结果目录路径

        Returns:
            批量评估报告
        """
        logger.info(f"[Action] batch_evaluation: test_dir={test_dir}")
        final_results = self._load_final_results(test_dir)
        if not self.model_api_key:
            raise ValueError("MODEL_API_KEY is required for batch_evaluation")
        try:
            result = await self._request(
                "POST",
                self.batch_evaluation_path,
                json={
                    "team_id": self.team_id,
                    "api_key": self.model_api_key,
                    "final_result": final_results,
                },
                _audit_context={
                    "action": "batch_evaluation",
                    "patient_id": "",
                },
            )
            self._write_batch_evaluation_report(test_dir, result)
            return result
        except httpx.HTTPStatusError as e:
            if e.response is None or e.response.status_code != 404:
                logger.error(f"[Action] batch_evaluation failed: {e}")
                raise
            logger.warning(
                "[Action] batch_evaluation endpoint %s not found; falling back to per-case evaluation",
                self.batch_evaluation_path,
            )
            return await self._batch_evaluation_via_case_evaluation(test_dir)
        except Exception as e:
            logger.error(f"[Action] batch_evaluation failed: {e}")
            raise

    def _resolve_final_results_file(self, test_dir: str) -> str:
        path = os.path.abspath(str(test_dir))
        if os.path.isdir(path):
            path = os.path.join(path, "final_results.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"final_results.jsonl not found: {path}")
        return path

    @staticmethod
    def _extract_final_result_entries(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        if isinstance(row.get("final_results"), list):
            entries.extend(item for item in row["final_results"] if isinstance(item, dict))
        elif isinstance(row.get("final_result"), dict):
            entries.append(row["final_result"])
        elif row.get("diagnosis") is not None or row.get("treatment_plan") is not None:
            entries.append(row)

        normalized: List[Dict[str, Any]] = []
        row_patient_id = (
            row.get("patient_id")
            or row.get("patientId")
            or row.get("caseId")
            or row.get("case_id")
        )
        for entry in entries:
            final_result = dict(entry)
            patient_id = (
                final_result.get("patient_id")
                or final_result.get("patientId")
                or final_result.get("caseId")
                or final_result.get("case_id")
                or row_patient_id
            )
            if patient_id:
                final_result["patient_id"] = str(patient_id)
                final_result.setdefault("caseId", str(patient_id))
            normalized.append(final_result)
        return normalized

    def _load_final_results(self, test_dir: str) -> List[Dict[str, Any]]:
        results_file = self._resolve_final_results_file(test_dir)
        final_results: List[Dict[str, Any]] = []
        with open(results_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
                if isinstance(row, dict):
                    final_results.extend(self._extract_final_result_entries(row))
        return final_results

    def _write_batch_evaluation_report(
        self, test_dir: str, report: Dict[str, Any]
    ) -> str:
        results_file = self._resolve_final_results_file(test_dir)
        report_path = os.path.join(os.path.dirname(results_file), "final_results_eval_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("[Action] batch_evaluation report saved: %s", report_path)
        return report_path

    @staticmethod
    def _avg_case_metric(reports: List[Dict[str, Any]], key: str) -> float:
        values = []
        for report in reports:
            value = report.get(key)
            try:
                if value is not None:
                    values.append(float(value))
            except (TypeError, ValueError):
                continue
        return round(sum(values) / len(values), 4) if values else 0.0

    async def _batch_evaluation_via_case_evaluation(self, test_dir: str) -> Dict[str, Any]:
        final_results = self._load_final_results(test_dir)
        case_reports: List[Dict[str, Any]] = []
        for final_result in final_results:
            patient_id = final_result.get("patient_id") or final_result.get("caseId")
            if not patient_id:
                case_reports.append({
                    "status": "failed",
                    "error": "missing patient_id",
                    "final_result": final_result,
                })
                continue
            try:
                case_reports.append(await self.evaluation(str(patient_id), final_result))
            except Exception as exc:
                case_reports.append({
                    "patientId": str(patient_id),
                    "status": "failed",
                    "error": str(exc),
                })

        evaluated = [r for r in case_reports if r.get("status") == "evaluated"]
        treatment_details = []
        for report in evaluated:
            detail = report.get("treatmentDetail") or {}
            treatment_details.append({
                "patient_id": report.get("patientId") or report.get("patient_id"),
                "overall_score": detail.get("overallScore", report.get("treatmentOverallScore")),
                "safety": detail.get("safety", report.get("treatmentSafety")),
                "effectiveness_alignment": detail.get(
                    "effectivenessAlignment",
                    report.get("treatmentEffectivenessAlignment"),
                ),
                "personalization": detail.get(
                    "personalization",
                    report.get("treatmentPersonalization"),
                ),
                "reasoning": detail.get("reasoning", ""),
            })

        report = {
            "diagnosis_accuracy": self._avg_case_metric(evaluated, "diagnosisAccuracy"),
            "examination_precision": self._avg_case_metric(evaluated, "examinationPrecision"),
            "treatment_overall_score": self._avg_case_metric(evaluated, "treatmentOverallScore"),
            "treatment_safety": self._avg_case_metric(evaluated, "treatmentSafety"),
            "treatment_effectiveness_alignment": self._avg_case_metric(
                evaluated, "treatmentEffectivenessAlignment"
            ),
            "treatment_personalization": self._avg_case_metric(
                evaluated, "treatmentPersonalization"
            ),
            "counts": {
                "final_results": len(final_results),
                "evaluated_patients": len(evaluated),
                "failed_patients": len(case_reports) - len(evaluated),
            },
            "treatment_details": treatment_details,
            "case_reports": case_reports,
            "submitted_at": datetime.now().astimezone().isoformat(),
            "fallback": "per_case_evaluation",
        }

        self._write_batch_evaluation_report(test_dir, report)
        return report

    async def close(self) -> None:
        """关闭 HTTP 客户端。"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class BaseDoctorAgent(ABC):
    """医生 Agent 基类。

    参赛者需要继承此类并实现 train 和 test 方法。
    基类提供：
    - self.actions: Actions 实例，用于调用比赛能力
    - self.config: 配置字典
    - run_train(): 训练入口
    - run_test(): 测试入口
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化 Agent。

        Args:
            config: 配置字典
        """
        self.config = config

        # 从请求级配置 / 环境变量 / 配置文件获取服务配置，避免修改进程全局环境。
        runtime_service = config.get("_runtime_service", {}) or {}
        service_config = config.get("service", {}) or {}
        base_url = (
            runtime_service.get("base_url")
            or os.environ.get("SERVICE_BASE_URL", "")
            or service_config.get("base_url", "")
        )
        token = (
            runtime_service.get("token")
            or os.environ.get("SERVICE_TRAIN_TOKEN", "")
            or service_config.get("token", "")
        )
        team_id = (
            runtime_service.get("team_id")
            or os.environ.get("TEAM_ID", "")
            or service_config.get("team_id", "")
        )
        endpoint_prefixes = (
            runtime_service.get("endpoint_prefixes")
            or service_config.get("endpoint_prefixes")
            or []
        )
        if isinstance(endpoint_prefixes, str):
            endpoint_prefixes = [endpoint_prefixes]
        llm_config = config.get("llm", {}) or {}
        model_api_key = (
            runtime_service.get("model_api_key")
            or runtime_service.get("api_key")
            or os.environ.get("MODEL_API_KEY", "")
            or llm_config.get("api_key", "")
            or service_config.get("model_api_key", "")
        )
        gateway_token = (
            runtime_service.get("gateway_token")
            or os.environ.get("MODELSCOPE_SDK_TOKEN", "")
            or service_config.get("gateway_token", "")
        )
        use_invoke = runtime_service.get("use_invoke", service_config.get("use_invoke", True))
        invoke_path = runtime_service.get("invoke_path", service_config.get("invoke_path", "/invoke"))
        exam_results_path = runtime_service.get(
            "exam_results_path", service_config.get("exam_results_path", "/exam/results")
        )
        case_evaluation_path = runtime_service.get(
            "case_evaluation_path", service_config.get("case_evaluation_path", "/evaluate/case")
        )
        batch_evaluation_path = runtime_service.get(
            "batch_evaluation_path", service_config.get("batch_evaluation_path", "/evaluate")
        )

        # 创建 Actions 实例
        self.actions = Actions(
            base_url=base_url,
            token=token,
            team_id=team_id,
            endpoint_prefixes=endpoint_prefixes,
            model_api_key=model_api_key,
            gateway_token=gateway_token,
            use_invoke=use_invoke,
            invoke_path=invoke_path,
            exam_results_path=exam_results_path,
            case_evaluation_path=case_evaluation_path,
            batch_evaluation_path=batch_evaluation_path,
        )

        # 输出目录
        self.output_dir = config.get("output_dir", "outputs")

    def _collect_runtime_audit(self) -> Dict[str, Any]:
        """Collect lightweight runtime audit even when a case fails early."""
        tool_records = []
        tool_summary: Dict[str, Any] = {}
        actions = getattr(self, "actions", None)
        if actions is not None:
            snapshot = getattr(actions, "snapshot_tool_audit", None)
            summary = getattr(actions, "tool_contract_summary", None)
            if callable(snapshot):
                tool_records = snapshot()
            if callable(summary):
                tool_summary = summary()
        llm_records = list(getattr(self, "_llm_call_audit", []) or [])
        llm_summary: Dict[str, Any] = {}
        llm_summary_builder = getattr(self, "_llm_contract_summary_from_audit", None)
        if callable(llm_summary_builder):
            llm_summary = llm_summary_builder(llm_records)
        return {
            "llm_call_audit": llm_records,
            "llm_contract_summary": llm_summary,
            "llm_context_audit": list(getattr(self, "_llm_context_audit", []) or []),
            "tool_call_audit": tool_records,
            "tool_contract_summary": tool_summary,
        }

    @abstractmethod
    async def train(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """训练流程：对单个患者进行诊疗。

        参赛者必须实现此方法。

        Args:
            patient_id: 患者 ID
        """
        ...

    @abstractmethod
    async def test(self, patient_id: str) -> None:
        """测试流程：对单个患者进行诊疗并提交结果。

        参赛者必须实现此方法。

        Args:
            patient_id: 患者 ID
        """
        ...

    async def _get_patient_list(
        self, mode: str = "train"
    ) -> List[str]:
        """从服务端获取患者列表。

        Args:
            mode: "train" 或 "test"

        Returns:
            患者 ID 列表
        """
        try:
            result = await self.actions._request(
                "GET",
                "/patients",
                params={"mode": mode, "team_id": self.actions.team_id},
            )
            patients = result.get("patients", result.get("data", result.get("patient_ids", [])))
            return [
                pid for pid in (self._normalize_patient_id(p) for p in patients) if pid
            ]
        except Exception as e:
            logger.warning(f"获取患者列表失败: {e}，使用配置文件中的列表")
            return []

    @staticmethod
    def _normalize_patient_id(value: Any) -> str:
        """Extract a stable patient id from service/config values."""
        if value is None:
            return ""
        if isinstance(value, dict):
            for key in ("patient_id", "patientId", "case_id", "caseId", "id"):
                item = value.get(key)
                if item:
                    return str(item)
            for key in ("patient", "case", "data"):
                nested = value.get(key)
                if isinstance(nested, dict):
                    found = BaseDoctorAgent._normalize_patient_id(nested)
                    if found:
                        return found
            return ""
        return str(value)

    def _get_patient_ids_from_config(self, mode: str = "train") -> List[str]:
        """从配置文件获取患者 ID 列表。

        Args:
            mode: "train" 或 "test"

        Returns:
            患者 ID 列表
        """
        mode_config = self.config.get(mode, {})
        patient_ids = mode_config.get("patient_ids", [])

        if patient_ids:
            return [
                pid for pid in (self._normalize_patient_id(item) for item in patient_ids) if pid
            ]

        # 如果没有指定患者 ID，返回空列表
        # 实际运行时会从服务端获取
        return []

    def _select_patient_ids(self, patient_ids: List[str], mode: str = "train") -> List[str]:
        """按 config.yaml 的 selection/patient_count/random_seed 选择患者。"""
        if not patient_ids:
            return []

        mode_config = self.config.get(mode, {}) or {}
        explicit_ids = mode_config.get("patient_ids", []) or []
        if explicit_ids:
            return [
                pid for pid in (self._normalize_patient_id(item) for item in explicit_ids) if pid
            ]

        selection = str(mode_config.get("selection", "forward") or "forward").lower()
        patient_count = mode_config.get("patient_count")
        try:
            limit = int(patient_count) if patient_count not in (None, "") else len(patient_ids)
        except (TypeError, ValueError):
            limit = len(patient_ids)
        limit = max(0, min(limit, len(patient_ids)))

        selected = [
            pid for pid in (self._normalize_patient_id(item) for item in patient_ids) if pid
        ]
        if selection == "random":
            seed = mode_config.get("random_seed", 42)
            rng = random.Random(seed)
            selected = selected[:]
            rng.shuffle(selected)
        elif selection == "reverse":
            selected = list(reversed(selected))
        elif selection != "forward":
            logger.warning("[%s] 未知 selection=%r，按 forward 处理", mode, selection)

        return selected[:limit] if limit else []

    async def _cleanup(self) -> None:
        """清理资源，关闭所有 HTTP 客户端。幂等方法，可安全多次调用。"""
        try:
            await self.actions.close()
        except Exception as e:
            logger.warning(f"[Cleanup] 关闭 actions 客户端失败: {e}")
        # 如果子类有 llm 客户端，也关闭它
        if hasattr(self, 'llm') and hasattr(self.llm, 'close'):
            try:
                await self.llm.close()
            except Exception as e:
                logger.warning(f"[Cleanup] 关闭 llm 客户端失败: {e}")

    async def run_train(self) -> Dict[str, Any]:
        """训练入口：获取患者列表并逐个训练。

        此方法由 train.py 调用。
        """
        logger.info("[run_train] 开始训练流程")

        # 获取患者列表
        patient_ids = self._get_patient_ids_from_config("train")

        if not patient_ids:
            # 尝试从服务端获取
            patient_ids = await self._get_patient_list("train")
        patient_ids = self._select_patient_ids(patient_ids, "train")

        if not patient_ids:
            # 使用配置中的数量
            train_config = self.config.get("train", {})
            patient_count = train_config.get("patient_count", 10)
            logger.warning(
                f"[run_train] 无法获取患者列表，"
                f"请确保 SERVICE_BASE_URL 正确或配置 patient_ids。"
                f"配置 patient_count={patient_count}"
            )
            return {
                "run_dir": "",
                "results_file": "",
                "summary_file": "",
                "results": [],
                "summary": summarize_training_results([]),
            }

        logger.info(f"[run_train] 训练患者数: {len(patient_ids)}")

        # 创建输出目录
        timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = os.path.join(self.output_dir, "train", timestamp)
        os.makedirs(run_dir, exist_ok=True)

        # 逐个训练
        success_count = 0
        fail_count = 0
        results: List[Dict[str, Any]] = []
        for i, patient_id in enumerate(patient_ids):
            logger.info(f"[run_train] 训练进度: {i + 1}/{len(patient_ids)}, patient_id={patient_id}")
            try:
                if hasattr(self.actions, "begin_case"):
                    self.actions.begin_case(patient_id)
                case_result = await self.train(patient_id)
                if not isinstance(case_result, dict):
                    case_result = {
                        "patient_id": patient_id,
                        "status": "completed",
                        "metrics": {},
                        "audit": self._collect_runtime_audit(),
                    }
                results.append(case_result)
                success_count += 1
            except Exception as e:
                logger.error(f"[run_train] 训练患者 {patient_id} 失败: {e}")
                results.append(
                    {
                        "patient_id": patient_id,
                        "status": "failed",
                        "error": str(e),
                        "metrics": {},
                        "audit": self._collect_runtime_audit(),
                    }
                )
                fail_count += 1

        logger.info(
            f"[run_train] 训练完成: 成功={success_count}, 失败={fail_count}"
        )

        results_file = os.path.join(run_dir, "training_results.jsonl")
        with open(results_file, "w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        summary = summarize_training_results(results)
        summary.update(
            {
                "selection": (self.config.get("train", {}) or {}).get("selection"),
                "random_seed": (self.config.get("train", {}) or {}).get("random_seed"),
                "requested_patient_count": (self.config.get("train", {}) or {}).get(
                    "patient_count"
                ),
                "success_count": success_count,
                "fail_count": fail_count,
            }
        )
        summary_file = os.path.join(run_dir, "training_summary.json")
        with open(summary_file, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        logger.info("[run_train] 训练汇总保存到: %s", summary_file)

        # 清理资源
        await self._cleanup()
        return {
            "run_dir": run_dir,
            "results_file": results_file,
            "summary_file": summary_file,
            "results": results,
            "summary": summary,
        }

    async def run_test(self) -> Dict[str, Any]:
        """测试入口：获取患者列表并逐个测试。

        此方法由 test.py 调用。
        """
        logger.info("[run_test] 开始测试流程")

        # 获取患者列表
        patient_ids = self._get_patient_ids_from_config("test")

        if not patient_ids:
            # 尝试从服务端获取
            patient_ids = await self._get_patient_list("test")
        patient_ids = self._select_patient_ids(patient_ids, "test")

        if not patient_ids:
            logger.warning("[run_test] 无法获取患者列表，请确保配置正确")
            return {
                "test_dir": "",
                "results_file": "",
                "results": [],
                "success_count": 0,
                "fail_count": 0,
            }

        logger.info(f"[run_test] 测试患者数: {len(patient_ids)}")

        # 创建输出目录
        timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        test_dir = os.path.join(self.output_dir, "test", timestamp)
        os.makedirs(test_dir, exist_ok=True)

        # 逐个测试
        results = []
        success_count = 0
        fail_count = 0
        for i, patient_id in enumerate(patient_ids):
            logger.info(f"[run_test] 测试进度: {i + 1}/{len(patient_ids)}, patient_id={patient_id}")
            try:
                await self.test(patient_id)
                success_count += 1
                # 收集测试结果（子类可在 test 中设置 self._last_test_result）
                result_entry = {"patient_id": patient_id, "status": "success"}
                if hasattr(self, '_last_test_result') and self._last_test_result:
                    result_entry.update(self._last_test_result)
                results.append(result_entry)
            except Exception as e:
                logger.error(f"[run_test] 测试患者 {patient_id} 失败: {e}")
                fail_count += 1
                # 失败也要给评测器一个可解析的 final_result 兜底，避免 "响应中未找到 final_result" 类错误
                _empty_final = {
                    "patient_id": patient_id,
                    "diagnosis": ["待明确诊断"],
                    "treatment_plan": "当前信息不足，建议进一步问诊并完善必要检查后制定治疗方案。",
                    "reasoning": f"agent failed: {e}",
                    "conversation_rounds": 0,
                    "ordered_examinations": [],
                    "finished": True,
                }
                results.append({
                    "patient_id": patient_id,
                    "status": "failed",
                    "error": str(e),
                    "diagnosis": _empty_final["diagnosis"],
                    "treatment_plan": _empty_final["treatment_plan"],
                    "reasoning": _empty_final["reasoning"],
                    "conversation_rounds": 0,
                    "ordered_examinations": [],
                    "finished": True,
                    "final_result": _empty_final,
                    "final_results": [_empty_final],
                })

        # 保存测试结果
        results_file = os.path.join(test_dir, "final_results.jsonl")
        tmp_results_file = os.path.join(
            test_dir,
            f".final_results.{os.getpid()}.{uuid.uuid4().hex}.tmp",
        )
        with open(tmp_results_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.replace(tmp_results_file, results_file)
        except OSError:
            # 某些受限 Windows 沙盒禁止重命名替换，退化为直接写入以保证功能可用。
            with open(results_file, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info(
            f"[run_test] 测试完成: 成功={success_count}, 失败={fail_count}"
        )
        logger.info(f"[run_test] 结果保存到: {results_file}")

        # 清理资源
        await self._cleanup()
        return {
            "test_dir": test_dir,
            "results_file": results_file,
            "results": results,
            "success_count": success_count,
            "fail_count": fail_count,
        }
