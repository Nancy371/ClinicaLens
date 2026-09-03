"""Stable trace enums for diagnostic trajectory capture."""

from enum import Enum


class TraceEventType(str, Enum):
    TRACE_STARTED = "trace.started"
    TRACE_COMPLETED = "trace.completed"
    TRACE_FAILED = "trace.failed"
    MODULE_STARTED = "module.started"
    MODULE_COMPLETED = "module.completed"
    MODULE_FAILED = "module.failed"
    ARTIFACT_CREATED = "artifact.created"
    DECISION_MADE = "decision.made"
    STATE_CHANGED = "state.changed"
    TOOL_CALLED = "tool.called"
    TOOL_RETURNED = "tool.returned"
    TOOL_FAILED = "tool.failed"
    SUBMISSION_CREATED = "submission.created"
    VALIDATION_FAILED = "validation.failed"


class TraceStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    WARNING = "warning"
    INCOMPLETE = "incomplete"


class ArtifactType(str, Enum):
    RAW_CASE = "raw_case"
    MODULE_INPUT = "module_input"
    MODULE_OUTPUT = "module_output"
    SUBMISSION_RESULT = "submission_result"
    EVIDENCE_SET = "evidence_set"
    CANDIDATE_SET = "candidate_set"
    CANDIDATE_SCORE_SET = "candidate_score_set"
    DIAGNOSIS_DECISION = "diagnosis_decision"
    EVIDENCE_GAP = "evidence_gap"
    GAP_RANKING = "gap_ranking"
    EXAM_PLAN = "exam_plan"
    EXAM_RESULT_INTENT_BINDING = "exam_result_intent_binding"
    TOOL_RESULT_RAW = "tool_result_raw"
    TOOL_RESULT_NORMALIZED = "tool_result_normalized"
    EVIDENCE_UPDATE = "evidence_update"
    RESOLVER_RESULT = "resolver_result"
    PATTERN_HYPOTHESIS = "pattern_hypothesis"
    PATTERN_HYPOTHESIS_VERIFICATION = "pattern_hypothesis_verification"
    PATTERN_ENTITY_LINK = "pattern_entity_link"
    PATTERN_RECALL_SIGNAL = "pattern_recall_signal"
    PATTERN_RECALL_AUDIT = "pattern_recall_audit"
    PATTERN_CANDIDATE_ADMISSION = "pattern_candidate_admission"


EVENT_TYPE_VALUES = {item.value for item in TraceEventType}
ARTIFACT_TYPE_VALUES = {item.value for item in ArtifactType}
STATUS_VALUES = {item.value for item in TraceStatus}
