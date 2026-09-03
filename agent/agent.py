"""
医生 Agent 主逻辑模块。

基于 LLM 驱动的智能诊疗流程，采用规划-执行-反思架构：
- 规划(Planner)：全局诊疗策略制定 + 阶段转换 + 动态重规划
- 执行(Executor)：问诊/检查/诊断/治疗等具体操作
- 反思(Reflection)：自我批判 + 策略调整 + 经验积累
- 记忆(Memory)：历史经验检索注入 + 低分病例重点学习

Planner 作为中枢，通过 Reflection/Criticism 机制将宏观目标分解为具体步骤，
LLM 进行深度推理后决策下一步操作（tool call）。
"""

import asyncio
import json
import logging
import os
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from hospital_agent import BaseDoctorAgent
from .llm import LLMClient
from .prompt import DoctorPrompt
from .memory import DoctorMemory
from .knowledge import KnowledgeBase
from .critic import DefectDetector
from .memory_system import DoctorAgentMemory
from .policy_store import PolicyStore
from .exam_strategy import ExamStrategyAgent
from .inquiry_strategy import InquiryStrategyAgent
from .qc import QualityAgent
from .treatment_strategy import TreatmentStrategyAgent
from .structural_diagnosis import StructuralDiagnosisAgent
from .evidence_engine import EvidenceDiagnosisEngine
from .clinical_evidence import (
    ClinicalEvidenceNormalizer,
    EvidenceAgent,
    EvidenceBundle,
    HybridEvidenceCompiler,
    Observation,
)
from .diagnosis_eligibility import DEFERRED, DIFFERENTIAL_ONLY, EXCLUDED, PRIMARY_ELIGIBLE
from .diagnosis_engine import DiagnosisDecisionEngine
from .diagnosis_critic import DiagnosisCritic
from .evidence_pattern_compiler import EvidencePatternCompiler
from .diagnostic_learning import DiagnosticLearningStore
from .candidate_policy_store import CandidatePolicyStore, RuleGeneralizer
from .treatment_safety import TreatmentSafetyGate
from .targeted_exam_result_parser import (
    ExamResultIntentBinding,
    TargetedExamParseResult,
    TargetedExamResultParser,
    binding_from_authorization_detail,
)
from .claim_resolution import (
    ClaimResolutionUpdater,
    claim_requirements_from_contract,
    materialize_candidate_claim_states,
    normalize_ledger,
)
from .context_compiler import StageContextCompiler
from .llm_contract import LLMContractExecutor
from .pattern_hypothesis import ThinkingSnapshot, evidence_snapshot_hash as pattern_evidence_snapshot_hash
from .trace import ArtifactType, TraceCollector

logger = logging.getLogger(__name__)


def _overall_score(report: Dict[str, Any]) -> Optional[float]:
    """从平台评估报告中提取综合分。取三项均值作为综合指标。"""
    if not report:
        return None
    keys = ("diagnosisAccuracy", "examinationPrecision", "treatmentOverallScore")
    vals = []
    for k in keys:
        v = report.get(k)
        try:
            if v is not None:
                vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def _compact_exam_name(value: Any) -> str:
    return re.sub(r"[\s_\-（）()［\]\[\]、，,。：:；;]+", "", str(value or "").lower())


# ============ 诊疗阶段枚举 ============

class Phase(str, Enum):
    """诊疗阶段状态机。"""
    INITIAL = "initial"        # 初始接诊
    INQUIRY = "inquiry"        # 问诊阶段
    EXAMINATION = "examination"  # 检查阶段
    DIAGNOSIS = "diagnosis"    # 诊断阶段
    TREATMENT = "treatment"    # 治疗阶段（提交方案）
    COMPLETED = "completed"    # 诊疗完成


class ActionType(str, Enum):
    """可执行的操作类型，对应 tool call。"""
    ASK_PATIENT = "ask_patient"           # 询问患者
    ORDER_EXAMINATION = "order_examination"  # 申请检查
    PRESCRIBE_TREATMENT = "prescribe_treatment"  # 提交诊疗方案
    REPLAN = "replan"                     # 重新规划


# ============ 规划器 ============

class Planner:
    """诊疗规划器 —— 中枢模块。

    职责：
    1. 制定全局诊疗策略（_plan）
    2. 决策下一步具体操作（decide_next_action）
    3. 阶段转换判断与执行
    4. 通过 Reflection/Criticism 机制自我修正策略
    5. 与 Memory 交互整合历史信息

    架构：
    Planner → LLM(深度推理) → Action(tool call)
         ↑                        |
         └── Reflection/Criticism ─┘
    """

    def __init__(
        self,
        prompt: DoctorPrompt,
        llm_chat_json,
        llm_chat,
        memory: DoctorMemory,
        mark_llm_consumer_result=None,
        compile_llm_context=None,
    ):
        """初始化规划器。

        Args:
            prompt: Prompt 模板管理器
            llm_chat_json: 异步 LLM JSON 调用函数
            llm_chat: 异步 LLM 文本调用函数
            memory: 记忆管理器
        """
        self.prompt = prompt
        self._llm_chat_json = llm_chat_json
        self._llm_chat = llm_chat
        self.memory = memory
        self._mark_llm_consumer_result = mark_llm_consumer_result
        self._compile_llm_context = compile_llm_context

        # 规划状态
        self.current_phase = Phase.INITIAL
        self.current_plan: Optional[Dict[str, Any]] = None
        self.action_history: List[Dict[str, Any]] = []
        self.criticism_result: Optional[Dict[str, Any]] = None

        # 阶段计数器
        self.inquiry_rounds = 0
        self.exam_rounds = 0

        # 配置
        self.max_inquiry_rounds = 5
        self.max_exam_rounds = 3
        self.max_total_actions = 15  # 安全上限，防止无限循环

        # 批判频率控制（成本优化）：仅前 N 次 plan 触发 criticism，后续 plan 跳过。
        # 因为初始规划质量最关键，后续多为微调，无需每次都走 2 次 LLM。
        self._plan_call_count = 0
        self.criticism_max_calls = 2

        # 跨患者软复用的教训种子（由 soft_reset 填充）
        self._carry_lessons: Optional[str] = None

        # 策略补丁库（由外部注入；None 时不启用）
        self.policy_store = None  # type: ignore[assignment]
        # 当前 plan 命中的补丁 ID 列表（供反思阶段 record_outcome 使用）
        self._last_used_patch_ids: List[str] = []

    def _mark_last_llm_consumer_result(
        self,
        purpose: str,
        accepted: bool,
        *,
        fallback_used: bool = False,
        fallback_trigger: str = "",
    ) -> None:
        if self._mark_llm_consumer_result is None:
            return
        self._mark_llm_consumer_result(
            purpose,
            accepted,
            fallback_used=fallback_used,
            fallback_trigger=fallback_trigger,
        )

    def _compile_context(self, stage: str, **state: Any) -> Dict[str, Any]:
        if self._compile_llm_context is None:
            return dict(state)
        return self._compile_llm_context(stage, **state)

    def _record_action(self, action_type: str, target: str, result_summary: str) -> None:
        """记录已执行的操作到历史。"""
        self.action_history.append({
            "type": action_type,
            "target": target,
            "result_summary": result_summary[:200],  # 截断防过长
            "phase": self.current_phase.value,
        })

    def soft_reset(self, keep_lessons: bool = True) -> None:
        """跨患者软复用：重置流程状态但保留可迁移的教训。

        清理项（下一位患者必须干净开始）:
            - current_phase / current_plan / action_history / criticism_result
            - inquiry_rounds / exam_rounds / _plan_call_count
        保留项（跨患者可复用的知识）:
            - self.memory 由外部单例持有，天然保留
            - self.max_* 配置
            - 若 keep_lessons=True 则保留最近一次 plan 的 lessons_learned 摘要
        """
        # 抽取上一位患者留下的教训（若有）
        lessons_carry = None
        if keep_lessons and self.current_plan:
            lessons_carry = self.current_plan.get("lessons_learned") or \
                            self.current_plan.get("strategy", {}).get("lessons_learned")

        # 重置流程状态
        self.current_phase = Phase.INITIAL
        self.current_plan = None
        self.action_history = []
        self.criticism_result = None
        self.inquiry_rounds = 0
        self.exam_rounds = 0
        self._plan_call_count = 0

        # 将教训作为轻量种子挂在 planner 上，下一次 plan() 时可注入 prompt
        self._carry_lessons = lessons_carry if lessons_carry else None
        logger.info(
            f"[规划] 软复用完成，教训种子={'有' if self._carry_lessons else '无'}"
        )

    async def plan(
        self,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        relevant_experience: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """制定或更新全局诊疗策略。

        这是规划器的核心方法，通过 LLM 深度推理生成策略规划，
        然后通过 Reflection/Criticism 机制进行自我审查和修正。

        Args:
            collected_info: 已收集的患者信息
            exam_results: 已有检查结果
            chat_history: 对话历史
            relevant_experience: 相关历史经验

        Returns:
            规划结果字典
        """
        # 1. 调用 LLM 生成策略规划
        planning_context = self._compile_context(
            "planning",
            collected_info=collected_info,
            exam_results=exam_results,
            chat_history=chat_history,
            relevant_experience=relevant_experience,
            previous_plan=self.current_plan,
        )
        planning_prompt = self.prompt.build_planning_prompt(
            collected_info=planning_context.get("collected_info", collected_info),
            exam_results=planning_context.get("exam_results", exam_results),
            chat_history=planning_context.get("chat_history", chat_history),
            phase=self.current_phase.value,
            relevant_experience=planning_context.get(
                "relevant_experience", relevant_experience
            ),
            previous_plan=planning_context.get("previous_plan", self.current_plan),
        )

        # 1.5 自迭代增强：拼接 (a) 教训种子 lessons  (b) 命中的策略补丁
        self._last_used_patch_ids = []
        _augment_parts: List[str] = []
        if self._carry_lessons:
            _augment_parts.append("【历史教训（跨患者复用）】\n" + str(self._carry_lessons))
        if self.policy_store is not None:
            try:
                _cands_for_match = []
                _dd = (self.current_plan or {}).get("differential_diagnoses") or []
                for _it in _dd:
                    if isinstance(_it, dict) and _it.get("disease"):
                        _cands_for_match.append(_it["disease"])
                    elif isinstance(_it, str):
                        _cands_for_match.append(_it)
                _ph = (self.current_plan or {}).get("primary_hypothesis")
                if _ph:
                    _cands_for_match.append(_ph)
                _hits = self.policy_store.match(
                    collected_info=collected_info,
                    candidate_diseases=_cands_for_match,
                    include_shadow=False,
                )
                if _hits:
                    _augment_parts.append(self.policy_store.render_for_prompt(_hits))
                    self._last_used_patch_ids = [p.get("id") for p in _hits if p.get("id")]
                    logger.info(
                        f"[规划] 命中策略补丁 {len(_hits)} 项: "
                        f"{[p.get('type') for p in _hits]}"
                    )
            except Exception as _e:
                logger.debug(f"[规划] 补丁匹配失败: {_e}")
        if _augment_parts:
            planning_prompt = "\n\n".join(_augment_parts) + "\n\n" + planning_prompt

        messages = [
            {"role": "system", "content": planning_prompt},
            {"role": "user", "content": "请制定诊疗策略规划。"},
        ]

        plan_result = await self._llm_chat_json(
            messages,
            temperature=0.3,
            purpose="planning",
        )

        if not plan_result or "strategy" not in plan_result:
            self._mark_last_llm_consumer_result(
                "planning",
                False,
                fallback_used=True,
                fallback_trigger="schema_missing_fields",
            )
            logger.warning("[规划] LLM 规划失败，使用回退策略")
            plan_result = self._fallback_plan(collected_info, exam_results)
        else:
            self._mark_last_llm_consumer_result("planning", True)

        # 2. Reflection/Criticism：自我批判审查（限流：仅前 N 次 plan 触发，节省成本）
        self._plan_call_count += 1
        criticism = None
        if self._plan_call_count <= self.criticism_max_calls:
            criticism = await self._reflect_and_criticize(
                plan_result, collected_info, exam_results
            )
        else:
            logger.info(
                f"[规划] 已达批判限流阈值({self.criticism_max_calls}次)，本次跳过 criticism 以节省 LLM 调用"
            )

        # 3. 根据批判结果修正计划
        if criticism and criticism.get("overall_assessment") == "needs_replan":
            logger.info("[规划] 批判建议重新规划，执行重规划")
            # 融合批判建议后重新规划
            revised_plan = self._apply_criticism_to_plan(plan_result, criticism)
            plan_result = revised_plan
        elif criticism and criticism.get("overall_assessment") == "needs_adjustment":
            logger.info("[规划] 批判建议微调，应用修正")
            plan_result = self._apply_criticism_to_plan(plan_result, criticism)

        # 4. 更新状态
        self.current_plan = plan_result
        self.criticism_result = criticism

        # 5. 同步阶段状态
        strategy = plan_result.get("strategy", {})
        plan_phase = strategy.get("current_phase", "")
        if plan_phase and plan_phase in [p.value for p in Phase]:
            self.current_phase = Phase(plan_phase)

        # 日志
        hypothesis = plan_result.get("primary_hypothesis", "未知")
        confidence = plan_result.get("hypothesis_confidence", 0)
        priority_actions = strategy.get("priority_actions", [])
        logger.info(f"[规划] 主要假设: {hypothesis} (置信度: {confidence})")
        logger.info(f"[规划] 当前阶段: {self.current_phase.value}")
        if priority_actions:
            first_action = priority_actions[0]
            logger.info(f"[规划] 首要行动: {first_action.get('action')} - {first_action.get('target')}")

        return plan_result

    async def _reflect_and_criticize(
        self,
        current_plan: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """对当前计划进行反思批判。

        通过 LLM 扮演"内在批判者"角色，从假设偏差、信息盲区、
        行动优先级、风险遗漏、阶段转换、认知偏差等维度审查计划。

        Args:
            current_plan: 当前诊疗计划
            collected_info: 已收集的患者信息
            exam_results: 已有检查结果

        Returns:
            批判结果字典，包含 criticisms、plan_revision、overall_assessment
        """
        # 获取最近的思考结果
        thinking_result = None
        if self.current_plan and "differential_diagnoses" in self.current_plan:
            thinking_result = {
                "differential_diagnosis": self.current_plan.get("differential_diagnoses", []),
                "key_unknowns": self.current_plan.get("strategy", {}).get("info_gaps", []),
            }

        criticism_context = self._compile_context(
            "planning_criticism",
            current_plan=current_plan,
            collected_info=collected_info,
            exam_results=exam_results,
            thinking_result=thinking_result,
            action_history=self.action_history,
        )
        criticism_prompt = self.prompt.build_reflection_criticism_prompt(
            current_plan=criticism_context.get("current_plan", current_plan),
            collected_info=criticism_context.get("collected_info", collected_info),
            exam_results=criticism_context.get("exam_results", exam_results),
            thinking_result=criticism_context.get("thinking_result", thinking_result),
            action_history=criticism_context.get("action_history", self.action_history),
        )
        messages = [
            {"role": "system", "content": criticism_prompt},
            {"role": "user", "content": "请对当前诊疗策略进行批判性审查。"},
        ]

        result = await self._llm_chat_json(
            messages,
            temperature=0.4,
            purpose="planning_criticism",
        )

        if result and "criticisms" in result:
            self._mark_last_llm_consumer_result("planning_criticism", True)
            high_severity = [c for c in result.get("criticisms", []) if c.get("severity") == "high"]
            assessment = result.get("overall_assessment", "on_track")
            confidence = result.get("confidence_in_plan", 0.5)
            logger.info(f"[批判] 评估: {assessment}, 计划置信度: {confidence}")
            if high_severity:
                for c in high_severity:
                    logger.warning(f"[批判] 高严重度问题: {c.get('issue')} → {c.get('suggestion')}")
            return result

        logger.info("[批判] 批判未产出有效结果，继续当前计划")
        self._mark_last_llm_consumer_result(
            "planning_criticism",
            False,
            fallback_used=True,
            fallback_trigger="consumer_rejected",
        )
        return None

    def _apply_criticism_to_plan(
        self, plan: Dict[str, Any], criticism: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将批判建议应用到计划中。

        Args:
            plan: 当前计划
            criticism: 批判结果

        Returns:
            修正后的计划
        """
        revised = plan.copy()
        revision = criticism.get("plan_revision", {})

        # 修正主要假设
        if revision.get("hypothesis_revised"):
            revised["primary_hypothesis"] = revision["hypothesis_revised"]
            logger.info(f"[规划修正] 假设修订: {revision['hypothesis_revised']}")

        # 修正优先行动
        if revision.get("priority_actions_revised"):
            revised["strategy"] = revised.get("strategy", {})
            revised["strategy"]["priority_actions"] = revision["priority_actions_revised"]
            logger.info("[规划修正] 优先行动已调整")

        # 阶段转换
        phase_change = revision.get("phase_change")
        if phase_change and phase_change.get("to"):
            new_phase = phase_change["to"]
            if new_phase in [p.value for p in Phase]:
                self.current_phase = Phase(new_phase)
                revised["strategy"]["current_phase"] = new_phase
                logger.info(f"[规划修正] 阶段转换: {phase_change.get('from')} → {new_phase}, 原因: {phase_change.get('reason')}")

        # 新增风险
        new_risks = revision.get("new_risks", [])
        if new_risks:
            risk_assessment = revised.get("risk_assessment", {})
            existing_urgent = risk_assessment.get("urgent_findings", [])
            risk_assessment["urgent_findings"] = existing_urgent + new_risks
            revised["risk_assessment"] = risk_assessment

        # 被忽略的信息
        missing_info = revision.get("missing_info", [])
        if missing_info:
            strategy = revised.get("strategy", {})
            existing_gaps = strategy.get("info_gaps", [])
            strategy["info_gaps"] = existing_gaps + missing_info
            revised["strategy"] = strategy

        return revised

    def decide_next_action(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """根据规划决策下一步具体操作（tool call）。

        这是规划器到执行器的桥梁：将宏观策略转化为具体的 tool call。

        Args:
            plan: 当前规划结果

        Returns:
            行动决策字典，包含：
            - action: ActionType 枚举值
            - target: 行动目标（如具体问题、检查项目等）
            - reason: 行动理由
            - phase: 当前阶段
        """
        strategy = plan.get("strategy", {})
        priority_actions = strategy.get("priority_actions", [])

        # 安全检查：总行动数上限
        if len(self.action_history) >= self.max_total_actions:
            logger.warning("[规划] 达到行动上限，强制进入诊断阶段")
            return {
                "action": ActionType.PRESCRIBE_TREATMENT,
                "target": "基于已有信息提交诊疗方案",
                "reason": "达到最大行动次数，必须提交方案",
                "phase": self.current_phase.value,
            }

        # 从优先行动列表中提取第一个行动
        if priority_actions:
            first = priority_actions[0]
            action_str = first.get("action", "")
            target = first.get("target", "")
            reason = first.get("reason", "")

            # 映射到 ActionType
            action_map = {
                "ask": ActionType.ASK_PATIENT,
                "examine": ActionType.ORDER_EXAMINATION,
                "diagnose": ActionType.PRESCRIBE_TREATMENT,
                "treat": ActionType.PRESCRIBE_TREATMENT,
            }
            action = action_map.get(action_str, ActionType.ASK_PATIENT)

            # 阶段一致性校验
            action = self._validate_phase_consistency(action)

            return {
                "action": action,
                "target": target,
                "reason": reason,
                "phase": self.current_phase.value,
            }

        # 回退：根据当前阶段决定默认行动
        return self._default_action_for_phase()

    def _validate_phase_consistency(self, action: ActionType) -> ActionType:
        """校验行动与当前阶段的一致性，必要时调整阶段。

        Args:
            action: 计划的行动

        Returns:
            校验后的行动
        """
        # 问诊轮次上限检查
        if action == ActionType.ASK_PATIENT and self.inquiry_rounds >= self.max_inquiry_rounds:
            logger.info(f"[规划] 问诊已达上限({self.max_inquiry_rounds}轮)，转入检查阶段")
            self.current_phase = Phase.EXAMINATION
            return ActionType.ORDER_EXAMINATION

        # 检查轮次上限检查
        if action == ActionType.ORDER_EXAMINATION and self.exam_rounds >= self.max_exam_rounds:
            logger.info(f"[规划] 检查已达上限({self.max_exam_rounds}轮)，转入诊断阶段")
            self.current_phase = Phase.DIAGNOSIS
            return ActionType.PRESCRIBE_TREATMENT

        # 阶段转换：问诊→检查
        if self.current_phase == Phase.INQUIRY and action == ActionType.ORDER_EXAMINATION:
            self.current_phase = Phase.EXAMINATION

        # 阶段转换：检查→诊断/治疗
        if self.current_phase == Phase.EXAMINATION and action == ActionType.PRESCRIBE_TREATMENT:
            self.current_phase = Phase.DIAGNOSIS

        return action

    def _default_action_for_phase(self) -> Dict[str, Any]:
        """根据当前阶段返回默认行动。"""
        defaults = {
            Phase.INITIAL: (ActionType.ASK_PATIENT, "收集主诉和基本病史"),
            Phase.INQUIRY: (ActionType.ASK_PATIENT, "继续追问细节"),
            Phase.EXAMINATION: (ActionType.ORDER_EXAMINATION, "申请检查项目"),
            Phase.DIAGNOSIS: (ActionType.PRESCRIBE_TREATMENT, "提交诊疗方案"),
            Phase.TREATMENT: (ActionType.PRESCRIBE_TREATMENT, "提交诊疗方案"),
            Phase.COMPLETED: (ActionType.PRESCRIBE_TREATMENT, "诊疗已完成"),
        }
        action, target = defaults.get(self.current_phase, (ActionType.PRESCRIBE_TREATMENT, "提交方案"))
        return {
            "action": action,
            "target": target,
            "reason": f"阶段 {self.current_phase.value} 的默认行动",
            "phase": self.current_phase.value,
        }

    def _fallback_plan(
        self, collected_info: Dict[str, Any], exam_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """规划失败时的回退策略。"""
        symptoms = collected_info.get("symptoms", [])
        has_chief_complaint = bool(collected_info.get("chief_complaint"))
        has_exam = bool(exam_results)

        if not has_chief_complaint:
            phase = "inquiry"
            priority = [{"action": "ask", "target": "主诉和现病史", "reason": "尚未收集基本信息"}]
        elif not has_exam:
            phase = "examination"
            priority = [{"action": "examine", "target": "基础检查", "reason": "问诊已有基础信息，需检查支持诊断"}]
        else:
            phase = "diagnosis"
            priority = [{"action": "diagnose", "target": "综合诊断", "reason": "信息已较充分，可以诊断"}]

        return {
            "primary_hypothesis": "待定",
            "hypothesis_confidence": 0.3,
            "differential_diagnoses": [],
            "strategy": {
                "current_phase": phase,
                "phase_goal": "完成诊疗",
                "priority_actions": priority,
                "info_gaps": [],
                "decision_points": [],
            },
            "phase_plan": {},
            "risk_assessment": {"urgent_findings": [], "red_flags": [], "safety_constraints": []},
        }

    def should_replan(self, new_info: Dict[str, Any] = None) -> bool:
        """判断是否需要重新规划。

        触发条件：
        1. 批判评估为 needs_replan
        2. 发现新的高风险信息
        3. 阶段转换时
        4. 关键假设被推翻

        Args:
            new_info: 新获取的信息

        Returns:
            是否需要重新规划
        """
        # 批判建议重规划
        if self.criticism_result and self.criticism_result.get("overall_assessment") == "needs_replan":
            return True

        # 发现高风险信息
        if new_info:
            red_flags = new_info.get("red_flags", [])
            if red_flags:
                logger.info(f"[规划] 发现危险信号: {red_flags}，触发重规划")
                return True

        return False

    def get_plan_summary(self) -> str:
        """获取当前规划的摘要文本，用于注入后续 prompt。"""
        if not self.current_plan:
            return "暂无诊疗规划。"

        lines = ["【当前诊疗规划】"]
        lines.append(f"主要假设: {self.current_plan.get('primary_hypothesis', '未知')}")
        lines.append(f"置信度: {self.current_plan.get('hypothesis_confidence', '?')}")

        strategy = self.current_plan.get("strategy", {})
        lines.append(f"当前阶段: {strategy.get('current_phase', '?')}")
        lines.append(f"阶段目标: {strategy.get('phase_goal', '?')}")

        dd = self.current_plan.get("differential_diagnoses", [])
        if dd:
            lines.append("鉴别诊断:")
            for i, d in enumerate(dd[:3], 1):
                lines.append(f"  {i}. {d.get('diagnosis', '?')} (可能性: {d.get('likelihood', '?')})")

        gaps = strategy.get("info_gaps", [])
        if gaps:
            lines.append(f"信息缺口: {', '.join(gaps[:5])}")

        risk = self.current_plan.get("risk_assessment", {})
        red_flags = risk.get("red_flags", [])
        if red_flags:
            lines.append(f"危险信号: {', '.join(red_flags)}")

        return "\n".join(lines)


class MyDoctorAgent(BaseDoctorAgent):
    """参赛医生 Agent 实现。

    采用规划-执行-反思架构：
    - Planner 作为中枢，制定全局策略并通过 Reflection/Criticism 自我修正
    - LLM 进行深度推理，决策下一步具体操作（tool call）
    - Memory 整合历史经验，指导规划决策
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.trace_collector = TraceCollector.from_config(config)
        self.actions.trace_collector = self.trace_collector
        self.prompt = DoctorPrompt()
        self.memory = DoctorMemory(config)
        self.llm = LLMClient(config)
        self.max_ask_rounds = config.get("max_ask_rounds", 5)
        self.max_exam_rounds = config.get("max_exam_rounds", 3)
        self.log_llm_prompts = config.get("log_llm_prompts", False)
        execution_config = config.get("execution", {}) or {}
        self.fast_mode = bool(execution_config.get("fast_mode", False))
        self.case_timeout_seconds = float(execution_config.get("case_timeout_seconds", 0) or 0)
        self.fallback_reserve_seconds = float(
            execution_config.get("fallback_reserve_seconds", 20) or 20
        )
        self.train_post_submit_reserve_seconds = float(
            execution_config.get("train_post_submit_reserve_seconds", 40) or 40
        )
        self.train_evaluation_timeout_seconds = float(
            execution_config.get("train_evaluation_timeout_seconds", 12) or 12
        )
        self.train_reflection_timeout_seconds = float(
            execution_config.get("train_reflection_timeout_seconds", 24) or 24
        )
        self.max_llm_calls_per_case = int(execution_config.get("max_llm_calls_per_case", 0) or 0)
        self.max_llm_repair_calls_per_case = int(
            execution_config.get("max_llm_repair_calls_per_case", 2) or 2
        )
        self.skip_train_reflection = bool(
            execution_config.get("skip_train_reflection", self.fast_mode)
        )
        self.planner_criticism_max_calls = int(
            execution_config.get("planner_criticism_max_calls", 0 if self.fast_mode else 2)
        )
        self.planner_max_total_actions = int(
            execution_config.get("planner_max_total_actions", 6 if self.fast_mode else 15)
        )
        self.fast_initial_question = execution_config.get(
            "fast_initial_question",
            "请描述这次最主要的不适、开始时间、伴随症状、既往病史、用药史和过敏史。",
        )
        self.fast_exam_items = execution_config.get("fast_exam_items") or [
            "体格检查",
            "超声心动图",
            "心电图（ECG）",
            "胸部X线检查（CXR）",
            "心导管检查",
            "血常规",
            "C反应蛋白",
            "胸部CT",
            "尿常规",
            "腹部B超",
            "甲状腺功能",
        ]
        self.fast_max_exam_items = int(execution_config.get("fast_max_exam_items", 10) or 10)
        contract_config = config.get("llm_contract", {}) or {}
        self.llm_contract_executor = LLMContractExecutor(
            enabled=bool(contract_config.get("enabled", True)),
            repair_enabled=bool(contract_config.get("repair_enabled", True)),
        )
        self.context_compiler = StageContextCompiler(config.get("context_compiler", {}) or {})
        self._llm_context_audit: List[Dict[str, Any]] = []

        learning_config = config.get("learning", {}) or {}
        self.freeze_active_learning = bool(
            learning_config.get("freeze_active_knowledge", True)
        )

        # 静态医学知识库（症状倒排 + 检查规范化 + RAG）
        ref_dir = config.get("ref_data_dir", "data/ref_data")
        self.knowledge = KnowledgeBase(
            ref_dir=ref_dir,
            allow_auto_alias_promotion=bool(
                learning_config.get("auto_promote_exam_aliases", False)
            )
            and not self.freeze_active_learning,
        )
        self.diagnosis_chain_enabled = bool(
            (config.get("diagnosis", {}) or {}).get("enabled", True)
        )
        self.legacy_candidate_submission = bool(
            (config.get("diagnosis", {}) or {}).get("legacy_candidate_submission", True)
        )
        self.clinical_normalizer = ClinicalEvidenceNormalizer(ref_dir=ref_dir)
        self.evidence_agent = EvidenceAgent(ref_dir=ref_dir, normalizer=self.clinical_normalizer)
        self.evidence_compiler = HybridEvidenceCompiler(
            normalizer=self.clinical_normalizer,
            ref_dir=ref_dir,
        )
        self.diagnosis_engine = DiagnosisDecisionEngine(config=config, ref_dir=ref_dir)
        diagnosis_config = config.get("diagnosis", {}) or {}
        self.exam_agent = ExamStrategyAgent(
            self.knowledge,
            discriminating_exam_max_items=int(
                diagnosis_config.get("discriminating_exam_max_items", 6) or 6
            ),
        )
        self.inquiry_agent = InquiryStrategyAgent(self.knowledge)
        self.quality_agent = QualityAgent(
            self.knowledge,
            allowed_diagnoses=self.diagnosis_engine.knowledge.allowed_names,
        )
        self.treatment_agent = TreatmentStrategyAgent(
            self.knowledge,
            diagnostic_knowledge=self.diagnosis_engine.knowledge,
        )
        self.treatment_safety = TreatmentSafetyGate(self.diagnosis_engine.knowledge)
        async def _diagnosis_critic_llm_chat_json(
            messages: List[Dict[str, str]],
            temperature: float = None,
            **kwargs: Any,
        ) -> Dict[str, Any]:
            return await self._llm_chat_json(
                messages,
                temperature=temperature,
                purpose="diagnosis_critic",
            )

        self.diagnosis_critic = DiagnosisCritic(
            config=config,
            knowledge=self.diagnosis_engine.knowledge,
            resolver=self.diagnosis_engine.resolver,
            llm_chat_json=_diagnosis_critic_llm_chat_json,
        )
        self.structural_agent = StructuralDiagnosisAgent()
        self.evidence_engine = EvidenceDiagnosisEngine(ref_dir=ref_dir)
        self.diagnostic_learning = DiagnosticLearningStore(
            path=diagnosis_config.get(
                "learning_path",
                os.path.join("outputs", "runtime_state", "pending_diagnostic_rules.json"),
            )
        )
        self._case_started_at = 0.0
        self._case_deadline = 0.0
        self._case_clinical_deadline = 0.0
        self._case_post_submit_reserve_seconds = 0.0
        self._last_diagnosis_audit: Dict[str, Any] = {}
        self._last_exam_authorization: List[Dict[str, Any]] = []
        self.targeted_exam_result_parser = TargetedExamResultParser()
        self.claim_resolution_updater = ClaimResolutionUpdater()
        self.exam_recovery_pattern_compiler = EvidencePatternCompiler(ref_dir=ref_dir)
        self._exam_result_intent_bindings: List[Dict[str, Any]] = []
        self._targeted_exam_result_parses: List[Dict[str, Any]] = []
        self._targeted_exam_observations: List[Observation] = []
        self._claim_resolution_ledger: Dict[str, Dict[str, Any]] = {}
        self._claim_resolution_update_audit: List[Dict[str, Any]] = []
        self._claim_match_events: List[Dict[str, Any]] = []
        self._claim_state_version = 0
        self._diagnostic_state_version = 0
        self._last_diagnosis_decision_obj: Any = None
        self._clinical_admission_audit: List[Dict[str, Any]] = []
        self._candidate_claim_contract_views: List[Dict[str, Any]] = []
        self._claim_state_materialization_audit: List[Dict[str, Any]] = []
        self._claim_state_invariant_audit: List[Dict[str, Any]] = []
        self._historical_claim_hydration_keys: set[str] = set()
        self._clinical_transition_trace: List[Dict[str, Any]] = []
        self._last_successful_clinical_transition: Dict[str, Any] = {}
        self._case_id_for_thinking = ""
        self._thinking_snapshots: List[Dict[str, Any]] = []

        # 规划器（延迟初始化，因为需要绑定异步方法）
        self._planner: Optional[Planner] = None

        # LLM 成本可观测：调用次数统计
        self._llm_call_count = 0
        self._llm_repair_call_count = 0
        self._llm_call_by_kind: Dict[str, int] = {}
        self._llm_call_audit: List[Dict[str, Any]] = []
        self._llm_logical_call_index = 0

        # 相关经验缓存：以 symptoms 元组为 key，避免每次 executor 重查
        self._exp_cache: Dict[tuple, List[Dict[str, Any]]] = {}

        # 自迭代闭环：缺陷检测器 + 策略补丁库（默认启用；可通过 config 关闭）
        self.self_improve_enabled = bool(config.get("self_improve_enabled", True))
        # P1 LLM 归因通道开关（默认开；关闭则 detect_all 退化为纯规则通道）
        self.self_improve_use_llm_attribute = bool(
            config.get("self_improve_use_llm_attribute", True)
        )
        self.detector: Optional[DefectDetector] = None
        self.policy_store: Optional[PolicyStore] = None
        self.candidate_policy_store: Optional[CandidatePolicyStore] = None
        self.rule_generalizer: Optional[RuleGeneralizer] = None
        if self.self_improve_enabled:
            try:
                # 注入 llm_chat 以启用 LLM 归因通道；critic 内部会在 use_llm=False 或 llm_chat=None 时自动退化
                self.detector = DefectDetector(
                    knowledge=self.knowledge,
                    llm_chat=self._llm_chat if self.self_improve_use_llm_attribute else None,
                )
                policy_path = config.get(
                    "policy_store_path", "outputs/runtime_state/policies.json"
                )
                candidate_policy_path = config.get(
                    "candidate_policy_store_path",
                    "outputs/runtime_state/candidate_policies.json",
                )
                self.policy_store = PolicyStore(store_path=policy_path)
                self.candidate_policy_store = CandidatePolicyStore(
                    path=candidate_policy_path
                )
                self.rule_generalizer = RuleGeneralizer()
                legacy = self.candidate_policy_store.ingest_legacy_patches(
                    self.policy_store.patches
                )
                sanitation = self.policy_store.sanitize_shadow_patches()
                logger.info(
                    "[SelfImprove] legacy policies copied to candidate store: "
                    "converted=%s quarantined=%s",
                    legacy.get("converted", 0),
                    legacy.get("quarantined", 0),
                )
                logger.info(
                    f"[自迭代] 已启用 detector + policy_store "
                    f"(现有补丁 {len(self.policy_store.patches)} 项, "
                    f"清退不可执行 shadow {sanitation['retired']} 项)"
                )
            except Exception as _e:
                logger.warning(f"[自迭代] 初始化失败，降级为无自迭代模式: {_e}")
                self.detector = None
                self.policy_store = None
                self.candidate_policy_store = None
                self.rule_generalizer = None
        self.memory_manager = DoctorAgentMemory(
            config=config,
            episodic_memory=self.memory,
            semantic_memory=self.knowledge,
            policy_store=self.policy_store,
        )
        logger.info(
            "[MemorySystem] 已启用结构化记忆层: %s",
            json.dumps(self.memory_manager.stats(), ensure_ascii=False),
        )

    def _get_cached_experience(self, collected_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于多维特征获取相关经验，带缓存。

        召回维度: symptoms + chief_complaint + past_history + age_bucket + gender。
        缓存 key 结合症状与年龄段/性别，避免不同患者互相污染。

        Args:
            collected_info: 已收集患者信息

        Returns:
            相关历史经验列表
        """
        symptoms = collected_info.get("symptoms", []) or []
        if not symptoms and not collected_info.get("chief_complaint"):
            return []
        key = (
            tuple(sorted(str(s) for s in symptoms)),
            str(collected_info.get("chief_complaint", ""))[:64],
            self.memory._bucket_age(collected_info.get("age", "")),
            str(collected_info.get("gender", "")),
        )
        if key not in self._exp_cache:
            try:
                # 优先多维召回，失败降级到症状召回
                if hasattr(self.memory, "search_relevant_experience_multi"):
                    self._exp_cache[key] = self.memory_manager.search_episodic(
                        collected_info, top_k=3
                    ) or []
                else:
                    self._exp_cache[key] = self.memory.search_relevant_experience(
                        list(symptoms), top_k=3
                    ) or []
            except Exception as e:
                logger.warning(f"[记忆] 检索相关经验失败: {e}")
                self._exp_cache[key] = []
        return self._exp_cache[key]

    def _bump_llm_counter(self, kind: str = "chat") -> None:
        """LLM 调用计数（用于成本监控）。"""
        if kind == "json_repair":
            self._llm_repair_call_count += 1
        else:
            self._llm_call_count += 1
        self._llm_call_by_kind[kind] = self._llm_call_by_kind.get(kind, 0) + 1

    def _reset_llm_counter(self) -> None:
        """重置计数器（每个患者独立统计）。"""
        self._llm_call_count = 0
        self._llm_repair_call_count = 0
        self._llm_call_by_kind = {}
        self._llm_call_audit = []
        self._llm_logical_call_index = 0
        self._llm_context_audit = []
        self._exp_cache = {}

    def _compile_llm_context(self, stage: str, **state: Any) -> Dict[str, Any]:
        compiled = self.context_compiler.compile(stage, **state)
        audit = dict(compiled.get("audit") or {})
        if audit:
            audit["claim_state_version"] = int(getattr(self, "_claim_state_version", 0) or 0)
            audit["diagnostic_state_version"] = int(
                getattr(self, "_diagnostic_state_version", 0) or 0
            )
            self._llm_context_audit.append(audit)
        return dict(compiled.get("context") or state)

    def _can_call_llm(self, kind: str, purpose: str = "") -> bool:
        """Return False when the per-case LLM budget has been exhausted."""
        if kind == "json_repair":
            if self.max_llm_repair_calls_per_case <= 0:
                logger.warning("[LLM] skip %s call: repair budget disabled", kind)
                return False
            remaining = self.max_llm_repair_calls_per_case - self._llm_repair_call_count
            if remaining <= 0:
                logger.warning(
                    "[LLM] skip %s call: repair budget reached (%s)",
                    kind,
                    self.max_llm_repair_calls_per_case,
                )
                return False
            priority = "medium"
            executor = getattr(self, "llm_contract_executor", None)
            if executor is not None and hasattr(executor, "repair_priority_for"):
                priority = str(executor.repair_priority_for(purpose) or "medium")
            if remaining == 1 and priority not in ("critical", "high"):
                logger.warning(
                    "[LLM] skip %s call for %s: reserve final repair budget for critical stages",
                    kind,
                    purpose or "unclassified",
                )
                return False
            if self._llm_repair_call_count < self.max_llm_repair_calls_per_case:
                return True
        if self.max_llm_calls_per_case <= 0:
            return True
        if self._llm_call_count < self.max_llm_calls_per_case:
            return True
        logger.warning(
            "[LLM] skip %s call: per-case budget reached (%s)",
            kind,
            self.max_llm_calls_per_case,
        )
        return False

    def _next_llm_logical_call_id(self) -> str:
        self._llm_logical_call_index += 1
        return f"L{self._llm_logical_call_index:04d}"

    def _llm_required_fields_for_purpose(self, purpose: str) -> List[str]:
        executor = getattr(self, "llm_contract_executor", None)
        if executor is not None and hasattr(executor, "required_fields_for"):
            return list(executor.required_fields_for(purpose))
        return []

    @staticmethod
    def _llm_failure_priority(flags: List[str]) -> str:
        priority = [
            "llm_budget_exhausted",
            "timeout",
            "rate_limited",
            "http_error",
            "connection_error",
            "raw_response_empty",
            "generation_truncated",
            "json_parse_failed",
            "unexpected_json_type",
            "schema_missing_fields",
            "schema_type_mismatch",
            "semantic_validation_failed",
            "contract_drift",
            "consumer_rejected",
            "unknown_exception",
        ]
        for item in priority:
            if item in flags:
                return item
        return ""

    @staticmethod
    def _parsed_type_name(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, dict):
            return "dict"
        if isinstance(value, list):
            return "list"
        if isinstance(value, str):
            return "string"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, (int, float)):
            return "number"
        return type(value).__name__

    def _classify_llm_audit_record(self, record: Dict[str, Any]) -> None:
        flags = list(dict.fromkeys(record.get("failure_flags") or []))
        status = record.get("http_status")
        exception_type = str(record.get("exception_type") or "")
        if record.get("model_invoked") is False and not record.get("call_started"):
            flags.append("llm_budget_exhausted")
        if exception_type:
            if "Timeout" in exception_type:
                flags.append("timeout")
            elif status == 429:
                flags.append("rate_limited")
            elif status:
                flags.append("http_error")
            elif "Connect" in exception_type or "Request" in exception_type:
                flags.append("connection_error")
            else:
                flags.append("unknown_exception")
        if status == 429:
            flags.append("rate_limited")
        elif isinstance(status, int) and status >= 400:
            flags.append("http_error")
        if str(record.get("finish_reason") or "").lower() == "length":
            flags.append("generation_truncated")
        if record.get("model_invoked") and record.get("http_status") and not record.get("raw_response_present"):
            flags.append("raw_response_empty")
        if record.get("json_expected"):
            if record.get("parse_success") is False:
                flags.append("json_parse_failed")
            elif record.get("required_fields") and record.get("parsed_type") not in ("dict", ""):
                flags.append("unexpected_json_type")
        if record.get("schema_success") is False:
            if record.get("missing_fields"):
                flags.append("schema_missing_fields")
            else:
                flags.append("schema_type_mismatch")
        contract_validation = record.get("contract_validation") or {}
        if isinstance(contract_validation, dict):
            if contract_validation.get("type_errors"):
                flags.append("schema_type_mismatch")
            if contract_validation.get("semantic_success") is False:
                flags.append("semantic_validation_failed")
        if record.get("consumer_accepted") is False:
            if record.get("schema_success") is True:
                flags.append("contract_drift")
                record["contract_drift_detected"] = True
            flags.append("consumer_rejected")
        flags = list(dict.fromkeys(flags))
        record["failure_flags"] = flags
        primary = self._llm_failure_priority(flags)
        record["primary_failure_reason"] = primary
        if record.get("fallback_used") and not record.get("fallback_trigger"):
            record["fallback_trigger"] = primary

    def _append_llm_audit(
        self,
        *,
        kind: str,
        purpose: str,
        stage: str = "",
        json_expected: bool,
        metadata: Optional[Dict[str, Any]] = None,
        parsed_value: Any = None,
        exception: Optional[BaseException] = None,
        model_invoked: bool = True,
        fallback_used: bool = False,
    ) -> Dict[str, Any]:
        metadata = dict(metadata or {})
        logical_id = str(metadata.get("logical_call_id") or "")
        if not logical_id:
            logical_id = self._next_llm_logical_call_id()
        attempt_index = int(metadata.get("attempt_index") or 1)
        now = round(time.time(), 3)
        parsed_type = self._parsed_type_name(parsed_value) if json_expected else ""
        parse_success: Optional[bool]
        if json_expected:
            parse_success = (
                parsed_value is not None
                and not (
                    isinstance(parsed_value, dict)
                    and set(parsed_value.keys()) == {"raw_response"}
                    and isinstance(parsed_value.get("raw_response"), str)
                )
            )
        else:
            parse_success = None
        required_fields = self._llm_required_fields_for_purpose(purpose)
        schema_applicable = bool(required_fields) and json_expected and parse_success is True
        missing_fields = [
            field
            for field in required_fields
            if not (isinstance(parsed_value, dict) and field in parsed_value)
        ]
        schema_success: Optional[bool] = None
        if schema_applicable:
            schema_success = not missing_fields
        contract_validation = metadata.get("contract_validation") or {}
        if isinstance(contract_validation, dict) and contract_validation.get("applicable"):
            schema_applicable = True
            schema_success = contract_validation.get("schema_success")
            missing_fields = list(contract_validation.get("missing_fields") or [])
            required_fields = list(
                dict.fromkeys(
                    required_fields
                    + list(contract_validation.get("critical_fields") or [])
                    + list(contract_validation.get("missing_fields") or [])
                )
            )
        record: Dict[str, Any] = {
            "call_id": f"{logical_id}-A{attempt_index}",
            "logical_call_id": logical_id,
            "attempt_index": attempt_index,
            "attempt_type": metadata.get("attempt_type") or "generate",
            "purpose": purpose or "unclassified",
            "stage": stage or purpose or "unclassified",
            "model": metadata.get("model") or getattr(self.llm, "model_name", ""),
            "call_started": now if model_invoked else None,
            "call_completed": now,
            "model_invoked": bool(model_invoked),
            "http_status": metadata.get("http_status"),
            "latency_ms": metadata.get("latency_ms"),
            "input_tokens": metadata.get("input_tokens"),
            "output_tokens": metadata.get("output_tokens"),
            "total_tokens": metadata.get("total_tokens"),
            "finish_reason": metadata.get("finish_reason"),
            "raw_response_present": bool(metadata.get("raw_response_present")),
            "response_chars": int(metadata.get("response_chars") or 0),
            "requested_max_tokens": metadata.get("requested_max_tokens"),
            "json_expected": bool(json_expected),
            "parse_success": parse_success,
            "parsed_type": parsed_type,
            "schema_applicable": bool(schema_applicable),
            "schema_success": schema_success,
            "required_fields": required_fields,
            "missing_fields": missing_fields if schema_applicable else [],
            "contract_version": metadata.get("contract_version") or (
                (metadata.get("contract_validation") or {}).get("contract_version")
                if isinstance(metadata.get("contract_validation"), dict)
                else ""
            ),
            "deterministic_normalizations": list(
                metadata.get("deterministic_normalizations") or []
            ),
            "contract_drift_detected": False,
            "consumer_acceptance_reason": metadata.get("consumer_acceptance_reason") or "",
            "consumer_rejection_code": metadata.get("consumer_rejection_code") or "",
            "consumer_accepted": (
                None
                if json_expected
                else (False if fallback_used else True)
            ),
            "failure_flags": [],
            "primary_failure_reason": "",
            "fallback_trigger": "",
            "fallback_used": bool(fallback_used),
            "contract_validation": dict(contract_validation or {}),
            "contract_repair_attempted": bool(metadata.get("contract_repair_attempted", False)),
            "contract_repair_succeeded": metadata.get("contract_repair_succeeded"),
            "exception_type": type(exception).__name__ if exception else metadata.get("exception_type", ""),
        }
        self._classify_llm_audit_record(record)
        self._llm_call_audit.append(record)
        return record

    def _mark_last_llm_consumer_result(
        self,
        purpose: str,
        accepted: bool,
        *,
        fallback_used: bool = False,
        fallback_trigger: str = "",
        consumer_acceptance_reason: str = "",
        consumer_rejection_code: str = "",
    ) -> None:
        for record in reversed(self._llm_call_audit):
            if record.get("purpose") == purpose and record.get("consumer_accepted") is None:
                record["consumer_accepted"] = bool(accepted)
                if accepted:
                    record["consumer_acceptance_reason"] = (
                        consumer_acceptance_reason or "ACCEPTED"
                    )
                else:
                    record["consumer_rejection_code"] = (
                        consumer_rejection_code or "LEGACY_CONSUMER_CONTRACT_DRIFT"
                    )
                    if record.get("schema_success") is True:
                        record["contract_drift_detected"] = True
                if fallback_used:
                    record["fallback_used"] = True
                if fallback_trigger:
                    record["fallback_trigger"] = fallback_trigger
                self._classify_llm_audit_record(record)
                return

    @staticmethod
    def _llm_contract_summary_from_audit(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_purpose: Dict[str, int] = {}
        failures: Dict[str, int] = {}
        fallback_purposes: List[str] = []
        fallback_calls = 0
        repair_calls = 0
        repair_success = 0
        contract_drift = 0
        deterministic_normalizations = 0
        for record in records or []:
            purpose = str(record.get("purpose") or "unclassified")
            by_purpose[purpose] = by_purpose.get(purpose, 0) + 1
            reason = str(record.get("primary_failure_reason") or "")
            if reason:
                failures[reason] = failures.get(reason, 0) + 1
            if record.get("fallback_used"):
                fallback_calls += 1
                if purpose not in fallback_purposes:
                    fallback_purposes.append(purpose)
            if record.get("contract_repair_attempted") or str(record.get("attempt_type") or "") == "repair":
                repair_calls += 1
                if record.get("contract_repair_succeeded") is True:
                    repair_success += 1
            if record.get("contract_drift_detected") or "contract_drift" in (
                record.get("failure_flags") or []
            ):
                contract_drift += 1
            deterministic_normalizations += len(record.get("deterministic_normalizations") or [])
        return {
            "total_calls": len(records or []),
            "fallback_calls": fallback_calls,
            "fallback_purposes": fallback_purposes,
            "call_count_by_purpose": by_purpose,
            "primary_failure_reasons": failures,
            "repair_attempts": repair_calls,
            "repair_successes": repair_success,
            "contract_drift_count": contract_drift,
            "deterministic_normalization_count": deterministic_normalizations,
        }

    @staticmethod
    def _deterministic_failure_attribution(
        *,
        report: Dict[str, Any],
        runtime_audit: Dict[str, Any],
        llm_call_audit: List[Dict[str, Any]],
        expected: List[str],
        top_twenty: List[str],
        submitted: List[str],
    ) -> Dict[str, Any]:
        """Classify the dominant failure domain without using another LLM."""
        diagnosis_accuracy = None
        for key in ("diagnosisAccuracy", "diagnosis_accuracy"):
            try:
                if report.get(key) is not None:
                    diagnosis_accuracy = float(report.get(key))
                    break
            except (TypeError, ValueError):
                pass
        if diagnosis_accuracy == 1.0:
            return {
                "primary_failure_domain": "NONE",
                "primary_failure_reason": "",
                "secondary_failure_reasons": [],
                "medical_failure_evaluable": True,
            }

        tool_summary = runtime_audit.get("tool_contract_summary") or {}
        if int(tool_summary.get("retry_exhausted_calls") or 0) > 0:
            return {
                "primary_failure_domain": "TOOL_BACKEND",
                "primary_failure_reason": "TOOL_RETRY_EXHAUSTED",
                "secondary_failure_reasons": [],
                "medical_failure_evaluable": False,
            }

        flags = [
            str(flag)
            for record in llm_call_audit
            for flag in (record.get("failure_flags") or [])
        ]
        purposes_by_failure: Dict[str, List[str]] = {}
        for record in llm_call_audit:
            purpose = str(record.get("purpose") or "unclassified")
            for flag in record.get("failure_flags") or []:
                purposes_by_failure.setdefault(str(flag), []).append(purpose)

        if "llm_budget_exhausted" in flags:
            return {
                "primary_failure_domain": "LLM_GENERATION",
                "primary_failure_reason": "LLM_BUDGET_EXHAUSTED",
                "secondary_failure_reasons": sorted(set(flags)),
                "medical_failure_evaluable": False,
                "affected_purposes": sorted(set(purposes_by_failure.get("llm_budget_exhausted", []))),
            }
        if "contract_drift" in flags:
            return {
                "primary_failure_domain": "CONTRACT",
                "primary_failure_reason": "CONTRACT_DRIFT",
                "secondary_failure_reasons": sorted(set(flags)),
                "medical_failure_evaluable": False,
            }
        if "schema_type_mismatch" in flags:
            return {
                "primary_failure_domain": "CONTRACT",
                "primary_failure_reason": "SCHEMA_TYPE_MISMATCH",
                "secondary_failure_reasons": sorted(set(flags)),
                "medical_failure_evaluable": False,
            }
        if "generation_truncated" in flags and any(
            str(record.get("purpose") or "") == "diagnosis"
            for record in llm_call_audit
            if "generation_truncated" in (record.get("failure_flags") or [])
        ):
            return {
                "primary_failure_domain": "LLM_GENERATION",
                "primary_failure_reason": "DIAGNOSIS_GENERATION_TRUNCATED",
                "secondary_failure_reasons": sorted(set(flags)),
                "medical_failure_evaluable": False,
            }

        if expected and not all(name in top_twenty for name in expected):
            return {
                "primary_failure_domain": "RECALL",
                "primary_failure_reason": "EXPECTED_NOT_IN_TOP20",
                "secondary_failure_reasons": [],
                "medical_failure_evaluable": True,
            }
        if expected and submitted and set(submitted) != set(expected):
            return {
                "primary_failure_domain": "ARBITRATION",
                "primary_failure_reason": "SUBMITTED_DIFFERS_FROM_EXPECTED",
                "secondary_failure_reasons": sorted(set(flags)),
                "medical_failure_evaluable": True,
            }
        return {
            "primary_failure_domain": "REASONING",
            "primary_failure_reason": "UNATTRIBUTED_MEDICAL_FAILURE",
            "secondary_failure_reasons": sorted(set(flags)),
            "medical_failure_evaluable": True,
        }

    def _get_planner(self) -> Planner:
        """获取或初始化规划器。"""
        if self._planner is None:
            self._planner = Planner(
                prompt=self.prompt,
                llm_chat_json=self._llm_chat_json,
                llm_chat=self._llm_chat,
                memory=self.memory,
                mark_llm_consumer_result=self._mark_last_llm_consumer_result,
                compile_llm_context=self._compile_llm_context,
            )
            self._planner.max_inquiry_rounds = self.max_ask_rounds
            self._planner.max_exam_rounds = self.max_exam_rounds
            self._planner.criticism_max_calls = max(0, self.planner_criticism_max_calls)
            self._planner.max_total_actions = max(1, self.planner_max_total_actions)
            # 注入策略补丁库（若 agent 已启用）
            if getattr(self, "policy_store", None) is not None:
                self._planner.policy_store = self.policy_store
        return self._planner

    async def _run_case_pipeline(
        self,
        patient_id: str,
        post_submit_reserve_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """Run one case with an optional fast path and hard timeout guard."""
        self._case_started_at = time.monotonic()
        post_submit_reserve_seconds = max(0.0, float(post_submit_reserve_seconds or 0))
        if self.case_timeout_seconds > 0:
            maximum_post_reserve = max(
                0.0,
                self.case_timeout_seconds - self.fallback_reserve_seconds - 1.0,
            )
            post_submit_reserve_seconds = min(
                post_submit_reserve_seconds,
                maximum_post_reserve,
            )
        self._case_post_submit_reserve_seconds = post_submit_reserve_seconds
        self._case_deadline = (
            self._case_started_at + self.case_timeout_seconds
            if self.case_timeout_seconds > 0
            else 0.0
        )
        self._case_clinical_deadline = (
            self._case_deadline - post_submit_reserve_seconds
            if self._case_deadline > 0
            else 0.0
        )
        self._last_diagnosis_audit = {}
        self._last_exam_authorization = []
        self._exam_result_intent_bindings = []
        self._targeted_exam_result_parses = []
        self._targeted_exam_observations = []
        self._claim_resolution_ledger = {}
        self._claim_resolution_update_audit = []
        self._claim_match_events = []
        self._claim_state_version = 0
        self._diagnostic_state_version = 0
        self._last_diagnosis_decision_obj = None
        self._clinical_admission_audit = []
        self._candidate_claim_contract_views = []
        self._claim_state_materialization_audit = []
        self._claim_state_invariant_audit = []
        self._historical_claim_hydration_keys = set()
        self._clinical_transition_trace = []
        self._last_successful_clinical_transition = {}
        self._case_id_for_thinking = patient_id
        self._thinking_snapshots = []
        self._llm_context_audit = []
        runner = (
            self._execute_fast_path(patient_id)
            if self.fast_mode
            else self._execute_with_planner(patient_id)
        )
        if self.case_timeout_seconds <= 0:
            return await runner
        main_timeout = max(
            1.0,
            self.case_timeout_seconds
            - post_submit_reserve_seconds
            - max(0.0, self.fallback_reserve_seconds),
        )
        try:
            return await asyncio.wait_for(runner, timeout=main_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "[CaseTimeout] patient=%s clinical path exceeded %.1fs; "
                "using %.1fs reserve for fallback submission",
                patient_id,
                main_timeout,
                max(0.0, self._remaining_case_seconds()),
            )
            return await self._build_timeout_final_result(
                patient_id,
                "临床诊疗链路超过 "
                f"{self.case_timeout_seconds - post_submit_reserve_seconds:.0f} 秒预算，"
                "已触发保底提交。",
            )

    def _remaining_case_seconds(self) -> float:
        deadline = self._case_clinical_deadline or self._case_deadline
        if deadline <= 0:
            return 10_000.0
        return max(0.0, deadline - time.monotonic())

    def _remaining_total_case_seconds(self) -> float:
        if self._case_deadline <= 0:
            return 10_000.0
        return max(0.0, self._case_deadline - time.monotonic())

    async def _build_timeout_final_result(self, patient_id: str, reason: str) -> Dict[str, Any]:
        """Submit a safe final result when the main execution path is interrupted."""
        collected_info = getattr(self, "_last_collected_info", {}) or {}
        exam_results = getattr(self, "_last_exam_results", {}) or {}
        planner = self._get_planner()
        conversation_rounds = len(
            [a for a in planner.action_history if a.get("type") == "ask_patient"]
        )
        fallback = self.quality_agent.default_final_result(reason)
        if self.diagnosis_chain_enabled:
            evidence = self._normalize_with_exam_recovery(collected_info, exam_results)
            decision = self.diagnosis_engine.decide(
                self._diagnosis_input_with_runtime_state(fallback),
                [],
                evidence,
            )
            self._last_diagnosis_decision_obj = decision
            admitted_views = self._materialize_admitted_candidate_claim_states()
            self._hydrate_claim_states_from_existing_exam_observations(
                admitted_views,
                stage="timeout_decision_claim_hydration",
            )
            fallback = self.diagnosis_engine.apply_to_result(fallback, decision, evidence)
        else:
            fallback = self.evidence_engine.review(
                fallback,
                collected_info=collected_info,
                exam_results=exam_results,
            )
            if not fallback.get("_trusted_diagnoses"):
                fallback = self.structural_agent.review(
                    fallback,
                    collected_info=collected_info,
                    exam_results=exam_results,
                )
        fallback = self.quality_agent.review_final_result(
            fallback,
            collected_info=collected_info,
            exam_results=exam_results,
            conversation_rounds=conversation_rounds,
        )
        fallback = self.treatment_agent.review(
            fallback,
            collected_info=collected_info,
            exam_results=exam_results,
        )
        fallback = self.treatment_safety.review(
            fallback,
            collected_info=collected_info,
            exam_results=exam_results,
        )
        try:
            remaining = self._remaining_case_seconds()
            if remaining <= 0.5:
                raise asyncio.TimeoutError("no fallback submission budget remaining")
            submit_result = await asyncio.wait_for(
                self.actions.prescribe_treatment(
                    patient_id=patient_id,
                    diagnosis=fallback.get("diagnosis", []),
                    treatment_plan=fallback.get("treatment_plan", ""),
                    reasoning=fallback.get("reasoning", ""),
                ),
                timeout=max(0.5, remaining - 0.25),
            )
            if isinstance(submit_result, dict):
                fallback.update({k: v for k, v in submit_result.items() if v not in (None, "")})
        except (Exception, asyncio.TimeoutError) as exc:
            logger.warning("[CaseTimeout] fallback submit failed for %s: %s", patient_id, exc)
            fallback.update(
                {
                    "patient_id": patient_id,
                    "caseId": patient_id,
                    "ordered_examinations": list(exam_results.keys()),
                    "finished": True,
                }
            )
        reviewed = self.quality_agent.review_final_result(
            fallback,
            collected_info=collected_info,
            exam_results=exam_results,
            conversation_rounds=conversation_rounds,
        )
        reviewed["_case_elapsed_seconds"] = round(
            max(0.0, time.monotonic() - self._case_started_at), 3
        )
        reviewed["_case_timed_out"] = True
        if self._last_diagnosis_audit:
            self._last_diagnosis_audit["elapsed_seconds"] = reviewed[
                "_case_elapsed_seconds"
            ]
            self._last_diagnosis_audit["timed_out"] = True
        return reviewed

    async def _execute_fast_path(self, patient_id: str) -> Dict[str, Any]:
        """A bounded case path for local training/testing speed and reliability."""
        planner = self._get_planner()
        working_memory = self.memory_manager.start_case(patient_id)
        chat_history: List[Dict[str, str]] = working_memory.chat_history
        collected_info: Dict[str, Any] = working_memory.collected_info
        exam_results: Dict[str, Any] = working_memory.exam_results
        self._last_collected_info = {}
        self._last_exam_results = {}

        logger.info("[FastPath] start patient=%s", patient_id)

        question = str(self.fast_initial_question)
        try:
            answer = await self.actions.ask_patient(
                patient_id=patient_id,
                input_data={"question": question, "chat_history": chat_history},
            )
        except Exception as exc:
            logger.warning("[FastPath] ask_patient failed for %s: %s", patient_id, exc)
            answer = ""

        chat_history.append({"from": "doctor", "text": question})
        chat_history.append({"from": "patient", "text": str(answer)})
        planner.current_phase = Phase.INQUIRY
        planner._record_action("ask_patient", "fast_initial_inquiry", str(answer)[:120])
        planner.inquiry_rounds = 1

        collected_info = self._fallback_parse_patient_response(str(answer), collected_info)
        if answer and not collected_info.get("chief_complaint"):
            collected_info["chief_complaint"] = str(answer)[:180]
        self.memory_manager.update_collected_info(patient_id, collected_info)
        self._last_collected_info = dict(collected_info)

        symptoms = collected_info.get("symptoms", []) or []
        disease_hits = self.knowledge.recall_diseases_by_symptoms(symptoms, top_k=5) if symptoms else []
        candidate_names = [item.get("name") for item in disease_hits if item.get("name")]
        self.memory_manager.update_candidates(patient_id, candidate_names)

        proposed_items = list(self._fallback_generate_examination_items(collected_info))
        for item in self.fast_exam_items:
            if item and item not in proposed_items:
                proposed_items.append(item)
        strategy = self.exam_agent.recommend(
            collected_info=collected_info,
            candidate_diseases=candidate_names,
            proposed_items=proposed_items,
            existing_results=exam_results,
        )
        if strategy.get("strict_diagnosis_driven") or strategy.get("blocked_items"):
            self._last_exam_authorization.append(
                {
                    "stage": "fast_exam",
                    "round": 1,
                    "strict_diagnosis_driven": bool(
                        strategy.get("strict_diagnosis_driven")
                    ),
                    "primary_diagnosis": strategy.get("primary_diagnosis", ""),
                    "authorized_items": list(strategy.get("items") or []),
                    "blocked_items": list(strategy.get("blocked_items") or []),
                }
            )
        normalized_fast_items, _ = self.knowledge.normalize_examinations(self.fast_exam_items)
        raw_exam_items = []
        for item in list(self.fast_exam_items) + normalized_fast_items + strategy.get("items", []):
            if item and item not in raw_exam_items:
                raw_exam_items.append(item)
        exam_items = self.exam_agent.prepare_order_items(
            raw_exam_items,
            collected_info=collected_info,
            candidate_diseases=candidate_names,
            existing_results=exam_results,
            max_items=self.fast_max_exam_items,
        )

        if exam_items:
            try:
                response = await self.actions.order_examination(
                    patient_id=patient_id,
                    items=exam_items,
                    reason="快速路径：基于主诉、症状召回和疾病画像补齐关键检查。",
                )
                new_results = {}
                if response and "results" in response:
                    for exam_name, exam_data in response["results"].items():
                        if isinstance(exam_data, dict) and exam_data.get("status") != "invalid":
                            new_results[exam_name] = exam_data
                if new_results:
                    self._record_targeted_exam_result_recovery(
                        patient_id=patient_id,
                        stage="fast_exam",
                        ordered_items=list(exam_items),
                        new_results=new_results,
                        strategy=strategy,
                    )
                    exam_results.update(new_results)
                    self.memory_manager.update_exam_results(patient_id, new_results)
                    planner._record_action(
                        "order_examination",
                        ",".join(new_results.keys()),
                        "fast_exam_batch",
                    )
                    planner.exam_rounds += 1
            except Exception as exc:
                logger.warning("[FastPath] order_examination failed for %s: %s", patient_id, exc)
        self._last_exam_results = dict(exam_results)

        planner.current_phase = Phase.TREATMENT
        final_result = await self._prescribe(
            patient_id,
            collected_info,
            exam_results,
            chat_history,
            self._get_cached_experience(collected_info),
        )
        planner._record_action("prescribe_treatment", "fast_final_submit", "")
        planner.current_phase = Phase.COMPLETED
        self._last_collected_info = dict(collected_info)
        self._last_exam_results = dict(exam_results)
        self.memory_manager.finish_case(patient_id)
        return final_result

    # ============ 规划驱动执行循环 ============

    async def _execute_with_planner(self, patient_id: str) -> Dict[str, Any]:
        """规划器驱动的核心执行循环。

        替代旧的硬编码流水线（初始问诊→追问→检查→诊断），
        由 Planner 作为中枢，通过 LLM 深度推理决策每一步操作。

        流程：
        1. 初始化状态 → Planner 制定全局策略
        2. 循环：decide_next_action → 执行 → 记录 → 检查重规划
        3. 提交诊疗方案 → 返回结果

        Args:
            patient_id: 患者 ID

        Returns:
            最终诊疗结果
        """
        planner = self._get_planner()
        working_memory = self.memory_manager.start_case(patient_id)
        chat_history: List[Dict[str, str]] = working_memory.chat_history
        collected_info: Dict[str, Any] = working_memory.collected_info
        exam_results: Dict[str, Any] = working_memory.exam_results
        relevant_experience: List[Dict[str, Any]] = []

        logger.info(f"[规划执行] 开始规划驱动诊疗，患者: {patient_id}")

        def _sync_plan(plan_obj: Dict[str, Any]) -> None:
            if isinstance(plan_obj, dict):
                self.memory_manager.update_candidates(
                    patient_id,
                    plan_obj.get("differential_diagnoses", []),
                )

        # 保存执行状态，供后续反思使用
        self._last_collected_info: Dict[str, Any] = {}
        self._last_exam_results: Dict[str, Any] = {}

        # ---- 阶段1：初始问诊（至少收集主诉） ----
        planner.current_phase = Phase.INQUIRY
        collected_info = await self._initial_inquiry(
            patient_id, chat_history, relevant_experience
        )
        self.memory_manager.update_collected_info(patient_id, collected_info)
        self._last_collected_info = dict(collected_info)
        planner._record_action("ask_patient", "初始问诊-主诉", str(collected_info.get("chief_complaint", ""))[:100])
        planner.inquiry_rounds = 1

        # ---- 阶段2：制定全局策略 ----
        plan = await planner.plan(
            collected_info=collected_info,
            exam_results=exam_results,
            chat_history=chat_history,
            relevant_experience=relevant_experience,
        )
        _sync_plan(plan)
        logger.info(f"[规划执行] 初始规划完成，阶段: {planner.current_phase.value}")

        # ---- 阶段3：规划驱动执行循环 ----
        for step in range(planner.max_total_actions):
            # 3.1 决策下一步操作
            action_decision = planner.decide_next_action(plan)
            action_type = action_decision["action"]
            target = action_decision["target"]
            reason = action_decision.get("reason", "")

            logger.info(
                f"[规划执行] 步骤{step + 1}: action={action_type.value}, "
                f"target={target}, phase={planner.current_phase.value}"
            )

            # 3.2 执行操作
            if action_type == ActionType.ASK_PATIENT:
                # 问诊：生成追问问题并询问患者
                result_info = await self._execute_ask_patient(
                    patient_id, chat_history, collected_info,
                    relevant_experience, target, reason,
                )
                if result_info:
                    collected_info = result_info
                    self.memory_manager.update_collected_info(patient_id, collected_info)
                    self._last_collected_info = dict(collected_info)
                    planner._record_action("ask_patient", target, reason)
                    planner.inquiry_rounds += 1
                else:
                    # 问诊被跳过（信息已足够或LLM判断无需追问），触发重规划
                    planner._record_action("ask_patient_skipped", target, "信息已足够，跳过追问")
                    logger.info("[规划执行] 问诊被跳过，触发重规划以转换阶段")
                    plan = await planner.plan(
                        collected_info=collected_info,
                        exam_results=exam_results,
                        chat_history=chat_history,
                        relevant_experience=relevant_experience,
                    )
                    _sync_plan(plan)
                    continue

            elif action_type == ActionType.ORDER_EXAMINATION:
                # 检查：申请检查项目
                result_exams = await self._execute_order_examination(
                    patient_id, collected_info, exam_results,
                    relevant_experience, target, reason,
                )
                if result_exams:
                    exam_results.update(result_exams)
                    self.memory_manager.update_exam_results(patient_id, result_exams)
                    self._last_exam_results = dict(exam_results)
                    planner._record_action("order_examination", target, reason)
                    planner.exam_rounds += 1
                else:
                    # 检查被跳过（检查已足够），触发重规划
                    planner._record_action("examination_skipped", target, "检查已足够，跳过")
                    logger.info("[规划执行] 检查被跳过，触发重规划以转换阶段")
                    plan = await planner.plan(
                        collected_info=collected_info,
                        exam_results=exam_results,
                        chat_history=chat_history,
                        relevant_experience=relevant_experience,
                    )
                    _sync_plan(plan)
                    continue

            elif action_type == ActionType.PRESCRIBE_TREATMENT:
                # 诊断/治疗：提交诊疗方案
                final_result = await self._prescribe(
                    patient_id, collected_info, exam_results,
                    chat_history, relevant_experience,
                )
                planner._record_action("prescribe_treatment", "提交诊疗方案", "")
                planner.current_phase = Phase.COMPLETED
                logger.info(f"[规划执行] 诊疗方案已提交，完成")
                # 保存执行状态供反思使用
                self._last_collected_info = collected_info
                self._last_exam_results = exam_results
                self.memory_manager.finish_case(patient_id)
                return final_result

            elif action_type == ActionType.REPLAN:
                # 重规划
                logger.info(f"[规划执行] 触发重规划")
                plan = await planner.plan(
                    collected_info=collected_info,
                    exam_results=exam_results,
                    chat_history=chat_history,
                    relevant_experience=relevant_experience,
                )
                _sync_plan(plan)
                continue

            # 3.3 检查是否需要重规划
            if planner.should_replan({"red_flags": collected_info.get("red_flags", [])}):
                logger.info(f"[规划执行] 检测到重规划需求")
                plan = await planner.plan(
                    collected_info=collected_info,
                    exam_results=exam_results,
                    chat_history=chat_history,
                    relevant_experience=relevant_experience,
                )
                _sync_plan(plan)
                continue

            # 3.4 阶段转换时更新策略（非每步都重规划，节省 LLM 调用）
            # 使用 Phase 枚举直接比较，避免字符串误比对
            plan_phase_str = plan.get("strategy", {}).get("current_phase", "")
            try:
                plan_phase = Phase(plan_phase_str) if plan_phase_str else planner.current_phase
            except ValueError:
                plan_phase = planner.current_phase
            if planner.current_phase != plan_phase:
                logger.info(
                    f"[规划执行] 阶段转换 {plan_phase.value} → {planner.current_phase.value}，更新策略"
                )
                relevant_experience = self._get_cached_experience(collected_info)
                plan = await planner.plan(
                    collected_info=collected_info,
                    exam_results=exam_results,
                    chat_history=chat_history,
                    relevant_experience=relevant_experience,
                )
                _sync_plan(plan)

            # 3.5 安全检查：如果循环结束仍未提交方案，强制提交
            if step == planner.max_total_actions - 1:
                logger.warning("[规划执行] 达到最大步骤数，强制提交诊疗方案")
                final_result = await self._prescribe(
                    patient_id, collected_info, exam_results,
                    chat_history, relevant_experience,
                )
                planner.current_phase = Phase.COMPLETED
                self._last_collected_info = collected_info
                self._last_exam_results = exam_results
                self.memory_manager.finish_case(patient_id)
                return final_result

        # 兜底返回
        self._last_collected_info = collected_info
        self._last_exam_results = exam_results
        final_result = await self._prescribe(
            patient_id, collected_info, exam_results,
            chat_history, relevant_experience,
        )
        self.memory_manager.finish_case(patient_id)
        return final_result

    async def _execute_ask_patient(
        self,
        patient_id: str,
        chat_history: List[Dict[str, str]],
        collected_info: Dict[str, Any],
        relevant_experience: List[Dict[str, Any]],
        target: str,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        """执行单次问诊操作。

        Args:
            patient_id: 患者 ID
            chat_history: 对话历史
            collected_info: 已收集信息
            relevant_experience: 相关经验
            target: 问诊目标（来自规划器）
            reason: 问诊理由（来自规划器）

        Returns:
            更新后的患者信息，失败返回 None
        """
        # 基于症状检索相关经验（带缓存）
        cached_exp = self._get_cached_experience(collected_info)
        if cached_exp:
            relevant_experience = cached_exp

        # 思考：生成鉴别诊断，判断信息是否足够
        thinking = await self._think(
            collected_info, {}, chat_history, "inquiry", relevant_experience
        )
        _cands = None
        if thinking and isinstance(thinking, dict):
            _cands = thinking.get("differential_diagnosis") or thinking.get("candidate_diseases")
        inquiry_strategy = self.inquiry_agent.recommend(
            collected_info=collected_info,
            candidate_diseases=_cands if isinstance(_cands, list) else None,
        )

        # 判断信息是否已足够
        if thinking and "is_sufficient" in thinking:
            if thinking.get("is_sufficient"):
                logger.info("[问诊] 思考判断信息已足够，跳过追问")
                return None

        # 构建追问 prompt（注入思考结果和规划目标）
        follow_up_prompt = self.prompt.build_follow_up_prompt(
            collected_info=collected_info,
            chat_history=chat_history,
            relevant_experience=relevant_experience,
            thinking=thinking,
        )
        messages = [
            {"role": "system", "content": follow_up_prompt},
            {
                "role": "user",
                "content": (
                    f"请针对以下目标生成追问问题：{target}。理由：{reason}。"
                    f"优先覆盖这些关键追问：{inquiry_strategy.get('questions', [])}。"
                    f"注意排查红旗信号：{inquiry_strategy.get('red_flags', [])}。"
                    "如果信息已足够，请返回空字符串。"
                ),
            },
        ]

        question = await self._llm_chat(
            messages,
            temperature=0.5,
            purpose="inquiry_question",
        )
        if not question or question.strip() == "":
            logger.info("[问诊] LLM 判断无需追问")
            return None

        logger.info(f"[问诊] 追问: {question}")

        answer = await self.actions.ask_patient(
            patient_id=patient_id,
            input_data={"question": question, "chat_history": chat_history},
        )

        chat_history.append({"from": "doctor", "text": question})
        chat_history.append({"from": "patient", "text": answer})

        # 提取结构化信息
        updated_info = await self._extract_patient_info(answer, collected_info)
        logger.info(f"[问诊] 追问回复: {answer[:200]}...")

        return updated_info

    def _pre_exam_judge_payload(
        self,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        thinking: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            evidence = self._normalize_with_exam_recovery(collected_info, exam_results)
            llm_result: Dict[str, Any] = {}
            if isinstance(thinking, dict):
                candidates = (
                    thinking.get("differential_diagnosis")
                    or thinking.get("candidate_diseases")
                    or thinking.get("diagnosis_candidates")
                    or []
                )
                if isinstance(candidates, list):
                    llm_result["diagnosis_candidates"] = candidates
            decision = self.diagnosis_engine.decide(
                self._diagnosis_input_with_runtime_state(llm_result),
                [],
                evidence,
            )
            payload = dict(getattr(decision, "judge_decision", None) or {})
            if payload:
                payload["stage"] = "pre_exam_judge"
                payload["pre_exam_runtime_state_version"] = int(self._diagnostic_state_version or 0)
                payload["pre_exam_engine_claim_state_version"] = int(
                    getattr(decision, "claim_state_version", 0) or 0
                )
                payload["pre_exam_claim_ledger_size"] = len(
                    normalize_ledger(self._claim_resolution_ledger)
                )
                active_gaps = [
                    gap for gap in payload.get("active_evidence_gaps", []) or []
                    if isinstance(gap, dict)
                ]
                payload["pre_exam_gap_count"] = len(active_gaps)
                hydrated = [
                    gap for gap in active_gaps
                    if gap.get("claim_resolutions") or gap.get("remaining_claims")
                ]
                payload["pre_exam_hydrated_gap_count"] = len(hydrated)
                payload["pre_exam_remaining_claims_by_gap"] = {
                    str(gap.get("gap_id") or ""): list(gap.get("remaining_claims") or [])
                    for gap in hydrated
                    if str(gap.get("gap_id") or "")
                }
                payload["pre_exam_stale_claim_state_detected"] = bool(
                    int(self._claim_state_version or 0) > 0
                    and (
                        int(getattr(decision, "claim_state_version", 0) or 0)
                        < int(self._claim_state_version or 0)
                        or len(normalize_ledger(self._claim_resolution_ledger)) > 0
                        and not hydrated
                    )
                )
                self._mark_clinical_transition(
                    "pre_exam_runtime_state_injected",
                    "pre_exam_judge",
                    pre_exam_claim_ledger_size=payload["pre_exam_claim_ledger_size"],
                    pre_exam_gap_count=payload["pre_exam_gap_count"],
                    pre_exam_hydrated_gap_count=payload["pre_exam_hydrated_gap_count"],
                    pre_exam_stale_claim_state_detected=payload[
                        "pre_exam_stale_claim_state_detected"
                    ],
                )
            return payload
        except Exception as exc:
            logger.debug("[Judge] pre-exam judge skipped: %s", exc)
            return {}

    def _collect_runtime_audit(self) -> Dict[str, Any]:
        audit = dict(super()._collect_runtime_audit())
        audit.update(self._collect_clinical_runtime_audit(audit))
        return audit

    def _collect_clinical_runtime_audit(
        self,
        runtime_audit: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ledger = normalize_ledger(getattr(self, "_claim_resolution_ledger", {}) or {})
        parses = [
            self._compact_targeted_parse(item)
            for item in getattr(self, "_targeted_exam_result_parses", []) or []
            if isinstance(item, dict)
        ]
        observations = [
            self._compact_targeted_observation(item)
            for item in getattr(self, "_targeted_exam_observations", []) or []
        ]
        claim_events = [
            self._compact_claim_match_event(item)
            for item in getattr(self, "_claim_match_events", []) or []
            if isinstance(item, dict)
        ]
        update_audit = [
            self._compact_claim_resolution_update(item)
            for item in getattr(self, "_claim_resolution_update_audit", []) or []
            if isinstance(item, dict)
        ]
        gap_state = self._clinical_gap_state(parses)
        anchor_state = self._clinical_anchor_state()
        eligibility_state = self._clinical_eligibility_state()
        payload = {
            "evidence_version": int(getattr(self, "_diagnostic_state_version", 0) or 0),
            "exam_result_intent_bindings": list(
                getattr(self, "_exam_result_intent_bindings", []) or []
            ),
            "targeted_exam_result_parses": parses,
            "exam_result_applicability": self._clinical_exam_result_applicability(parses),
            "targeted_exam_observations": observations,
            "clinical_admission_audit": list(
                getattr(self, "_clinical_admission_audit", []) or []
            ),
            "candidate_claim_contract_views": list(
                getattr(self, "_candidate_claim_contract_views", []) or []
            ),
            "claim_state_materialization_audit": list(
                getattr(self, "_claim_state_materialization_audit", []) or []
            ),
            "claim_state_invariant_audit": list(
                getattr(self, "_claim_state_invariant_audit", []) or []
            ),
            "claim_match_events": claim_events,
            "claim_resolution_ledger": ledger,
            "claim_resolution_update_audit": update_audit,
            "claim_state_version": int(getattr(self, "_claim_state_version", 0) or 0),
            "diagnostic_state_version": int(
                getattr(self, "_diagnostic_state_version", 0) or 0
            ),
            "gap_state": gap_state,
            "anchor_state": anchor_state,
            "eligibility_state": eligibility_state,
            "last_completed_stage": str(
                (getattr(self, "_last_successful_clinical_transition", {}) or {}).get(
                    "stage"
                )
                or ""
            ),
            "last_successful_clinical_transition": dict(
                getattr(self, "_last_successful_clinical_transition", {}) or {}
            ),
            "clinical_transition_trace": list(
                getattr(self, "_clinical_transition_trace", []) or []
            ),
        }
        payload["failure_stage"] = self._infer_clinical_failure_stage(
            payload,
            runtime_audit or {},
        )
        return payload

    def _mark_clinical_transition(
        self,
        name: str,
        stage: str,
        **payload: Any,
    ) -> None:
        event = {
            "name": str(name or ""),
            "stage": str(stage or ""),
            "timestamp": round(time.time(), 3),
            "claim_state_version": int(getattr(self, "_claim_state_version", 0) or 0),
            "diagnostic_state_version": int(
                getattr(self, "_diagnostic_state_version", 0) or 0
            ),
            "counts": self._clinical_transition_counts(),
        }
        for key, value in (payload or {}).items():
            event[key] = self._compact_runtime_value(value)
        trace = list(getattr(self, "_clinical_transition_trace", []) or [])
        trace.append(event)
        self._clinical_transition_trace = trace[-200:]
        self._last_successful_clinical_transition = event

    def _clinical_transition_counts(self) -> Dict[str, int]:
        return {
            "exam_result_intent_bindings": len(
                getattr(self, "_exam_result_intent_bindings", []) or []
            ),
            "targeted_exam_result_parses": len(
                getattr(self, "_targeted_exam_result_parses", []) or []
            ),
            "targeted_exam_observations": len(
                getattr(self, "_targeted_exam_observations", []) or []
            ),
            "claim_match_events": len(getattr(self, "_claim_match_events", []) or []),
            "claim_resolution_ledger": len(
                normalize_ledger(getattr(self, "_claim_resolution_ledger", {}) or {})
            ),
        }

    @staticmethod
    def _compact_runtime_value(value: Any) -> Any:
        if isinstance(value, dict):
            compact: Dict[str, Any] = {}
            for key, item in value.items():
                text_key = str(key)
                if text_key in {"raw_result", "raw_response", "prompt", "source_text"}:
                    compact[f"{text_key}_chars"] = len(str(item or ""))
                    continue
                compact[text_key] = MyDoctorAgent._compact_runtime_value(item)
            return compact
        if isinstance(value, list):
            return [MyDoctorAgent._compact_runtime_value(item) for item in value[:30]]
        if isinstance(value, str):
            return value if len(value) <= 240 else value[:240] + "...[truncated]"
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _compact_targeted_parse(item: Dict[str, Any]) -> Dict[str, Any]:
        observations = []
        for obs in item.get("observations", []) or []:
            if isinstance(obs, dict):
                observations.append(
                    {
                        "finding": obs.get("finding"),
                        "polarity": obs.get("polarity"),
                        "source_exam": obs.get("source_exam"),
                        "order_id": obs.get("order_id"),
                        "target_gap_ids": list(obs.get("target_gap_ids") or []),
                        "entity_id": obs.get("entity_id"),
                    }
                )
        return {
            "binding_id": item.get("binding_id"),
            "order_id": item.get("order_id"),
            "entity_id": item.get("entity_id"),
            "target_gap_ids": list(item.get("target_gap_ids") or []),
            "status": item.get("status"),
            "binding_status": item.get("binding_status"),
            "stage": item.get("stage"),
            "ordered_exam": item.get("ordered_exam"),
            "actual_result_exam": item.get("actual_result_exam"),
            "binding_source": item.get("binding_source"),
            "applicability_reason": item.get("applicability_reason"),
            "gap_closure_assessment": item.get("gap_closure_assessment"),
            "gap_resolution_status": item.get("gap_resolution_status"),
            "claim_matches": list(item.get("claim_matches") or []),
            "observations": observations,
            "observation_count": len(observations),
            "atomic_observation_count": len(item.get("atomic_observations") or []),
            "relation_observation_count": len(item.get("relation_observations") or []),
            "claim_resolution_update": MyDoctorAgent._compact_runtime_value(
                item.get("claim_resolution_update") or {}
            ),
        }

    @staticmethod
    def _compact_targeted_observation(item: Any) -> Dict[str, Any]:
        payload = item.to_dict() if hasattr(item, "to_dict") else dict(item or {})
        return {
            "finding": payload.get("finding"),
            "polarity": payload.get("polarity"),
            "source": payload.get("source"),
            "source_exam": payload.get("source_exam"),
            "order_id": payload.get("order_id"),
            "target_gap_ids": list(payload.get("target_gap_ids") or []),
            "entity_id": payload.get("entity_id"),
            "verification_method": payload.get("verification_method"),
            "source_refs": list(payload.get("source_refs") or []),
        }

    @staticmethod
    def _compact_claim_match_event(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "event_id": item.get("event_id"),
            "candidate_id": item.get("candidate_id"),
            "entity_id": item.get("entity_id"),
            "claim_id": item.get("claim_id"),
            "contract_id": item.get("contract_id"),
            "contract_version": item.get("contract_version"),
            "match_status": item.get("match_status"),
            "supporting_evidence_refs": list(item.get("supporting_evidence_refs") or []),
            "contradicting_evidence_refs": list(
                item.get("contradicting_evidence_refs") or []
            ),
            "source_type": item.get("source_type"),
            "source_route_id": item.get("source_route_id"),
            "source_exam": item.get("source_exam"),
            "evidence_version": item.get("evidence_version"),
        }

    @staticmethod
    def _compact_claim_resolution_update(item: Dict[str, Any]) -> Dict[str, Any]:
        before = item.get("resolution_before") or {}
        after = item.get("resolution_after") or {}
        return {
            "event_id": item.get("event_id"),
            "claim_key": item.get("claim_key"),
            "source_route": item.get("source_route"),
            "incoming_match_status": item.get("incoming_match_status"),
            "resolution_before_status": before.get("resolution_status"),
            "resolution_after_status": after.get("resolution_status"),
            "supporting_refs_added": list(item.get("supporting_refs_added") or []),
            "contradicting_refs_added": list(item.get("contradicting_refs_added") or []),
            "merge_decision": item.get("merge_decision"),
            "idempotent_replay": bool(item.get("idempotent_replay")),
            "conflict_created": bool(item.get("conflict_created")),
            "claim_state_version_delta": int(item.get("claim_state_version_delta") or 0),
            "route_attempt_state_delta": int(item.get("route_attempt_state_delta") or 0),
        }

    @staticmethod
    def _clinical_exam_result_applicability(parses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for item in parses or []:
            if not item.get("binding_id") and not item.get("actual_result_exam"):
                continue
            result.append(
                {
                    "binding_id": item.get("binding_id"),
                    "entity_id": item.get("entity_id"),
                    "target_gap_ids": list(item.get("target_gap_ids") or []),
                    "actual_result_exam": item.get("actual_result_exam"),
                    "binding_source": item.get("binding_source"),
                    "applicability_reason": item.get("applicability_reason"),
                    "claim_match_count": len(item.get("claim_matches") or []),
                    "supported_claims": [
                        match.get("target_claim")
                        for match in item.get("claim_matches", []) or []
                        if isinstance(match, dict)
                        and match.get("claim_status") == "SUPPORTED"
                    ],
                }
            )
        return result

    @staticmethod
    def _clinical_gap_state(parses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for item in parses or []:
            update = item.get("claim_resolution_update") or {}
            gap_eval = update.get("gap_closure_evaluation") or {}
            if not gap_eval:
                continue
            result.append(
                {
                    "gap_id": gap_eval.get("gap_id"),
                    "entity_id": gap_eval.get("entity_id"),
                    "contract_id": gap_eval.get("contract_id"),
                    "contract_version": gap_eval.get("contract_version"),
                    "gap_closure_level": gap_eval.get("gap_closure_level"),
                    "resolved_claims": list(gap_eval.get("resolved_claims") or []),
                    "remaining_claims": list(gap_eval.get("remaining_claims") or []),
                    "contradicted_claims": list(gap_eval.get("contradicted_claims") or []),
                    "conflicted_claims": list(gap_eval.get("conflicted_claims") or []),
                }
            )
        return result

    def _clinical_anchor_state(self) -> List[Dict[str, Any]]:
        audit = getattr(self, "_last_diagnosis_audit", {}) or {}
        decision = audit.get("diagnosis_decision") or {}
        candidates = decision.get("candidates") if isinstance(decision, dict) else []
        result = []
        for candidate in candidates or []:
            if not isinstance(candidate, dict):
                continue
            anchor = (
                candidate.get("claim_anchor_evaluation")
                or candidate.get("anchor_evaluation")
                or {}
            )
            if not anchor:
                continue
            result.append(
                {
                    "diagnosis": candidate.get("diagnosis"),
                    "entity_id": candidate.get("entity_id"),
                    "anchor_status": anchor.get("anchor_status_after")
                    or candidate.get("anchor_status"),
                    "required_claims": list(anchor.get("required_claims") or []),
                    "satisfied_claims": list(anchor.get("satisfied_claims") or []),
                    "unresolved_claims": list(anchor.get("unresolved_claims") or []),
                }
            )
        return result

    def _clinical_eligibility_state(self) -> List[Dict[str, Any]]:
        audit = getattr(self, "_last_diagnosis_audit", {}) or {}
        decision = audit.get("diagnosis_decision") or {}
        candidates = decision.get("candidates") if isinstance(decision, dict) else []
        result = []
        for candidate in candidates or []:
            if not isinstance(candidate, dict):
                continue
            status = (
                candidate.get("eligibility_status")
                or candidate.get("primary_eligibility")
                or candidate.get("primary_status")
                or ""
            )
            if not status:
                continue
            result.append(
                {
                    "diagnosis": candidate.get("diagnosis"),
                    "entity_id": candidate.get("entity_id"),
                    "eligibility_status": status,
                    "rank": candidate.get("rank"),
                    "score": candidate.get("score"),
                }
            )
        return result

    @staticmethod
    def _infer_clinical_failure_stage(
        clinical: Dict[str, Any],
        runtime_audit: Dict[str, Any],
    ) -> str:
        tool_summary = runtime_audit.get("tool_contract_summary") or {}
        if int(tool_summary.get("retry_exhausted_calls") or 0) > 0 or int(
            tool_summary.get("terminal_failure_logical_calls") or 0
        ) > 0:
            return "tool_backend"
        transitions = list(clinical.get("clinical_transition_trace") or [])
        llm_summary = runtime_audit.get("llm_contract_summary") or {}
        if int(llm_summary.get("fallback_calls") or 0) > 0 and not transitions:
            return "llm_contract"
        parses = list(clinical.get("targeted_exam_result_parses") or [])
        events = list(clinical.get("claim_match_events") or [])
        ledger = normalize_ledger(clinical.get("claim_resolution_ledger") or {})
        if clinical.get("exam_result_intent_bindings") and not parses:
            return "exam_result_parsing"
        if parses and not any(item.get("binding_source") for item in parses):
            return "claim_contract_binding"
        if any((item.get("observation_count") or 0) > 0 for item in parses) and not events:
            return "claim_matching"
        resolvable = [
            item
            for item in events
            if item.get("match_status") in {"SUPPORTED", "CONTRADICTED"}
        ]
        if resolvable and not ledger:
            return "claim_resolution_writeback"
        if int(clinical.get("claim_state_version") or 0) > 0:
            transition_names = {
                str(item.get("name") or "") for item in transitions if isinstance(item, dict)
            }
            if not (
                "post_result_reevaluation_started" in transition_names
                or "diagnosis_decision_started" in transition_names
                or "diagnosis_decision_completed" in transition_names
            ):
                return "post_result_re_evaluation"
        if clinical.get("anchor_state") or clinical.get("eligibility_state"):
            return "diagnosis_arbitration"
        if int(llm_summary.get("fallback_calls") or 0) > 0:
            return "llm_contract"
        return "unknown_runtime_failure"

    def _strategy_order_items(
        self,
        strategy: Dict[str, Any],
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]],
        existing_results: Dict[str, Any],
        max_items: Optional[int],
        add_strong_verification: bool = False,
    ) -> List[str]:
        items = list(strategy.get("items") or [])
        details = [
            item
            for item in strategy.get("exam_authorization_details", []) or []
            if isinstance(item, dict)
        ]
        has_reserved_gap = any(
            str(item.get("exam_source") or "") == "deferred_gap_closure_exam"
            and (
                bool(item.get("priority_override"))
                or str(item.get("priority_bucket") or "")
                == "high_value_deferred_gap_closure"
            )
            for item in details
        )
        if not (strategy.get("differential_driven") or has_reserved_gap):
            detail_by_exam = self._exam_authorization_detail_by_exam(details)
            prepared = self.exam_agent.prepare_order_items(
                items,
                collected_info=collected_info,
                candidate_diseases=candidate_diseases,
                existing_results=existing_results,
                max_items=max_items,
                add_strong_verification=add_strong_verification,
            )
            return self._filter_repeat_unauthorized_exams(
                prepared,
                existing_results=existing_results,
                strategy=strategy,
                detail_by_exam=detail_by_exam,
            )

        detail_by_exam = self._exam_authorization_detail_by_exam(details)
        existing_valid, _ = self.knowledge.normalize_examinations(
            list((existing_results or {}).keys())
        )
        existing_set = set((existing_results or {}).keys()) | set(existing_valid)
        prepared: List[str] = []
        for item in items:
            exam = str(item or "").strip()
            if not exam or exam in prepared:
                continue
            detail = detail_by_exam.get(exam, {})
            authorization = self._authorize_exam_route(
                exam,
                existing_results,
                authorization_detail=detail,
            )
            if not authorization.get("authorized"):
                self._record_exam_repeat_audit(
                    strategy,
                    exam=exam,
                    blocked=True,
                    reason=str((authorization.get("reason_codes") or [""])[0]),
                    detail={**detail, **authorization},
                )
                continue
            if detail:
                self._record_exam_repeat_audit(
                    strategy,
                    exam=exam,
                    blocked=False,
                    reason=str((authorization.get("reason_codes") or ["EXAM_ROUTE_AUTHORIZED"])[0]),
                    detail={**detail, **authorization},
                )
            prepared.append(exam)
        if max_items is None or len(prepared) <= max_items:
            return prepared
        limit = max(0, int(max_items))
        reserved = [
            item
            for item in prepared
            if str(detail_by_exam.get(item, {}).get("exam_source") or "")
            == "deferred_gap_closure_exam"
            and (
                bool(detail_by_exam.get(item, {}).get("priority_override"))
                or str(detail_by_exam.get(item, {}).get("priority_bucket") or "")
                == "high_value_deferred_gap_closure"
            )
        ]
        urgent = [
            item
            for item in prepared
            if str(detail_by_exam.get(item, {}).get("priority_bucket") or "")
            == "urgent_safety"
        ]
        selected: List[str] = []
        for item in urgent + reserved + prepared:
            if item not in selected:
                selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def _filter_repeat_unauthorized_exams(
        self,
        items: List[str],
        *,
        existing_results: Optional[Dict[str, Any]],
        strategy: Optional[Dict[str, Any]] = None,
        detail_by_exam: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[str]:
        filtered: List[str] = []
        detail_by_exam = detail_by_exam or {}
        for item in items or []:
            exam = str(item or "").strip()
            if not exam or exam in filtered:
                continue
            detail = self._authorization_detail_for_order_exam(exam, detail_by_exam)
            authorization = self._authorize_exam_route(
                exam,
                existing_results,
                authorization_detail=detail,
            )
            if not authorization.get("authorized"):
                self._record_exam_repeat_audit(
                    strategy,
                    exam=exam,
                    blocked=True,
                    reason=str((authorization.get("reason_codes") or [""])[0]),
                    detail={**detail, **authorization},
                )
                continue
            if detail:
                self._record_exam_repeat_audit(
                    strategy,
                    exam=exam,
                    blocked=False,
                    reason=str((authorization.get("reason_codes") or ["EXAM_ROUTE_AUTHORIZED"])[0]),
                    detail={**detail, **authorization},
                )
            filtered.append(exam)
        return filtered

    def _exam_authorization_detail_by_exam(
        self,
        details: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for detail in details or []:
            if not isinstance(detail, dict):
                continue
            keys = [
                detail.get("exam"),
                detail.get("requested_exam"),
                detail.get("resolved_exam"),
            ]
            for key in keys:
                text = str(key or "").strip()
                if text and text not in result:
                    result[text] = detail
        return result

    def _authorization_detail_for_order_exam(
        self,
        exam: str,
        detail_by_exam: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if exam in detail_by_exam:
            return detail_by_exam[exam]
        normalized, _ = self.knowledge.normalize_examinations([exam])
        for name in [exam] + list(normalized or []):
            if name in detail_by_exam:
                return detail_by_exam[name]
        requested_family = self._exam_repeat_family(exam)
        if requested_family:
            for key, detail in detail_by_exam.items():
                if self._exam_repeat_family(key) == requested_family:
                    return detail
        return {}

    def _completed_exam_duplicate_reason(
        self,
        exam: Any,
        existing_results: Optional[Dict[str, Any]],
        *,
        authorization_detail: Optional[Dict[str, Any]] = None,
    ) -> str:
        authorization = self._authorize_exam_route(
            exam,
            existing_results,
            authorization_detail=authorization_detail,
        )
        if authorization.get("authorized"):
            return ""
        reasons = list(authorization.get("reason_codes") or [])
        return str(reasons[0]) if reasons else ""

    def _authorize_exam_route(
        self,
        exam: Any,
        existing_results: Optional[Dict[str, Any]],
        *,
        authorization_detail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = str(exam or "").strip()
        detail = authorization_detail or {}
        result: Dict[str, Any] = {
            "exam": text,
            "authorized": True,
            "reason_codes": [],
            "repeat_requested": bool(detail.get("repeat_requested")),
            "repeat_authorized": bool(detail.get("repeat_authorized")),
            "repeat_reason_codes": list(detail.get("repeat_reason_codes") or []),
            "exam_source": str(detail.get("exam_source") or "generic_workup"),
            "target_gap_ids": list(detail.get("target_gaps") or []),
            "target_claim_ids": list(detail.get("target_claims") or []),
            "route_target_claim_ids": list(detail.get("route_target_claims") or []),
            "closure_route_ids": [
                str(route.get("route_id") or route.get("id") or "")
                for route in detail.get("closure_routes", []) or []
                if isinstance(route, dict)
            ],
            "source_evidence_version": detail.get("source_evidence_version"),
            "prior_result_state": "none",
        }
        if not text:
            result["authorized"] = False
            result["reason_codes"] = ["EMPTY_EXAM"]
            return result
        if not existing_results:
            result["reason_codes"] = ["NO_PRIOR_RESULT"]
            return result
        if self._detail_has_valid_repeat_authorization(detail):
            result["reason_codes"] = list(
                dict.fromkeys(
                    list(result["repeat_reason_codes"]) or ["EXPLICIT_REPEAT_AUTHORIZED"]
                )
            )
            return result
        normalized, _ = self.knowledge.normalize_examinations([text])
        exam_names = set([text] + list(normalized or []))
        existing_valid, _ = self.knowledge.normalize_examinations(
            list((existing_results or {}).keys())
        )
        existing_names = set((existing_results or {}).keys()) | set(existing_valid)
        if exam_names & existing_names:
            result["authorized"] = False
            result["prior_result_state"] = "completed_same_exam"
            result["reason_codes"] = [
                self._completed_route_reason(detail) or "COMPLETED_EXAM_DUPLICATE"
            ]
            return result
        requested_family = self._exam_repeat_family(text)
        if not requested_family:
            result["reason_codes"] = ["NO_PRIOR_EQUIVALENT_RESULT"]
            return result
        for existing in existing_names:
            if self._exam_repeat_family(existing) == requested_family:
                result["authorized"] = False
                result["prior_result_state"] = "completed_equivalent_exam_family"
                result["reason_codes"] = [
                    self._completed_route_reason(detail)
                    or "GENERIC_WORKUP_DUPLICATE_BLOCKED"
                ]
                return result
        result["reason_codes"] = ["NO_PRIOR_EQUIVALENT_RESULT"]
        return result

    @staticmethod
    def _detail_has_valid_repeat_authorization(detail: Dict[str, Any]) -> bool:
        if not detail:
            return False
        if bool(detail.get("repeat_authorized")):
            return True
        if bool(detail.get("repeat_requested")) and str(detail.get("repeat_authorized")).lower() == "true":
            return True
        reasons = {
            str(item or "").strip()
            for item in detail.get("repeat_reason_codes", []) or []
            if str(item or "").strip()
        }
        allowed = {
            "NEW_TARGET_CLAIM",
            "PRIOR_RESULT_INADEQUATE",
            "PRIOR_TOOL_FAILURE",
            "NEW_MATERIAL_EVIDENCE",
            "LONGITUDINAL_MONITORING",
            "EVIDENCE_STALE",
            "CONTRADICTION_RESOLUTION",
            "TECHNICAL_FAILURE_RETRY",
        }
        return bool(reasons & allowed)

    @staticmethod
    def _completed_route_reason(detail: Dict[str, Any]) -> str:
        route_claims = {
            str(item or "").strip()
            for item in detail.get("route_target_claims", []) or []
            if str(item or "").strip()
        }
        all_claims = {
            str(item or "").strip()
            for item in detail.get("target_claims", []) or []
            if str(item or "").strip()
        }
        if route_claims:
            return "CLAIM_ROUTE_ALREADY_RESOLVED"
        if all_claims:
            return "COMPLETED_EXAM_DUPLICATE"
        if str(detail.get("exam_source") or "") == "generic_workup":
            return "GENERIC_WORKUP_DUPLICATE_BLOCKED"
        return ""

    @staticmethod
    def _exam_repeat_family(exam: Any) -> str:
        compact = "".join(ch for ch in str(exam or "").lower() if ch.isalnum())
        if not compact:
            return ""
        if "cta" in compact:
            return "cta"
        if "ct" in compact:
            return "ct"
        if "cxr" in compact:
            return "cxr"
        if "xray" in compact or "x线" in compact:
            return "xray"
        return ""

    @staticmethod
    def _record_exam_repeat_audit(
        strategy: Optional[Dict[str, Any]],
        *,
        exam: str,
        blocked: bool,
        reason: str,
        detail: Dict[str, Any],
    ) -> None:
        if strategy is None:
            return
        audit = strategy.setdefault("exam_repeat_authorization_audit", [])
        audit.append(
            {
                "exam": exam,
                "blocked": bool(blocked),
                "repeat_authorized": not blocked,
                "reason_codes": list(
                    dict.fromkeys(
                        ([reason] if reason else [])
                        + list(detail.get("repeat_reason_codes") or [])
                    )
                ),
                "exam_source": str(detail.get("exam_source") or "generic_workup"),
                "target_gap_ids": list(detail.get("target_gaps") or []),
                "target_claim_ids": list(detail.get("target_claims") or []),
                "route_target_claim_ids": list(detail.get("route_target_claims") or []),
                "closure_route_ids": [
                    str(route.get("route_id") or route.get("id") or "")
                    for route in detail.get("closure_routes", []) or []
                    if isinstance(route, dict)
                ],
                "source_evidence_version": detail.get("source_evidence_version"),
                "prior_result_state": str(detail.get("prior_result_state") or ""),
            }
        )

    def _record_exam_route_authorization_summary(
        self,
        *,
        stage: str,
        strategy: Dict[str, Any],
        authorized_items: Optional[List[str]] = None,
        target: str = "",
    ) -> None:
        audit = [
            item
            for item in strategy.get("exam_repeat_authorization_audit", []) or []
            if isinstance(item, dict)
        ]
        if not audit:
            return
        summary = {
            "stage": stage,
            "target": target,
            "authorized_items": list(authorized_items or []),
            "exam_repeat_authorization_audit": audit,
            "exam_route_authorization_blocked_count": sum(
                1 for item in audit if item.get("blocked")
            ),
            "exam_route_authorization_allowed_count": sum(
                1 for item in audit if not item.get("blocked")
            ),
        }
        if self._last_exam_authorization and str(
            self._last_exam_authorization[-1].get("stage") or ""
        ) == stage:
            self._last_exam_authorization[-1].update(summary)
        else:
            self._last_exam_authorization.append(summary)

    def _normalize_with_exam_recovery(
        self,
        collected_info: Optional[Dict[str, Any]],
        exam_results: Optional[Dict[str, Any]],
        raw_case_text: str = "",
    ) -> EvidenceBundle:
        base = self.clinical_normalizer.normalize(
            collected_info,
            exam_results,
            raw_case_text=raw_case_text,
        )
        if not self._targeted_exam_observations:
            return base
        observations = self.evidence_compiler.merge_observations(
            list(base.observations),
            list(self._targeted_exam_observations),
        )
        return self._append_exam_recovery_patterns(
            EvidenceBundle(self.clinical_normalizer._finalize_observations(observations))
        )

    def _compile_evidence_with_exam_recovery(
        self,
        collected_info: Optional[Dict[str, Any]],
        exam_results: Optional[Dict[str, Any]],
        diagnosis_result: Optional[Dict[str, Any]] = None,
        raw_case_text: str = "",
    ) -> EvidenceBundle:
        bundle = self.evidence_compiler.compile(
            collected_info,
            exam_results,
            diagnosis_result,
            raw_case_text=raw_case_text,
            additional_observations=list(self._targeted_exam_observations),
        )
        return self._append_exam_recovery_patterns(bundle)

    def _append_exam_recovery_patterns(self, evidence: EvidenceBundle) -> EvidenceBundle:
        if not self._targeted_exam_observations:
            return evidence
        derived = self.exam_recovery_pattern_compiler.compile([], evidence)
        if not derived:
            return evidence
        observations = self.evidence_compiler.merge_observations(
            list(evidence.observations),
            list(derived),
        )
        return EvidenceBundle(self.clinical_normalizer._finalize_observations(observations))

    def _record_targeted_exam_result_recovery(
        self,
        *,
        patient_id: str,
        stage: str,
        ordered_items: List[str],
        new_results: Dict[str, Any],
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not new_results:
            return
        admitted_claim_views = self._materialize_admitted_candidate_claim_states()
        details = self._exam_authorization_details_for_result_recovery(strategy)
        pairs = self._match_ordered_items_to_results(ordered_items, new_results)
        observations_before = len(self._targeted_exam_observations)
        for index, (ordered_exam, actual_exam, raw_result) in enumerate(pairs, start=1):
            self._mark_clinical_transition(
                "exam_result_received",
                "exam_result_recovery",
                ordered_exam=ordered_exam,
                actual_result_exam=actual_exam,
                result_chars=len(str(raw_result or "")),
            )
            applicable_details = self._authorization_details_for_exam_result(
                ordered_exam,
                actual_exam,
                list(details)
                + self._candidate_claim_contract_details_for_exam_result(
                    ordered_exam,
                    actual_exam,
                    admitted_claim_views,
                ),
            )
            if not applicable_details:
                self._targeted_exam_result_parses.append(
                    {
                        "status": "unbound",
                        "ordered_exam": ordered_exam,
                        "actual_result_exam": actual_exam,
                        "gap_closure_allowed": False,
                        "stage": stage,
                    }
                )
                self._mark_clinical_transition(
                    "claim_contracts_bound",
                    "exam_result_recovery",
                    ordered_exam=ordered_exam,
                    actual_result_exam=actual_exam,
                    applicable_contract_count=0,
                )
                continue
            self._mark_clinical_transition(
                "claim_contracts_bound",
                "exam_result_recovery",
                ordered_exam=ordered_exam,
                actual_result_exam=actual_exam,
                applicable_contract_count=len(applicable_details),
                entity_ids=list(
                    dict.fromkeys(
                        str(item.get("entity_id") or "")
                        for item in applicable_details
                        if str(item.get("entity_id") or "")
                    )
                ),
                target_gap_ids=list(
                    dict.fromkeys(
                        str(gap_id or "")
                        for item in applicable_details
                        for gap_id in item.get("target_gaps", []) or []
                        if str(gap_id or "")
                    )
                ),
            )
            parse_detail = self._neutral_parse_detail_for_applicable_contracts(
                applicable_details
            )
            parse_binding = binding_from_authorization_detail(
                detail=parse_detail,
                requested_exam=ordered_exam,
                actual_result_exam=actual_exam,
                patient_id=patient_id,
                stage=stage,
                order_index=len(self._exam_result_intent_bindings) + index,
            )
            neutral_parsed = self.targeted_exam_result_parser.parse(raw_result, parse_binding)
            self._mark_clinical_transition(
                "observations_parsed",
                "exam_result_recovery",
                ordered_exam=ordered_exam,
                actual_result_exam=actual_exam,
                observation_count=len(neutral_parsed.observations or []),
                atomic_observation_count=len(neutral_parsed.atomic_observations or []),
                relation_observation_count=len(neutral_parsed.relation_observations or []),
                findings=list(
                    dict.fromkeys(
                        obs.finding
                        for obs in neutral_parsed.observations or []
                        if getattr(obs, "finding", "")
                    )
                ),
            )
            result_claim_delta = 0
            result_route_attempt_delta = 0
            observations_recorded = False
            for detail_index, detail in enumerate(applicable_details, start=1):
                binding = binding_from_authorization_detail(
                    detail=detail,
                    requested_exam=ordered_exam,
                    actual_result_exam=actual_exam,
                    patient_id=patient_id,
                    stage=stage,
                    order_index=(
                        len(self._exam_result_intent_bindings) + index * 100 + detail_index
                    ),
                )
                binding_payload = binding.to_dict()
                binding_payload["binding_source"] = str(
                    detail.get("_result_binding_source") or "RESULT_APPLICABILITY"
                )
                self._exam_result_intent_bindings.append(binding_payload)
                if detail is parse_detail:
                    parsed = neutral_parsed
                else:
                    parsed = self.targeted_exam_result_parser.rematch_claims_for_binding(
                        neutral_parsed,
                        binding,
                    )
                payload = parsed.to_dict()
                payload["stage"] = stage
                payload["ordered_exam"] = ordered_exam
                payload["binding_source"] = binding_payload["binding_source"]
                payload["applicability_reason"] = str(
                    detail.get("_applicability_reason") or ""
                )
                payload["gap_closure_allowed"] = parsed.gap_closure_assessment in {
                    "positive_closed",
                    "negative_closed",
                }
                claim_matches = list(payload.get("claim_matches") or [])
                self._mark_clinical_transition(
                    "claim_matches_generated",
                    "claim_resolution",
                    binding_id=binding_payload.get("binding_id"),
                    entity_id=binding_payload.get("entity_id"),
                    target_gap_ids=list(binding_payload.get("target_gap_ids") or []),
                    claim_match_count=len(claim_matches),
                    supported_claims=[
                        item.get("target_claim")
                        for item in claim_matches
                        if isinstance(item, dict)
                        and item.get("claim_status") == "SUPPORTED"
                    ],
                    contradicted_claims=[
                        item.get("target_claim")
                        for item in claim_matches
                        if isinstance(item, dict)
                        and item.get("claim_status") == "CONTRADICTED"
                    ],
                )
                claim_update = self.claim_resolution_updater.update_from_parse(
                    ledger=self._claim_resolution_ledger,
                    parsed_result=payload,
                    intent_binding=binding_payload,
                    gap_contract=self._claim_gap_contract_from_authorization_detail(detail),
                )
                self._claim_resolution_ledger = normalize_ledger(
                    claim_update.get("ledger") or {}
                )
                self._claim_match_events.extend(
                    list(claim_update.get("claim_match_events") or [])
                )
                update_audit = list(
                    claim_update.get("claim_resolution_update_audit") or []
                )
                self._claim_resolution_update_audit.extend(update_audit)
                claim_delta = int(
                    claim_update.get("persisted_claim_resolution_delta_count") or 0
                )
                self._mark_clinical_transition(
                    "claim_ledger_updated",
                    "claim_resolution",
                    binding_id=binding_payload.get("binding_id"),
                    entity_id=binding_payload.get("entity_id"),
                    claim_match_event_count=int(
                        claim_update.get("claim_match_event_count") or 0
                    ),
                    resolvable_claim_match_count=int(
                        claim_update.get("resolvable_claim_match_count") or 0
                    ),
                    persisted_claim_resolution_delta_count=claim_delta,
                    ledger_size=len(self._claim_resolution_ledger),
                )
                route_delta = sum(
                    int(item.get("route_attempt_state_delta") or 0)
                    for item in update_audit
                    if isinstance(item, dict)
                )
                result_claim_delta += claim_delta
                result_route_attempt_delta += route_delta
                payload["claim_resolution_update"] = {
                    key: value
                    for key, value in claim_update.items()
                    if key != "ledger"
                }
                self._targeted_exam_result_parses.append(payload)
                if not observations_recorded:
                    self._targeted_exam_observations.extend(neutral_parsed.observations)
                    observations_recorded = True
            if result_claim_delta:
                self._claim_state_version += 1
                self._diagnostic_state_version += 1
                self._mark_clinical_transition(
                    "claim_state_transaction_committed",
                    "claim_resolution",
                    ordered_exam=ordered_exam,
                    actual_result_exam=actual_exam,
                    persisted_claim_resolution_delta_count=result_claim_delta,
                    route_attempt_state_delta_count=result_route_attempt_delta,
                    claim_state_version_after=int(self._claim_state_version or 0),
                    diagnostic_state_version_after=int(
                        self._diagnostic_state_version or 0
                    ),
                )
                self._targeted_exam_result_parses.append(
                    {
                        "status": "claim_state_transaction_committed",
                        "stage": stage,
                        "ordered_exam": ordered_exam,
                        "actual_result_exam": actual_exam,
                        "persisted_claim_resolution_delta_count": result_claim_delta,
                        "route_attempt_state_delta_count": result_route_attempt_delta,
                        "claim_state_version_after": int(self._claim_state_version or 0),
                        "diagnostic_state_version_after": int(
                            self._diagnostic_state_version or 0
                        ),
                    }
                )
        if len(self._targeted_exam_observations) > observations_before:
            findings = [
                item.finding
                for item in self._targeted_exam_observations[observations_before:]
            ]
            logger.info(
                "[ExamEvidenceRecovery] targeted evidence recovered: %s",
                list(dict.fromkeys(findings)),
            )

    def _exam_authorization_details_for_result_recovery(
        self,
        strategy: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        for item in (strategy or {}).get("exam_authorization_details", []) or []:
            if isinstance(item, dict) and item.get("target_gaps"):
                details.append(dict(item))
        for record in getattr(self, "_last_exam_authorization", []) or []:
            if not isinstance(record, dict):
                continue
            for item in record.get("exam_authorization_details", []) or []:
                if isinstance(item, dict) and item.get("target_gaps"):
                    details.append(dict(item))
        seen: set[tuple] = set()
        result: List[Dict[str, Any]] = []
        for detail in details:
            key = (
                _compact_exam_name(detail.get("exam")),
                tuple(str(item or "") for item in detail.get("target_gaps", []) or []),
                str(detail.get("entity_id") or ""),
                tuple(str(item or "") for item in detail.get("route_target_claims", []) or []),
                str(detail.get("claim_closure_plan_version") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(detail)
        return result

    def _clinical_admission_top_k(self) -> int:
        judge = getattr(getattr(self, "diagnosis_engine", None), "judge", None)
        return int(getattr(judge, "filtered_pool_max_size", 0) or 8)

    def _clinical_admitted_candidate_views(self) -> List[Dict[str, Any]]:
        decision = getattr(self, "_last_diagnosis_decision_obj", None)
        candidates = list(getattr(decision, "candidates", []) or [])
        if not candidates:
            return []
        top_k = max(1, self._clinical_admission_top_k())
        current_primary = str(getattr(decision, "judge_primary", "") or "")
        bridge_protected = {
            str(item or "").strip()
            for item in getattr(decision, "bridge_protected_candidates", []) or []
            if str(item or "").strip()
        }
        arbitration_entities: set[str] = set()
        arbitration_names: set[str] = set()
        judge_decision = dict(getattr(decision, "judge_decision", {}) or {})
        for item in judge_decision.get("primary_arbitration_candidates", []) or []:
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("entity_id") or "").strip()
            name = str(item.get("diagnosis") or item.get("candidate") or "").strip()
            if entity_id:
                arbitration_entities.add(entity_id)
            if name:
                arbitration_names.add(name)
        for item in judge_decision.get("candidate_disposition_audit", []) or []:
            if not isinstance(item, dict) or not item.get("arbitration_pool_member"):
                continue
            entity_id = str(item.get("entity_id") or item.get("candidate_id") or "").strip()
            name = str(item.get("candidate") or "").strip()
            if entity_id:
                arbitration_entities.add(entity_id)
            if name:
                arbitration_names.add(name)

        records: List[Dict[str, Any]] = []
        admission_audit: List[Dict[str, Any]] = []
        seen_entities: set[str] = set()
        for index, candidate in enumerate(candidates, start=1):
            diagnosis = str(getattr(candidate, "diagnosis", "") or "").strip()
            entity_id = str(getattr(candidate, "entity_id", "") or "").strip()
            if not entity_id and self.diagnosis_engine.knowledge:
                entity_id = self.diagnosis_engine.knowledge.entity_id_for(diagnosis)
            if not diagnosis and not entity_id:
                continue
            reasons: List[str] = []
            if diagnosis and diagnosis == current_primary:
                reasons.append("CURRENT_PRIMARY")
            if index <= top_k:
                reasons.append("TOP_K")
            eligibility = str(getattr(candidate, "eligibility_status", "") or "")
            anchor = str(getattr(candidate, "eligibility_anchor_status", "") or "")
            if eligibility == "PrimaryEligible":
                reasons.append("PRIMARY_ELIGIBLE")
            if anchor == "AnchorSatisfied":
                reasons.append("ANCHOR_SATISFIED")
            if diagnosis in bridge_protected or entity_id in bridge_protected:
                reasons.append("PROTECTED_RECALL")
            if (
                int(getattr(candidate, "actionable_gap_count", 0) or 0) > 0
                or getattr(candidate, "required_gaps", None)
                or getattr(candidate, "evidence_gaps", None)
            ):
                reasons.append("ACTIVE_OR_PENDING_WORKUP")
            if diagnosis in arbitration_names or entity_id in arbitration_entities:
                reasons.append("ARBITRATION_MEMBER")
            material_evidence = bool(
                getattr(candidate, "matched_evidence", None)
                or getattr(candidate, "core_matched_evidence", None)
                or getattr(candidate, "diagnostic_matched_evidence", None)
                or getattr(candidate, "evidence_contributions", None)
            )
            if material_evidence:
                reasons.append("MATERIAL_EVIDENCE_FOR_EXISTING_CANDIDATE")
            if not reasons:
                continue
            entry = self.diagnosis_engine.knowledge.get(diagnosis) if diagnosis else {}
            contract = dict(entry.get("claim_anchor_contract") or {})
            if not contract and entity_id:
                entry = self._knowledge_entry_for_entity(entity_id)
                contract = dict(entry.get("claim_anchor_contract") or {})
            record = {
                "candidate": diagnosis,
                "diagnosis": diagnosis,
                "entity_id": entity_id,
                "rank": index,
                "clinical_admitted": True,
                "clinical_admission_reasons": list(dict.fromkeys(reasons)),
                "admission_state_version": int(
                    getattr(self, "_diagnostic_state_version", 0) or 0
                ),
                "eligibility_status": eligibility,
                "anchor_status": anchor,
                "claim_schema_available": bool(contract),
                "claim_anchor_contract": contract,
            }
            admission_audit.append({k: v for k, v in record.items() if k != "claim_anchor_contract"})
            if entity_id and entity_id in seen_entities:
                continue
            if entity_id:
                seen_entities.add(entity_id)
            records.append(record)
        self._clinical_admission_audit = admission_audit
        return records

    def _knowledge_entry_for_entity(self, entity_id: str) -> Dict[str, Any]:
        if not entity_id:
            return {}
        knowledge = getattr(self.diagnosis_engine, "knowledge", None)
        for entry in getattr(knowledge, "entries", {}).values():
            if str(entry.get("entity_id") or "") == entity_id:
                return dict(entry)
        return {}

    def _materialize_admitted_candidate_claim_states(self) -> List[Dict[str, Any]]:
        views = self._clinical_admitted_candidate_views()
        self._candidate_claim_contract_views = [
            self._compact_candidate_claim_contract_view(item)
            for item in views
            if item.get("claim_anchor_contract")
        ]
        active_entities = [
            str(item.get("entity_id") or "")
            for item in views
            if str(item.get("entity_id") or "")
        ]
        ledger, audit = materialize_candidate_claim_states(
            ledger=self._claim_resolution_ledger,
            contract_views=views,
            active_entity_ids=active_entities,
        )
        self._claim_resolution_ledger = normalize_ledger(ledger)
        if audit.get("contract_view_count") or audit.get("missing_claim_contracts"):
            self._claim_state_materialization_audit.append(audit)
            self._mark_clinical_transition(
                "candidate_claim_state_materialized",
                "claim_materialization",
                contract_view_count=audit.get("contract_view_count", 0),
                materialized_claim_state_count=audit.get(
                    "materialized_claim_state_count", 0
                ),
                reactivated_claim_state_count=audit.get(
                    "reactivated_claim_state_count", 0
                ),
                missing_claim_contract_count=len(audit.get("missing_claim_contracts") or []),
            )
        invariants = self._candidate_claim_state_invariants(views)
        if invariants:
            self._claim_state_invariant_audit.extend(invariants)
        return views

    @staticmethod
    def _compact_candidate_claim_contract_view(view: Dict[str, Any]) -> Dict[str, Any]:
        contract = dict(view.get("claim_anchor_contract") or {})
        requirements = claim_requirements_from_contract(contract)
        return {
            "candidate": view.get("candidate") or view.get("diagnosis") or "",
            "entity_id": view.get("entity_id"),
            "contract_id": contract.get("contract_id"),
            "contract_version": contract.get("contract_version"),
            "claims": [
                {
                    "claim_id": item.get("claim_id"),
                    "required_for_anchor": bool(item.get("required_for_anchor", True)),
                    "allowed_evidence_types": list(item.get("allowed_evidence_types") or []),
                    "allowed_exam_types": list(item.get("allowed_exam_types") or []),
                    "allowed_relation_types": list(item.get("allowed_relation_types") or []),
                }
                for item in requirements
            ],
            "closure_routes": list(contract.get("closure_routes") or []),
            "clinical_admission_reasons": list(
                view.get("clinical_admission_reasons") or []
            ),
            "contract_source": "disease_entity.claim_anchor_contract",
        }

    def _candidate_claim_state_invariants(
        self,
        views: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ledger = normalize_ledger(self._claim_resolution_ledger)
        records: List[Dict[str, Any]] = []
        for view in views or []:
            entity_id = str(view.get("entity_id") or "").strip()
            if not entity_id:
                continue
            contract = dict(view.get("claim_anchor_contract") or {})
            reasons = list(view.get("clinical_admission_reasons") or [])
            high_requirement = any(
                item in {"CURRENT_PRIMARY", "PRIMARY_ELIGIBLE", "ANCHOR_SATISFIED", "ARBITRATION_MEMBER"}
                for item in reasons
            )
            if not contract:
                records.append(
                    {
                        "entity_id": entity_id,
                        "candidate": view.get("candidate") or view.get("diagnosis") or "",
                        "invariant_code": (
                            "CLAIM_SCHEMA_REQUIRED_BUT_MISSING"
                            if high_requirement
                            else "NO_CLAIM_SCHEMA_AVAILABLE"
                        ),
                        "clinical_admission_reasons": reasons,
                        "diagnostic_state_version": int(
                            getattr(self, "_diagnostic_state_version", 0) or 0
                        ),
                    }
                )
                continue
            contract_id = str(contract.get("contract_id") or f"claim_anchor_contract:{entity_id}")
            contract_version = str(contract.get("contract_version") or "1")
            missing: List[str] = []
            for requirement in claim_requirements_from_contract(contract):
                claim_id = str(requirement.get("claim_id") or "").strip()
                if not claim_id:
                    continue
                key = "|".join([entity_id, claim_id, contract_id, contract_version])
                if key not in ledger:
                    missing.append(claim_id)
            if missing:
                records.append(
                    {
                        "entity_id": entity_id,
                        "candidate": view.get("candidate") or view.get("diagnosis") or "",
                        "invariant_code": "CLAIM_STATE_MATERIALIZATION_MISSING",
                        "missing_claims": missing,
                        "clinical_admission_reasons": reasons,
                        "diagnostic_state_version": int(
                            getattr(self, "_diagnostic_state_version", 0) or 0
                        ),
                    }
                )
        return records

    def _candidate_claim_contract_details_for_exam_result(
        self,
        ordered_exam: str,
        actual_exam: str,
        views: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        details: List[Dict[str, Any]] = []
        for view in views or []:
            contract = dict(view.get("claim_anchor_contract") or {})
            entity_id = str(view.get("entity_id") or "").strip()
            if not entity_id or not contract:
                continue
            routes = [
                dict(route)
                for route in contract.get("closure_routes", []) or []
                if isinstance(route, dict)
                and str(route.get("route_type") or "") == "exam_result"
                and self._closure_route_matches_exam(
                    route,
                    ordered_exam=ordered_exam,
                    actual_exam=actual_exam,
                )
            ]
            if not routes:
                continue
            route_claims = list(
                dict.fromkeys(
                    str(claim_id or "").strip()
                    for route in routes
                    for claim_id in route.get("target_claims", []) or []
                    if str(claim_id or "").strip()
                )
            )
            if not route_claims:
                continue
            requirements = claim_requirements_from_contract(contract)
            target_claims = [
                str(item.get("claim_id") or "").strip()
                for item in requirements
                if str(item.get("claim_id") or "").strip()
            ]
            detail = {
                "exam": ordered_exam or actual_exam,
                "requested_exam": ordered_exam or actual_exam,
                "resolved_exam": actual_exam or ordered_exam,
                "exam_source": "candidate_claim_state_applicability",
                "target_gaps": [f"claim_contract:{entity_id}"],
                "entity_id": entity_id,
                "target_candidates": [
                    view.get("candidate") or view.get("diagnosis") or entity_id
                ],
                "target_claims": target_claims,
                "route_target_claims": route_claims,
                "claim_requirements": requirements,
                "closure_routes": routes,
                "claim_closure_plan_version": str(
                    contract.get("contract_version")
                    or contract.get("claim_closure_plan_version")
                    or "1"
                ),
                "contract_id": str(
                    contract.get("contract_id") or f"claim_anchor_contract:{entity_id}"
                ),
                "contract_version": str(
                    contract.get("contract_version")
                    or contract.get("claim_closure_plan_version")
                    or "1"
                ),
                "_result_binding_source": "RESULT_APPLICABILITY",
                "_applicability_reason": "candidate_claim_contract_compatibility",
                "clinical_admission_reasons": list(
                    view.get("clinical_admission_reasons") or []
                ),
                "contract_source": "candidate_claim_state",
            }
            details.append(detail)
        return details

    def _closure_route_matches_exam(
        self,
        route: Dict[str, Any],
        *,
        ordered_exam: str,
        actual_exam: str,
    ) -> bool:
        route_exam = str(route.get("exam") or route.get("requested_exam") or "").strip()
        if not route_exam:
            return True
        route_key = _compact_exam_name(route_exam)
        ordered_key = _compact_exam_name(ordered_exam)
        actual_key = _compact_exam_name(actual_exam)
        if route_key and route_key in {ordered_key, actual_key}:
            return True
        route_family = self._exam_repeat_family(route_exam)
        ordered_family = self._exam_repeat_family(ordered_exam)
        actual_family = self._exam_repeat_family(actual_exam)
        if route_family and route_family in {ordered_family, actual_family}:
            return True
        return bool(route_key and (route_key in ordered_key or route_key in actual_key))

    def _hydrate_claim_states_from_existing_exam_observations(
        self,
        views: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        stage: str = "historical_claim_hydration",
    ) -> None:
        observations = list(getattr(self, "_targeted_exam_observations", []) or [])
        if not observations:
            return
        contract_views = list(views or self._candidate_claim_contract_views or [])
        if not contract_views:
            return
        by_exam: Dict[str, List[Observation]] = {}
        for obs in observations:
            exam = str(getattr(obs, "source_exam", "") or "historical_exam")
            by_exam.setdefault(exam, []).append(obs)
        total_delta = 0
        total_events = 0
        for exam, exam_observations in by_exam.items():
            details = self._candidate_claim_contract_details_for_exam_result(
                exam,
                exam,
                contract_views,
            )
            if not details:
                continue
            finding_key = ",".join(
                sorted(
                    {
                        str(getattr(obs, "finding", "") or "")
                        for obs in exam_observations
                        if str(getattr(obs, "finding", "") or "")
                    }
                )
            )
            neutral = TargetedExamParseResult(
                binding_id=f"historical-neutral:{exam}",
                order_id=f"historical-order:{exam}",
                target_gap_ids=[],
                entity_id="",
                parser_profile="historical_observation_hydration",
                observations=list(exam_observations),
                actual_result_exam=exam,
                execution_status="historical",
            )
            for index, detail in enumerate(details, start=1):
                contract_id = str(detail.get("contract_id") or "")
                entity_id = str(detail.get("entity_id") or "")
                hydration_key = "|".join([entity_id, contract_id, exam, finding_key])
                if hydration_key in self._historical_claim_hydration_keys:
                    continue
                self._historical_claim_hydration_keys.add(hydration_key)
                binding = binding_from_authorization_detail(
                    detail=detail,
                    requested_exam=exam,
                    actual_result_exam=exam,
                    patient_id=str(getattr(self, "_case_id_for_thinking", "") or ""),
                    stage=stage,
                    order_index=900000 + index,
                )
                parsed = self.targeted_exam_result_parser.rematch_claims_for_binding(
                    neutral,
                    binding,
                )
                payload = parsed.to_dict()
                payload["stage"] = stage
                payload["ordered_exam"] = exam
                payload["binding_source"] = "HISTORICAL_RESULT_APPLICABILITY"
                payload["applicability_reason"] = "historical_candidate_claim_hydration"
                claim_update = self.claim_resolution_updater.update_from_parse(
                    ledger=self._claim_resolution_ledger,
                    parsed_result=payload,
                    intent_binding=binding.to_dict(),
                    gap_contract=self._claim_gap_contract_from_authorization_detail(detail),
                )
                self._claim_resolution_ledger = normalize_ledger(
                    claim_update.get("ledger") or {}
                )
                self._claim_match_events.extend(
                    list(claim_update.get("claim_match_events") or [])
                )
                self._claim_resolution_update_audit.extend(
                    list(claim_update.get("claim_resolution_update_audit") or [])
                )
                total_events += int(claim_update.get("claim_match_event_count") or 0)
                total_delta += int(
                    claim_update.get("persisted_claim_resolution_delta_count") or 0
                )
                payload["claim_resolution_update"] = {
                    key: value
                    for key, value in claim_update.items()
                    if key != "ledger"
                }
                self._targeted_exam_result_parses.append(payload)
        if total_delta:
            self._claim_state_version += 1
            self._diagnostic_state_version += 1
            self._mark_clinical_transition(
                "historical_claim_state_transaction_committed",
                "claim_resolution",
                claim_match_event_count=total_events,
                persisted_claim_resolution_delta_count=total_delta,
                claim_state_version_after=int(self._claim_state_version or 0),
                diagnostic_state_version_after=int(self._diagnostic_state_version or 0),
            )

    def _authorization_details_for_exam_result(
        self,
        ordered_exam: str,
        actual_exam: str,
        details: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ordered_key = _compact_exam_name(ordered_exam)
        actual_key = _compact_exam_name(actual_exam)
        ordered_family = self._exam_repeat_family(ordered_exam)
        actual_family = self._exam_repeat_family(actual_exam)
        primary = self._authorization_detail_for_exam(ordered_exam, actual_exam, details)
        result: List[Dict[str, Any]] = []
        seen: set[tuple] = set()

        def add(detail: Dict[str, Any], source: str, reason: str) -> None:
            key = (
                _compact_exam_name(detail.get("exam")),
                tuple(str(item or "") for item in detail.get("target_gaps", []) or []),
                str(detail.get("entity_id") or ""),
                tuple(str(item or "") for item in detail.get("route_target_claims", []) or []),
                str(detail.get("claim_closure_plan_version") or ""),
            )
            if key in seen:
                return
            seen.add(key)
            payload = dict(detail)
            payload["_result_binding_source"] = str(
                detail.get("_result_binding_source") or source
            )
            payload["_applicability_reason"] = str(
                detail.get("_applicability_reason") or reason
            )
            result.append(payload)

        if primary:
            add(primary, "AUTHORIZED_TARGET", "ordered_or_actual_exam_match")
        for detail in details or []:
            keys = {
                _compact_exam_name(detail.get("exam")),
                _compact_exam_name(detail.get("requested_exam")),
                _compact_exam_name(detail.get("resolved_exam")),
            }
            if (ordered_key and ordered_key in keys) or (actual_key and actual_key in keys):
                add(detail, "SHARED_AUTHORIZATION", "same_exam_authorization")
                continue
            detail_family = self._exam_repeat_family(
                detail.get("resolved_exam") or detail.get("exam") or detail.get("requested_exam")
            )
            if detail_family and detail_family in {ordered_family, actual_family}:
                add(detail, "RESULT_APPLICABILITY", "same_exam_family")
                continue
            if self._detail_has_applicable_exam_route(
                detail,
                ordered_exam=ordered_exam,
                actual_exam=actual_exam,
            ):
                add(detail, "RESULT_APPLICABILITY", "claim_closure_route_exam_match")
        return result

    def _neutral_parse_detail_for_applicable_contracts(
        self,
        details: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        for detail in details or []:
            if not self._authorization_detail_is_pavm(detail):
                return detail
        return details[0] if details else {}

    @staticmethod
    def _authorization_detail_is_pavm(detail: Dict[str, Any]) -> bool:
        text = _compact_exam_name(
            " ".join(
                str(item or "")
                for item in list(detail.get("target_candidates") or [])
                + list(detail.get("target_claims") or [])
                + list(detail.get("target_gaps") or [])
                + [detail.get("entity_id")]
            )
        )
        return any(marker in text for marker in ("d100055", "pavm", "pulmonaryav"))

    def _detail_has_applicable_exam_route(
        self,
        detail: Dict[str, Any],
        *,
        ordered_exam: str,
        actual_exam: str,
    ) -> bool:
        ordered_family = self._exam_repeat_family(ordered_exam)
        actual_family = self._exam_repeat_family(actual_exam)
        ordered_key = _compact_exam_name(ordered_exam)
        actual_key = _compact_exam_name(actual_exam)
        for route in detail.get("closure_routes", []) or []:
            if not isinstance(route, dict):
                continue
            if str(route.get("route_type") or "") != "exam_result":
                continue
            route_exam = str(route.get("exam") or route.get("requested_exam") or "").strip()
            if not route_exam:
                return True
            route_key = _compact_exam_name(route_exam)
            route_family = self._exam_repeat_family(route_exam)
            if route_key and route_key in {ordered_key, actual_key}:
                return True
            if route_family and route_family in {ordered_family, actual_family}:
                return True
            if route_key and (route_key in ordered_key or route_key in actual_key):
                return True
        return False

    @staticmethod
    def _match_ordered_items_to_results(
        ordered_items: List[str],
        new_results: Dict[str, Any],
    ) -> List[tuple]:
        remaining = list((new_results or {}).items())
        pairs: List[tuple] = []
        for ordered in ordered_items or []:
            match_index = next(
                (
                    idx
                    for idx, (name, _) in enumerate(remaining)
                    if _compact_exam_name(name) == _compact_exam_name(ordered)
                ),
                -1,
            )
            if match_index < 0 and remaining:
                match_index = 0
            if match_index < 0:
                continue
            actual, raw = remaining.pop(match_index)
            pairs.append((ordered, actual, raw))
        for actual, raw in remaining:
            pairs.append(("", actual, raw))
        return pairs

    @staticmethod
    def _claim_gap_contract_from_authorization_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
        contract = dict(detail or {})
        target_gaps = [
            str(item or "").strip()
            for item in contract.get("target_gaps", []) or []
            if str(item or "").strip()
        ]
        if target_gaps and not contract.get("gap_id"):
            contract["gap_id"] = target_gaps[0]
        entity_id = str(contract.get("entity_id") or "").strip()
        if not entity_id:
            candidates = [
                str(item or "").strip()
                for item in contract.get("target_candidates", []) or []
                if str(item or "").strip()
            ]
            compact = " ".join(candidates + target_gaps).lower()
            if "d100058" in compact or "radiation" in compact:
                entity_id = "D100058"
            elif "d100055" in compact or "pavm" in compact:
                entity_id = "D100055"
        if entity_id:
            contract["entity_id"] = entity_id
        contract.setdefault("contract_id", f"claim_anchor_contract:{entity_id or 'unknown'}")
        contract.setdefault(
            "contract_version",
            str(contract.get("claim_closure_plan_version") or "1"),
        )
        return contract

    @staticmethod
    def _authorization_detail_for_exam(
        ordered_exam: str,
        actual_exam: str,
        details: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        ordered_key = _compact_exam_name(ordered_exam)
        actual_key = _compact_exam_name(actual_exam)
        for detail in details:
            keys = {
                _compact_exam_name(detail.get("exam")),
                _compact_exam_name(detail.get("requested_exam")),
                _compact_exam_name(detail.get("resolved_exam")),
            }
            if ordered_key and ordered_key in keys:
                return detail
            if actual_key and actual_key in keys:
                return detail
        if len(details) == 1:
            return details[0]
        return None

    async def _execute_order_examination(
        self,
        patient_id: str,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        relevant_experience: List[Dict[str, Any]],
        target: str,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        """执行单次检查操作。

        Args:
            patient_id: 患者 ID
            collected_info: 已收集信息
            exam_results: 已有检查结果
            relevant_experience: 相关经验
            target: 检查目标（来自规划器）
            reason: 检查理由（来自规划器）

        Returns:
            新增检查结果字典，失败返回 None
        """
        # 基于症状检索相关经验（带缓存）
        cached_exp = self._get_cached_experience(collected_info)
        if cached_exp:
            relevant_experience = cached_exp

        # 思考：更新鉴别诊断，判断检查是否足够
        thinking = await self._think(
            collected_info, exam_results, [], "examination", relevant_experience
        )

        # 判断检查是否已足够
        if exam_results and thinking and "is_sufficient" in thinking:
            if thinking.get("is_sufficient"):
                logger.info("[检查] 思考判断检查已足够，跳过")
                return None

        # 从 knowledge 构建 RAG 上下文（症状 + 候选疾病）
        _sym = collected_info.get("symptoms") or []
        _cands = None
        if thinking and isinstance(thinking, dict):
            _cands = thinking.get("differential_diagnosis") or thinking.get("candidate_diseases")
        pre_exam_judge = self._pre_exam_judge_payload(
            collected_info,
            exam_results,
            thinking=thinking if isinstance(thinking, dict) else None,
        )
        if pre_exam_judge.get("differential_candidates"):
            _cands = [
                str(item).strip()
                for item in pre_exam_judge.get("differential_candidates") or []
                if str(item).strip()
            ]
        try:
            knowledge_context = self.knowledge.build_rag_context(_sym, _cands)
        except Exception:
            knowledge_context = ""
        try:
            if getattr(self, "memory_manager", None):
                memory_context = self.memory_manager.build_semantic_context(
                    collected_info, _cands
                )
                if memory_context:
                    knowledge_context = memory_context
        except Exception:
            pass

        # 构建检查申请 prompt（注入思考结果和规划目标）
        exam_prompt = self.prompt.build_examination_prompt(
            collected_info=collected_info,
            exam_results=exam_results,
            relevant_experience=relevant_experience,
            thinking=thinking,
            knowledge_context=knowledge_context,
        )
        messages = [
            {"role": "system", "content": exam_prompt},
            {"role": "user", "content": f"请针对以下目标申请检查：{target}。理由：{reason}。输出 JSON 数组格式的检查项目列表。"},
        ]

        exam_items = await self._llm_generate_examination_items(messages, collected_info)
        strategy = self.exam_agent.recommend(
            collected_info=collected_info,
            candidate_diseases=_cands if isinstance(_cands, list) else None,
            proposed_items=exam_items,
            existing_results=exam_results,
            judge_decision=pre_exam_judge or None,
        )
        if strategy.get("strong_verification_items"):
            logger.info(f"[检查策略] 强验证检查: {strategy['strong_verification_items']}")
        if strategy.get("red_flag_items"):
            logger.info(f"[检查策略] 红旗补查检查: {strategy['red_flag_items']}")
        if strategy.get("evidence_driven_items"):
            logger.info(f"[检查策略] 证据驱动补查检查: {strategy['evidence_driven_items']}")
        if strategy.get("added_required"):
            logger.info(f"[检查策略] 补齐必查检查: {strategy['added_required']}")
        if strategy.get("invalid_items"):
            logger.info(f"[检查策略] 过滤无效检查项: {strategy['invalid_items']}")
        if (
            strategy.get("strict_diagnosis_driven")
            or strategy.get("differential_driven")
            or strategy.get("blocked_items")
        ):
            self._last_exam_authorization.append(
                {
                    "stage": "planner_exam",
                    "target": target,
                    "strict_diagnosis_driven": bool(
                        strategy.get("strict_diagnosis_driven")
                    ),
                    "differential_driven": bool(strategy.get("differential_driven")),
                    "primary_diagnosis": strategy.get("primary_diagnosis", ""),
                    "differential_candidates": list(
                        strategy.get("differential_candidates") or []
                    ),
                    "discriminating_items": list(
                        strategy.get("discriminating_items") or []
                    ),
                    "authorized_items": list(strategy.get("items") or []),
                    "reserved_gap_items": list(strategy.get("reserved_gap_items") or []),
                    "source_decision_version": strategy.get("source_decision_version", 0),
                    "source_evidence_version": strategy.get("source_evidence_version", 0),
                    "blocked_items": list(strategy.get("blocked_items") or []),
                    "exam_authorization_details": list(
                        strategy.get("exam_authorization_details") or []
                    ),
                    "generic_exam_suppression_count": int(
                        strategy.get("generic_exam_suppression_count", 0) or 0
                    ),
                }
            )
        exam_items = self._strategy_order_items(
            strategy,
            collected_info=collected_info,
            candidate_diseases=_cands if isinstance(_cands, list) else None,
            existing_results=exam_results,
            max_items=self.exam_agent.max_new_items,
            add_strong_verification=False,
        )
        self._record_exam_route_authorization_summary(
            stage="planner_exam",
            target=target,
            strategy=strategy,
            authorized_items=exam_items,
        )
        if not exam_items:
            logger.info("[检查] 无检查项目需要申请")
            return None

        # 去重
        new_items = [item for item in exam_items if item not in exam_results]
        if not new_items:
            logger.info("[检查] 所有推荐检查已完成")
            return None

        logger.info(f"[检查] 申请检查: {new_items}")

        # 申请检查
        response = await self.actions.order_examination(
            patient_id=patient_id,
            items=new_items,
            reason=reason or "基于问诊信息，需要进一步检查以明确诊断。",
        )

        # 合并检查结果
        new_results = {}
        if response and "results" in response:
            for exam_name, exam_data in response["results"].items():
                if exam_data.get("status") != "invalid":
                    new_results[exam_name] = exam_data
        if new_results:
            self._record_targeted_exam_result_recovery(
                patient_id=patient_id,
                stage="planner_exam",
                ordered_items=list(new_items),
                new_results=new_results,
                strategy=strategy,
            )

        return new_results if new_results else None

    # ============ 训练流程（规划驱动） ============

    async def train(self, patient_id: str) -> Dict[str, Any]:
        """训练流程：使用规划器驱动诊疗，并在结束后评估反思。

        Args:
            patient_id: 患者 ID
        """
        logger.info(f"[Train] 开始训练患者: {patient_id}")
        if hasattr(self.actions, "begin_case"):
            self.actions.begin_case(patient_id)
        trace = getattr(self, "trace_collector", None)
        case_span_id = None
        if trace and trace.enabled:
            trace.start_trace(
                patient_id,
                {
                    "mode": "train",
                    "agent": self.__class__.__name__,
                    "fast_mode": bool(self.fast_mode),
                    "diagnosis_chain_enabled": bool(self.diagnosis_chain_enabled),
                    "case_timeout_seconds": self.case_timeout_seconds,
                },
            )
            case_span_id = trace.start_span(
                "case_orchestrator",
                self.__class__.__name__,
                "train_case",
            )
            trace.create_artifact(ArtifactType.RAW_CASE, {"patient_id": patient_id})
        self._reset_llm_counter()
        # 跨患者软复用：重置流程状态但保留可迁移的教��
        if self._planner is not None:
            self._planner.soft_reset(keep_lessons=True)

        # 规划器驱动诊疗
        try:
            final_result = await self._run_case_pipeline(
                patient_id,
                post_submit_reserve_seconds=self.train_post_submit_reserve_seconds,
            )
        except Exception as exc:
            if trace and trace.enabled:
                trace.end_span(case_span_id, status="failed", payload={"error": str(exc)})
                trace.fail_trace(exc)
            raise

        # 训练阶段：先独立获取评估，再执行反思。反思失败不能伪装成评估失败。
        report: Dict[str, Any] = {}
        evaluation_error = ""
        reflection_error = ""
        try:
            evaluation_call = self.actions.evaluation(
                patient_id=patient_id, final_result=final_result
            )
            if self.case_timeout_seconds > 0:
                available = (
                    self._remaining_total_case_seconds()
                    - self.train_reflection_timeout_seconds
                    - 1.0
                )
                evaluation_budget = min(
                    self.train_evaluation_timeout_seconds,
                    max(0.0, available),
                )
                if evaluation_budget < 0.5:
                    evaluation_call.close()
                    raise asyncio.TimeoutError("no evaluation budget remaining")
                raw_report = await asyncio.wait_for(
                    evaluation_call,
                    timeout=evaluation_budget,
                )
            else:
                raw_report = await evaluation_call
            if not isinstance(raw_report, dict):
                raise TypeError("evaluation response must be a JSON object")
            report = raw_report
            logger.info(
                f"[Train] 患者 {patient_id} 评估报告: "
                f"{json.dumps(report, ensure_ascii=False, indent=2)}"
            )
        except asyncio.TimeoutError:
            evaluation_error = (
                f"evaluation exceeded {self.train_evaluation_timeout_seconds:.0f}s budget"
            )
            logger.warning(f"[Train] {evaluation_error}")
        except Exception as exc:
            evaluation_error = str(exc)
            logger.warning(f"[Train] 获取评估报告失败: {exc}")

        if report:
            notes_before_reflection = len(getattr(self.memory, "notes", []) or [])
            try:
                if self.skip_train_reflection:
                    self._save_fast_reflection(
                        patient_id,
                        report,
                        self._last_collected_info,
                        self._last_exam_results,
                    )
                else:
                    reflection_call = self._reflect_and_save(
                        patient_id, report,
                        self._last_collected_info,
                        self._last_exam_results,
                    )
                    if self.case_timeout_seconds > 0:
                        reflection_budget = min(
                            self.train_reflection_timeout_seconds,
                            max(0.0, self._remaining_total_case_seconds() - 1.0),
                        )
                        if reflection_budget < 0.5:
                            reflection_call.close()
                            raise asyncio.TimeoutError("no reflection budget remaining")
                        await asyncio.wait_for(
                            reflection_call,
                            timeout=reflection_budget,
                        )
                    else:
                        await reflection_call
            except asyncio.TimeoutError:
                reflection_error = (
                    f"reflection exceeded {self.train_reflection_timeout_seconds:.0f}s budget"
                )
                logger.warning(f"[Train] {reflection_error}; using deterministic reflection")
                if len(getattr(self.memory, "notes", []) or []) == notes_before_reflection:
                    try:
                        self._save_fast_reflection(
                            patient_id,
                            report,
                            self._last_collected_info,
                            self._last_exam_results,
                        )
                    except Exception as fallback_exc:
                        reflection_error += f"; fallback failed: {fallback_exc}"
            except Exception as exc:
                reflection_error = str(exc)
                logger.warning(f"[Train] 反思或记忆保存失败: {exc}")

        training_elapsed = round(max(0.0, time.monotonic() - self._case_started_at), 3)
        if self._last_diagnosis_audit:
            self._last_diagnosis_audit["training_elapsed_seconds"] = training_elapsed

        # 输出记忆统计
        stats = self.memory.get_statistics()
        logger.info(f"[Train] 记忆统计: {json.dumps(stats, ensure_ascii=False)}")
        logger.info(
            f"[Train] LLM 调用统计: 总{self._llm_call_count}次, "
            f"明细={json.dumps(self._llm_call_by_kind, ensure_ascii=False)}"
        )
        train_result = self._build_training_result(
            patient_id=patient_id,
            final_result=final_result,
            report=report,
            evaluation_error=evaluation_error,
            reflection_error=reflection_error,
        )
        self._last_train_result = train_result
        if trace and trace.enabled:
            self._emit_trace_case_artifacts(patient_id, final_result, train_result=train_result)
            trace.end_span(case_span_id, status="success")
            trace.complete_trace(final_result)
        logger.info(f"[Train] 完成训练患者: {patient_id}")
        return train_result

    def _emit_trace_case_artifacts(
        self,
        patient_id: str,
        final_result: Dict[str, Any],
        *,
        train_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        trace = getattr(self, "trace_collector", None)
        if not trace or not trace.enabled:
            return
        audit = self._last_diagnosis_audit or {}
        try:
            evidence_payload = audit.get("evidence") or {
                "collected_info": getattr(self, "_last_collected_info", {}) or {},
                "exam_results": getattr(self, "_last_exam_results", {}) or {},
            }
            trace.create_artifact(ArtifactType.EVIDENCE_SET, evidence_payload)

            decision_payload = audit.get("diagnosis_decision") or {}
            decision_ref = None
            if decision_payload:
                decision_ref = trace.create_artifact(
                    ArtifactType.DIAGNOSIS_DECISION,
                    decision_payload,
                )
                trace.emit_decision(
                    "diagnosis_decision",
                    {
                        "candidate_count": len(decision_payload.get("candidate_scores") or []),
                        "final_diagnoses": decision_payload.get("final_diagnoses") or [],
                        "primary_status": (
                            (decision_payload.get("judge_decision") or {}).get("primary_status")
                            if isinstance(decision_payload.get("judge_decision"), dict)
                            else None
                        ),
                    },
                    refs=[decision_ref] if decision_ref else [],
                )
                pattern_hypotheses = decision_payload.get("llm_pattern_hypotheses") or []
                if pattern_hypotheses:
                    trace.create_artifact(ArtifactType.PATTERN_HYPOTHESIS, pattern_hypotheses)
                pattern_verifications = (
                    list(decision_payload.get("verified_pattern_hypotheses") or [])
                    + list(decision_payload.get("rejected_pattern_hypotheses") or [])
                )
                if pattern_verifications:
                    trace.create_artifact(
                        ArtifactType.PATTERN_HYPOTHESIS_VERIFICATION,
                        pattern_verifications,
                    )
                    entity_links = []
                    for item in pattern_verifications:
                        if isinstance(item, dict):
                            entity_links.extend(item.get("entity_links") or [])
                    if entity_links:
                        trace.create_artifact(ArtifactType.PATTERN_ENTITY_LINK, entity_links)
                pattern_signals = decision_payload.get("pattern_recall_signals") or []
                if pattern_signals:
                    trace.create_artifact(ArtifactType.PATTERN_RECALL_SIGNAL, pattern_signals)
                pattern_audit = decision_payload.get("pattern_recall_audit") or {}
                if pattern_audit:
                    trace.create_artifact(ArtifactType.PATTERN_RECALL_AUDIT, pattern_audit)
                pattern_admissions = decision_payload.get("pattern_candidate_admissions") or []
                if pattern_admissions:
                    trace.create_artifact(
                        ArtifactType.PATTERN_CANDIDATE_ADMISSION,
                        pattern_admissions,
                    )

            judge_payload = (
                decision_payload.get("judge_decision")
                if isinstance(decision_payload, dict)
                else {}
            ) or {}
            active_gaps = (
                judge_payload.get("active_evidence_gaps")
                or judge_payload.get("evidence_gaps")
                or audit.get("active_evidence_gaps")
                or []
            )
            if active_gaps:
                gap_ref = trace.create_artifact(ArtifactType.EVIDENCE_GAP, active_gaps)
                trace.emit_decision(
                    "evidence_gap",
                    {
                        "gap_count": len(active_gaps),
                        "active_gap_ids": [
                            str(item.get("gap_id"))
                            for item in active_gaps
                            if isinstance(item, dict) and item.get("gap_id")
                        ],
                    },
                    refs=[gap_ref] if gap_ref else [],
                    stage="judge",
                    component="DiagnosisJudge",
                    action="emit_evidence_gaps",
                )

            if self._last_exam_authorization:
                exam_plan_ref = trace.create_artifact(
                    ArtifactType.EXAM_PLAN,
                    self._last_exam_authorization,
                )
                trace.emit_decision(
                    "exam_plan",
                    {
                        "plan_count": len(self._last_exam_authorization),
                        "authorized_exam_count": sum(
                            len(item.get("authorized_items") or [])
                            for item in self._last_exam_authorization
                            if isinstance(item, dict)
                        ),
                    },
                    refs=[exam_plan_ref] if exam_plan_ref else [],
                    stage="exam_strategy",
                    component="ExamStrategy",
                    action="recommend_exams",
                )

            if self._exam_result_intent_bindings:
                trace.create_artifact(
                    ArtifactType.EXAM_RESULT_INTENT_BINDING,
                    self._exam_result_intent_bindings,
                )

            if self._targeted_exam_result_parses or self._targeted_exam_observations:
                evidence_update_ref = trace.create_artifact(
                    ArtifactType.EVIDENCE_UPDATE,
                    {
                        "targeted_exam_result_parses": self._targeted_exam_result_parses,
                        "targeted_exam_observations": [
                            item.to_dict() if hasattr(item, "to_dict") else item
                            for item in self._targeted_exam_observations
                        ],
                        "claim_resolution_ledger": normalize_ledger(
                            self._claim_resolution_ledger
                        ),
                        "claim_match_events": list(self._claim_match_events),
                        "claim_resolution_update_audit": list(
                            self._claim_resolution_update_audit
                        ),
                        "claim_state_version": int(self._claim_state_version or 0),
                        "diagnostic_state_version": int(
                            self._diagnostic_state_version or 0
                        ),
                    },
                )
                trace.emit_event(
                    "state.changed",
                    payload={
                        "state_type": "evidence_feedback",
                        "added_evidence_count": len(self._targeted_exam_observations),
                        "parse_count": len(self._targeted_exam_result_parses),
                    },
                    output_refs=[evidence_update_ref] if evidence_update_ref else [],
                    stage="evidence_feedback",
                    component="TargetedExamResultParser",
                    action="recover_exam_evidence",
                )

            submitted = final_result.get("diagnosis") if isinstance(final_result, dict) else []
            if isinstance(submitted, str):
                submitted = [submitted]
            submission_ref = trace.create_artifact(ArtifactType.SUBMISSION_RESULT, final_result)
            trace.emit_submission(
                {
                    "submitted_diagnoses": list(submitted or []),
                    "judge_decision_available": bool(decision_payload),
                    "submission_status": "created",
                    "termination_reason": ""
                    if submitted
                    else "NO_PRIMARY_ELIGIBLE_CANDIDATE",
                },
                refs=[submission_ref] if submission_ref else [],
            )

            if train_result:
                trace.create_artifact(ArtifactType.MODULE_OUTPUT, train_result)
        except Exception as exc:
            if not getattr(getattr(trace, "config", None), "fail_open", True):
                raise
            logger.warning("[Trace] failed to emit case artifacts for %s: %s", patient_id, exc)

    def _build_training_result(
        self,
        patient_id: str,
        final_result: Dict[str, Any],
        report: Dict[str, Any],
        evaluation_error: str = "",
        reflection_error: str = "",
    ) -> Dict[str, Any]:
        """Build a compact, secret-free record for batch training reports."""
        runtime_audit = self._collect_runtime_audit()
        detail = report.get("diagnosisDetail") or report.get("diagnosis_detail") or {}
        if not isinstance(detail, dict):
            detail = {}

        def _names(value: Any) -> List[str]:
            if isinstance(value, str):
                value = [value]
            return list(
                dict.fromkeys(
                    str(item).strip() for item in (value or []) if str(item).strip()
                )
            )

        def _metric(*keys: str) -> Optional[float]:
            for key in keys:
                value = report.get(key)
                try:
                    if value is not None:
                        return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        audit = self._last_diagnosis_audit or {}
        decision = audit.get("diagnosis_decision") or {}
        candidates = decision.get("candidates") or []
        top_five = [
            str(item.get("diagnosis"))
            for item in candidates[:5]
            if isinstance(item, dict) and item.get("diagnosis")
        ]
        top_twenty = [
            str(item.get("diagnosis"))
            for item in candidates[:20]
            if isinstance(item, dict) and item.get("diagnosis")
        ]
        expected = _names(detail.get("expected") or report.get("finalDiagnosis"))
        submitted = _names(
            detail.get("submitted")
            or report.get("diagnosis")
            or final_result.get("diagnosis")
        )
        recall_at_five = all(name in top_five for name in expected) if expected else None
        recall_at_twenty = all(name in top_twenty for name in expected) if expected else None
        ranking_accuracy = (
            bool(expected and top_twenty)
            and top_twenty[0] in set(expected)
        ) if expected else None
        authorized = _names(
            decision.get("authorized_diagnoses")
            or decision.get("final_diagnoses")
            or final_result.get("_authorized_diagnoses")
        )
        submission_alignment = (
            authorized == submitted
            if authorized or submitted
            else None
        )
        retriever_top1 = str(
            decision.get("retriever_top1")
            or (top_twenty[0] if top_twenty else "")
            or ""
        )
        judge_primary = str(
            decision.get("judge_primary")
            or (authorized[0] if authorized else "")
            or ""
        )
        submitter_final = _names(
            decision.get("submitter_final")
            or decision.get("authorized_diagnoses")
            or final_result.get("_authorized_diagnoses")
            or submitted
        )
        raw_override = decision.get("decision_override")
        decision_override_rate = (
            bool(raw_override)
            if raw_override is not None
            else bool(retriever_top1 and judge_primary and retriever_top1 != judge_primary)
        )
        required_gap_authorized_diagnoses = _names(
            decision.get("required_gap_authorized_diagnoses")
        )
        judge_payload = decision.get("judge_decision") or {}
        case_board_payload = dict(decision.get("case_board") or {})
        case_board_audit = {}
        for item in case_board_payload.get("audit_events", []) or []:
            if isinstance(item, dict):
                case_board_audit.update(item)
        eligibility_distribution = dict(
            decision.get("eligibility_distribution")
            or judge_payload.get("eligibility_distribution")
            or {}
        )
        decision_conflicts = list(
            decision.get("evidence_conflicts")
            or judge_payload.get("evidence_conflicts")
            or []
        )
        blocked_records = list(decision.get("blocked_diagnoses") or [])
        reasoning_structured_conflict_count = sum(
            1
            for item in decision_conflicts
            if isinstance(item, dict)
            and str(item.get("conflict_type") or "")
            == "reasoning_structured_polarity_conflict"
        )
        conflict_deferred_primary_count = int(
            bool(
                reasoning_structured_conflict_count
                and str(judge_payload.get("primary_status") or "") == "deferred"
                and judge_payload.get("needs_discriminating_exams")
            )
        )
        conflict_blocked_final_count = sum(
            1
            for item in blocked_records
            if isinstance(item, dict)
            and str(item.get("reason") or "")
            == "unresolved reasoning-structured evidence conflict"
        )
        pairwise_comparisons = list(judge_payload.get("pairwise_comparisons") or [])
        pool_filter_summary = dict(judge_payload.get("pool_filter_summary") or {})
        root_cause_payload = dict(
            decision.get("root_cause_arbitration")
            or judge_payload.get("root_cause_arbitration")
            or {}
        )
        root_cause_secondary = _names(
            root_cause_payload.get("root_cause_secondary")
            or judge_payload.get("root_cause_secondary")
            or []
        )
        root_cause_arbitration_count = int(bool(root_cause_payload.get("applied")))
        root_cause_primary_override_count = int(
            bool(root_cause_payload.get("primary_override"))
        )
        root_cause_secondary_submission_count = len(
            [name for name in root_cause_secondary if name in set(submitter_final)]
        )
        try:
            root_cause_coverage = float(
                root_cause_payload.get("root_cause_coverage", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            root_cause_coverage = 0.0
        differential_candidates = _names(judge_payload.get("differential_candidates"))
        discriminating_exams = _names(judge_payload.get("discriminating_exams"))
        discriminating_findings = _names(judge_payload.get("discriminating_findings"))
        deferred_evidence_gaps = [
            item
            for item in judge_payload.get("deferred_evidence_gaps", []) or []
            if isinstance(item, dict)
        ]
        exam_priority_overrides = [
            item
            for item in judge_payload.get("exam_priority_overrides", []) or []
            if isinstance(item, dict)
        ]
        deferred_gap_closure_tasks = [
            item
            for item in judge_payload.get("deferred_gap_closure_tasks", []) or []
            if isinstance(item, dict)
        ]
        active_evidence_gaps = [
            item
            for item in judge_payload.get("active_evidence_gaps", []) or []
            if isinstance(item, dict)
        ]
        deferred_gap_target_ids = {
            str(gap.get("gap_id") or "")
            for item in exam_priority_overrides
            for gap in item.get("evidence_gaps", []) or []
            if isinstance(gap, dict) and str(gap.get("gap_id") or "")
        }
        dynamic_trace = list(judge_payload.get("dynamic_rerank_trace") or [])
        dynamic_rerank_changed_primary = bool(
            judge_payload.get("dynamic_rerank_changed_primary")
            or any(
                isinstance(item, dict) and item.get("changed_primary")
                for item in dynamic_trace
            )
        )
        primary_unlock_reason = str(judge_payload.get("primary_unlock_reason") or "")
        explanation_score_changed_ranking = bool(
            judge_payload.get("explanation_score_changed_ranking")
        )
        gap_state_distribution = dict(judge_payload.get("gap_state_distribution") or {})
        judge_gap_authorization_rate = False
        judge_primary_accuracy = (
            bool(expected and judge_primary in set(expected))
            if expected
            else None
        )

        def _priority_expected_names(names: List[str]) -> List[str]:
            priority: List[str] = []
            for name in names or []:
                entry = self.diagnosis_engine.knowledge.get(name)
                dtype = str(entry.get("diagnosis_type") or "").lower()
                if dtype in {"etiology", "metabolic", "structural", "systemic"}:
                    priority.append(name)
            return priority

        priority_expected = _priority_expected_names(expected)
        etiology_preference = (
            any(name in set(submitted) for name in priority_expected)
            if priority_expected
            else None
        )
        unauthorized_exam_count = sum(
            len(item.get("blocked_items") or [])
            for item in getattr(self, "_last_exam_authorization", []) or []
            if isinstance(item, dict)
        )
        evidence_payload = audit.get("evidence") or {}
        observations = [
            item
            for item in (evidence_payload.get("observations") or [])
            if isinstance(item, dict) and item.get("finding")
        ]
        evidence_compiler_audit = dict(audit.get("evidence_compiler") or {})
        reasoning_inference_findings = {
            str(item.get("finding"))
            for item in observations
            if item.get("source") == "reasoning_inference"
            and item.get("polarity", "positive") == "positive"
        }
        raw_case_findings = {
            str(item.get("finding"))
            for item in observations
            if item.get("source") == "raw_case_finding"
            and item.get("polarity", "positive") == "positive"
        }
        blocked_reasoning_inference_count = int(
            evidence_compiler_audit.get("blocked_reasoning_inference_count", 0) or 0
        )
        positive_findings = {
            str(item.get("finding"))
            for item in observations
            if item.get("polarity", "positive") == "positive"
        }
        negative_findings = {
            str(item.get("finding"))
            for item in observations
            if item.get("polarity") == "negative"
        }
        diagnostic_findings = {
            finding
            for finding in positive_findings
            if not finding.startswith(("field:", "symptom:"))
        }
        high_information_observations = [
            item
            for item in observations
            if item.get("polarity", "positive") == "positive"
            and not item.get("shadowed_by")
            and float(item.get("information_value") or 0.0) >= 0.75
        ]
        shadowed_observations = [
            item for item in observations if item.get("shadowed_by")
        ]
        information_values = [
            float(item.get("information_value") or 0.0)
            for item in observations
            if item.get("polarity", "positive") == "positive"
        ]
        high_information_findings = {
            str(item.get("finding"))
            for item in high_information_observations
            if item.get("finding")
        }
        shadowed_findings = [
            {
                "finding": str(item.get("finding") or ""),
                "shadowed_by": str(item.get("shadowed_by") or ""),
            }
            for item in shadowed_observations
            if item.get("finding")
        ]
        finding_extraction_summary = {
            "observation_count": len(observations),
            "positive_finding_count": len(positive_findings),
            "negative_finding_count": len(negative_findings),
            "diagnostic_finding_count": len(diagnostic_findings),
            "diagnostic_findings": sorted(diagnostic_findings)[:24],
            "high_information_finding_count": len(high_information_findings),
            "high_information_findings": sorted(high_information_findings)[:24],
            "generic_finding_shadowed_count": len(shadowed_observations),
            "shadowed_findings": shadowed_findings[:24],
            "reasoning_inference_finding_count": len(reasoning_inference_findings),
            "reasoning_inference_findings": sorted(reasoning_inference_findings)[:24],
            "raw_case_finding_count": len(raw_case_findings),
            "raw_case_findings": sorted(raw_case_findings)[:24],
            "blocked_reasoning_inference_count": blocked_reasoning_inference_count,
            "evidence_information_value_mean": (
                round(sum(information_values) / max(1, len(information_values)), 4)
                if information_values
                else None
            ),
        }
        top_candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        matched_count = len(top_candidate.get("matched_evidence") or [])
        gap_count = len(top_candidate.get("required_gaps") or [])
        if top_candidate:
            required_evidence_coverage = (
                1.0
                if top_candidate.get("required_met")
                else (
                    matched_count / max(1, matched_count + gap_count)
                    if matched_count or gap_count
                    else 0.0
                )
            )
        else:
            required_evidence_coverage = None
        soft_contradiction_count = sum(
            len(item.get("soft_contradicted_evidence") or [])
            for item in candidates
            if isinstance(item, dict)
        )
        hard_contradiction_count = sum(
            len(item.get("hard_contradicted_evidence") or [])
            for item in candidates
            if isinstance(item, dict)
        )
        exam_authorization_records = list(
            getattr(self, "_last_exam_authorization", []) or []
        )
        exam_authorization_mode = (
            "differential_driven"
            if any(
                isinstance(item, dict) and item.get("differential_driven")
                for item in exam_authorization_records
            )
            else (
                "strict_diagnosis_driven"
                if any(
                    isinstance(item, dict) and item.get("strict_diagnosis_driven")
                    for item in exam_authorization_records
                )
                else ("authorized" if exam_authorization_records else "not_recorded")
            )
        )
        ordered_exam_names = _names(final_result.get("ordered_examinations"))
        exam_authorization_details = [
            detail
            for record in exam_authorization_records
            if isinstance(record, dict)
            for detail in (record.get("exam_authorization_details") or [])
            if isinstance(detail, dict)
        ]
        exam_route_authorization_audit = [
            item
            for record in exam_authorization_records
            if isinstance(record, dict)
            for item in (record.get("exam_repeat_authorization_audit") or [])
            if isinstance(item, dict)
        ]
        ordered_exam_set = set(ordered_exam_names)
        ordered_authorization_details = [
            detail
            for detail in exam_authorization_details
            if str(detail.get("exam") or "") in ordered_exam_set
        ]
        deferred_ordered_details = [
            detail
            for detail in ordered_authorization_details
            if str(detail.get("exam_source") or "") == "deferred_gap_closure_exam"
            or bool(detail.get("priority_override"))
        ]
        gap_value_ordered_details = [
            detail
            for detail in ordered_authorization_details
            if bool(detail.get("score_gap_decoupled"))
            or float(detail.get("source_gap_value") or 0.0) > 0.0
        ]
        ordered_deferred_gap_ids = {
            str(gap_id or "")
            for detail in deferred_ordered_details
            for gap_id in detail.get("target_gaps", []) or []
            if str(gap_id or "")
        }
        top_gap_value = max(
            [float(gap.get("gap_value") or 0.0) for gap in active_evidence_gaps],
            default=0.0,
        )
        top_gap_ids = {
            str(gap.get("gap_id") or "")
            for gap in active_evidence_gaps
            if str(gap.get("gap_id") or "")
            and float(gap.get("gap_value") or 0.0) >= max(0.0, top_gap_value - 1e-9)
        }
        top_gap_ordered_ids = {
            str(gap_id or "")
            for detail in gap_value_ordered_details
            for gap_id in detail.get("target_gaps", []) or []
            if str(gap_id or "")
        }
        deferred_exam_coverage = (
            len(deferred_gap_target_ids & ordered_deferred_gap_ids)
            / max(1, len(deferred_gap_target_ids))
            if deferred_gap_target_ids
            else None
        )
        gap_value_exam_selection_rate = (
            len(gap_value_ordered_details) / max(1, len(ordered_exam_names))
            if ordered_exam_names and active_evidence_gaps
            else None
        )
        reserved_highest_gap_survival_rate = (
            len(top_gap_ids & top_gap_ordered_ids) / max(1, len(top_gap_ids))
            if top_gap_ids
            else None
        )
        exam_priority_alignment = (
            len(deferred_ordered_details) / max(1, len(ordered_exam_names))
            if ordered_exam_names and deferred_gap_target_ids
            else None
        )
        wrong_primary_exam_drift = (
            1.0 - float(exam_priority_alignment or 0.0)
            if ordered_exam_names and deferred_gap_target_ids
            else None
        )
        special_discriminator_rate = (
            sum(
                1
                for detail in ordered_authorization_details
                if str(detail.get("exam_type") or "") == "special_discriminator"
            )
            / max(1, len(ordered_authorization_details))
            if ordered_authorization_details
            else None
        )
        multi_candidate_exam_rate = (
            sum(
                1
                for detail in ordered_authorization_details
                if len(detail.get("target_candidates") or []) >= 2
            )
            / max(1, len(ordered_authorization_details))
            if ordered_authorization_details
            else None
        )
        generic_exam_suppression_count = sum(
            int(record.get("generic_exam_suppression_count", 0) or 0)
            for record in exam_authorization_records
            if isinstance(record, dict)
        )
        exam_route_blocked_count = sum(
            1 for item in exam_route_authorization_audit if item.get("blocked")
        )
        exam_route_repeat_authorized_count = sum(
            1
            for item in exam_route_authorization_audit
            if not item.get("blocked") and item.get("repeat_authorized")
        )
        exam_route_claim_resolved_block_count = sum(
            1
            for item in exam_route_authorization_audit
            if "CLAIM_ROUTE_ALREADY_RESOLVED" in (item.get("reason_codes") or [])
        )
        exam_route_generic_duplicate_block_count = sum(
            1
            for item in exam_route_authorization_audit
            if "GENERIC_WORKUP_DUPLICATE_BLOCKED" in (item.get("reason_codes") or [])
        )
        post_exam_primary_recomputed_rate = bool(
            any(
                isinstance(item, dict)
                and str(item.get("stage") or "") == "after_discriminating_exams"
                for item in dynamic_trace
            )
        )
        differential_source_items = {
            str(detail.get("exam") or "")
            for detail in exam_authorization_details
            if str(detail.get("exam_source") or "") == "judge_discriminating_exam"
        }
        authorized_source_items = {
            str(detail.get("exam") or "")
            for detail in exam_authorization_details
            if str(detail.get("exam") or "")
        }
        differential_exam_contribution_rate = (
            len(set(ordered_exam_names) & differential_source_items)
            / max(1, len(ordered_exam_names))
            if ordered_exam_names
            else None
        )
        legacy_exam_package_contribution_rate = (
            len(set(ordered_exam_names) - authorized_source_items)
            / max(1, len(ordered_exam_names))
            if ordered_exam_names and exam_authorization_mode == "differential_driven"
            else None
        )
        differential_exam_precision = (
            len(set(ordered_exam_names) & set(discriminating_exams))
            / max(1, len(ordered_exam_names))
            if ordered_exam_names and discriminating_exams
            else None
        )
        discriminating_exam_recall = (
            len(set(ordered_exam_names) & set(discriminating_exams))
            / max(1, len(discriminating_exams))
            if ordered_exam_names and discriminating_exams
            else None
        )
        exam_information_gain = (
            round(
                (
                    float(differential_exam_precision or 0.0)
                    + float(discriminating_exam_recall or 0.0)
                )
                / 2,
                4,
            )
            if differential_exam_precision is not None
            or discriminating_exam_recall is not None
            else None
        )
        discriminating_gap_closed_rate = (
            len(set(diagnostic_findings) & set(discriminating_findings))
            / max(1, len(discriminating_findings))
            if discriminating_findings
            else None
        )
        gap_closure_rate = discriminating_gap_closed_rate
        primary_candidate = next(
            (
                item
                for item in candidates
                if isinstance(item, dict)
                and item.get("diagnosis") == judge_primary
            ),
            top_candidate,
        ) or {}
        primary_matched_evidence = {
            str(item)
            for item in (primary_candidate.get("matched_evidence") or [])
            if str(item)
        }
        reasoning_inference_used_by_primary = bool(
            primary_matched_evidence & reasoning_inference_findings
        )

        def _component_value(candidate: Dict[str, Any], key: str) -> float:
            try:
                return float(
                    ((candidate.get("component_scores") or {}).get(key, 0.0))
                    or 0.0
                )
            except (AttributeError, TypeError, ValueError):
                return 0.0

        generic_penalized_candidates = [
            item
            for item in candidates
            if isinstance(item, dict)
            and _component_value(item, "generic_parent_penalty") > 0.0
        ]
        generic_primary_block_count = sum(
            1
            for item in generic_penalized_candidates
            if str(item.get("diagnosis") or "") not in set(authorized)
        )
        primary_core_evidence_score = _component_value(
            primary_candidate, "core_evidence_score"
        )
        primary_diagnostic_evidence_score = _component_value(
            primary_candidate, "diagnostic_evidence_score"
        )
        try:
            primary_core_coverage_value = float(
                primary_candidate.get(
                    "core_explanatory_coverage",
                    (primary_candidate.get("component_scores") or {}).get(
                        "core_explanatory_coverage", 0.0
                    ),
                )
                or 0.0
            )
        except (AttributeError, TypeError, ValueError):
            primary_core_coverage_value = 0.0
        specific_over_generic_preference_count = int(
            bool(primary_candidate)
            and (primary_core_evidence_score > 0.0 or primary_diagnostic_evidence_score > 0.0)
            and any(
                bool(item.get("required_met"))
                and str(item.get("diagnosis") or "") not in set(authorized)
                for item in generic_penalized_candidates
            )
        )
        core_evidence_primary_alignment = (
            primary_core_evidence_score > 0.0
            or primary_core_coverage_value >= 0.40
            if primary_candidate
            else None
        )
        diagnostic_evidence_primary_alignment = (
            primary_diagnostic_evidence_score > 0.0
            if primary_candidate
            else None
        )
        residual_core_penalty_applied_count = sum(
            1
            for item in candidates
            if isinstance(item, dict)
            and _component_value(item, "residual_core_penalty") > 0.0
        )
        explanatory_coverage = (
            judge_payload.get("explanatory_coverage")
            if judge_payload.get("explanatory_coverage") is not None
            else primary_candidate.get(
                "explanatory_coverage",
                primary_candidate.get("coverage_score"),
            )
        )
        core_explanatory_coverage = (
            judge_payload.get("core_explanatory_coverage")
            if judge_payload.get("core_explanatory_coverage") is not None
            else primary_candidate.get(
                "core_explanatory_coverage",
                (primary_candidate.get("component_scores") or {}).get(
                    "core_explanatory_coverage"
                ),
            )
        )
        residual_evidence_score = (
            judge_payload.get("residual_evidence_score")
            if judge_payload.get("residual_evidence_score") is not None
            else primary_candidate.get(
                "residual_evidence_score",
                primary_candidate.get("residual_score"),
            )
        )
        residual_core_evidence_count = (
            judge_payload.get("residual_core_evidence_count")
            if judge_payload.get("residual_core_evidence_count") is not None
            else primary_candidate.get(
                "residual_core_evidence_count",
                (primary_candidate.get("component_scores") or {}).get(
                    "residual_core_evidence_count"
                ),
            )
        )
        expected_set = set(expected)
        differential_candidate_set = set(differential_candidates)
        differential_pool_expected_included = (
            bool(expected_set & differential_candidate_set)
            if expected_set and differential_candidates
            else None
        )
        differential_pool_precision = (
            len(expected_set & differential_candidate_set)
            / max(1, len(differential_candidates))
            if expected_set and differential_candidates
            else None
        )
        pairwise_noise_rejection_count = pool_filter_summary.get(
            "pairwise_noise_rejection_count"
        )
        cluster_gate_rejection_count = pool_filter_summary.get(
            "cluster_gate_rejection_count"
        )
        core_evidence_coverage = pool_filter_summary.get("core_evidence_coverage")
        pairwise_relevant = [
            item
            for item in pairwise_comparisons
            if isinstance(item, dict)
            and (
                str(item.get("left") or "") in expected_set
                or str(item.get("right") or "") in expected_set
            )
        ]
        pairwise_judge_accuracy = (
            sum(
                1
                for item in pairwise_relevant
                if str(item.get("preferred") or "") in expected_set
            )
            / max(1, len(pairwise_relevant))
            if expected_set and pairwise_relevant
            else None
        )
        critic = audit.get("critic") or {}
        elapsed = audit.get(
            "training_elapsed_seconds",
            audit.get("elapsed_seconds", final_result.get("_case_elapsed_seconds")),
        )
        try:
            elapsed = float(elapsed) if elapsed is not None else None
        except (TypeError, ValueError):
            elapsed = None

        targeted_bindings = [
            item
            for item in getattr(self, "_exam_result_intent_bindings", []) or []
            if isinstance(item, dict)
        ]
        targeted_parses = [
            item
            for item in getattr(self, "_targeted_exam_result_parses", []) or []
            if isinstance(item, dict)
        ]
        bound_parse_count = sum(
            1
            for item in targeted_parses
            if str(item.get("binding_status") or "") == "bound"
        )
        targeted_parser_count = sum(
            1
            for item in targeted_parses
            if str(item.get("binding_status") or "") == "bound"
            and str(item.get("status") or "") not in {"unbound", ""}
        )
        targeted_recovery_count = sum(
            1
            for item in targeted_parses
            if str(item.get("status") or "") in {"positive", "negative"}
        )
        targeted_gap_closed_count = sum(
            1
            for item in targeted_parses
            if str(item.get("gap_closure_assessment") or "")
            in {"positive_closed", "negative_closed"}
        )
        claim_events = [
            item
            for item in getattr(self, "_claim_match_events", []) or []
            if isinstance(item, dict)
        ]
        claim_update_audit = [
            item
            for item in getattr(self, "_claim_resolution_update_audit", []) or []
            if isinstance(item, dict)
        ]
        resolvable_claim_match_count = sum(
            1
            for item in claim_events
            if str(item.get("match_status") or "") in {"SUPPORTED", "CONTRADICTED"}
        )
        persisted_claim_resolution_delta_count = sum(
            int(item.get("claim_state_version_delta") or 0)
            for item in claim_update_audit
        )
        claim_resolution_writeback_missing_count = int(
            bool(resolvable_claim_match_count and persisted_claim_resolution_delta_count == 0)
        )
        silent_exam_substitution_count = sum(
            1
            for item in targeted_bindings
            if str(item.get("requested_exam") or "")
            and str(item.get("actual_result_exam") or "")
            and str(item.get("requested_exam") or "") != str(item.get("actual_result_exam") or "")
            and not str(item.get("actual_closure_level") or "")
        )
        unverified_exam_evidence_leakage_count = sum(
            1
            for item in observations
            if item.get("source") == "targeted_exam_result_parser"
            and not (
                item.get("target_gap_ids")
                and item.get("order_id")
                and item.get("verification_method") == "targeted_exam_result_parser"
            )
        )

        public_final = {
            key: final_result.get(key)
            for key in (
                "patient_id",
                "caseId",
                "diagnosis",
                "treatment_plan",
                "reasoning",
                "ordered_examinations",
                "conversation_rounds",
                "finished",
            )
            if key in final_result
        }
        pattern_recall_audit = (
            decision.get("pattern_recall_audit")
            if isinstance(decision, dict)
            else {}
        ) or {}
        pattern_pipeline_audit = (
            pattern_recall_audit.get("pattern_pipeline_audit")
            if isinstance(pattern_recall_audit, dict)
            else {}
        ) or {}
        policy_summary = (
            self.candidate_policy_store.summary()
            if getattr(self, "candidate_policy_store", None) is not None
            else {}
        )
        failure_attribution = self._deterministic_failure_attribution(
            report=report,
            runtime_audit=runtime_audit,
            llm_call_audit=list(self._llm_call_audit),
            expected=expected,
            top_twenty=top_twenty,
            submitted=submitted,
        )
        treatment_strategy = (
            final_result.get("_treatment_strategy")
            if isinstance(final_result, dict)
            and isinstance(final_result.get("_treatment_strategy"), dict)
            else {}
        )
        treatment_protocol_coverage_rate = float(
            treatment_strategy.get("treatment_protocol_coverage_rate") or 0.0
        )
        treatment_uncovered_diagnosis_count = len(
            treatment_strategy.get("uncovered_diagnoses") or []
        )
        treatment_actionability_section_count = len(
            treatment_strategy.get("actionability_sections") or []
        )
        return {
            "patient_id": patient_id,
            "status": "evaluated" if report else "evaluation_failed",
            "final_result": public_final,
            "expected_diagnoses": expected,
            "submitted_diagnoses": submitted,
            "error_types": self._classify_diagnosis_errors(report) if report else [],
            "metrics": {
                "diagnosis_accuracy": _metric("diagnosisAccuracy", "diagnosis_accuracy"),
                "examination_precision": _metric(
                    "examinationPrecision", "examination_precision"
                ),
                "treatment_overall_score": _metric(
                    "treatmentOverallScore", "treatment_overall_score"
                ),
                "treatment_safety": _metric("treatmentSafety", "treatment_safety"),
                "treatment_protocol_coverage_rate": treatment_protocol_coverage_rate,
                "treatment_uncovered_diagnosis_count": treatment_uncovered_diagnosis_count,
                "treatment_actionability_section_count": treatment_actionability_section_count,
                "candidate_recall_at_20": recall_at_twenty,
                "candidate_recall_at_5": recall_at_five,
                "ranking_accuracy": ranking_accuracy,
                "submission_alignment": submission_alignment,
                "submission_override_count": int(
                    decision.get("submission_override_count", 0) or 0
                ) if isinstance(decision, dict) else 0,
                "etiology_preference": etiology_preference,
                "decision_override_rate": decision_override_rate,
                "judge_gap_authorization_rate": judge_gap_authorization_rate,
                "required_gap_authorized_count": 0,
                "primary_eligible_count": eligibility_distribution.get(PRIMARY_ELIGIBLE),
                "deferred_needs_anchor_count": eligibility_distribution.get(DEFERRED),
                "differential_only_count": eligibility_distribution.get(DIFFERENTIAL_ONLY),
                "excluded_count": eligibility_distribution.get(EXCLUDED),
                "judge_primary_accuracy": judge_primary_accuracy,
                "explanatory_coverage": explanatory_coverage,
                "core_explanatory_coverage": core_explanatory_coverage,
                "residual_evidence_score": residual_evidence_score,
                "residual_core_evidence_count": residual_core_evidence_count,
                "differential_exam_precision": differential_exam_precision,
                "discriminating_exam_recall": discriminating_exam_recall,
                "exam_information_gain": exam_information_gain,
                "deferred_gap_closure_rate": deferred_exam_coverage,
                "deferred_exam_coverage": deferred_exam_coverage,
                "gap_value_exam_selection_rate": gap_value_exam_selection_rate,
                "reserved_highest_gap_survival_rate": reserved_highest_gap_survival_rate,
                "exam_priority_alignment": exam_priority_alignment,
                "wrong_primary_exam_drift": wrong_primary_exam_drift,
                "deferred_gap_count": len(deferred_evidence_gaps),
                "exam_priority_override_count": len(exam_priority_overrides),
                "special_discriminator_rate": special_discriminator_rate,
                "multi_candidate_exam_rate": multi_candidate_exam_rate,
                "generic_exam_suppression_count": generic_exam_suppression_count,
                "exam_route_authorization_blocked_count": exam_route_blocked_count,
                "exam_route_repeat_authorized_count": exam_route_repeat_authorized_count,
                "exam_route_claim_resolved_block_count": exam_route_claim_resolved_block_count,
                "exam_route_generic_duplicate_block_count": exam_route_generic_duplicate_block_count,
                "post_exam_primary_recomputed_rate": post_exam_primary_recomputed_rate,
                "discriminating_gap_closed_rate": discriminating_gap_closed_rate,
                "gap_closure_rate": gap_closure_rate,
                "dynamic_rerank_changed_primary": dynamic_rerank_changed_primary,
                "explanation_score_changed_ranking_rate": explanation_score_changed_ranking,
                "primary_unlock_rate": bool(primary_unlock_reason),
                "legacy_exam_package_contribution_rate": legacy_exam_package_contribution_rate,
                "differential_exam_contribution_rate": differential_exam_contribution_rate,
                "gap_state_satisfied_count": gap_state_distribution.get("satisfied"),
                "gap_state_actionable_count": gap_state_distribution.get("actionable_gap"),
                "gap_state_nonblocking_count": gap_state_distribution.get("nonblocking_gap"),
                "gap_state_unsupported_count": gap_state_distribution.get("unsupported_gap"),
                "gap_state_hard_blocked_count": (
                    gap_state_distribution.get("hard_contradiction")
                    or gap_state_distribution.get("hard_blocked")
                ),
                "gap_state_partially_satisfied_count": gap_state_distribution.get(
                    "partially_satisfied"
                ),
                "fallback_to_pre_discrimination_primary": bool(
                    judge_payload.get("fallback_to_pre_discrimination_primary")
                ),
                "pairwise_judge_accuracy": pairwise_judge_accuracy,
                "differential_pool_precision": differential_pool_precision,
                "differential_pool_expected_included": differential_pool_expected_included,
                "generic_primary_block_count": generic_primary_block_count,
                "specific_over_generic_preference_count": specific_over_generic_preference_count,
                "core_evidence_primary_alignment": core_evidence_primary_alignment,
                "diagnostic_evidence_primary_alignment": diagnostic_evidence_primary_alignment,
                "residual_core_penalty_applied_count": residual_core_penalty_applied_count,
                "pairwise_noise_rejection_count": pairwise_noise_rejection_count,
                "cluster_gate_rejection_count": cluster_gate_rejection_count,
                "core_evidence_coverage": core_evidence_coverage,
                "judge_deferred_primary": bool(
                    judge_payload.get("needs_discriminating_exams")
                ),
                "unauthorized_exam_count": unauthorized_exam_count,
                "required_evidence_coverage": required_evidence_coverage,
                "soft_contradiction_count": soft_contradiction_count,
                "hard_contradiction_count": hard_contradiction_count,
                "high_information_finding_count": len(high_information_findings),
                "generic_finding_shadowed_count": len(shadowed_observations),
                "reasoning_inference_finding_count": len(reasoning_inference_findings),
                "raw_case_finding_count": len(raw_case_findings),
                "reasoning_inference_used_by_primary": reasoning_inference_used_by_primary,
                "blocked_reasoning_inference_count": blocked_reasoning_inference_count,
                "evidence_information_value_mean": finding_extraction_summary[
                    "evidence_information_value_mean"
                ],
                "generic_only_candidate_count": pool_filter_summary.get(
                    "generic_only_candidate_count"
                ),
                "reasoning_structured_conflict_count": reasoning_structured_conflict_count,
                "conflict_deferred_primary_count": conflict_deferred_primary_count,
                "conflict_blocked_final_count": conflict_blocked_final_count,
                "root_cause_arbitration_count": root_cause_arbitration_count,
                "root_cause_primary_override_count": root_cause_primary_override_count,
                "root_cause_secondary_submission_count": root_cause_secondary_submission_count,
                "root_cause_coverage": root_cause_coverage,
                "candidate_policy_count": policy_summary.get("candidate_policy_count"),
                "policy_promotion_count": policy_summary.get("policy_promotion_count"),
                "policy_quarantine_count": policy_summary.get("policy_quarantine_count"),
                "policy_rejected_count": policy_summary.get("policy_rejected_count"),
                "policy_conflict_count": policy_summary.get("policy_conflict_count"),
                "failure_stage_distribution": policy_summary.get(
                    "failure_stage_distribution"
                ),
                "evidence_hypothesis_count": case_board_audit.get(
                    "evidence_hypothesis_count"
                ),
                "evidence_query_task_count": case_board_audit.get("query_task_count"),
                "evidence_hypothesis_verification_rate": case_board_audit.get(
                    "evidence_hypothesis_verification_rate"
                ),
                "evidence_recovery_count": case_board_audit.get(
                    "evidence_recovery_count"
                ),
                "evidence_recovery_rate": case_board_audit.get(
                    "evidence_recovery_rate"
                ),
                "false_evidence_injection_rate": case_board_audit.get(
                    "false_evidence_injection_rate"
                ),
                "unverified_evidence_leakage": case_board_audit.get(
                    "unverified_evidence_leakage"
                ),
                "conflict_closure_rate": case_board_audit.get(
                    "conflict_closure_rate"
                ),
                "protected_candidate_rescue_count": case_board_audit.get(
                    "protected_candidate_rescue_count"
                ),
                "derived_pattern_count": case_board_audit.get(
                    "derived_pattern_count"
                ),
                "targeted_exam_result_binding_rate": (
                    bound_parse_count / max(1, len(targeted_parses))
                    if targeted_parses
                    else None
                ),
                "targeted_parser_coverage": (
                    targeted_parser_count / max(1, bound_parse_count)
                    if bound_parse_count
                    else None
                ),
                "targeted_gap_evidence_recovery_rate": (
                    targeted_recovery_count / max(1, targeted_parser_count)
                    if targeted_parser_count
                    else None
                ),
                "gap_closure_after_result_rate": (
                    targeted_gap_closed_count / max(1, targeted_parser_count)
                    if targeted_parser_count
                    else None
                ),
                "claim_match_event_count": len(claim_events),
                "resolvable_claim_match_count": resolvable_claim_match_count,
                "persisted_claim_resolution_delta_count": persisted_claim_resolution_delta_count,
                "parser_to_claim_ledger_writeback_rate": (
                    persisted_claim_resolution_delta_count
                    / max(1, resolvable_claim_match_count)
                    if resolvable_claim_match_count
                    else None
                ),
                "claim_resolution_writeback_missing_count": (
                    claim_resolution_writeback_missing_count
                ),
                "claim_state_version": int(getattr(self, "_claim_state_version", 0) or 0),
                "diagnostic_state_version": int(
                    getattr(self, "_diagnostic_state_version", 0) or 0
                ),
                "pavm_anchor_recovery_count": sum(
                    1
                    for item in observations
                    if item.get("source") == "targeted_exam_result_parser"
                    and item.get("finding")
                    in {
                        "pulmonary_cta_positive",
                        "enhanced_ct_vascular_malformation",
                        "bubble_echo_right_to_left_shunt",
                    }
                ),
                "unverified_exam_evidence_leakage_count": unverified_exam_evidence_leakage_count,
                "silent_exam_substitution_count": silent_exam_substitution_count,
            },
            "top_candidates": top_twenty,
            "retriever_top1": retriever_top1,
            "judge_primary": judge_primary,
            "judge_primary_status": str(judge_payload.get("primary_status") or ""),
            "submitter_final": submitter_final,
            "required_gap_authorized_diagnoses": required_gap_authorized_diagnoses,
            "authorized_diagnoses": authorized,
            "blocked_diagnoses": blocked_records
            if isinstance(decision, dict)
            else [],
            "audit": {
                "elapsed_seconds": elapsed,
                "timed_out": bool(
                    audit.get("timed_out", final_result.get("_case_timed_out", False))
                    or "exceeded" in evaluation_error
                    or "exceeded" in reflection_error
                    or (
                        self.case_timeout_seconds > 0
                        and elapsed is not None
                        and elapsed >= self.case_timeout_seconds
                    )
                ),
                "critic_issues": list(critic.get("issues") or []),
                "critic_llm_used": bool(critic.get("llm_used", False)),
                "llm_calls": self._llm_call_count,
                "llm_calls_by_kind": dict(self._llm_call_by_kind),
                "llm_call_audit": list(self._llm_call_audit),
                "llm_contract_summary": self._llm_contract_summary_from_audit(
                    list(self._llm_call_audit)
                ),
                "llm_context_audit": list(runtime_audit.get("llm_context_audit") or []),
                "tool_call_audit": list(runtime_audit.get("tool_call_audit") or []),
                "tool_contract_summary": dict(
                    runtime_audit.get("tool_contract_summary") or {}
                ),
                "failure_attribution": failure_attribution,
                "treatment_strategy": treatment_strategy,
                "exam_authorization": exam_authorization_records,
                "exam_authorization_mode": exam_authorization_mode,
                "exam_route_authorization_audit": exam_route_authorization_audit,
                "exam_result_intent_bindings": targeted_bindings,
                "exam_execution_resolution": targeted_bindings,
                "targeted_exam_result_parses": targeted_parses,
                "claim_resolution_ledger": normalize_ledger(
                    getattr(self, "_claim_resolution_ledger", {}) or {}
                ),
                "claim_match_events": claim_events,
                "claim_resolution_update_audit": claim_update_audit,
                "claim_state_version": int(getattr(self, "_claim_state_version", 0) or 0),
                "diagnostic_state_version": int(
                    getattr(self, "_diagnostic_state_version", 0) or 0
                ),
                "evidence_version": runtime_audit.get("evidence_version"),
                "exam_result_applicability": list(
                    runtime_audit.get("exam_result_applicability") or []
                ),
                "targeted_exam_observations": list(
                    runtime_audit.get("targeted_exam_observations") or []
                ),
                "gap_state": list(runtime_audit.get("gap_state") or []),
                "anchor_state": list(runtime_audit.get("anchor_state") or []),
                "eligibility_state": list(runtime_audit.get("eligibility_state") or []),
                "last_completed_stage": runtime_audit.get("last_completed_stage"),
                "failure_stage": runtime_audit.get("failure_stage"),
                "last_successful_clinical_transition": dict(
                    runtime_audit.get("last_successful_clinical_transition") or {}
                ),
                "clinical_transition_trace": list(
                    runtime_audit.get("clinical_transition_trace") or []
                ),
                "gap_evidence_recovery": [
                    {
                        "binding_id": item.get("binding_id"),
                        "order_id": item.get("order_id"),
                        "target_gap_ids": item.get("target_gap_ids"),
                        "status": item.get("status"),
                        "gap_closure_assessment": item.get("gap_closure_assessment"),
                        "observed_findings": [
                            obs.get("finding")
                            for obs in item.get("observations", []) or []
                            if isinstance(obs, dict)
                        ],
                    }
                    for item in targeted_parses
                ],
                "unbound_gap_exam_results": [
                    item
                    for item in targeted_parses
                    if str(item.get("binding_status") or "") != "bound"
                ],
                "unresolved_targeted_exam_results": [
                    item
                    for item in targeted_parses
                    if str(item.get("status") or "") in {"unresolved", "inconclusive"}
                ],
                "finding_extraction_summary": finding_extraction_summary,
                "evidence_compiler": evidence_compiler_audit,
                "case_board_evidence": case_board_audit,
                "pattern_recall_audit": pattern_recall_audit,
                "pattern_pipeline_audit": pattern_pipeline_audit,
                "pairwise_comparison_count": len(pairwise_comparisons),
                "judge_primary_status": str(judge_payload.get("primary_status") or ""),
                "needs_discriminating_exams": bool(
                    judge_payload.get("needs_discriminating_exams")
                ),
                "provisional_primary": str(
                    judge_payload.get("provisional_primary") or ""
                ),
                "locked_primary": str(judge_payload.get("locked_primary") or ""),
                "defer_reason": str(judge_payload.get("defer_reason") or ""),
                "differential_candidates": differential_candidates,
                "excluded_from_pairwise": list(
                    judge_payload.get("excluded_from_pairwise") or []
                ),
                "pool_filter_reasons": dict(
                    judge_payload.get("pool_filter_reasons") or {}
                ),
                "cluster_assignments": dict(
                    judge_payload.get("cluster_assignments") or {}
                ),
                "pairwise_allowed_matrix": list(
                    judge_payload.get("pairwise_allowed_matrix") or []
                ),
                "clinical_reasoning_comparisons": list(
                    judge_payload.get("clinical_reasoning_comparisons") or []
                ),
                "primary_arbitration_candidates": list(
                    judge_payload.get("primary_arbitration_candidates") or []
                ),
                "primary_arbitration_decision": dict(
                    judge_payload.get("primary_arbitration_decision") or {}
                ),
                "primary_anchor_revalidation": dict(
                    judge_payload.get("primary_anchor_revalidation") or {}
                ),
                "arbitration_winner": str(judge_payload.get("arbitration_winner") or ""),
                "arbitration_loser": str(judge_payload.get("arbitration_loser") or ""),
                "arbitration_action": str(judge_payload.get("arbitration_action") or ""),
                "arbitration_reason_codes": list(
                    judge_payload.get("arbitration_reason_codes") or []
                ),
                "pairwise_discriminating_gaps": list(
                    judge_payload.get("pairwise_discriminating_gaps") or []
                ),
                "pool_filter_summary": pool_filter_summary,
                "discriminating_exams": discriminating_exams,
                "discriminating_exam_tasks": list(
                    judge_payload.get("discriminating_exam_tasks") or []
                ),
                "discriminating_findings": discriminating_findings,
                "primary_unlock_reason": primary_unlock_reason,
                "explanation_score_changed_ranking": explanation_score_changed_ranking,
                "gap_state_distribution": gap_state_distribution,
                "eligibility_distribution": eligibility_distribution,
                "primary_eligible_candidates": _names(
                    decision.get("primary_eligible_candidates")
                    or judge_payload.get("primary_eligible_candidates")
                    or []
                ),
                "deferred_anchor_candidates": _names(
                    decision.get("deferred_anchor_candidates")
                    or judge_payload.get("deferred_anchor_candidates")
                    or []
                ),
                "excluded_candidates": _names(
                    decision.get("excluded_candidates")
                    or judge_payload.get("excluded_candidates")
                    or []
                ),
                "explanatory_coverage": explanatory_coverage,
                "core_explanatory_coverage": core_explanatory_coverage,
                "residual_evidence_score": residual_evidence_score,
                "residual_core_evidence_count": residual_core_evidence_count,
                "high_value_gap_candidates": _names(
                    judge_payload.get("high_value_gap_candidates")
                ),
                "deferred_evidence_gaps": deferred_evidence_gaps,
                "exam_priority_overrides": exam_priority_overrides,
                "deferred_gap_closure_tasks": deferred_gap_closure_tasks,
                "active_evidence_gaps": active_evidence_gaps,
                "deferred_exam_coverage": deferred_exam_coverage,
                "gap_value_exam_selection_rate": gap_value_exam_selection_rate,
                "reserved_highest_gap_survival_rate": reserved_highest_gap_survival_rate,
                "exam_priority_alignment": exam_priority_alignment,
                "wrong_primary_exam_drift": wrong_primary_exam_drift,
                "deferred_substatus_distribution": dict(
                    judge_payload.get("deferred_substatus_distribution") or {}
                ),
                "root_cause_arbitration": root_cause_payload,
                "root_cause_primary": str(
                    root_cause_payload.get("root_cause_primary")
                    or judge_payload.get("root_cause_primary")
                    or ""
                ),
                "root_cause_secondary": root_cause_secondary,
                "candidate_explanation_edges": list(
                    root_cause_payload.get("candidate_explanation_edges")
                    or judge_payload.get("candidate_explanation_edges")
                    or []
                ),
            },
            "evaluation_error": evaluation_error,
            "reflection_error": reflection_error,
        }

    # ============ 测试流程（规划驱动） ============

    async def test(self, patient_id: str) -> None:
        """测试流程：使用规划器驱动诊疗并提交结果。

        Args:
            patient_id: 患者 ID
        """
        logger.info(f"[Test] 开始测试患者: {patient_id}")
        trace = getattr(self, "trace_collector", None)
        case_span_id = None
        if trace and trace.enabled:
            trace.start_trace(
                patient_id,
                {
                    "mode": "test",
                    "agent": self.__class__.__name__,
                    "fast_mode": bool(self.fast_mode),
                    "diagnosis_chain_enabled": bool(self.diagnosis_chain_enabled),
                },
            )
            case_span_id = trace.start_span(
                "case_orchestrator",
                self.__class__.__name__,
                "test_case",
            )
            trace.create_artifact(ArtifactType.RAW_CASE, {"patient_id": patient_id})
        self._reset_llm_counter()
        # 跨患者软复用：保留 planner 中的经验教训种子
        if self._planner is not None:
            self._planner.soft_reset(keep_lessons=True)

        # 规划器驱动诊疗
        try:
            final_result = await self._run_case_pipeline(patient_id)
        except Exception as exc:
            if trace and trace.enabled:
                trace.end_span(case_span_id, status="failed", payload={"error": str(exc)})
                trace.fail_trace(exc)
            raise

        # 保存测试结果供 run_test 收集
        planner = self._get_planner()
        _rounds = int(
            final_result.get("conversation_rounds")
            or len([a for a in planner.action_history if a["type"] == "ask_patient"])
        )
        final_result = self.quality_agent.review_final_result(
            final_result,
            collected_info=getattr(self, "_last_collected_info", {}),
            exam_results=getattr(self, "_last_exam_results", {}),
            conversation_rounds=_rounds,
        )
        _dx = final_result.get("diagnosis", [])
        _tp = final_result.get("treatment_plan", "")
        # 关键：final_result / final_results 双字段，兼容评测器的字段名约定
        _final_payload = {
            "patient_id": patient_id,
            "caseId": patient_id,
            "diagnosis": _dx,
            "treatment_plan": _tp,
            "reasoning": final_result.get("reasoning", "") if isinstance(final_result, dict) else "",
            "conversation_rounds": _rounds,
            "ordered_examinations": final_result.get("ordered_examinations", []) if isinstance(final_result, dict) else [],
            "finished": bool(final_result.get("finished", True)) if isinstance(final_result, dict) else True,
        }
        self._last_test_result = {
            "final_result": _final_payload,
            "final_results": [_final_payload],
            "caseId": patient_id,
            "diagnosis": _dx,
            "treatment_plan": _tp,
            "reasoning": _final_payload["reasoning"],
            "conversation_rounds": _rounds,
            "ordered_examinations": _final_payload["ordered_examinations"],
            "finished": _final_payload["finished"],
        }
        if trace and trace.enabled:
            self._emit_trace_case_artifacts(patient_id, _final_payload)
            trace.end_span(case_span_id, status="success")
            trace.complete_trace(_final_payload)

        logger.info(
            f"[Test] 完成测试患者: {patient_id}, 结果: "
            f"{json.dumps(final_result, ensure_ascii=False)}"
        )
        logger.info(
            f"[Test] LLM 调用统计: 总{self._llm_call_count}次, "
            f"明细={json.dumps(self._llm_call_by_kind, ensure_ascii=False)}"
        )

    # ============ 初始问诊 ============

    async def _initial_inquiry(
        self,
        patient_id: str,
        chat_history: List[Dict[str, str]],
        relevant_experience: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """初始问诊：收集主诉和基本病史。

        Args:
            patient_id: 患者 ID
            chat_history: 对话历史
            relevant_experience: 相关历史经验

        Returns:
            收集到的患者信息
        """
        # 检索相关经验
        # 初始阶段还没有症状信息，使用空列表
        # 经验将在后续阶段注入

        # 构建初始问诊 prompt
        system_prompt = self.prompt.build_initial_inquiry_prompt(relevant_experience)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请生成一个初始问诊问题，引导患者描述主要症状。"},
        ]

        # 使用 LLM 生成初始问题
        question = await self._llm_chat(
            messages,
            purpose="inquiry_question",
        )
        if not question:
            question = "请描述这次最主要的不适、开始时间和伴随症状。"

        logger.info(f"[问诊] 初始问诊问题: {question}")

        # 询问患者
        answer = await self.actions.ask_patient(
            patient_id=patient_id,
            input_data={
                "question": question,
                "chat_history": chat_history,
            },
        )

        chat_history.append({"from": "doctor", "text": question})
        chat_history.append({"from": "patient", "text": answer})

        # 使用 LLM 结构化提取患者回复
        collected_info = await self._extract_patient_info(answer, {})
        logger.info(f"[问诊] 初始问诊回复: {answer[:200]}...")
        logger.info(f"[问诊] 提取信息: {json.dumps(collected_info, ensure_ascii=False, indent=2)[:200]}...")

        return collected_info

    # ============ 追问细节 ============

    async def _follow_up_inquiry(
        self,
        patient_id: str,
        chat_history: List[Dict[str, str]],
        collected_info: Dict[str, Any],
        relevant_experience: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """追问细节：根据初始问诊结果，追问病史和症状细节。

        Args:
            patient_id: 患者 ID
            chat_history: 对话历史
            collected_info: 已收集的信息
            relevant_experience: 相关历史经验

        Returns:
            更新后的患者信息
        """
        # 基于当前症状检索相关经验
        symptoms = collected_info.get("symptoms", [])
        if symptoms:
            relevant_experience = self.memory.search_relevant_experience(symptoms, top_k=3)

        for round_idx in range(self.max_ask_rounds - 1):
            # 思考：生成鉴别诊断，判断信息是否足够
            thinking = await self._think(
                collected_info, {}, chat_history, "inquiry", relevant_experience
            )
            _cands_follow = None
            if thinking and isinstance(thinking, dict):
                _cands_follow = thinking.get("differential_diagnosis") or thinking.get("candidate_diseases")
            inquiry_strategy = self.inquiry_agent.recommend(
                collected_info=collected_info,
                candidate_diseases=_cands_follow if isinstance(_cands_follow, list) else None,
            )

            # 判断信息是否已足够（从思考结果中获取，回退到旧方法）
            if thinking and "is_sufficient" in thinking:
                is_sufficient = thinking["is_sufficient"]
                if thinking.get("key_unknowns"):
                    logger.info(f"[问诊] 信息缺口: {thinking['key_unknowns']}")
            else:
                is_sufficient = await self._check_info_sufficient(
                    collected_info, round_idx + 2, self.max_ask_rounds
                )

            if is_sufficient:
                logger.info(f"[问诊] 信息已足够，结束问诊")
                break

            # 构建追问 prompt（注入思考结果）
            follow_up_prompt = self.prompt.build_follow_up_prompt(
                collected_info=collected_info,
                chat_history=chat_history,
                relevant_experience=relevant_experience,
                thinking=thinking,
            )
            messages = [
                {"role": "system", "content": follow_up_prompt},
                {
                    "role": "user",
                    "content": (
                        "请生成一个追问问题。"
                        f"优先覆盖这些关键追问：{inquiry_strategy.get('questions', [])}。"
                        f"注意排查红旗信号：{inquiry_strategy.get('red_flags', [])}。"
                        "如果信息已足够，请返回空字符串。"
                    ),
                },
            ]

            # 使用 LLM 生成追问问题
            question = await self._llm_chat(
                messages,
                temperature=0.5,
                purpose="inquiry_question",
            )

            if not question or question.strip() == "":
                logger.info(f"[问诊] LLM 判断无需追问，结束问诊")
                break

            logger.info(f"[问诊] 第{round_idx + 2}轮追问: {question}")

            answer = await self.actions.ask_patient(
                patient_id=patient_id,
                input_data={
                    "question": question,
                    "chat_history": chat_history,
                },
            )

            chat_history.append({"from": "doctor", "text": question})
            chat_history.append({"from": "patient", "text": answer})

            # 使用 LLM 结构化提取追问回复
            collected_info = await self._extract_patient_info(answer, collected_info)

            logger.info(f"[问诊] 第{round_idx + 2}轮追问回复: {answer[:200]}...")

        return collected_info

    # ============ 申请检查 ============

    async def _order_examinations(
        self,
        patient_id: str,
        collected_info: Dict[str, Any],
        relevant_experience: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """根据问诊信息申请检查。

        Args:
            patient_id: 患者 ID
            collected_info: 收集到的患者信息
            relevant_experience: 相关历史经验

        Returns:
            检查结果
        """
        # 基于症状检索相关经验
        symptoms = collected_info.get("symptoms", [])
        if symptoms:
            relevant_experience = self.memory.search_relevant_experience(symptoms, top_k=3)

        exam_results = {}

        for exam_round in range(self.max_exam_rounds):
            # 思考：更新鉴别诊断，判断检查是否足够
            thinking = await self._think(
                collected_info, exam_results, [], "examination", relevant_experience
            )

            # 判断检查是否已足够（从思考结果中获取，回退到旧方法）
            if exam_results:
                if thinking and "is_sufficient" in thinking:
                    is_sufficient = thinking["is_sufficient"]
                    if thinking.get("key_unknowns"):
                        logger.info(f"[检查] 建议补充: {thinking['key_unknowns']}")
                else:
                    is_sufficient = await self._check_exam_sufficient(
                        collected_info, exam_results, exam_round + 1, self.max_exam_rounds
                    )
                if is_sufficient:
                    logger.info(f"[检查] 检查已足够，结束检查")
                    break

            # 从 knowledge 构建 RAG 上下文
            _sym2 = collected_info.get("symptoms") or []
            _cands2 = None
            if thinking and isinstance(thinking, dict):
                _cands2 = thinking.get("differential_diagnosis") or thinking.get("candidate_diseases")
            pre_exam_judge = self._pre_exam_judge_payload(
                collected_info,
                exam_results,
                thinking=thinking if isinstance(thinking, dict) else None,
            )
            if pre_exam_judge.get("differential_candidates"):
                _cands2 = [
                    str(item).strip()
                    for item in pre_exam_judge.get("differential_candidates") or []
                    if str(item).strip()
                ]
            try:
                knowledge_context2 = self.knowledge.build_rag_context(_sym2, _cands2)
            except Exception:
                knowledge_context2 = ""
            try:
                if getattr(self, "memory_manager", None):
                    memory_context2 = self.memory_manager.build_semantic_context(
                        collected_info, _cands2
                    )
                    if memory_context2:
                        knowledge_context2 = memory_context2
            except Exception:
                pass

            # 构建检查申请 prompt（注入思考结果）
            exam_prompt = self.prompt.build_examination_prompt(
                collected_info=collected_info,
                exam_results=exam_results,
                relevant_experience=relevant_experience,
                thinking=thinking,
                knowledge_context=knowledge_context2,
            )
            messages = [
                {"role": "system", "content": exam_prompt},
                {"role": "user", "content": "请输出需要申请的检查项目列表（JSON 数组格式）。"},
            ]

            # 使用 LLM 生成检查项目
            exam_items = await self._llm_generate_examination_items(messages, collected_info)
            strategy = self.exam_agent.recommend(
                collected_info=collected_info,
                candidate_diseases=_cands2 if isinstance(_cands2, list) else None,
                proposed_items=exam_items,
                existing_results=exam_results,
                judge_decision=pre_exam_judge or None,
            )
            if strategy.get("strong_verification_items"):
                logger.info(f"[检查策略] 强验证检查: {strategy['strong_verification_items']}")
            if strategy.get("red_flag_items"):
                logger.info(f"[检查策略] 红旗补查检查: {strategy['red_flag_items']}")
            if strategy.get("evidence_driven_items"):
                logger.info(f"[检查策略] 证据驱动补查检查: {strategy['evidence_driven_items']}")
            if strategy.get("added_required"):
                logger.info(f"[检查策略] 补齐必查检查: {strategy['added_required']}")
            if strategy.get("invalid_items"):
                logger.info(f"[检查策略] 过滤无效检查项: {strategy['invalid_items']}")
            if (
                strategy.get("strict_diagnosis_driven")
                or strategy.get("differential_driven")
                or strategy.get("blocked_items")
            ):
                self._last_exam_authorization.append(
                    {
                        "stage": "initial_exam",
                        "round": exam_round + 1,
                        "strict_diagnosis_driven": bool(
                            strategy.get("strict_diagnosis_driven")
                        ),
                        "differential_driven": bool(strategy.get("differential_driven")),
                        "primary_diagnosis": strategy.get("primary_diagnosis", ""),
                        "differential_candidates": list(
                            strategy.get("differential_candidates") or []
                        ),
                        "discriminating_items": list(
                            strategy.get("discriminating_items") or []
                        ),
                        "authorized_items": list(strategy.get("items") or []),
                        "reserved_gap_items": list(strategy.get("reserved_gap_items") or []),
                        "source_decision_version": strategy.get("source_decision_version", 0),
                        "source_evidence_version": strategy.get("source_evidence_version", 0),
                        "blocked_items": list(strategy.get("blocked_items") or []),
                        "exam_authorization_details": list(
                            strategy.get("exam_authorization_details") or []
                        ),
                        "generic_exam_suppression_count": int(
                            strategy.get("generic_exam_suppression_count", 0) or 0
                        ),
                    }
                )
            exam_items = self._strategy_order_items(
                strategy,
                collected_info=collected_info,
                candidate_diseases=_cands2 if isinstance(_cands2, list) else None,
                existing_results=exam_results,
                max_items=self.exam_agent.max_new_items,
                add_strong_verification=False,
            )

            if not exam_items:
                logger.info(f"[检查] 无更多检查需要申请")
                break

            # 去重：排除已检查的项目
            new_items = [item for item in exam_items if item not in exam_results]
            if not new_items:
                logger.info(f"[检查] 所有推荐检查已完成")
                break

            logger.info(f"[检查] 第{exam_round + 1}轮检查: {new_items}")

            # 申请检查
            response = await self.actions.order_examination(
                patient_id=patient_id,
                items=new_items,
                reason=f"基于问诊信息，需要进一步检查以明确诊断。",
            )

            # 合并检查结果
            if response and "results" in response:
                for exam_name, exam_data in response["results"].items():
                    if exam_data.get("status") != "invalid":
                        exam_results[exam_name] = exam_data
                self._record_targeted_exam_result_recovery(
                    patient_id=patient_id,
                    stage="initial_exam",
                    ordered_items=list(new_items),
                    new_results={
                        exam_name: exam_data
                        for exam_name, exam_data in (response or {}).get("results", {}).items()
                        if isinstance(exam_data, dict) and exam_data.get("status") != "invalid"
                    },
                    strategy=strategy,
                )

        return exam_results

    # ============ 提交诊疗方案 ============

    def _raw_case_text_from_state(
        self,
        collected_info: Dict[str, Any],
        chat_history: List[Dict[str, str]],
    ) -> str:
        parts: List[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str):
                text = " ".join(value.split())
                if text and text not in parts:
                    parts.append(text)
                return
            if isinstance(value, list):
                for item in value:
                    add(item)
                return
            if isinstance(value, dict):
                for key in (
                    "raw_case_text",
                    "raw_text",
                    "case_text",
                    "patient_text",
                    "original_case",
                    "chief_complaint",
                    "history",
                    "history_present_illness",
                    "physical_exam",
                ):
                    if key in value:
                        add(value.get(key))

        add(collected_info or {})
        for message in chat_history or []:
            if not isinstance(message, dict):
                continue
            sender = str(message.get("from") or message.get("role") or "").lower()
            if sender in {"patient", "user", "患者", "病人"}:
                add(message.get("content") or message.get("text") or "")
        return "\n".join(parts)

    async def _prescribe(
        self,
        patient_id: str,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        relevant_experience: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """综合分析并提交诊疗方案。

        Args:
            patient_id: 患者 ID
            collected_info: 收集到的患者信息
            exam_results: 检查结果
            chat_history: 对话历史
            relevant_experience: 相关历史经验

        Returns:
            最终诊疗结果
        """
        conversation_rounds = len([m for m in chat_history if m.get("from") == "doctor"])
        relevant_experience = self._get_cached_experience(collected_info)
        decision = None
        evidence = None
        raw_case_text = self._raw_case_text_from_state(collected_info, chat_history)

        if self.diagnosis_chain_enabled:
            evidence = self._normalize_with_exam_recovery(
                collected_info,
                exam_results,
                raw_case_text=raw_case_text,
            )
            evidence_graph = evidence.to_graph()
            planner_candidates = self._planner_candidate_names()
            retrieval_views = self.diagnosis_engine.build_retrieval_views(evidence)
            rag_query = evidence.to_query()
            if planner_candidates:
                rag_query += " 当前鉴别诊断 " + " ".join(planner_candidates)
            rag_chunks = self.memory_manager.search_rag(
                collected_info=collected_info,
                query=rag_query,
                candidate_diseases=planner_candidates or None,
                retrieval_views=retrieval_views,
            )
            rag_context = self.memory_manager.render_rag_chunks(rag_chunks)
            preview = self.diagnosis_engine.decide({}, rag_chunks, evidence)
            candidate_table = self.diagnosis_engine.render_candidate_table(preview)
            diagnosis_context = self._compile_llm_context(
                "diagnosis",
                collected_info=collected_info,
                exam_results=exam_results,
                chat_history=chat_history,
                relevant_experience=relevant_experience,
                standard_diseases=self.diagnosis_engine.knowledge.allowed_names,
                rag_context=rag_context,
                evidence_summary=evidence.render_summary(),
                candidate_table=candidate_table,
            )
            diagnosis_prompt = self.prompt.build_diagnosis_prompt(
                collected_info=diagnosis_context.get("collected_info", collected_info),
                exam_results=diagnosis_context.get("exam_results", exam_results),
                chat_history=diagnosis_context.get("chat_history", chat_history),
                relevant_experience=diagnosis_context.get(
                    "relevant_experience", relevant_experience
                ),
                standard_diseases=diagnosis_context.get(
                    "standard_diseases", self.diagnosis_engine.knowledge.allowed_names
                ),
                rag_context=diagnosis_context.get("rag_context", rag_context),
                evidence_summary=diagnosis_context.get(
                    "evidence_summary", evidence.render_summary()
                ),
                candidate_table=diagnosis_context.get("candidate_table", candidate_table),
            )
            messages = [
                {"role": "system", "content": diagnosis_prompt},
                {"role": "user", "content": "请做出诊断并制定治疗方案，以 JSON 格式输出。"},
            ]
            diagnosis_result = await self._llm_generate_diagnosis(messages)
            evidence = self._compile_evidence_with_exam_recovery(
                collected_info,
                exam_results,
                diagnosis_result,
                raw_case_text=raw_case_text,
            )
            pattern_recall_context = self.diagnosis_engine.build_pattern_recall_context(
                diagnosis_result,
                evidence,
                case_id=patient_id,
                thinking_snapshots=list(self._thinking_snapshots),
            )
            evidence_graph = evidence.to_graph()
            retrieval_views = self.diagnosis_engine.build_retrieval_views(evidence)
            llm_resolutions = self.diagnosis_engine.resolve_open_candidates(diagnosis_result)
            llm_candidates = []
            for item in llm_resolutions:
                if item.raw_name:
                    llm_candidates.append(item.raw_name)
                if item.canonical_name:
                    llm_candidates.append(item.canonical_name)
            llm_candidates = list(dict.fromkeys(llm_candidates))
            final_candidates = list(
                dict.fromkeys(planner_candidates + llm_candidates)
            )
            final_query = evidence.to_query()
            if final_candidates:
                final_query += " 当前鉴别诊断 " + " ".join(final_candidates)
            final_rag_chunks = self.memory_manager.search_rag(
                collected_info=collected_info,
                query=final_query,
                candidate_diseases=final_candidates or None,
                retrieval_views=retrieval_views,
            )
            if final_rag_chunks:
                rag_chunks = final_rag_chunks
            self._mark_clinical_transition(
                "diagnosis_decision_started",
                "diagnosis",
                candidate_count=len(final_candidates),
                rag_chunk_count=len(rag_chunks or []),
                evidence_finding_count=len(evidence.observations or []),
            )
            decision = self.diagnosis_engine.decide(
                self._diagnosis_input_with_runtime_state(diagnosis_result),
                rag_chunks,
                evidence,
                pattern_recall_context=pattern_recall_context,
            )
            self._last_diagnosis_decision_obj = decision
            admitted_views = self._materialize_admitted_candidate_claim_states()
            self._hydrate_claim_states_from_existing_exam_observations(
                admitted_views,
                stage="diagnosis_decision_claim_hydration",
            )
            self._mark_clinical_transition(
                "diagnosis_decision_completed",
                "diagnosis",
                candidate_count=len(getattr(decision, "candidates", []) or []),
                final_diagnosis_count=len(
                    getattr(decision, "final_diagnoses", []) or []
                ),
                judge_primary=str(getattr(decision, "judge_primary", "") or ""),
            )
            critic = await self.diagnosis_critic.review(
                decision,
                evidence,
                remaining_seconds=self._remaining_case_seconds(),
                allow_llm=True,
            )
            self._apply_critic_selection(decision, critic.selected_diagnoses, critic.reason)
            self._restore_legacy_candidate_submission(decision, diagnosis_result, critic)
            self.diagnosis_engine.judge_and_submit(decision)
            pre_corrective_judge = dict(getattr(decision, "judge_decision", None) or {})
            pre_corrective_primary = str(getattr(decision, "judge_primary", "") or "")

            evidence_gap_exams = self._recommend_evidence_gap_exams(
                decision=decision,
                collected_info=collected_info,
                exam_results=exam_results,
            )
            if pre_corrective_judge.get("needs_discriminating_exams"):
                recommended_exams = list(dict.fromkeys(evidence_gap_exams))
            else:
                recommended_exams = list(
                    dict.fromkeys(
                        evidence_gap_exams + list(critic.recommended_exams or [])
                    )
                )
            corrective_targets = self._evidence_gap_target_diagnoses(decision) or list(
                decision.final_diagnoses or []
            )
            corrective_candidate_diseases = corrective_targets
            if pre_corrective_judge.get("needs_discriminating_exams"):
                differential_candidates = [
                    str(item).strip()
                    for item in pre_corrective_judge.get("differential_candidates") or []
                    if str(item).strip()
                ]
                corrective_candidate_diseases = (
                    list(dict.fromkeys(corrective_targets + differential_candidates))
                    or corrective_targets
                )
            corrective_results = await self._maybe_order_critic_exams(
                patient_id=patient_id,
                recommended_exams=recommended_exams,
                exam_results=exam_results,
                collected_info=collected_info,
                candidate_diseases=corrective_candidate_diseases,
                judge_decision=pre_corrective_judge,
                add_strong_verification=not bool(
                    pre_corrective_judge.get("needs_discriminating_exams")
                ),
                force_deferred_anchor_round=self._has_deferred_anchor_target(
                    decision,
                    corrective_targets,
                ),
            )
            if corrective_results:
                exam_results.update(corrective_results)
                self.memory_manager.update_exam_results(patient_id, corrective_results)
                self._last_exam_results = dict(exam_results)
                evidence = self._compile_evidence_with_exam_recovery(
                    collected_info,
                    exam_results,
                    diagnosis_result,
                    raw_case_text=raw_case_text,
                )
                pattern_recall_context = self.diagnosis_engine.build_pattern_recall_context(
                    diagnosis_result,
                    evidence,
                    case_id=patient_id,
                    thinking_snapshots=list(self._thinking_snapshots),
                )
                evidence_graph = evidence.to_graph()
                rag_chunks = self.memory_manager.search_rag(
                    collected_info=collected_info,
                    query=(
                        evidence.to_query()
                        + " 当前鉴别诊断 "
                        + " ".join(decision.final_diagnoses)
                    ),
                    candidate_diseases=decision.final_diagnoses or None,
                )
                self._mark_clinical_transition(
                    "post_result_reevaluation_started",
                    "diagnosis",
                    corrective_exam_count=len(corrective_results),
                    ordered_exams=list(corrective_results.keys()),
                    evidence_finding_count=len(evidence.observations or []),
                )
                decision = self.diagnosis_engine.decide(
                    self._diagnosis_input_with_runtime_state(diagnosis_result),
                    rag_chunks,
                    evidence,
                    pattern_recall_context=pattern_recall_context,
                )
                self._last_diagnosis_decision_obj = decision
                admitted_views = self._materialize_admitted_candidate_claim_states()
                self._hydrate_claim_states_from_existing_exam_observations(
                    admitted_views,
                    stage="post_result_decision_claim_hydration",
                )
                self._mark_clinical_transition(
                    "post_result_reevaluation_completed",
                    "diagnosis",
                    candidate_count=len(getattr(decision, "candidates", []) or []),
                    final_diagnosis_count=len(
                        getattr(decision, "final_diagnoses", []) or []
                    ),
                    judge_primary=str(getattr(decision, "judge_primary", "") or ""),
                )
                final_critic = await self.diagnosis_critic.review(
                    decision,
                    evidence,
                    remaining_seconds=self._remaining_case_seconds(),
                    allow_llm=False,
                )
                self._apply_critic_selection(
                    decision,
                    final_critic.selected_diagnoses,
                    final_critic.reason,
                )
                self._restore_legacy_candidate_submission(decision, diagnosis_result, final_critic)
                self.diagnosis_engine.judge_and_submit(decision)
                post_corrective_judge = dict(getattr(decision, "judge_decision", None) or {})
                post_corrective_primary = str(getattr(decision, "judge_primary", "") or "")
                fallback_applied = self._apply_pre_discrimination_fallback(
                    decision,
                    pre_corrective_judge,
                    post_corrective_judge,
                )
                if fallback_applied:
                    post_corrective_judge = dict(
                        getattr(decision, "judge_decision", None) or post_corrective_judge
                    )
                    post_corrective_primary = str(
                        getattr(decision, "judge_primary", "") or post_corrective_primary
                    )
                rerank_trace = list(
                    pre_corrective_judge.get("dynamic_rerank_trace") or []
                )
                rerank_trace.extend(post_corrective_judge.get("dynamic_rerank_trace") or [])
                rerank_trace.append(
                    {
                        "stage": "after_discriminating_exams",
                        "ordered_exams": list(corrective_results.keys()),
                        "previous_primary": pre_corrective_primary,
                        "primary": post_corrective_primary,
                        "changed_primary": bool(
                            pre_corrective_primary
                            and post_corrective_primary
                            and pre_corrective_primary != post_corrective_primary
                        ),
                        "fallback_to_pre_discrimination_primary": bool(
                            fallback_applied
                        ),
                    }
                )
                post_corrective_judge["dynamic_rerank_trace"] = rerank_trace
                post_corrective_judge["dynamic_rerank_changed_primary"] = any(
                    bool(item.get("changed_primary"))
                    for item in rerank_trace
                    if isinstance(item, dict)
                )
                decision.judge_decision = post_corrective_judge
                critic.issues = list(dict.fromkeys(critic.issues + final_critic.issues))

            diagnosis_result = self.diagnosis_engine.apply_to_result(
                diagnosis_result,
                decision,
                evidence,
            )
            diagnosis_result["_critic_review"] = critic.to_dict()
            diagnosis_result["_rag_chunks"] = [
                {
                    "id": item.get("id"),
                    "type": item.get("type"),
                    "title": item.get("title"),
                    "score": item.get("score"),
                }
                for item in rag_chunks
            ]
            self._last_diagnosis_audit = {
                "evidence": evidence.to_dict(),
                "evidence_graph": evidence_graph.to_dict(),
                "diagnosis_decision": decision.to_dict(),
                "critic": critic.to_dict(),
                "rag_chunks": diagnosis_result["_rag_chunks"],
                "llm_candidates": llm_candidates,
                "diagnosis_name_resolution": [
                    item.to_dict() for item in llm_resolutions
                ],
                "evidence_compiler": dict(self.evidence_compiler.last_audit),
                "exam_result_intent_bindings": list(self._exam_result_intent_bindings),
                "targeted_exam_result_parses": list(self._targeted_exam_result_parses),
                "clinical_admission_audit": list(self._clinical_admission_audit),
                "candidate_claim_contract_views": list(
                    self._candidate_claim_contract_views
                ),
                "claim_state_materialization_audit": list(
                    self._claim_state_materialization_audit
                ),
                "claim_state_invariant_audit": list(self._claim_state_invariant_audit),
                "claim_resolution_ledger": normalize_ledger(self._claim_resolution_ledger),
                "claim_match_events": list(self._claim_match_events),
                "claim_resolution_update_audit": list(self._claim_resolution_update_audit),
                "claim_state_version": int(self._claim_state_version or 0),
                "diagnostic_state_version": int(self._diagnostic_state_version or 0),
                "llm_call_audit": list(self._llm_call_audit),
                "llm_contract_summary": self._llm_contract_summary_from_audit(
                    list(self._llm_call_audit)
                ),
                "llm_context_audit": list(self._llm_context_audit),
                "tool_call_audit": self.actions.snapshot_tool_audit()
                if hasattr(self.actions, "snapshot_tool_audit")
                else [],
                "tool_contract_summary": self.actions.tool_contract_summary()
                if hasattr(self.actions, "tool_contract_summary")
                else {},
            }
        else:
            diagnosis_context = self._compile_llm_context(
                "diagnosis",
                collected_info=collected_info,
                exam_results=exam_results,
                chat_history=chat_history,
                relevant_experience=relevant_experience,
                standard_diseases=self.knowledge.get_disease_catalog_names(),
            )
            diagnosis_prompt = self.prompt.build_diagnosis_prompt(
                collected_info=diagnosis_context.get("collected_info", collected_info),
                exam_results=diagnosis_context.get("exam_results", exam_results),
                chat_history=diagnosis_context.get("chat_history", chat_history),
                relevant_experience=diagnosis_context.get(
                    "relevant_experience", relevant_experience
                ),
                standard_diseases=diagnosis_context.get(
                    "standard_diseases", self.knowledge.get_disease_catalog_names()
                ),
            )
            messages = [
                {"role": "system", "content": diagnosis_prompt},
                {"role": "user", "content": "请做出诊断并制定治疗方案，以 JSON 格式输出。"},
            ]
            diagnosis_result = await self._llm_generate_diagnosis(messages)
            diagnosis_result = self.evidence_engine.review(
                diagnosis_result,
                collected_info=collected_info,
                exam_results=exam_results,
            )
            if not diagnosis_result.get("_trusted_diagnoses"):
                diagnosis_result = self.structural_agent.review(
                    diagnosis_result,
                    collected_info=collected_info,
                    exam_results=exam_results,
                )

        diagnosis_result = self.quality_agent.review_final_result(
            diagnosis_result,
            collected_info=collected_info,
            exam_results=exam_results,
            conversation_rounds=conversation_rounds,
        )
        diagnosis_result = self.treatment_agent.review(
            diagnosis_result,
            collected_info=collected_info,
            exam_results=exam_results,
        )
        diagnosis_result = self.treatment_safety.review(
            diagnosis_result,
            collected_info=collected_info,
            exam_results=exam_results,
        )
        diagnosis_result = self.quality_agent.review_final_result(
            diagnosis_result,
            collected_info=collected_info,
            exam_results=exam_results,
            conversation_rounds=conversation_rounds,
        )
        if (
            decision is not None
            and evidence is not None
        ):
            diagnosis_result = self._refilter_diagnosis_result(
                diagnosis_result,
                decision,
                evidence,
            )
        if decision is not None and decision.final_diagnoses:
            filtered_names = list(decision.final_diagnoses)
        else:
            filtered_names = self._remove_suppressed_diagnosis_names(
                self._diagnosis_names_from_result(diagnosis_result)
            )
        if filtered_names:
            diagnosis_result["diagnosis"] = filtered_names
        if diagnosis_result.get("_qc_issues"):
            logger.info(f"[质控] 诊疗方案修复/提示: {diagnosis_result['_qc_issues']}")

        logger.info(
            f"[诊断] 诊断结果: {json.dumps(diagnosis_result, ensure_ascii=False, indent=2)}"
        )

        if decision is not None and decision.final_diagnoses:
            submission_diagnoses = list(decision.final_diagnoses)
        else:
            submission_diagnoses = self._remove_suppressed_diagnosis_names(
                self._diagnosis_names_from_result(diagnosis_result)
            )
        if submission_diagnoses:
            diagnosis_result["diagnosis"] = submission_diagnoses

        # 提交诊疗方案
        submit_result = await self.actions.prescribe_treatment(
            patient_id=patient_id,
            diagnosis=submission_diagnoses or diagnosis_result.get("diagnosis", []),
            treatment_plan=diagnosis_result.get("treatment_plan", ""),
            reasoning=diagnosis_result.get("reasoning", ""),
        )

        final_result = dict(diagnosis_result)
        if isinstance(submit_result, dict):
            for key, value in submit_result.items():
                if value is not None and value != "":
                    final_result[key] = value
        reviewed = self.quality_agent.review_final_result(
            final_result,
            collected_info=collected_info,
            exam_results=exam_results,
            conversation_rounds=conversation_rounds,
        )
        if submission_diagnoses:
            reviewed["diagnosis"] = submission_diagnoses
        elapsed = round(max(0.0, time.monotonic() - self._case_started_at), 3)
        reviewed["_case_elapsed_seconds"] = elapsed
        reviewed["_case_timed_out"] = False
        if self._last_diagnosis_audit:
            if decision is not None:
                self._last_diagnosis_audit["diagnosis_decision"] = decision.to_dict()
            self._last_diagnosis_audit["elapsed_seconds"] = elapsed
            self._last_diagnosis_audit["timed_out"] = False
        return reviewed

    def _planner_candidate_names(self) -> List[str]:
        plan = getattr(self._planner, "current_plan", None) or {}
        names: List[str] = []
        primary = plan.get("primary_hypothesis") or plan.get("primary_diagnosis")
        if primary:
            names.append(str(primary))
        for item in plan.get("differential_diagnoses", []) or []:
            if isinstance(item, str):
                value = item
            elif isinstance(item, dict):
                value = item.get("disease") or item.get("diagnosis") or item.get("name")
            else:
                value = None
            if value:
                names.append(str(value))
        return list(dict.fromkeys(names))[:8]

    @staticmethod
    def _diagnosis_names_from_result(result: Any) -> List[str]:
        if not isinstance(result, dict):
            return []
        values = result.get("diagnosis") or result.get("diagnoses") or []
        if isinstance(values, str):
            values = [values]
        return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))

    def _diagnosis_input_with_runtime_state(self, result: Any) -> Dict[str, Any]:
        payload = dict(result or {}) if isinstance(result, dict) else {}
        payload["_claim_resolution_ledger"] = normalize_ledger(
            self._claim_resolution_ledger
        )
        payload["_claim_match_events"] = list(self._claim_match_events)
        payload["_claim_state_version"] = int(self._claim_state_version or 0)
        payload["_diagnostic_state_version"] = int(self._diagnostic_state_version or 0)
        return payload

    def _diagnosis_input_with_claim_state(self, result: Any) -> Dict[str, Any]:
        return self._diagnosis_input_with_runtime_state(result)

    def _augment_evidence_from_reasoning(
        self,
        evidence: EvidenceBundle,
        diagnosis_result: Dict[str, Any],
    ) -> EvidenceBundle:
        if not isinstance(evidence, EvidenceBundle) or not isinstance(diagnosis_result, dict):
            return evidence
        additions = self.evidence_compiler.reasoning_adapter.adapt(diagnosis_result)
        if not additions:
            return evidence

        observations = self.evidence_compiler.merge_observations(
            evidence.observations,
            additions,
        )
        observations = self.clinical_normalizer._finalize_observations(observations)
        added_findings = [item.finding for item in additions]
        if added_findings:
            logger.info(
                "[诊断证据] reasoning 推论证据: %s",
                list(dict.fromkeys(added_findings)),
            )
        return EvidenceBundle(observations)

    def _reasoning_evidence_texts(self, result: Dict[str, Any]) -> List[str]:
        texts: List[str] = []

        def add_text(value: Any) -> None:
            if isinstance(value, str):
                text = value.strip()
                if text:
                    texts.append(text)
                return
            if isinstance(value, list):
                for item in value:
                    add_text(item)
                return
            if isinstance(value, dict):
                for key in (
                    "supporting_evidence",
                    "evidence",
                    "evidence_summary",
                    "reasoning",
                    "reason",
                    "rationale",
                ):
                    if key in value:
                        add_text(value.get(key))

        add_text(result.get("reasoning"))
        for key in (
            "diagnosis_candidates",
            "candidate_diagnoses",
            "open_diagnosis_candidates",
        ):
            add_text(result.get(key))
        return list(dict.fromkeys(texts))

    def _reasoning_inference_observations(
        self,
        text: str,
        field_path: str,
    ) -> List[Observation]:
        raw_text = " ".join(str(text or "").split())
        if not raw_text:
            return []
        findings: List[tuple] = []

        def add(finding: str, confidence: float, direction: str = "") -> None:
            if finding not in [item[0] for item in findings]:
                findings.append((finding, confidence, direction))

        if (
            self._reasoning_has_assertive_term(
                raw_text,
                ("镁负荷保留率升高", "镁保留率升高"),
            )
            or self._reasoning_regex_assertive(
                raw_text,
                r"镁负荷.{0,16}(?:保留率|保留).{0,16}(?:升高|增高|偏高|高于|>|＞|\d+(?:\.\d+)?%)",
            )
        ):
            add("magnesium_load_retention_high", 0.9, "high")
            add("magnesium_depletion", 0.88)
        if (
            self._reasoning_has_assertive_term(
                raw_text,
                ("24小时尿镁降低", "尿镁降低", "尿镁偏低"),
            )
            or self._reasoning_regex_assertive(
                raw_text,
                r"(?:24小时)?尿镁.{0,16}(?:降低|减低|偏低|低于|<|＜)",
            )
        ):
            add("low_urine_magnesium", 0.86, "low")
            add("magnesium_depletion", 0.86)
        if self._reasoning_has_assertive_term(
            raw_text,
            ("镁储备不足", "镁储备缺乏", "镁缺乏"),
        ):
            add("magnesium_depletion", 0.88)
        if self._reasoning_has_assertive_term(
            raw_text,
            ("血镁降低", "血镁偏低", "低血镁"),
        ):
            add("low_magnesium", 0.9, "low")

        if self._reasoning_has_assertive_term(raw_text, ("肺肾综合征",)):
            add("pulmonary_hemorrhage", 0.82)
            add("renal_impairment", 0.78)
        if self._reasoning_has_assertive_term(
            raw_text,
            ("肺泡出血", "弥漫性肺泡出血", "肺出血"),
        ):
            add("pulmonary_hemorrhage", 0.88)
        if self._reasoning_has_assertive_term(
            raw_text,
            ("镜下血尿", "显微镜下血尿", "血尿", "尿红细胞增多"),
        ):
            add("microscopic_hematuria", 0.84)
        if (
            self._reasoning_has_assertive_term(raw_text, ("尿色深", "尿色变深"))
            and self._reasoning_has_assertive_term(raw_text, ("血尿", "肾小球肾炎"))
        ):
            add("microscopic_hematuria", 0.8)
        if self._reasoning_has_assertive_term(raw_text, ("蛋白尿", "尿蛋白阳性")):
            add("proteinuria", 0.82)
        if self._reasoning_has_assertive_term(
            raw_text,
            ("肾功能受损", "肾功能损害", "肾损害", "肌酐升高", "肾小球滤过率降低"),
        ):
            add("renal_impairment", 0.84)
        if self._reasoning_has_assertive_term(
            raw_text,
            ("MPO-ANCA阳性", "MPO-ANCA 阳性", "MPO抗体阳性", "抗MPO阳性"),
        ):
            add("mpo_anca_positive", 0.9)
            add("anca_positive", 0.84)
        if self._reasoning_has_assertive_term(
            raw_text,
            ("p-ANCA阳性", "P-ANCA阳性", "p-ANCA 阳性"),
        ):
            add("p_anca_positive", 0.86)
            add("anca_positive", 0.82)
        if self._reasoning_has_assertive_term(
            raw_text,
            ("ANCA阳性", "ANCA 阳性", "ANCA谱阳性"),
        ):
            add("anca_positive", 0.82)

        heart_failure_terms = (
            "BNP升高",
            "BNP增高",
            "NT-proBNP升高",
            "NT-proBNP增高",
            "EF下降",
            "射血分数降低",
            "肺淤血",
            "肺水肿",
            "心影增大",
            "心脏扩大",
            "容量超负荷",
        )
        if self._reasoning_has_assertive_term(raw_text, heart_failure_terms):
            add("heart_failure_state", 0.86)
        if (
            self._reasoning_has_assertive_term(raw_text, ("心力衰竭", "心衰"))
            and self._reasoning_has_assertive_term(raw_text, ("端坐呼吸",))
            and self._reasoning_has_assertive_term(raw_text, ("水肿",))
        ):
            add("heart_failure_state", 0.84)

        if self._reasoning_has_assertive_term(
            raw_text,
            ("肺动脉瓣狭窄", "肺动脉瓣口狭窄"),
        ):
            add("pulmonary_valve_stenosis", 0.92)
            add("diagnosis:肺动脉瓣狭窄", 0.9)
        if self._reasoning_regex_assertive(
            raw_text,
            r"(?:肺动脉瓣|跨瓣|峰值).{0,12}(?:压差|压力阶差).{0,12}(?:升高|增高|[5-9]\d\s*mmHg|\d{2,3}\s*mmHg)",
        ):
            add("pulmonary_valve_gradient", 0.9, "high")
        if self._reasoning_has_assertive_term(raw_text, ("右心室肥厚", "右室肥厚")):
            add("right_ventricular_hypertrophy", 0.84)
        if self._reasoning_has_assertive_term(raw_text, ("肺动脉高压", "肺动脉压升高")):
            add("pulmonary_hypertension", 0.84)
        if self._reasoning_has_assertive_term(
            raw_text,
            ("室间隔缺损", "大型VSD", "大型室间隔缺损", "VSD"),
        ):
            add("ventricular_septal_defect", 0.9)
            add("diagnosis:室间隔缺损（VSD）", 0.86)
        if self._reasoning_has_assertive_term(raw_text, ("右向左分流", "右至左分流")):
            add("right_to_left_shunt", 0.88)
        if self._reasoning_has_assertive_term(
            raw_text,
            ("先天性心脏病", "先心病", "先天性心脏缺陷", "紫绀型先天性心脏病"),
        ):
            add("congenital_heart_defect", 0.9)
            add("diagnosis:先天性心脏病", 0.88)

        return [
            Observation(
                finding=finding,
                source="reasoning_inference",
                direction=direction,
                polarity="positive",
                confidence=confidence,
                raw_text=raw_text[:240],
                field_path=field_path,
            )
            for finding, confidence, direction in findings
        ]

    def _reasoning_has_assertive_term(self, text: str, terms: tuple) -> bool:
        for term in terms:
            search_from = 0
            while True:
                start = str(text).find(term, search_from)
                if start < 0:
                    break
                if not self._reasoning_window_blocked(text, start, start + len(term)):
                    return True
                search_from = start + len(term)
        return False

    def _reasoning_regex_assertive(self, text: str, pattern: str) -> bool:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if not self._reasoning_window_blocked(text, match.start(), match.end()):
                return True
        return False

    @staticmethod
    def _reasoning_window_blocked(text: str, start: int, end: int) -> bool:
        window = text[max(0, start - 24): min(len(text), end + 36)]
        blockers = (
            "不支持", "排除", "不能解释", "缺乏", "无", "未见", "未发现",
            "阴性", "正常", "鉴别", "待鉴别", "需鉴别", "待排", "待查",
            "需查", "建议", "排查", "除外", "可能", "疑似",
        )
        return any(token in window for token in blockers)

    def _refilter_diagnosis_result(self, result: Dict[str, Any], decision, evidence) -> Dict[str, Any]:
        if getattr(decision, "judge_decision", None):
            self.diagnosis_engine.judge_and_submit(decision)
            return self.diagnosis_engine.apply_to_result(result, decision, evidence)
        original_names = self._diagnosis_names_from_result(result)
        names = self._remove_suppressed_diagnosis_names(original_names)
        if not names:
            return self.diagnosis_engine.apply_to_result(result, decision, evidence)
        self.diagnosis_engine.authorize_final_diagnoses(
            decision,
            names,
            respect_differential_only=True,
        )
        if not decision.final_diagnoses:
            return self.diagnosis_engine.apply_to_result(result, decision, evidence)
        filtered_names = list(decision.final_diagnoses)
        if filtered_names == original_names:
            return self.diagnosis_engine.apply_to_result(result, decision, evidence)
        score_by_name = {item.diagnosis: item for item in decision.candidates}
        if filtered_names and filtered_names[0] in score_by_name:
            decision.confidence = score_by_name[filtered_names[0]].score
        return self.diagnosis_engine.apply_to_result(result, decision, evidence)

    def _remove_suppressed_diagnosis_names(self, names: List[str]) -> List[str]:
        ordered = list(dict.fromkeys(names or []))
        selected = set(ordered)
        suppressed = set()
        for name in ordered:
            entry = self.diagnosis_engine.knowledge.get(name)
            suppressed.update(str(item) for item in entry.get("suppress_diagnoses", []) or [])
            parent = str(entry.get("parent_diagnosis") or "")
            if parent and parent in selected:
                suppressed.add(parent)
        return [name for name in ordered if name not in suppressed]

    def _apply_critic_selection(
        self,
        decision,
        selected_diagnoses: List[str],
        reason: str,
    ) -> None:
        if not selected_diagnoses:
            return
        score_by_name = {item.diagnosis: item for item in decision.candidates}
        base_final = list(decision.final_diagnoses or [])
        for name in selected_diagnoses:
            candidate = score_by_name.get(name)
            if not candidate:
                continue
            if (
                not candidate.hard_contradiction
                and candidate.trusted
                and not candidate.differential_only
                and not candidate.differential_only_reason
                and not self._critic_submission_eligible(candidate, base_final)
            ):
                self.diagnosis_engine._mark_differential_only(
                    candidate,
                    "作为 critic 提出的鉴别诊断保留，但缺少直接诊断证据、独立并发证据或足够高的证据评分，不作为最终诊断提交。",
                )
        self.diagnosis_engine.authorize_final_diagnoses(
            decision,
            base_final,
            respect_differential_only=True,
        )
        if reason:
            extra = self.diagnosis_engine.differential_only_reasoning(decision.candidates)
            suffix = " Final diagnosis authorization reviewed critic suggestions: " + str(reason).strip()
            if extra:
                suffix += extra
            if suffix not in decision.evidence_reasoning:
                decision.evidence_reasoning = decision.evidence_reasoning + suffix
        return

    def _critic_submission_eligible(self, candidate, base_final: List[str]) -> bool:
        if not candidate:
            return False
        if candidate.diagnosis in set(base_final or []):
            return True
        if f"diagnosis:{candidate.diagnosis}" in set(candidate.matched_evidence or []):
            return True
        if (
            self.diagnosis_engine._is_secondary_manifestation(candidate)
            and self.diagnosis_engine._has_independent_state_evidence(candidate)
            and any(
                self.diagnosis_engine._diagnoses_submission_related(candidate.diagnosis, name)
                for name in base_final or []
            )
        ):
            return True
        return candidate.score >= self.diagnosis_engine.trusted_threshold

    def _restore_legacy_candidate_submission(
        self,
        decision,
        llm_result: Dict[str, Any],
        critic,
    ) -> None:
        """Restore the earlier broad-submission behavior for evidence-backed candidates.

        This keeps the Evidence-first ranking, but lets an LLM primary diagnosis or
        Critic-supported candidate re-enter the final submission when it is already
        represented in the scored candidate table and has no hard contradiction.
        """
        if not self.legacy_candidate_submission or not decision:
            return
        score_by_name = {item.diagnosis: item for item in decision.candidates}
        primary_names = self._resolved_llm_primary_names(llm_result)
        supporting_names = self._resolved_llm_candidate_names(llm_result)
        supporting_names.extend(self._resolved_critic_names(critic))

        if decision.final_diagnoses:
            base_final = list(decision.final_diagnoses or [])
            for name in list(dict.fromkeys(primary_names + supporting_names)):
                if name in base_final:
                    continue
                candidate = score_by_name.get(name)
                if candidate and self._legacy_submission_eligible(candidate):
                    self.diagnosis_engine._mark_differential_only(
                        candidate,
                        "legacy candidate retained for audit; final submission is locked to the authorized evidence-first decision",
                    )
            self.diagnosis_engine.authorize_final_diagnoses(
                decision,
                base_final,
                respect_differential_only=True,
            )
            return

        primary_eligible = [
            name for name in primary_names
            if self._legacy_submission_eligible(score_by_name.get(name))
        ]
        support_eligible = [
            name for name in supporting_names
            if self._legacy_submission_eligible(score_by_name.get(name))
        ]

        if primary_eligible and self._should_prefer_legacy_primary(primary_eligible, score_by_name):
            names = list(primary_eligible)
        else:
            names = list(decision.final_diagnoses or [])

        for name in support_eligible:
            if name not in names:
                names.append(name)
            if len(names) >= self.diagnosis_engine.max_final_diagnoses:
                break

        if not names:
            return
        names = self._remove_suppressed_diagnosis_names(names)
        filtered = self.diagnosis_engine.filter_final_diagnoses(
            [
                name for name in names
                if name in score_by_name
                and self._legacy_submission_eligible(score_by_name[name])
            ],
            decision.candidates,
            respect_differential_only=True,
        )
        if not filtered:
            return
        filtered_names = [item.diagnosis for item in filtered]
        self.diagnosis_engine.authorize_final_diagnoses(
            decision,
            filtered_names,
            respect_differential_only=True,
        )
        if filtered_names == list(decision.final_diagnoses or []):
            return
        decision.differential_only_diagnoses = self.diagnosis_engine.differential_only_details(
            decision.candidates
        )
        suffix = (
            "。提交前审查：Legacy 候选恢复已通过最终诊断 gate；"
            "仅保留 required 已满足、非仅鉴别且无硬反证的标准诊断。"
        )
        if suffix not in decision.evidence_reasoning:
            decision.evidence_reasoning = decision.evidence_reasoning.rstrip("。") + suffix

    def _resolved_llm_primary_names(self, result: Dict[str, Any]) -> List[str]:
        if not isinstance(result, dict):
            return []
        values = result.get("diagnosis") or result.get("diagnoses") or []
        if isinstance(values, str):
            values = [values]
        return self._resolve_diagnosis_values(values)

    def _resolved_llm_candidate_names(self, result: Dict[str, Any]) -> List[str]:
        if not isinstance(result, dict):
            return []
        values: List[Any] = []
        for key in (
            "diagnosis_candidates",
            "open_diagnosis_candidates",
            "candidate_diagnoses",
            "differential_diagnoses",
        ):
            current = result.get(key)
            if current:
                values.extend(current if isinstance(current, list) else [current])
        return self._resolve_diagnosis_values(values)

    def _resolved_critic_names(self, critic) -> List[str]:
        values: List[Any] = []
        values.extend(list(getattr(critic, "selected_diagnoses", []) or []))
        return self._resolve_diagnosis_values(values)

    def _resolve_diagnosis_values(self, values: List[Any]) -> List[str]:
        names: List[str] = []
        for value in values or []:
            confidence = 1.0
            if isinstance(value, dict):
                raw = value.get("name") or value.get("diagnosis") or value.get("disease")
                try:
                    confidence = float(value.get("confidence", 1.0) or 1.0)
                except (TypeError, ValueError):
                    confidence = 1.0
            else:
                raw = value
            if confidence < 0.45:
                continue
            resolved = self.diagnosis_engine.resolver.resolve(raw, model_confidence=confidence)
            if resolved.canonical_name and resolved.canonical_name not in names:
                names.append(resolved.canonical_name)
        return names

    def _extract_allowed_diagnoses_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        names: List[str] = []
        for name in sorted(self.diagnosis_engine.knowledge.allowed_names, key=len, reverse=True):
            if name not in text:
                continue
            if self._diagnosis_mention_is_negated(text, name):
                continue
            if name not in names:
                names.append(name)
        return names[: self.diagnosis_engine.max_final_diagnoses * 2]

    @staticmethod
    def _diagnosis_mention_is_negated(text: str, name: str) -> bool:
        index = text.find(name)
        if index < 0:
            return False
        window = text[max(0, index - 18): index + len(name) + 28]
        negators = (
            "不支持", "排除", "不选", "未选", "缺乏", "无", "不能解释",
            "证据极弱", "可能性低", "不符合", "否定",
        )
        return any(token in window for token in negators)

    def _legacy_submission_eligible(self, candidate) -> bool:
        if candidate is None:
            return False
        return (
            candidate.trusted
            and bool(candidate.required_met)
            and bool(candidate.matched_evidence)
            and not candidate.hard_contradiction
            and not candidate.differential_only
            and not candidate.differential_only_reason
            and candidate.score >= self.diagnosis_engine.differential_threshold
        )

    def _should_prefer_legacy_primary(
        self,
        primary_names: List[str],
        score_by_name: Dict[str, Any],
    ) -> bool:
        for name in primary_names:
            candidate = score_by_name.get(name)
            if not candidate:
                continue
            dtype = str(getattr(candidate, "diagnosis_type", "") or "").lower()
            if dtype in {"etiology", "metabolic", "structural"}:
                return True
        return False

    def _recommend_evidence_gap_exams(
        self,
        decision,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> List[str]:
        if not decision:
            return []
        judge_payload = getattr(decision, "judge_decision", None) or {}
        needs_discriminating = bool(judge_payload.get("needs_discriminating_exams"))
        explicit_gap_targets = [
            str(item).strip()
            for item in (judge_payload.get("evidence_gap_targets") or [])
            if str(item).strip()
        ]
        safe_gap_targets = self._evidence_gap_target_diagnoses(decision)
        strict_stop = self._strict_primary_exam_stop_active(decision)
        if strict_stop and not needs_discriminating:
            safe_set = {
                self.knowledge.normalize_diagnosis(str(item)) or str(item)
                for item in safe_gap_targets
                if str(item).strip()
            }
            targets = [
                item
                for item in dict.fromkeys(explicit_gap_targets)
                if (self.knowledge.normalize_diagnosis(str(item)) or str(item)) in safe_set
            ] or safe_gap_targets
        else:
            targets = list(dict.fromkeys(explicit_gap_targets)) or safe_gap_targets
        if strict_stop and not needs_discriminating and not targets:
            return []
        if not targets and not needs_discriminating:
            return []
        if targets:
            targets = self._prioritize_evidence_gap_exam_targets(decision, targets)
        differential_names = [
            str(item).strip()
            for item in judge_payload.get("differential_candidates") or []
            if str(item).strip()
        ]

        target_proposed: List[str] = []
        for name in targets:
            entry = self.diagnosis_engine.knowledge.get(name)
            for exam in entry.get("discriminating_exams", []) or []:
                text = str(exam).strip()
                if text and text not in target_proposed:
                    target_proposed.append(text)

        judge_proposed: List[str] = []
        for exam in judge_payload.get("discriminating_exams", []) or []:
            text = str(exam).strip()
            if text and text not in judge_proposed:
                judge_proposed.append(text)

        if needs_discriminating:
            proposed = list(dict.fromkeys(target_proposed + judge_proposed))
            strategy_judge_payload = dict(judge_payload)
            strategy_judge_payload["discriminating_exams"] = list(proposed)
            strategy_judge_payload["differential_candidates"] = list(
                dict.fromkeys(
                    list(judge_payload.get("differential_candidates") or [])
                    + targets
                )
            )
            if targets:
                target_set = {
                    self.knowledge.normalize_diagnosis(str(item)) or str(item)
                    for item in targets
                    if str(item).strip()
                }

                def gap_targets_candidate(item: Dict[str, Any]) -> bool:
                    candidate = str(item.get("candidate") or "").strip()
                    normalized = self.knowledge.normalize_diagnosis(candidate) or candidate
                    if normalized in target_set:
                        return True
                    return any(
                        (self.knowledge.normalize_diagnosis(str(name)) or str(name))
                        in target_set
                        for name in item.get("target_candidates", []) or []
                        if str(name).strip()
                    )

                strategy_judge_payload["active_evidence_gaps"] = [
                    item
                    for item in strategy_judge_payload.get("active_evidence_gaps", []) or []
                    if isinstance(item, dict) and gap_targets_candidate(item)
                ]
                strategy_judge_payload["deferred_gap_closure_tasks"] = [
                    item
                    for item in strategy_judge_payload.get("deferred_gap_closure_tasks", []) or []
                    if isinstance(item, dict) and gap_targets_candidate(item)
                ]
                strategy_judge_payload["exam_priority_overrides"] = [
                    item
                    for item in strategy_judge_payload.get("exam_priority_overrides", []) or []
                    if isinstance(item, dict) and gap_targets_candidate(item)
                ]
        else:
            proposed = list(dict.fromkeys(target_proposed + judge_proposed))
            # When a concrete evidence-gap target already exposes discriminating exams,
            # keep that disease-specific workup ahead of generic TopK differential tests.
            # The generic Judge exams are still available as later fill-ins.
            strategy_judge_payload = judge_payload if not target_proposed else None

        pulmonary_renal_priority: List[str] = []
        if (
            self.exam_agent._needs_pulmonary_renal_workup(
                collected_info,
                list(dict.fromkeys(targets + differential_names)),
            )
        ):
            pulmonary_renal_priority, _ = self.exam_agent.knowledge.normalize_examinations(
                [
                    "尿液分析（UA）",
                    "肾功能检查（RFTs）",
                    "抗中性粒细胞胞质抗体（ANCA）谱",
                    "MPO-ANCA",
                ]
            )
            proposed = list(dict.fromkeys(pulmonary_renal_priority + proposed))
            if strategy_judge_payload is not None:
                strategy_judge_payload = dict(strategy_judge_payload)
                strategy_judge_payload["discriminating_exams"] = list(proposed)

        strategy = self.exam_agent.recommend(
            collected_info=collected_info,
            candidate_diseases=targets,
            proposed_items=proposed,
            existing_results=exam_results,
            judge_decision=strategy_judge_payload,
        )
        if needs_discriminating:
            ordered_items = list(
                dict.fromkeys(target_proposed + list(strategy.get("items", []) or []))
            )
        else:
            ordered_items = list(
                dict.fromkeys(proposed + list(strategy.get("items", []) or []))
            )
        if strategy.get("items"):
            strategy_for_items = dict(strategy)
            strategy_for_items["items"] = ordered_items
            items = self._strategy_order_items(
                strategy_for_items,
                collected_info=collected_info,
                candidate_diseases=targets,
                existing_results=exam_results,
                max_items=self.diagnosis_critic.max_corrective_exam_items,
                add_strong_verification=False,
            )
            route_audit_strategy = strategy_for_items
        else:
            items = self.exam_agent.prepare_order_items(
                ordered_items,
                collected_info=collected_info,
                candidate_diseases=targets,
                existing_results=exam_results,
                max_items=self.diagnosis_critic.max_corrective_exam_items,
                add_strong_verification=False,
            )
            route_audit_strategy = strategy
        if needs_discriminating and target_proposed:
            items = list(
                dict.fromkeys(
                    list(target_proposed)
                    + list(items or [])
                )
            )[: self.diagnosis_critic.max_corrective_exam_items]
        if pulmonary_renal_priority:
            items = list(
                dict.fromkeys(
                    list(pulmonary_renal_priority)
                    + list(items or [])
                )
            )[: self.diagnosis_critic.max_corrective_exam_items]
        if route_audit_strategy.get("exam_repeat_authorization_audit"):
            strategy["exam_repeat_authorization_audit"] = list(
                route_audit_strategy.get("exam_repeat_authorization_audit") or []
            )
        if (
            strategy.get("strict_diagnosis_driven")
            or strategy.get("differential_driven")
            or strategy.get("blocked_items")
            or strategy.get("exam_repeat_authorization_audit")
        ):
            self._last_exam_authorization.append(
                {
                    "stage": "evidence_gap_exam",
                    "strict_diagnosis_driven": bool(
                        strategy.get("strict_diagnosis_driven")
                    ),
                    "differential_driven": bool(strategy.get("differential_driven")),
                    "primary_diagnosis": strategy.get("primary_diagnosis", ""),
                    "differential_candidates": list(
                        strategy.get("differential_candidates") or []
                    ),
                    "discriminating_items": list(
                        strategy.get("discriminating_items") or []
                    ),
                    "authorized_items": list(items or []),
                    "reserved_gap_items": list(strategy.get("reserved_gap_items") or []),
                    "source_decision_version": strategy.get("source_decision_version", 0),
                    "source_evidence_version": strategy.get("source_evidence_version", 0),
                    "blocked_items": list(strategy.get("blocked_items") or []),
                    "exam_authorization_details": list(
                        strategy.get("exam_authorization_details") or []
                    ),
                    "generic_exam_suppression_count": int(
                        strategy.get("generic_exam_suppression_count", 0) or 0
                    ),
                    "exam_repeat_authorization_audit": list(
                        strategy.get("exam_repeat_authorization_audit") or []
                    ),
                }
            )
        return items

    def _apply_pre_discrimination_fallback(
        self,
        decision,
        pre_judge: Dict[str, Any],
        post_judge: Dict[str, Any],
    ) -> bool:
        if not decision or not pre_judge or not pre_judge.get("needs_discriminating_exams"):
            return False
        pre_primary = str(
            pre_judge.get("pre_discrimination_primary")
            or pre_judge.get("provisional_primary")
            or pre_judge.get("primary")
            or pre_judge.get("judge_primary")
            or ""
        ).strip()
        if not pre_primary:
            return False
        post_primary = str(
            post_judge.get("primary") or post_judge.get("judge_primary") or ""
        ).strip()
        if post_primary == pre_primary:
            return False
        by_name = {item.diagnosis: item for item in getattr(decision, "candidates", [])}
        fallback = by_name.get(pre_primary)
        if not fallback or getattr(fallback, "hard_contradiction", False):
            return False
        if str(getattr(fallback, "eligibility_status", "") or "") != PRIMARY_ELIGIBLE:
            return False
        if not getattr(fallback, "matched_evidence", None):
            return False
        current = by_name.get(post_primary)
        try:
            fallback_score = self.diagnosis_engine.judge._primary_eligibility_score(
                fallback
            )
            current_score = (
                self.diagnosis_engine.judge._primary_eligibility_score(current)
                if current
                else -1.0
            )
        except Exception:
            fallback_score = float(getattr(fallback, "score", 0.0) or 0.0)
            current_score = float(getattr(current, "score", 0.0) or -1.0) if current else -1.0
        if fallback_score + 0.04 < current_score:
            return False

        fallback.differential_only = False
        fallback.differential_only_reason = ""
        self.diagnosis_engine.authorize_final_diagnoses(
            decision,
            [pre_primary],
            respect_differential_only=True,
        )
        if not decision.final_diagnoses or decision.final_diagnoses[0] != pre_primary:
            return False
        payload = dict(post_judge or {})
        payload.update(
            {
                "primary": pre_primary,
                "judge_primary": pre_primary,
                "primary_status": "locked",
                "locked_primary": pre_primary,
                "provisional_primary": "",
                "fallback_primary": pre_primary,
                "fallback_to_pre_discrimination_primary": True,
                "fallback_reason": (
                    "discriminating exams did not produce a clearly superior "
                    "primary; returning to pre-discrimination explanatory primary"
                ),
                "required_gap_authorized_diagnoses": [],
                "final_diagnoses": list(decision.final_diagnoses),
                "discrimination_attempted": True,
                "discrimination_resolved": False,
            }
        )
        decision.judge_primary = pre_primary
        decision.submitter_final = list(decision.final_diagnoses)
        decision.required_gap_authorized_diagnoses = []
        decision.judge_decision = payload
        return True

    def _prioritize_evidence_gap_exam_targets(
        self,
        decision,
        targets: List[str],
    ) -> List[str]:
        if not decision or not targets:
            return targets
        by_name = {item.diagnosis: item for item in getattr(decision, "candidates", [])}
        judge_payload = getattr(decision, "judge_decision", None) or {}
        review_scores = {
            str(item.get("diagnosis") or ""): float(item.get("judge_score", 0.0) or 0.0)
            for item in (judge_payload.get("reviews") or [])
            if isinstance(item, dict)
        }

        def key(name: str) -> tuple:
            candidate = by_name.get(name)
            if not candidate:
                return (0, 0, 0, 0, 0, 0.0, 0.0)
            has_gap = 1 if getattr(candidate, "required_gaps", None) else 0
            priority = 1 if self._is_etiology_priority_candidate(candidate) else 0
            matched = set(getattr(candidate, "matched_evidence", None) or [])
            objective = 1 if matched else 0
            specific_matches = len(
                [
                    item
                    for item in matched
                    if not str(item).startswith(("field:", "symptom:"))
                ]
            )
            match_count = len(matched)
            judge_score = review_scores.get(
                name,
                float(getattr(candidate, "score", 0.0) or 0.0),
            )
            evidence_specificity = float(
                getattr(candidate, "evidence_specificity_score", 0.0) or 0.0
            )
            return (
                has_gap,
                priority,
                objective,
                specific_matches,
                match_count,
                judge_score,
                evidence_specificity,
            )

        return sorted(list(dict.fromkeys(targets)), key=key, reverse=True)

    def _evidence_gap_target_diagnoses(self, decision) -> List[str]:
        judge_payload = getattr(decision, "judge_decision", None) or {}
        by_name = {item.diagnosis: item for item in decision.candidates}
        selected = by_name.get((decision.final_diagnoses or [""])[0])
        selected_names = set(decision.final_diagnoses or [])
        unexplained = set(decision.unexplained_evidence or [])
        strict_stop_active = (
            self._strict_primary_exam_stop_active(decision)
            and not bool(judge_payload.get("needs_discriminating_exams"))
        )
        targets: List[str] = []

        def add_deferred_anchor_target(name: str) -> None:
            candidate = by_name.get(str(name).strip())
            if not candidate:
                return
            if (
                str(getattr(candidate, "eligibility_status", "") or "") == DEFERRED
                and str(getattr(candidate, "eligibility_reason", "") or "") == "NeedsAnchor"
                and candidate.diagnosis not in targets
            ):
                if strict_stop_active:
                    matched = set(getattr(candidate, "matched_evidence", None) or [])
                    residual_matches = unexplained & matched
                    objective_residual_matches = [
                        item
                        for item in residual_matches
                        if not str(item).startswith(("field:", "symptom:"))
                    ]
                    specific_matches = [
                        item
                        for item in matched
                        if not str(item).startswith(("field:", "symptom:"))
                    ]
                    if not objective_residual_matches or len(specific_matches) < 2:
                        return
                targets.append(candidate.diagnosis)

        judge_targets = [
            str(item).strip()
            for item in (judge_payload.get("evidence_gap_targets") or [])
            if str(item).strip()
        ]
        for name in dict.fromkeys(judge_targets):
            add_deferred_anchor_target(name)
        deferred_anchor_names = list(
            getattr(decision, "deferred_anchor_candidates", []) or []
        ) + list(judge_payload.get("deferred_anchor_candidates") or [])
        for name in dict.fromkeys(
            str(item).strip() for item in deferred_anchor_names if str(item).strip()
        ):
            add_deferred_anchor_target(name)

        if strict_stop_active:
            if targets:
                prioritized = self._prioritize_evidence_gap_exam_targets(decision, targets)
                limit = getattr(self.diagnosis_engine, "max_evidence_gap_targets", 2)
                return prioritized[: max(1, int(limit or 2))]
            return []

        close_margin = getattr(self.diagnosis_engine, "etiology_close_margin", 0.12)
        coverage_threshold = getattr(
            self.diagnosis_engine,
            "evidence_gap_coverage_threshold",
            0.32,
        )
        residual_threshold = getattr(
            self.diagnosis_engine,
            "evidence_gap_residual_threshold",
            0.72,
        )
        gap_candidates = []
        for item in decision.candidates:
            if getattr(item, "differential_only", False) and not getattr(item, "required_gaps", None):
                continue
            if self._is_actionable_evidence_gap_candidate(
                item,
                selected=selected,
                selected_names=selected_names,
                unexplained=unexplained,
                close_margin=close_margin,
                coverage_threshold=coverage_threshold,
                residual_threshold=residual_threshold,
            ):
                gap_candidates.append(item)
        gap_candidates.sort(
            key=lambda item: (
                getattr(item, "coverage_score", 0.0),
                1.0 - getattr(item, "residual_score", 1.0),
                getattr(item, "evidence_specificity_score", 0.0),
                item.score,
            ),
            reverse=True,
        )

        has_gap = bool(
            decision.low_confidence
            or decision.unexplained_evidence
            or gap_candidates
            or self._has_close_etiology_candidate(decision)
        )
        if not has_gap:
            return []

        for item in gap_candidates:
            if item.diagnosis not in targets:
                targets.append(item.diagnosis)

        for name in (decision.final_diagnoses or [])[:1]:
            if name in by_name and name not in targets:
                targets.append(name)

        if unexplained:
            for item in decision.candidates:
                if item.hard_contradiction:
                    continue
                if getattr(item, "differential_only", False) and not getattr(item, "required_gaps", None):
                    continue
                if (
                    unexplained & set(item.matched_evidence or [])
                    and self._is_gap_candidate_competitive_with_selected(
                        item,
                        selected=selected,
                        close_margin=close_margin,
                    )
                    and (
                        not self._evidence_gap_scope_gate_active(selected)
                        or self._evidence_gap_companion_allowed(item, selected)
                    )
                ):
                    if item.diagnosis not in targets:
                        targets.append(item.diagnosis)
                    break

        limit = getattr(self.diagnosis_engine, "max_evidence_gap_targets", 2)
        return targets[: max(1, int(limit or 2))]

    def _strict_primary_exam_stop_active(self, decision) -> bool:
        if not decision or not getattr(decision, "final_diagnoses", None):
            return False
        judge_payload = getattr(decision, "judge_decision", None) or {}
        if bool(judge_payload.get("needs_discriminating_exams")):
            return False
        by_name = {item.diagnosis: item for item in getattr(decision, "candidates", [])}
        primary = by_name.get(decision.final_diagnoses[0])
        if not primary:
            return False
        if (
            not getattr(primary, "required_met", False)
            or getattr(primary, "required_gaps", None)
            or getattr(primary, "hard_contradiction", False)
            or float(getattr(primary, "score", 0.0) or 0.0)
            < float(getattr(self.diagnosis_engine, "trusted_threshold", 0.65) or 0.65)
            or not self._is_etiology_priority_candidate(primary)
        ):
            return False
        try:
            strong_items = self.exam_agent._strong_verification_items_for_disease(
                primary.diagnosis
            )
        except AttributeError:
            strong_items = []
        if not strong_items:
            return False
        matched = set(getattr(primary, "matched_evidence", None) or [])
        objective = float(
            (getattr(primary, "component_scores", None) or {}).get(
                "objective_evidence",
                0.0,
            )
            or 0.0
        )
        return objective >= 1.0 or f"diagnosis:{primary.diagnosis}" in matched

    def _is_actionable_evidence_gap_candidate(
        self,
        candidate,
        selected,
        selected_names: set,
        unexplained: set,
        close_margin: float,
        coverage_threshold: float,
        residual_threshold: float,
    ) -> bool:
        if not (
            getattr(candidate, "required_gaps", None)
            and getattr(candidate, "matched_evidence", None)
            and not getattr(candidate, "hard_contradiction", False)
            and self._is_etiology_priority_candidate(candidate)
        ):
            return False
        status = str(getattr(candidate, "eligibility_status", "") or "")
        if status and status != DEFERRED:
            return False
        if candidate.diagnosis in selected_names:
            return True
        quality_signal = (
            getattr(candidate, "coverage_score", 0.0) >= coverage_threshold
            or getattr(candidate, "residual_score", 1.0) <= residual_threshold
            or getattr(candidate, "source_prior", 0.0) >= 0.45
        )
        if not quality_signal:
            return False
        if selected is None:
            return True
        if self._evidence_gap_scope_gate_active(selected) and not self._evidence_gap_companion_allowed(candidate, selected):
            return False
        if candidate.score >= selected.score + close_margin:
            return True
        if (
            (getattr(candidate, "source_prior", 0.0) >= 0.45 or unexplained & set(candidate.matched_evidence or []))
            and self._is_gap_candidate_competitive_with_selected(
                candidate,
                selected=selected,
                close_margin=close_margin,
            )
        ):
            return True
        if unexplained & set(candidate.matched_evidence or []):
            return (
                candidate.score >= max(0.0, selected.score - 2 * close_margin)
                or getattr(candidate, "coverage_score", 0.0) + close_margin
                >= getattr(selected, "coverage_score", 0.0)
                or getattr(candidate, "residual_score", 1.0)
                <= getattr(selected, "residual_score", 1.0) + close_margin
            )
        return False

    @staticmethod
    def _is_gap_candidate_competitive_with_selected(
        candidate,
        selected,
        close_margin: float,
    ) -> bool:
        if selected is None:
            return True
        if candidate.score < max(0.0, selected.score - close_margin):
            return False
        return (
            getattr(candidate, "coverage_score", 0.0) + close_margin
            >= getattr(selected, "coverage_score", 0.0)
            or getattr(candidate, "residual_score", 1.0)
            <= getattr(selected, "residual_score", 1.0) + close_margin
        )

    def _evidence_gap_companion_allowed(self, candidate, selected) -> bool:
        if selected is None or candidate is None:
            return True
        if getattr(candidate, "diagnosis", "") == getattr(selected, "diagnosis", ""):
            return True
        try:
            left = self.diagnosis_engine.knowledge.get(candidate.diagnosis)
            right = self.diagnosis_engine.knowledge.get(selected.diagnosis)
        except Exception:
            return True
        if self._diagnosis_graph_related(candidate.diagnosis, selected.diagnosis, left, right):
            return True
        left_system = str(left.get("body_system") or "")
        right_system = str(right.get("body_system") or "")
        left_family = str(left.get("disease_family") or left.get("family") or "")
        right_family = str(right.get("disease_family") or right.get("family") or "")
        if left_system and right_system and left_system == right_system:
            return bool(left_family and right_family and left_family == right_family)
        return False

    def _evidence_gap_scope_gate_active(self, selected) -> bool:
        if selected is None:
            return False
        if not (
            getattr(selected, "required_met", False)
            and not getattr(selected, "required_gaps", None)
            and not getattr(selected, "hard_contradiction", False)
            and bool(getattr(selected, "matched_evidence", None))
        ):
            return False
        trusted_threshold = float(
            getattr(self.diagnosis_engine, "trusted_threshold", 0.65) or 0.65
        )
        if float(getattr(selected, "score", 0.0) or 0.0) < trusted_threshold:
            return False
        try:
            strong_items = self.exam_agent._strong_verification_items_for_disease(
                selected.diagnosis
            )
        except AttributeError:
            strong_items = []
        return bool(strong_items and self._is_etiology_priority_candidate(selected))

    @staticmethod
    def _diagnosis_graph_related(
        left_name: str,
        right_name: str,
        left: Dict[str, Any],
        right: Dict[str, Any],
    ) -> bool:
        if left_name == right_name:
            return True
        left_related = set(str(item) for item in left.get("related_complications", []) or [])
        right_related = set(str(item) for item in right.get("related_complications", []) or [])
        if right_name in left_related or left_name in right_related:
            return True
        left_causes = set(str(item) for item in left.get("causes", []) or [])
        right_causes = set(str(item) for item in right.get("causes", []) or [])
        left_caused_by = set(str(item) for item in left.get("caused_by", []) or [])
        right_caused_by = set(str(item) for item in right.get("caused_by", []) or [])
        return (
            right_name in left_causes
            or left_name in right_causes
            or right_name in left_caused_by
            or left_name in right_caused_by
        )

    def _has_close_etiology_candidate(self, decision) -> bool:
        if not decision.candidates or not decision.final_diagnoses:
            return False
        by_name = {item.diagnosis: item for item in decision.candidates}
        selected = by_name.get(decision.final_diagnoses[0])
        if not selected:
            return False
        close_margin = getattr(self.diagnosis_engine, "etiology_close_margin", 0.12)
        coverage_threshold = getattr(
            self.diagnosis_engine,
            "evidence_gap_coverage_threshold",
            0.32,
        )
        residual_threshold = getattr(
            self.diagnosis_engine,
            "evidence_gap_residual_threshold",
            0.72,
        )
        for item in decision.candidates:
            if item.diagnosis == selected.diagnosis:
                continue
            if (
                item.matched_evidence
                and not item.hard_contradiction
                and self._is_etiology_priority_candidate(item)
                and (
                    not self._evidence_gap_scope_gate_active(selected)
                    or self._evidence_gap_companion_allowed(item, selected)
                )
                and (
                    item.required_gaps
                    or item.source_prior >= 0.45
                    or getattr(item, "coverage_score", 0.0) >= coverage_threshold
                    or getattr(item, "residual_score", 1.0) <= residual_threshold
                    or item.score >= max(0.0, selected.score - close_margin)
                )
            ):
                return True
        return False

    @staticmethod
    def _is_etiology_priority_candidate(candidate) -> bool:
        dtype = str(getattr(candidate, "diagnosis_type", "") or "").lower()
        return dtype in {"etiology", "metabolic", "structural"}

    async def _maybe_order_critic_exams(
        self,
        patient_id: str,
        recommended_exams: List[str],
        exam_results: Dict[str, Any],
        collected_info: Optional[Dict[str, Any]] = None,
        candidate_diseases: Optional[List[Any]] = None,
        judge_decision: Optional[Dict[str, Any]] = None,
        add_strong_verification: bool = True,
        force_deferred_anchor_round: bool = False,
    ) -> Dict[str, Any]:
        has_judge_gap_plan = bool(
            judge_decision
            and (
                (judge_decision.get("deferred_gap_closure_tasks") or [])
                or (judge_decision.get("exam_priority_overrides") or [])
            )
        )
        if (
            not recommended_exams
            and not has_judge_gap_plan
            or self._remaining_case_seconds() < self.diagnosis_critic.corrective_exam_min_seconds
        ):
            return {}
        planner = self._get_planner()
        if planner.exam_rounds >= self.max_exam_rounds and not force_deferred_anchor_round:
            return {}
        strategy: Dict[str, Any] = {}
        if judge_decision:
            strategy = self.exam_agent.recommend(
                collected_info=collected_info or {},
                candidate_diseases=candidate_diseases,
                proposed_items=recommended_exams or [],
                existing_results=exam_results,
                judge_decision=judge_decision,
            )
        if strategy.get("items"):
            items = self._strategy_order_items(
                strategy,
                collected_info=collected_info or {},
                candidate_diseases=candidate_diseases,
                existing_results=exam_results,
                max_items=self.diagnosis_critic.max_corrective_exam_items,
                add_strong_verification=False,
            )
        else:
            items = self.exam_agent.prepare_order_items(
                recommended_exams,
                collected_info=collected_info or {},
                candidate_diseases=candidate_diseases,
                existing_results=exam_results,
                max_items=self.diagnosis_critic.max_corrective_exam_items,
                add_strong_verification=add_strong_verification,
            )
        normalized_recommended, _ = self.knowledge.normalize_examinations(
            recommended_exams or []
        )
        blocked_items = [
            item for item in normalized_recommended
            if item not in set(items or [])
        ]
        if blocked_items or strategy.get("exam_authorization_details"):
            self._last_exam_authorization.append(
                {
                    "stage": "critic_corrective_exam",
                    "strict_diagnosis_driven": True,
                    "differential_driven": bool(strategy.get("differential_driven")),
                    "primary_diagnosis": (
                        str(candidate_diseases[0])
                        if candidate_diseases
                        else ""
                    ),
                    "authorized_items": list(items or []),
                    "reserved_gap_items": list(strategy.get("reserved_gap_items") or []),
                    "source_decision_version": strategy.get("source_decision_version", 0),
                    "source_evidence_version": strategy.get("source_evidence_version", 0),
                    "blocked_items": list(dict.fromkeys(blocked_items)),
                    "exam_authorization_details": list(
                        strategy.get("exam_authorization_details") or []
                    ),
                }
            )
        if not items:
            return {}
        try:
            response = await self.actions.order_examination(
                patient_id=patient_id,
                items=items,
                reason="提交前诊断审查发现低置信或未解释证据，补充最具鉴别价值的检查。",
            )
        except Exception as exc:
            logger.warning("[DiagnosisCritic] corrective examination failed: %s", exc)
            return {}
        new_results: Dict[str, Any] = {}
        for exam_name, exam_data in (response or {}).get("results", {}).items():
            if isinstance(exam_data, dict) and exam_data.get("status") != "invalid":
                new_results[exam_name] = exam_data
        if new_results:
            self._record_targeted_exam_result_recovery(
                patient_id=patient_id,
                stage="critic_corrective_exam",
                ordered_items=list(items),
                new_results=new_results,
                strategy=strategy,
            )
            planner.exam_rounds += 1
            planner._record_action(
                "order_examination",
                ",".join(new_results.keys()),
                "diagnosis_critic_corrective_exam",
            )
        return new_results

    @staticmethod
    def _has_deferred_anchor_target(decision, targets: Optional[List[str]]) -> bool:
        if not decision or not targets:
            return False
        target_set = {str(item).strip() for item in targets or [] if str(item).strip()}
        if not target_set:
            return False
        for candidate in getattr(decision, "candidates", []) or []:
            if str(getattr(candidate, "diagnosis", "") or "") not in target_set:
                continue
            if (
                str(getattr(candidate, "eligibility_status", "") or "") == DEFERRED
                and str(getattr(candidate, "eligibility_reason", "") or "") == "NeedsAnchor"
            ):
                return True
        return False

    # ============ 反思并保存经验 ============

    def _save_fast_reflection(
        self,
        patient_id: str,
        report: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> None:
        """Save a lightweight training note without an extra reflection LLM call."""
        diagnosis_accuracy = report.get("diagnosisAccuracy", report.get("diagnosis_accuracy", 0))
        exam_precision = report.get("examinationPrecision", report.get("examination_precision", 0))
        treatment_score = report.get("treatmentOverallScore", report.get("treatment_overall_score", 0))
        reflection = (
            "快速训练反思："
            f"诊断准确率={diagnosis_accuracy}，"
            f"检查精确率={exam_precision}，"
            f"治疗评分={treatment_score}。"
            "本轮为快速路径，已保留问诊、检查和评估结果供后续检索。"
        )
        error_types = self._classify_diagnosis_errors(report)
        audit = self._last_diagnosis_audit or {}
        self.memory.save_case_experience(
            patient_id=patient_id,
            report=report,
            reflection=reflection,
            collected_info=collected_info,
            exam_results=exam_results,
            evidence=audit.get("evidence"),
            diagnosis_decision=audit.get("diagnosis_decision"),
            error_types=error_types,
        )
        self.memory.save_diagnostic_replay(
            patient_id=patient_id,
            collected_info=collected_info,
            exam_results=exam_results,
            evidence=audit.get("evidence") or {},
            diagnosis_decision=audit.get("diagnosis_decision") or {},
            report=report,
            error_types=error_types,
            llm_candidates=audit.get("llm_candidates") or [],
            rag_chunks=audit.get("rag_chunks") or [],
            case_audit=audit,
        )
        self._record_exam_alias_feedback(patient_id, report, exam_results)
        self._record_diagnostic_rule_feedback(patient_id, report, collected_info, exam_results)
        logger.info("[Reflection] fast reflection saved for %s", patient_id)

    def _classify_diagnosis_errors(self, report: Dict[str, Any]) -> List[str]:
        """Classify evaluation failures into actionable diagnosis subsystems."""
        errors: List[str] = []
        detail = report.get("diagnosisDetail") or report.get("diagnosis_detail") or {}
        if not isinstance(detail, dict):
            detail = {}
        expected = detail.get("expected") or report.get("finalDiagnosis") or []
        submitted = detail.get("submitted") or report.get("diagnosis") or []
        if isinstance(expected, str):
            expected = [expected]
        if isinstance(submitted, str):
            submitted = [submitted]
        audit = self._last_diagnosis_audit or {}
        decision = audit.get("diagnosis_decision") or {}
        candidates = decision.get("candidates") or []
        candidate_names = [str(item.get("diagnosis")) for item in candidates if isinstance(item, dict)]
        final_names = [str(item) for item in decision.get("final_diagnoses", []) or []]
        for name in expected:
            name = str(name)
            if not self.diagnosis_engine.knowledge.is_allowed(name):
                errors.append("namespace_error")
            elif name not in candidate_names[:5]:
                errors.append("candidate_recall_error")
            elif name not in final_names:
                errors.append("candidate_ranking_error")
            matched = next(
                (
                    item.get("matched_evidence") or []
                    for item in candidates
                    if isinstance(item, dict) and item.get("diagnosis") == name
                ),
                [],
            )
            if not matched:
                errors.append("evidence_extraction_failure")
        if any(not self.diagnosis_engine.knowledge.is_allowed(item) for item in submitted):
            errors.append("namespace_error")
        examination_detail = report.get("examinationDetail") or report.get("examination_detail") or {}
        try:
            if float(examination_detail.get("coverage", 1.0)) < 0.5:
                errors.append("insufficient_examination")
        except (TypeError, ValueError):
            pass
        treatment_detail = report.get("treatmentDetail") or report.get("treatment_detail") or {}
        try:
            if float(treatment_detail.get("safety", 1.0)) < 0.8:
                errors.append("treatment_safety_failure")
        except (TypeError, ValueError):
            pass
        return list(dict.fromkeys(errors))

    def _record_exam_alias_feedback(
        self,
        patient_id: str,
        report: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> None:
        """Collect exam alias evidence from training feedback without breaking training."""
        try:
            if not hasattr(self.knowledge, "record_exam_alias_feedback"):
                return
            submitted_items = list((exam_results or {}).keys())
            stats = self.knowledge.record_exam_alias_feedback(
                patient_id=patient_id,
                report=report,
                submitted_items=submitted_items,
            )
            if stats.get("pending") or stats.get("promoted"):
                logger.info(
                    "[ExamAlias] feedback recorded: pending=%s, promoted=%s",
                    stats.get("pending", 0),
                    stats.get("promoted", 0),
                )
        except Exception as exc:
            logger.warning("[ExamAlias] feedback collection failed: %s", exc)

    def _record_diagnostic_rule_feedback(
        self,
        patient_id: str,
        report: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> None:
        """Collect evaluation feedback as shadow evidence for replay validation."""
        try:
            audit = self._last_diagnosis_audit or {}
            stats = self.diagnostic_learning.record_feedback(
                patient_id=patient_id,
                report=report,
                evidence=audit.get("evidence") or {},
                diagnosis_decision=audit.get("diagnosis_decision") or {},
                error_types=self._classify_diagnosis_errors(report),
            )
            if stats.get("pending") or stats.get("updated"):
                logger.info(
                    "[EvidenceRules] shadow candidates added=%s updated=%s",
                    stats.get("pending", 0),
                    stats.get("updated", 0),
                )
        except Exception as exc:
            logger.warning("[EvidenceRules] feedback collection failed: %s", exc)

    async def _reflect_and_save(
        self,
        patient_id: str,
        report: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
    ) -> None:
        """反思评估结果并保存经验。

        Args:
            patient_id: 患者 ID
            report: 评估报告
            collected_info: 收集到的患者信息
            exam_results: 检查结果
        """
        # 三源 RAG：knowledge + memory + 当前病例
        _sym_r = (collected_info or {}).get("symptoms") or []
        _final_dx = None
        try:
            _final_dx = (report or {}).get("finalDiagnosis") or (report or {}).get("diagnosis")
        except Exception:
            _final_dx = None
        _cands_r = [_final_dx] if _final_dx else None
        try:
            knowledge_context_r = self.knowledge.build_rag_context(_sym_r, _cands_r)
        except Exception:
            knowledge_context_r = ""
        try:
            if getattr(self, "memory_manager", None):
                memory_context_r = self.memory_manager.build_semantic_context(
                    collected_info, _cands_r
                )
                if memory_context_r:
                    knowledge_context_r = memory_context_r
        except Exception:
            pass

        # 多维召回历史经验
        try:
            relevant_exp_r = self._get_cached_experience(collected_info) or ""
        except Exception:
            relevant_exp_r = ""

        # 构建反思 prompt
        reflect_prompt = self.prompt.build_reflection_prompt(
            report=report,
            collected_info=collected_info,
            exam_results=exam_results,
            knowledge_context=knowledge_context_r,
            relevant_experience=relevant_exp_r,
        )
        messages = [
            {"role": "system", "content": reflect_prompt},
            {"role": "user", "content": "请进行结构化反思，总结经验教训。"},
        ]

        # 使用 LLM 生成反思总结
        reflection = await self._llm_chat(
            messages,
            temperature=0.5,
            purpose="reflection",
        )

        if not reflection:
            # 回退到简单反思
            diagnosis_accuracy = report.get("diagnosisAccuracy", 0)
            exam_precision = report.get("examinationPrecision", 0)
            treatment_score = report.get("treatmentOverallScore", 0)
            reflection = (
                f"诊断准确率: {diagnosis_accuracy}, "
                f"检查精确率: {exam_precision}, "
                f"治疗评分: {treatment_score}。"
            )
            if diagnosis_accuracy < 0.8:
                reflection += " 需要改进问诊策略，收集更多鉴别诊断信息。"
            if exam_precision < 0.8:
                reflection += " 需要优化检查选择，减少不必要的检查。"
            if treatment_score < 0.8:
                reflection += " 需要改进治疗方案，提高个性化和有效性。"

        # 保存到记忆。失败病例仅作为纠错教训渲染，并单独保留完整诊断回放。
        error_types = self._classify_diagnosis_errors(report)
        audit = self._last_diagnosis_audit or {}
        self.memory.save_case_experience(
            patient_id=patient_id,
            report=report,
            reflection=reflection,
            collected_info=collected_info,
            exam_results=exam_results,
            evidence=audit.get("evidence"),
            diagnosis_decision=audit.get("diagnosis_decision"),
            error_types=error_types,
        )
        self.memory.save_diagnostic_replay(
            patient_id=patient_id,
            collected_info=collected_info,
            exam_results=exam_results,
            evidence=audit.get("evidence") or {},
            diagnosis_decision=audit.get("diagnosis_decision") or {},
            report=report,
            error_types=error_types,
            llm_candidates=audit.get("llm_candidates") or [],
            rag_chunks=audit.get("rag_chunks") or [],
            case_audit=audit,
        )
        self._record_exam_alias_feedback(patient_id, report, exam_results)
        self._record_diagnostic_rule_feedback(patient_id, report, collected_info, exam_results)

        # ============ 自迭代闭环 ============
        # 1) 缺陷检测 → 2) 编译为策略补丁 → 3) 反馈本例 ΔScore
        try:
            if (
                self.detector is not None
                and self.policy_store is not None
                and self.candidate_policy_store is not None
                and self.rule_generalizer is not None
            ):
                # 步骤1：缺陷检测（规则通道 + LLM 归因通道合并；后者受 config 开关与 llm_chat 注入双重控制）
                _defects = await self.detector.detect_all(
                    report=report,
                    collected_info=collected_info,
                    exam_results=exam_results,
                    action_history=(self._planner.action_history
                                    if self._planner else None),
                    use_llm=self.self_improve_use_llm_attribute,
                )
                # 步骤2：单例失败只写候选策略库，不直接写 active/shadow PolicyStore。
                _policies = self.rule_generalizer.generalize(
                    _defects,
                    source_case=patient_id,
                )
                _candidate_stats = self.candidate_policy_store.upsert_many(_policies)
                # 步骤3：本例用到的补丁 → 结合 ΔScore 反馈
                _used_ids = []
                if self._planner is not None:
                    _used_ids = list(self._planner._last_used_patch_ids or [])
                if _used_ids:
                    _overall = _overall_score(report)
                    _baseline = self.memory.get_score_baseline(
                        final_dx=(report or {}).get("finalDiagnosis")
                                 or (report or {}).get("diagnosis")
                    ) if hasattr(self.memory, "get_score_baseline") else 0.7
                    _delta = None
                    if _overall is not None and _baseline is not None:
                        _delta = _overall - _baseline
                    self.policy_store.record_outcome(_used_ids, _delta or 0.0)
                    logger.info(
                        f"[自迭代] 本例补丁反馈: used={len(_used_ids)}, "
                        f"score={_overall}, baseline={_baseline}, delta={_delta}"
                    )
                # 步骤4：每 5 例做一次 audit（元迭代）
                _n_cases = len(getattr(self.memory, "notes", []) or [])
                if (
                    not self.freeze_active_learning
                    and _n_cases > 0
                    and _n_cases % 5 == 0
                ):
                    self.policy_store.audit()

                if _defects:
                    logger.info(
                        "[SelfImprove] failure attributions=%s candidate_policies=%s "
                        "updated=%s quarantined=%s",
                        len(_defects),
                        _candidate_stats.get("candidate", 0),
                        _candidate_stats.get("updated", 0),
                        _candidate_stats.get("quarantined", 0),
                    )
        except Exception as _e:
            logger.warning(f"[自迭代] 反思阶段闭环失败(不影响主流程): {_e}")

        logger.info(f"[反思] 患者 {patient_id} 反思已保存")

    # ============ 思考链 ============

    def _record_thinking_snapshot(
        self,
        thinking: Dict[str, Any],
        *,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        phase: str,
    ) -> None:
        if not isinstance(thinking, dict) or not self._case_id_for_thinking:
            return
        try:
            raw_case_text = self._raw_case_text_from_state(collected_info, chat_history)
            evidence = self._normalize_with_exam_recovery(
                collected_info,
                exam_results or {},
                raw_case_text=raw_case_text,
            )
            snapshot_id = pattern_evidence_snapshot_hash(evidence)
        except Exception:
            snapshot_id = ""
        try:
            round_number = len(exam_results or {})
            snapshot = ThinkingSnapshot.from_thinking(
                thinking,
                case_id=self._case_id_for_thinking,
                patient_id=self._case_id_for_thinking,
                phase=phase,
                round_id=f"round_{min(max(round_number, 0), 99):02d}",
                case_version=round_number,
                evidence_snapshot_id=snapshot_id,
            )
            payload = snapshot.to_dict()
            existing_ids = {
                str(item.get("snapshot_id") or "")
                for item in self._thinking_snapshots
                if isinstance(item, dict)
            }
            if payload.get("snapshot_id") not in existing_ids:
                self._thinking_snapshots.append(payload)
        except Exception as exc:
            logger.warning("[PatternRecall] failed to record thinking snapshot: %s", exc)

    async def _think(
        self,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        phase: str,
        relevant_experience: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """思考当前诊疗状态，生成鉴别诊断和下一步行动指引。

        合并了充分性判断功能：返回结果中包含 is_sufficient 字段。

        Args:
            collected_info: 已收集的患者信息
            exam_results: 已有检查结果
            chat_history: 对话历史
            phase: 当前阶段（"inquiry" 或 "examination"）
            relevant_experience: 相关历史经验

        Returns:
            思考结果，包含：
            - differential_diagnosis: 鉴别诊断列表
            - key_unknowns: 关键未知项
            - is_sufficient: 信息是否足够
            - next_action: 下一步行动建议
            - action_reasoning: 行动理由
        """
        # 前置知识库召回：基于症状检索候选疾病，缩小 LLM 搜索空间
        knowledge_context = ""
        try:
            symptoms = collected_info.get("symptoms", []) or []
            if symptoms and getattr(self, "memory_manager", None):
                memory_context = self.memory_manager.build_semantic_context(
                    collected_info
                )
                if memory_context:
                    knowledge_context = memory_context
                    logger.info("[思考] 已注入结构化语义记忆上下文")
            elif symptoms and getattr(self, "knowledge", None):
                knowledge_context = self.knowledge.build_rag_context(symptoms=symptoms)
                if knowledge_context:
                    logger.info("[思考] 已注入知识库 RAG 上下文")
        except Exception as e:
            logger.warning(f"[思考] 知识库召回失败: {e}")

        evidence_summary = ""
        try:
            raw_case_text = self._raw_case_text_from_state(collected_info, chat_history)
            thinking_evidence = self._normalize_with_exam_recovery(
                collected_info,
                exam_results or {},
                raw_case_text=raw_case_text,
            )
            evidence_summary = thinking_evidence.render_summary(limit=18)
        except Exception as exc:
            logger.warning("[PatternRecall] failed to build thinking evidence catalog: %s", exc)

        thinking_context = self._compile_llm_context(
            "thinking",
            collected_info=collected_info,
            exam_results=exam_results or {},
            chat_history=chat_history,
            phase=phase,
            relevant_experience=relevant_experience,
            knowledge_context=knowledge_context,
            evidence_summary=evidence_summary,
        )
        thinking_prompt = self.prompt.build_thinking_prompt(
            collected_info=thinking_context.get("collected_info", collected_info),
            exam_results=thinking_context.get("exam_results", exam_results or {}),
            chat_history=thinking_context.get("chat_history", chat_history),
            phase=phase,
            relevant_experience=thinking_context.get(
                "relevant_experience", relevant_experience
            ),
            knowledge_context=thinking_context.get("knowledge_context", knowledge_context),
            evidence_summary=thinking_context.get("evidence_summary", evidence_summary),
        )
        messages = [
            {"role": "system", "content": thinking_prompt},
            {"role": "user", "content": "请进行临床推理分析。"},
        ]

        result = await self._llm_chat_json(
            messages,
            temperature=0.3,
            purpose="thinking",
        )

        if result and "differential_diagnosis" in result:
            self._mark_last_llm_consumer_result("thinking", True)
            self._record_thinking_snapshot(
                result,
                collected_info=collected_info,
                exam_results=exam_results or {},
                chat_history=chat_history,
                phase=phase,
            )
            dd_names = [d.get("diagnosis", "?") for d in result.get("differential_diagnosis", [])]
            logger.info(f"[思考] 阶段={phase}, 鉴别诊断: {dd_names}")
            logger.info(f"[思考] 关键未知项: {result.get('key_unknowns', [])}")
            logger.info(f"[思考] 信息充分: {result.get('is_sufficient', False)}")
            logger.info(f"[思考] 建议下一步: {result.get('next_action', '')}")
            return result

        logger.warning("[思考] LLM 思考失败，返回空思考结果")
        return {}

    # ============ LLM 辅助方法 ============

    async def _llm_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        purpose: str = "unclassified",
    ) -> str:
        """调用 LLM 进行对话。

        Args:
            messages: 消息列表
            temperature: 生成温度

        Returns:
            LLM 响应文本
        """
        if not self._can_call_llm("chat"):
            self._append_llm_audit(
                kind="chat",
                purpose=purpose,
                json_expected=False,
                model_invoked=False,
                fallback_used=True,
            )
            return ""
        try:
            if self.log_llm_prompts:
                logger.debug(f"[LLM] Prompt: {json.dumps(messages, ensure_ascii=False)[:500]}...")

            response = await self.llm.chat(messages, temperature=temperature)
            self._bump_llm_counter("chat")
            self._append_llm_audit(
                kind="chat",
                purpose=purpose,
                json_expected=False,
                metadata=getattr(self.llm, "last_call_metadata", {}) or {},
                fallback_used=not bool(response.strip()),
            )

            if self.log_llm_prompts:
                logger.debug(f"[LLM] Response: {response[:500]}...")

            return response.strip()

        except Exception as e:
            self._append_llm_audit(
                kind="chat",
                purpose=purpose,
                json_expected=False,
                metadata=getattr(self.llm, "last_call_metadata", {}) or {},
                exception=e,
                fallback_used=True,
            )
            logger.error(f"[LLM] 调用失败: {e}")
            return ""

    async def _llm_chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        purpose: str = "unclassified",
    ) -> Dict[str, Any]:
        """调用 LLM 并解析 JSON 响应。

        Args:
            messages: 消息列表
            temperature: 生成温度

        Returns:
            解析后的 JSON 字典
        """
        if not self._can_call_llm("json"):
            self._append_llm_audit(
                kind="json",
                purpose=purpose,
                json_expected=True,
                model_invoked=False,
                fallback_used=True,
            )
            return {}
        logical_call_id = self._next_llm_logical_call_id()
        try:
            max_tokens = self.llm_contract_executor.output_tokens_for(purpose)
            result = await self.llm.chat_json(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._bump_llm_counter("json")
            parse_failed = (
                isinstance(result, dict)
                and set(result.keys()) == {"raw_response"}
                and isinstance(result.get("raw_response"), str)
            )
            normalization = self.llm_contract_executor.normalize(result, purpose)
            result = normalization.value
            validation = self.llm_contract_executor.validate(result, purpose)
            metadata = dict(getattr(self.llm, "last_call_metadata", {}) or {})
            metadata.update(
                {
                    "logical_call_id": logical_call_id,
                    "attempt_index": 1,
                    "attempt_type": "generate",
                    "contract_validation": validation.to_audit(),
                    "contract_version": validation.contract_version,
                    "deterministic_normalizations": normalization.normalizations,
                    "requested_max_tokens": max_tokens,
                }
            )
            self._append_llm_audit(
                kind="json",
                purpose=purpose,
                json_expected=True,
                metadata=metadata,
                parsed_value=result,
                fallback_used=parse_failed,
            )
            if validation.accepted or not self.llm_contract_executor.should_repair(validation):
                return result
            repaired = await self._repair_llm_json_contract(
                messages=messages,
                previous_value=result,
                validation=validation,
                logical_call_id=logical_call_id,
                purpose=purpose,
            )
            if repaired is not None:
                return repaired
            return result
        except Exception as e:
            metadata = dict(getattr(self.llm, "last_call_metadata", {}) or {})
            metadata.update(
                {
                    "logical_call_id": logical_call_id,
                    "attempt_index": 1,
                    "attempt_type": "generate",
                }
            )
            self._append_llm_audit(
                kind="json",
                purpose=purpose,
                json_expected=True,
                metadata=metadata,
                exception=e,
                fallback_used=True,
            )
            logger.error(f"[LLM] JSON 调用失败: {e}")
            return {}

    async def _repair_llm_json_contract(
        self,
        *,
        messages: List[Dict[str, str]],
        previous_value: Any,
        validation: Any,
        logical_call_id: str,
        purpose: str,
    ) -> Optional[Dict[str, Any]]:
        if not self._can_call_llm("json_repair", purpose):
            self._append_llm_audit(
                kind="json_repair",
                purpose=purpose,
                json_expected=True,
                metadata={
                    "logical_call_id": logical_call_id,
                    "attempt_index": 2,
                    "attempt_type": "repair",
                    "contract_validation": validation.to_audit(),
                    "contract_repair_attempted": True,
                    "contract_repair_succeeded": False,
                },
                model_invoked=False,
                fallback_used=True,
            )
            return None
        try:
            repair_messages = self.llm_contract_executor.build_repair_messages(
                original_messages=messages,
                previous_value=previous_value,
                validation=validation,
            )
            max_tokens = self.llm_contract_executor.repair_output_tokens_for(purpose)
            repair_payload = await self.llm.chat_json(
                repair_messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            self._bump_llm_counter("json_repair")
            parse_failed = (
                isinstance(repair_payload, dict)
                and set(repair_payload.keys()) == {"raw_response"}
                and isinstance(repair_payload.get("raw_response"), str)
            )
            merged = self.llm_contract_executor.merge_repair(
                previous_value=previous_value,
                repair_value=repair_payload,
                validation=validation,
                purpose=purpose,
            )
            repaired = merged.value
            repair_validation = self.llm_contract_executor.validate(repaired, purpose)
            metadata = dict(getattr(self.llm, "last_call_metadata", {}) or {})
            metadata.update(
                {
                    "logical_call_id": logical_call_id,
                    "attempt_index": 2,
                    "attempt_type": "repair",
                    "contract_validation": repair_validation.to_audit(),
                    "contract_version": repair_validation.contract_version,
                    "deterministic_normalizations": merged.normalizations,
                    "contract_repair_attempted": True,
                    "contract_repair_succeeded": bool(repair_validation.accepted),
                    "requested_max_tokens": max_tokens,
                }
            )
            self._append_llm_audit(
                kind="json_repair",
                purpose=purpose,
                json_expected=True,
                metadata=metadata,
                parsed_value=repaired,
                fallback_used=parse_failed,
            )
            if repair_validation.accepted:
                return repaired
        except Exception as exc:
            metadata = dict(getattr(self.llm, "last_call_metadata", {}) or {})
            metadata.update(
                {
                    "logical_call_id": logical_call_id,
                    "attempt_index": 2,
                    "attempt_type": "repair",
                    "contract_validation": validation.to_audit(),
                    "contract_repair_attempted": True,
                    "contract_repair_succeeded": False,
                }
            )
            self._append_llm_audit(
                kind="json_repair",
                purpose=purpose,
                json_expected=True,
                metadata=metadata,
                exception=exc,
                fallback_used=True,
            )
            logger.warning("[LLMContract] repair failed for %s: %s", purpose, exc)
        return None

    async def _extract_patient_info(
        self, patient_response: str, existing_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用 LLM 从患者回复中提取结构化信息。

        Args:
            patient_response: 患者回复文本
            existing_info: 已有的患者信息

        Returns:
            更新后的患者信息
        """
        # 构建 LLM 信息提取 prompt
        extraction_prompt = self.prompt.build_info_extraction_prompt(
            patient_response=patient_response,
            existing_info=existing_info,
        )
        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": "请从患者回复中提取结构化信息。"},
        ]

        # 调用 LLM 提取信息
        extracted = await self._llm_chat_json(
            messages,
            temperature=0.3,
            purpose="info_extraction",
        )

        if not extracted or "raw_response" in extracted:
            self._mark_last_llm_consumer_result(
                "info_extraction",
                False,
                fallback_used=True,
                fallback_trigger="consumer_rejected",
            )
            # LLM 提取失败，回退到关键词提取
            logger.warning("[信息提取] LLM 提取失败，回退到关键词提取")
            return self._fallback_parse_patient_response(patient_response, existing_info)
        self._mark_last_llm_consumer_result("info_extraction", True)

        # 合并提取结果到已有信息
        info = existing_info.copy()

        # 合并各字段（保留已有非空值）
        for key in [
            "chief_complaint", "present_illness", "past_history",
            "medication_history", "allergy_history", "family_history",
            "personal_history",
        ]:
            value = extracted.get(key, "")
            if value and (not info.get(key) or info.get(key) == ""):
                info[key] = value

        # 合并症状列表
        new_symptoms = extracted.get("symptoms", [])
        existing_symptoms = info.get("symptoms", [])
        info["symptoms"] = list(set(existing_symptoms + new_symptoms))

        # 合并症状详情
        new_details = extracted.get("symptom_details", {})
        existing_details = info.get("symptom_details", {})
        existing_details.update(new_details)
        info["symptom_details"] = existing_details

        # 保留原始回复
        if not info.get("raw_responses"):
            info["raw_responses"] = []
        info["raw_responses"].append(patient_response)

        return info

    def _fallback_parse_patient_response(
        self, response: str, existing_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """回退的患者回复解析（关键词提取）。

        Args:
            response: 患者回复文本
            existing_info: 已有的患者信息

        Returns:
            更新后的患者信息
        """
        info = existing_info.copy()

        if not info.get("raw_responses"):
            info["raw_responses"] = []
        info["raw_responses"].append(response)

        # 关键词提取
        symptoms = info.get("symptoms", [])
        symptom_keywords = [
            "发热", "咳嗽", "咳痰", "胸痛", "腹痛", "腹泻",
            "头痛", "头晕", "心悸", "胸闷", "气短", "恶心",
            "呕吐", "乏力", "食欲不振", "失眠", "水肿",
        ]
        symptom_keywords.extend([
            "呼吸困难", "呼吸急促", "气促", "喘息", "喘不上气", "发绀",
            "胸口闷", "出汗", "畏寒", "寒战", "便秘", "停经", "月经异常",
            "多饮", "多尿", "尿频", "尿急", "尿痛", "关节痛", "皮疹",
        ])
        for keyword in symptom_keywords:
            if keyword in response and keyword not in symptoms:
                symptoms.append(keyword)
        info["symptoms"] = symptoms

        return info

    async def _check_info_sufficient(
        self, collected_info: Dict[str, Any], ask_rounds: int, max_ask_rounds: int
    ) -> bool:
        """使用 LLM 判断信息是否已足够。

        Args:
            collected_info: 已收集的信息
            ask_rounds: 当前问诊轮次
            max_ask_rounds: 最大问诊轮次

        Returns:
            是否信息足够
        """
        # 快速判断：如果已达最大轮次，直接返回 True
        if ask_rounds >= max_ask_rounds:
            return True

        # 快速判断：如果关键信息缺失，直接返回 False
        if not collected_info.get("chief_complaint") and not collected_info.get("symptoms"):
            return False

        # 使用 LLM 判断
        prompt = self.prompt.build_info_sufficiency_prompt(
            collected_info=collected_info,
            ask_rounds=ask_rounds,
            max_ask_rounds=max_ask_rounds,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请判断信息是否足够。"},
        ]

        result = await self._llm_chat_json(
            messages,
            temperature=0.3,
            purpose="sufficiency_check",
        )

        if result and "is_sufficient" in result:
            self._mark_last_llm_consumer_result("sufficiency_check", True)
            is_sufficient = result["is_sufficient"]
            missing = result.get("missing_aspects", [])
            if missing:
                logger.info(f"[问诊] 信息缺口: {missing}")
            return bool(is_sufficient)

        # 回退：简单判断
        self._mark_last_llm_consumer_result(
            "sufficiency_check",
            False,
            fallback_used=True,
            fallback_trigger="consumer_rejected",
        )
        return len(collected_info.get("symptoms", [])) >= 3

    async def _check_exam_sufficient(
        self,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        exam_rounds: int,
        max_exam_rounds: int,
    ) -> bool:
        """使用 LLM 判断检查是否已足够。

        Args:
            collected_info: 收集到的患者信息
            exam_results: 检查结果
            exam_rounds: 当前检查轮次
            max_exam_rounds: 最大检查轮次

        Returns:
            是否检查足够
        """
        # 快速判断：如果已达最大轮次，直接返回 True
        if exam_rounds >= max_exam_rounds:
            return True

        # 使用 LLM 判断
        prompt = self.prompt.build_exam_sufficiency_prompt(
            collected_info=collected_info,
            exam_results=exam_results,
            exam_rounds=exam_rounds,
            max_exam_rounds=max_exam_rounds,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请判断检查是否足够。"},
        ]

        result = await self._llm_chat_json(
            messages,
            temperature=0.3,
            purpose="sufficiency_check",
        )

        if result and "is_sufficient" in result:
            self._mark_last_llm_consumer_result("sufficiency_check", True)
            is_sufficient = result["is_sufficient"]
            additional = result.get("additional_exams_needed", [])
            if additional:
                logger.info(f"[检查] 建议补充检查: {additional}")
            return bool(is_sufficient)

        # 回退：至少有一项检查结果
        self._mark_last_llm_consumer_result(
            "sufficiency_check",
            False,
            fallback_used=True,
            fallback_trigger="consumer_rejected",
        )
        return len(exam_results) >= 1

    async def _llm_generate_examination_items(
        self, messages: List[Dict[str, str]], collected_info: Dict[str, Any]
    ) -> List[str]:
        """使用 LLM 生成检查项目列表。

        Args:
            messages: LLM 消息列表
            collected_info: 已收集的患者信息

        Returns:
            检查项目列表
        """
        result = await self._llm_chat_json(
            messages,
            temperature=0.3,
            purpose="exam_generation",
        )

        if isinstance(result, list):
            self._mark_last_llm_consumer_result("exam_generation", True)
            return [str(item) for item in result if item]

        if isinstance(result, dict):
            # 尝试从字典中提取列表
            for key in ["items", "examinations", "exams", "checks"]:
                if key in result and isinstance(result[key], list):
                    self._mark_last_llm_consumer_result("exam_generation", True)
                    return [str(item) for item in result[key] if item]

        # 回退到规则推荐
        logger.warning("[检查] LLM 生成检查项目失败，回退到规则推荐")
        self._mark_last_llm_consumer_result(
            "exam_generation",
            False,
            fallback_used=True,
            fallback_trigger="consumer_rejected",
        )
        return self._fallback_generate_examination_items(collected_info)

    def _fallback_generate_examination_items(
        self, collected_info: Dict[str, Any]
    ) -> List[str]:
        """回退的检查项目推荐（基于规则）。

        Args:
            collected_info: 已收集的患者信息

        Returns:
            检查项目列表
        """
        items = []
        symptoms = collected_info.get("symptoms", [])

        if any(s in str(symptoms) for s in ["发热", "咳嗽", "咳痰", "胸痛"]):
            items.extend(["血常规", "胸部CT"])
        if any(s in str(symptoms) for s in ["腹痛", "腹泻", "恶心"]):
            items.extend(["血常规", "腹部B超"])
        if any(s in str(symptoms) for s in ["头痛", "头晕", "意识障碍"]):
            items.extend(["血常规", "头颅CT"])
        if any(s in str(symptoms) for s in ["心悸", "胸闷", "气短"]):
            items.extend(["心电图", "血常规"])

        if not items:
            items = ["血常规", "尿常规"]

        return items

    async def _llm_generate_diagnosis(
        self, messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """使用 LLM 生成诊断和治疗方案。

        Args:
            messages: LLM 消息列表

        Returns:
            包含 diagnosis, treatment_plan, reasoning 的字典
        """
        result = await self._llm_chat_json(
            messages,
            temperature=0.5,
            purpose="diagnosis",
        )

        if result and "diagnosis" in result:
            self._mark_last_llm_consumer_result("diagnosis", True)
            return result

        # 回退
        logger.warning("[诊断] LLM 生成诊断失败，返回标准目录内兜底诊断")
        self._mark_last_llm_consumer_result(
            "diagnosis",
            False,
            fallback_used=True,
            fallback_trigger="schema_missing_fields",
        )
        return {
            "diagnosis": ["上呼吸道感染"],
            "treatment_plan": "建议进一步完善检查，明确诊断后制定治疗方案。",
            "reasoning": "基于当前问诊和检查信息，暂未明确诊断，需进一步检查。",
        }


if __name__ == "__main__":
    # 当通过 python -m agent 运行时，启动 HTTP 服务
    # 注意：直接运行 python agent/agent.py 不支持（相对导入限制）
    # 请使用 python -m agent 代替
    from agent.server import run_server

    port = int(os.environ.get("PORT", "7860"))
    run_server(host="0.0.0.0", port=port)
