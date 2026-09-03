"""影子回放:补丁上线前的 A/B 验证。

给定一个候选补丁与若干历史案例, ShadowReplay 会:
  1) 让 Planner 在**不注入**补丁的条件下生成 baseline 计划
  2) 让 Planner 在**注入**补丁的条件下生成 candidate 计划
  3) 用可配置的评分器对两个计划打分,输出 ΔScore
  4) 汇总多例的成功率/平均 ΔScore, 用于判断该补丁是否值得从 shadow 提权到 active

设计要点:
- 不依赖真实平台调用 —— 只重放 planner 内部推理链
- 评分器可插拔:默认使用启发式评分器 heuristic_plan_score(),用户可自定义
- Planner 侧只需要满足 plan(collected_info, current_plan) -> plan 语义
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from .clinical_evidence import ClinicalEvidenceNormalizer, EvidenceBundle
from .candidate_policy_store import promotion_decision
from .diagnosis_engine import DiagnosisDecisionEngine

logger = logging.getLogger(__name__)


# ---------------- 默认启发式评分器 ----------------

def heuristic_plan_score(plan: Any) -> float:
    """对 planner 输出的 plan 做启发式打分,范围 [0,1]。

    维度:
      - 覆盖度:differential_diagnoses 条数(上限归一)
      - 深度:是否有 primary_hypothesis
      - 检查计划完备性:examinations 条数
      - 治疗合理性:是否有 treatments 且非空
      - 风险识别:是否含 risks / red_flags 类字段
    """
    if plan is None:
        return 0.0
    d = _as_dict(plan)

    score = 0.0
    # 1) 鉴别诊断覆盖度: 每条 +0.08, 上限 0.3
    diffs = d.get("differential_diagnoses") or d.get("differentials") or []
    if isinstance(diffs, list):
        score += min(len(diffs) * 0.08, 0.3)

    # 2) 主假设存在: +0.15
    if d.get("primary_hypothesis") or d.get("primary_diagnosis"):
        score += 0.15

    # 3) 检查计划: 每条 +0.05, 上限 0.2
    exams = d.get("examinations") or d.get("recommended_exams") or []
    if isinstance(exams, list):
        score += min(len(exams) * 0.05, 0.2)

    # 4) 治疗方案存在且非空: +0.15
    treats = d.get("treatments") or d.get("treatment_plan") or []
    if isinstance(treats, list) and len(treats) > 0:
        score += 0.15
    elif isinstance(treats, dict) and treats:
        score += 0.15

    # 5) 风险识别: +0.1
    risks = d.get("risks") or d.get("red_flags") or d.get("warnings")
    if risks:
        score += 0.1

    # 6) 追问计划: +0.1
    followups = d.get("follow_up_questions") or d.get("next_questions") or []
    if isinstance(followups, list) and len(followups) > 0:
        score += 0.1

    return max(0.0, min(1.0, score))


def _as_dict(plan: Any) -> Dict[str, Any]:
    if isinstance(plan, dict):
        return plan
    # 兼容 pydantic / dataclass / 自定义对象
    for attr in ("model_dump", "dict", "to_dict"):
        f = getattr(plan, attr, None)
        if callable(f):
            try:
                return f()
            except Exception:
                pass
    if hasattr(plan, "__dict__"):
        return {k: v for k, v in plan.__dict__.items() if not k.startswith("_")}
    return {}


# ---------------- ShadowReplay 主体 ----------------

class ShadowReplay:
    """补丁影子回放器。

    使用方式:
        replay = ShadowReplay(planner, policy_store, score_fn=heuristic_plan_score)
        result = await replay.replay_patch(case, patch_id)
        stats  = await replay.batch_evaluate(cases, patch_id)
        if stats["should_promote"]:
            policy_store.promote(patch_id)  # 或由 policy_store.audit 处理
    """

    def __init__(
        self,
        planner,
        policy_store,
        score_fn: Optional[Callable[[Any], float]] = None,
    ):
        self.planner = planner
        self.policy_store = policy_store
        self.score_fn = score_fn or heuristic_plan_score

    # ---------- 单例回放 ----------

    async def replay_patch(
        self,
        case: Dict[str, Any],
        patch_id: str,
    ) -> Dict[str, Any]:
        """对单个 case 做 A/B 回放。

        case 字段约定:
          - collected_info: dict
          - exam_results:   dict，可选
          - chat_history:   list，可选
          - relevant_experience: list，可选
          - current_plan:   dict/obj，可选，用于回放前恢复 planner.current_plan
          - case_id:        标识(可选)
        返回:
          {case_id, patch_id, baseline_score, candidate_score, delta, hit}
        """
        patch = self._find_patch(patch_id)
        if patch is None:
            raise ValueError(f"patch {patch_id!r} 不存在")

        collected_info = case.get("collected_info") or {}
        exam_results = case.get("exam_results") or {}
        chat_history = case.get("chat_history") or []
        relevant_experience = case.get("relevant_experience") or []
        current_plan = case.get("current_plan")

        # A: baseline (临时清空 store 中该补丁的匹配)
        baseline_plan = await self._plan_without_patch(
            collected_info, exam_results, chat_history, relevant_experience, current_plan, patch_id
        )
        baseline_score = self.score_fn(baseline_plan)

        # B: candidate (强制注入该补丁)
        candidate_plan = await self._plan_with_forced_patch(
            collected_info, exam_results, chat_history, relevant_experience, current_plan, patch
        )
        candidate_score = self.score_fn(candidate_plan)

        delta = candidate_score - baseline_score
        return {
            "case_id": case.get("case_id"),
            "patch_id": patch_id,
            "baseline_score": round(baseline_score, 4),
            "candidate_score": round(candidate_score, 4),
            "delta": round(delta, 4),
            "hit": delta > 0,
        }

    # ---------- 批量评估 ----------

    async def batch_evaluate(
        self,
        cases: List[Dict[str, Any]],
        patch_id: str,
        min_cases: int = 5,
        min_avg_delta: float = 0.03,
        min_success_ratio: float = 0.6,
    ) -> Dict[str, Any]:
        """批量回放并给出提权建议。

        返回:
          {n, successes, failures, avg_delta, success_ratio, per_case, should_promote}
        """
        per_case: List[Dict[str, Any]] = []
        for c in cases:
            try:
                r = await self.replay_patch(c, patch_id)
                per_case.append(r)
            except Exception as e:
                logger.warning(f"[replay] case {c.get('case_id')} 失败: {e}")

        n = len(per_case)
        successes = sum(1 for r in per_case if r["delta"] > 0)
        failures = sum(1 for r in per_case if r["delta"] < 0)
        avg_delta = sum(r["delta"] for r in per_case) / n if n else 0.0
        ratio = successes / n if n else 0.0

        should_promote = (
            n >= min_cases
            and avg_delta >= min_avg_delta
            and ratio >= min_success_ratio
        )

        summary = {
            "patch_id": patch_id,
            "n": n,
            "successes": successes,
            "failures": failures,
            "avg_delta": round(avg_delta, 4),
            "success_ratio": round(ratio, 4),
            "should_promote": bool(should_promote),
            "per_case": per_case,
        }
        logger.info(
            f"[replay] patch={patch_id} n={n} avg_delta={summary['avg_delta']} "
            f"ratio={summary['success_ratio']} promote={summary['should_promote']}"
        )
        return summary

    # ---------- 内部:控制补丁注入的双跑 ----------

    async def _plan_without_patch(
        self,
        collected_info,
        exam_results,
        chat_history,
        relevant_experience,
        current_plan,
        patch_id,
    ):
        """临时把该补丁从可匹配集合里屏蔽,再让 planner 正常 plan。"""
        original = getattr(self.planner, "policy_store", None)
        original_plan = getattr(self.planner, "current_plan", None)
        try:
            if hasattr(self.planner, "current_plan"):
                self.planner.current_plan = current_plan
            # 用一个包装:match() 时过滤掉 patch_id
            self.planner.policy_store = _MaskedStore(self.policy_store, mask_id=patch_id)
            return await _call_planner_plan(
                self.planner, collected_info, exam_results, chat_history, relevant_experience
            )
        finally:
            self.planner.policy_store = original
            if hasattr(self.planner, "current_plan"):
                self.planner.current_plan = original_plan

    async def _plan_with_forced_patch(
        self,
        collected_info,
        exam_results,
        chat_history,
        relevant_experience,
        current_plan,
        patch,
    ):
        """强制注入该补丁,即使 trigger 未命中也返回它。"""
        original = getattr(self.planner, "policy_store", None)
        original_plan = getattr(self.planner, "current_plan", None)
        try:
            if hasattr(self.planner, "current_plan"):
                self.planner.current_plan = current_plan
            self.planner.policy_store = _ForcedStore(self.policy_store, forced_patch=patch)
            return await _call_planner_plan(
                self.planner, collected_info, exam_results, chat_history, relevant_experience
            )
        finally:
            self.planner.policy_store = original
            if hasattr(self.planner, "current_plan"):
                self.planner.current_plan = original_plan

    def _find_patch(self, patch_id: str) -> Optional[Dict[str, Any]]:
        patches = getattr(self.policy_store, "patches", None) or []
        for p in patches:
            if p.get("id") == patch_id:
                return p
        return None


class DiagnosticReplay:
    """Deterministically replay diagnosis traces without calling an LLM/service."""

    def __init__(
        self,
        decision_engine: DiagnosisDecisionEngine,
        normalizer: Optional[ClinicalEvidenceNormalizer] = None,
    ):
        self.decision_engine = decision_engine
        self.normalizer = normalizer or ClinicalEvidenceNormalizer(
            ref_dir=decision_engine.knowledge.ref_dir
        )

    def evaluate(
        self,
        source: Union[str, Iterable[Dict[str, Any]]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        rows = self._load_rows(source)
        per_case: List[Dict[str, Any]] = []
        recall_hits = 0
        top1_hits = 0
        exact_hits = 0
        legal_results = 0
        negative_checks = 0
        negative_false_positives = 0

        for row in rows:
            expected = self._normalize_expected(row.get("expected"))
            if not expected:
                continue
            evidence = EvidenceBundle.from_dict(row.get("evidence") or {})
            if not evidence.observations:
                evidence = self.normalizer.normalize(
                    row.get("collected_info") or {},
                    row.get("exam_results") or {},
                )

            prior_names = row.get("llm_candidates") or []
            if not prior_names:
                old_decision = row.get("diagnosis_decision") or {}
                prior_names = old_decision.get("llm_candidates") or []
            rag_chunks = row.get("rag_chunks") or []
            decision = self.decision_engine.decide(
                {"diagnosis": prior_names},
                rag_chunks,
                evidence,
            )
            ranked = [item.diagnosis for item in decision.candidates]
            final = list(decision.final_diagnoses)
            negative_diagnoses = self._normalize_expected(
                row.get("negative_diagnoses") or row.get("forbidden_diagnoses")
            )
            recall = any(name in ranked[:top_k] for name in expected)
            top1 = bool(ranked and ranked[0] in expected)
            exact = set(final) == set(expected)
            legal = bool(final) and all(
                self.decision_engine.knowledge.is_allowed(name) for name in final
            )
            recall_hits += int(recall)
            top1_hits += int(top1)
            exact_hits += int(exact)
            legal_results += int(legal)
            negative_checks += len(negative_diagnoses)
            false_positive_names = [name for name in negative_diagnoses if name in final]
            negative_false_positives += len(false_positive_names)
            per_case.append(
                {
                    "patient_id": row.get("patient_id") or row.get("case_id"),
                    "expected": expected,
                    "final": final,
                    "top5": ranked[:top_k],
                    "recall_at_5": recall,
                    "top1_hit": top1,
                    "exact_match": exact,
                    "namespace_legal": legal,
                    "negative_diagnoses": negative_diagnoses,
                    "negative_false_positives": false_positive_names,
                    "confidence": decision.confidence,
                    "error_types": list(row.get("error_types") or []),
                }
            )

        n = len(per_case)
        return {
            "cases": n,
            "candidate_recall_at_5": round(recall_hits / n, 4) if n else 0.0,
            "top1_accuracy": round(top1_hits / n, 4) if n else 0.0,
            "exact_match_rate": round(exact_hits / n, 4) if n else 0.0,
            "namespace_legal_rate": round(legal_results / n, 4) if n else 0.0,
            "negation_false_positive_rate": (
                round(negative_false_positives / negative_checks, 4)
                if negative_checks else 0.0
            ),
            "negation_false_positive_count": negative_false_positives,
            "targets": {
                "candidate_recall_at_5": 0.9,
                "top1_accuracy": 0.7,
                "namespace_legal_rate": 1.0,
            },
            "maximum_targets": {"negation_false_positive_rate": 0.0},
            "per_case": per_case,
        }

    @staticmethod
    def promotion_summary(gains_by_case: Dict[str, float]) -> Dict[str, Any]:
        """Apply the shadow-to-active gate from independently keyed replays."""
        gains = [float(value) for value in gains_by_case.values()]
        count = len(gains)
        successes = sum(1 for value in gains if value > 0)
        ratio = successes / count if count else 0.0
        average = sum(gains) / count if count else 0.0
        return {
            "independent_cases": count,
            "success_ratio": round(ratio, 4),
            "avg_diagnosis_gain": round(average, 4),
            "should_promote": bool(count >= 3 and ratio >= 0.6 and average >= 0.1),
        }

    @staticmethod
    def policy_promotion_summary(per_case: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a candidate policy against target/neighbor/counterexample buckets."""
        rows = [dict(item) for item in per_case or [] if isinstance(item, dict)]

        def bucket(*names: str) -> List[Dict[str, Any]]:
            allowed = set(names)
            return [item for item in rows if str(item.get("bucket") or "") in allowed]

        def accuracy_delta(items: List[Dict[str, Any]]) -> float:
            if not items:
                return 0.0
            deltas = [
                float(bool(item.get("candidate_correct")))
                - float(bool(item.get("baseline_correct")))
                for item in items
            ]
            return sum(deltas) / len(deltas)

        def false_positive_delta(items: List[Dict[str, Any]]) -> float:
            if not items:
                return 0.0
            deltas = [
                float(item.get("candidate_false_positives", 0) or 0)
                - float(item.get("baseline_false_positives", 0) or 0)
                for item in items
            ]
            return sum(deltas) / len(deltas)

        def unsafe_delta(items: List[Dict[str, Any]]) -> float:
            if not items:
                return 0.0
            deltas = [
                float(bool(item.get("candidate_unsafe_submission")))
                - float(bool(item.get("baseline_unsafe_submission")))
                for item in items
            ]
            return sum(deltas) / len(deltas)

        target_rows = bucket("target", "source", "same_pattern_positive")
        neighbor_rows = bucket("neighbor", "neighboring_differential")
        counter_rows = bucket("counterexample", "negative")
        target_fix_rate = (
            sum(1 for item in target_rows if item.get("candidate_correct"))
            / len(target_rows)
            if target_rows
            else 0.0
        )
        metrics = {
            "target_fix_rate": round(target_fix_rate, 4),
            "neighboring_accuracy_delta": round(accuracy_delta(neighbor_rows), 4),
            "false_positive_increase": round(false_positive_delta(counter_rows), 4),
            "global_accuracy_delta": round(accuracy_delta(rows), 4),
            "unsafe_submission_delta": round(unsafe_delta(rows), 4),
            "bucket_counts": {
                "target": len(target_rows),
                "neighbor": len(neighbor_rows),
                "counterexample": len(counter_rows),
                "historical_stable": len(bucket("historical_stable", "stable")),
            },
        }
        decision = promotion_decision(metrics)
        return {
            **metrics,
            "promote_allowed": decision.promote_allowed,
            "failed_gates": list(decision.failed_gates),
            "should_promote": decision.promote_allowed,
        }

    @staticmethod
    def _normalize_expected(value: Any) -> List[str]:
        if isinstance(value, str):
            value = [value]
        return list(dict.fromkeys(str(item).strip() for item in (value or []) if str(item).strip()))

    @staticmethod
    def _load_rows(
        source: Union[str, Iterable[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if not isinstance(source, (str, os.PathLike)):
            return [dict(item) for item in source if isinstance(item, dict)]
        rows: List[Dict[str, Any]] = []
        try:
            with open(source, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(item, dict):
                        rows.append(item)
        except OSError as exc:
            logger.warning("[diagnostic-replay] unable to load %s: %s", source, exc)
        return rows


# ---------------- 内部包装器 ----------------

class _MaskedStore:
    """代理 PolicyStore, 让 match() 过滤指定 patch_id。"""
    def __init__(self, inner, mask_id):
        self._inner = inner
        self._mask_id = mask_id

    def match(self, collected_info, candidate_diseases, include_shadow=False):
        hits = self._inner.match(collected_info, candidate_diseases, include_shadow)
        return [h for h in hits if h.get("id") != self._mask_id]

    def render_for_prompt(self, patches):
        return self._inner.render_for_prompt(patches)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _ForcedStore:
    """代理 PolicyStore, match() 始终返回指定补丁(与其它命中合并去重)。"""
    def __init__(self, inner, forced_patch):
        self._inner = inner
        self._forced = forced_patch

    def match(self, collected_info, candidate_diseases, include_shadow=False):
        hits = self._inner.match(collected_info, candidate_diseases, include_shadow)
        ids = {h.get("id") for h in hits}
        if self._forced.get("id") not in ids:
            hits = [self._forced] + hits
        return hits

    def render_for_prompt(self, patches):
        return self._inner.render_for_prompt(patches)

    def __getattr__(self, name):
        return getattr(self._inner, name)


async def _maybe_await(x):
    import inspect
    if inspect.isawaitable(x):
        return await x
    return x


async def _call_planner_plan(
    planner,
    collected_info,
    exam_results,
    chat_history,
    relevant_experience,
):
    """兼容真实 Planner.plan 与测试 MockPlanner.plan。"""
    import inspect

    sig = inspect.signature(planner.plan)
    params = list(sig.parameters)
    if "exam_results" in params or len(params) >= 4:
        return await _maybe_await(planner.plan(
            collected_info=collected_info,
            exam_results=exam_results,
            chat_history=chat_history,
            relevant_experience=relevant_experience,
        ))
    return await _maybe_await(planner.plan(collected_info, None))
