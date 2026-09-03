"""策略补丁库：把缺陷检测输出编译成可持久化、可命中匹配、可注入 Planner 的补丁。

补丁 (Patch) 数据结构：
{
    "id":           "p_xxxxxxxx"                  # 唯一 ID
    "type":         "exam_mandatory" | "exam_prune" | "inquiry_deepen" | "treatment_personalize"
    "trigger":      { symptoms_any / final_dx / age_min / age_max / gender / always }
    "action":       "自然语言描述（会拼进 Planner 的 system prompt）"
    "items":        [...]                         # 类型相关的结构化字段（如强制检查清单）
    "stats": {
        "hits":         int,                      # 命中次数
        "successes":    int,                      # 命中后带来 ΔScore>0 的次数
        "failures":     int,                      # 命中后带来 ΔScore<=0 的次数
        "created_at":   iso ts,
        "last_used_at": iso ts,
        "status":       "shadow" | "active" | "retired"  # 影子/生效/退役
    },
    "source": {
        "signal":       原缺陷 signal,
        "severity":     原严重度,
    }
}

命中匹配（match_patches）：对传入的 collected_info + 候选诊断，返回所有命中的 active 补丁。
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from .candidate_policy_store import promotion_decision

logger = logging.getLogger(__name__)


class PolicyStore:
    """策略补丁持久化 + 命中匹配 + 元迭代（audit/retire）。"""

    def __init__(self, store_path: str = "outputs/runtime_state/policies.json"):
        self.store_path = store_path
        self.patches: List[Dict[str, Any]] = []
        self._load()

    # ---------------- I/O ----------------

    def _load(self) -> None:
        if not os.path.isfile(self.store_path):
            self.patches = []
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self.patches = data
            elif isinstance(data, dict) and "patches" in data:
                self.patches = data["patches"]
            else:
                self.patches = []
            self.patches = [p for p in self.patches if isinstance(p, dict)]
            if self._normalize_loaded_patches():
                self._save()
                logger.info("[policy] 已迁移历史补丁 trigger 为标准字典结构")
        except Exception as e:
            logger.warning(f"[policy] 加载补丁库失败: {e}, 使用空库")
            self.patches = []

    def _normalize_loaded_patches(self) -> bool:
        """Normalize legacy patch shapes loaded from disk."""
        changed = False
        for patch in self.patches:
            old_trigger = patch.get("trigger")
            new_trigger = _normalize_trigger(old_trigger)
            if old_trigger != new_trigger:
                patch["trigger"] = new_trigger
                changed = True
            if not isinstance(patch.get("items"), list):
                item = patch.get("items")
                patch["items"] = [item] if isinstance(item, str) and item else []
                changed = True
            if not isinstance(patch.get("stats"), dict):
                patch["stats"] = {
                    "hits": 0,
                    "successes": 0,
                    "failures": 0,
                    "created_at": _now_iso(),
                    "last_used_at": None,
                    "status": "shadow",
                }
                changed = True
            if not isinstance(patch.get("source"), dict):
                patch["source"] = {"signal": str(patch.get("source") or ""), "severity": "low"}
                changed = True
            action_text = json.dumps(patch.get("action") or "", ensure_ascii=False)
            source_text = json.dumps(patch.get("source") or {}, ensure_ascii=False)
            if "required_gap_authorized" in action_text or "required_gap_authorized" in source_text:
                stats = patch.setdefault("stats", {})
                if stats.get("status") != "quarantined":
                    stats["status"] = "quarantined"
                    patch.setdefault("source", {})[
                        "quarantine_reason"
                    ] = "required_gap_authorized policies are deprecated"
                    changed = True
        return changed

    def _save(self) -> None:
        tmp_path = ""
        try:
            target_dir = os.path.dirname(self.store_path) or "."
            os.makedirs(target_dir, exist_ok=True)
            tmp_path = os.path.join(
                target_dir,
                f".{os.path.basename(self.store_path)}.{os.getpid()}.{uuid.uuid4().hex}.tmp",
            )
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.patches, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(tmp_path, self.store_path)
                tmp_path = ""
            except OSError:
                # 某些受限 Windows 沙盒禁止重命名替换，退化为直接写入以保证功能可用。
                with open(self.store_path, "w", encoding="utf-8") as f:
                    json.dump(self.patches, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            logger.warning(f"[policy] 保存补丁库失败: {e}")

    def sanitize_shadow_patches(self) -> Dict[str, int]:
        """Retire zero-hit shadow patches that cannot be matched safely.

        Historical LLM attribution sometimes emitted free-form ``signal``
        triggers or empty ``final_dx`` values. They are retained for audit, but
        must not be promoted or injected until converted to executable fields.
        """
        retired = 0
        normalized_count = 0
        for patch in self.patches:
            stats = patch.setdefault("stats", {})
            if stats.get("status", "shadow") != "shadow":
                continue
            if int(stats.get("hits", 0) or 0) > 0:
                continue

            trigger = _normalize_trigger(patch.get("trigger"))
            executable: Dict[str, Any] = {}
            if trigger.get("always") is True:
                executable["always"] = True
            symptoms = [str(item).strip() for item in _as_list(trigger.get("symptoms_any")) if str(item).strip()]
            if symptoms:
                executable["symptoms_any"] = symptoms
            final_dx = str(trigger.get("final_dx") or "").strip()
            if final_dx:
                executable["final_dx"] = final_dx
            for key in ("age_min", "age_max"):
                try:
                    if trigger.get(key) not in (None, ""):
                        executable[key] = int(float(trigger[key]))
                except (TypeError, ValueError):
                    pass
            gender = str(trigger.get("gender") or "").strip()
            if gender:
                executable["gender"] = gender

            if executable != trigger:
                source = patch.setdefault("source", {})
                source.setdefault("legacy_trigger", trigger)
                patch["trigger"] = executable
                normalized_count += 1
            usable = bool(executable)
            if usable:
                continue

            stats["status"] = "retired"
            source = patch.setdefault("source", {})
            source["retired_reason"] = "unexecutable_zero_hit_shadow_trigger"
            retired += 1

        if retired or normalized_count:
            self._save()
            logger.info(
                "[policy] sanitized zero-hit shadow patches: normalized=%s retired=%s",
                normalized_count,
                retired,
            )
        return {"retired": retired, "normalized": normalized_count}

    # ---------------- emit / dedup ----------------

    def emit_from_defects(
        self,
        defects: List[Dict[str, Any]],
        default_status: str = "shadow",
    ) -> List[Dict[str, Any]]:
        """把 DefectDetector 输出的 defects 编译为补丁；对相似补丁做去重/合并。

        Args:
            defects: DefectDetector.detect 的返回值
            default_status: 新补丁初始状态。首例建议 shadow，收敛后可 active

        Returns:
            实际新增或更新的补丁列表
        """
        touched: List[Dict[str, Any]] = []
        now = _now_iso()
        for d in defects or []:
            if not isinstance(d, dict):
                continue
            fix = d.get("suggested_fix") or {}
            if not isinstance(fix, dict) or not fix.get("type"):
                continue
            items = fix.get("items") or []
            if isinstance(items, str):
                items = [items]
            elif not isinstance(items, list):
                items = []
            candidate = {
                "id": _mk_id(),
                "type": fix.get("type"),
                "trigger": _normalize_trigger(fix.get("trigger", {"always": True})) or {"always": True},
                "action": fix.get("action") or "",
                "items": items,
                "stats": {
                    "hits": 0,
                    "successes": 0,
                    "failures": 0,
                    "created_at": now,
                    "last_used_at": None,
                    "status": default_status,
                },
                "source": {
                    "signal": d.get("signal"),
                    "severity": d.get("severity"),
                },
            }
            existing = self._find_similar(candidate)
            if existing is not None:
                # 合并 items（去重），提升严重度到较高者
                merged_items = list({*_as_list(existing.get("items")), *candidate["items"]})
                existing["items"] = merged_items
                existing["action"] = existing.get("action") or candidate["action"]
                # 严重度取更高：high > medium > low
                cur_sev = (existing.get("source") or {}).get("severity") or "low"
                new_sev = candidate["source"]["severity"] or "low"
                if _sev_rank(new_sev) > _sev_rank(cur_sev):
                    existing.setdefault("source", {})["severity"] = new_sev
                touched.append(existing)
            else:
                self.patches.append(candidate)
                touched.append(candidate)

        if touched:
            self._save()
            logger.info(f"[policy] emit 补丁 {len(touched)} 项 (库大小={len(self.patches)})")
        return touched

    def upsert_policy_candidate(self, policy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Install a validated active policy candidate into the active store.

        Candidate and temporary policies are intentionally ignored here; runtime
        failures must pass replay/promotion before they affect normal behavior.
        """
        if not isinstance(policy, dict):
            return None
        if str(policy.get("status") or "") != "active":
            return None
        action = policy.get("action") or {}
        action_text = json.dumps(action, ensure_ascii=False, sort_keys=True)
        if "required_gap_authorized" in action_text:
            return None
        now = _now_iso()
        patch = {
            "id": policy.get("policy_id") or _mk_id(),
            "type": "layer_policy",
            "target_layer": policy.get("target_layer"),
            "policy_type": policy.get("policy_type", "general_rule"),
            "trigger": {
                "policy_conditions": list(policy.get("trigger_conditions") or []),
            },
            "action": action,
            "items": [],
            "stats": {
                "hits": 0,
                "successes": 0,
                "failures": 0,
                "created_at": policy.get("created_at") or now,
                "last_used_at": None,
                "status": "active",
            },
            "source": {
                "policy_id": policy.get("policy_id"),
                "source_cases": list(policy.get("source_cases") or []),
                "validation_cases": list(policy.get("validation_cases") or []),
                "validation_metrics": dict(policy.get("validation_metrics") or {}),
                "priority": policy.get("priority"),
                "priority_class": policy.get("priority_class"),
            },
        }
        for existing in self.patches:
            if existing.get("id") == patch["id"]:
                existing.update(patch)
                self._save()
                return existing
        self.patches.append(patch)
        self._save()
        return patch

    def _find_similar(self, cand: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """同 type + trigger 关键字段一致 视为同一补丁。"""
        for p in self.patches:
            if p.get("type") != cand.get("type"):
                continue
            if _trigger_equal(p.get("trigger") or {}, cand.get("trigger") or {}):
                return p
        return None

    # ---------------- match / inject ----------------

    def match(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[str]] = None,
        include_shadow: bool = False,
    ) -> List[Dict[str, Any]]:
        """按当前上下文匹配 active（可选含 shadow）补丁。"""
        collected_info = collected_info or {}
        symptoms = [str(s) for s in (collected_info.get("symptoms") or [])]
        age = collected_info.get("age")
        gender = collected_info.get("gender")
        cands = [str(x) for x in (candidate_diseases or []) if x]

        allow_statuses = {"active"}
        if include_shadow:
            allow_statuses.add("shadow")

        hits: List[Dict[str, Any]] = []
        for p in self.patches:
            status = (p.get("stats") or {}).get("status", "shadow")
            if status not in allow_statuses:
                continue
            if _trigger_hit(p.get("trigger") or {}, symptoms, cands, age, gender):
                hits.append(p)
        return hits

    def render_for_prompt(self, patches: List[Dict[str, Any]]) -> str:
        """把命中的补丁渲染成一段可直接拼入 system prompt 的文字。"""
        if not patches:
            return ""
        lines = ["【策略补丁：以下规则由既往教训沉淀，务必遵守】"]
        for i, p in enumerate(patches, 1):
            action = p.get("action") or ""
            items = p.get("items") or []
            tag = f"[{p.get('type')}]"
            if items:
                lines.append(f"{i}. {tag} {action}（关键项: {items}）")
            else:
                lines.append(f"{i}. {tag} {action}")
        return "\n".join(lines)

    # ---------------- feedback / audit ----------------

    def record_outcome(
        self,
        used_patch_ids: List[str],
        delta_score: float,
    ) -> None:
        """记录本例使用了哪些补丁、以及本例的 ΔScore（相对该场景基线）。"""
        if not used_patch_ids:
            return
        now = _now_iso()
        for p in self.patches:
            if p.get("id") in used_patch_ids:
                stats = p.setdefault("stats", {})
                stats["hits"] = int(stats.get("hits", 0)) + 1
                stats["last_used_at"] = now
                if delta_score is not None and delta_score > 0:
                    stats["successes"] = int(stats.get("successes", 0)) + 1
                elif delta_score is not None and delta_score <= 0:
                    stats["failures"] = int(stats.get("failures", 0)) + 1
        self._save()

    def record_diagnostic_replay(
        self,
        patch_id: str,
        gains_by_case: Dict[str, float],
        promotion_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Persist independent replay evidence used by the promotion gate."""
        unique = {
            str(case_id): float(gain)
            for case_id, gain in (gains_by_case or {}).items()
            if str(case_id).strip()
        }
        count = len(unique)
        successes = sum(1 for gain in unique.values() if gain > 0)
        decision = promotion_decision(promotion_metrics or {})
        summary = {
            "independent_cases": count,
            "success_ratio": round(successes / count, 4) if count else 0.0,
            "avg_diagnosis_gain": round(sum(unique.values()) / count, 4) if count else 0.0,
            "case_gains": unique,
            "promotion_metrics": dict(promotion_metrics or {}),
            "promotion_decision": decision.to_dict(),
            "updated_at": _now_iso(),
        }
        for patch in self.patches:
            if patch.get("id") == patch_id:
                patch.setdefault("stats", {})["diagnostic_replay"] = summary
                self._save()
                return summary
        raise ValueError(f"patch {patch_id!r} does not exist")

    def audit(
        self,
        min_hits: int = 5,
        min_success_ratio: float = 0.6,
        min_replay_cases: int = 3,
        min_avg_delta: float = 0.1,
    ) -> Dict[str, int]:
        """Audit patches; shadow promotion requires explicit replay evidence.

        Runtime hits may retire a harmful patch but cannot promote it. A shadow
        patch additionally needs independent diagnostic replay evidence.
        """
        promoted, retired = 0, 0
        for p in self.patches:
            stats = p.get("stats") or {}
            hits = int(stats.get("hits", 0))
            succ = int(stats.get("successes", 0))
            fail = int(stats.get("failures", 0))
            status = stats.get("status", "shadow")
            replay = stats.get("diagnostic_replay") or {}
            replay_cases = int(replay.get("independent_cases", 0) or 0)
            replay_ratio = float(replay.get("success_ratio", 0.0) or 0.0)
            replay_delta = float(replay.get("avg_diagnosis_gain", 0.0) or 0.0)
            decision_payload = replay.get("promotion_decision") or {}
            if (
                status == "shadow"
                and replay_cases >= min_replay_cases
                and replay_ratio >= min_success_ratio
                and replay_delta >= min_avg_delta
                and bool(decision_payload.get("promote_allowed"))
            ):
                stats["status"] = "active"
                promoted += 1
                continue
            if hits < min_hits:
                continue
            ratio = succ / max(hits, 1)
            if ratio < min_success_ratio and fail >= succ:
                stats["status"] = "retired"
                retired += 1
        if promoted or retired:
            self._save()
            logger.info(f"[policy] audit 完成: promoted={promoted}, retired={retired}")
        return {"promoted": promoted, "retired": retired}


# ---------------- helpers ----------------

def _mk_id() -> str:
    return "p_" + uuid.uuid4().hex[:8]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _sev_rank(s: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(s or "low", 1)


def _normalize_trigger(trigger: Any) -> Dict[str, Any]:
    """Convert legacy or malformed trigger values into a safe dict shape."""
    if isinstance(trigger, dict):
        return trigger
    if trigger is None:
        return {}
    text = str(trigger).strip()
    if not text:
        return {}
    # LLM-generated string triggers are kept for provenance/dedup, but are not
    # executable match conditions unless later promoted into a structured rule.
    return {"signal": text}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value else []
    return [value]


def _trigger_equal(a: Any, b: Any) -> bool:
    """判定两个 trigger 是否语义相同（键集合一致 + 关键字段一致）。"""
    a = _normalize_trigger(a)
    b = _normalize_trigger(b)
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a.keys():
        va, vb = a.get(k), b.get(k)
        if isinstance(va, list) and isinstance(vb, list):
            if set(map(str, va)) != set(map(str, vb)):
                return False
        else:
            if va != vb:
                return False
    return True


def _trigger_hit(
    trigger: Any,
    symptoms: List[str],
    candidate_diseases: List[str],
    age: Any,
    gender: Any,
) -> bool:
    """判定单个补丁的 trigger 是否命中当前上下文。"""
    trigger = _normalize_trigger(trigger)
    if not trigger:
        return False
    if trigger.get("always"):
        return True

    has_condition = False

    # symptoms_any: 任一命中即可
    syms_any = trigger.get("symptoms_any") or []
    if syms_any:
        has_condition = True
        joined = " ".join(symptoms)
        if not any(str(x) and str(x) in joined for x in syms_any):
            return False

    # final_dx: 目标疾病与候选之一匹配（子串）
    final_dx = trigger.get("final_dx")
    if final_dx:
        has_condition = True
        fd = str(final_dx)
        joined_cands = " ".join(candidate_diseases)
        if fd not in joined_cands:
            return False

    # 年龄区间
    from_age = trigger.get("age_min")
    to_age = trigger.get("age_max")
    if from_age is not None or to_age is not None:
        has_condition = True
        age_num = _age_to_int(age)
        if age_num is None:
            return False
        if from_age is not None and age_num < int(from_age):
            return False
        if to_age is not None and age_num > int(to_age):
            return False

    # 性别
    g = trigger.get("gender")
    if g:
        has_condition = True
        if str(gender or "").upper() != str(g).upper():
            return False

    return has_condition


def _age_to_int(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(v)
    import re
    m = re.search(r"\d+", str(v))
    return int(m.group(0)) if m else None
