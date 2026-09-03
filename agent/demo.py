"""Public ClinicaLens demo catalog, projection, jobs, and rate limiting."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA_DIR = ROOT / "data" / "demo"
DEFAULT_JOB_TTL_SECONDS = 30 * 60


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return deepcopy(default)


def _public_case(case: Dict[str, Any]) -> Dict[str, Any]:
    allowed = (
        "key",
        "number",
        "eyebrow",
        "title",
        "summary",
        "focus",
        "tags",
        "offline_path",
        "live_supported",
    )
    return {key: deepcopy(case.get(key)) for key in allowed if key in case}


class DemoCatalog:
    """Read-only catalog of sanitized public demo assets."""

    def __init__(self, data_dir: Path = DEMO_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        payload = _read_json(
            self.data_dir / "cases.json",
            {"product": "ClinicaLens", "default_case": "", "cases": []},
        )
        self.product = str(payload.get("product") or "ClinicaLens")
        self.default_case = str(payload.get("default_case") or "")
        self._cases = {
            str(item.get("key")): dict(item)
            for item in payload.get("cases", [])
            if isinstance(item, dict) and item.get("key")
        }

    def list_cases(self) -> List[Dict[str, Any]]:
        return [_public_case(item) for item in self._cases.values()]

    def has_case(self, case_key: str) -> bool:
        return str(case_key or "") in self._cases

    def case_meta(self, case_key: str) -> Dict[str, Any]:
        return _public_case(self._cases.get(str(case_key or ""), {}))

    def offline_case(self, case_key: str) -> Optional[Dict[str, Any]]:
        if not self.has_case(case_key):
            return None
        payload = _read_json(self.data_dir / f"{case_key}.json", None)
        return payload if isinstance(payload, dict) else None

    def metrics(self) -> Dict[str, Any]:
        value = _read_json(self.data_dir / "metrics.json", {})
        return value if isinstance(value, dict) else {}


def _flatten_exam_evidence(exam_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for exam_name, exam in (exam_results or {}).items():
        if not isinstance(exam, dict):
            continue
        status = str(exam.get("status") or "")
        payload = exam.get("result") if isinstance(exam.get("result"), dict) else exam
        for field, value in payload.items():
            if field == "status" or isinstance(value, (dict, list)):
                continue
            evidence.append(
                {
                    "label": str(field),
                    "value": str(value),
                    "source": str(exam_name),
                    "role": "supporting" if status != "normal" else "context",
                    "reason": "来自真实检查结果，已进入诊断证据链。",
                }
            )
    return evidence[:30]


def _project_candidates(decision: Any) -> List[Dict[str, Any]]:
    if decision is None:
        return []
    final_names = set(getattr(decision, "final_diagnoses", []) or [])
    projected: List[Dict[str, Any]] = []
    for rank, candidate in enumerate((getattr(decision, "candidates", []) or [])[:5], start=1):
        diagnosis = str(getattr(candidate, "diagnosis", "") or "")
        contradicted = [str(item) for item in getattr(candidate, "contradicted_evidence", []) or []]
        if diagnosis in final_names:
            status = "selected"
            summary = "当前证据覆盖充分，并通过提交授权。"
        elif float(getattr(candidate, "score", 0) or 0) <= 0 or contradicted:
            status = "blocked"
            summary = "存在反证或未满足必要确证条件。"
        else:
            status = "differential"
            summary = "保留为鉴别方向，解释覆盖弱于首选结论。"
        projected.append(
            {
                "rank": rank,
                "diagnosis": diagnosis,
                "score": round(float(getattr(candidate, "score", 0) or 0), 4),
                "coverage": round(
                    float(getattr(candidate, "explanatory_coverage", 0) or 0), 4
                ),
                "status": status,
                "summary": summary,
                "supporting_evidence": [
                    str(item).replace("symptom:", "")
                    for item in (getattr(candidate, "matched_evidence", []) or [])[:4]
                ],
                "contradicting_evidence": contradicted[:4],
                "unresolved_gap_count": len(
                    getattr(candidate, "required_gaps", []) or []
                ),
            }
        )
    return projected


def build_live_projection(
    *,
    agent: Any,
    case_meta: Dict[str, Any],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Build a public DTO from processed Agent state without raw prompts or ids."""
    last = dict(getattr(agent, "_last_test_result", {}) or {})
    final_result = dict(last.get("final_result") or {})
    collected_info = dict(getattr(agent, "_last_collected_info", {}) or {})
    exam_results = dict(getattr(agent, "_last_exam_results", {}) or {})
    planner = agent._get_planner()

    evidence: List[Dict[str, Any]] = []
    for symptom in collected_info.get("symptoms", []) or []:
        evidence.append(
            {
                "label": str(symptom),
                "value": "患者主诉",
                "source": "问诊",
                "role": "supporting",
                "reason": "由真实问诊结果提取。",
            }
        )
    evidence.extend(_flatten_exam_evidence(exam_results))

    action_labels = {
        "ask_patient": ("信息采集", "补充影响判断的关键信息"),
        "order_examination": ("检查解析", "获取能够区分候选的检查"),
        "prescribe_treatment": ("辅助判断", "整理证据方向与就医准备"),
        "replan": ("动态重规划", "根据新证据调整下一步"),
    }
    timeline: List[Dict[str, Any]] = []
    for index, action in enumerate(planner.action_history or [], start=1):
        action_type = str(action.get("type") or "")
        phase, title = action_labels.get(action_type, ("诊疗执行", "更新诊疗状态"))
        timeline.append(
            {
                "step": index,
                "phase": phase,
                "title": title,
                "summary": str(action.get("target") or "")[:180],
                "meta": str(action.get("result_summary") or "")[:180],
            }
        )

    diagnoses = [str(item) for item in final_result.get("diagnosis", []) or []]
    decision = getattr(agent, "_last_diagnosis_decision_obj", None)
    confidence = float(getattr(decision, "confidence", 0) or 0)
    low_confidence = bool(getattr(decision, "low_confidence", False))
    if not timeline:
        timeline = [
            {
                "step": 1,
                "phase": "结论生成",
                "title": "完成真实 Agent 运行",
                "summary": "、".join(diagnoses),
                "meta": "诊疗流程已安全结束",
            }
        ]

    return {
        "schema_version": "sample-assessment.v1",
        "case": _public_case(case_meta),
        "mode": "live",
        "status": "completed",
        "timeline": timeline,
        "evidence": evidence,
        "candidates": _project_candidates(decision),
        "conclusion": {
            "leading_direction": diagnoses or ["信息不足，暂不形成候选方向"],
            "status": "requires_doctor_confirmation",
            "care_navigation": "请携带已展示的病历与检查来源线下就医，由医生决定检查与治疗。",
            "doctor_plan": None,
            "reasoning": str(final_result.get("reasoning") or ""),
            "uncertainty": {
                "level": "medium" if low_confidence or confidence < 0.65 else "low",
                "label": "仍需补充证据" if low_confidence else "候选方向较集中",
                "detail": (
                    "Agent 已明确标记低置信度，结论应结合更多检查复核。"
                    if low_confidence
                    else "该方向来自当前脱敏病例证据，仍需医生结合完整信息确认。"
                ),
            },
            "conversation_rounds": int(final_result.get("conversation_rounds", 0) or 0),
            "ordered_examinations": [
                str(item) for item in final_result.get("ordered_examinations", []) or []
            ],
            "disclaimer": "这是 AI 辅助判断，不是医生确诊；检查、治疗和用药由医生负责。",
        },
        "metrics": {"confidence": round(confidence, 4)},
        "runtime": {
            "engine": "live-agent-harness",
            "source": "authorized virtual patient service",
            "duration_seconds": round(max(0.0, elapsed_seconds), 3),
            "timed_out": bool(final_result.get("_case_timed_out", False)),
            "fallback_used": not bool(getattr(decision, "candidates", []) or []),
            "snapshot_notice": "真实 Agent 运行结果；病例标识已脱敏。",
        },
    }


@dataclass
class LimitDecision:
    allowed: bool
    remaining: int = 0
    reset_at: float = 0.0
    reason: str = ""


class PersistentDemoLimiter:
    """Upstash-backed global and hashed-client fixed-window limits."""

    def __init__(self) -> None:
        self.global_limit = _bounded_int(os.getenv("DEMO_DAILY_LIMIT"), 10, 1, 100)
        self.client_limit = _bounded_int(os.getenv("DEMO_IP_DAILY_LIMIT"), 2, 1, 20)
        self.salt = os.getenv("DEMO_RATE_LIMIT_SALT", "")
        self._global = None
        self._client = None
        url = os.getenv("UPSTASH_REDIS_REST_URL", "")
        token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
        if not (url and token and len(self.salt) >= 16):
            return
        try:
            from upstash_ratelimit.asyncio import FixedWindow, Ratelimit
            from upstash_redis.asyncio import Redis

            redis = Redis(url=url, token=token)
            self._global = Ratelimit(
                redis=redis,
                limiter=FixedWindow(max_requests=self.global_limit, window=1, unit="d"),
                prefix="clinicalens:global:v1",
            )
            self._client = Ratelimit(
                redis=redis,
                limiter=FixedWindow(max_requests=self.client_limit, window=1, unit="d"),
                prefix="clinicalens:client:v1",
            )
        except Exception:
            logger.exception("[Demo] unable to configure persistent limiter")
            self._global = None
            self._client = None

    @property
    def configured(self) -> bool:
        return self._global is not None and self._client is not None

    def hash_client(self, value: str) -> str:
        return hmac.new(
            self.salt.encode("utf-8"),
            str(value or "unknown").encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def status(self) -> LimitDecision:
        if not self.configured:
            return LimitDecision(False, reason="persistent_rate_limit_unavailable")
        try:
            remaining = int(await self._global.get_remaining("daily"))
            reset_at = float(await self._global.get_reset("daily"))
            return LimitDecision(True, max(0, remaining), reset_at)
        except Exception:
            logger.exception("[Demo] rate limit status failed")
            return LimitDecision(False, reason="persistent_rate_limit_unavailable")

    async def consume(self, client_identity: str) -> LimitDecision:
        if not self.configured:
            return LimitDecision(False, reason="persistent_rate_limit_unavailable")
        client_key = self.hash_client(client_identity)
        try:
            if int(await self._client.get_remaining(client_key)) <= 0:
                return LimitDecision(
                    False,
                    reset_at=float(await self._client.get_reset(client_key)),
                    reason="client_daily_limit",
                )
            if int(await self._global.get_remaining("daily")) <= 0:
                return LimitDecision(
                    False,
                    reset_at=float(await self._global.get_reset("daily")),
                    reason="global_daily_limit",
                )
            client_result = await self._client.limit(client_key)
            if not client_result.allowed:
                return LimitDecision(
                    False,
                    reset_at=float(client_result.reset),
                    reason="client_daily_limit",
                )
            global_result = await self._global.limit("daily")
            if not global_result.allowed:
                return LimitDecision(
                    False,
                    reset_at=float(global_result.reset),
                    reason="global_daily_limit",
                )
            return LimitDecision(
                True,
                remaining=max(0, int(global_result.remaining)),
                reset_at=float(global_result.reset),
            )
        except Exception:
            logger.exception("[Demo] persistent rate limit check failed")
            return LimitDecision(False, reason="persistent_rate_limit_unavailable")


LiveRunner = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]


class DemoRuntime:
    """In-memory job queue. Completed public DTOs expire after a short TTL."""

    def __init__(
        self,
        *,
        catalog: Optional[DemoCatalog] = None,
        limiter: Optional[PersistentDemoLimiter] = None,
        live_runner: Optional[LiveRunner] = None,
        run_timeout_seconds: Optional[float] = None,
    ) -> None:
        self.catalog = catalog or DemoCatalog()
        self.limiter = limiter or PersistentDemoLimiter()
        self.case_map = self._load_case_map()
        self.requested_live = _truthy(os.getenv("DEMO_LIVE_ENABLED", "false"))
        self.credentials_ready = self._credential_ready()
        self.max_queue = _bounded_int(os.getenv("DEMO_MAX_QUEUE"), 3, 1, 10)
        self.job_ttl_seconds = _bounded_int(
            os.getenv("DEMO_JOB_TTL_SECONDS"),
            DEFAULT_JOB_TTL_SECONDS,
            60,
            86400,
        )
        self.run_timeout_seconds = (
            max(0.01, float(run_timeout_seconds))
            if run_timeout_seconds is not None
            else float(
                _bounded_int(
                    os.getenv("DEMO_RUN_TIMEOUT_SECONDS"),
                    240,
                    30,
                    600,
                )
            )
        )
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.max_queue)
        self._worker: Optional[asyncio.Task] = None
        self._active_job_id = ""
        self._live_runner = live_runner or self._run_agent

    @staticmethod
    def _load_case_map() -> Dict[str, str]:
        try:
            payload = json.loads(os.getenv("DEMO_CASE_MAP_JSON", "{}"))
        except (TypeError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): str(value)
            for key, value in payload.items()
            if str(key) and str(value)
        }

    def _credential_ready(self) -> bool:
        required = ("SERVICE_BASE_URL", "SERVICE_TRAIN_TOKEN", "MODEL_API_KEY", "TEAM_ID")
        return all(os.getenv(name) for name in required)

    @property
    def live_enabled(self) -> bool:
        return bool(
            self.requested_live
            and self.credentials_ready
            and self.case_map
            and self.limiter.configured
        )

    def disabled_reason(self) -> str:
        if not self.requested_live:
            return "live_mode_disabled"
        if not self.credentials_ready:
            return "live_credentials_missing"
        if not self.case_map:
            return "live_case_map_missing"
        if not self.limiter.configured:
            return "persistent_rate_limit_unavailable"
        return ""

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._worker_loop(), name="clinicalens-demo-worker")

    async def close(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    async def status(self) -> Dict[str, Any]:
        limit = await self.limiter.status() if self.live_enabled else LimitDecision(False)
        return {
            "service": "ClinicaLens live agent",
            "live_enabled": self.live_enabled,
            "disabled_reason": self.disabled_reason(),
            "remaining_today": limit.remaining if limit.allowed else 0,
            "daily_limit": self.limiter.global_limit,
            "client_daily_limit": self.limiter.client_limit,
            "reset_at": limit.reset_at,
            "running": bool(self._active_job_id),
            "queued": self.queue.qsize(),
            "max_queue": self.max_queue,
        }

    def _cleanup_jobs(self) -> None:
        cutoff = time.time() - self.job_ttl_seconds
        stale = [
            job_id
            for job_id, job in self.jobs.items()
            if float(job.get("updated_at", 0) or 0) < cutoff
            and job.get("status") in {"completed", "failed"}
        ]
        for job_id in stale:
            self.jobs.pop(job_id, None)

    async def submit(self, case_key: str, client_identity: str) -> Dict[str, Any]:
        self._cleanup_jobs()
        if not self.catalog.has_case(case_key):
            return {"accepted": False, "http_status": 404, "error": "unknown_case"}
        if case_key not in self.case_map:
            return {"accepted": False, "http_status": 409, "error": "case_not_live_enabled"}
        if not self.live_enabled:
            return {
                "accepted": False,
                "http_status": 503,
                "error": self.disabled_reason() or "live_mode_unavailable",
            }
        if self.queue.full():
            return {"accepted": False, "http_status": 503, "error": "queue_full"}

        limit = await self.limiter.consume(client_identity)
        if not limit.allowed:
            return {
                "accepted": False,
                "http_status": 429,
                "error": limit.reason or "rate_limited",
                "remaining_today": limit.remaining,
                "reset_at": limit.reset_at,
            }

        job_id = uuid.uuid4().hex
        now = time.time()
        self.jobs[job_id] = {
            "run_id": job_id,
            "case_key": case_key,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": "",
        }
        self.queue.put_nowait(job_id)
        return {
            "accepted": True,
            "http_status": 202,
            "run_id": job_id,
            "status": "queued",
            "remaining_today": limit.remaining,
            "reset_at": limit.reset_at,
        }

    def get_job(self, run_id: str) -> Optional[Dict[str, Any]]:
        self._cleanup_jobs()
        job = self.jobs.get(str(run_id or ""))
        if not job:
            return None
        public = {
            "run_id": job["run_id"],
            "case_key": job["case_key"],
            "status": job["status"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
        }
        if job.get("result") is not None:
            public["result"] = deepcopy(job["result"])
        if job.get("error"):
            public["error"] = str(job["error"])
        if job.get("error_code"):
            public["error_code"] = str(job["error_code"])
        return public

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self.queue.get()
            job = self.jobs.get(job_id)
            if not job:
                self.queue.task_done()
                continue
            self._active_job_id = job_id
            job["status"] = "running"
            job["updated_at"] = time.time()
            try:
                job["result"] = await asyncio.wait_for(
                    self._live_runner(
                        str(job["case_key"]),
                        self.catalog.case_meta(str(job["case_key"])),
                    ),
                    timeout=self.run_timeout_seconds,
                )
                job["status"] = "completed"
            except asyncio.TimeoutError:
                logger.warning("[Demo] live run timed out: run_id=%s", job_id)
                job["status"] = "failed"
                job["error_code"] = "run_timed_out"
                job["error"] = "Agent 运行超时，已安全终止；请查看离线回放。"
            except Exception as exc:
                logger.error(
                    "[Demo] live run failed: run_id=%s error_type=%s",
                    job_id,
                    type(exc).__name__,
                )
                job["status"] = "failed"
                job["error"] = "Agent 运行失败，请查看离线回放或稍后重试。"
            finally:
                job["updated_at"] = time.time()
                self._active_job_id = ""
                self.queue.task_done()

    async def _run_agent(self, case_key: str, case_meta: Dict[str, Any]) -> Dict[str, Any]:
        import yaml

        from agent.agent import MyDoctorAgent

        config_path = Path(os.getenv("CONFIG_PATH", str(ROOT / "config.yaml")))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config["test"] = {
            **(config.get("test") or {}),
            "patient_ids": [self.case_map[case_key]],
            "patient_count": 1,
        }
        agent = MyDoctorAgent(config)
        started = time.monotonic()
        try:
            await agent.test(self.case_map[case_key])
            return build_live_projection(
                agent=agent,
                case_meta=case_meta,
                elapsed_seconds=time.monotonic() - started,
            )
        finally:
            await agent._cleanup()
