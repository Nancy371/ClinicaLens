"""
Agent HTTP 服务模块。

在端口 7860 上启动 HTTP 服务，暴露 POST /test 接口，
供 ModelScope Studio 平台调用进行测试。
"""

import asyncio
import hmac
import json
import logging
import os
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiohttp import web

from agent.care import CareError, CareRuntime, MAX_UPLOAD_BYTES, new_journey
from agent.demo import DemoRuntime

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_WEB_DIR = _ROOT / "web"
DEMO_RUNTIME_KEY = web.AppKey("demo_runtime", DemoRuntime)
CARE_RUNTIME_KEY = web.AppKey("care_runtime", CareRuntime)
SESSION_COOKIE = "clinicalens_session"

_SENSITIVE_KEYS = {
    "contestServiceToken",
    "serviceToken",
    "token",
    "api_key",
    "apiKey",
    "MODEL_API_KEY",
    "SERVICE_TRAIN_TOKEN",
    "authorization",
    "Authorization",
}


def _default_final_result(reason: str = "") -> Dict[str, Any]:
    """返回评测器可解析的兜底结果。"""
    return {
        "diagnosis": ["待明确诊断"],
        "treatment_plan": "当前信息不足，建议进一步问诊并完善必要检查后制定治疗方案。",
        "reasoning": reason or "Agent 未能生成完整诊疗结果，返回兜底可评估结构。",
        "conversation_rounds": 0,
    }


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None and str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _find_first_value(value: Any, keys: tuple, depth: int = 0) -> Any:
    """递归查找请求体中的服务字段，兼容平台包在 input/data 的情况。"""
    if depth > 4:
        return None
    if isinstance(value, dict):
        for key in keys:
            if value.get(key):
                return value[key]
        for item in value.values():
            found = _find_first_value(item, keys, depth + 1)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_first_value(item, keys, depth + 1)
            if found:
                return found
    return None


def _normalize_final_result(value: Any, reason: str = "") -> Dict[str, Any]:
    """把不同来源的结果整理成评测器需要的稳定字段。"""
    if not isinstance(value, dict):
        return _default_final_result(reason)

    normalized = dict(value)
    diagnosis = (
        _as_str_list(value.get("diagnosis"))
        or _as_str_list(value.get("diagnoses"))
        or _as_str_list(value.get("final_diagnosis"))
        or _as_str_list(value.get("finalDiagnosis"))
    )
    if not diagnosis:
        diagnosis = ["待明确诊断"]

    treatment_plan = (
        value.get("treatment_plan")
        or value.get("treatment")
        or value.get("plan")
        or "当前信息不足，建议进一步问诊并完善必要检查后制定治疗方案。"
    )

    normalized["diagnosis"] = diagnosis
    normalized["treatment_plan"] = str(treatment_plan)
    normalized["reasoning"] = str(
        value.get("reasoning")
        or value.get("reason")
        or reason
        or "已返回兜底诊疗结果。"
    )
    try:
        normalized["conversation_rounds"] = int(value.get("conversation_rounds", 0) or 0)
    except (TypeError, ValueError):
        normalized["conversation_rounds"] = 0
    return normalized


def _extract_patient_ids(body: Dict[str, Any]) -> List[str]:
    """兼容平台可能传入的 patient_id / caseId / cases 等字段。"""
    id_keys = ("patient_id", "patientId", "case_id", "caseId", "id")
    list_keys = ("patient_ids", "patientIds", "case_ids", "caseIds", "ids")

    patient_ids: List[str] = []
    for key in id_keys:
        if body.get(key):
            patient_ids.append(str(body[key]))

    for key in list_keys:
        for item in _as_str_list(body.get(key)):
            patient_ids.append(item)

    cases = body.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, dict):
                for key in id_keys:
                    if case.get(key):
                        patient_ids.append(str(case[key]))
                        break
            elif case is not None:
                patient_ids.append(str(case))

    # 有些平台会把病例包在 input/data 里。
    for wrapper_key in ("input", "inputs", "data"):
        wrapped = body.get(wrapper_key)
        if isinstance(wrapped, dict):
            patient_ids.extend(_extract_patient_ids(wrapped))
        elif isinstance(wrapped, list):
            for item in wrapped:
                if isinstance(item, dict):
                    patient_ids.extend(_extract_patient_ids(item))
                elif item is not None:
                    patient_ids.append(str(item))

    return list(dict.fromkeys(patient_ids))


def _redact_for_log(value: Any, depth: int = 0) -> Any:
    """生成适合日志记录的脱敏摘要，避免泄露 token 和病例详情。"""
    if depth > 2:
        return "<nested>"
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if str(key) in _SENSITIVE_KEYS:
                redacted[key] = "<redacted>"
            elif key in {"config", "input", "inputs", "data", "cases"}:
                redacted[key] = "<omitted>"
            else:
                redacted[key] = _redact_for_log(item, depth + 1)
        return redacted
    if isinstance(value, list):
        return [_redact_for_log(item, depth + 1) for item in value[:5]]
    if isinstance(value, str) and len(value) > 80:
        return value[:80] + "..."
    return value


def _apply_allowed_config_overrides(config: Dict[str, Any], override: Any) -> Dict[str, Any]:
    """只接受安全白名单配置覆盖，避免请求改写文件路径或外部服务地址。"""
    merged = deepcopy(config)
    if not isinstance(override, dict):
        return merged

    if isinstance(override.get("test"), dict):
        test_override = override["test"]
        allowed_test = {}
        if "patient_ids" in test_override:
            allowed_test["patient_ids"] = _as_str_list(test_override.get("patient_ids"))
        if "patient_count" in test_override:
            try:
                allowed_test["patient_count"] = max(1, int(test_override["patient_count"]))
            except (TypeError, ValueError):
                pass
        if allowed_test:
            merged.setdefault("test", {}).update(allowed_test)

    for key in ("max_ask_rounds", "max_exam_rounds"):
        if key in override:
            try:
                merged[key] = max(1, int(override[key]))
            except (TypeError, ValueError):
                logger.warning("[Server] 忽略非法配置覆盖: %s=%r", key, override[key])

    ignored_keys = sorted(set(override.keys()) - {"test", "max_ask_rounds", "max_exam_rounds"})
    if ignored_keys:
        logger.warning("[Server] 已忽略非白名单配置覆盖: %s", ignored_keys)

    return merged


@web.middleware
async def _demo_security_middleware(
    request: web.Request,
    handler: Any,
) -> web.StreamResponse:
    """Apply strict cross-origin and browser security policy to public demo routes."""
    public_deployment = str(os.getenv("DEMO_PUBLIC_DEPLOYMENT", "")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    try:
        if (
            public_deployment
            and request.method == "POST"
            and request.path in {"/", "/test"}
        ):
            expected = os.getenv("SERVER_TEST_ACCESS_TOKEN", "")
            supplied = request.headers.get("Authorization", "")
            authorized = bool(expected) and hmac.compare_digest(
                supplied,
                f"Bearer {expected}",
            )
            if not authorized:
                response = web.json_response(
                    {"error": "legacy_endpoint_disabled_on_public_demo"},
                    status=403,
                )
            else:
                response = await handler(request)
        elif request.method == "OPTIONS" and request.path.startswith("/api/"):
            response = web.Response(status=204)
        else:
            response = await handler(request)
    except CareError as exc:
        response = web.json_response(
            {"error": exc.code, "message": exc.message},
            status=exc.status,
        )

    origin = request.headers.get("Origin", "")
    allowed_origins = {
        item.strip()
        for item in (
            os.environ.get("DEMO_ALLOWED_ORIGINS", "")
            + ","
            + os.environ.get("CARE_ALLOWED_ORIGINS", "")
        ).split(",")
        if item.strip()
    }
    if origin and origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-CSRF-Token"
        response.headers["Access-Control-Max-Age"] = "600"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; "
        "script-src 'self'; connect-src 'self' https:; frame-ancestors 'none'; base-uri 'self'"
    )
    return response


async def _start_demo_runtime(app: web.Application) -> None:
    await app[DEMO_RUNTIME_KEY].start()


async def _close_demo_runtime(app: web.Application) -> None:
    await app[DEMO_RUNTIME_KEY].close()


async def _start_care_runtime(app: web.Application) -> None:
    await app[CARE_RUNTIME_KEY].start()


async def _close_care_runtime(app: web.Application) -> None:
    await app[CARE_RUNTIME_KEY].close()


def create_app(
    demo_runtime: DemoRuntime = None,
    care_runtime: CareRuntime = None,
) -> web.Application:
    """创建 aiohttp 应用。

    Returns:
        配置好路由的 aiohttp Application
    """
    app = web.Application(
        client_max_size=MAX_UPLOAD_BYTES + 1024 * 1024,
        middlewares=[_demo_security_middleware],
    )
    app[DEMO_RUNTIME_KEY] = demo_runtime or DemoRuntime()
    app[CARE_RUNTIME_KEY] = care_runtime or CareRuntime()
    app.add_routes([
        web.post("/", handle_test),
        web.post("/test", handle_test),
        web.get("/health", handle_health),
        web.get("/", handle_demo_index),
        web.get("/index.html", handle_demo_index),
        web.get("/config.js", handle_demo_public_file),
        web.get("/manifest.webmanifest", handle_demo_public_file),
        web.get("/sw.js", handle_demo_public_file),
        web.get("/og.png", handle_demo_public_file),
        web.get("/robots.txt", handle_demo_public_file),
        web.get("/favicon.ico", handle_empty_favicon),
        web.get("/api/demo/status", handle_demo_status),
        web.get("/api/demo/cases", handle_demo_cases),
        web.get("/api/demo/cases/{case_key}", handle_demo_offline_case),
        web.get("/api/demo/metrics", handle_demo_metrics),
        web.post("/api/demo/runs", handle_demo_run),
        web.get("/api/demo/runs/{run_id}", handle_demo_job),
        web.get("/api/sample/status", handle_demo_status),
        web.get("/api/sample/cases", handle_demo_cases),
        web.get("/api/sample/cases/{case_key}", handle_demo_offline_case),
        web.get("/api/sample/metrics", handle_demo_metrics),
        web.post("/api/sample/runs", handle_demo_run),
        web.get("/api/sample/runs/{run_id}", handle_demo_job),
        web.get("/api/sample/journey", handle_care_public_sample),
        web.post("/api/v1/auth/otp/request", handle_care_otp_request),
        web.post("/api/v1/auth/otp/verify", handle_care_otp_verify),
        web.get("/api/v1/session", handle_care_session),
        web.delete("/api/v1/session", handle_care_logout),
        web.post("/api/v1/hospital-connections", handle_care_hospital_connection),
        web.post("/api/v1/records/sync", handle_care_record_sync),
        web.post("/api/v1/record-imports", handle_care_record_import),
        web.get("/api/v1/record-imports/{import_id}", handle_care_record_import_status),
        web.post("/api/v1/care-access-grants", handle_care_access_grant_create),
        web.post("/api/v1/care-access-grants/redeem", handle_care_access_grant_redeem),
        web.get("/api/v1/care-team-links", handle_care_team_links),
        web.delete("/api/v1/care-team-links/{link_id}", handle_care_team_link_delete),
        web.get("/api/v1/journeys", handle_care_journeys),
        web.post("/api/v1/journeys", handle_care_create_journey),
        web.get("/api/v1/journeys/{journey_id}", handle_care_journey),
        web.get("/api/v1/journeys/{journey_id}/exam-reports", handle_care_exam_reports),
        web.patch("/api/v1/journeys/{journey_id}/exam-reports/{report_id}", handle_care_exam_report_review),
        web.get("/api/v1/journeys/{journey_id}/patient-explanations", handle_care_patient_explanations),
        web.post("/api/v1/journeys/{journey_id}/consultation/messages", handle_care_consultation_message),
        web.get("/api/v1/journeys/{journey_id}/consultation-case-documents", handle_care_consultation_case_documents),
        web.post("/api/v1/journeys/{journey_id}/consultation-case-documents", handle_care_consultation_case_generate),
        web.patch("/api/v1/journeys/{journey_id}/consultation-case-documents/{document_id}", handle_care_consultation_case_confirm),
        web.patch("/api/v1/journeys/{journey_id}/clinical-history", handle_care_clinical_history),
        web.post("/api/v1/journeys/{journey_id}/record-batches/{batch_key}/sync", handle_care_record_batch_sync),
        web.get("/api/v1/journeys/{journey_id}/assessment-versions", handle_care_assessment_versions),
        web.post("/api/v1/journeys/{journey_id}/triage", handle_care_triage),
        web.patch("/api/v1/journeys/{journey_id}/records/{record_id}", handle_care_record_review),
        web.post("/api/v1/journeys/{journey_id}/assessments", handle_care_assessment),
        web.get("/api/v1/assessment-runs/{run_id}", handle_care_assessment_run),
        web.post("/api/v1/journeys/{journey_id}/appointment-plan", handle_care_appointment_plan),
        web.patch("/api/v1/journeys/{journey_id}/appointment-plan", handle_care_booking_status),
        web.post("/api/v1/journeys/{journey_id}/doctor-documents", handle_care_doctor_document),
        web.get("/api/v1/followups", handle_care_followups),
        web.get("/api/v1/medications", handle_care_medications),
        web.post("/api/v1/medications/{medication_id}/events", handle_care_medication_event),
        web.post("/api/v1/push-subscriptions", handle_care_push_subscription),
        web.delete("/api/v1/push-subscriptions", handle_care_delete_push_subscription),
        web.get("/api/v1/account/export", handle_care_export),
        web.delete("/api/v1/account", handle_care_delete_account),
        web.get("/api/v1/clinician/journeys", handle_clinician_journeys),
        web.get("/api/v1/clinician/journeys/{journey_id}", handle_clinician_journey),
        web.get("/api/v1/clinician/journeys/{journey_id}/exam-recommendations", handle_clinician_exam_recommendations),
        web.post("/api/v1/clinician/journeys/{journey_id}/exam-recommendations/{recommendation_id}/decision", handle_clinician_exam_recommendation_decision),
        web.post("/api/v1/clinician/journeys/{journey_id}/treatment-recommendations/{recommendation_id}/decision", handle_clinician_treatment_recommendation_decision),
        web.post("/api/v1/clinician/journeys/{journey_id}/prescription-drafts/{draft_id}/sign", handle_clinician_prescription_draft_sign),
        web.post("/api/v1/clinician/journeys/{journey_id}/prescription-drafts/{draft_id}/cancel", handle_clinician_prescription_draft_cancel),
        web.post("/api/v1/clinician/journeys/{journey_id}/assessments", handle_clinician_assessment),
    ])
    assets_dir = _WEB_DIR / "assets"
    web_data_dir = _WEB_DIR / "data"
    if assets_dir.exists():
        app.router.add_static("/assets/", str(assets_dir), show_index=False)
    if web_data_dir.exists():
        app.router.add_static("/data/", str(web_data_dir), show_index=False)
    app.on_startup.append(_start_demo_runtime)
    app.on_startup.append(_start_care_runtime)
    app.on_cleanup.append(_close_demo_runtime)
    app.on_cleanup.append(_close_care_runtime)
    return app


async def handle_health(request: web.Request) -> web.Response:
    """健康检查端点。

    Args:
        request: HTTP 请求

    Returns:
        健康状态响应
    """
    return web.json_response({"status": "ok", "service": "hospital-agent"})


async def handle_demo_index(request: web.Request) -> web.StreamResponse:
    """Serve the installable C-end application shell."""
    index_path = _WEB_DIR / "index.html"
    if index_path.exists():
        return web.FileResponse(index_path)
    return await handle_health(request)


async def handle_demo_public_file(request: web.Request) -> web.StreamResponse:
    """Serve only the public root assets needed by the PWA."""
    filename = request.path.lstrip("/")
    if filename not in {"config.js", "og.png", "robots.txt", "manifest.webmanifest", "sw.js"}:
        raise web.HTTPNotFound()
    path = _WEB_DIR / filename
    if not path.exists():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


async def handle_empty_favicon(request: web.Request) -> web.Response:
    return web.Response(status=204)


async def handle_demo_status(request: web.Request) -> web.Response:
    return web.json_response(await request.app[DEMO_RUNTIME_KEY].status())


async def handle_demo_cases(request: web.Request) -> web.Response:
    runtime = request.app[DEMO_RUNTIME_KEY]
    return web.json_response(
        {
            "product": runtime.catalog.product,
            "default_case": runtime.catalog.default_case,
            "cases": runtime.catalog.list_cases(),
        }
    )


async def handle_demo_offline_case(request: web.Request) -> web.Response:
    payload = request.app[DEMO_RUNTIME_KEY].catalog.offline_case(
        request.match_info.get("case_key", "")
    )
    if payload is None:
        return web.json_response({"error": "unknown_case"}, status=404)
    return web.json_response(payload)


async def handle_demo_metrics(request: web.Request) -> web.Response:
    return web.json_response(request.app[DEMO_RUNTIME_KEY].catalog.metrics())


def _demo_client_identity(request: web.Request) -> str:
    if str(os.getenv("DEMO_TRUST_PROXY", "")).lower() in {"1", "true", "yes", "on"}:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",", 1)[0].strip()
    return str(request.remote or "unknown")


async def handle_demo_run(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    case_key = str(body.get("case_key") or "").strip()
    result = await request.app[DEMO_RUNTIME_KEY].submit(
        case_key,
        _demo_client_identity(request),
    )
    status = int(result.pop("http_status", 500))
    return web.json_response(result, status=status)


async def handle_demo_job(request: web.Request) -> web.Response:
    job = request.app[DEMO_RUNTIME_KEY].get_job(
        request.match_info.get("run_id", "")
    )
    if job is None:
        return web.json_response({"error": "run_not_found_or_expired"}, status=404)
    return web.json_response(job)


async def handle_care_public_sample(request: web.Request) -> web.Response:
    """Return the complete fictional journey used for signed-out exploration."""
    audience = str(request.query.get("audience") or "legacy").lower()
    if audience not in {"legacy", "patient", "clinician"}:
        raise CareError("invalid_audience", "预览角色无效。")
    return web.json_response(await request.app[CARE_RUNTIME_KEY].public_sample(audience))


async def _care_json(request: web.Request) -> Dict[str, Any]:
    try:
        value = await request.json()
    except Exception as exc:
        raise CareError("invalid_json", "请求内容必须是 JSON。") from exc
    if not isinstance(value, dict):
        raise CareError("invalid_json", "请求内容必须是 JSON 对象。")
    return value


async def _care_auth(
    request: web.Request,
    *,
    require_csrf: bool = False,
    role: Optional[str] = None,
) -> Dict[str, Any]:
    token = request.cookies.get(SESSION_COOKIE, "")
    session = await request.app[CARE_RUNTIME_KEY].repository.get_session(token)
    if session is None:
        raise CareError("authentication_required", "请先登录后继续。", 401)
    if require_csrf:
        supplied = request.headers.get("X-CSRF-Token", "")
        if not supplied or not hmac.compare_digest(supplied, str(session["csrf_token"])):
            raise CareError("csrf_validation_failed", "页面安全状态已过期，请刷新后重试。", 403)
    if role and str(session.get("role") or "patient") != role:
        raise CareError("role_forbidden", "当前账户无权访问该角色功能。", 403)
    return session


def _care_cookie_options(runtime: CareRuntime) -> Dict[str, Any]:
    return {
        "httponly": True,
        "secure": runtime.cookie_secure,
        "samesite": "None" if runtime.cookie_secure else "Lax",
        "max_age": 7 * 86400,
        "path": "/",
    }


async def handle_care_otp_request(request: web.Request) -> web.Response:
    body = await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].request_otp(
        str(body.get("phone") or ""),
        _demo_client_identity(request),
    )
    return web.json_response(result, status=202)


async def handle_care_otp_verify(request: web.Request) -> web.Response:
    body = await _care_json(request)
    runtime = request.app[CARE_RUNTIME_KEY]
    result = await runtime.verify_otp(
        str(body.get("phone") or ""),
        str(body.get("code") or ""),
    )
    response = web.json_response(
        {
            "status": "authenticated",
            "user": result["user"],
            "csrf_token": result["csrf_token"],
        }
    )
    response.set_cookie(SESSION_COOKIE, result["token"], **_care_cookie_options(runtime))
    return response


async def handle_care_session(request: web.Request) -> web.Response:
    session = await _care_auth(request)
    role = str(session.get("role") or "patient")
    if role == "clinician":
        journeys = await request.app[CARE_RUNTIME_KEY].list_clinician_journeys(session["user_id"])
    else:
        journeys = await request.app[CARE_RUNTIME_KEY].list_patient_journeys(session["user_id"])
    return web.json_response(
        {
            "authenticated": True,
            "user": {"id": session["user_id"], "phone_masked": session["phone_masked"], "role": role},
            "csrf_token": session["csrf_token"],
            "journeys": journeys,
        }
    )


async def handle_care_logout(request: web.Request) -> web.Response:
    await _care_auth(request, require_csrf=True)
    token = request.cookies.get(SESSION_COOKIE, "")
    await request.app[CARE_RUNTIME_KEY].repository.delete_session(token)
    response = web.json_response({"status": "signed_out"})
    response.del_cookie(SESSION_COOKIE, path="/")
    return response


async def handle_care_hospital_connection(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].connect_hospital(
        session["user_id"],
        bool(body.get("consent")),
    )
    return web.json_response(result, status=201)


async def handle_care_record_sync(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].sync_records(
        session["user_id"],
        str(body.get("simulate") or "success"),
    )
    if result.get("journey"):
        result["journey"] = await request.app[CARE_RUNTIME_KEY].get_patient_journey(
            session["user_id"], result["journey"]["id"]
        )
    return web.json_response(result)


async def handle_care_record_import(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    if not request.content_type.startswith("multipart/"):
        raise CareError("multipart_required", "请使用表单上传 PDF 或图片。", 415)
    reader = await request.multipart()
    document_kind = "medical_report"
    filename = ""
    content_type = ""
    file_data = bytearray()
    async for part in reader:
        if part.name == "document_kind":
            document_kind = (await part.text()).strip()
        elif part.name == "file" and part.filename:
            filename = part.filename
            content_type = part.headers.get("Content-Type", "application/octet-stream").split(";", 1)[0].strip().lower()
            while True:
                chunk = await part.read_chunk(size=64 * 1024)
                if not chunk:
                    break
                file_data.extend(chunk)
                if len(file_data) > MAX_UPLOAD_BYTES:
                    raise CareError("invalid_file_size", "文件不得超过 10MB。", 413)
    if not filename:
        raise CareError("file_required", "请选择需要上传的报告文件。")
    result = await request.app[CARE_RUNTIME_KEY].import_document(
        session["user_id"],
        filename=filename,
        content_type=content_type,
        data=bytes(file_data),
        document_kind=document_kind,
    )
    return web.json_response(result, status=202)


async def handle_care_record_import_status(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="patient")
    result = await request.app[CARE_RUNTIME_KEY].repository.get_import(
        session["user_id"],
        request.match_info.get("import_id", ""),
    )
    if result is None:
        raise CareError("record_import_not_found", "导入任务不存在。", 404)
    return web.json_response(result)


async def handle_care_access_grant_create(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    journey_id = str(body.get("journey_id") or "")
    result = await request.app[CARE_RUNTIME_KEY].create_care_access_grant(session["user_id"], journey_id)
    return web.json_response(result, status=201)


async def handle_care_access_grant_redeem(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="clinician")
    body = await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].redeem_care_access_grant(
        session["user_id"], str(body.get("code") or "")
    )
    return web.json_response(result, status=201)


async def handle_care_team_links(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="patient")
    items = await request.app[CARE_RUNTIME_KEY].list_patient_care_team_links(
        session["user_id"], str(request.query.get("journey_id") or "")
    )
    safe = [
        {
            "id": str(item.get("id")), "journey_id": str(item.get("journey_id")),
            "status": item.get("status"), "created_at": item.get("created_at"),
            "revoked_at": item.get("revoked_at"),
        }
        for item in items
    ]
    return web.json_response({"care_team_links": safe})


async def handle_care_team_link_delete(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    result = await request.app[CARE_RUNTIME_KEY].revoke_care_team_link(
        session["user_id"], request.match_info.get("link_id", "")
    )
    return web.json_response(result)


async def handle_care_journeys(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="patient")
    journeys = await request.app[CARE_RUNTIME_KEY].list_patient_journeys(session["user_id"])
    return web.json_response({"journeys": journeys})


async def handle_care_create_journey(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    journey = new_journey(session["user_id"])
    await request.app[CARE_RUNTIME_KEY].repository.save_journey(session["user_id"], journey)
    return web.json_response(await request.app[CARE_RUNTIME_KEY].get_patient_journey(session["user_id"], journey["id"]), status=201)


async def handle_care_journey(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="patient")
    journey = await request.app[CARE_RUNTIME_KEY].get_patient_journey(
        session["user_id"], request.match_info.get("journey_id", "")
    )
    return web.json_response(journey)


async def handle_care_exam_reports(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="patient")
    reports = await request.app[CARE_RUNTIME_KEY].get_exam_reports(
        session["user_id"], request.match_info.get("journey_id", "")
    )
    return web.json_response({"exam_reports": reports})


async def handle_care_exam_report_review(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    if not bool(body.get("disputed")):
        raise CareError("invalid_report_review", "当前接口仅用于提出检查结果争议。")
    result = await request.app[CARE_RUNTIME_KEY].dispute_exam_report(
        session["user_id"], request.match_info.get("journey_id", ""),
        request.match_info.get("report_id", ""), str(body.get("reason") or ""),
    )
    return web.json_response(result)


async def handle_care_patient_explanations(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="patient")
    explanations = await request.app[CARE_RUNTIME_KEY].get_patient_explanations(
        session["user_id"], request.match_info.get("journey_id", "")
    )
    return web.json_response({"patient_explanations": explanations})


async def handle_care_consultation_message(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    signs = body.get("danger_signs") if isinstance(body.get("danger_signs"), list) else None
    correction = body.get("correction") if isinstance(body.get("correction"), dict) else None
    result = await request.app[CARE_RUNTIME_KEY].send_consultation(
        session["user_id"],
        request.match_info.get("journey_id", ""),
        str(body.get("message") or ""),
        [str(item) for item in signs] if signs is not None else None,
        confirm_summary=body.get("summary_confirmed") is True,
        correction=correction,
    )
    return web.json_response(result, status=201)


async def handle_care_consultation_case_documents(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="patient")
    documents = await request.app[CARE_RUNTIME_KEY].consultation_case_documents(
        session["user_id"], request.match_info.get("journey_id", "")
    )
    return web.json_response({"consultation_case_documents": documents})


async def handle_care_consultation_case_generate(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    await _care_json(request)
    document = await request.app[CARE_RUNTIME_KEY].generate_consultation_case_document(
        session["user_id"], request.match_info.get("journey_id", "")
    )
    return web.json_response({"case_document": document}, status=201)


async def handle_care_consultation_case_confirm(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    corrections = body.get("corrections") if isinstance(body.get("corrections"), list) else []
    document = await request.app[CARE_RUNTIME_KEY].confirm_consultation_case_document(
        session["user_id"], request.match_info.get("journey_id", ""),
        request.match_info.get("document_id", ""), [str(item) for item in corrections],
    )
    return web.json_response({"case_document": document})


async def handle_care_clinical_history(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].update_clinical_history(
        session["user_id"], request.match_info.get("journey_id", ""), body
    )
    return web.json_response(result)


async def handle_care_record_batch_sync(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].sync_record_batch(
        session["user_id"],
        request.match_info.get("journey_id", ""),
        request.match_info.get("batch_key", ""),
    )
    if result.get("journey"):
        result["journey"] = await request.app[CARE_RUNTIME_KEY].get_patient_journey(
            session["user_id"], result["journey"]["id"]
        )
    return web.json_response(result)


async def handle_care_assessment_versions(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="patient")
    versions = await request.app[CARE_RUNTIME_KEY].get_patient_explanations(
        session["user_id"], request.match_info.get("journey_id", "")
    )
    return web.json_response({"patient_explanations": versions, "notice": "患者端不返回内部候选排序和评分。"})


async def handle_care_triage(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    signs = body.get("danger_signs") if isinstance(body.get("danger_signs"), list) else []
    result = await request.app[CARE_RUNTIME_KEY].triage(
        session["user_id"], request.match_info.get("journey_id", ""), [str(item) for item in signs]
    )
    return web.json_response(result)


async def handle_care_record_review(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].confirm_record(
        session["user_id"],
        request.match_info.get("journey_id", ""),
        request.match_info.get("record_id", ""),
        bool(body.get("confirmed")),
        str(body.get("correction") or ""),
    )
    return web.json_response(result)


async def handle_care_assessment(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    await _care_json(request)
    job = await request.app[CARE_RUNTIME_KEY].start_assessment(
        session["user_id"], request.match_info.get("journey_id", "")
    )
    return web.json_response(
        {"status": "accepted", "run_id": job["id"], "poll_url": f"/api/v1/assessment-runs/{job['id']}"},
        status=202,
    )


async def handle_care_assessment_run(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="patient")
    job = await request.app[CARE_RUNTIME_KEY].repository.get_job(
        session["user_id"], request.match_info.get("run_id", "")
    )
    if job is None:
        raise CareError("assessment_run_not_found", "分析任务不存在。", 404)
    safe = {key: deepcopy(value) for key, value in job.items() if key not in {"result"}}
    journey = await request.app[CARE_RUNTIME_KEY].get_patient_journey(session["user_id"], job["journey_id"])
    safe["result"] = {
        "patient_explanation": (journey.get("patient_explanations") or [None])[-1],
        "current_stage": journey.get("current_stage"),
    }
    return web.json_response(safe)


async def handle_care_appointment_plan(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    await _care_json(request)
    plan = await request.app[CARE_RUNTIME_KEY].create_appointment_plan(
        session["user_id"], request.match_info.get("journey_id", "")
    )
    return web.json_response(plan, status=201)


async def handle_care_booking_status(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    plan = await request.app[CARE_RUNTIME_KEY].update_booking(
        session["user_id"], request.match_info.get("journey_id", ""), bool(body.get("booked"))
    )
    return web.json_response(plan)


async def handle_care_doctor_document(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].apply_doctor_document(
        session["user_id"], request.match_info.get("journey_id", ""), body
    )
    return web.json_response(result, status=201)


async def handle_care_followups(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="patient")
    journeys = await request.app[CARE_RUNTIME_KEY].list_patient_journeys(session["user_id"])
    return web.json_response({"followups": [item for journey in journeys for item in journey.get("followups", [])]})


async def handle_care_medications(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="patient")
    journeys = await request.app[CARE_RUNTIME_KEY].list_patient_journeys(session["user_id"])
    return web.json_response({"medications": [item for journey in journeys for item in journey.get("medications", [])]})


async def handle_care_medication_event(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    event = await request.app[CARE_RUNTIME_KEY].add_medication_event(
        session["user_id"], request.match_info.get("medication_id", ""), str(body.get("type") or ""), str(body.get("note") or "")
    )
    return web.json_response(event, status=201)


async def handle_care_push_subscription(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    subscription = body.get("subscription") if isinstance(body.get("subscription"), dict) else body
    await request.app[CARE_RUNTIME_KEY].repository.add_push_subscription(session["user_id"], subscription)
    return web.json_response({"status": "subscribed"}, status=201)


async def handle_care_delete_push_subscription(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    body = await _care_json(request)
    await request.app[CARE_RUNTIME_KEY].repository.remove_push_subscription(
        session["user_id"], str(body.get("endpoint") or "")
    )
    return web.json_response({"status": "unsubscribed"})


async def handle_care_export(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="patient")
    return web.json_response(await request.app[CARE_RUNTIME_KEY].export_user_data(session["user_id"]))


async def handle_care_delete_account(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="patient")
    await request.app[CARE_RUNTIME_KEY].document_store.delete_owner(session["user_id"])
    await request.app[CARE_RUNTIME_KEY].repository.delete_user_data(session["user_id"])
    response = web.json_response({"status": "deleted"})
    response.del_cookie(SESSION_COOKIE, path="/")
    return response


async def handle_clinician_journeys(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="clinician")
    journeys = await request.app[CARE_RUNTIME_KEY].list_clinician_journeys(session["user_id"])
    return web.json_response({"journeys": journeys})


async def handle_clinician_journey(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="clinician")
    journey = await request.app[CARE_RUNTIME_KEY].get_clinician_journey(
        session["user_id"], request.match_info.get("journey_id", "")
    )
    return web.json_response(journey)


async def handle_clinician_exam_recommendations(request: web.Request) -> web.Response:
    session = await _care_auth(request, role="clinician")
    items = await request.app[CARE_RUNTIME_KEY].clinician_exam_recommendations(
        session["user_id"], request.match_info.get("journey_id", "")
    )
    return web.json_response({"exam_recommendations": items})


async def handle_clinician_exam_recommendation_decision(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="clinician")
    body = await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].decide_exam_recommendation(
        session["user_id"], request.match_info.get("journey_id", ""),
        request.match_info.get("recommendation_id", ""), body,
    )
    return web.json_response(result, status=201)


async def handle_clinician_treatment_recommendation_decision(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="clinician")
    body = await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].decide_treatment_recommendation(
        session["user_id"], request.match_info.get("journey_id", ""),
        request.match_info.get("recommendation_id", ""), body,
    )
    return web.json_response(result, status=201)


async def handle_clinician_prescription_draft_sign(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="clinician")
    body = await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].sign_prescription_draft(
        session["user_id"], request.match_info.get("journey_id", ""),
        request.match_info.get("draft_id", ""), body,
    )
    return web.json_response(result, status=201)


async def handle_clinician_prescription_draft_cancel(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="clinician")
    body = await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].cancel_prescription_draft(
        session["user_id"], request.match_info.get("journey_id", ""),
        request.match_info.get("draft_id", ""), str(body.get("rationale") or ""),
    )
    return web.json_response({"prescription_draft": result})


async def handle_clinician_assessment(request: web.Request) -> web.Response:
    session = await _care_auth(request, require_csrf=True, role="clinician")
    await _care_json(request)
    result = await request.app[CARE_RUNTIME_KEY].rerun_clinician_assessment(
        session["user_id"], request.match_info.get("journey_id", "")
    )
    return web.json_response(result, status=201)


async def handle_test(request: web.Request) -> web.Response:
    """处理 POST /test 请求。

    平台调用此接口触发 Agent 测试流程。
    请求体可包含：
    - patient_id: 单个患者 ID（可选）
    - patient_ids: 患者ID列表（可选）
    - config: 覆盖配置（可选）

    Args:
        request: HTTP 请求

    Returns:
        测试结果响应
    """
    try:
        # 解析请求体
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {"data": body}

        requested_patient_ids = _extract_patient_ids(body)
        logger.info(
            "[Server] 收到测试请求: path=%s, patient_ids=%s, body=%s",
            request.path,
            requested_patient_ids,
            json.dumps(_redact_for_log(body), ensure_ascii=False) if body else "empty",
        )

        # 加载配置
        import yaml
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        # 用请求体中的白名单字段覆盖配置
        if body.get("config"):
            config = _apply_allowed_config_overrides(config, body["config"])

        # 如果请求指定了患者/病例 ID，覆盖配置。
        if requested_patient_ids:
            config.setdefault("test", {})["patient_ids"] = requested_patient_ids

        # 请求级服务凭证：只作用于当前 Agent 实例，不污染全局环境变量。
        runtime_service = config.setdefault("_runtime_service", {})
        request_token = _find_first_value(
            body,
            (
                "contestServiceToken",
                "contest_service_token",
                "serviceToken",
                "service_token",
            ),
        )
        request_team_id = _find_first_value(
            body,
            ("teamId", "team_id", "TEAM_ID"),
        )
        if request_token:
            runtime_service["token"] = str(request_token)
            logger.info("[Server] 使用请求级服务 token")

        if request_team_id:
            runtime_service["team_id"] = str(request_team_id)
            logger.info("[Server] 使用请求级 team id")

        # 创建 Agent 实例并运行测试
        from agent.agent import MyDoctorAgent
        agent = MyDoctorAgent(config)

        # 运行测试（确保资源清理）
        try:
            run_info = await agent.run_test()
        finally:
            await agent._cleanup()

        results = []
        test_dir = ""
        results_file = ""
        if isinstance(run_info, dict):
            results = run_info.get("results", []) or []
            test_dir = run_info.get("test_dir", "") or ""
            results_file = run_info.get("results_file", "") or ""

        # 关键：评测器 (batch_evaluation) 在响应顶层找 final_result / final_results。
        # 把 results 中每个 case 的 final_result 提到顶层，兼容单条/多条场景。
        _final_results_top: List[Dict[str, Any]] = []
        for _r in results:
            if isinstance(_r, dict) and isinstance(_r.get("final_results"), list) and _r["final_results"]:
                _final_results_top.extend(
                    _normalize_final_result(item, _r.get("error", ""))
                    for item in _r["final_results"]
                )
            elif isinstance(_r, dict) and _r.get("final_result") is not None:
                _final_results_top.append(_normalize_final_result(_r["final_result"], _r.get("error", "")))
            else:
                # 兜底空结构，保证评测器一定能取到可解析字段
                _final_results_top.append(
                    _default_final_result(_r.get("error", "") if isinstance(_r, dict) else "")
                )

        if not _final_results_top:
            _final_results_top = [_default_final_result("no results")]

        for idx, final_result in enumerate(_final_results_top):
            if requested_patient_ids:
                patient_id = requested_patient_ids[min(idx, len(requested_patient_ids) - 1)]
                final_result.setdefault("patient_id", patient_id)
                final_result.setdefault("caseId", patient_id)

        _final_result_single = _final_results_top[-1]

        response_data = {
            "status": "success",
            "message": "Test completed",
            "results": results,
            "result_count": len(results),
            "test_dir": test_dir,
            "results_file": results_file,
            # ↓↓↓ 评测器约定字段：顶层必须直接可读 ↓↓↓
            "final_result": _final_result_single,
            "final_results": _final_results_top,
        }
        if requested_patient_ids:
            response_data["patient_id"] = requested_patient_ids[0]
            response_data["caseId"] = requested_patient_ids[0]

        logger.info(f"[Server] 测试完成, 结果数={len(results)}, final_results顶层字段已注入")

        return web.json_response(response_data)

    except Exception as e:
        error_msg = f"Test failed: {str(e)}"
        error_trace = traceback.format_exc()
        logger.error(f"[Server] {error_msg}\n{error_trace}")
        public_error = "Test failed; see server logs."

        return web.json_response(
            {
                "status": "error",
                "message": public_error,
                "final_result": _default_final_result(public_error),
                "final_results": [_default_final_result(public_error)],
            },
        )


def run_server(host: str = "0.0.0.0", port: int = 7860) -> None:
    """启动 HTTP 服务。

    Args:
        host: 监听地址
        port: 监听端口
    """
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"[Server] 启动 Hospital Agent 服务, 监听 {host}:{port}")

    app = create_app()
    web.run_app(app, host=host, port=port, print=logger.info)
