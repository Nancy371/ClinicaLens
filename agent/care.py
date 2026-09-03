"""Longitudinal consumer care-journey domain for ClinicaLens.

The public product surface intentionally separates AI decision support from
doctor-authored diagnoses, prescriptions, and follow-up plans.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from agent.care_product import (
    all_sample_records as product_sample_records,
    answer_consultation,
    assessment_version as product_assessment_version,
    build_evidence as product_build_evidence,
    hydrate_journey,
    medication_education,
    public_sample_journey,
    quick_questions,
    sample_clinical_history,
    sample_patient_profile,
    sample_raw_case_document,
    sample_record_batches,
    treatment_reference,
)
from agent.care_roles import (
    build_patient_explanations,
    clinician_journey_dto,
    hydrate_journey_v3,
    patient_explanation,
    patient_journey_dto,
    public_sample_projection,
    reports_for_batches,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / ".care_data"
PHONE_RE = re.compile(r"^1[3-9]\d{9}$")
ALLOWED_UPLOAD_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _copy(value: Any) -> Any:
    return deepcopy(value)


def _mask_phone(phone: str) -> str:
    return f"{phone[:3]}****{phone[-4:]}"


def _safe_text(value: Any, limit: int = 180) -> str:
    return str(value or "").strip()[:limit]


class CareError(RuntimeError):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class SQLiteCareRepository:
    """Small durable repository used locally and in tests.

    Production deployments can keep the domain contract and replace this
    repository with the PostgreSQL schema shipped in ``db/postgres.sql``.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        raw = path or Path(os.getenv("CARE_DB_PATH", DEFAULT_DATA_DIR / "clinicalens.db"))
        self.path = Path(raw)
        self._db: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id TEXT PRIMARY KEY,
              phone_hash TEXT UNIQUE NOT NULL,
              phone_masked TEXT NOT NULL,
              role TEXT NOT NULL DEFAULT 'patient' CHECK(role IN ('patient', 'clinician')),
              created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
              token_hash TEXT PRIMARY KEY,
              user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              csrf_token TEXT NOT NULL,
              expires_at REAL NOT NULL,
              created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS otp_challenges (
              id TEXT PRIMARY KEY,
              phone_hash TEXT NOT NULL,
              client_hash TEXT NOT NULL,
              code_hash TEXT NOT NULL,
              expires_at REAL NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0,
              used INTEGER NOT NULL DEFAULT 0,
              created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS otp_phone_created_idx
              ON otp_challenges(phone_hash, created_at);
            CREATE INDEX IF NOT EXISTS otp_client_created_idx
              ON otp_challenges(client_hash, created_at);
            CREATE TABLE IF NOT EXISTS journeys (
              id TEXT PRIMARY KEY,
              owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              payload_json TEXT NOT NULL,
              updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS journeys_owner_idx ON journeys(owner_id);
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              kind TEXT NOT NULL,
              status TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS record_imports (
              id TEXT PRIMARY KEY,
              owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              status TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS push_subscriptions (
              id TEXT PRIMARY KEY,
              owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              endpoint_hash TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              UNIQUE(owner_id, endpoint_hash)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
              id TEXT PRIMARY KEY,
              owner_id TEXT,
              action TEXT NOT NULL,
              object_type TEXT NOT NULL,
              object_id TEXT,
              detail_json TEXT NOT NULL,
              created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS care_access_grants (
              id TEXT PRIMARY KEY,
              journey_id TEXT NOT NULL,
              owner_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              code_hash TEXT UNIQUE NOT NULL,
              expires_at REAL NOT NULL,
              used_at REAL,
              created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_care_access_grants_owner_journey
              ON care_access_grants(owner_id, journey_id);
            CREATE TABLE IF NOT EXISTS care_team_links (
              id TEXT PRIMARY KEY,
              journey_id TEXT NOT NULL,
              patient_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              clinician_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
              status TEXT NOT NULL DEFAULT 'active',
              created_at REAL NOT NULL,
              revoked_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_care_team_links_clinician_status
              ON care_team_links(clinician_id, status);
            CREATE INDEX IF NOT EXISTS idx_care_team_links_patient_journey
              ON care_team_links(patient_id, journey_id);
            """
        )
        columns = {row["name"] for row in self._db.execute("PRAGMA table_info(users)").fetchall()}
        if "role" not in columns:
            self._db.execute(
                "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'patient'"
            )
        self._db.execute("PRAGMA optimize")
        self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("Care repository is not started")
        return self._db

    async def create_otp(
        self,
        *,
        phone_hash: str,
        client_hash: str,
        code_hash: str,
        expires_at: float,
    ) -> None:
        async with self._lock:
            db = self._conn()
            cutoff = time.time() - 15 * 60
            phone_count = db.execute(
                "SELECT COUNT(*) FROM otp_challenges WHERE phone_hash=? AND created_at>=?",
                (phone_hash, cutoff),
            ).fetchone()[0]
            client_count = db.execute(
                "SELECT COUNT(*) FROM otp_challenges WHERE client_hash=? AND created_at>=?",
                (client_hash, cutoff),
            ).fetchone()[0]
            if phone_count >= 5 or client_count >= 12:
                raise CareError("otp_rate_limited", "验证码请求过于频繁，请稍后再试。", 429)
            db.execute(
                "INSERT INTO otp_challenges VALUES (?, ?, ?, ?, ?, 0, 0, ?)",
                (str(uuid.uuid4()), phone_hash, client_hash, code_hash, expires_at, time.time()),
            )
            db.commit()

    async def consume_otp(self, phone_hash: str, code_hash: str) -> bool:
        async with self._lock:
            db = self._conn()
            row = db.execute(
                """SELECT * FROM otp_challenges
                   WHERE phone_hash=? AND used=0 ORDER BY created_at DESC LIMIT 1""",
                (phone_hash,),
            ).fetchone()
            if row is None or row["expires_at"] < time.time() or row["attempts"] >= 5:
                return False
            valid = hmac.compare_digest(str(row["code_hash"]), code_hash)
            db.execute(
                "UPDATE otp_challenges SET attempts=attempts+1, used=? WHERE id=?",
                (1 if valid else 0, row["id"]),
            )
            db.commit()
            return valid

    async def get_or_create_user(self, phone_hash: str, phone_masked: str) -> Dict[str, Any]:
        async with self._lock:
            db = self._conn()
            row = db.execute("SELECT * FROM users WHERE phone_hash=?", (phone_hash,)).fetchone()
            if row is None:
                user_id = str(uuid.uuid4())
                db.execute(
                    "INSERT INTO users(id, phone_hash, phone_masked, role, created_at) VALUES (?, ?, ?, 'patient', ?)",
                    (user_id, phone_hash, phone_masked, time.time()),
                )
                db.commit()
                row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            return dict(row)

    async def set_user_role(self, user_id: str, role: str) -> None:
        if role not in {"patient", "clinician"}:
            raise CareError("invalid_role", "账户角色无效。")
        async with self._lock:
            db = self._conn()
            db.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
            db.commit()

    async def create_access_grant(
        self, owner_id: str, journey_id: str, code_hash: str, expires_at: float
    ) -> Dict[str, Any]:
        item = {
            "id": str(uuid.uuid4()), "journey_id": journey_id, "owner_id": owner_id,
            "expires_at": expires_at, "used_at": None, "created_at": time.time(),
        }
        async with self._lock:
            db = self._conn()
            db.execute(
                """INSERT INTO care_access_grants
                   (id, journey_id, owner_id, code_hash, expires_at, used_at, created_at)
                   VALUES (?, ?, ?, ?, ?, NULL, ?)""",
                (item["id"], journey_id, owner_id, code_hash, expires_at, item["created_at"]),
            )
            db.commit()
        return item

    async def redeem_access_grant(self, clinician_id: str, code_hash: str) -> Dict[str, Any]:
        async with self._lock:
            db = self._conn()
            row = db.execute(
                "SELECT * FROM care_access_grants WHERE code_hash=?", (code_hash,)
            ).fetchone()
            if row is None:
                raise CareError("access_grant_not_found", "授权码无效。", 404)
            if row["used_at"] is not None:
                raise CareError("access_grant_used", "授权码已被使用。", 409)
            if float(row["expires_at"]) < time.time():
                raise CareError("access_grant_expired", "授权码已过期，请患者重新生成。", 410)
            if str(row["owner_id"]) == clinician_id:
                raise CareError("access_grant_self_redeem", "不能领取自己的病例授权。", 409)
            now = time.time()
            db.execute("UPDATE care_access_grants SET used_at=? WHERE id=?", (now, row["id"]))
            existing = db.execute(
                """SELECT * FROM care_team_links WHERE journey_id=? AND patient_id=?
                   AND clinician_id=? AND status='active' ORDER BY created_at DESC LIMIT 1""",
                (row["journey_id"], row["owner_id"], clinician_id),
            ).fetchone()
            if existing is None:
                link_id = str(uuid.uuid4())
                db.execute(
                    """INSERT INTO care_team_links
                       (id, journey_id, patient_id, clinician_id, status, created_at, revoked_at)
                       VALUES (?, ?, ?, ?, 'active', ?, NULL)""",
                    (link_id, row["journey_id"], row["owner_id"], clinician_id, now),
                )
                existing = db.execute("SELECT * FROM care_team_links WHERE id=?", (link_id,)).fetchone()
            db.commit()
            return dict(existing)

    async def revoke_care_team_link(self, patient_id: str, link_id: str) -> bool:
        async with self._lock:
            db = self._conn()
            cursor = db.execute(
                """UPDATE care_team_links SET status='revoked', revoked_at=?
                   WHERE id=? AND patient_id=? AND status='active'""",
                (time.time(), link_id, patient_id),
            )
            db.commit()
            return cursor.rowcount > 0

    async def list_clinician_links(self, clinician_id: str) -> List[Dict[str, Any]]:
        async with self._lock:
            rows = self._conn().execute(
                """SELECT * FROM care_team_links WHERE clinician_id=? AND status='active'
                   ORDER BY created_at DESC""", (clinician_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    async def get_clinician_link(self, clinician_id: str, journey_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            row = self._conn().execute(
                """SELECT * FROM care_team_links WHERE clinician_id=? AND journey_id=?
                   AND status='active' ORDER BY created_at DESC LIMIT 1""",
                (clinician_id, journey_id),
            ).fetchone()
            return dict(row) if row is not None else None

    async def list_patient_links(self, patient_id: str, journey_id: str = "") -> List[Dict[str, Any]]:
        async with self._lock:
            query = "SELECT * FROM care_team_links WHERE patient_id=?"
            params: Tuple[Any, ...] = (patient_id,)
            if journey_id:
                query += " AND journey_id=?"
                params = (patient_id, journey_id)
            query += " ORDER BY created_at DESC"
            rows = self._conn().execute(query, params).fetchall()
            return [dict(row) for row in rows]

    async def create_session(self, user_id: str, ttl_seconds: int = 7 * 86400) -> Tuple[str, str]:
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        async with self._lock:
            db = self._conn()
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                (hashlib.sha256(token.encode()).hexdigest(), user_id, csrf, time.time() + ttl_seconds, time.time()),
            )
            db.commit()
        return token, csrf

    async def get_session(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        async with self._lock:
            db = self._conn()
            row = db.execute(
                """SELECT sessions.*, users.phone_masked, users.role FROM sessions
                   JOIN users ON users.id=sessions.user_id
                   WHERE token_hash=? AND expires_at>?""",
                (token_hash, time.time()),
            ).fetchone()
            return dict(row) if row is not None else None

    async def delete_session(self, token: str) -> None:
        async with self._lock:
            db = self._conn()
            db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))
            db.commit()

    async def list_journeys(self, owner_id: str) -> List[Dict[str, Any]]:
        async with self._lock:
            rows = self._conn().execute(
                "SELECT payload_json FROM journeys WHERE owner_id=? ORDER BY updated_at DESC",
                (owner_id,),
            ).fetchall()
            return [json.loads(row["payload_json"]) for row in rows]

    async def get_journey(self, owner_id: str, journey_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            row = self._conn().execute(
                "SELECT payload_json FROM journeys WHERE owner_id=? AND id=?",
                (owner_id, journey_id),
            ).fetchone()
            return json.loads(row["payload_json"]) if row is not None else None

    async def save_journey(self, owner_id: str, journey: Dict[str, Any]) -> None:
        journey["updated_at"] = utc_now()
        async with self._lock:
            db = self._conn()
            db.execute(
                """INSERT INTO journeys(id, owner_id, payload_json, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET payload_json=excluded.payload_json,
                     updated_at=excluded.updated_at""",
                (journey["id"], owner_id, _json(journey), time.time()),
            )
            db.commit()

    async def delete_user_data(self, owner_id: str) -> None:
        async with self._lock:
            db = self._conn()
            db.execute("DELETE FROM users WHERE id=?", (owner_id,))
            db.commit()

    async def save_job(self, owner_id: str, job: Dict[str, Any]) -> None:
        now = time.time()
        async with self._lock:
            db = self._conn()
            db.execute(
                """INSERT INTO jobs(id, owner_id, kind, status, payload_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET status=excluded.status,
                     payload_json=excluded.payload_json, updated_at=excluded.updated_at""",
                (job["id"], owner_id, job["kind"], job["status"], _json(job), now, now),
            )
            db.commit()

    async def get_job(self, owner_id: str, job_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            row = self._conn().execute(
                "SELECT payload_json FROM jobs WHERE owner_id=? AND id=?",
                (owner_id, job_id),
            ).fetchone()
            return json.loads(row["payload_json"]) if row is not None else None

    async def save_import(self, owner_id: str, item: Dict[str, Any]) -> None:
        now = time.time()
        async with self._lock:
            db = self._conn()
            db.execute(
                "INSERT INTO record_imports VALUES (?, ?, ?, ?, ?, ?)",
                (item["id"], owner_id, item["status"], _json(item), now, now),
            )
            db.commit()

    async def get_import(self, owner_id: str, import_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            row = self._conn().execute(
                "SELECT payload_json FROM record_imports WHERE owner_id=? AND id=?",
                (owner_id, import_id),
            ).fetchone()
            return json.loads(row["payload_json"]) if row is not None else None

    async def add_push_subscription(self, owner_id: str, subscription: Dict[str, Any]) -> None:
        endpoint = str(subscription.get("endpoint") or "")
        if not endpoint.startswith("https://"):
            raise CareError("invalid_push_subscription", "推送订阅地址无效。")
        endpoint_hash = hashlib.sha256(endpoint.encode()).hexdigest()
        async with self._lock:
            db = self._conn()
            db.execute(
                """INSERT INTO push_subscriptions VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(owner_id, endpoint_hash) DO UPDATE SET payload_json=excluded.payload_json""",
                (str(uuid.uuid4()), owner_id, endpoint_hash, _json(subscription), time.time()),
            )
            db.commit()

    async def list_push_subscriptions(self, owner_id: str) -> List[Dict[str, Any]]:
        async with self._lock:
            rows = self._conn().execute(
                "SELECT payload_json FROM push_subscriptions WHERE owner_id=?",
                (owner_id,),
            ).fetchall()
            return [json.loads(row["payload_json"]) for row in rows]

    async def list_all_journeys(self) -> List[Tuple[str, Dict[str, Any]]]:
        async with self._lock:
            rows = self._conn().execute("SELECT owner_id, payload_json FROM journeys").fetchall()
            return [(str(row["owner_id"]), json.loads(row["payload_json"])) for row in rows]

    async def remove_push_subscription(self, owner_id: str, endpoint: str) -> None:
        async with self._lock:
            db = self._conn()
            db.execute(
                "DELETE FROM push_subscriptions WHERE owner_id=? AND endpoint_hash=?",
                (owner_id, hashlib.sha256(endpoint.encode()).hexdigest()),
            )
            db.commit()

    async def audit(
        self,
        owner_id: Optional[str],
        action: str,
        object_type: str,
        object_id: str = "",
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        async with self._lock:
            db = self._conn()
            db.execute(
                "INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), owner_id, action, object_type, object_id, _json(detail or {}), time.time()),
            )
            db.commit()


class PostgresCareRepository:
    """Production repository backed by PostgreSQL through asyncpg."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._pool: Any = None

    @staticmethod
    def _uuid(value: str) -> uuid.UUID:
        return uuid.UUID(str(value))

    async def start(self) -> None:
        try:
            import asyncpg
        except ImportError as exc:
            raise RuntimeError("asyncpg is required when CARE_DATABASE_URL uses PostgreSQL") from exc
        self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5, command_timeout=15)
        schema = (ROOT / "db" / "postgres.sql").read_text(encoding="utf-8")
        async with self._pool.acquire() as connection:
            await connection.execute(schema)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _required_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("Care repository is not started")
        return self._pool

    async def create_otp(self, *, phone_hash: str, client_hash: str, code_hash: str, expires_at: float) -> None:
        pool = self._required_pool()
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=15)
        async with pool.acquire() as connection, connection.transaction():
            phone_count = await connection.fetchval("SELECT COUNT(*) FROM otp_challenges WHERE phone_hash=$1 AND created_at >= $2", phone_hash, cutoff)
            client_count = await connection.fetchval("SELECT COUNT(*) FROM otp_challenges WHERE client_hash=$1 AND created_at >= $2", client_hash, cutoff)
            if phone_count >= 5 or client_count >= 12:
                raise CareError("otp_rate_limited", "验证码请求过于频繁，请稍后再试。", 429)
            await connection.execute(
                """INSERT INTO otp_challenges(id, phone_hash, client_hash, code_hash, expires_at)
                   VALUES($1, $2, $3, $4, $5)""",
                uuid.uuid4(), phone_hash, client_hash, code_hash, datetime.fromtimestamp(expires_at, timezone.utc),
            )

    async def consume_otp(self, phone_hash: str, code_hash: str) -> bool:
        pool = self._required_pool()
        async with pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """SELECT * FROM otp_challenges WHERE phone_hash=$1 AND used=FALSE
                   ORDER BY created_at DESC LIMIT 1 FOR UPDATE""", phone_hash
            )
            if row is None or row["expires_at"] < datetime.now(timezone.utc) or row["attempts"] >= 5:
                return False
            valid = hmac.compare_digest(str(row["code_hash"]), code_hash)
            await connection.execute("UPDATE otp_challenges SET attempts=attempts+1, used=$1 WHERE id=$2", valid, row["id"])
            return valid

    async def get_or_create_user(self, phone_hash: str, phone_masked: str) -> Dict[str, Any]:
        pool = self._required_pool()
        row = await pool.fetchrow("SELECT * FROM users WHERE phone_hash=$1", phone_hash)
        if row is None:
            await pool.execute(
                """INSERT INTO users(id, phone_hash, phone_masked) VALUES($1, $2, $3)
                   ON CONFLICT(phone_hash) DO NOTHING""", uuid.uuid4(), phone_hash, phone_masked
            )
            row = await pool.fetchrow("SELECT * FROM users WHERE phone_hash=$1", phone_hash)
        return {"id": str(row["id"]), "phone_hash": row["phone_hash"], "phone_masked": row["phone_masked"], "role": row["role"]}

    async def set_user_role(self, user_id: str, role: str) -> None:
        if role not in {"patient", "clinician"}:
            raise CareError("invalid_role", "账户角色无效。")
        await self._required_pool().execute(
            "UPDATE users SET role=$1 WHERE id=$2", role, self._uuid(user_id)
        )

    async def create_access_grant(
        self, owner_id: str, journey_id: str, code_hash: str, expires_at: float
    ) -> Dict[str, Any]:
        item = {
            "id": str(uuid.uuid4()), "journey_id": journey_id, "owner_id": owner_id,
            "expires_at": expires_at, "used_at": None, "created_at": time.time(),
        }
        await self._required_pool().execute(
            """INSERT INTO care_access_grants(id, journey_id, owner_id, code_hash, expires_at)
               VALUES($1, $2, $3, $4, $5)""",
            self._uuid(item["id"]), self._uuid(journey_id), self._uuid(owner_id), code_hash,
            datetime.fromtimestamp(expires_at, timezone.utc),
        )
        return item

    async def redeem_access_grant(self, clinician_id: str, code_hash: str) -> Dict[str, Any]:
        pool = self._required_pool()
        async with pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT * FROM care_access_grants WHERE code_hash=$1 FOR UPDATE", code_hash
            )
            if row is None:
                raise CareError("access_grant_not_found", "授权码无效。", 404)
            if row["used_at"] is not None:
                raise CareError("access_grant_used", "授权码已被使用。", 409)
            if row["expires_at"] < datetime.now(timezone.utc):
                raise CareError("access_grant_expired", "授权码已过期，请患者重新生成。", 410)
            if str(row["owner_id"]) == clinician_id:
                raise CareError("access_grant_self_redeem", "不能领取自己的病例授权。", 409)
            await connection.execute(
                "UPDATE care_access_grants SET used_at=NOW() WHERE id=$1", row["id"]
            )
            existing = await connection.fetchrow(
                """SELECT * FROM care_team_links WHERE journey_id=$1 AND patient_id=$2
                   AND clinician_id=$3 AND status='active' ORDER BY created_at DESC LIMIT 1""",
                row["journey_id"], row["owner_id"], self._uuid(clinician_id),
            )
            if existing is None:
                link_id = uuid.uuid4()
                await connection.execute(
                    """INSERT INTO care_team_links(id, journey_id, patient_id, clinician_id)
                       VALUES($1, $2, $3, $4)""",
                    link_id, row["journey_id"], row["owner_id"], self._uuid(clinician_id),
                )
                existing = await connection.fetchrow("SELECT * FROM care_team_links WHERE id=$1", link_id)
            return {key: str(value) if isinstance(value, uuid.UUID) else value.isoformat() if isinstance(value, datetime) else value for key, value in dict(existing).items()}

    async def revoke_care_team_link(self, patient_id: str, link_id: str) -> bool:
        status = await self._required_pool().execute(
            """UPDATE care_team_links SET status='revoked', revoked_at=NOW()
               WHERE id=$1 AND patient_id=$2 AND status='active'""",
            self._uuid(link_id), self._uuid(patient_id),
        )
        return status.endswith(" 1")

    async def list_clinician_links(self, clinician_id: str) -> List[Dict[str, Any]]:
        rows = await self._required_pool().fetch(
            """SELECT * FROM care_team_links WHERE clinician_id=$1 AND status='active'
               ORDER BY created_at DESC""", self._uuid(clinician_id)
        )
        return [{key: str(value) if isinstance(value, uuid.UUID) else value.isoformat() if isinstance(value, datetime) else value for key, value in dict(row).items()} for row in rows]

    async def get_clinician_link(self, clinician_id: str, journey_id: str) -> Optional[Dict[str, Any]]:
        row = await self._required_pool().fetchrow(
            """SELECT * FROM care_team_links WHERE clinician_id=$1 AND journey_id=$2
               AND status='active' ORDER BY created_at DESC LIMIT 1""",
            self._uuid(clinician_id), self._uuid(journey_id),
        )
        if row is None:
            return None
        return {key: str(value) if isinstance(value, uuid.UUID) else value.isoformat() if isinstance(value, datetime) else value for key, value in dict(row).items()}

    async def list_patient_links(self, patient_id: str, journey_id: str = "") -> List[Dict[str, Any]]:
        if journey_id:
            rows = await self._required_pool().fetch(
                "SELECT * FROM care_team_links WHERE patient_id=$1 AND journey_id=$2 ORDER BY created_at DESC",
                self._uuid(patient_id), self._uuid(journey_id),
            )
        else:
            rows = await self._required_pool().fetch(
                "SELECT * FROM care_team_links WHERE patient_id=$1 ORDER BY created_at DESC",
                self._uuid(patient_id),
            )
        return [{key: str(value) if isinstance(value, uuid.UUID) else value.isoformat() if isinstance(value, datetime) else value for key, value in dict(row).items()} for row in rows]

    async def create_session(self, user_id: str, ttl_seconds: int = 7 * 86400) -> Tuple[str, str]:
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(24)
        await self._required_pool().execute(
            "INSERT INTO sessions(token_hash, user_id, csrf_token, expires_at) VALUES($1, $2, $3, $4)",
            hashlib.sha256(token.encode()).hexdigest(), self._uuid(user_id), csrf, datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
        )
        return token, csrf

    async def get_session(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        row = await self._required_pool().fetchrow(
            """SELECT sessions.*, users.phone_masked, users.role FROM sessions JOIN users ON users.id=sessions.user_id
               WHERE token_hash=$1 AND expires_at>NOW()""", hashlib.sha256(token.encode()).hexdigest()
        )
        if row is None:
            return None
        return {"token_hash": row["token_hash"], "user_id": str(row["user_id"]), "csrf_token": row["csrf_token"], "phone_masked": row["phone_masked"], "role": row["role"]}

    async def delete_session(self, token: str) -> None:
        await self._required_pool().execute("DELETE FROM sessions WHERE token_hash=$1", hashlib.sha256(token.encode()).hexdigest())

    async def list_journeys(self, owner_id: str) -> List[Dict[str, Any]]:
        rows = await self._required_pool().fetch("SELECT payload FROM journeys WHERE owner_id=$1 ORDER BY updated_at DESC", self._uuid(owner_id))
        return [_copy(row["payload"]) for row in rows]

    async def get_journey(self, owner_id: str, journey_id: str) -> Optional[Dict[str, Any]]:
        row = await self._required_pool().fetchrow("SELECT payload FROM journeys WHERE owner_id=$1 AND id=$2", self._uuid(owner_id), self._uuid(journey_id))
        return _copy(row["payload"]) if row is not None else None

    async def save_journey(self, owner_id: str, journey: Dict[str, Any]) -> None:
        journey["updated_at"] = utc_now()
        await self._required_pool().execute(
            """INSERT INTO journeys(id, owner_id, payload, updated_at) VALUES($1, $2, $3::jsonb, NOW())
               ON CONFLICT(id) DO UPDATE SET payload=EXCLUDED.payload, updated_at=NOW()""",
            self._uuid(journey["id"]), self._uuid(owner_id), _json(journey),
        )

    async def delete_user_data(self, owner_id: str) -> None:
        await self._required_pool().execute("DELETE FROM users WHERE id=$1", self._uuid(owner_id))

    async def save_job(self, owner_id: str, job: Dict[str, Any]) -> None:
        await self._required_pool().execute(
            """INSERT INTO jobs(id, owner_id, kind, status, payload) VALUES($1, $2, $3, $4, $5::jsonb)
               ON CONFLICT(id) DO UPDATE SET status=EXCLUDED.status, payload=EXCLUDED.payload, updated_at=NOW()""",
            self._uuid(job["id"]), self._uuid(owner_id), job["kind"], job["status"], _json(job),
        )

    async def get_job(self, owner_id: str, job_id: str) -> Optional[Dict[str, Any]]:
        row = await self._required_pool().fetchrow("SELECT payload FROM jobs WHERE owner_id=$1 AND id=$2", self._uuid(owner_id), self._uuid(job_id))
        return _copy(row["payload"]) if row is not None else None

    async def save_import(self, owner_id: str, item: Dict[str, Any]) -> None:
        await self._required_pool().execute(
            "INSERT INTO record_imports(id, owner_id, status, payload) VALUES($1, $2, $3, $4::jsonb)",
            self._uuid(item["id"]), self._uuid(owner_id), item["status"], _json(item),
        )

    async def get_import(self, owner_id: str, import_id: str) -> Optional[Dict[str, Any]]:
        row = await self._required_pool().fetchrow("SELECT payload FROM record_imports WHERE owner_id=$1 AND id=$2", self._uuid(owner_id), self._uuid(import_id))
        return _copy(row["payload"]) if row is not None else None

    async def add_push_subscription(self, owner_id: str, subscription: Dict[str, Any]) -> None:
        endpoint = str(subscription.get("endpoint") or "")
        if not endpoint.startswith("https://"):
            raise CareError("invalid_push_subscription", "推送订阅地址无效。")
        endpoint_hash = hashlib.sha256(endpoint.encode()).hexdigest()
        await self._required_pool().execute(
            """INSERT INTO push_subscriptions(id, owner_id, endpoint_hash, payload) VALUES($1, $2, $3, $4::jsonb)
               ON CONFLICT(owner_id, endpoint_hash) DO UPDATE SET payload=EXCLUDED.payload""",
            uuid.uuid4(), self._uuid(owner_id), endpoint_hash, _json(subscription),
        )

    async def list_push_subscriptions(self, owner_id: str) -> List[Dict[str, Any]]:
        rows = await self._required_pool().fetch("SELECT payload FROM push_subscriptions WHERE owner_id=$1", self._uuid(owner_id))
        return [_copy(row["payload"]) for row in rows]

    async def list_all_journeys(self) -> List[Tuple[str, Dict[str, Any]]]:
        rows = await self._required_pool().fetch("SELECT owner_id, payload FROM journeys")
        return [(str(row["owner_id"]), _copy(row["payload"])) for row in rows]

    async def remove_push_subscription(self, owner_id: str, endpoint: str) -> None:
        await self._required_pool().execute(
            "DELETE FROM push_subscriptions WHERE owner_id=$1 AND endpoint_hash=$2",
            self._uuid(owner_id), hashlib.sha256(endpoint.encode()).hexdigest(),
        )

    async def audit(self, owner_id: Optional[str], action: str, object_type: str, object_id: str = "", detail: Optional[Dict[str, Any]] = None) -> None:
        await self._required_pool().execute(
            "INSERT INTO audit_events(id, owner_id, action, object_type, object_id, detail) VALUES($1, $2, $3, $4, $5, $6::jsonb)",
            uuid.uuid4(), self._uuid(owner_id) if owner_id else None, action, object_type, object_id, _json(detail or {}),
        )


def create_repository() -> Any:
    database_url = os.getenv("CARE_DATABASE_URL", os.getenv("DATABASE_URL", "")).strip()
    if database_url.startswith(("postgres://", "postgresql://")):
        return PostgresCareRepository(database_url)
    return SQLiteCareRepository()


class LocalDocumentStore:
    """Development-only upload storage; production must configure S3."""

    def __init__(self, root: Optional[Path] = None, public_deployment: bool = False) -> None:
        self.root = Path(root or os.getenv("CARE_UPLOAD_DIR", DEFAULT_DATA_DIR / "uploads"))
        self.public_deployment = public_deployment

    async def put(self, owner_id: str, filename: str, content_type: str, data: bytes) -> Dict[str, Any]:
        if self.public_deployment:
            raise CareError("object_storage_unconfigured", "生产环境未配置加密对象存储，上传入口已关闭。", 503)
        if content_type not in ALLOWED_UPLOAD_TYPES:
            raise CareError("unsupported_file_type", "仅支持 PDF、JPG、PNG 或 WebP 文件。", 415)
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise CareError("invalid_file_size", "文件不能为空且不得超过 10MB。", 413)
        suffix = {"application/pdf": ".pdf", "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[content_type]
        key = f"{owner_id}/{uuid.uuid4()}{suffix}"
        target = (self.root / key).resolve()
        base = self.root.resolve()
        if base not in target.parents:
            raise CareError("invalid_upload_path", "上传路径无效。")
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)
        return {
            "storage": "development-local",
            "object_key": key,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "content_type": content_type,
            "original_name": Path(filename).name[:120],
        }

    async def delete_owner(self, owner_id: str) -> None:
        target = (self.root / owner_id).resolve()
        base = self.root.resolve()
        if target != base and base in target.parents and target.exists():
            await asyncio.to_thread(shutil.rmtree, target)


class S3DocumentStore:
    """S3-compatible production storage with server-side encryption."""

    def __init__(self) -> None:
        self.bucket = os.getenv("CARE_S3_BUCKET", "").strip()
        if not self.bucket:
            raise RuntimeError("CARE_S3_BUCKET is required")
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("boto3 is required for S3 document storage") from exc
        self._client = boto3.client(
            "s3",
            endpoint_url=os.getenv("CARE_S3_ENDPOINT") or None,
            region_name=os.getenv("CARE_S3_REGION") or None,
            aws_access_key_id=os.getenv("CARE_S3_ACCESS_KEY_ID") or None,
            aws_secret_access_key=os.getenv("CARE_S3_SECRET_ACCESS_KEY") or None,
        )

    async def put(self, owner_id: str, filename: str, content_type: str, data: bytes) -> Dict[str, Any]:
        if content_type not in ALLOWED_UPLOAD_TYPES:
            raise CareError("unsupported_file_type", "仅支持 PDF、JPG、PNG 或 WebP 文件。", 415)
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise CareError("invalid_file_size", "文件不能为空且不得超过 10MB。", 413)
        suffix = {"application/pdf": ".pdf", "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[content_type]
        key = f"health-documents/{owner_id}/{uuid.uuid4()}{suffix}"
        digest = hashlib.sha256(data).hexdigest()
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            ServerSideEncryption="AES256",
            Metadata={"sha256": digest},
        )
        return {
            "storage": "s3-encrypted",
            "object_key": key,
            "sha256": digest,
            "size": len(data),
            "content_type": content_type,
            "original_name": Path(filename).name[:120],
        }

    async def delete_owner(self, owner_id: str) -> None:
        prefix = f"health-documents/{owner_id}/"
        response = await asyncio.to_thread(self._client.list_objects_v2, Bucket=self.bucket, Prefix=prefix)
        objects = [{"Key": item["Key"]} for item in response.get("Contents", [])]
        if objects:
            await asyncio.to_thread(self._client.delete_objects, Bucket=self.bucket, Delete={"Objects": objects, "Quiet": True})


def _sample_patient_profile() -> Dict[str, Any]:
    """Return a complete, explicitly fictional identity for the sandbox journey."""
    return sample_patient_profile()
    return {
        "name": "周予安",
        "sex": "男",
        "age": 46,
        "date_of_birth": "1980-04-18",
        "phone": "138 0000 2468",
        "address": "北京市朝阳区望京街道（虚构）",
        "hospital_record_no": "SBX-20260902-001",
        "emergency_contact": "林女士 · 139 0000 1357（虚构）",
        "is_fictional": True,
        "notice": "以上身份信息均为产品沙箱虚构数据，不对应任何真实患者。",
    }


def _sample_records() -> List[Dict[str, Any]]:
    return product_sample_records()
    return [
        {
            "id": "record-symptoms",
            "kind": "symptom",
            "title": "当前症状",
            "items": ["间断咯血", "活动后呼吸困难", "关节痛"],
            "observed_at": "2026-08-31T08:40:00+08:00",
            "source": {"type": "sandbox", "label": "沙箱医院 · 门诊病历", "locator": "主诉"},
            "verification_status": "imported",
            "abnormal": True,
        },
        {
            "id": "record-urinalysis",
            "kind": "laboratory",
            "title": "尿常规",
            "items": ["尿红细胞 50 个/HPF（参考值 0–3）", "尿蛋白阳性"],
            "observed_at": "2026-08-31T10:12:00+08:00",
            "source": {"type": "sandbox", "label": "沙箱医院 · 检验报告", "locator": "尿常规第 3、7 项"},
            "verification_status": "imported",
            "abnormal": True,
        },
        {
            "id": "record-renal",
            "kind": "laboratory",
            "title": "肾功能",
            "items": ["肌酐 220 μmol/L（参考值 44–133）"],
            "observed_at": "2026-08-31T10:15:00+08:00",
            "source": {"type": "sandbox", "label": "沙箱医院 · 生化检验", "locator": "肾功能第 2 项"},
            "verification_status": "imported",
            "abnormal": True,
        },
        {
            "id": "record-ct",
            "kind": "imaging",
            "title": "胸部 CT",
            "items": ["影像提示弥漫性肺泡出血"],
            "observed_at": "2026-08-31T13:30:00+08:00",
            "source": {"type": "sandbox", "label": "沙箱医院 · 影像报告", "locator": "影像结论"},
            "verification_status": "imported",
            "abnormal": True,
        },
        {
            "id": "record-anca",
            "kind": "laboratory",
            "title": "免疫学检查",
            "items": ["MPO-ANCA 阳性"],
            "observed_at": "2026-09-01T09:05:00+08:00",
            "source": {"type": "sandbox", "label": "沙箱医院 · 抗体检测", "locator": "ANCA 谱"},
            "verification_status": "imported",
            "abnormal": True,
        },
        {
            "id": "record-infection",
            "kind": "laboratory",
            "title": "感染筛查",
            "items": ["血培养未检出细菌", "痰病原学筛查阴性"],
            "observed_at": "2026-09-01T11:20:00+08:00",
            "source": {"type": "sandbox", "label": "完整虚构沙箱病例", "locator": "感染筛查"},
            "verification_status": "imported",
            "abnormal": False,
            "scenario_note": "该阴性结果仅用于演示反证如何影响候选，不计入 7 例回放指标。",
        },
    ]


def new_journey(owner_id: str) -> Dict[str, Any]:
    now = utc_now()
    return hydrate_journey_v3(hydrate_journey({
        "schema_version": "care-journey.v3",
        "id": str(uuid.uuid4()),
        "owner_id": owner_id,
        "title": "肺与肾的多项异常需要一起理解",
        "status": "active",
        "current_stage": "consultation",
        "created_at": now,
        "updated_at": now,
        "patient_profile": sample_patient_profile(),
        "clinical_history": sample_clinical_history(confirmed=False),
        "raw_case_document": sample_raw_case_document(),
        "consultation": {"messages": [], "quick_questions": quick_questions()},
        "synced_batches": [],
        "assessment_versions": [],
        "treatment_reference": treatment_reference(),
        "hospital_connection": None,
        "triage": {"status": "pending", "danger_signs": [], "checked_at": None},
        "records": [],
        "evidence": [],
        "assessment": None,
        "appointment_plan": None,
        "doctor_plan": None,
        "followups": [],
        "medications": [],
        "reminders": [],
        "timeline": [{"id": str(uuid.uuid4()), "type": "journey_created", "title": "健康事件已建立", "detail": "从问诊和安全分流开始，等待用户确认病史。", "source": "user", "created_at": now}],
        "consents": {"hospital": False, "ai_analysis": False, "push": False},
    }))
    return {
        "schema_version": "care-journey.v1",
        "id": str(uuid.uuid4()),
        "owner_id": owner_id,
        "title": "肺与肾的多项异常需要一起理解",
        "status": "active",
        "current_stage": "connect_records",
        "created_at": now,
        "updated_at": now,
        "patient_profile": _sample_patient_profile(),
        "hospital_connection": None,
        "triage": {"status": "pending", "danger_signs": [], "checked_at": None},
        "records": [],
        "evidence": [],
        "assessment": None,
        "appointment_plan": None,
        "doctor_plan": None,
        "followups": [],
        "medications": [],
        "reminders": [],
        "timeline": [
            {
                "id": str(uuid.uuid4()),
                "type": "journey_created",
                "title": "发现需要持续跟进的异常",
                "detail": "建立健康事件，等待用户授权获取病历。",
                "source": "user",
                "created_at": now,
            }
        ],
        "consents": {"hospital": False, "ai_analysis": False, "push": False},
    }


def _build_evidence(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    record_ids = {item.get("id") for item in records}
    batches = [key for key, batch in sample_record_batches().items() if any(record.get("id") in record_ids for record in batch["records"])]
    return product_build_evidence(records, batches)
    evidence: List[Dict[str, Any]] = []
    for record in records:
        role = "contradicting" if record["id"] == "record-infection" else "supporting"
        for index, item in enumerate(record.get("items") or []):
            evidence.append(
                {
                    "id": f"evidence-{record['id']}-{index}",
                    "label": item,
                    "role": role,
                    "record_id": record["id"],
                    "source": _copy(record["source"]),
                    "observed_at": record["observed_at"],
                    "verification_status": record["verification_status"],
                    "reason": (
                        "阴性病原学结果降低感染方向，但不能单独排除所有感染。"
                        if role == "contradicting"
                        else "已进入候选方向的证据覆盖检查。"
                    ),
                }
            )
    evidence.extend(
        [
            {
                "id": "gap-biopsy",
                "label": "肾活检或其他组织学依据",
                "role": "unresolved",
                "record_id": None,
                "source": {"type": "system", "label": "候选确证条件", "locator": "信息缺口"},
                "observed_at": None,
                "verification_status": "unresolved",
                "reason": "影响医生最终确诊和治疗选择。",
            },
            {
                "id": "gap-full-infection",
                "label": "是否已由医生充分排除感染",
                "role": "unresolved",
                "record_id": None,
                "source": {"type": "system", "label": "安全边界", "locator": "信息缺口"},
                "observed_at": None,
                "verification_status": "unresolved",
                "reason": "部分阴性检查不等于医生已完成感染排除。",
            },
        ]
    )
    return evidence


def build_assessment(journey: Dict[str, Any]) -> Dict[str, Any]:
    return product_assessment_version(hydrate_journey(journey))
    return {
        "schema_version": "assessment.v1",
        "authority": "decision_support",
        "label": "AI 辅助判断",
        "status": "completed",
        "created_at": utc_now(),
        "urgency": {
            "level": "urgent_specialist",
            "label": "建议尽快由风湿免疫科与肾内科联合评估",
            "reason": "肺泡出血表现与肾功能损害同时存在，可能涉及器官威胁。",
        },
        "candidate_history": [
            {
                "stage": "症状进入",
                "value": "建立需要区分的问题空间",
                "candidates": [
                    {"name": "肺部感染方向", "strength": "中", "trend": "新出现", "reason": "咯血与呼吸困难可见于感染。"},
                    {"name": "局部肺血管/结构异常", "strength": "中", "trend": "新出现", "reason": "单看咯血仍需要排查局部肺部病变。"},
                    {"name": "系统性小血管炎方向", "strength": "低", "trend": "待验证", "reason": "关节痛提示系统性可能，但缺少器官证据。"},
                ],
            },
            {
                "stage": "肺肾异常合并",
                "value": "发现单一肺部问题无法覆盖全部异常",
                "candidates": [
                    {"name": "系统性小血管炎方向", "strength": "中", "trend": "上升", "reason": "血尿、蛋白尿和肌酐升高提示肾小球受累。"},
                    {"name": "肺部感染方向", "strength": "低", "trend": "下降", "reason": "不能单独解释肾小球损害。"},
                    {"name": "局部肺血管/结构异常", "strength": "低", "trend": "下降", "reason": "不能覆盖肾脏异常。"},
                ],
            },
            {
                "stage": "免疫与影像证据进入",
                "value": "形成可验证的跨器官模式",
                "candidates": [
                    {"name": "显微镜下多血管炎方向", "strength": "强", "trend": "上升", "reason": "肺泡出血、肾小球损害与 MPO-ANCA 阳性形成一致模式。"},
                    {"name": "肺部感染方向", "strength": "低", "trend": "下降", "reason": "病原学阴性结果构成反证，但尚不能完全排除感染。"},
                    {"name": "局部肺血管/结构异常", "strength": "低", "trend": "受阻", "reason": "弥漫性肺泡出血不支持局灶结构病变成为统一解释。"},
                ],
            },
        ],
        "evidence_summary": {
            "supporting_count": sum(item["role"] == "supporting" for item in journey["evidence"]),
            "contradicting_count": sum(item["role"] == "contradicting" for item in journey["evidence"]),
            "unresolved_count": sum(item["role"] == "unresolved" for item in journey["evidence"]),
        },
        "leading_direction": {
            "name": "显微镜下多血管炎方向",
            "status": "需要医生进一步确诊",
            "reasoning": "肺泡出血、肾小球损害和 MPO-ANCA 阳性可以形成统一解释；感染筛查阴性降低感染方向，但尚缺医生综合评估与必要确证。",
        },
        "uncertainty": {
            "level": "medium",
            "label": "方向较集中，但不能标记为确诊",
            "gaps": ["肾活检或其他组织学依据", "医生对感染排除程度的判断", "完整病史、查体与器官威胁评估"],
        },
        "care_navigation": {
            "departments": ["风湿免疫科", "肾内科"],
            "materials": ["尿常规与肾功能报告", "胸部 CT 原始影像与报告", "ANCA 检测报告", "当前用药清单"],
            "questions_for_doctor": ["这些肺部和肾脏异常是否可能由同一病因造成？", "还需要哪些检查才能确诊？", "哪些症状出现时应立即急诊？"],
            "exam_discussion_items": [
                {
                    "name": "尿沉渣镜检（红细胞形态与管型）",
                    "priority": "优先向医生确认",
                    "purpose": "判断血尿是否更符合肾小球来源，并寻找红细胞管型。",
                    "resolves": "血尿来源仍未完全确认",
                    "status": "pending_doctor_confirmation",
                },
                {
                    "name": "尿蛋白/肌酐比或 24 小时尿蛋白定量",
                    "priority": "优先向医生确认",
                    "purpose": "量化蛋白尿，帮助医生评估肾脏受累程度。",
                    "resolves": "目前只有尿蛋白阳性，缺少定量",
                    "status": "pending_doctor_confirmation",
                },
                {
                    "name": "复查肾功能、电解质与 eGFR",
                    "priority": "优先向医生确认",
                    "purpose": "观察肌酐变化速度，并评估当前肾功能风险。",
                    "resolves": "只有单次肌酐结果，缺少趋势",
                    "status": "pending_doctor_confirmation",
                },
                {
                    "name": "MPO-ANCA/PR3-ANCA 定量复核、抗 GBM 抗体、补体 C3/C4、ANA 谱",
                    "priority": "由专科医生选择",
                    "purpose": "复核自身抗体并帮助区分其他可造成肺肾综合征的方向。",
                    "resolves": "免疫学鉴别仍不完整",
                    "status": "pending_doctor_confirmation",
                },
                {
                    "name": "血常规、CRP及医生认为必要的感染病原学检查",
                    "priority": "就诊时确认",
                    "purpose": "查看贫血和炎症表现，并继续评估感染是否已充分排除。",
                    "resolves": "现有阴性筛查不能排除全部感染",
                    "status": "pending_doctor_confirmation",
                },
                {
                    "name": "肾活检可行性与必要性评估",
                    "priority": "由肾内科决定",
                    "purpose": "医生结合出血风险和病情判断是否需要组织学确证。",
                    "resolves": "缺少组织学依据",
                    "status": "pending_doctor_confirmation",
                },
            ],
        },
        "limitations": [
            "该结果是 AI 辅助判断，不是医生确诊。",
            "系统未进行体格检查，也没有完整排除感染。",
            "任何检查、治疗和用药决定必须由医生作出。",
        ],
    }


def _stage_after_records(journey: Dict[str, Any]) -> str:
    if journey["triage"]["status"] == "emergency":
        return "emergency"
    if not journey["records"]:
        return "connect_records"
    if journey["triage"]["status"] != "stable":
        return "safety_triage"
    accepted = {"user_confirmed", "hospital_confirmed", "doctor_confirmed"}
    if any(item.get("verification_status") not in accepted for item in journey["records"]):
        return "confirm_records"
    if journey["assessment"] is None:
        return "ready_for_assessment"
    if journey["appointment_plan"] is None:
        return "appointment_preparation"
    if journey["doctor_plan"] is None:
        return "awaiting_doctor"
    if journey["medications"]:
        return "medication_active"
    return "followup_active"


class CareRuntime:
    def __init__(
        self,
        repository: Optional[SQLiteCareRepository] = None,
        document_store: Optional[LocalDocumentStore] = None,
    ) -> None:
        self.public_deployment = str(os.getenv("CARE_PUBLIC_DEPLOYMENT", "false")).lower() in {"1", "true", "yes", "on"}
        self.secret = os.getenv("CARE_AUTH_SECRET", "clinicalens-local-development-secret")
        self.repository = repository or create_repository()
        if document_store is not None:
            self.document_store = document_store
        elif os.getenv("CARE_S3_BUCKET"):
            self.document_store = S3DocumentStore()
        else:
            self.document_store = LocalDocumentStore(public_deployment=self.public_deployment)
        self.cookie_secure = str(os.getenv("CARE_COOKIE_SECURE", "false")).lower() in {"1", "true", "yes", "on"}
        self._reminder_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        await self.repository.start()
        if os.getenv("CARE_VAPID_PRIVATE_KEY") and os.getenv("CARE_VAPID_SUBJECT"):
            self._reminder_task = asyncio.create_task(self._reminder_loop(), name="clinicalens-reminders")

    async def close(self) -> None:
        if self._reminder_task is not None:
            self._reminder_task.cancel()
            try:
                await self._reminder_task
            except asyncio.CancelledError:
                pass
            self._reminder_task = None
        await self.repository.close()

    @staticmethod
    def _due(value: str) -> bool:
        try:
            return datetime.fromisoformat(value).astimezone(timezone.utc) <= datetime.now(timezone.utc)
        except (TypeError, ValueError):
            return False

    async def _reminder_loop(self) -> None:
        interval = max(30, min(300, int(os.getenv("CARE_REMINDER_POLL_SECONDS", "60"))))
        while True:
            try:
                await self.deliver_due_reminders()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Reminder delivery may fail without affecting records or in-app tasks.
                pass
            await asyncio.sleep(interval)

    async def deliver_due_reminders(self) -> int:
        try:
            from pywebpush import webpush
        except ImportError:
            return 0
        private_key = os.getenv("CARE_VAPID_PRIVATE_KEY", "")
        subject = os.getenv("CARE_VAPID_SUBJECT", "")
        if not (private_key and subject):
            return 0
        delivered = 0
        for owner_id, journey in await self.repository.list_all_journeys():
            subscriptions = await self.repository.list_push_subscriptions(owner_id)
            if not subscriptions:
                continue
            changed = False
            for reminder in journey.get("reminders", []):
                if reminder.get("status") != "scheduled" or not self._due(str(reminder.get("scheduled_at") or "")):
                    continue
                medication = next((item for item in journey.get("medications", []) if item.get("id") == reminder.get("medication_id")), None)
                payload = _json(
                    {
                        "title": "ClinicaLens 用药提醒",
                        "body": f"请按医生处方核对并记录：{medication.get('name', '用药任务')}" if medication else "你有一项医生计划相关任务待处理。",
                        "url": "/#aftercare",
                        "tag": f"clinicalens-{reminder['id']}",
                    }
                )
                success = False
                for subscription in subscriptions:
                    try:
                        await asyncio.to_thread(
                            webpush,
                            subscription_info=subscription,
                            data=payload,
                            vapid_private_key=private_key,
                            vapid_claims={"sub": subject},
                        )
                        success = True
                    except Exception:
                        continue
                if success:
                    reminder["status"] = "delivered"
                    reminder["delivered_at"] = utc_now()
                    changed = True
                    delivered += 1
            if changed:
                await self.repository.save_journey(owner_id, journey)
        return delivered

    def _digest(self, value: str) -> str:
        return hmac.new(self.secret.encode(), value.encode(), hashlib.sha256).hexdigest()

    async def request_otp(self, phone: str, client_identity: str) -> Dict[str, Any]:
        phone = re.sub(r"\s+", "", phone or "")
        if not PHONE_RE.fullmatch(phone):
            raise CareError("invalid_phone", "请输入有效的中国大陆手机号。")
        if self.public_deployment and len(self.secret) < 24:
            raise CareError("auth_unavailable", "登录服务尚未安全配置。", 503)
        code = os.getenv("CARE_DEV_OTP_CODE", "246810")
        provider_url = os.getenv("CARE_SMS_PROVIDER_URL", "").strip()
        if self.public_deployment and not provider_url:
            raise CareError("sms_provider_unconfigured", "短信服务尚未配置，未发送验证码。", 503)
        phone_hash = self._digest(phone)
        await self.repository.create_otp(
            phone_hash=phone_hash,
            client_hash=self._digest(client_identity),
            code_hash=self._digest(f"{phone_hash}:{code}"),
            expires_at=time.time() + 5 * 60,
        )
        if provider_url:
            try:
                import httpx

                headers = {}
                if os.getenv("CARE_SMS_PROVIDER_TOKEN"):
                    headers["Authorization"] = f"Bearer {os.getenv('CARE_SMS_PROVIDER_TOKEN')}"
                async with httpx.AsyncClient(timeout=8.0) as client:
                    response = await client.post(provider_url, json={"phone": phone, "code": code}, headers=headers)
                    response.raise_for_status()
            except Exception as exc:
                raise CareError("sms_delivery_failed", "验证码发送失败，请稍后重试。", 503) from exc
        result = {"status": "sent", "expires_in": 300, "delivery": "sms" if provider_url else "development"}
        if not self.public_deployment:
            result["development_code"] = code
        return result

    async def verify_otp(self, phone: str, code: str) -> Dict[str, Any]:
        phone = re.sub(r"\s+", "", phone or "")
        phone_hash = self._digest(phone)
        valid = await self.repository.consume_otp(phone_hash, self._digest(f"{phone_hash}:{code}"))
        if not valid:
            raise CareError("invalid_otp", "验证码无效或已过期。", 401)
        user = await self.repository.get_or_create_user(phone_hash, _mask_phone(phone))
        configured = os.getenv("CARE_DEV_CLINICIAN_PHONES", "13900000000")
        dev_clinicians = {item.strip() for item in configured.split(",") if item.strip()}
        if not self.public_deployment and phone in dev_clinicians and user.get("role") != "clinician":
            await self.repository.set_user_role(user["id"], "clinician")
            user["role"] = "clinician"
        token, csrf = await self.repository.create_session(user["id"])
        existing_journeys = await self.repository.list_journeys(user["id"])
        if user.get("role", "patient") == "patient" and not existing_journeys:
            await self.repository.save_journey(user["id"], new_journey(user["id"]))
        else:
            for existing in existing_journeys:
                await self.repository.save_journey(user["id"], hydrate_journey_v3(hydrate_journey(existing)))
        await self.repository.audit(user["id"], "session_created", "session")
        return {
            "token": token,
            "csrf_token": csrf,
            "user": {"id": user["id"], "phone_masked": user["phone_masked"], "role": user.get("role", "patient")},
        }

    async def connect_hospital(self, owner_id: str, consent: bool) -> Dict[str, Any]:
        if not consent:
            raise CareError("consent_required", "需要先同意病历连接授权。")
        journeys = await self.repository.list_journeys(owner_id)
        journey = hydrate_journey_v3(hydrate_journey(journeys[0])) if journeys else new_journey(owner_id)
        journey["hospital_connection"] = {
            "id": str(uuid.uuid4()),
            "provider": "SandboxHospitalConnector",
            "display_name": "沙箱医院",
            "mode": "sandbox",
            "status": "connected",
            "connected_at": utc_now(),
        }
        journey["consents"]["hospital"] = True
        journey["timeline"].append(
            {"id": str(uuid.uuid4()), "type": "hospital_connected", "title": "已连接沙箱医院", "detail": "仅用于产品流程验证，不是真实医疗机构连接。", "source": "sandbox", "created_at": utc_now()}
        )
        await self.repository.save_journey(owner_id, journey)
        await self.repository.audit(owner_id, "hospital_connected", "journey", journey["id"], {"provider": "sandbox"})
        return journey["hospital_connection"]

    async def sync_records(self, owner_id: str, simulate: str = "success") -> Dict[str, Any]:
        journeys = await self.repository.list_journeys(owner_id)
        if not journeys:
            raise CareError("journey_not_found", "没有进行中的健康事件。", 404)
        journey = hydrate_journey_v3(hydrate_journey(journeys[0]))
        if not journey.get("hospital_connection"):
            raise CareError("hospital_not_connected", "请先授权连接病历来源。", 409)
        if simulate == "timeout":
            journey["hospital_sync_status"] = "failed"
            journey["last_hospital_sync_at"] = utc_now()
            await self.repository.save_journey(owner_id, journey)
            raise CareError("hospital_sync_timeout", "医院接口暂时无响应，可稍后重试或上传报告。", 504)
        records = product_sample_records()
        for record in records:
            record["verification_status"] = "hospital_confirmed"
            record["source"] = {
                "type": "sandbox_hospital", "label": "完整虚构病例 · 沙箱医院",
                "locator": record.get("source", {}).get("locator") or record.get("title"),
            }
        journey["patient_profile"] = _sample_patient_profile()
        journey["clinical_history"] = sample_clinical_history(confirmed=False)
        journey["raw_case_document"] = sample_raw_case_document()
        journey["synced_batches"] = list(sample_record_batches())
        journey["records"] = records
        journey["exam_reports"] = reports_for_batches(journey["synced_batches"])
        journey["evidence"] = product_build_evidence(records, journey["synced_batches"])
        journey["current_stage"] = "safety_triage"
        journey["hospital_sync_status"] = "completed"
        journey["last_hospital_sync_at"] = utc_now()
        journey["timeline"].append(
            {"id": str(uuid.uuid4()), "type": "records_synced", "title": "已从沙箱医院获取 4 批检查", "detail": "医院签名结果可直接进入辅助判断；患者仍可提出结果争议。", "source": "sandbox_hospital", "created_at": utc_now()}
        )
        if self._assessment_is_ready(journey):
            self._append_assessment_version(journey)
        await self.repository.save_journey(owner_id, journey)
        return {"status": "completed", "journey": journey}

    async def public_sample(self, audience: str = "legacy") -> Dict[str, Any]:
        sample = hydrate_journey_v3(public_sample_journey())
        if audience in {"patient", "clinician"}:
            return public_sample_projection(sample, audience)
        return sample

    async def list_journeys(self, owner_id: str) -> List[Dict[str, Any]]:
        journeys = await self.repository.list_journeys(owner_id)
        upgraded: List[Dict[str, Any]] = []
        for journey in journeys:
            hydrated = hydrate_journey_v3(hydrate_journey(journey))
            if hydrated != journey:
                await self.repository.save_journey(owner_id, hydrated)
            upgraded.append(hydrated)
        return upgraded

    async def get_journey(self, owner_id: str, journey_id: str) -> Dict[str, Any]:
        return await self._owned_journey(owner_id, journey_id)

    async def list_patient_journeys(self, owner_id: str) -> List[Dict[str, Any]]:
        return [patient_journey_dto(item) for item in await self.list_journeys(owner_id)]

    async def get_patient_journey(self, owner_id: str, journey_id: str) -> Dict[str, Any]:
        return patient_journey_dto(await self._owned_journey(owner_id, journey_id))

    async def create_care_access_grant(self, owner_id: str, journey_id: str) -> Dict[str, Any]:
        await self._owned_journey(owner_id, journey_id)
        alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
        code = "".join(secrets.choice(alphabet) for _ in range(8))
        expires_at = time.time() + 10 * 60
        item = await self.repository.create_access_grant(
            owner_id, journey_id, self._digest(f"care-access:{code}"), expires_at
        )
        await self.repository.audit(owner_id, "care_access_grant_created", "journey", journey_id)
        return {
            "id": item["id"], "journey_id": journey_id, "code": code,
            "expires_at": datetime.fromtimestamp(expires_at, timezone.utc).isoformat(),
            "expires_in": 600, "single_use": True,
        }

    async def redeem_care_access_grant(self, clinician_id: str, code: str) -> Dict[str, Any]:
        clean = re.sub(r"[^A-Z0-9]", "", str(code or "").upper())
        if len(clean) != 8:
            raise CareError("invalid_access_grant", "请输入 8 位病例授权码。")
        link = await self.repository.redeem_access_grant(
            clinician_id, self._digest(f"care-access:{clean}")
        )
        await self.repository.audit(
            clinician_id, "care_access_grant_redeemed", "care_team_link", str(link["id"]),
            {"journey_id": str(link["journey_id"])},
        )
        return link

    async def revoke_care_team_link(self, patient_id: str, link_id: str) -> Dict[str, Any]:
        revoked = await self.repository.revoke_care_team_link(patient_id, link_id)
        if not revoked:
            raise CareError("care_team_link_not_found", "病例授权不存在或已经撤销。", 404)
        await self.repository.audit(patient_id, "care_team_link_revoked", "care_team_link", link_id)
        return {"id": link_id, "status": "revoked", "revoked_at": utc_now()}

    async def list_patient_care_team_links(self, patient_id: str, journey_id: str = "") -> List[Dict[str, Any]]:
        return await self.repository.list_patient_links(patient_id, journey_id)

    async def list_clinician_journeys(self, clinician_id: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for link in await self.repository.list_clinician_links(clinician_id):
            journey = await self.repository.get_journey(str(link["patient_id"]), str(link["journey_id"]))
            if journey is not None:
                results.append(clinician_journey_dto(hydrate_journey_v3(hydrate_journey(journey)), link))
        return results

    async def get_clinician_journey(self, clinician_id: str, journey_id: str) -> Dict[str, Any]:
        _, journey, link = await self._clinician_journey(clinician_id, journey_id)
        return clinician_journey_dto(journey, link)

    async def _clinician_journey(
        self, clinician_id: str, journey_id: str
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        link = await self.repository.get_clinician_link(clinician_id, journey_id)
        if link is None:
            raise CareError("clinician_access_denied", "尚未获得该病例授权，或患者已撤销授权。", 403)
        patient_id = str(link["patient_id"])
        journey = await self.repository.get_journey(patient_id, journey_id)
        if journey is None:
            raise CareError("journey_not_found", "授权病例不存在。", 404)
        hydrated = hydrate_journey_v3(hydrate_journey(journey))
        if hydrated != journey:
            await self.repository.save_journey(patient_id, hydrated)
        return patient_id, hydrated, link

    async def clinician_exam_recommendations(self, clinician_id: str, journey_id: str) -> List[Dict[str, Any]]:
        _, journey, _ = await self._clinician_journey(clinician_id, journey_id)
        return _copy(journey.get("exam_recommendations") or [])

    async def decide_exam_recommendation(
        self, clinician_id: str, journey_id: str, recommendation_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        patient_id, journey, _ = await self._clinician_journey(clinician_id, journey_id)
        recommendation = next((item for item in journey.get("exam_recommendations", []) if item.get("id") == recommendation_id), None)
        if recommendation is None:
            raise CareError("exam_recommendation_not_found", "检查建议不存在。", 404)
        action = str(payload.get("action") or "")
        rationale = _safe_text(payload.get("rationale"), 500)
        if action not in {"confirmed", "modified", "rejected"}:
            raise CareError("invalid_recommendation_action", "请选择确认、修改或拒绝。")
        if not rationale:
            raise CareError("recommendation_rationale_required", "医生必须填写决策理由。")
        edits = payload.get("edits") if isinstance(payload.get("edits"), dict) else {}
        if action == "modified":
            allowed_edits = {"clinical_question", "items", "priority", "timing", "prerequisites", "risks", "expected_impact"}
            for key, value in edits.items():
                if key not in allowed_edits:
                    continue
                if isinstance(value, list):
                    recommendation[key] = [_safe_text(item, 180) for item in value[:20] if _safe_text(item, 180)]
                else:
                    recommendation[key] = _safe_text(value, 500)
        decision = {
            "id": str(uuid.uuid4()), "recommendation_id": recommendation_id,
            "recommendation_type": "exam", "clinician_id": clinician_id,
            "action": action, "edits": _copy(edits) if action == "modified" else {},
            "rationale": rationale, "decided_at": utc_now(),
        }
        recommendation["status"] = action
        recommendation["decision"] = _copy(decision)
        journey.setdefault("recommendation_decisions", []).append(_copy(decision))
        order = None
        if action in {"confirmed", "modified"}:
            order = {
                "id": str(uuid.uuid4()), "recommendation_id": recommendation_id,
                "items": _copy(recommendation.get("items") or []), "status": "sandbox_ordered",
                "source": {"type": "clinician", "clinician_id": clinician_id, "connector": "SandboxHospitalConnector"},
                "ordered_at": utc_now(), "notice": "沙箱检查医嘱；未向真实医院下单。",
            }
            journey.setdefault("exam_orders", []).append(order)
        await self.repository.save_journey(patient_id, journey)
        await self.repository.audit(clinician_id, "exam_recommendation_decided", "journey", journey_id, {"recommendation_id": recommendation_id, "action": action})
        return {"recommendation": recommendation, "decision": decision, "exam_order": order}

    async def decide_treatment_recommendation(
        self, clinician_id: str, journey_id: str, recommendation_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        patient_id, journey, _ = await self._clinician_journey(clinician_id, journey_id)
        recommendation = next((item for item in journey.get("treatment_recommendations", []) if item.get("id") == recommendation_id), None)
        if recommendation is None:
            raise CareError("treatment_recommendation_not_found", "治疗路径建议不存在。", 404)
        action = str(payload.get("action") or "")
        rationale = _safe_text(payload.get("rationale"), 500)
        if action not in {"confirmed", "modified", "rejected"}:
            raise CareError("invalid_recommendation_action", "请选择确认、修改或拒绝。")
        if not rationale:
            raise CareError("recommendation_rationale_required", "医生必须填写决策理由。")
        edits = payload.get("edits") if isinstance(payload.get("edits"), dict) else {}
        if action == "modified":
            for key in ("goals", "pathways", "prerequisites", "risks", "monitoring"):
                if key in edits and isinstance(edits[key], list):
                    recommendation[key] = _copy(edits[key][:20])
        decision = {
            "id": str(uuid.uuid4()), "recommendation_id": recommendation_id,
            "recommendation_type": "treatment", "clinician_id": clinician_id,
            "action": action, "edits": _copy(edits) if action == "modified" else {},
            "rationale": rationale, "decided_at": utc_now(),
            "boundary": "本决策未创建处方、剂量、用药任务或提醒。",
        }
        recommendation["status"] = action
        recommendation["decision"] = _copy(decision)
        journey.setdefault("recommendation_decisions", []).append(_copy(decision))
        if action in {"confirmed", "modified"}:
            journey["confirmed_treatment_direction"] = {
                "title": recommendation.get("title"), "status": action,
                "rationale": rationale, "source": "clinician", "confirmed_at": utc_now(),
                "boundary": "这是医生确认的治疗方向，不是处方；实际用药只读取医生处方。",
            }
        elif action == "rejected":
            journey["confirmed_treatment_direction"] = None
        medication_snapshot = _json(journey.get("medications") or [])
        await self.repository.save_journey(patient_id, journey)
        if medication_snapshot != _json(journey.get("medications") or []):
            raise RuntimeError("Treatment recommendation decision must not mutate medications")
        await self.repository.audit(clinician_id, "treatment_recommendation_decided", "journey", journey_id, {"recommendation_id": recommendation_id, "action": action})
        return {"recommendation": recommendation, "decision": decision, "created_prescription": False, "created_medication_task": False}

    async def rerun_clinician_assessment(self, clinician_id: str, journey_id: str) -> Dict[str, Any]:
        patient_id, journey, _ = await self._clinician_journey(clinician_id, journey_id)
        if not self._assessment_is_ready(journey):
            raise CareError("assessment_prerequisites_missing", "当前证据未满足重新运行条件。", 409)
        assessment = self._append_assessment_version(journey, force=True)
        await self.repository.save_journey(patient_id, journey)
        await self.repository.audit(
            clinician_id, "clinician_assessment_rerun", "journey", journey_id,
            {"version": assessment.get("version")},
        )
        return {
            "assessment": assessment,
            "patient_explanation": _copy(journey.get("patient_explanations", [])[-1]),
        }

    async def send_consultation(
        self,
        owner_id: str,
        journey_id: str,
        text: str,
        danger_signs: List[str],
    ) -> Dict[str, Any]:
        journey = await self._owned_journey(owner_id, journey_id)
        clean = _safe_text(text, 500)
        if not clean:
            raise CareError("message_required", "请输入你现在最担心的问题。")
        answer = answer_consultation(clean, danger_signs)
        now = utc_now()
        journey["consultation"]["messages"].extend([
            {"id": str(uuid.uuid4()), "role": "user", "text": clean, "created_at": now},
            {"id": str(uuid.uuid4()), "role": "assistant", "answer": answer, "created_at": utc_now()},
        ])
        if answer["urgency"] == "emergency":
            journey["triage"] = {"status": "emergency", "danger_signs": list(danger_signs), "checked_at": now, "message": answer["direct_answer"]}
            journey["current_stage"] = "emergency"
        elif isinstance(danger_signs, list):
            journey["triage"] = {"status": "stable", "danger_signs": [], "checked_at": now, "message": "当前未报告列出的急性危险信号；症状仍需按建议线下评估。"}
            journey["current_stage"] = "history_confirmation"
        journey["timeline"].append({"id": str(uuid.uuid4()), "type": "consultation_answered", "title": "完成一次问诊导航", "detail": answer["direct_answer"], "source": "decision_support", "created_at": now})
        await self.repository.save_journey(owner_id, journey)
        await self.repository.audit(owner_id, "consultation_answered", "journey", journey_id, {"intent": answer["intent"], "urgency": answer["urgency"]})
        return {"answer": answer, "triage": journey["triage"], "messages": journey["consultation"]["messages"]}

    async def update_clinical_history(self, owner_id: str, journey_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        journey = await self._owned_journey(owner_id, journey_id)
        required = ("conditions", "surgeries", "current_medications", "allergies", "family_history", "social_history")
        history = _copy(journey.get("clinical_history") or sample_clinical_history())
        for key in required:
            if key in payload:
                value = payload[key]
                if key == "social_history" and isinstance(value, dict):
                    history[key] = {name: _safe_text(item, 180) for name, item in value.items() if name in {"smoking", "alcohol", "occupation", "exposures"}}
                elif isinstance(value, list):
                    history[key] = _copy(value[:20])
        statuses = payload.get("field_statuses") if isinstance(payload.get("field_statuses"), dict) else {}
        for key in required:
            status = str(statuses.get(key) or history.get("field_statuses", {}).get(key) or "unconfirmed")
            if status not in {"confirmed", "unknown"}:
                status = "unconfirmed"
            history.setdefault("field_statuses", {})[key] = status
        complete = all(history["field_statuses"].get(key) in {"confirmed", "unknown"} for key in required)
        history["confirmation_status"] = "confirmed" if complete else "unconfirmed"
        history["confirmed_at"] = utc_now() if complete else None
        history["source"] = "用户对照完整虚构病例确认"
        journey["clinical_history"] = history
        journey["assessment"] = None
        journey["appointment_plan"] = None
        journey["current_stage"] = "record_sync" if complete else "history_confirmation"
        journey["timeline"].append({"id": str(uuid.uuid4()), "type": "clinical_history_updated", "title": "关键病史已更新", "detail": "旧判断已撤回；历史评估版本仍保留用于对比。", "source": "user", "created_at": utc_now()})
        await self.repository.save_journey(owner_id, journey)
        await self.repository.audit(owner_id, "clinical_history_updated", "journey", journey_id, {"complete": complete})
        return {"clinical_history": history, "current_stage": journey["current_stage"], "assessment_withdrawn": True}

    async def sync_record_batch(self, owner_id: str, journey_id: str, batch_key: str) -> Dict[str, Any]:
        journey = await self._owned_journey(owner_id, journey_id)
        batches = sample_record_batches()
        if batch_key not in batches:
            raise CareError("record_batch_not_found", "检查批次不存在。", 404)
        if not journey.get("hospital_connection"):
            raise CareError("hospital_not_connected", "请先授权连接病历来源。", 409)
        ordered = list(batches)
        required_previous = ordered[: ordered.index(batch_key)]
        if any(key not in journey["synced_batches"] for key in required_previous):
            raise CareError("previous_batch_required", "请先按时间顺序同步前一批记录。", 409)
        if batch_key in journey["synced_batches"]:
            return {"status": "already_synced", "batch": batches[batch_key], "journey": journey}
        existing_ids = {item["id"] for item in journey["records"]}
        for record in batches[batch_key]["records"]:
            if record["id"] not in existing_ids:
                imported = _copy(record)
                imported["verification_status"] = "hospital_confirmed"
                imported["source"] = {
                    "type": "sandbox_hospital", "label": "完整虚构病例 · 沙箱医院",
                    "locator": imported.get("source", {}).get("locator") or imported.get("title"),
                }
                journey["records"].append(imported)
        journey["synced_batches"].append(batch_key)
        current_report_ids = {item.get("id") for item in journey.get("exam_reports", [])}
        for report in reports_for_batches([batch_key]):
            if report["id"] not in current_report_ids:
                journey.setdefault("exam_reports", []).append(report)
        journey["evidence"] = product_build_evidence(journey["records"], journey["synced_batches"])
        journey["assessment"] = None
        journey["appointment_plan"] = None
        journey["hospital_sync_status"] = "completed"
        journey["last_hospital_sync_at"] = utc_now()
        journey["current_stage"] = _stage_after_records(journey)
        journey["timeline"].append({"id": str(uuid.uuid4()), "type": "record_batch_synced", "title": f"已同步：{batches[batch_key]['label']}", "detail": "医院签名结果已进入当前证据集；患者可对有误结果提出争议。", "source": "sandbox_hospital", "created_at": utc_now()})
        if self._assessment_is_ready(journey):
            self._append_assessment_version(journey)
        await self.repository.save_journey(owner_id, journey)
        return {"status": "completed", "batch_key": batch_key, "batch": batches[batch_key], "journey": journey}

    async def assessment_versions(self, owner_id: str, journey_id: str) -> List[Dict[str, Any]]:
        journey = await self._owned_journey(owner_id, journey_id)
        return _copy(journey.get("assessment_versions") or [])

    async def triage(self, owner_id: str, journey_id: str, danger_signs: List[str]) -> Dict[str, Any]:
        journey = await self._owned_journey(owner_id, journey_id)
        allowed = {"active_hemoptysis", "severe_dyspnea", "altered_consciousness", "low_oxygen"}
        signs = [item for item in danger_signs if item in allowed]
        journey["triage"] = {
            "status": "emergency" if signs else "stable",
            "danger_signs": signs,
            "checked_at": utc_now(),
            "message": (
                "当前存在危险信号，请立即前往急诊或拨打 120，不要等待系统分析。"
                if signs
                else "当前未报告上述危险信号，可继续核对病历并尽快线下就医。"
            ),
        }
        journey["current_stage"] = _stage_after_records(journey)
        journey["timeline"].append(
            {"id": str(uuid.uuid4()), "type": "triage_completed", "title": "安全分流完成", "detail": journey["triage"]["message"], "source": "user", "created_at": utc_now()}
        )
        await self.repository.save_journey(owner_id, journey)
        return journey["triage"]

    async def confirm_record(
        self,
        owner_id: str,
        journey_id: str,
        record_id: str,
        confirmed: bool,
        correction: str = "",
    ) -> Dict[str, Any]:
        journey = await self._owned_journey(owner_id, journey_id)
        record = next((item for item in journey["records"] if item["id"] == record_id), None)
        if record is None:
            raise CareError("record_not_found", "病历记录不存在。", 404)
        record["verification_status"] = "user_confirmed" if confirmed else "needs_correction"
        record["user_correction"] = _safe_text(correction, 300) if not confirmed else ""
        for evidence in journey["evidence"]:
            if evidence.get("record_id") == record_id:
                evidence["verification_status"] = record["verification_status"]
        journey["assessment"] = None
        journey["appointment_plan"] = None
        journey["current_stage"] = _stage_after_records(journey)
        await self.repository.save_journey(owner_id, journey)
        await self.repository.audit(owner_id, "record_reviewed", "record", record_id, {"confirmed": confirmed})
        return {"record": record, "current_stage": journey["current_stage"]}

    async def get_exam_reports(self, owner_id: str, journey_id: str) -> List[Dict[str, Any]]:
        journey = await self._owned_journey(owner_id, journey_id)
        return _copy(journey.get("exam_reports") or [])

    async def get_patient_explanations(self, owner_id: str, journey_id: str) -> List[Dict[str, Any]]:
        journey = await self._owned_journey(owner_id, journey_id)
        return _copy(journey.get("patient_explanations") or [])

    async def dispute_exam_report(
        self, owner_id: str, journey_id: str, report_id: str, reason: str
    ) -> Dict[str, Any]:
        journey = await self._owned_journey(owner_id, journey_id)
        report = next((item for item in journey.get("exam_reports", []) if item.get("id") == report_id), None)
        if report is None:
            raise CareError("exam_report_not_found", "检查报告不存在。", 404)
        clean_reason = _safe_text(reason, 400)
        if not clean_reason:
            raise CareError("dispute_reason_required", "请说明哪一项结果可能有误。")
        report["verification_status"] = "disputed"
        report["dispute"] = {"reason": clean_reason, "reported_at": utc_now(), "status": "pending_hospital_review"}
        for observation in report.get("observations", []):
            observation["verification_status"] = "disputed"
            observation["disputed"] = True
        affected_ids = set(report.get("record_ids") or [])
        for record in journey.get("records", []):
            if record.get("id") in affected_ids:
                record["verification_status"] = "needs_correction"
                record["user_correction"] = clean_reason
        for evidence in journey.get("evidence", []):
            if evidence.get("record_id") in affected_ids:
                evidence["verification_status"] = "disputed"
        journey["assessment"] = None
        journey["appointment_plan"] = None
        journey["current_stage"] = "confirm_records"
        journey.setdefault("timeline", []).append({
            "id": str(uuid.uuid4()), "type": "exam_report_disputed", "title": f"检查结果已提出争议：{report.get('title')}",
            "detail": "该报告已退出当前 Agent 证据集；历史诊断版本保留但不再视为当前判断。",
            "source": "user", "created_at": utc_now(),
        })
        await self.repository.save_journey(owner_id, journey)
        await self.repository.audit(owner_id, "exam_report_disputed", "exam_report", report_id)
        return {"report": report, "assessment_withdrawn": True, "current_stage": journey["current_stage"]}

    @staticmethod
    def _assessment_is_ready(journey: Dict[str, Any]) -> bool:
        accepted = {"user_confirmed", "hospital_confirmed", "doctor_confirmed"}
        return bool(
            journey.get("records")
            and journey.get("triage", {}).get("status") == "stable"
            and journey.get("clinical_history", {}).get("confirmation_status") == "confirmed"
            and all(item.get("verification_status") in accepted for item in journey.get("records", []))
        )

    def _append_assessment_version(self, journey: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
        stage = len(journey.get("synced_batches") or [])
        latest = (journey.get("assessment_versions") or [None])[-1]
        if not force and latest and int(latest.get("batch_stage") or 0) == stage:
            journey["assessment"] = _copy(latest)
            return latest
        assessment = product_assessment_version(journey)
        assessment["id"] = f"assessment-v{assessment['version']}"
        journey["assessment"] = assessment
        journey.setdefault("assessment_versions", []).append(_copy(assessment))
        journey["treatment_reference"] = _copy(assessment.get("treatment_reference") or treatment_reference())
        explanation = patient_explanation(assessment, journey.get("doctor_plan"))
        journey.setdefault("patient_explanations", []).append(_copy(explanation))
        journey.setdefault("consultation", {}).setdefault("messages", []).append({
            "id": str(uuid.uuid4()), "role": "assistant", "kind": "assessment_update",
            "assessment_version_id": explanation["assessment_version_id"],
            "patient_explanation": _copy(explanation), "created_at": utc_now(),
        })
        journey["current_stage"] = _stage_after_records(journey)
        journey.setdefault("timeline", []).append({
            "id": str(uuid.uuid4()), "type": "assessment_completed", "title": f"检查结果更新 · v{assessment['version']}",
            "detail": "病例页和问诊引用同一份患者解读；内部候选排序不会进入患者端。",
            "source": "decision_support", "created_at": utc_now(),
        })
        return assessment

    async def start_assessment(self, owner_id: str, journey_id: str) -> Dict[str, Any]:
        journey = await self._owned_journey(owner_id, journey_id)
        if journey["triage"]["status"] == "emergency":
            raise CareError("emergency_path_active", "危险信号未解除，不能用分析替代急诊。", 409)
        if journey["triage"]["status"] != "stable":
            raise CareError("triage_required", "请先完成安全分流。", 409)
        if journey.get("clinical_history", {}).get("confirmation_status") != "confirmed":
            raise CareError("clinical_history_not_confirmed", "请先逐项确认既往史、用药史、过敏史、家族史和暴露史。", 409)
        accepted = {"user_confirmed", "hospital_confirmed", "doctor_confirmed"}
        if not journey["records"] or any(item.get("verification_status") not in accepted for item in journey["records"]):
            raise CareError("records_not_confirmed", "所有病历字段确认后才能进入辅助判断。", 409)
        journey["consents"]["ai_analysis"] = True
        self._append_assessment_version(journey)
        await self.repository.save_journey(owner_id, journey)
        job = {
            "id": str(uuid.uuid4()),
            "kind": "assessment",
            "status": "completed",
            "journey_id": journey_id,
            "result": {"assessment": journey["assessment"], "current_stage": journey["current_stage"]},
            "created_at": utc_now(),
        }
        await self.repository.save_job(owner_id, job)
        return job

    async def create_appointment_plan(self, owner_id: str, journey_id: str) -> Dict[str, Any]:
        journey = await self._owned_journey(owner_id, journey_id)
        if not journey.get("assessment"):
            raise CareError("assessment_required", "请先完成辅助判断。", 409)
        official_url = os.getenv("CARE_APPOINTMENT_URL", "https://www.114yygh.com/").strip()
        allowed_hosts = {"www.114yygh.com", "114yygh.com"}
        allowed_hosts.update(item.strip().lower() for item in os.getenv("CARE_APPOINTMENT_HOSTS", "").split(",") if item.strip())
        parsed = urlparse(official_url)
        if parsed.scheme != "https" or (parsed.hostname or "").lower() not in allowed_hosts:
            raise CareError("appointment_url_untrusted", "医院官方挂号入口未通过安全校验。", 503)
        plan = {
            "status": "ready",
            "source": "care_navigation",
            "official_url": official_url,
            "link_label": "前往北京市预约挂号统一平台",
            "departments": journey["assessment"]["care_navigation"]["departments"],
            "reason": journey["assessment"]["urgency"]["reason"],
            "materials": journey["assessment"]["care_navigation"]["materials"],
            "questions_for_doctor": journey["assessment"]["care_navigation"]["questions_for_doctor"],
            "exam_discussion_items": journey["assessment"]["care_navigation"]["exam_discussion_items"],
            "booking_status": "not_confirmed",
            "disclaimer": "ClinicaLens 不读取号源，也不会声称已经挂号成功。",
            "created_at": utc_now(),
        }
        journey["appointment_plan"] = plan
        journey["current_stage"] = _stage_after_records(journey)
        journey["timeline"].append(
            {"id": str(uuid.uuid4()), "type": "appointment_plan_created", "title": "就医准备清单已生成", "detail": "科室建议、材料和问题清单已准备，可前往官方入口挂号。", "source": "agent", "created_at": utc_now()}
        )
        await self.repository.save_journey(owner_id, journey)
        return plan

    async def update_booking(self, owner_id: str, journey_id: str, booked: bool) -> Dict[str, Any]:
        journey = await self._owned_journey(owner_id, journey_id)
        if not journey.get("appointment_plan"):
            raise CareError("appointment_plan_required", "尚未生成就医准备清单。", 409)
        journey["appointment_plan"]["booking_status"] = "user_confirmed" if booked else "not_confirmed"
        journey["appointment_plan"]["confirmed_at"] = utc_now() if booked else None
        journey["current_stage"] = _stage_after_records(journey)
        await self.repository.save_journey(owner_id, journey)
        return journey["appointment_plan"]

    async def import_document(
        self,
        owner_id: str,
        *,
        filename: str,
        content_type: str,
        data: bytes,
        document_kind: str,
    ) -> Dict[str, Any]:
        if document_kind not in {"medical_report", "doctor_visit", "prescription"}:
            raise CareError("invalid_document_kind", "文档类型无效。")
        stored = await self.document_store.put(owner_id, filename, content_type, data)
        item = {
            "id": str(uuid.uuid4()),
            "status": "awaiting_user_confirmation",
            "document_kind": document_kind,
            "storage": stored,
            "provenance": {"source": "uploaded_document", "verification_status": "extracted"},
            "created_at": utc_now(),
            "notice": "开发环境未启用 OCR，请对照原文填写并确认结构化字段。",
        }
        await self.repository.save_import(owner_id, item)
        await self.repository.audit(owner_id, "document_uploaded", "record_import", item["id"], {"kind": document_kind, "sha256": stored["sha256"]})
        return item

    async def apply_doctor_document(
        self,
        owner_id: str,
        journey_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        journey = await self._owned_journey(owner_id, journey_id)
        source_type = str(payload.get("source_type") or "")
        import_id = str(payload.get("import_id") or "")
        if source_type == "uploaded_document":
            imported = await self.repository.get_import(owner_id, import_id)
            if not imported or imported.get("document_kind") not in {"doctor_visit", "prescription"}:
                raise CareError("doctor_document_required", "需要先上传医生门诊记录或处方。", 409)
            source_label = "医生文档 · 用户对照原文确认"
            verification_status = "user_confirmed"
        elif source_type == "sandbox_hospital" and journey.get("hospital_connection", {}).get("mode") == "sandbox":
            source_label = "沙箱医院 · 医生就诊记录"
            verification_status = "doctor_confirmed"
        else:
            raise CareError("invalid_doctor_source", "确诊与处方必须来自医院回传或医生文档。", 409)
        diagnoses = [_safe_text(item, 80) for item in payload.get("diagnoses", []) if _safe_text(item, 80)]
        if not diagnoses:
            raise CareError("diagnosis_required", "医生记录中至少需要一个诊断。")
        plan = {
            "authority": "doctor_plan",
            "diagnoses": diagnoses,
            "source": {"type": source_type, "label": source_label, "import_id": import_id or None},
            "verification_status": verification_status,
            "confirmed_at": utc_now(),
            "care_summary": _safe_text(payload.get("care_summary"), 400),
            "comparison": {
                "result": "doctor_confirmed_or_revised",
                "ai_primary": _safe_text((journey.get("assessment") or {}).get("primary_diagnosis", {}).get("name"), 120),
                "doctor_diagnosis": diagnoses[0],
                "confirmed_evidence": [
                    _safe_text(item, 180)
                    for item in (payload.get("confirmed_evidence") or [])[:10]
                    if _safe_text(item, 180)
                ],
                "revisions": [
                    _safe_text(item, 180)
                    for item in (payload.get("revisions") or [])[:10]
                    if _safe_text(item, 180)
                ],
            },
            "examination_orders": [
                {
                    "name": _safe_text(item if isinstance(item, str) else item.get("name"), 160),
                    "status": _safe_text(item.get("status"), 40) if isinstance(item, dict) else "doctor_ordered",
                    "source": "doctor_plan",
                }
                for item in (payload.get("examination_orders") or [])
                if _safe_text(item if isinstance(item, str) else item.get("name"), 160)
            ],
            "treatments": [
                {
                    "name": _safe_text(item.get("name"), 120),
                    "route": _safe_text(item.get("route"), 80),
                    "schedule": _safe_text(item.get("schedule"), 160),
                    "source": "doctor_plan",
                }
                for item in (payload.get("treatments") or []) if isinstance(item, dict) and _safe_text(item.get("name"), 120)
            ],
        }
        followup_at = _safe_text(payload.get("followup_at"), 40)
        journey["doctor_plan"] = plan
        journey["patient_explanations"] = build_patient_explanations(journey)
        explanation_map = {
            item["assessment_version_id"]: item for item in journey["patient_explanations"]
        }
        for message in journey.get("consultation", {}).get("messages", []):
            version_id = message.get("assessment_version_id")
            if message.get("kind") == "assessment_update" and version_id in explanation_map:
                message["patient_explanation"] = _copy(explanation_map[version_id])
        if followup_at:
            journey["followups"] = [
                {"id": str(uuid.uuid4()), "title": "按医生计划复诊", "scheduled_at": followup_at, "status": "scheduled", "source": _copy(plan["source"])}
            ]
        medications: List[Dict[str, Any]] = []
        allergy_terms = [
            str(item.get("allergen") or "")
            for item in journey.get("clinical_history", {}).get("allergies", [])
            if isinstance(item, dict) and item.get("status") not in {"known_none", "unknown"}
        ]
        for raw in payload.get("prescriptions", []) or []:
            name = _safe_text(raw.get("name"), 80)
            dose = _safe_text(raw.get("dose"), 80)
            frequency = _safe_text(raw.get("frequency"), 80)
            if not (name and dose and frequency):
                continue
            if any(term and (term in name or (term == "磺胺" and "磺胺" in name)) for term in allergy_terms):
                raise CareError("medication_allergy_conflict", f"医生处方中的{name}与已确认过敏史存在冲突，请联系医生或药师核对。", 409)
            medications.append(
                {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "dose": dose,
                    "frequency": frequency,
                    "course": _safe_text(raw.get("course"), 80),
                    "route": _safe_text(raw.get("route"), 60) or "口服",
                    "purpose": _safe_text(raw.get("purpose"), 180),
                    "prescription_original": _safe_text(raw.get("prescription_original"), 260) or f"{name} {dose}，{frequency}",
                    "next_at": _safe_text(raw.get("next_at"), 40),
                    "status": "active",
                    "source": _copy(plan["source"]),
                    "events": [],
                    "education": medication_education(name),
                    "boundary": "仅执行医生处方；如需改量、停药或换药，请联系医生。",
                }
            )
        journey["medications"] = medications
        journey["reminders"] = [
            {"id": str(uuid.uuid4()), "kind": "medication", "medication_id": item["id"], "scheduled_at": item["next_at"], "status": "scheduled"}
            for item in medications if item["next_at"]
        ]
        journey["current_stage"] = _stage_after_records(journey)
        journey["timeline"].append(
            {"id": str(uuid.uuid4()), "type": "doctor_plan_received", "title": "已收到医生确认结果", "detail": "AI 辅助判断与医生结论已分层保存。", "source": source_type, "created_at": utc_now()}
        )
        await self.repository.save_journey(owner_id, journey)
        await self.repository.audit(owner_id, "doctor_plan_received", "journey", journey_id, {"source": source_type, "medication_count": len(medications)})
        return {"doctor_plan": plan, "followups": journey["followups"], "medications": medications, "current_stage": journey["current_stage"]}

    async def add_medication_event(
        self,
        owner_id: str,
        medication_id: str,
        event_type: str,
        note: str = "",
    ) -> Dict[str, Any]:
        if event_type not in {"taken", "missed", "adverse"}:
            raise CareError("invalid_medication_event", "用药记录类型无效。")
        journeys = await self.list_journeys(owner_id)
        for journey in journeys:
            medication = next((item for item in journey["medications"] if item["id"] == medication_id), None)
            if medication is None:
                continue
            if medication.get("source", {}).get("type") not in {"sandbox_hospital", "uploaded_document", "hospital"}:
                raise CareError("doctor_source_required", "该药物没有可验证的医生来源。", 409)
            event = {"id": str(uuid.uuid4()), "type": event_type, "note": _safe_text(note, 300), "recorded_at": utc_now()}
            medication["events"].append(event)
            await self.repository.save_journey(owner_id, journey)
            await self.repository.audit(owner_id, "medication_event_recorded", "medication", medication_id, {"event": event_type})
            return event
        raise CareError("medication_not_found", "用药计划不存在。", 404)

    async def export_user_data(self, owner_id: str) -> Dict[str, Any]:
        return {"exported_at": utc_now(), "journeys": await self.list_patient_journeys(owner_id), "notice": "导出内容不包含登录凭证、医生内部建议或内部审计日志。"}

    async def _owned_journey(self, owner_id: str, journey_id: str) -> Dict[str, Any]:
        journey = await self.repository.get_journey(owner_id, journey_id)
        if journey is None:
            raise CareError("journey_not_found", "健康事件不存在或无权访问。", 404)
        upgraded = hydrate_journey_v3(hydrate_journey(journey))
        if upgraded != journey:
            await self.repository.save_journey(owner_id, upgraded)
        return upgraded


__all__ = [
    "ALLOWED_UPLOAD_TYPES",
    "CareError",
    "CareRuntime",
    "LocalDocumentStore",
    "MAX_UPLOAD_BYTES",
    "PostgresCareRepository",
    "S3DocumentStore",
    "SQLiteCareRepository",
    "build_assessment",
    "new_journey",
]
