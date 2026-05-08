"""
WS-9: persistent dead-letter queue for failed webhook deliveries.

The existing ``WebhookDispatcher`` (``src/queue/webhook.py``) already
implements:
    * HMAC-SHA256 signing
    * SSRF guards (scheme + private-network blocklist)
    * Exponential-backoff retry (default 3 attempts)

What's missing — and what this module adds — is **durability**: when
all retries are exhausted, the failure is logged and the payload
disappears. A worker restart, a brief receiver outage, or an
operator-initiated redeliver are all impossible without persisting
the failed payload somewhere.

``WebhookDLQ`` is a tiny SQLite-backed store with three operations:

    enqueue_failed(subscription_id, payload, last_error, attempts)
    claim_due(now, limit) -> list of entries ready for retry
    mark_delivered(entry_id) / mark_dead(entry_id)

A Celery beat task (``redeliver_failed_webhooks``) runs every minute
to claim due entries and re-attempt delivery via ``WebhookDispatcher``.
Operators can also force redelivery via the admin API
(``POST /api/v1/webhooks/{id}/dlq/{entry_id}/redeliver``).

Schema:

    CREATE TABLE webhook_dlq (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subscription_id TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        first_failed_at TEXT NOT NULL,
        next_retry_at TEXT NOT NULL,
        attempts INTEGER NOT NULL,
        max_attempts INTEGER NOT NULL,
        last_error TEXT,
        status TEXT NOT NULL,  -- 'pending' | 'delivered' | 'dead'
        delivered_at TEXT,
        dead_at TEXT
    );
    CREATE INDEX idx_dlq_due ON webhook_dlq (status, next_retry_at);
    CREATE INDEX idx_dlq_subscription ON webhook_dlq (subscription_id);

Status transitions:
    pending → delivered (success on a retry)
    pending → pending   (transient retry — bumps attempts + next_retry_at)
    pending → dead      (max_attempts reached)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.config import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS webhook_dlq (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subscription_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    first_failed_at TEXT NOT NULL,
    next_retry_at TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    max_attempts INTEGER NOT NULL,
    last_error TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    delivered_at TEXT,
    dead_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_dlq_due
    ON webhook_dlq (status, next_retry_at);
CREATE INDEX IF NOT EXISTS idx_dlq_subscription
    ON webhook_dlq (subscription_id);
"""


_DEFAULT_MAX_ATTEMPTS: int = 10
_DEFAULT_BACKOFF_BASE_SECONDS: int = 60


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DLQEntry:
    """A single failed-delivery record.

    ``payload`` is the dict form of the original ``WebhookPayload`` so
    we can rebuild it on retry without depending on the WebhookPayload
    type at this layer.
    """

    id: int
    subscription_id: str
    payload: dict[str, Any]
    first_failed_at: datetime
    next_retry_at: datetime
    attempts: int
    max_attempts: int
    last_error: str | None
    status: str  # 'pending' | 'delivered' | 'dead'
    delivered_at: datetime | None = None
    dead_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Datetime → ISO for JSON-friendly responses.
        for k in ("first_failed_at", "next_retry_at", "delivered_at", "dead_at"):
            v = d.get(k)
            d[k] = v.isoformat() if isinstance(v, datetime) else v
        return d


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class WebhookDLQ:
    """SQLite-backed dead-letter queue for failed webhook deliveries."""

    def __init__(
        self,
        db_path: str | Path = ".webhook_dlq/dlq.db",
        *,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        backoff_base_seconds: int = _DEFAULT_BACKOFF_BASE_SECONDS,
    ) -> None:
        """
        Args:
            db_path: Path to the SQLite database file. The parent
                directory is created if it doesn't exist. Use
                ``":memory:"`` for tests.
            max_attempts: Hard upper bound on retry attempts before an
                entry is marked dead and stops being claimed.
            backoff_base_seconds: Base delay for the exponential backoff
                that schedules the next retry. The Nth retry waits
                ``backoff_base_seconds * 2 ** (attempts - 1)`` seconds.
        """
        self._db_path = str(db_path)
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base_seconds
        # Single-process serialisation. Multi-process correctness is
        # delegated to SQLite's own file-level locking.
        self._lock = threading.Lock()

        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # For ``:memory:`` databases the connection has to live for the
        # life of the store; otherwise each ``_connect()`` opens its
        # own transient connection.
        self._memory_conn: sqlite3.Connection | None = None
        if self._db_path == ":memory:":
            self._memory_conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
            self._memory_conn.executescript(_SCHEMA_SQL)
            self._memory_conn.commit()
        else:
            with self._connect() as conn:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()

    # --- connection management ---------------------------------------

    @contextmanager
    def _connect(self) -> Any:
        if self._memory_conn is not None:
            yield self._memory_conn
            return
        conn = sqlite3.connect(self._db_path, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # --- public API --------------------------------------------------

    def enqueue_failed(
        self,
        *,
        subscription_id: str,
        payload: dict[str, Any],
        last_error: str | None,
        attempts: int = 1,
    ) -> int:
        """Record a failed delivery and schedule a retry.

        Args:
            subscription_id: The owning webhook subscription.
            payload: ``WebhookPayload.to_dict()`` form of the event.
            last_error: Stringified error from the last delivery
                attempt; may be None.
            attempts: How many times the dispatcher already tried
                before giving up on this round (≥ 1).

        Returns:
            The DLQ row id.
        """
        now = datetime.now(UTC)
        next_retry = self._compute_next_retry(now, attempts)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO webhook_dlq (
                    subscription_id, payload_json, first_failed_at,
                    next_retry_at, attempts, max_attempts, last_error,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    subscription_id,
                    json.dumps(payload, default=str),
                    now.isoformat(),
                    next_retry.isoformat(),
                    attempts,
                    self._max_attempts,
                    last_error,
                ),
            )
            conn.commit()
            entry_id = cur.lastrowid
            assert entry_id is not None
            logger.info(
                "webhook_dlq_enqueued",
                entry_id=entry_id,
                subscription_id=subscription_id,
                attempts=attempts,
                next_retry_at=next_retry.isoformat(),
            )
            return entry_id

    def claim_due(self, *, now: datetime | None = None, limit: int = 50) -> list[DLQEntry]:
        """Return up to ``limit`` pending entries whose ``next_retry_at`` ≤ now."""
        now = now or datetime.now(UTC)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM webhook_dlq
                WHERE status = 'pending'
                  AND next_retry_at <= ?
                ORDER BY next_retry_at ASC
                LIMIT ?
                """,
                (now.isoformat(), limit),
            ).fetchall()
            return [self._row_to_entry(row) for row in rows]

    def mark_delivered(self, entry_id: int) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE webhook_dlq SET status='delivered', delivered_at=? WHERE id=?",
                (now, entry_id),
            )
            conn.commit()
        logger.info("webhook_dlq_delivered", entry_id=entry_id)

    def mark_dead(self, entry_id: int, *, last_error: str | None = None) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE webhook_dlq
                SET status='dead', dead_at=?, last_error=COALESCE(?, last_error)
                WHERE id=?
                """,
                (now, last_error, entry_id),
            )
            conn.commit()
        logger.warning("webhook_dlq_dead", entry_id=entry_id, error=last_error)

    def reschedule_failed_attempt(
        self,
        entry_id: int,
        *,
        last_error: str | None,
    ) -> bool:
        """Bump attempts and either reschedule or mark dead.

        Returns True if the entry was rescheduled; False if it hit the
        max-attempts ceiling and was marked dead.
        """
        now = datetime.now(UTC)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT attempts, max_attempts FROM webhook_dlq WHERE id=?",
                (entry_id,),
            ).fetchone()
            if row is None:
                return False
            new_attempts = int(row["attempts"]) + 1
            if new_attempts >= int(row["max_attempts"]):
                conn.execute(
                    "UPDATE webhook_dlq SET status='dead', dead_at=?, attempts=?, last_error=? WHERE id=?",
                    (now.isoformat(), new_attempts, last_error, entry_id),
                )
                conn.commit()
                logger.warning(
                    "webhook_dlq_dead_after_retries",
                    entry_id=entry_id,
                    attempts=new_attempts,
                )
                return False

            next_retry = self._compute_next_retry(now, new_attempts)
            conn.execute(
                """
                UPDATE webhook_dlq
                SET attempts=?, next_retry_at=?, last_error=?
                WHERE id=?
                """,
                (new_attempts, next_retry.isoformat(), last_error, entry_id),
            )
            conn.commit()
            logger.info(
                "webhook_dlq_rescheduled",
                entry_id=entry_id,
                attempts=new_attempts,
                next_retry_at=next_retry.isoformat(),
            )
            return True

    def list_for_subscription(
        self,
        subscription_id: str,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[DLQEntry]:
        sql = "SELECT * FROM webhook_dlq WHERE subscription_id=?"
        params: list[Any] = [subscription_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        # Newest first; ``id DESC`` is the tie-breaker for sub-microsecond
        # enqueues that share the same ``first_failed_at`` timestamp.
        sql += " ORDER BY first_failed_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [self._row_to_entry(row) for row in rows]

    def get(self, entry_id: int) -> DLQEntry | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM webhook_dlq WHERE id=?", (entry_id,)
            ).fetchone()
            return self._row_to_entry(row) if row else None

    # --- internals ---------------------------------------------------

    def _compute_next_retry(self, now: datetime, attempts: int) -> datetime:
        # Capped exponential backoff. After ~10 attempts we'd be at ~17h
        # backoff with the default base of 60s; cap at 24h to keep a
        # daily ceiling regardless of base.
        seconds = self._backoff_base * (2 ** max(0, attempts - 1))
        seconds = min(seconds, 86_400)
        return now + timedelta(seconds=seconds)

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> DLQEntry:
        def _parse(value: Any) -> datetime | None:
            return datetime.fromisoformat(value) if value else None

        return DLQEntry(
            id=int(row["id"]),
            subscription_id=str(row["subscription_id"]),
            payload=json.loads(row["payload_json"]),
            first_failed_at=_parse(row["first_failed_at"]),  # type: ignore[arg-type]
            next_retry_at=_parse(row["next_retry_at"]),  # type: ignore[arg-type]
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            last_error=row["last_error"],
            status=str(row["status"]),
            delivered_at=_parse(row["delivered_at"]),
            dead_at=_parse(row["dead_at"]),
        )


__all__ = ["DLQEntry", "WebhookDLQ"]
