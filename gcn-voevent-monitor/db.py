"""SQLite database layer for GCN VOEvent storage."""

import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "gcn_events.db"
_local = threading.local()


def get_conn() -> sqlite3.Connection:
    """Return a per-thread connection."""
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db() -> None:
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ivorn       TEXT UNIQUE,
            topic       TEXT,
            role        TEXT,
            received_at TEXT NOT NULL,
            event_time  TEXT,
            author      TEXT,
            description TEXT,
            ra          REAL,
            dec         REAL,
            error_radius REAL,
            raw_xml     TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_received_at ON events(received_at)")
    conn.commit()


def insert_event(event: dict) -> bool:
    """Insert event; return True if inserted, False if duplicate."""
    conn = get_conn()
    try:
        conn.execute(
            """
            INSERT INTO events
                (ivorn, topic, role, received_at, event_time,
                 author, description, ra, dec, error_radius, raw_xml)
            VALUES
                (:ivorn, :topic, :role, :received_at, :event_time,
                 :author, :description, :ra, :dec, :error_radius, :raw_xml)
            """,
            event,
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def prune_old_events(days: int = 31) -> int:
    """Delete events older than `days`. Returns count deleted."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = get_conn()
    cur = conn.execute("DELETE FROM events WHERE received_at < ?", (cutoff,))
    conn.commit()
    return cur.rowcount


def query_events(hours: int | None = 24, limit: int = 500) -> list[dict]:
    conn = get_conn()
    if hours is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cur = conn.execute(
            "SELECT * FROM events WHERE received_at >= ? ORDER BY received_at DESC LIMIT ?",
            (cutoff, limit),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM events ORDER BY received_at DESC LIMIT ?", (limit,)
        )
    return [dict(row) for row in cur.fetchall()]


def count_events(hours: int | None = None) -> int:
    conn = get_conn()
    if hours is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cur = conn.execute(
            "SELECT COUNT(*) FROM events WHERE received_at >= ?", (cutoff,)
        )
    else:
        cur = conn.execute("SELECT COUNT(*) FROM events")
    return cur.fetchone()[0]


def stats_by_topic(hours: int = 24) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    conn = get_conn()
    cur = conn.execute(
        """
        SELECT topic, COUNT(*) AS cnt
        FROM events
        WHERE received_at >= ?
        GROUP BY topic
        ORDER BY cnt DESC
        """,
        (cutoff,),
    )
    return [dict(row) for row in cur.fetchall()]


def events_per_hour(hours: int = 24) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    conn = get_conn()
    cur = conn.execute(
        """
        SELECT strftime('%Y-%m-%dT%H:00:00', received_at) AS hour,
               COUNT(*) AS cnt
        FROM events
        WHERE received_at >= ?
        GROUP BY hour
        ORDER BY hour
        """,
        (cutoff,),
    )
    return [dict(row) for row in cur.fetchall()]
