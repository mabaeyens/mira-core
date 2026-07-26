"""SQLite persistence layer for conversations and messages.

Schema
------
projects      : id, name, local_path, github_repo, created_at, last_used
conversations : id, title, created_at, updated_at, model_name, project_id
messages      : id, conversation_id, role, content, created_at
memories      : id, text, created_at

Only 'user' and 'assistant' roles are stored — tool / search messages are
ephemeral and re-generated on each turn.  Content is stored as plain text
(the original user message, not the RAG-augmented version).
"""

import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .config import DB_PATH, MAX_CONVERSATIONS

_local = threading.local()


# ── Connection ────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection, creating it on first use."""
    if not getattr(_local, "conn", None):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return _local.conn


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup."""
    # Migrate from old in-package location if the new path doesn't exist yet.
    old_path = Path(__file__).parent / "conversations.db"
    if old_path.exists() and not DB_PATH.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, DB_PATH)

    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id          TEXT    PRIMARY KEY,
                name        TEXT    NOT NULL,
                local_path  TEXT,
                github_repo TEXT,
                created_at  INTEGER NOT NULL,
                last_used   INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT    PRIMARY KEY,
                title      TEXT    NOT NULL DEFAULT 'New conversation',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                model_name TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT    NOT NULL
                                REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                created_at      INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                text       TEXT    NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reminders (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                text         TEXT    NOT NULL,
                scheduled_at INTEGER NOT NULL,
                fired        INTEGER NOT NULL DEFAULT 0,
                created_at   INTEGER NOT NULL
            );
        """)
        # Migration: add project_id to existing conversations tables
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN project_id TEXT")
        except Exception:
            pass  # column already exists
        # Migration: add thinking_content to existing messages tables
        try:
            conn.execute("ALTER TABLE messages ADD COLUMN thinking_content TEXT")
        except Exception:
            pass  # column already exists
        # FTS5 index for conversation search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
            USING fts5(content, conversation_id UNINDEXED)
        """)
        # One-time backfill for pre-existing messages (when FTS table is newly created)
        fts_count = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        if fts_count == 0:
            conn.execute(
                "INSERT INTO messages_fts (content, conversation_id)"
                " SELECT content, conversation_id FROM messages"
            )


# ── Projects ──────────────────────────────────────────────────────────────────

def create_project(name: str, local_path: Optional[str] = None, github_repo: Optional[str] = None) -> str:
    project_id = uuid.uuid4().hex
    now = int(time.time())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, local_path, github_repo, created_at, last_used)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, name, local_path, github_repo, now, now),
        )
    return project_id


def list_projects() -> List[Dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT p.id, p.name, p.local_path, p.github_repo, p.created_at, p.last_used,"
            " (SELECT COUNT(*) FROM conversations c WHERE c.project_id = p.id) AS conversation_count"
            " FROM projects p ORDER BY p.last_used DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_project(project_id: str) -> Optional[Dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, name, local_path, github_repo, created_at, last_used"
            " FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_project(project_id: str) -> None:
    """Delete a project and unfile its conversations, which are NOT deleted.

    Clearing project_id matters now that conversations can be moved between
    projects: without it the rows keep pointing at a project that no longer
    exists, and a client that groups by project has nowhere to put them.
    """
    with _conn() as conn:
        conn.execute(
            "UPDATE conversations SET project_id = NULL WHERE project_id = ?",
            (project_id,),
        )
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))


def touch_project(project_id: str) -> None:
    """Update last_used timestamp when a project's conversation becomes active."""
    with _conn() as conn:
        conn.execute(
            "UPDATE projects SET last_used = ? WHERE id = ?",
            (int(time.time()), project_id),
        )


# ── Conversations ─────────────────────────────────────────────────────────────

def create_conversation(model_name: str, project_id: Optional[str] = None, conv_id: Optional[str] = None) -> str:
    """Insert a new conversation row and evict old ones in a single transaction."""
    conv_id = conv_id or uuid.uuid4().hex
    now = int(time.time())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, model_name, project_id)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, "New conversation", now, now, model_name, project_id),
        )
        conn.execute("""
            DELETE FROM conversations WHERE id IN (
                SELECT id FROM conversations
                ORDER BY updated_at DESC
                LIMIT -1 OFFSET ?
            )
        """, (MAX_CONVERSATIONS,))
    if project_id:
        touch_project(project_id)
    return conv_id


def list_conversations() -> List[Dict]:
    """Return all conversations ordered by most recently updated first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at, c.model_name, c.project_id,"
            " (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id)"
            " AS message_count"
            " FROM conversations c ORDER BY c.updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str) -> Optional[Dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at, model_name, project_id"
            " FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_conversation(conv_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


def update_title(conv_id: str, title: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id)
        )


def update_project(conv_id: str, project_id: Optional[str]) -> None:
    """Move a conversation into a project, or out of every project when None."""
    with _conn() as conn:
        conn.execute(
            "UPDATE conversations SET project_id = ? WHERE id = ?",
            (project_id, conv_id),
        )
    if project_id:
        touch_project(project_id)


# ── Messages ──────────────────────────────────────────────────────────────────

def save_messages(conv_id: str, messages: List[Dict]) -> None:
    """Append messages and bump updated_at."""
    now = int(time.time())
    with _conn() as conn:
        for msg in messages:
            content = str(msg.get("content", ""))
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, thinking_content, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (conv_id, msg["role"], content,
                 msg.get("thinking_content") or None, now),
            )
            conn.execute(
                "INSERT INTO messages_fts (content, conversation_id) VALUES (?, ?)",
                (content, conv_id),
            )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
        )


def load_messages(conv_id: str) -> List[Dict]:
    """Return messages as [{role, content, thinking_content}] ordered by insertion."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, thinking_content FROM messages"
            " WHERE conversation_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"],
             "thinking_content": r["thinking_content"]} for r in rows]


def replace_messages(conv_id: str, messages: List[Dict]) -> None:
    """Replace all messages for a conversation (used after summarize-and-compress)."""
    now = int(time.time())
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM messages_fts WHERE conversation_id = ?", (conv_id,))
        for msg in messages:
            content = str(msg.get("content", ""))
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, thinking_content, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (conv_id, msg["role"], content,
                 msg.get("thinking_content") or None, now),
            )
            conn.execute(
                "INSERT INTO messages_fts (content, conversation_id) VALUES (?, ?)",
                (content, conv_id),
            )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
        )


# ── Memories ──────────────────────────────────────────────────────────────────

def get_memories() -> List[Dict]:
    rows = _conn().execute(
        "SELECT id, text, created_at FROM memories ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def add_memory(text: str) -> int:
    now = int(time.time())
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO memories (text, created_at) VALUES (?, ?)", (text, now)
        )
        return cur.lastrowid


def delete_memory(memory_id: int) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))


# ── Search ────────────────────────────────────────────────────────────────────

def search_conversations(query: str, limit: int = 10) -> List[Dict]:
    """Full-text search over message content. Returns matching conversations ordered by recency."""
    if not query.strip():
        return []
    safe_q = '"{}"'.format(query.strip().replace('"', '""'))
    with _conn() as conn:
        try:
            rows = conn.execute(
                """
                SELECT c.id AS conversation_id, c.title, c.created_at, c.updated_at,
                       substr(f.content, 1, 200) AS snippet
                FROM messages_fts f
                JOIN conversations c ON c.id = f.conversation_id
                WHERE f MATCH ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (safe_q, limit),
            ).fetchall()
        except Exception:
            # FTS unavailable or query syntax error — fall back to LIKE
            rows = conn.execute(
                """
                SELECT c.id AS conversation_id, c.title, c.created_at, c.updated_at,
                       substr(m.content, 1, 200) AS snippet
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE m.content LIKE ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (f"%{query.strip()}%", limit),
            ).fetchall()
    return [dict(r) for r in rows]


# ── Reminders ─────────────────────────────────────────────────────────────────

def add_reminder(text: str, scheduled_at: int) -> int:
    now = int(time.time())
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO reminders (text, scheduled_at, fired, created_at) VALUES (?, ?, 0, ?)",
            (text, scheduled_at, now),
        )
        return cur.lastrowid


def get_pending_reminders() -> List[Dict]:
    now = int(time.time())
    rows = _conn().execute(
        "SELECT id, text, scheduled_at, created_at FROM reminders"
        " WHERE fired = 0 AND scheduled_at <= ? ORDER BY scheduled_at",
        (now,),
    ).fetchall()
    return [dict(r) for r in rows]


def list_reminders() -> List[Dict]:
    rows = _conn().execute(
        "SELECT id, text, scheduled_at, fired, created_at FROM reminders"
        " WHERE fired = 0 ORDER BY scheduled_at"
    ).fetchall()
    return [dict(r) for r in rows]


def mark_reminder_fired(reminder_id: int) -> None:
    with _conn() as conn:
        conn.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))


def delete_reminder(reminder_id: int) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))


