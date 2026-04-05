"""SQLite persistence layer for conversations and messages.

Schema
------
conversations : id, title, created_at, updated_at, model_name
messages      : id, conversation_id, role, content, created_at

Only 'user' and 'assistant' roles are stored — tool / search messages are
ephemeral and re-generated on each turn.  Content is stored as plain text
(the original user message, not the RAG-augmented version).
"""

import sqlite3
import time
import uuid
from typing import Dict, List, Optional

from config import DB_PATH, MAX_CONVERSATIONS


# ── Connection ────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they do not exist. Safe to call on every startup."""
    with _conn() as conn:
        conn.executescript("""
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
        """)


# ── Conversations ─────────────────────────────────────────────────────────────

def create_conversation(model_name: str) -> str:
    """Insert a new conversation row and return its ID."""
    conv_id = uuid.uuid4().hex
    now = int(time.time())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, model_name)"
            " VALUES (?, ?, ?, ?, ?)",
            (conv_id, "New conversation", now, now, model_name),
        )
    _evict_old()
    return conv_id


def list_conversations() -> List[Dict]:
    """Return all conversations ordered by most recently updated first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at, model_name"
            " FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conv_id: str) -> Optional[Dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at, model_name"
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


# ── Messages ──────────────────────────────────────────────────────────────────

def save_messages(conv_id: str, messages: List[Dict]) -> None:
    """Append messages and bump updated_at."""
    now = int(time.time())
    with _conn() as conn:
        for msg in messages:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at)"
                " VALUES (?, ?, ?, ?)",
                (conv_id, msg["role"], str(msg.get("content", "")), now),
            )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
        )


def load_messages(conv_id: str) -> List[Dict]:
    """Return messages as [{role, content}] ordered by insertion."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages"
            " WHERE conversation_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def replace_messages(conv_id: str, messages: List[Dict]) -> None:
    """Replace all messages for a conversation (used after summarize-and-compress)."""
    now = int(time.time())
    with _conn() as conn:
        conn.execute(
            "DELETE FROM messages WHERE conversation_id = ?", (conv_id,)
        )
        for msg in messages:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at)"
                " VALUES (?, ?, ?, ?)",
                (conv_id, msg["role"], str(msg.get("content", "")), now),
            )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
        )


# ── Eviction ──────────────────────────────────────────────────────────────────

def _evict_old() -> None:
    """Delete conversations beyond MAX_CONVERSATIONS (oldest updated_at first)."""
    with _conn() as conn:
        conn.execute("""
            DELETE FROM conversations WHERE id IN (
                SELECT id FROM conversations
                ORDER BY updated_at DESC
                LIMIT -1 OFFSET ?
            )
        """, (MAX_CONVERSATIONS,))
