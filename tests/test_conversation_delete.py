"""Deleting a conversation has to take its search-index copy with it.

`messages` cascades off the conversations row via a foreign key, but
`messages_fts` is an FTS5 virtual table maintained by hand in db.py and sits
outside that cascade, so every deleted conversation left its full text behind in
the index. Found 2026-08-01 on the live database: 94 live messages against 1266
indexed rows — 1172 orphans belonging to 539 conversations that no longer existed,
roughly twelve times more dead index than live data, growing with every delete.

**It was not a disclosure path.** `search_conversations()` INNER JOINs
`conversations`, so orphaned rows were filtered out and never reached a caller.
That JOIN is load-bearing and `test_deleted_content_is_not_findable` below pins
it; that test passes with or without the purge, and is here to keep the join,
not to cover this bug. The two tests either side of it are the regression.
"""
import pytest

from core import db


def _fts_rows(conv_id):
    with db._conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE conversation_id = ?", (conv_id,)
        ).fetchone()[0]


def _msg_rows(conv_id):
    with db._conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conv_id,)
        ).fetchone()[0]


@pytest.fixture
def conv_with_messages():
    conv_id = db.create_conversation("test-model")
    db.save_messages(conv_id, [
        {"role": "user", "content": "unmistakable-needle-alpha"},
        {"role": "assistant", "content": "unmistakable-needle-beta"},
    ])
    yield conv_id
    db.delete_conversation(conv_id)  # idempotent; safe if the test already deleted


def test_delete_removes_messages_and_index_together(conv_with_messages):
    conv_id = conv_with_messages
    assert _msg_rows(conv_id) == 2
    assert _fts_rows(conv_id) == 2, "precondition: save_messages indexes both rows"

    db.delete_conversation(conv_id)

    assert _msg_rows(conv_id) == 0, "FK cascade should have taken the messages"
    assert _fts_rows(conv_id) == 0, "the index copy must go too, or it stays searchable"


def test_deleted_content_is_not_findable(conv_with_messages):
    """Pins the INNER JOIN in search_conversations, which is what kept the orphans
    invisible. Passes with or without the FTS purge — drop the join to `messages_fts`
    alone and this is the test that fails."""
    conv_id = conv_with_messages
    assert db.search_conversations("unmistakable-needle-alpha")

    db.delete_conversation(conv_id)

    hits = db.search_conversations("unmistakable-needle-alpha")
    assert not hits, f"deleted conversation still searchable: {hits}"


def test_a_second_delete_leaves_nothing_behind(conv_with_messages):
    """Delete twice: the second call must be a no-op, not a partial state.

    Deliberately NOT a global `SELECT COUNT(*) ... NOT IN (SELECT id FROM
    conversations)` assertion. These tests run against the real
    ~/.local/share/mira/conversations.db, shared with every other module and with
    Miguel's own data, so a global invariant fails on any orphan left by an older
    build or an interrupted run rather than on anything this test did. Scoped to
    the row under test. A whole-table sweep belongs in a maintenance script.
    """
    conv_id = conv_with_messages
    db.delete_conversation(conv_id)
    db.delete_conversation(conv_id)
    assert _msg_rows(conv_id) == 0
    assert _fts_rows(conv_id) == 0
