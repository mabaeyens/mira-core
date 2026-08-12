"""A retry replaces the failed turn instead of stacking on top of it.

Demonstrated live on 2026-08-12 before the fix: two identical posts to /chat on
one conversation id left four rows, two of them the same user message, with the
broken reply still sitting between them. Every later turn was then built on both
copies. These tests pin the two halves of the fix — the database, and the
in-memory history the orchestrator prompts from.
"""
import pytest

from core import db
from server import _rollback_to_last_user


@pytest.fixture
def conv():
    """A fresh conversation in the test database conftest already redirected.

    Deliberately not a per-test DB_PATH: `db._conn()` caches a per-thread
    connection on first use, so re-binding DB_PATH afterwards changes nothing and
    the test would quietly assert against whatever database the suite opened
    first. Isolation here is by conversation id, which is what the code isolates
    by anyway — and it means every assertion has to be scoped to this
    conversation, including the full-text ones.
    """
    return db.create_conversation("test-model")


def _roles(conv_id):
    return [m["role"] for m in db.load_messages(conv_id)]


def _contents(conv_id):
    return [m["content"] for m in db.load_messages(conv_id)]


class TestDropLastTurn:
    def test_drops_the_question_and_its_answer(self, conv):
        db.save_messages(conv, [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
        ])
        db.save_messages(conv, [
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "!!!!"},
        ])

        removed = db.drop_last_turn(conv)

        assert removed == 2
        assert _contents(conv) == ["first", "first answer"]

    def test_drops_everything_after_the_last_user_message(self, conv):
        """A turn can persist more than two rows; the unit is the whole turn."""
        db.save_messages(conv, [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": "tool output"},
            {"role": "assistant", "content": "follow-up"},
        ])

        assert db.drop_last_turn(conv) == 4
        assert _roles(conv) == []

    def test_empty_conversation_is_not_an_error(self, conv):
        assert db.drop_last_turn(conv) == 0

    def test_assistant_only_history_is_left_alone(self, conv):
        """No user row means nothing a retry could be replacing."""
        db.save_messages(conv, [{"role": "assistant", "content": "greeting"}])

        assert db.drop_last_turn(conv) == 0
        assert _contents(conv) == ["greeting"]

    def test_other_conversations_are_untouched(self, conv):
        other = db.create_conversation("test-model")
        db.save_messages(other, [
            {"role": "user", "content": "theirs"},
            {"role": "assistant", "content": "theirs answer"},
        ])
        db.save_messages(conv, [
            {"role": "user", "content": "mine"},
            {"role": "assistant", "content": "mine answer"},
        ])

        db.drop_last_turn(conv)

        assert _contents(other) == ["theirs", "theirs answer"]

    def test_search_no_longer_finds_the_dropped_turn(self, conv):
        """The FTS index has to lose the row too, or search resurrects it."""
        db.save_messages(conv, [
            {"role": "user", "content": "kept question"},
            {"role": "assistant", "content": "kept answer"},
        ])
        db.save_messages(conv, [
            {"role": "user", "content": "xyzzyunrepeatable"},
            {"role": "assistant", "content": "broken"},
        ])

        db.drop_last_turn(conv)

        def matches(term):
            with db._conn() as c:
                return c.execute(
                    "SELECT count(*) AS n FROM messages_fts"
                    " WHERE messages_fts MATCH ? AND conversation_id = ?",
                    (term, conv),
                ).fetchone()["n"]

        assert matches("xyzzyunrepeatable") == 0
        assert matches("kept") == 2

    def test_two_retries_in_a_row_do_not_dig_further_each_time(self, conv):
        """Retrying twice replaces twice; it must not eat the earlier turn."""
        db.save_messages(conv, [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
        ])
        for attempt in ("bad", "worse"):
            db.save_messages(conv, [
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": attempt},
            ])
            db.drop_last_turn(conv)

        assert _contents(conv) == ["first", "first answer"]


class TestRollbackToLastUser:
    def test_cuts_the_last_user_message_and_what_followed(self):
        history = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "!!!!"},
        ]

        assert _rollback_to_last_user(history) == 2
        assert [m["role"] for m in history] == ["system", "user", "assistant"]

    def test_mutates_in_place(self):
        """The orchestrator holds this list — rebinding it would fix nothing."""
        history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        same_object = history

        _rollback_to_last_user(history)

        assert same_object is history
        assert history == []

    def test_takes_tool_messages_with_it(self):
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "calling"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "answer"},
        ]

        assert _rollback_to_last_user(history) == 4
        assert history == []

    def test_no_user_message_means_no_cut(self):
        history = [{"role": "system", "content": "rules"}]

        assert _rollback_to_last_user(history) == 0
        assert len(history) == 1

    def test_empty_history(self):
        assert _rollback_to_last_user([]) == 0
