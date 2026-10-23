from pathlib import Path
import json
import sqlite3
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "chat_history.db"


def _connect():
    """Create a SQLite connection with foreign-key support."""
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    """Create chat-history tables and migrate older databases safely."""
    with _connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                timings TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)

        # Add mode to existing databases without deleting old chat history.
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "mode" not in columns:
            connection.execute(
                "ALTER TABLE messages ADD COLUMN mode TEXT NOT NULL DEFAULT 'chat'"
            )

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, id)
        """)


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def create_conversation(title="New Chat"):
    """Create a conversation and return its integer ID."""
    timestamp = _now()
    cleaned_title = " ".join((title or "").split()) or "New Chat"
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO conversations (title, created_at, updated_at)
            VALUES (?, ?, ?)
            """,
            (cleaned_title[:100], timestamp, timestamp),
        )
        return cursor.lastrowid


def list_conversations():
    """Return conversations from newest activity to oldest."""
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def get_conversation(conversation_id):
    """Return one conversation, or None if it does not exist."""
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
        return dict(row) if row else None


def rename_conversation(conversation_id, title):
    """Rename a conversation."""
    cleaned_title = " ".join((title or "").split())
    if not cleaned_title:
        raise ValueError("Conversation title cannot be empty.")
    with _connect() as connection:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (cleaned_title[:100], _now(), conversation_id),
        )
        if cursor.rowcount == 0:
            raise ValueError("Conversation not found.")


def delete_conversation(conversation_id):
    """Delete a conversation and its messages."""
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        return cursor.rowcount > 0


def save_message(
    conversation_id,
    role,
    content,
    sources=None,
    timings=None,
    mode="chat",
):
    """Save a user/assistant message, its mode, and conversation activity."""
    if role not in {"user", "assistant"}:
        raise ValueError("Role must be 'user' or 'assistant'.")
    if mode not in {"chat", "comparison"}:
        raise ValueError("Mode must be 'chat' or 'comparison'.")

    timestamp = _now()
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    timings_json = json.dumps(timings or {}, ensure_ascii=False)

    with _connect() as connection:
        conversation = connection.execute(
            "SELECT title FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

        if not conversation:
            raise ValueError("Conversation not found.")

        connection.execute(
            """
            INSERT INTO messages
                (conversation_id, role, content, sources, timings, created_at, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                sources_json,
                timings_json,
                timestamp,
                mode,
            ),
        )

        # Give a new normal chat a useful title from its first question.
        if (
            mode == "chat"
            and role == "user"
            and conversation["title"] == "New Chat"
        ):
            title = " ".join(content.split())
            if title:
                if len(title) > 48:
                    title = title[:45].rstrip() + "..."
                connection.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?",
                    (title, conversation_id),
                )

        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (timestamp, conversation_id),
        )


def get_messages(conversation_id):
    """Load messages and their mode in chronological order."""
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT role, content, sources, timings, created_at, mode
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

    messages = []
    for row in rows:
        try:
            sources = json.loads(row["sources"] or "[]")
        except (TypeError, json.JSONDecodeError):
            sources = []

        try:
            timings = json.loads(row["timings"] or "{}")
        except (TypeError, json.JSONDecodeError):
            timings = {}

        messages.append({
            "role": row["role"],
            "content": row["content"],
            "sources": sources,
            "timings": timings,
            "created_at": row["created_at"],
            "mode": row["mode"] or "chat",
        })

    return messages


def clear_conversation_messages(conversation_id):
    """Clear one conversation's messages but keep the conversation itself."""
    with _connect() as connection:
        connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.execute(
            """
            UPDATE conversations
            SET title = 'New Chat', updated_at = ?
            WHERE id = ?
            """,
            (_now(), conversation_id),
        )


initialize_database()
