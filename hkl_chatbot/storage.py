"""Bounded conversation storage and atomic rate limits for a single host.

SQLite WAL supports multiple WSGI workers sharing the same local file. It is not
a distributed database; use a shared service before scaling across hosts.
"""

from contextlib import contextmanager
import hashlib
import hmac
from pathlib import Path
import secrets
import sqlite3
import time
import uuid


class ConversationMissing(Exception):
    pass


class CapacityExceeded(Exception):
    pass


class Store:
    def __init__(self, path: str, ttl: int, max_conversations: int, max_history: int):
        self.path = path
        self.ttl = ttl
        self.max_conversations = max_conversations
        self.max_history = max_history
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, expires REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversation_expiry ON conversations(expires);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL,
                    user TEXT NOT NULL, bot TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS message_conversation ON messages(conversation_id, id);
                CREATE TABLE IF NOT EXISTS rate_limits (
                    key TEXT PRIMARY KEY, count INTEGER NOT NULL, expires REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS rate_expiry ON rate_limits(expires);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create(self) -> tuple[str, str]:
        now = time.time()
        conversation_id, token = uuid.uuid4().hex, secrets.token_urlsafe(32)
        with self.connection() as db:
            # Reserve capacity and insert in one transaction across threads/workers.
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM conversations WHERE expires <= ?", (now,))
            if db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] >= self.max_conversations:
                raise CapacityExceeded
            db.execute("INSERT INTO conversations VALUES (?, ?, ?)",
                       (conversation_id, self.digest(token), now + self.ttl))
        return conversation_id, token

    def _authorize(self, db, conversation_id: str, token: str):
        row = db.execute("SELECT token_hash, expires FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if not row or row[1] <= time.time() or not hmac.compare_digest(row[0], self.digest(token)):
            raise ConversationMissing

    def authorize(self, conversation_id: str, token: str):
        with self.connection() as db:
            self._authorize(db, conversation_id, token)

    def append(self, conversation_id: str, token: str, user: str, bot: str):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._authorize(db, conversation_id, token)
            db.execute("INSERT INTO messages (conversation_id, user, bot) VALUES (?, ?, ?)",
                       (conversation_id, user, bot))
            db.execute("""DELETE FROM messages WHERE conversation_id = ? AND id NOT IN
                          (SELECT id FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?)""",
                       (conversation_id, conversation_id, self.max_history))

    def history(self, conversation_id: str, token: str) -> list[dict[str, str]]:
        with self.connection() as db:
            # A read transaction gives authorization and history a consistent snapshot.
            db.execute("BEGIN")
            self._authorize(db, conversation_id, token)
            rows = db.execute("SELECT user, bot FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)).fetchall()
        return [{"user": user, "bot": bot} for user, bot in rows]

    def delete(self, conversation_id: str, token: str):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._authorize(db, conversation_id, token)
            db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def prune(self):
        """Also run via the CLI scheduler so expired data is removed when idle."""
        with self.connection() as db:
            db.execute("DELETE FROM conversations WHERE expires <= ?", (time.time(),))
            db.execute("DELETE FROM rate_limits WHERE expires <= ?", (time.time(),))

    def allow(self, identity: str, limit: int, window: int = 60) -> bool:
        now = time.time()
        key = self.digest(identity)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM rate_limits WHERE expires <= ?", (now,))
            row = db.execute("SELECT count FROM rate_limits WHERE key = ?", (key,)).fetchone()
            if row:
                if row[0] >= limit:
                    return False
                db.execute("UPDATE rate_limits SET count = count + 1 WHERE key = ?", (key,))
            else:
                # Bound state even when presented with many distinct client addresses.
                if db.execute("SELECT COUNT(*) FROM rate_limits").fetchone()[0] >= 10000:
                    return False
                db.execute("INSERT INTO rate_limits VALUES (?, 1, ?)", (key, now + window))
        return True
