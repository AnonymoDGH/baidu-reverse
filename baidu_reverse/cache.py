"""Persistent response cache.

Identical stateless requests (same model, flags and query) are answered from a
local SQLite file instead of hitting Baidu again. That saves latency and, more
importantly, keeps you away from the ``kunlun`` rate limit when a script or an
agent re-asks the same thing.

Only complete, non-empty answers of *stateless* calls are cached (calls made
inside a :class:`~baidu_reverse.session.Conversation` depend on server state and
are never cached).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Optional


class ResponseCache:
    """SQLite-backed ``(model, flags, query) -> events`` cache with TTL and size cap.

    ``path=":memory:"`` gives a process-local cache; ``path=None`` uses
    ``~/.cache/baidu_reverse/responses.sqlite``.
    """

    def __init__(self, path: Optional[str] = None, ttl: float = 3600.0, max_entries: int = 5000):
        if path is None:
            from .paths import cache_dir

            path = str(cache_dir() / "responses.sqlite")
        self.path = path
        self.ttl = ttl
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "key TEXT PRIMARY KEY, created REAL NOT NULL, last_hit REAL NOT NULL, "
            "hits INTEGER NOT NULL DEFAULT 0, model TEXT, value TEXT NOT NULL)"
        )
        self._db.commit()

    @staticmethod
    def make_key(model: str, deep_search: bool, internet_search: bool, query: str) -> str:
        blob = json.dumps([model, bool(deep_search), bool(internet_search), query], ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[list[dict]]:
        now = time.time()
        with self._lock:
            row = self._db.execute("SELECT created, value FROM responses WHERE key=?", (key,)).fetchone()
            if row is None:
                self.misses += 1
                return None
            created, value = row
            if self.ttl and now - created > self.ttl:
                self._db.execute("DELETE FROM responses WHERE key=?", (key,))
                self._db.commit()
                self.misses += 1
                return None
            self._db.execute("UPDATE responses SET hits=hits+1, last_hit=? WHERE key=?", (now, key))
            self._db.commit()
        self.hits += 1
        return json.loads(value)

    def put(self, key: str, events: list[dict], model: str = "") -> None:
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT INTO responses(key, created, last_hit, hits, model, value) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET created=excluded.created, last_hit=excluded.last_hit, value=excluded.value",
                (key, now, now, 0, model, json.dumps(events, ensure_ascii=False)),
            )
            (count,) = self._db.execute("SELECT COUNT(*) FROM responses").fetchone()
            if count > self.max_entries:
                self._db.execute(
                    "DELETE FROM responses WHERE key IN (SELECT key FROM responses ORDER BY last_hit ASC LIMIT ?)",
                    (count - self.max_entries,),
                )
            self._db.commit()

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM responses")
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def stats(self) -> dict:
        return {"entries": len(self), "hits": self.hits, "misses": self.misses, "ttl": self.ttl}

    def close(self) -> None:
        with self._lock:
            self._db.close()
