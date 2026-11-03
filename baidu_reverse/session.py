"""Conversation state and caches for long chats.

Baidu keeps the context of a conversation **server side**: the first request has
``ori_lid=""`` and the answer carries a ``sessionId``; follow-up requests send that
id as ``ori_lid`` and an incremented ``rank`` (turn number), and only the *new*
message. So a long chat never has to resend its history.

This module builds on that:

* :class:`Conversation` -- the tiny server-side handle (``session_id`` + ``turn``).
* :class:`PrefixSessionCache` -- maps an OpenAI-style ``messages`` list to a live
  server conversation (hash chain over the message prefixes), so stateless
  clients (an OpenAI-compatible server, an agent loop) automatically send only the
  *delta* of each request.
* :class:`ChatSession` -- a stateful chat with a local transcript, rolling-summary
  memory compaction (when the server context gets too big), automatic recovery if
  the server forgets the session, and JSON persistence (:class:`SessionStore`).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from .tool_calling import clip_middle, content_to_text


# ----------------------------------------------------------------------------
# Server-side conversation handle
# ----------------------------------------------------------------------------
@dataclass
class Conversation:
    """Handle of a Baidu server-side conversation.

    ``turn`` is the number of *completed* turns; the next request is turn
    ``turn + 1`` (that is the ``rank`` the web client sends).
    """

    session_id: str = ""
    turn: int = 0

    def clone(self) -> "Conversation":
        return Conversation(self.session_id, self.turn)

    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "turn": self.turn}

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        return cls(str(data.get("session_id", "")), int(data.get("turn", 0)))


# ----------------------------------------------------------------------------
# Prefix -> server conversation cache (for stateless OpenAI-style clients)
# ----------------------------------------------------------------------------
def canonical_message(msg: dict) -> str:
    """Stable text form of a message (ignores ids / whitespace noise)."""
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                pass
        calls.append([fn.get("name", ""), json.dumps(args, sort_keys=True, ensure_ascii=False)])
    return json.dumps(
        [msg.get("role", ""), content_to_text(msg.get("content")).strip(), calls],
        ensure_ascii=False,
    )


def prefix_hashes(messages: list, salt: str = "") -> list[str]:
    """``hashes[i]`` identifies ``messages[: i + 1]`` (a hash chain)."""
    out: list[str] = []
    h = hashlib.sha1(salt.encode("utf-8")).hexdigest()
    for m in messages:
        h = hashlib.sha1((h + canonical_message(m)).encode("utf-8")).hexdigest()
        out.append(h)
    return out


@dataclass
class SessionHit:
    conversation: Conversation
    consumed: int  # messages[:consumed] are already known to the server
    chars: int  # characters already sent in this server session


@dataclass
class _Entry:
    conversation: Conversation
    chars: int
    stamp: float = field(default_factory=time.time)


class PrefixSessionCache:
    """LRU + TTL map ``hash(messages prefix) -> live server conversation``.

    After answering ``messages`` the caller stores ``messages + [assistant_reply]``;
    when the client comes back with that history plus new messages, ``lookup``
    returns the conversation and how many messages are already known, so only the
    delta has to be sent. A miss simply means "send everything in a new session".
    """

    def __init__(self, max_entries: int = 256, ttl: float = 3600.0, max_session_chars: int = 150_000):
        self.max_entries = max_entries
        self.ttl = ttl
        self.max_session_chars = max_session_chars
        self._d: "OrderedDict[str, _Entry]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._d)

    def _purge(self) -> None:
        now = time.time()
        for k in [k for k, e in self._d.items() if now - e.stamp > self.ttl]:
            del self._d[k]
        while len(self._d) > self.max_entries:
            self._d.popitem(last=False)

    def lookup(self, messages: list, model: str = "", tools_sig: str = "") -> Optional[SessionHit]:
        self._purge()
        hs = prefix_hashes(messages, f"{model}|{tools_sig}")
        for k in range(len(messages) - 1, 0, -1):
            e = self._d.get(hs[k - 1])
            if e is None or messages[k - 1].get("role") != "assistant":
                continue
            if e.chars > self.max_session_chars:
                continue  # server context is getting big: rebuild a fresh session
            self._d.move_to_end(hs[k - 1])
            self.hits += 1
            return SessionHit(e.conversation.clone(), k, e.chars)
        self.misses += 1
        return None

    def store(self, messages: list, model: str, tools_sig: str, conversation: Conversation, chars: int) -> None:
        if not messages or not conversation.session_id:
            return
        key = prefix_hashes(messages, f"{model}|{tools_sig}")[-1]
        self._d[key] = _Entry(conversation.clone(), chars)
        self._d.move_to_end(key)
        self._purge()

    def clear(self) -> None:
        self._d.clear()


# ----------------------------------------------------------------------------
# Stateful chat with memory compaction
# ----------------------------------------------------------------------------
SUMMARY_PROMPT = """You maintain the long-term memory of an ongoing conversation. \
Update the memory below with the new messages.
Keep: facts about the user, names, numbers, decisions, code and identifiers, open tasks, \
preferences, and anything the assistant promised. Drop pleasantries and repetition.
Write in the same language as the conversation, as compact bullet points, at most {words} words.
Output ONLY the updated memory, nothing else.

[Current memory]
{memory}

[New messages]
{transcript}
"""


def _role_label(role: str) -> str:
    return {"user": "User", "assistant": "Assistant", "system": "System"}.get(role, role.capitalize())


def render_transcript(messages: list[dict], per_message_chars: int = 6000) -> str:
    lines = []
    for m in messages:
        text = clip_middle(m.get("content", ""), per_message_chars)
        lines.append(f"{_role_label(m.get('role', 'user'))}: {text}")
    return "\n".join(lines)


class ChatSession:
    """A multi-turn chat that stays fast and coherent however long it gets.

    * Sends only the new message each turn (server-side memory).
    * Keeps the full transcript locally (``messages``) and persists it.
    * When the server session grows past ``max_context_chars`` (or
      ``max_turns_per_session`` turns) the older part is folded into a rolling
      ``summary`` and a **fresh** server session is bootstrapped from
      ``system + summary + recent turns``.
    * If the server answers nothing (it forgot the session) the session is
      rebuilt from the local transcript and the turn is retried once.

    Example::

        async with BaiduChat() as chat:
            s = ChatSession(chat, model="deepseek-v4-flash", system="Answer briefly.")
            print(await s.send("My name is Ana"))
            print(await s.send("What is my name?"))
            s.save("ana.json")
    """

    def __init__(
        self,
        chat: Any,
        model: Optional[str] = None,
        system: Optional[str] = None,
        *,
        id: Optional[str] = None,
        store: Optional["SessionStore"] = None,
        deep_search: bool = False,
        internet_search: bool = False,
        max_context_chars: int = 60_000,
        keep_recent_turns: int = 6,
        max_turns_per_session: int = 40,
        summary_model: str = "deepseek-v4-flash",
        summary_words: int = 350,
        autosave: bool = True,
    ):
        from .models import DEFAULT_MODEL

        self.chat = chat
        self.model = model or DEFAULT_MODEL
        self.system = system
        self.id = id or uuid.uuid4().hex[:12]
        self.store = store
        self.deep_search = deep_search
        self.internet_search = internet_search
        self.max_context_chars = max_context_chars
        self.keep_recent_turns = keep_recent_turns
        self.max_turns_per_session = max_turns_per_session
        self.summary_model = summary_model
        self.summary_words = summary_words
        self.autosave = autosave

        self.messages: list[dict] = []
        self.summary: str = ""
        self.summarized_upto: int = 0
        self.conversation = Conversation()
        self.session_chars = 0
        self.rebuilds = 0
        self.created = time.time()

    # -- prompt composition ---------------------------------------------------
    def _bootstrap(self, text: str) -> str:
        parts: list[str] = []
        if self.system:
            parts.append(self.system.strip())
        recent = self.messages[self.summarized_upto :]
        if self.summary or recent:
            if self.summary:
                parts.append("[Conversation memory - summary of earlier messages]\n" + self.summary)
            if recent:
                budget = max(2000, int(self.max_context_chars * 0.6))
                parts.append("[Recent messages]\n" + clip_middle(render_transcript(recent), budget))
            parts.append("[Current message - continue the conversation naturally]\n" + text)
        else:
            parts.append(text)
        return "\n\n".join(parts)

    def _compose(self, text: str) -> str:
        if self.conversation.turn == 0:
            return self._bootstrap(text)
        return text

    def _reset_server_session(self) -> None:
        self.conversation = Conversation()
        self.session_chars = 0

    # -- memory compaction -----------------------------------------------------
    async def _summarize(self, memory: str, msgs: list[dict]) -> str:
        # fold in pieces so a huge backlog never exceeds what one request can hold
        pieces: list[list[dict]] = [[]]
        size = 0
        for m in msgs:
            n = min(len(m.get("content", "")), 6000)
            if size + n > 24_000 and pieces[-1]:
                pieces.append([])
                size = 0
            pieces[-1].append(m)
            size += n
        for piece in pieces:
            prompt = SUMMARY_PROMPT.format(
                words=self.summary_words,
                memory=memory.strip() or "(empty)",
                transcript=render_transcript(piece),
            )
            try:
                out = (await self.chat.generate(prompt, model=self.summary_model)).strip()
            except Exception:
                out = ""
            memory = out or self._fallback_summary(memory, piece)
        return memory

    @staticmethod
    def _fallback_summary(memory: str, msgs: list[dict]) -> str:
        bits = [memory] if memory else []
        for m in msgs:
            first = " ".join(m.get("content", "").split())[:160]
            if first:
                bits.append(f"- {_role_label(m.get('role', 'user'))}: {first}")
        return clip_middle("\n".join(bits), 4000)

    async def compact(self, force: bool = False) -> bool:
        """Fold older messages into ``summary``. Returns True if anything changed."""
        keep = max(0, self.keep_recent_turns * 2)
        target = max(self.summarized_upto, len(self.messages) - keep)
        if target <= self.summarized_upto:
            return False
        if not force and self.session_chars <= self.max_context_chars and (
            self.conversation.turn < self.max_turns_per_session
        ):
            return False
        chunk = self.messages[self.summarized_upto : target]
        self.summary = await self._summarize(self.summary, chunk)
        self.summarized_upto = target
        return True

    async def _maybe_rebuild(self) -> None:
        c = self.conversation
        if c.turn > 0 and (self.session_chars > self.max_context_chars or c.turn >= self.max_turns_per_session):
            await self.compact(force=True)
            self._reset_server_session()
            self.rebuilds += 1

    # -- chat -----------------------------------------------------------------
    async def stream(self, text: str) -> AsyncIterator[dict]:
        """Yield normalized events (``text`` / ``thinking`` / ``done`` ...) for one turn."""
        await self._maybe_rebuild()
        reply = ""
        prompt = ""
        for attempt in (0, 1):
            prompt = self._compose(text)
            parts: list[str] = []
            pending_done: Optional[dict] = None
            async for ev in self.chat.stream(
                prompt,
                self.model,
                self.deep_search,
                self.internet_search,
                conversation=self.conversation,
            ):
                if ev["type"] == "text":
                    parts.append(ev["content"])
                if ev["type"] == "done":
                    pending_done = ev
                    continue
                yield ev
            reply = "".join(parts)
            if reply.strip() or attempt == 1:
                if pending_done is not None:
                    yield pending_done
                break
            # empty answer: the server probably lost our session -> rebuild and retry
            self._reset_server_session()
            self.rebuilds += 1

        if reply.strip():
            self.messages.append({"role": "user", "content": text})
            self.messages.append({"role": "assistant", "content": reply})
            self.session_chars += len(prompt) + len(reply)
            if self.store is not None and self.autosave:
                self.store.save(self)

    async def send(self, text: str) -> str:
        parts: list[str] = []
        async for ev in self.stream(text):
            if ev["type"] == "text":
                parts.append(ev["content"])
        return "".join(parts)

    def reset(self) -> None:
        """Forget everything (local and server side)."""
        self.messages.clear()
        self.summary = ""
        self.summarized_upto = 0
        self._reset_server_session()

    # -- persistence ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "id": self.id,
            "model": self.model,
            "system": self.system,
            "deep_search": self.deep_search,
            "internet_search": self.internet_search,
            "summary": self.summary,
            "summarized_upto": self.summarized_upto,
            "messages": self.messages,
            "conversation": self.conversation.to_dict(),
            "session_chars": self.session_chars,
            "created": self.created,
            "updated": time.time(),
        }

    def save(self, path: Optional[str] = None) -> str:
        if path is None:
            if self.store is None:
                raise ValueError("no path given and no SessionStore attached")
            return str(self.store.save(self))
        _atomic_write(Path(path), json.dumps(self.to_dict(), ensure_ascii=False, indent=1))
        return str(path)

    @classmethod
    def from_dict(cls, chat: Any, data: dict, store: Optional["SessionStore"] = None, **kw: Any) -> "ChatSession":
        s = cls(
            chat,
            model=data.get("model"),
            system=data.get("system"),
            id=data.get("id"),
            store=store,
            deep_search=data.get("deep_search", False),
            internet_search=data.get("internet_search", False),
            **kw,
        )
        s.summary = data.get("summary", "")
        s.summarized_upto = int(data.get("summarized_upto", 0))
        s.messages = list(data.get("messages", []))
        s.created = data.get("created", s.created)
        # The server may have dropped the old session: start a fresh one and let
        # the first turn replay summary + recent messages (safe default).
        s.conversation = Conversation()
        return s

    @classmethod
    def load(cls, chat: Any, path_or_id: str, store: Optional["SessionStore"] = None, **kw: Any) -> "ChatSession":
        p = Path(path_or_id)
        if not p.exists() and store is not None:
            p = store.path_for(path_or_id)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(chat, data, store=store, **kw)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class SessionStore:
    """Directory of ``<id>.json`` chat sessions."""

    def __init__(self, directory: Optional[str] = None):
        if directory is None:
            from .paths import cache_dir

            directory = str(cache_dir() / "sessions")
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def save(self, session: ChatSession) -> Path:
        p = self.path_for(session.id)
        _atomic_write(p, json.dumps(session.to_dict(), ensure_ascii=False, indent=1))
        return p

    def load(self, chat: Any, session_id: str, **kw: Any) -> ChatSession:
        return ChatSession.load(chat, session_id, store=self, **kw)

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            first = next((m["content"] for m in d.get("messages", []) if m.get("role") == "user"), "")
            out.append(
                {
                    "id": d.get("id", p.stem),
                    "model": d.get("model"),
                    "messages": len(d.get("messages", [])),
                    "title": " ".join(first.split())[:60],
                    "updated": d.get("updated"),
                }
            )
        return out

    def delete(self, session_id: str) -> bool:
        p = self.path_for(session_id)
        if p.exists():
            p.unlink()
            return True
        return False
