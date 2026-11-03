"""Shared test helpers: a scripted fake of Baidu's server (no network)."""

from __future__ import annotations

from typing import Callable, Optional

import pytest

from baidu_reverse import BaiduChat


def msg_event(text: str, sid: str, end: bool = False, extra_content: Optional[dict] = None) -> dict:
    content: dict = {"generator": {"component": "markdown-yiyan", "data": {"value": text}}} if text else {}
    if extra_content:
        content.update(extra_content)
    return {
        "event": "message",
        "data": {
            "sessionId": sid,
            "data": {
                "message": {
                    "content": content,
                    "metaData": {"state": "generate-complete" if end else "generating-resp", "endTurn": end},
                }
            },
        },
    }


class ScriptedChat(BaiduChat):
    """A BaiduChat whose ``chat()`` is answered by ``replier(query, history, sid)``.

    Simulates the server-side session memory: every session id has a list of the
    queries it received; ``forget()`` wipes them so the next follow-up is "unknown".
    """

    def __init__(self, replier: Callable[[str, list, str], str], **kw):
        super().__init__(min_interval=0, **kw)
        self.replier = replier
        self.sessions: dict[str, list[str]] = {}
        self.sent: list[dict] = []
        self.block_next = 0  # number of upcoming calls answered with the kunlun block
        self.thinking = ""
        self._n = 0

    def forget(self) -> None:
        self.sessions.clear()

    async def ready(self) -> None:  # no network
        self._token, self._lid = "t", "l"

    async def chat(self, query, model="deepseek-v4-pro", deep_search=False, internet_search=False,
                   rank=None, conversation=None):
        known = conversation is not None and conversation.session_id in self.sessions
        sid = conversation.session_id if conversation is not None and conversation.session_id else ""
        if not sid:
            self._n += 1
            sid = f"sess{self._n}"
            if conversation is not None:
                conversation.session_id = sid
        self.sent.append(
            {
                "sid": sid,
                "query": query,
                "turn": conversation.turn if conversation is not None else 0,
                "model": model,
                "known": known,
            }
        )
        yield {"event": "basedata", "data": {"status": 0, **({"chatHitKunlun": "kunlun_popup"} if self.block_next else {})}}
        if self.block_next:
            self.block_next -= 1
            yield msg_event("", sid, end=True, extra_content={"hints": {"parts": [{"type": "kunlunHint", "text": "oops"}]}})
            return
        # a follow-up to a session the server does not know -> it "forgot"
        if conversation is not None and conversation.turn > 0 and not known:
            yield msg_event("", sid, end=True)
            return
        hist = self.sessions.setdefault(sid, [])
        hist.append(query)
        if self.thinking:
            yield {
                "event": "message",
                "data": {
                    "sessionId": sid,
                    "data": {"message": {"content": {"generator": {"component": "thinkingSteps",
                             "data": {"reasoningContentArr": [self.thinking]}}}, "metaData": {"state": "generating-resp"}}},
                },
            }
        text = self.replier(query, list(hist), sid)
        for i in range(0, len(text), 5):
            yield msg_event(text[i : i + 5], sid)
        yield msg_event("", sid, end=True)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Never touch the user's real cookie file / caches during tests."""
    monkeypatch.setenv("BAIDU_REVERSE_HOME", str(tmp_path / "home"))


@pytest.fixture
def echo_chat():
    return ScriptedChat(lambda q, h, sid: f"echo:{q}")
