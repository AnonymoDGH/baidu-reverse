import json

import pytest

from baidu_reverse import ChatSession, Conversation, SessionStore
from baidu_reverse.session import prefix_hashes

from conftest import ScriptedChat


def memory_replier(q, hist, sid):
    """A tiny 'assistant' that really remembers: answers 'name?' from the session history."""
    if q.startswith("SUMMARIZE_MARK") or "You maintain the long-term memory" in q:
        return "- user name is Ana\n- likes tea"
    import re

    if "name?" in q.lower():
        for past in hist:  # facts can come from a raw message or from a replayed summary
            m = re.search(r"\bname is (\w+)", past, re.I)
            if m:
                return f"Your name is {m.group(1).capitalize()}"
    return "ok"


@pytest.mark.asyncio
async def test_follow_ups_send_only_the_new_message():
    chat = ScriptedChat(memory_replier)
    s = ChatSession(chat, system="Be brief.")
    await s.send("My name is Ana")
    assert await s.send("what is my name?") == "Your name is Ana"
    assert chat.sent[0]["query"].startswith("Be brief.") and "My name is Ana" in chat.sent[0]["query"]
    assert chat.sent[1]["query"] == "what is my name?"          # no history resent
    assert chat.sent[0]["sid"] == chat.sent[1]["sid"]            # same server session
    assert [m["role"] for m in s.messages] == ["user", "assistant", "user", "assistant"]


@pytest.mark.asyncio
async def test_forgotten_server_session_is_rebuilt_from_the_local_transcript():
    chat = ScriptedChat(memory_replier)
    s = ChatSession(chat)
    await s.send("My name is Ana")
    chat.forget()  # the server loses the session
    assert await s.send("what is my name?") == "Your name is Ana"
    assert s.rebuilds == 1
    replay = chat.sent[-1]
    assert "My name is Ana" in replay["query"] and "what is my name?" in replay["query"]
    assert replay["sid"] != chat.sent[0]["sid"]


@pytest.mark.asyncio
async def test_empty_answers_are_not_recorded_in_the_transcript():
    chat = ScriptedChat(lambda q, h, s: "")
    s = ChatSession(chat)
    assert await s.send("hello") == ""
    assert s.messages == []


@pytest.mark.asyncio
async def test_compaction_folds_old_turns_into_a_summary_and_starts_a_fresh_session():
    chat = ScriptedChat(memory_replier)
    s = ChatSession(chat, system="SYS", max_context_chars=300, keep_recent_turns=2)
    await s.send("My name is Ana")
    for i in range(6):
        await s.send(f"filler message number {i} " + "z" * 60)
    sessions_before = {m["sid"] for m in chat.sent}
    # next turn triggers the rebuild (session got bigger than max_context_chars)
    out = await s.send("what is my name?")
    assert out == "Your name is Ana"                       # survived through the summary
    assert s.summary and "Ana" in s.summary
    assert s.summarized_upto > 0 and s.rebuilds >= 1
    last = chat.sent[-1]
    assert last["sid"] not in sessions_before              # brand-new server session
    assert "[Conversation memory" in last["query"] and "SYS" in last["query"]
    assert "filler message number 5" in last["query"]      # recent turns kept verbatim
    assert "filler message number 0" not in last["query"]  # old ones only via the summary
    # the summarisation call itself ran in its own isolated session on the cheap model
    summ = [c for c in chat.sent if "long-term memory" in c["query"]]
    assert summ and all(c["model"] == "deepseek-v4-flash" for c in summ)


@pytest.mark.asyncio
async def test_summary_failure_falls_back_to_a_heuristic():
    chat = ScriptedChat(lambda q, h, s: "ok")

    async def boom(*a, **k):
        raise RuntimeError("down")

    s = ChatSession(chat, keep_recent_turns=1)
    for i in range(4):
        await s.send(f"message {i}")
    chat.generate = boom
    assert await s.compact(force=True) is True
    assert "message 0" in s.summary


def test_persistence_roundtrip(tmp_path):
    chat = ScriptedChat(memory_replier)
    store = SessionStore(str(tmp_path / "sessions"))
    s = ChatSession(chat, model="smartmode", system="S", store=store)
    s.messages = [{"role": "user", "content": "My name is Ana"}, {"role": "assistant", "content": "ok"}]
    s.summary = "mem"
    s.summarized_upto = 0
    path = s.save()
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["id"] == s.id and data["model"] == "smartmode"

    back = store.load(chat, s.id)
    assert back.messages == s.messages and back.summary == "mem" and back.system == "S"
    assert back.conversation == Conversation()          # server session is NOT trusted after a reload
    assert [x["id"] for x in store.list()] == [s.id]
    assert store.delete(s.id) and store.list() == []


@pytest.mark.asyncio
async def test_resumed_session_replays_context_in_its_first_turn(tmp_path):
    chat = ScriptedChat(memory_replier)
    store = SessionStore(str(tmp_path))
    s = ChatSession(chat, store=store)
    await s.send("My name is Ana")
    chat2 = ScriptedChat(memory_replier)               # "next day": fresh process, server forgot everything
    back = ChatSession.load(chat2, s.id, store=store)
    assert await back.send("what is my name?") == "Your name is Ana"
    assert "My name is Ana" in chat2.sent[0]["query"]


@pytest.mark.asyncio
async def test_autosave_after_each_turn(tmp_path):
    chat = ScriptedChat(lambda q, h, s: "r")
    store = SessionStore(str(tmp_path))
    s = ChatSession(chat, store=store)
    await s.send("a")
    assert json.loads(store.path_for(s.id).read_text(encoding="utf-8"))["messages"][0]["content"] == "a"


def test_prefix_hashes_are_a_chain():
    a = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    b = a + [{"role": "user", "content": "z"}]
    ha, hb = prefix_hashes(a, "m"), prefix_hashes(b, "m")
    assert hb[:2] == ha and hb[2] != ha[1]
    assert prefix_hashes(a, "other")[0] != ha[0]       # salt (model/tools) separates caches
    # whitespace/ids don't change identity
    c = [{"role": "user", "content": "x "}, {"role": "assistant", "content": "y"}]
    assert prefix_hashes(c, "m") == ha
