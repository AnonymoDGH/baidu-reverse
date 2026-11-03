import asyncio
import base64
import hashlib
import json
import time
import urllib.parse
import warnings

import pytest

from baidu_reverse import (
    BaiduBlockedError,
    BaiduChat,
    Conversation,
    DEFAULT_MODEL,
    MODEL_ALIASES,
    MODELS,
    ResponseCache,
    public_models,
)
from baidu_reverse import client as client_mod
from baidu_reverse.models import resolve_model

from conftest import ScriptedChat, msg_event


# --------------------------------------------------------------------------- models
def test_public_exports():
    assert callable(BaiduChat)
    assert DEFAULT_MODEL == "deepseek-v4-pro"
    assert set(MODELS) == {"smartmode", "ernie-5.1", "deepseek-v4-pro", "deepseek-v4-flash"}
    assert "deepseek-r1" not in MODELS  # removed upstream
    assert callable(public_models)


def test_model_aliases_consistent():
    assert MODEL_ALIASES["wenxin"] == "ernie-5.1"
    assert MODEL_ALIASES["smart"] == "smartmode"
    assert MODEL_ALIASES["flash"] == "deepseek-v4-flash"
    # the removed upstream models point at a live replacement
    assert MODEL_ALIASES["deepseek-r1"] == "deepseek-v4-pro"
    assert MODEL_ALIASES["deepseek"] == "deepseek-v4-pro"


def test_public_models_include_variants():
    ids = public_models()
    for needed in ("smartmode", "smartmode-thinking", "ernie-5.1-thinking", "deepseek-v4-pro-nothinking",
                   "deepseek-v4-flash-nothinking"):
        assert needed in ids


def test_resolve_model_fallback_and_suffix():
    with pytest.warns(RuntimeWarning):
        assert client_mod._resolve_model("unknown-thing") == "deepseek-v4-pro"
    assert client_mod._resolve_model("ernine-5.1") == "ernie-5.1"
    assert client_mod._resolve_model("smartMode") == "smartmode"
    # suffixes are stripped before the lookup (ernie-5.1-thinking used to fall back to the default!)
    assert resolve_model("ernie-5.1-thinking") == "ernie-5.1"
    assert resolve_model("DeepSeek-V4-Flash-NoThinking") == "deepseek-v4-flash"


def test_deprecated_models_warn():
    with pytest.warns(DeprecationWarning):
        assert resolve_model("deepseek-r1") == "deepseek-v4-pro"


def test_model_suffix_thinking():
    base = MODELS["deepseek-v4-flash"]
    no = client_mod._apply_model_suffix(base, "deepseek-v4-flash-nothinking")
    assert no["usedModel"]["modelFunction"]["thinkMode"] == "0"
    th = client_mod._apply_model_suffix(base, "deepseek-v4-flash-thinking")
    assert th["usedModel"]["modelFunction"]["thinkMode"] == "1"
    # the shared catalog is never mutated
    assert MODELS["deepseek-v4-flash"]["usedModel"]["modelFunction"]["thinkMode"] == "0"


# --------------------------------------------------------------------------- payload / headers
def _ready_chat(**kw):
    chat = BaiduChat(**kw)
    chat._token, chat._lid = "tok123", "lid456"
    return chat


def test_generate_chat_token_format():
    chat = _ready_chat()
    token = chat._generate_chat_token("hola")
    assert token.endswith("-lid456-3")
    decoded = base64.b64decode(token.rsplit("-", 2)[0]).decode()
    parts = decoded.split("|")
    assert parts[0] == "tok123"
    assert parts[1] == hashlib.md5(b"hola").hexdigest()
    assert parts[3] == "lid456"
    assert int(parts[2]) <= int(time.time() * 1000) + 1000


def test_new_calls_start_a_fresh_server_session():
    """Regression: every call used to share ori_lid, so independent calls leaked context."""
    chat = _ready_chat()
    p = chat._build_message_payload("pregunta", model="smartmode")
    msg = p["message"]
    assert msg["inputMethod"] == "chat_search"
    assert msg["query"][0]["data"]["text"]["query"] == "pregunta"
    assert msg["searchInfo"]["usedModel"]["modelName"] == "smartMode"
    assert msg["searchInfo"]["chatParams"]["chat_token"].endswith("-lid456-3")
    assert msg["searchInfo"]["ori_lid"] == ""
    assert p["rank"] == 1


def test_conversation_continues_server_session():
    chat = _ready_chat()
    conv = Conversation(session_id="777", turn=2)
    p = chat._build_message_payload("q", model="smartmode", conversation=conv)
    assert p["message"]["searchInfo"]["ori_lid"] == "777"
    assert p["rank"] == 3 and p["message"]["searchInfo"]["re_rank"] == "3"
    assert json.loads(p["message"]["content"]["agentInfo"]["params"])["agt_rk"] == 3


def test_deep_search_flag():
    p = _ready_chat()._build_message_payload("q", model="smartmode", deep_search=True)
    assert p["message"]["searchInfo"]["usedModel"]["modelFunction"]["deepSearch"] == "1"
    assert MODELS["smartmode"]["usedModel"]["modelFunction"]["deepSearch"] == "0"


def test_header_query_is_clipped():
    """A long X-Chat-Message header makes Baidu answer nothing (kunlun) - keep it short."""
    chat = _ready_chat()
    long_q = "x" * 5000
    h = chat._build_headers(long_q, "smartmode", 1)
    q_enc = h["X-Chat-Message"].split(",anti_ext:")[0][len("query:"):]
    assert len(q_enc) <= client_mod.HEADER_QUERY_MAX
    # CJK expands 9x when encoded: still bounded, and never cut mid-sequence
    h2 = chat._build_headers("你好" * 500, "smartmode", 1)
    q2 = h2["X-Chat-Message"].split(",anti_ext:")[0][len("query:"):]
    assert len(q2) <= client_mod.HEADER_QUERY_MAX
    assert urllib.parse.unquote(q2) == "你好" * (len(urllib.parse.unquote(q2)) // 2)
    # the JSON body keeps the full text
    p = chat._build_message_payload(long_q, model="smartmode")
    assert p["message"]["query"][0]["data"]["text"]["query"] == long_q


def test_build_headers_x_chat_message():
    h = _ready_chat()._build_headers("mi query", "deepseek-v4-flash", 1)
    assert h["X-Chat-Message"].startswith("query:")
    assert "modelName:DeepSeek-V4-Flash" in h["X-Chat-Message"]
    assert h["isDeepseek"] == "1" and h["source"] == "pc_csaitab"


def test_build_headers_smart_is_deepseek_flag_like_the_web():
    assert _ready_chat()._build_headers("q", "smartmode", 1)["isDeepseek"] == "1"
    assert _ready_chat()._build_headers("q", "ernie-5.1", 1)["isDeepseek"] == "0"


# --------------------------------------------------------------------------- cookies
def test_cookie_string_is_sent_verbatim_in_order():
    chat = BaiduChat(cookies="BAIDUID=abc; ZFY=xyz")
    assert chat._cookie_string() == "BAIDUID=abc; ZFY=xyz"
    assert chat._base_headers()["Cookie"] == "BAIDUID=abc; ZFY=xyz"


def test_cookie_file_roundtrip(tmp_path):
    f = tmp_path / "c.json"
    a = BaiduChat(cookies="A=1; B=2", cookie_file=str(f))
    a._cookies_dirty = True
    a._save_cookie_file()
    b = BaiduChat(cookie_file=str(f))
    assert b._cookie_string() == "A=1; B=2"
    assert BaiduChat(cookies="X=9", cookie_file=str(f))._cookie_string() == "X=9"  # explicit wins


def test_server_set_cookies_are_absorbed():
    chat = BaiduChat(cookies="A=1")

    class Resp:
        from http.cookies import SimpleCookie

        cookies = SimpleCookie("BA_HECTOR=zzz; Path=/")

    chat._absorb_cookies(Resp())
    assert chat._cookie_string() == "A=1; BA_HECTOR=zzz"


# --------------------------------------------------------------------------- SSE
@pytest.mark.asyncio
async def test_sse_iter_parses_events():
    raw = (
        'event:basedata\ndata:{"status":0,"lid":"l1"}\n\n'
        "event:ping\n\n"
        'event:message\ndata:{"data":{"message":{"content":{"generator":{"component":"markdown-yiyan",'
        '"data":{"value":"hola"}},"metaData":{"state":"generating-resp"}}}}}\n\n'
    )

    class FakeContent:
        async def __aiter__(self):
            yield raw.encode("utf-8")

    class FakeResp:
        content = FakeContent()

    events = [e async for e in BaiduChat()._iter_sse(FakeResp())]
    assert events[0]["event"] == "basedata" and events[0]["data"]["lid"] == "l1"
    assert events[1]["event"] == "message"
    assert events[1]["data"]["data"]["message"]["content"]["generator"]["component"] == "markdown-yiyan"


@pytest.mark.asyncio
async def test_sse_multibyte_split_across_chunks():
    raw = 'event:message\ndata:{"v":"你好世界"}\n\n'.encode("utf-8")

    class FakeContent:
        async def __aiter__(self):
            for i in range(0, len(raw), 3):  # cuts through UTF-8 sequences
                yield raw[i : i + 3]

    class FakeResp:
        content = FakeContent()

    events = [e async for e in BaiduChat()._iter_sse(FakeResp())]
    assert events[0]["data"]["v"] == "你好世界"


@pytest.mark.asyncio
async def test_sse_stall_guard_ends_the_stream():
    class FakeContent:
        async def __aiter__(self):
            yield b'event:ping\ndata:{"a":1}\n\n'
            await asyncio.sleep(5)  # server goes silent
            yield b'event:ping\ndata:{"a":2}\n\n'

    class FakeResp:
        content = FakeContent()

    t0 = time.time()
    events = [e async for e in BaiduChat(stall_seconds=0.2)._iter_sse(FakeResp())]
    assert len(events) == 1 and time.time() - t0 < 2


# --------------------------------------------------------------------------- normalisation / stream
@pytest.mark.asyncio
async def test_stream_normalizes_markdown_and_thinking():
    async def fake_chat(query, model, deep_search, internet_search, rank=None):
        yield {"event": "basedata", "data": {"status": 0, "lid": "l1"}}
        yield {"event": "message", "data": {"data": {"message": {
            "content": {"generator": {"component": "thinkingSteps", "data": {"reasoningContentArr": ["pensando ", "mas"]}}},
            "metaData": {"state": "generating-resp"}}}}}
        yield {"event": "message", "data": {"data": {"message": {
            "content": {"generator": {"component": "markdown-yiyan", "data": {"value": "respuesta"}}},
            "metaData": {"state": "generating-resp"}}}}}
        yield {"event": "message", "data": {"data": {"message": {"content": {}, "metaData": {"state": "generate-complete", "endTurn": True}}}}}

    chat = _ready_chat()
    chat.chat = fake_chat
    got = [c async for c in chat.stream("q")]
    assert [g["type"] for g in got] == ["metadata", "thinking", "text", "done"]
    assert "".join(g["content"] for g in got if g["type"] == "text") == "respuesta"
    assert "pensando mas" in "".join(g["content"] for g in got if g["type"] == "thinking")


@pytest.mark.asyncio
async def test_generate_concatenates_text():
    chat = _ready_chat()

    async def fake_stream(q, model, deep_search, internet_search):
        yield {"type": "thinking", "content": "razon"}
        yield {"type": "text", "content": "hola "}
        yield {"type": "text", "content": "mundo"}
        yield {"type": "done", "content": ""}

    chat.stream = fake_stream
    assert await chat.generate("q") == "hola mundo"


@pytest.mark.asyncio
async def test_stream_tracks_conversation(echo_chat):
    conv = Conversation()
    assert await echo_chat.generate("one", conversation=conv) == "echo:one"
    assert conv.session_id == "sess1" and conv.turn == 1
    assert await echo_chat.generate("two", conversation=conv) == "echo:two"
    assert conv.turn == 2
    assert [s["sid"] for s in echo_chat.sent] == ["sess1", "sess1"]  # same server session
    assert [s["turn"] for s in echo_chat.sent] == [0, 1]
    # without a conversation every call is its own fresh session
    await echo_chat.generate("three")
    await echo_chat.generate("four")
    assert echo_chat.sent[-1]["sid"] != echo_chat.sent[-2]["sid"]


# --------------------------------------------------------------------------- block / captcha recovery
@pytest.mark.asyncio
async def test_blocked_stream_raises(echo_chat):
    echo_chat.block_next = 1
    with pytest.raises(BaiduBlockedError) as ei:
        await echo_chat.generate("hi")
    assert "auto_solve" in str(ei.value) and ei.value.hint


@pytest.mark.asyncio
async def test_blocked_stream_does_not_leak_done(echo_chat):
    echo_chat.block_next = 1
    seen = []
    with pytest.raises(BaiduBlockedError):
        async for ev in echo_chat.stream("hi"):
            seen.append(ev["type"])
    assert "done" not in seen and "text" not in seen


@pytest.mark.asyncio
async def test_auto_solve_recovers_and_retries():
    chat = ScriptedChat(lambda q, h, s: "fine", auto_solve=True)
    chat.block_next = 1
    solved = []

    async def fake_solve():
        solved.append(1)
        chat._set_cookies("BAIDUID=fresh")
        return "BAIDUID=fresh"

    chat.solve_captcha = fake_solve
    assert await chat.generate("hi") == "fine"
    assert solved == [1]


@pytest.mark.asyncio
async def test_auto_solve_gives_up_after_max_retries():
    chat = ScriptedChat(lambda q, h, s: "fine", auto_solve=True, max_block_retries=2)
    chat.block_next = 10

    async def fake_solve():
        return ""

    chat.solve_captcha = fake_solve
    with pytest.raises(BaiduBlockedError):
        await chat.generate("hi")
    assert len(chat.sent) == 3  # first try + 2 retries


# --------------------------------------------------------------------------- response cache
@pytest.mark.asyncio
async def test_response_cache_hits_for_stateless_calls():
    chat = ScriptedChat(lambda q, h, s: f"answer to {q}", cache=ResponseCache(":memory:"))
    first = await chat.generate("same question", model="smartmode")
    second = await chat.generate("same question", model="smartmode")
    assert first == second == "answer to same question"
    assert len(chat.sent) == 1  # second one never reached the "server"
    await chat.generate("same question", model="deepseek-v4-flash")  # other model -> miss
    assert len(chat.sent) == 2
    assert chat.cache.stats()["hits"] == 1


@pytest.mark.asyncio
async def test_response_cache_never_used_inside_a_conversation():
    chat = ScriptedChat(lambda q, h, s: "x", cache=ResponseCache(":memory:"))
    for _ in range(2):
        await chat.generate("again", conversation=Conversation())
    assert len(chat.sent) == 2
    assert len(chat.cache) == 0


def test_response_cache_ttl_and_cap(tmp_path):
    c = ResponseCache(str(tmp_path / "r.sqlite"), ttl=0.05, max_entries=2)
    k = c.make_key("m", False, False, "q")
    c.put(k, [{"type": "text", "content": "a"}])
    assert c.get(k) == [{"type": "text", "content": "a"}]
    time.sleep(0.08)
    assert c.get(k) is None  # expired
    c2 = ResponseCache(":memory:", max_entries=2)
    for i in range(5):
        c2.put(f"k{i}", [{"type": "text", "content": str(i)}])
    assert len(c2) == 2
