import base64
import hashlib
import time

import pytest

from baidu_reverse import BaiduChat, DEFAULT_MODEL, MODEL_ALIASES, MODELS, public_models
from baidu_reverse import client as client_mod
from baidu_reverse.models import MODEL_ALIASES as MODELS_ALIASES


def test_public_exports():
    assert callable(BaiduChat)
    assert DEFAULT_MODEL == "deepseek-v4-pro"
    assert "ernie-5.1" in MODELS
    assert "deepseek-r1" in MODELS
    assert callable(public_models)


def test_model_aliases_consistent():
    assert MODELS_ALIASES is MODEL_ALIASES
    assert MODEL_ALIASES["wenxin"] == "ernie-5.1"
    assert MODEL_ALIASES["smart"] == "smartmode"
    assert MODEL_ALIASES["deepseek"] == "deepseek-r1"
    assert MODEL_ALIASES["flash"] == "deepseek-v4-flash"


def test_public_models_include_variants():
    ids = public_models()
    assert "smartmode" in ids
    assert "smartmode-thinking" in ids
    assert "deepseek-v4-pro-nothinking" in ids
    assert "deepseek-v4-flash-nothinking" in ids


def test_resolve_model_fallback():
    assert client_mod._resolve_model("unknown-thing") == "deepseek-v4-pro"
    assert client_mod._resolve_model("ernine-5.1") == "ernie-5.1"
    assert client_mod._resolve_model("smartMode") == "smartmode"


def test_model_suffix_thinking():
    base = MODELS["deepseek-v4-flash"]
    no = client_mod._apply_model_suffix(base, "deepseek-v4-flash-nothinking")
    assert no["usedModel"]["modelFunction"]["thinkMode"] == "0"
    th = client_mod._apply_model_suffix(base, "deepseek-v4-flash-thinking")
    assert th["usedModel"]["modelFunction"]["thinkMode"] == "1"


def test_generate_chat_token_format():
    chat = BaiduChat()
    chat._token = "tok123"
    chat._lid = "lid456"
    token = chat._generate_chat_token("hola")
    assert token.endswith("-lid456-3")
    b64 = token.rsplit("-", 2)[0]
    decoded = base64.b64decode(b64).decode()
    parts = decoded.split("|")
    assert parts[0] == "tok123"
    assert parts[1] == hashlib.md5(b"hola").hexdigest()
    assert parts[3] == "lid456"
    assert int(parts[2]) <= int(time.time() * 1000) + 1000


def test_build_message_payload_structure():
    chat = BaiduChat()
    chat._token = "t"
    chat._lid = "l"
    payload = chat._build_message_payload("pregunta", model="smartmode")
    msg = payload["message"]
    assert msg["inputMethod"] == "chat_search"
    assert msg["query"][0]["data"]["text"]["query"] == "pregunta"
    assert msg["searchInfo"]["usedModel"]["modelName"] == "smartMode"
    assert msg["searchInfo"]["chatParams"]["chat_token"].endswith("-l-3")
    assert payload["rank"] == 1


def test_deep_search_flag():
    chat = BaiduChat()
    chat._token = "t"
    chat._lid = "l"
    payload = chat._build_message_payload("q", model="smartmode", deep_search=True)
    assert payload["message"]["searchInfo"]["usedModel"]["modelFunction"]["deepSearch"] == "1"


def test_build_headers_x_chat_message():
    chat = BaiduChat()
    chat._token = "t"
    chat._lid = "l"
    headers = chat._build_headers("mi query", "deepseek-r1", 1)
    assert headers["X-Chat-Message"].startswith("query:")
    assert "modelName:DeepSeek-R1" in headers["X-Chat-Message"]
    assert headers["isDeepseek"] == "1"
    assert headers["source"] == "pc_csaitab"


def test_build_headers_smart_not_deepseek():
    chat = BaiduChat()
    chat._token = "t"
    chat._lid = "l"
    headers = chat._build_headers("q", "smartmode", 1)
    # the web sends isDeepseek=1 for smartMode too (verified in capture)
    assert headers["isDeepseek"] == "1"


@pytest.mark.asyncio
async def test_sse_iter_parses_events():
    raw = (
        "event:basedata\ndata:{\"status\":0,\"lid\":\"l1\"}\n\n"
        "event:ping\n\n"
        "event:message\ndata:{\"data\":{\"message\":{\"content\":{\"generator\":{\"component\":\"markdown-yiyan\",\"data\":{\"value\":\"hola\"}},\"metaData\":{\"state\":\"generating-resp\"}}}}}\n\n"
    )

    class FakeContent:
        async def __aiter__(self):
            for chunk in [raw.encode("utf-8")]:
                yield chunk

    class FakeResp:
        content = FakeContent()

    chat = BaiduChat()
    events = [e async for e in chat._iter_sse(FakeResp())]
    assert events[0]["event"] == "basedata"
    assert events[0]["data"]["lid"] == "l1"
    assert events[1]["event"] == "message"
    assert events[1]["data"]["data"]["message"]["content"]["generator"]["component"] == "markdown-yiyan"


@pytest.mark.asyncio
async def test_stream_normalizes_markdown_and_thinking():
    import json

    async def fake_chat(query, model, deep_search, internet_search, rank=None):
        yield {
            "event": "basedata",
            "data": {"status": 0, "lid": "l1"},
        }
        yield {
            "event": "message",
            "data": {
                "data": {
                    "message": {
                        "content": {
                            "generator": {
                                "component": "thinkingSteps",
                                "data": {"reasoningContentArr": ["pensando ", "mas"]},
                            }
                        },
                        "metaData": {"state": "generating-resp"},
                    }
                }
            },
        }
        yield {
            "event": "message",
            "data": {
                "data": {
                    "message": {
                        "content": {
                            "generator": {
                                "component": "markdown-yiyan",
                                "data": {"value": "respuesta"},
                            }
                        },
                        "metaData": {"state": "generating-resp"},
                    }
                }
            },
        }
        yield {
            "event": "message",
            "data": {
                "data": {
                    "message": {"content": {}, "metaData": {"state": "generate-complete", "endTurn": True}}
                }
            },
        }

    chat = BaiduChat()
    chat._token = "t"
    chat._lid = "l"

    from unittest.mock import patch

    with patch.object(chat, "chat", fake_chat):
        got = [c async for c in chat.stream("q")]
    types = [g["type"] for g in got]
    assert types == ["metadata", "thinking", "text", "done"]
    texts = [g["content"] for g in got if g["type"] == "text"]
    assert "".join(texts) == "respuesta"
    thinking = "".join(g["content"] for g in got if g["type"] == "thinking")
    assert "pensando mas" in thinking


@pytest.mark.asyncio
async def test_generate_concatenates_text():
    from unittest.mock import patch

    chat = BaiduChat()
    chat._token = "t"
    chat._lid = "l"

    async def fake_stream(q, model, deep_search, internet_search):
        yield {"type": "thinking", "content": "razon"}
        yield {"type": "text", "content": "hola "}
        yield {"type": "text", "content": "mundo"}
        yield {"type": "done", "content": ""}

    with patch.object(chat, "stream", fake_stream):
        result = await chat.generate("q")
    assert result == "hola mundo"


def test_cookie_injection():
    import asyncio

    chat = BaiduChat(cookies="BAIDUID=abc; ZFY=xyz")
    assert chat._cookie_string() == "BAIDUID=abc; ZFY=xyz"
    asyncio.run(chat._get_session())
    assert chat._cookie_jar is not None
    asyncio.run(chat.close())
