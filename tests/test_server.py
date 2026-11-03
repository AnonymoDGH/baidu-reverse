import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from baidu_reverse.server import create_app

from conftest import ScriptedChat

XML = '<tool_calls><tool_call><name>get_weather</name><arguments>{"city":"Lima"}</arguments></tool_call></tool_calls>'
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "w",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


async def make_client(replier, **app_kw):
    chat = ScriptedChat(replier)
    client = TestClient(TestServer(create_app(chat, **app_kw)))
    await client.start_server()
    return chat, client


@pytest.mark.asyncio
async def test_models_and_health():
    chat, c = await make_client(lambda q, h, s: "x")
    try:
        ids = [m["id"] for m in (await (await c.get("/v1/models")).json())["data"]]
        assert "deepseek-v4-pro" in ids and "ernie-5.1-thinking" in ids and "deepseek-r1" not in ids
        assert (await (await c.get("/health")).json())["status"] == "ok"
    finally:
        await c.close()


@pytest.mark.asyncio
async def test_non_streaming_completion():
    chat, c = await make_client(lambda q, h, s: "hola mundo")
    try:
        r = await c.post("/v1/chat/completions", json={"model": "smartmode", "messages": [{"role": "user", "content": "hi"}]})
        body = await r.json()
        assert r.status == 200 and body["object"] == "chat.completion" and body["model"] == "smartmode"
        ch = body["choices"][0]
        assert ch["message"] == {"role": "assistant", "content": "hola mundo"} and ch["finish_reason"] == "stop"
        assert body["usage"]["total_tokens"] > 0
    finally:
        await c.close()


@pytest.mark.asyncio
async def test_reasoning_and_streaming_chunks():
    chat = ScriptedChat(lambda q, h, s: "hello there")
    chat.thinking = "pondering"
    client = TestClient(TestServer(create_app(chat)))
    await client.start_server()
    try:
        r = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
        assert r.status == 200 and r.headers["Content-Type"].startswith("text/event-stream")
        raw = (await r.read()).decode()
        events = [json.loads(l[6:]) for l in raw.split("\n\n") if l.startswith("data: {")]
        assert raw.rstrip().endswith("data: [DONE]")
        deltas = [e["choices"][0]["delta"] for e in events]
        assert deltas[0] == {"role": "assistant", "content": ""}
        assert "".join(d.get("content", "") for d in deltas) == "hello there"
        assert "".join(d.get("reasoning_content", "") for d in deltas) == "pondering"
        assert events[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_calls_non_streaming_and_streaming():
    chat, c = await make_client(lambda q, h, s: XML)
    try:
        body = await (await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "w?"}], "tools": [TOOL]})).json()
        ch = body["choices"][0]
        assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] is None
        call = ch["message"]["tool_calls"][0]
        assert call["function"]["name"] == "get_weather" and json.loads(call["function"]["arguments"]) == {"city": "Lima"}

        r = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "w?"}], "tools": [TOOL], "stream": True})
        raw = (await r.read()).decode()
        events = [json.loads(l[6:]) for l in raw.split("\n\n") if l.startswith("data: {")]
        tc = [e["choices"][0]["delta"]["tool_calls"] for e in events if "tool_calls" in e["choices"][0]["delta"]]
        assert tc and tc[0][0]["index"] == 0 and tc[0][0]["function"]["name"] == "get_weather"
        assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
    finally:
        await c.close()


@pytest.mark.asyncio
async def test_long_chat_resending_history_only_costs_the_delta_upstream():
    chat, c = await make_client(lambda q, h, s: f"answer{len(h)}")
    try:
        msgs = [{"role": "user", "content": "q1 " + "x" * 2000}]
        for i in range(3):
            body = await (await c.post("/v1/chat/completions", json={"messages": msgs})).json()
            msgs += [{"role": "assistant", "content": body["choices"][0]["message"]["content"]},
                     {"role": "user", "content": f"q{i + 2}"}]
        assert len({s["sid"] for s in chat.sent}) == 1                 # one upstream session for the whole chat
        assert [len(s["query"]) > 2000 for s in chat.sent] == [True, False, False]
        assert (await (await c.get("/health")).json())["session_hits"] == 2
    finally:
        await c.close()


@pytest.mark.asyncio
async def test_validation_auth_and_upstream_errors():
    chat, c = await make_client(lambda q, h, s: "x", api_key="sekret")
    try:
        r = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status == 401
        hdr = {"Authorization": "Bearer sekret"}
        assert (await c.post("/v1/chat/completions", json={"messages": []}, headers=hdr)).status == 400
        assert (await c.post("/v1/chat/completions", data="not json", headers=hdr)).status == 400
        assert (await c.get("/health")).status == 200                  # health is not behind the key
        chat.block_next = 5
        r = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}, headers=hdr)
        assert r.status == 503 and (await r.json())["error"]["type"] == "baidu_blocked"
        r = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True}, headers=hdr)
        assert r.status == 503                                          # blocked is reported before streaming starts
    finally:
        await c.close()
