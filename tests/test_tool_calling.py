import json

import pytest

from baidu_reverse import BaiduChat, build_tool_system_prompt, messages_to_prompt, parse_tool_calls
from baidu_reverse.tool_calling import compact_messages

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_build_tool_system_prompt_contains_tool():
    prompt = build_tool_system_prompt([WEATHER_TOOL])
    assert "<tool_calls>" in prompt
    assert "get_weather" in prompt


def test_messages_to_prompt_basic():
    messages = [{"role": "user", "content": "hola"}]
    prompt = messages_to_prompt(messages)
    assert "User: hola" in prompt


def test_messages_to_prompt_system_first():
    messages = [
        {"role": "system", "content": "Eres util"},
        {"role": "user", "content": "que hora es"},
    ]
    prompt = messages_to_prompt(messages)
    assert prompt.index("Eres util") < prompt.index("User: que hora es")


def test_compact_messages_trims():
    messages = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
    out = compact_messages(messages, max_messages=5)
    assert len(out) == 5


def test_parse_tool_calls_xml():
    content = (
        "Voy a consultar el clima.\n"
        "<tool_calls>\n"
        "  <tool_call><name>get_weather</name><arguments>{\"city\":\"Madrid\"}</arguments></tool_call>\n"
        "</tool_calls>"
    )
    cleaned, calls = parse_tool_calls(content)
    assert "Voy a consultar" in cleaned
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Madrid"}


def test_parse_tool_calls_no_xml():
    cleaned, calls = parse_tool_calls("respuesta normal")
    assert cleaned == "respuesta normal"
    assert calls == []


def test_parse_tool_calls_truncated_wrapper():
    # el modelo corto la salida: falta </tool_calls> final
    content = (
        '<tool_calls>\n  <tool_call><name>get_weather</name>'
        '<arguments>{"city":"Lima"}</arguments>\n  </tool_call>\n'
    )
    cleaned, calls = parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Lima"}
    # el texto residuo XML se limpia
    assert "<tool_calls" not in cleaned


def test_parse_tool_calls_loose_without_wrapper():
    # tool_call individual sin el wrapper <tool_calls>
    content = '<tool_call><name>get_weather</name><arguments>{"city":"CDMX"}</arguments></tool_call>'
    cleaned, calls = parse_tool_calls(content)
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "CDMX"}


def test_messages_to_prompt_with_tools():
    messages = [{"role": "user", "content": "clima en Madrid"}]
    prompt = messages_to_prompt(messages, tools=[WEATHER_TOOL])
    assert "System:" in prompt
    assert "get_weather" in prompt
    assert "User: clima en Madrid" in prompt


@pytest.mark.asyncio
async def test_chat_completions_emits_tool_calls(monkeypatch):
    chat = BaiduChat()
    chat._token = "t"
    chat._lid = "l"

    xml = (
        "<tool_calls><tool_call><name>get_weather</name>"
        "<arguments>{\"city\":\"Bogota\"}</arguments></tool_call></tool_calls>"
    )

    async def fake_stream(query, model, deep_search, internet_search):
        yield {"type": "thinking", "content": "razonando"}
        yield {"type": "text", "content": "Voy a buscar. " + xml}
        yield {"type": "done", "content": ""}

    from unittest.mock import patch

    with patch.object(chat, "stream", fake_stream):
        got = [c async for c in chat.chat_completions(
            [{"role": "user", "content": "clima"}], tools=[WEATHER_TOOL]
        )]
    types = [g["type"] for g in got]
    assert "reasoning_content" in types
    assert "tool_calls" in types
    tool_ev = [g for g in got if g["type"] == "tool_calls"][0]
    assert tool_ev["tool_calls"][0]["function"]["name"] == "get_weather"
    assert got[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_chat_completions_plain_text(monkeypatch):
    chat = BaiduChat()
    chat._token = "t"
    chat._lid = "l"

    async def fake_stream(query, model, deep_search, internet_search):
        yield {"type": "text", "content": "hola "}
        yield {"type": "text", "content": "mundo"}
        yield {"type": "done", "content": ""}

    from unittest.mock import patch

    with patch.object(chat, "stream", fake_stream):
        got = [c async for c in chat.chat_completions([{"role": "user", "content": "saluda"}])]
    contents = "".join(g["content"] for g in got if g["type"] == "content")
    assert contents == "hola mundo"
    assert got[-1]["type"] == "done"
