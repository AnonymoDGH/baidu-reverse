import json

import pytest

from baidu_reverse import (
    BaiduChat,
    PrefixSessionCache,
    ToolStreamFilter,
    build_tool_system_prompt,
    extract_tool_calls,
    messages_to_prompt,
    parse_tool_calls,
)
from baidu_reverse import tool_calling as tc
from baidu_reverse.tool_calling import compact_messages

from conftest import ScriptedChat

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "default": 1},
                "metric": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["city"],
        },
    },
}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    },
}
XML = "<tool_calls><tool_call><name>get_weather</name><arguments>{\"city\":\"Bogota\"}</arguments></tool_call></tool_calls>"


# --------------------------------------------------------------------------- prompt building
def test_build_tool_system_prompt_is_compact_and_has_example():
    prompt = build_tool_system_prompt([WEATHER_TOOL])
    assert "<tool_calls>" in prompt and "get_weather" in prompt
    assert "city (string, required) - City name" in prompt
    assert "days (integer, optional)" in prompt and "[default: 1]" in prompt
    assert '<name>get_weather</name><arguments>{"city": "..."}</arguments>' in prompt  # example from the schema


def test_tool_choice_instructions():
    assert "MUST call the tool `get_weather`" in build_tool_system_prompt(
        [WEATHER_TOOL], {"type": "function", "function": {"name": "get_weather"}})
    assert "MUST call at least one tool" in build_tool_system_prompt([WEATHER_TOOL], "required")
    assert tc.wants_tool_call("required") and not tc.wants_tool_call("auto") and not tc.wants_tool_call(None)


def test_single_plain_user_message_is_sent_bare():
    assert messages_to_prompt([{"role": "user", "content": "hola"}]) == "hola"


def test_messages_to_prompt_system_first():
    prompt = messages_to_prompt([
        {"role": "system", "content": "Eres util"},
        {"role": "user", "content": "que hora es"},
    ])
    assert prompt.index("Eres util") < prompt.index("User: que hora es")


def test_messages_to_prompt_with_tools():
    prompt = messages_to_prompt([{"role": "user", "content": "clima en Madrid"}], tools=[WEATHER_TOOL])
    assert "System:" in prompt and "get_weather" in prompt and "User: clima en Madrid" in prompt


def test_history_renders_tool_protocol_consistently():
    msgs = [
        {"role": "user", "content": "clima?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Lima\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "22C"},
    ]
    prompt = messages_to_prompt(msgs, tools=[WEATHER_TOOL])
    # the assistant's past call is shown in the SAME xml dialect the model must produce
    assert "<tool_call><name>get_weather</name><arguments>{\"city\":\"Lima\"}</arguments></tool_call>" in prompt
    # tool results are labelled with the tool name (resolved from the call id)
    assert '<tool_result name="get_weather" id="call_1">\n22C\n</tool_result>' in prompt
    assert prompt.rstrip().endswith("give the final answer in plain text.")


def test_render_delta_only_new_messages():
    delta = [{"role": "tool", "tool_call_id": "c9", "content": "ok"}]
    out = tc.render_delta(delta, [WEATHER_TOOL], id_names={"c9": "get_weather"})
    assert '<tool_result name="get_weather" id="c9">' in out and "Tools available: get_weather" in out
    assert tc.render_delta([{"role": "user", "content": "hi"}]) == "hi"


def test_compact_messages_trims_and_keeps_system():
    messages = [{"role": "system", "content": "S"}] + [{"role": "user", "content": f"msg {i}"} for i in range(30)]
    out = compact_messages(messages, max_messages=5)
    assert len(out) == 6 and out[0]["role"] == "system"


def test_compact_messages_never_starts_with_orphan_tool_result():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
            {"role": "tool", "tool_call_id": "x", "content": "r"}, {"role": "user", "content": "c"}]
    out = compact_messages(msgs, max_messages=2)
    assert out[0]["role"] != "tool"


def test_long_messages_are_clipped_in_the_middle():
    long = "A" * 20000 + "B" * 20000
    out = tc.clip_middle(long, 1000)
    assert len(out) <= 1000 and out.startswith("A") and out.endswith("B") and "compressed" in out


# --------------------------------------------------------------------------- parsing
def test_parse_tool_calls_xml():
    content = "Voy a consultar.\n<tool_calls>\n  <tool_call><name>get_weather</name><arguments>{\"city\":\"Madrid\"}</arguments></tool_call>\n</tool_calls>"
    cleaned, calls = parse_tool_calls(content)
    assert "Voy a consultar" in cleaned and "<tool" not in cleaned
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Madrid"}
    assert calls[0]["id"].startswith("call_") and calls[0]["type"] == "function"


def test_parse_tool_calls_no_xml():
    assert parse_tool_calls("respuesta normal") == ("respuesta normal", [])


def test_parse_truncated_wrapper_and_arguments():
    cleaned, calls = parse_tool_calls('<tool_calls>\n<tool_call><name>get_weather</name><arguments>{"city":"Lima"}</arguments>\n</tool_call>\n')
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Lima"} and "<tool_calls" not in cleaned
    # cut right after the arguments, before </tool_call>
    _, calls = parse_tool_calls('<tool_call><name>get_weather</name><arguments>{"city":"Quito"}</arguments>')
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Quito"}


def test_parse_loose_without_wrapper_and_multiple():
    c = '<tool_call><name>a</name><arguments>{"x":1}</arguments></tool_call><tool_call><name>b</name><arguments>{}</arguments></tool_call>'
    _, calls = parse_tool_calls(c)
    assert [x["function"]["name"] for x in calls] == ["a", "b"]


def test_parse_markdown_fenced_and_cdata():
    c = '```xml\n<tool_calls><tool_call><name>get_weather</name><arguments><![CDATA[{"city":"Cusco"}]]></arguments></tool_call></tool_calls>\n```'
    cleaned, calls = parse_tool_calls(c)
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Cusco"}
    assert "```" not in cleaned


def test_parse_escaped_xml_and_unescaped_specials_in_args():
    esc = "&lt;tool_calls&gt;&lt;tool_call&gt;&lt;name&gt;search_web&lt;/name&gt;&lt;arguments&gt;{&quot;q&quot;:&quot;a &amp; b&quot;}&lt;/arguments&gt;&lt;/tool_call&gt;&lt;/tool_calls&gt;"
    _, calls = parse_tool_calls(esc)
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": "a & b"}
    # raw '<' and '&' inside a JSON string used to break the strict XML parser
    raw = '<tool_call><name>search_web</name><arguments>{"q": "1 < 2 && 3 > 2"}</arguments></tool_call>'
    _, calls = parse_tool_calls(raw)
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": "1 < 2 && 3 > 2"}


def test_parse_json_body_and_attribute_name_styles():
    _, calls = parse_tool_calls('<tool_call>{"name": "get_weather", "arguments": {"city": "Rio"}}</tool_call>')
    assert calls[0]["function"]["name"] == "get_weather"
    _, calls = parse_tool_calls('<tool_call name="get_weather">{"city": "Rio"}</tool_call>')
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Rio"}
    _, calls = parse_tool_calls('<invoke name="get_weather"><parameter name="city">Rio</parameter><parameter name="days">3</parameter></invoke>')
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Rio", "days": 3}


def test_parse_sloppy_json_arguments():
    for raw in ("{'city': 'Lima',}", '{"city": "Lima",}', "{\"city\": \"Lima\"} trailing words"):
        _, calls = parse_tool_calls(f"<tool_call><name>get_weather</name><arguments>{raw}</arguments></tool_call>")
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Lima"}, raw


def test_invalid_arguments_are_reported_not_silently_emptied():
    res = extract_tool_calls("<tool_call><name>get_weather</name><arguments>not json at all</arguments></tool_call>", [WEATHER_TOOL])
    assert res.calls == [] and res.found_markup and "not a valid JSON object" in res.errors[0]


def test_validation_against_schema():
    # fuzzy / namespaced names, string->int/bool/array coercion
    res = extract_tool_calls(
        '<tool_call><name>functions.Get_Weather</name><arguments>{"city":"Lima","days":"3","metric":"true","tags":"[\\"a\\"]"}</arguments></tool_call>',
        [WEATHER_TOOL])
    args = json.loads(res.calls[0]["function"]["arguments"])
    assert res.calls[0]["function"]["name"] == "get_weather"
    assert args == {"city": "Lima", "days": 3, "metric": True, "tags": ["a"]}
    # typo'd name is repaired
    res = extract_tool_calls('<tool_call><name>get_wether</name><arguments>{"city":"x"}</arguments></tool_call>', [WEATHER_TOOL])
    assert res.calls[0]["function"]["name"] == "get_weather"
    # unknown tool and missing required argument are rejected with a reason
    res = extract_tool_calls('<tool_call><name>delete_all</name><arguments>{}</arguments></tool_call>', [WEATHER_TOOL])
    assert res.calls == [] and "unknown tool" in res.errors[0]
    res = extract_tool_calls('<tool_call><name>get_weather</name><arguments>{"days": 2}</arguments></tool_call>', [WEATHER_TOOL])
    assert res.calls == [] and "missing required argument(s): city" in res.errors[0]


def test_json_only_replies_count_as_calls_when_tools_are_known():
    cases = [
        '{"name": "get_weather", "arguments": {"city": "Lima"}}',
        '```json\n{"tool_calls": [{"function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Lima\\"}"}}]}\n```',
        '[{"name": "get_weather", "parameters": {"city": "Lima"}}]',
    ]
    for c in cases:
        res = extract_tool_calls(c, [WEATHER_TOOL])
        assert [x["function"]["name"] for x in res.calls] == ["get_weather"], c
        assert res.text == ""
    # without a tool list a bare JSON object is NOT mistaken for a call
    assert parse_tool_calls('{"name": "get_weather", "arguments": {}}')[1] == []
    # and prose that merely mentions json is untouched
    assert extract_tool_calls('Use {"name": "x"} like this.', [WEATHER_TOOL]).calls == []


# --------------------------------------------------------------------------- stream filter
def _feed(chunks):
    f = ToolStreamFilter()
    out = [f.feed(c) for c in chunks]
    out.append(f.finish())
    return f, "".join(out)


def test_filter_streams_prose_and_holds_markup():
    text = f"Voy a buscar. {XML}"
    for size in (1, 2, 3, 7, 50):
        f, shown = _feed([text[i:i + size] for i in range(0, len(text), size)])
        assert shown == "Voy a buscar.", size
        assert f.triggered and f.held.strip().startswith("<tool_calls>")


def test_filter_never_swallows_normal_angle_brackets():
    for text in ("a < b and <b>bold</b> done", "List<String> x = new ArrayList<>();", "5 <3"):
        f, shown = _feed([text[i:i + 2] for i in range(0, len(text), 2)])
        assert shown == text and not f.holding


def test_filter_buffers_bare_json_and_fences_but_not_code_blocks():
    f, shown = _feed(['{"name": "get_', 'weather", "arguments": {}}'])
    assert shown == "" and f.buffer_all
    f, shown = _feed(["```js", "on\n{}\n```"])
    assert shown == "" and f.buffer_all
    f, shown = _feed(["```pyth", "on\nprint(1)\n```"])
    assert shown == "```python\nprint(1)\n```" and not f.holding


# --------------------------------------------------------------------------- chat_completions
@pytest.mark.asyncio
async def test_chat_completions_emits_tool_calls_and_reasoning():
    chat = ScriptedChat(lambda q, h, s: "Voy a buscar. " + XML)
    chat.thinking = "razonando"
    got = [c async for c in chat.chat_completions([{"role": "user", "content": "clima"}], tools=[WEATHER_TOOL])]
    types = [g["type"] for g in got]
    assert "reasoning_content" in types and "tool_calls" in types and types[-1] == "done"
    assert "".join(g["content"] for g in got if g["type"] == "content") == "Voy a buscar."
    calls = next(g for g in got if g["type"] == "tool_calls")["tool_calls"]
    assert calls[0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_prose_streams_live_before_the_xml_is_complete():
    chat = ScriptedChat(lambda q, h, s: "Un momento por favor. " + XML)
    order = []
    async for c in chat.chat_completions([{"role": "user", "content": "x"}], tools=[WEATHER_TOOL]):
        order.append(c["type"])
    assert order.index("content") < order.index("tool_calls")
    assert order.count("content") >= 1


@pytest.mark.asyncio
async def test_chat_completions_plain_text(echo_chat):
    got = [c async for c in echo_chat.chat_completions([{"role": "user", "content": "saluda"}])]
    assert "".join(g["content"] for g in got if g["type"] == "content") == "echo:saluda"
    assert got[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_required_tool_choice_retries_once_with_a_correction():
    replies = iter(["I think the weather is nice.", XML])
    chat = ScriptedChat(lambda q, h, s: next(replies))
    out = await chat.complete([{"role": "user", "content": "weather in Bogota"}], tools=[WEATHER_TOOL], tool_choice="required")
    assert out["tool_calls"][0]["function"]["name"] == "get_weather"
    assert len(chat.sent) == 2
    retry = chat.sent[1]
    assert retry["sid"] == chat.sent[0]["sid"] and "could not be used as a tool call" in retry["query"]
    assert len(retry["query"]) < 600  # a correction, not a replay of the whole prompt


@pytest.mark.asyncio
async def test_malformed_call_is_corrected_with_the_validation_error():
    replies = iter(['<tool_call><name>get_weather</name><arguments>{"days": 2}</arguments></tool_call>', XML])
    chat = ScriptedChat(lambda q, h, s: next(replies))
    out = await chat.complete([{"role": "user", "content": "w"}], tools=[WEATHER_TOOL])
    assert "missing required argument(s): city" in chat.sent[1]["query"]
    assert json.loads(out["tool_calls"][0]["function"]["arguments"]) == {"city": "Bogota"}


@pytest.mark.asyncio
async def test_gives_up_after_retries_and_returns_text():
    chat = ScriptedChat(lambda q, h, s: "no tools for me")
    out = await chat.complete([{"role": "user", "content": "w"}], tools=[WEATHER_TOOL], tool_choice="required", tool_retries=1)
    assert out["tool_calls"] is None and out["content"] == "no tools for me"
    assert len(chat.sent) == 2


@pytest.mark.asyncio
async def test_tool_choice_none_disables_tools():
    chat = ScriptedChat(lambda q, h, s: "plain")
    await chat.complete([{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL], tool_choice="none")
    assert "get_weather" not in chat.sent[0]["query"]


# --------------------------------------------------------------------------- session cache (delta sends)
def _conversation(chat, cache, messages, **kw):
    return chat.complete(messages, session_cache=cache, **kw)


@pytest.mark.asyncio
async def test_prefix_cache_sends_only_the_delta_of_follow_ups():
    chat = ScriptedChat(lambda q, h, s: f"r{len(h)}")
    cache = PrefixSessionCache()
    msgs = [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "first question " + "x" * 3000}]
    a = await _conversation(chat, cache, msgs)
    msgs += [{"role": "assistant", "content": a["content"]}, {"role": "user", "content": "second"}]
    b = await _conversation(chat, cache, msgs)
    msgs += [{"role": "assistant", "content": b["content"]}, {"role": "user", "content": "third"}]
    await _conversation(chat, cache, msgs)

    assert [s["sid"] for s in chat.sent] == ["sess1"] * 3          # one live server session
    assert len(chat.sent[0]["query"]) > 3000                        # first turn carries everything
    assert chat.sent[1]["query"] == "second" and chat.sent[2]["query"] == "third"  # then only deltas
    assert [s["turn"] for s in chat.sent] == [0, 1, 2]
    assert cache.hits == 2


@pytest.mark.asyncio
async def test_prefix_cache_miss_on_edited_history_starts_a_new_session():
    chat = ScriptedChat(lambda q, h, s: "ok")
    cache = PrefixSessionCache()
    m = [{"role": "user", "content": "q1"}]
    a = await _conversation(chat, cache, m)
    edited = [{"role": "user", "content": "q1 (edited)"}, {"role": "assistant", "content": a["content"]},
              {"role": "user", "content": "q2"}]
    await _conversation(chat, cache, edited)
    assert chat.sent[1]["sid"] != chat.sent[0]["sid"]
    assert "q1 (edited)" in chat.sent[1]["query"]  # full history re-sent


@pytest.mark.asyncio
async def test_prefix_cache_with_tool_round_trip_sends_tool_result_only():
    replies = iter([XML, "It is 22C in Bogota."])
    chat = ScriptedChat(lambda q, h, s: next(replies))
    cache = PrefixSessionCache()
    msgs = [{"role": "user", "content": "weather in Bogota?"}]
    first = await _conversation(chat, cache, msgs, tools=[WEATHER_TOOL])
    call = first["tool_calls"][0]
    msgs += [{"role": "assistant", "content": first["content"], "tool_calls": first["tool_calls"]},
             {"role": "tool", "tool_call_id": call["id"], "content": "22C"}]
    second = await _conversation(chat, cache, msgs, tools=[WEATHER_TOOL])
    assert second["content"] == "It is 22C in Bogota."
    assert chat.sent[1]["sid"] == chat.sent[0]["sid"]
    q2 = chat.sent[1]["query"]
    assert '<tool_result name="get_weather"' in q2 and "weather in Bogota?" not in q2


@pytest.mark.asyncio
async def test_prefix_cache_rebuilds_when_server_session_gets_too_big():
    chat = ScriptedChat(lambda q, h, s: "y" * 50)
    cache = PrefixSessionCache(max_session_chars=300)
    msgs = [{"role": "user", "content": "a" * 400}]
    a = await _conversation(chat, cache, msgs)
    msgs += [{"role": "assistant", "content": a["content"]}, {"role": "user", "content": "next"}]
    await _conversation(chat, cache, msgs)
    assert chat.sent[1]["sid"] != chat.sent[0]["sid"]  # over budget -> fresh session with the full history
    assert "a" * 400 in chat.sent[1]["query"]
