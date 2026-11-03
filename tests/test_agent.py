import asyncio
import json
from typing import Literal, Optional

import pytest

from baidu_reverse import Agent, Tool, tool

from conftest import ScriptedChat


def call_xml(name, **args):
    return f"<tool_calls><tool_call><name>{name}</name><arguments>{json.dumps(args)}</arguments></tool_call></tool_calls>"


# --------------------------------------------------------------------------- schema
@tool
def get_weather(city: str, unit: Literal["c", "f"] = "c", days: int = 1, tags: Optional[list[str]] = None) -> str:
    """Get the current weather.

    Longer explanation that continues here.

    Args:
        city: City name, e.g. "Lima".
        unit: Temperature unit.
        days: How many days ahead.
        tags: Free tags.

    Returns:
        A sentence.
    """
    return f"22 {unit} in {city} (+{days}d)"


def test_schema_from_signature_and_docstring():
    assert isinstance(get_weather, Tool) and get_weather.name == "get_weather"
    assert get_weather.description == "Get the current weather."
    p = get_weather.parameters
    assert p["required"] == ["city"]
    assert p["properties"]["city"] == {"type": "string", "description": 'City name, e.g. "Lima".'}
    assert p["properties"]["unit"] == {"type": "string", "enum": ["c", "f"], "description": "Temperature unit.", "default": "c"}
    assert p["properties"]["days"]["type"] == "integer" and p["properties"]["days"]["default"] == 1
    assert p["properties"]["tags"] == {"type": "array", "items": {"type": "string"}, "description": "Free tags."}
    assert get_weather.spec()["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_tool_run_sync_async_and_result_formats():
    assert await get_weather.run({"city": "Lima", "bogus": 1}) == "22 c in Lima (+1d)"  # unknown args ignored

    @tool(name="slow", timeout=0.05)
    async def slow() -> str:
        await asyncio.sleep(1)
        return "never"

    with pytest.raises(asyncio.TimeoutError):
        await slow.run({})

    @tool
    def structured() -> dict:
        return {"a": [1, 2], "b": "ñ"}

    assert json.loads(await structured.run({})) == {"a": [1, 2], "b": "ñ"}

    @tool
    def nothing() -> None:
        return None

    assert await nothing.run({}) == "ok"


# --------------------------------------------------------------------------- agent loop
@pytest.mark.asyncio
async def test_agent_calls_the_tool_and_answers_with_the_result():
    replies = iter([call_xml("get_weather", city="Lima", days="2"), "It is 22 c in Lima."])
    chat = ScriptedChat(lambda q, h, s: next(replies))
    agent = Agent(chat, [get_weather], system="You are helpful.")
    result = await agent.run("weather in Lima?")
    assert result.text == "It is 22 c in Lima." and result.stopped == "final"
    assert result.steps[0]["name"] == "get_weather"
    assert result.steps[0]["arguments"] == {"city": "Lima", "days": 2}      # "2" coerced to int by the schema
    assert result.steps[0]["result"] == "22 c in Lima (+2d)"
    # second model call = same server session, only the tool result is sent
    assert chat.sent[0]["sid"] == chat.sent[1]["sid"]
    assert '<tool_result name="get_weather"' in chat.sent[1]["query"]
    assert "weather in Lima?" not in chat.sent[1]["query"]
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_agent_handles_parallel_calls_and_tool_errors():
    @tool
    def boom(x: int) -> str:
        """Always fails."""
        raise ValueError("kaput")

    both = ("<tool_calls>"
            "<tool_call><name>get_weather</name><arguments>{\"city\":\"A\"}</arguments></tool_call>"
            "<tool_call><name>boom</name><arguments>{\"x\":1}</arguments></tool_call>"
            "</tool_calls>")
    replies = iter([both, "done"])
    chat = ScriptedChat(lambda q, h, s: next(replies))
    result = await Agent(chat, [get_weather, boom]).run("go")
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert [m["name"] for m in tool_msgs] == ["get_weather", "boom"]
    assert tool_msgs[1]["content"] == "Error: ValueError: kaput"       # the model sees the error and can recover
    assert result.text == "done"


@pytest.mark.asyncio
async def test_agent_multi_step_chain_with_stream_events():
    replies = iter([call_xml("get_weather", city="A"), call_xml("get_weather", city="B"), "A and B are fine"])
    chat = ScriptedChat(lambda q, h, s: next(replies))
    agent = Agent(chat, [get_weather])
    kinds = [ev["type"] async for ev in agent.stream("compare A and B")]
    assert kinds.count("tool_call") == 2 and kinds.count("tool_result") == 2 and kinds[-1] == "final"
    assert len({s["sid"] for s in chat.sent}) == 1 and len(chat.sent) == 3


@pytest.mark.asyncio
async def test_agent_is_conversational_across_runs():
    replies = iter([call_xml("get_weather", city="A"), "first answer", "second answer"])
    chat = ScriptedChat(lambda q, h, s: next(replies))
    agent = Agent(chat, [get_weather])
    await agent.run("one")
    r = await agent.run("two")
    assert r.text == "second answer"
    assert chat.sent[-1]["query"].startswith("two") and chat.sent[-1]["sid"] == chat.sent[0]["sid"]


@pytest.mark.asyncio
async def test_agent_stops_a_model_stuck_calling_the_same_tool():
    chat = ScriptedChat(lambda q, h, s: call_xml("get_weather", city="A"))
    result = await Agent(chat, [get_weather], max_steps=20).run("loop")
    assert result.stopped == "repeated_call" and len(result.steps) == 5


@pytest.mark.asyncio
async def test_agent_max_steps():
    n = iter(range(100))
    chat = ScriptedChat(lambda q, h, s: call_xml("get_weather", city=f"c{next(n)}"))
    result = await Agent(chat, [get_weather], max_steps=3).run("x")
    assert result.stopped == "max_steps" and len(result.steps) == 3


@pytest.mark.asyncio
async def test_agent_raw_tool_dicts_need_handlers():
    spec = {"type": "function", "function": {"name": "ping", "description": "p", "parameters": {"type": "object", "properties": {}}}}
    with pytest.raises(ValueError):
        Agent(ScriptedChat(lambda *a: ""), [spec])
    replies = iter([call_xml("ping"), "pong!"])
    chat = ScriptedChat(lambda q, h, s: next(replies))
    result = await Agent(chat, [spec], handlers={"ping": lambda: "PONG"}).run("ping it")
    assert result.steps[0]["result"] == "PONG" and result.text == "pong!"
