"""Make Baidu models use tools: ``@tool`` + an :class:`Agent` loop.

Baidu has no function-calling API, so :class:`~baidu_reverse.BaiduChat` emulates
it (XML protocol, tolerant parser, schema validation, one corrective retry).
This module closes the loop::

    @tool
    def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
        '''Get the current weather.

        Args:
            city: City name, e.g. "Lima".
            unit: Temperature unit.
        '''
        return f"22 degrees {unit} in {city}"

    async with BaiduChat() as chat:
        agent = Agent(chat, [get_weather], model="deepseek-v4-pro")
        result = await agent.run("What's the weather in Lima?")
        print(result.text)

The agent keeps an OpenAI-style ``messages`` list; with the built-in
:class:`~baidu_reverse.session.PrefixSessionCache` every step after the first
sends only the new tool results (the server remembers the rest).
"""

from __future__ import annotations

import asyncio
import enum
import inspect
import json
import re
import time
import types
import typing
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional, Union

from .models import DEFAULT_MODEL
from .session import PrefixSessionCache
from .tool_calling import clip_middle


# ----------------------------------------------------------------------------
# schema from Python signatures
# ----------------------------------------------------------------------------
def _json_schema(tp: Any) -> dict:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp is inspect.Parameter.empty or tp is Any:
        return {}
    if origin is typing.Literal:
        vals = list(args)
        kind = {str: "string", int: "integer", float: "number", bool: "boolean"}.get(type(vals[0]), "string")
        return {"type": kind, "enum": vals}
    if origin in (Union, getattr(types, "UnionType", Union)):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _json_schema(non_none[0])
        return {"anyOf": [_json_schema(a) for a in non_none]}
    if origin in (list, set, frozenset, tuple) or tp in (list, set, frozenset, tuple):
        schema: dict = {"type": "array"}
        if args:
            schema["items"] = _json_schema(args[0])
        return schema
    if origin is dict or tp is dict:
        return {"type": "object"}
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return {"type": "string", "enum": [m.value for m in tp]}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}
    return {"type": "string"}


def _parse_docstring(doc: Optional[str]) -> tuple[str, dict]:
    """Google-style docstring -> (description, {param: description})."""
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    m = re.search(r"\n\s*(Args|Arguments|Parameters):", doc)
    head, rest = (doc[: m.start()], doc[m.start() :]) if m else (doc, "")
    params: dict[str, str] = {}
    current = None
    in_args = False
    for line in rest.splitlines():
        if re.match(r"^\s*(Args|Arguments|Parameters):\s*$", line):
            in_args = True
            continue
        if not in_args:
            continue
        if re.match(r"^\s*(Returns?|Raises?|Yields?|Examples?):", line):
            break
        m = re.match(r"^\s+(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
        if m:
            current = m.group(1)
            params[current] = m.group(2).strip()
        elif current and line.strip():
            params[current] += " " + line.strip()
    first_paragraph = re.split(r"\n\s*\n", head.strip(), maxsplit=1)[0]  # keep tool descriptions short
    return " ".join(first_paragraph.split()), params


@dataclass
class Tool:
    """A callable exposed to the model."""

    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]
    timeout: float = 60.0

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        timeout: float = 60.0,
    ) -> "Tool":
        sig = inspect.signature(func)
        try:
            hints = typing.get_type_hints(func)
        except Exception:
            hints = {}
        doc_desc, doc_params = _parse_docstring(func.__doc__)
        props: dict[str, dict] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            schema = _json_schema(hints.get(pname, p.annotation))
            if not schema:
                schema = {"type": "string"}
            if pname in doc_params:
                schema["description"] = doc_params[pname]
            if p.default is not inspect.Parameter.empty:
                if isinstance(p.default, (str, int, float, bool)):  # None = merely optional: no noise
                    schema["default"] = p.default
            else:
                required.append(pname)
            props[pname] = schema
        parameters: dict = {"type": "object", "properties": props}
        if required:
            parameters["required"] = required
        return cls(
            name=name or func.__name__,
            description=description or doc_desc or func.__name__,
            parameters=parameters,
            func=func,
            timeout=timeout,
        )

    async def run(self, arguments: dict) -> str:
        sig = inspect.signature(self.func)
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        kwargs = {k: v for k, v in arguments.items() if accepts_kwargs or k in sig.parameters}
        if inspect.iscoroutinefunction(self.func):
            result = await asyncio.wait_for(self.func(**kwargs), self.timeout)
        else:
            result = await asyncio.wait_for(asyncio.to_thread(self.func, **kwargs), self.timeout)
            if inspect.isawaitable(result):
                result = await result
        if isinstance(result, str):
            return result
        if result is None:
            return "ok"
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)


def tool(func: Optional[Callable[..., Any]] = None, *, name: Optional[str] = None,
         description: Optional[str] = None, timeout: float = 60.0):
    """Decorator turning a (sync or async) function into a :class:`Tool`.

    The JSON schema comes from the type hints and the Google-style docstring.
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool.from_function(f, name=name, description=description, timeout=timeout)

    return wrap(func) if func is not None else wrap


# ----------------------------------------------------------------------------
# Agent loop
# ----------------------------------------------------------------------------
@dataclass
class AgentResult:
    text: str
    reasoning: str = ""
    steps: list = field(default_factory=list)  # [{"name", "arguments", "result", "seconds"}]
    messages: list = field(default_factory=list)
    stopped: str = "final"  # final | max_steps | repeated_call


class Agent:
    """Run a model that can call Python tools until it gives a final answer.

    Args:
        chat: a :class:`~baidu_reverse.BaiduChat`.
        tools: :class:`Tool` objects, ``@tool`` functions, plain callables or raw
            OpenAI tool dicts (the latter need ``handlers``).
        handlers: ``{name: callable}`` for raw tool dicts.
        tool_choice: applied to the first step only (``"required"`` forces a call).
        max_steps: maximum model calls per :meth:`run`.
    """

    def __init__(
        self,
        chat: Any,
        tools: list,
        model: str = DEFAULT_MODEL,
        system: Optional[str] = None,
        *,
        handlers: Optional[dict] = None,
        max_steps: int = 8,
        deep_search: bool = False,
        internet_search: bool = False,
        tool_choice: Any = None,
        max_result_chars: int = 8000,
        session_cache: Optional[PrefixSessionCache] = None,
        tool_retries: int = 1,
    ):
        self.chat = chat
        self.model = model
        self.max_steps = max_steps
        self.deep_search = deep_search
        self.internet_search = internet_search
        self.tool_choice = tool_choice
        self.max_result_chars = max_result_chars
        self.tool_retries = tool_retries
        self.session_cache = session_cache if session_cache is not None else PrefixSessionCache()
        self.tools: dict[str, Tool] = {}
        self._specs: list[dict] = []
        for t in tools:
            if isinstance(t, Tool):
                tool_obj = t
            elif callable(t):
                tool_obj = Tool.from_function(t)
            elif isinstance(t, dict):
                fn = t.get("function", t)
                handler = (handlers or {}).get(fn.get("name"))
                if handler is None:
                    raise ValueError(f"no handler for tool {fn.get('name')!r}")
                tool_obj = Tool(fn["name"], fn.get("description", ""), fn.get("parameters", {}), handler)
            else:
                raise TypeError(f"unsupported tool: {t!r}")
            self.tools[tool_obj.name] = tool_obj
            self._specs.append(tool_obj.spec())
        self.messages: list[dict] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def reset(self) -> None:
        self.messages = [m for m in self.messages if m.get("role") == "system"]

    async def _execute(self, call: dict) -> tuple[str, dict, str, float]:
        fn = call["function"]
        name = fn["name"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        t0 = time.time()
        tool_obj = self.tools.get(name)
        if tool_obj is None:
            out = f"Error: unknown tool {name!r}. Available: {', '.join(self.tools)}"
        else:
            try:
                out = await tool_obj.run(args)
            except asyncio.TimeoutError:
                out = f"Error: tool {name!r} timed out after {tool_obj.timeout:.0f}s"
            except Exception as exc:  # the model gets to see and recover from it
                out = f"Error: {type(exc).__name__}: {exc}"
        return name, args, clip_middle(out, self.max_result_chars), time.time() - t0

    async def stream(self, user_message: str) -> AsyncIterator[dict]:
        """Yield events: ``reasoning``, ``content``, ``tool_call``, ``tool_result``, ``final``."""
        self.messages.append({"role": "user", "content": user_message})
        steps: list[dict] = []
        reasoning_all: list[str] = []
        seen: dict[str, int] = {}
        stopped = "max_steps"
        final_text = ""

        for step in range(self.max_steps):
            content: list[str] = []
            calls: list[dict] = []
            async for ch in self.chat.chat_completions(
                self.messages,
                model=self.model,
                tools=self._specs,
                tool_choice=self.tool_choice if step == 0 else None,
                deep_search=self.deep_search,
                internet_search=self.internet_search,
                session_cache=self.session_cache,
                tool_retries=self.tool_retries,
            ):
                ct = ch["type"]
                if ct == "content":
                    content.append(ch["content"])
                    yield {"type": "content", "content": ch["content"]}
                elif ct == "reasoning_content":
                    reasoning_all.append(ch["content"])
                    yield {"type": "reasoning", "content": ch["content"]}
                elif ct == "tool_calls":
                    calls.extend(ch["tool_calls"])

            assistant: dict = {"role": "assistant", "content": "".join(content).strip()}
            if calls:
                assistant["tool_calls"] = calls
            self.messages.append(assistant)

            if not calls:
                stopped, final_text = "final", assistant["content"]
                break

            for c in calls:
                yield {"type": "tool_call", "name": c["function"]["name"], "arguments": c["function"]["arguments"], "id": c["id"]}
            results = await asyncio.gather(*(self._execute(c) for c in calls))
            repeated = False
            for c, (name, args, out, secs) in zip(calls, results):
                sig = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                seen[sig] = seen.get(sig, 0) + 1
                if seen[sig] >= 3:
                    out += "\n(You already called this tool with identical arguments; use the earlier results and answer.)"
                if seen[sig] >= 5:
                    repeated = True
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "name": name, "content": out})
                steps.append({"name": name, "arguments": args, "result": out, "seconds": round(secs, 3)})
                yield {"type": "tool_result", "name": name, "id": c["id"], "content": out, "seconds": secs}
            if repeated:
                stopped = "repeated_call"
                break
        else:
            final_text = next(
                (m["content"] for m in reversed(self.messages) if m["role"] == "assistant" and m.get("content")), ""
            )

        yield {
            "type": "final",
            "result": AgentResult(
                text=final_text,
                reasoning="".join(reasoning_all),
                steps=steps,
                messages=list(self.messages),
                stopped=stopped,
            ),
        }

    async def run(self, user_message: str) -> AgentResult:
        result: Optional[AgentResult] = None
        async for ev in self.stream(user_message):
            if ev["type"] == "final":
                result = ev["result"]
        assert result is not None
        return result
