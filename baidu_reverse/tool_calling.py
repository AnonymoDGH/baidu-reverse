"""Tool-calling emulation for baidu-reverse.

Baidu Wenxin has no native function-calling API. We make models use tools by:

1. describing the tools in a compact prompt and asking for a strict
   ``<tool_calls>`` XML reply (:func:`build_tool_system_prompt`),
2. rendering OpenAI-style histories (assistant ``tool_calls`` and ``tool``
   results) in that same XML dialect so the model sees a consistent protocol
   (:func:`messages_to_prompt`, :func:`render_delta`),
3. parsing the reply back into OpenAI ``tool_calls`` with a tolerant parser that
   copes with truncation, fences, JSON bodies, ``<invoke>`` style, unescaped
   characters, fuzzy tool names and string-typed arguments, and validates
   calls against the declared JSON schema (:func:`extract_tool_calls`),
4. filtering a token stream so normal prose streams live while the XML is held
   back (:class:`ToolStreamFilter`).
"""

from __future__ import annotations

import ast
import difflib
import html
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

# ----------------------------------------------------------------------------
# small text helpers
# ----------------------------------------------------------------------------


def content_to_text(content: Any) -> str:
    """OpenAI ``content`` (str | list of parts | None) -> plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and item.get("type") in ("text", "input_text", "output_text"):
                texts.append(str(item.get("text", "")))
        return "\n".join(t for t in texts if t)
    return str(content)


def clip_middle(text: str, limit: int) -> str:
    """Shorten ``text`` to ``limit`` chars keeping head and tail."""
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 40:
        return text[:limit]
    marker = "\n...[compressed]...\n"
    head = (limit - len(marker)) // 2
    tail = limit - len(marker) - head
    return f"{text[:head]}{marker}{text[-tail:]}"


# kept for backwards compatibility
_content_to_text = content_to_text
_clip_middle = clip_middle


# ----------------------------------------------------------------------------
# tool descriptions
# ----------------------------------------------------------------------------


def _fn(tool: Any) -> dict:
    """Return the ``function`` dict of an OpenAI tool (also accepts flat dicts)."""
    if not isinstance(tool, dict):
        return {}
    fn = tool.get("function")
    return fn if isinstance(fn, dict) else tool


def tool_names(tools: Optional[list]) -> list[str]:
    return [n for n in (_fn(t).get("name", "") for t in tools or []) if n]


def tools_signature(tools: Optional[list]) -> str:
    """Stable fingerprint of a tool set (used as cache key)."""
    if not tools:
        return ""
    import hashlib

    blob = json.dumps([_fn(t) for t in tools], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _type_label(prop: dict) -> str:
    if not isinstance(prop, dict):
        return "any"
    if "enum" in prop:
        return " | ".join(json.dumps(v, ensure_ascii=False) for v in prop["enum"])
    t = prop.get("type", "any")
    if isinstance(t, list):
        t = "|".join(str(x) for x in t)
    if t == "array":
        items = prop.get("items")
        return f"array<{_type_label(items)}>" if items else "array"
    return str(t)


def summarize_tool(tool: Any) -> str:
    fn = _fn(tool)
    name = fn.get("name", "")
    desc = (fn.get("description") or "").strip().replace("\n", " ")
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    lines = [f"- {name}: {desc}" if desc else f"- {name}"]
    if not props:
        lines.append("    (no parameters)")
    for pname, p in props.items():
        p = p if isinstance(p, dict) else {}
        line = f"    {pname} ({_type_label(p)}, {'required' if pname in required else 'optional'})"
        if p.get("description"):
            line += f" - {str(p['description']).strip()}"
        if "default" in p:
            line += f" [default: {json.dumps(p['default'], ensure_ascii=False)}]"
        nested = p.get("type") == "object" and p.get("properties")
        nested = nested or (p.get("type") == "array" and isinstance(p.get("items"), dict) and p["items"].get("properties"))
        if nested:
            line += " schema=" + json.dumps(p, ensure_ascii=False, separators=(",", ":"))
        lines.append(line)
    return "\n".join(lines)


def _example_value(prop: dict) -> Any:
    if not isinstance(prop, dict):
        return "..."
    if "enum" in prop and prop["enum"]:
        return prop["enum"][0]
    if "default" in prop:
        return prop["default"]
    t = prop.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    return {"string": "...", "integer": 1, "number": 1.5, "boolean": True, "array": [], "object": {}}.get(t, "...")


def _example_call(tools: list) -> tuple[str, str]:
    for t in tools:
        fn = _fn(t)
        if not fn.get("name"):
            continue
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        req = [k for k in (params.get("required") or []) if k in props] or list(props)[:1]
        args = {k: _example_value(props[k]) for k in req}
        return fn["name"], json.dumps(args, ensure_ascii=False)
    return "tool_name", "{}"


def wants_tool_call(tool_choice: Any) -> bool:
    """True when ``tool_choice`` forces the model to call a tool."""
    if tool_choice in ("required", "any"):
        return True
    if isinstance(tool_choice, dict):
        name = _fn(tool_choice).get("name") or tool_choice.get("name")
        return bool(name)
    return False


def _format_tool_choice(tool_choice: Any) -> str:
    if not tool_choice or tool_choice == "auto":
        return ""
    if tool_choice == "none":
        return "Do not call any tool in your next reply; answer in plain text."
    if tool_choice in ("required", "any"):
        return "You MUST call at least one tool in your next reply."
    if isinstance(tool_choice, dict):
        name = _fn(tool_choice).get("name") or tool_choice.get("name") or ""
        if name:
            return f"You MUST call the tool `{name}` in your next reply."
    return ""


TOOL_SYSTEM_PROMPT = """# Tools
You can call tools to fetch information or perform actions. Available tools:

{tools}

# How to call tools
To call one or more tools, reply with ONLY this XML - no markdown fences, no text before or after it:
<tool_calls>
  <tool_call>
    <name>TOOL_NAME</name>
    <arguments>{{"param": "value"}}</arguments>
  </tool_call>
</tool_calls>

Rules:
- <arguments> is ONE valid JSON object using exactly the parameter names above (strings in double quotes).
- Use only the tool names listed. Never invent tools or parameters.
- To call several tools at once, put several <tool_call> blocks inside the same <tool_calls>.
- After calling tools, STOP and wait: results come back in the next message as <tool_result> blocks. Never guess or fabricate results.
- When you have what you need, or no tool is needed, answer normally in plain text with NO XML.
{tool_choice}
Example of a valid call:
<tool_calls><tool_call><name>{ex_name}</name><arguments>{ex_args}</arguments></tool_call></tool_calls>
"""


def build_tool_system_prompt(tools: list, tool_choice: Any = None) -> str:
    summaries = [summarize_tool(t) for t in tools if _fn(t).get("name")]
    ex_name, ex_args = _example_call(tools)
    choice = _format_tool_choice(tool_choice)
    return TOOL_SYSTEM_PROMPT.format(
        tools="\n".join(summaries),
        tool_choice=(choice + "\n") if choice else "",
        ex_name=ex_name,
        ex_args=ex_args,
    )


def tool_reminder(tools: list, tool_choice: Any = None) -> str:
    """Short reminder appended to follow-up turns of an already-primed session."""
    names = ", ".join(tool_names(tools))
    choice = _format_tool_choice(tool_choice)
    return (
        f"(Tools available: {names}. To call tools reply ONLY with the <tool_calls> XML exactly as before "
        f"(arguments = one JSON object); otherwise answer in plain text. {choice})".strip()
    )


# ----------------------------------------------------------------------------
# history -> prompt
# ----------------------------------------------------------------------------


def render_tool_calls_xml(tool_calls: list) -> str:
    blocks = []
    for tc in tool_calls or []:
        fn = _fn(tc) if isinstance(tc, dict) else {}
        name = fn.get("name", "")
        args = fn.get("arguments", "{}")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        blocks.append(f"  <tool_call><name>{name}</name><arguments>{args}</arguments></tool_call>")
    return "<tool_calls>\n" + "\n".join(blocks) + "\n</tool_calls>"


def render_tool_result(name: str, call_id: Optional[str], content: str) -> str:
    ident = f' id="{call_id}"' if call_id else ""
    return f'<tool_result name="{name}"{ident}>\n{content}\n</tool_result>'


def compact_messages(messages: list, max_messages: int = 60, max_message_chars: int = 12000) -> list:
    valid = [m for m in messages if isinstance(m, dict)]
    if not valid:
        return []
    system_messages = [m for m in valid if m.get("role") in ("system", "developer")]
    non_system = [m for m in valid if m.get("role") not in ("system", "developer")]
    recent = non_system[-max_messages:] if max_messages > 0 else non_system
    # never start the window with an orphan tool result
    while recent and recent[0].get("role") == "tool" and len(recent) > 1:
        recent = recent[1:]
    out = []
    for msg in system_messages + recent:
        copied = dict(msg)
        content = content_to_text(copied.get("content"))
        if max_message_chars > 0 and len(content) > max_message_chars:
            copied["content"] = clip_middle(content, max_message_chars)
        out.append(copied)
    return out


def _call_id_names(messages: list) -> dict:
    out = {}
    for m in messages:
        for tc in (m.get("tool_calls") or []) if isinstance(m, dict) else []:
            if isinstance(tc, dict) and tc.get("id"):
                out[tc["id"]] = _fn(tc).get("name", "")
    return out


def _render_message(msg: dict, id_names: dict) -> list[str]:
    role = msg.get("role", "user")
    content = content_to_text(msg.get("content"))
    if role == "tool":
        call_id = msg.get("tool_call_id")
        name = msg.get("name") or id_names.get(call_id) or "tool"
        return [render_tool_result(name, call_id, content)]
    if role == "assistant":
        out = []
        if content:
            out.append(f"Assistant: {content}")
        if msg.get("tool_calls"):
            out.append("Assistant: " + render_tool_calls_xml(msg["tool_calls"]))
        return out
    label = "System" if role == "developer" else role.capitalize()
    return [f"{label}: {content}"] if content else []


_AFTER_TOOL_RESULTS = (
    "Using the tool results above, either call more tools (reply with ONLY the <tool_calls> XML) "
    "or give the final answer in plain text."
)


def messages_to_prompt(
    messages: list,
    tools: Optional[list] = None,
    tool_choice: Any = None,
    max_chars: int = 40000,
    max_messages: int = 60,
    max_message_chars: int = 12000,
) -> str:
    """Render an OpenAI ``messages`` list as one Baidu query string."""
    if not isinstance(messages, list):
        return ""
    compacted = compact_messages(messages, max_messages=max_messages, max_message_chars=max_message_chars)
    non_system = [m for m in compacted if m.get("role") not in ("system", "developer")]
    has_system = any(m.get("role") in ("system", "developer") for m in compacted)

    # one plain user message, nothing else: send it bare (best answers, no framing)
    if not tools and not has_system and len(non_system) == 1 and non_system[0].get("role") == "user":
        text = content_to_text(non_system[0].get("content")).strip()
        return clip_middle(text, max_chars) if max_chars > 0 else text

    id_names = _call_id_names(messages)
    parts: list[str] = []
    if tools:
        parts.append(f"System: {build_tool_system_prompt(tools, tool_choice)}")
    for msg in compacted:
        parts.extend(_render_message(msg, id_names))
    if tools and compacted and compacted[-1].get("role") == "tool":
        parts.append(_AFTER_TOOL_RESULTS)
    prompt = "\n\n".join(p for p in parts if p).strip()
    if max_chars > 0 and len(prompt) > max_chars:
        prompt = clip_middle(prompt, max_chars)
    return prompt


def render_delta(messages: list, tools: Optional[list] = None, tool_choice: Any = None, id_names: Optional[dict] = None) -> str:
    """Render only NEW messages for a turn of an already-primed server session."""
    id_names = id_names or {}
    only_plain_user = len(messages) == 1 and messages[0].get("role") == "user"
    parts: list[str] = []
    if only_plain_user:
        parts.append(content_to_text(messages[0].get("content")).strip())  # bare, no framing
    else:
        for msg in messages:
            if msg.get("role") in ("system", "developer"):
                parts.append(f"System: {content_to_text(msg.get('content'))}")
            else:
                parts.extend(_render_message(msg, id_names))
    if tools:
        if messages and messages[-1].get("role") == "tool":
            parts.append(_AFTER_TOOL_RESULTS)
        parts.append(tool_reminder(tools, tool_choice))
    return "\n\n".join(p for p in parts if p).strip()


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------


@dataclass
class ParseResult:
    text: str
    calls: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    found_markup: bool = False  # tool markup was present (even if invalid)


_TOOL_CALL_RE = re.compile(r"<tool_call\b([^>]*)>(.*?)(?:</tool_call\s*>|(?=<tool_call\b)|\Z)", re.S | re.I)
_INVOKE_RE = re.compile(r"<invoke\b([^>]*)>(.*?)(?:</invoke\s*>|\Z)", re.S | re.I)
_FUNCTION_CALL_RE = re.compile(r"<function_call\b([^>]*)>(.*?)(?:</function_call\s*>|\Z)", re.S | re.I)
_WRAPPER_RE = re.compile(r"</?\s*(?:tool_calls|function_calls)\s*>", re.I)
_EMPTY_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*```")
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```\s*$", re.S)


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = _FENCE_RE.match(t)
    return m.group(1).strip() if m else t


def _strip_cdata(text: str) -> str:
    m = re.fullmatch(r"\s*<!\[CDATA\[(.*?)\]\]>\s*", text, re.S)
    return m.group(1) if m else text


def _first_balanced(s: str, open_c: str = "{", close_c: str = "}") -> Optional[str]:
    start = s.find(open_c)
    if start < 0:
        return None
    depth, in_str, esc, quote = 0, False, False, ""
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str, quote = True, ch
        elif ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _lenient_json(text: str) -> Any:
    """Parse JSON the way a model writes it. Returns None when hopeless."""
    t = _strip_cdata(_strip_fences(text or "")).strip()
    if not t:
        return None
    for cand in (t, html.unescape(t)):
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            pass
    obj = _first_balanced(t)
    if obj:
        for cand in (obj, re.sub(r",\s*([}\]])", r"\1", obj), html.unescape(obj)):
            try:
                return json.loads(cand)
            except (json.JSONDecodeError, ValueError):
                pass
        try:
            return ast.literal_eval(obj)
        except (ValueError, SyntaxError):
            pass
    return None


def _args_to_dict(raw: Optional[str]) -> Optional[dict]:
    """Arguments text -> dict (``None`` = unparsable)."""
    if raw is None or not raw.strip():
        return {}
    val = _lenient_json(raw)
    if isinstance(val, str):  # double-encoded JSON
        val = _lenient_json(val)
    return val if isinstance(val, dict) else None


def _attr(attrs: str, key: str) -> str:
    m = re.search(rf"""\b{key}\s*=\s*(?:"([^"]*)"|'([^']*)')""", attrs or "", re.I)
    return (m.group(1) or m.group(2) or "").strip() if m else ""


def _loose_value(text: str) -> Any:
    t = html.unescape(_strip_cdata(text)).strip()
    try:
        return json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return t


def _parse_block(attrs: str, body: str) -> tuple[str, Optional[dict]]:
    name = _attr(attrs, "name")
    m = re.search(r"<name\s*>(.*?)</name\s*>", body, re.S | re.I)
    if m:
        name = html.unescape(m.group(1)).strip()
    m = re.search(r"<(arguments|parameters|args|input)\s*>(.*?)(?:</\1\s*>|\Z)", body, re.S | re.I)
    if m:
        return name, _args_to_dict(m.group(2))
    params = re.findall(r"""<parameter\b[^>]*\bname\s*=\s*["']([^"']+)["'][^>]*>(.*?)</parameter\s*>""", body, re.S | re.I)
    if params:
        return name, {k: _loose_value(v) for k, v in params}
    stripped = _strip_fences(body).strip()
    if stripped.startswith("{"):
        obj = _lenient_json(stripped)
        if isinstance(obj, dict):
            fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
            if not name and fn.get("name"):
                name = str(fn["name"])
                a = fn.get("arguments", fn.get("parameters", {}))
                return name, _args_to_dict(a if isinstance(a, str) else json.dumps(a))
            if "arguments" in obj or "parameters" in obj:
                a = obj.get("arguments", obj.get("parameters"))
                return name, _args_to_dict(a if isinstance(a, str) else json.dumps(a))
            return name, obj
        return name, None
    return name, {}


def _match_name(name: str, names: list[str]) -> Optional[str]:
    if not name:
        return None
    if name in names:
        return name
    low = {n.lower(): n for n in names}
    if name.lower() in low:
        return low[name.lower()]
    short = re.split(r"[.:/]", name)[-1]
    if short in names:
        return short
    if short.lower() in low:
        return low[short.lower()]
    close = difflib.get_close_matches(short.lower(), list(low), n=1, cutoff=0.8)
    return low[close[0]] if close else None


def _coerce(value: Any, prop: dict) -> Any:
    if not isinstance(prop, dict):
        return value
    t = prop.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    try:
        if t == "integer":
            if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
                return int(value)
            if isinstance(value, float) and value.is_integer():
                return int(value)
        elif t == "number":
            if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                return float(value) if "." in value else int(value)
        elif t == "boolean":
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("true", "yes", "1"):
                    return True
                if v in ("false", "no", "0"):
                    return False
        elif t in ("array", "object") and isinstance(value, str):
            parsed = _lenient_json(value)
            if isinstance(parsed, list if t == "array" else dict):
                return parsed
        elif t == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    except (ValueError, TypeError):
        pass
    return value


def _build_call(name: str, args: dict) -> dict:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _json_fallback(content: str, names: list[str]) -> list[tuple[str, Any]]:
    """Calls written as bare JSON (no XML). Only when the whole reply is that JSON."""
    t = _strip_fences(content).strip()
    if not t or t[0] not in "{[":
        return []
    obj = _lenient_json(t)
    items: list = []
    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
        items = obj["tool_calls"]
    elif isinstance(obj, dict):
        items = [obj]
    elif isinstance(obj, list):
        items = obj
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        fn = it.get("function") if isinstance(it.get("function"), dict) else it
        name = fn.get("name") or fn.get("tool") or fn.get("tool_name")
        args = fn.get("arguments", fn.get("parameters", fn.get("args", {})))
        if name and _match_name(str(name), names):
            out.append((str(name), args if isinstance(args, str) else json.dumps(args)))
    return out


def extract_tool_calls(content: str, tools: Optional[list] = None) -> ParseResult:
    """Parse tool calls out of a model reply.

    With ``tools`` given, names are fuzzy-matched, arguments are coerced to the
    declared types and calls missing required arguments are rejected (reported
    in ``errors``). Without ``tools`` everything that parses is returned.
    """
    if not content:
        return ParseResult(text=content or "")
    work = content
    if re.search(r"&lt;\s*tool_call", work, re.I):  # escaped markup
        work = html.unescape(work)

    spans: list[tuple[int, int]] = []
    raw: list[tuple[str, Optional[dict]]] = []
    for rx in (_TOOL_CALL_RE, _INVOKE_RE, _FUNCTION_CALL_RE):
        for m in rx.finditer(work):
            raw.append(_parse_block(m.group(1), m.group(2)))
            spans.append(m.span())
    found_markup = bool(spans) or bool(_WRAPPER_RE.search(work))

    names = tool_names(tools)
    if not raw and tools:
        for name, args in _json_fallback(work, names):
            raw.append((name, _args_to_dict(args)))
            found_markup = True
        if raw:
            spans = [(0, len(work))]

    props_by_name: dict[str, dict] = {}
    required_by_name: dict[str, list] = {}
    for t in tools or []:
        fn = _fn(t)
        params = fn.get("parameters") or {}
        props_by_name[fn.get("name", "")] = params.get("properties") or {}
        required_by_name[fn.get("name", "")] = list(params.get("required") or [])

    calls: list[dict] = []
    errors: list[str] = []
    for name, args in raw:
        if tools:
            matched = _match_name(name, names)
            if matched is None:
                errors.append(f"unknown tool {name!r} (available: {', '.join(names)})")
                continue
            name = matched
        if not name:
            errors.append("a tool call had no name")
            continue
        if args is None:
            errors.append(f"{name}: arguments are not a valid JSON object")
            continue
        if tools:
            props = props_by_name.get(name, {})
            args = {k: _coerce(v, props.get(k, {})) for k, v in args.items()}
            missing = [r for r in required_by_name.get(name, []) if r not in args or args[r] is None]
            if missing:
                errors.append(f"{name}: missing required argument(s): {', '.join(missing)}")
                continue
        calls.append(_build_call(name, args))

    # text without the tool markup
    cleaned = work
    for s, e in sorted(spans, reverse=True):
        cleaned = cleaned[:s] + cleaned[e:]
    cleaned = _WRAPPER_RE.sub("", cleaned)
    cleaned = _EMPTY_FENCE_RE.sub("", cleaned)
    cleaned = re.sub(r"<tool_calls\b[^>]*>[\s\S]*$", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return ParseResult(text=cleaned, calls=calls, errors=errors, found_markup=found_markup)


def parse_tool_calls(content: str, tools: Optional[list] = None):
    """Backwards compatible wrapper: ``(cleaned_text, tool_calls)``."""
    res = extract_tool_calls(content, tools)
    return res.text, res.calls


def build_retry_prompt(errors: list, tools: Optional[list], tool_choice: Any = None) -> str:
    problem = "; ".join(errors) if errors else "no valid <tool_calls> XML was found in it"
    forced = _format_tool_choice(tool_choice)
    return (
        f"Your previous reply could not be used as a tool call: {problem}.\n"
        "Reply again with ONLY the <tool_calls> XML, using exactly the tool and parameter names "
        "listed in the tools section (arguments must be one valid JSON object). " + forced
    ).strip()


# ----------------------------------------------------------------------------
# streaming filter
# ----------------------------------------------------------------------------

_MARKERS = ("<tool_call", "<invoke", "<function_call", "<tool_use")
_FENCE_LEADS = ("```json", "```xml", "```tool")


class ToolStreamFilter:
    """Pass prose through live, hold back tool markup.

    ``feed(chunk)`` returns the text that is safe to show now. Once a marker
    (``<tool_call`` ...) shows up, or the reply starts like bare JSON / a
    json|xml fence, everything else is held (see :attr:`held`) so the caller can
    parse it at the end.
    """

    def __init__(self) -> None:
        self.pending = ""
        self.held = ""
        self.emitted = ""
        self.triggered = False  # a marker was seen
        self.buffer_all = False  # reply looks like bare JSON / fenced payload

    @property
    def holding(self) -> bool:
        return self.triggered or self.buffer_all

    def _emit(self, text: str) -> str:
        self.emitted += text
        return text

    def feed(self, chunk: str) -> str:
        if self.holding:
            self.held += chunk
            return ""
        self.pending += chunk
        if not self.emitted:
            lead = self.pending.lstrip()
            if not lead:
                return ""
            if lead[0] in "{[" or lead.startswith(_FENCE_LEADS):
                self.buffer_all = True
                self.held, self.pending = self.pending, ""
                return ""
            if lead.startswith("```") and len(lead) < 8:
                return ""  # could still become ```json
            if any(f.startswith(lead) for f in _FENCE_LEADS) and len(lead) < 7:
                return ""
        low = self.pending.lower()
        idx = [i for i in (low.find(m) for m in _MARKERS) if i >= 0]
        if idx:
            i = min(idx)
            visible = self.pending[:i].rstrip()
            out, self.held = visible, self.pending[len(visible) :]
            self.pending = ""
            self.triggered = True
            return self._emit(out)
        # keep a tail that could be the start of a marker
        hold = 0
        for m in _MARKERS:
            for k in range(min(len(m) - 1, len(low)), 0, -1):
                if low.endswith(m[:k]):
                    hold = max(hold, k)
                    break
        out = self.pending[: len(self.pending) - hold] if hold else self.pending
        # also keep trailing whitespace: it may precede a tool block and shouldn't be shown
        out = out.rstrip()
        self.pending = self.pending[len(out) :]
        return self._emit(out)

    def finish(self) -> str:
        """Flush what is still pending (only when nothing was held)."""
        if self.holding or not self.pending:
            return ""
        out, self.pending = self.pending, ""
        return self._emit(out)
