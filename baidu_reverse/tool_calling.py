"""Tool-calling emulation for baidu-reverse.

Baidu Wenxin has no native function-calling API, so (matching the approach
used by the community ``baidu2api2`` project) we instruct the model to emit
``<tool_calls>`` XML and parse it back into OpenAI-style ``tool_calls``.
"""

import json
import re
import uuid
import xml.etree.ElementTree as ET
from typing import Any

TOOL_SYSTEM_PROMPT = """You can call tools when they are useful.

When calling tools, do not answer in natural language. Output only XML in this exact format:
<tool_calls>
  <tool_call>
    <name>tool_name</name>
    <arguments>{{"arg":"value"}}</arguments>
  </tool_call>
</tool_calls>

Rules:
- Use only tool names listed below.
- arguments must be a valid JSON object.
- If a required tool is specified, call that tool.
- If no tool is needed, answer normally without XML.

Available tools:
{tools}

{tool_choice}
"""


def build_tool_system_prompt(tools: list, tool_choice: Any = None) -> str:
    summaries = []
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name", "")
        if not name:
            continue
        summaries.append(json.dumps({
            "name": name,
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
        }, ensure_ascii=False))
    return TOOL_SYSTEM_PROMPT.format(
        tools="\n".join(summaries),
        tool_choice=_format_tool_choice(tool_choice),
    )


def messages_to_prompt(
    messages: list,
    tools: list | None = None,
    tool_choice: Any = None,
    max_chars: int = 12000,
    max_messages: int = 16,
    max_message_chars: int = 2000,
) -> str:
    if not isinstance(messages, list):
        return ""
    parts = []
    if tools:
        parts.append(f"System: {build_tool_system_prompt(tools, tool_choice)}")
    compacted = compact_messages(messages, max_messages=max_messages, max_message_chars=max_message_chars)
    for msg in compacted:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = _content_to_text(msg.get("content", ""))
        if content:
            parts.append(f"{role.capitalize()}: {content}")
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            parts.append(f"Assistant tool_calls: {json.dumps(tool_calls, ensure_ascii=False)}")
        if role == "tool":
            tool_name = msg.get("name") or msg.get("tool_call_id") or "tool"
            if parts:
                parts[-1] = f"Tool({tool_name}): {content}" if content else parts[-1]
    prompt = "\n\n".join(parts).strip()
    if max_chars > 0 and len(prompt) > max_chars:
        prompt = _clip_middle(prompt, max_chars)
    return prompt


def compact_messages(
    messages: list,
    max_messages: int = 16,
    max_message_chars: int = 2000,
) -> list:
    valid = [m for m in messages if isinstance(m, dict)]
    if not valid:
        return []
    system_messages = [m for m in valid if m.get("role") == "system"]
    non_system = [m for m in valid if m.get("role") != "system"]
    recent = non_system[-max_messages:] if max_messages > 0 else non_system
    compacted = []
    for msg in system_messages + recent:
        copied = dict(msg)
        content = _content_to_text(copied.get("content", ""))
        if max_message_chars > 0 and len(content) > max_message_chars:
            copied["content"] = _clip_middle(content, max_message_chars)
        compacted.append(copied)
    return compacted


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
        return "\n".join(t for t in texts if t)
    return ""


def _clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 20:
        return text[:limit]
    head = limit // 2
    tail = limit - head - 18
    return f"{text[:head]}\n...[compressed]...\n{text[-tail:]}"


def _format_tool_choice(tool_choice: Any) -> str:
    if not tool_choice or tool_choice == "auto":
        return ""
    if tool_choice == "none":
        return "Required tool: none. Do not call tools."
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function", {})
        name = function.get("name", "") if isinstance(function, dict) else ""
        if name:
            return f"Required tool: {name}"
    return ""


def parse_tool_calls(content: str):
    if not content:
        return content, []
    # Buscar bloques completos <tool_calls>...</tool_calls>
    blocks = re.findall(r"<tool_calls\b[^>]*>.*?</tool_calls>", content, flags=re.DOTALL)
    tool_calls = []
    for block in blocks:
        tool_calls.extend(_parse_tool_call_block(block))

    # Tolerante a truncamiento: si el cierre </tool_calls> no llego pero hay
    # <tool_call> completos (p.ej. el modelo corto la salida), parsearlos igual.
    if not tool_calls and "<tool_call" in content:
        tool_calls.extend(_parse_loose_tool_calls(content))

    cleaned = content
    for block in blocks:
        cleaned = cleaned.replace(block, "")
    if not blocks:
        # quitar cualquier resto de XML incompleto
        cleaned = re.sub(r"<tool_calls\b[^>]*>[\s\S]*$", "", cleaned).strip()
    return cleaned.strip(), tool_calls


def _parse_loose_tool_calls(content: str) -> list:
    """Parse individual <tool_call>...</tool_call> blocks even if the
    surrounding <tool_calls> wrapper is truncated/incomplete."""
    calls = []
    for block in re.findall(r"<tool_call\b[^>]*>.*?</tool_call>", content, flags=re.DOTALL):
        calls.extend(_parse_tool_call_block(f"<tool_calls>{block}</tool_calls>"))
    return calls


def _parse_tool_call_block(block: str) -> list:
    try:
        root = ET.fromstring(block)
    except ET.ParseError:
        return []
    parsed = []
    for node in root.findall("tool_call"):
        name = (node.findtext("name") or "").strip()
        raw_args = (node.findtext("arguments") or "{}").strip()
        if not name:
            continue
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        parsed.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return parsed
