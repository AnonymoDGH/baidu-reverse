"""Prueba de tool calling en vivo contra chat.baidu.com.

Requiere ``BAIDU_COOKIES`` (cookies de chat.baidu.com de un navegador).

Uso:
    $env:BAIDU_COOKIES="BAIDUID=...; ZFY=..." ; python scripts/test_tools.py
"""

import asyncio
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from baidu_reverse import BaiduChat

COOKIES = os.environ.get("BAIDU_COOKIES", "")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Obtener el clima actual de una ciudad",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

MODELS = ["deepseek-r1", "smartmode", "deepseek-v4-pro", "deepseek-v4-flash", "ernie-5.1"]


async def test_tool(chat: BaiduChat, model: str) -> dict:
    got = []
    async for chunk in chat.chat_completions(
        [{"role": "user", "content": "¿Qué clima hace en Buenos Aires?"}],
        model=model,
        tools=[WEATHER_TOOL],
    ):
        got.append(chunk)
    tc = [g for g in got if g["type"] == "tool_calls"]
    if tc:
        calls = tc[0]["tool_calls"]
        return {"model": model, "tool_calls": calls}
    texts = "".join(g["content"] for g in got if g["type"] == "content")
    return {"model": model, "tool_calls": [], "text": texts.strip()[:150]}


async def main():
    if not COOKIES:
        print("Set BAIDU_COOKIES env var", file=sys.stderr)
        return 1
    async with BaiduChat(cookies=COOKIES) as chat:
        print("=" * 70)
        for model in MODELS:
            try:
                r = await test_tool(chat, model)
            except Exception as e:
                print(f"  {model:<20} ERROR {type(e).__name__}: {str(e)[:100]}")
                continue
            if r["tool_calls"]:
                for call in r["tool_calls"]:
                    print(f"  {model:<20} TOOL  {call['function']['name']}({call['function']['arguments']})")
            else:
                print(f"  {model:<20} NO-TOOL  {r['text']!r}")
            await asyncio.sleep(2)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
