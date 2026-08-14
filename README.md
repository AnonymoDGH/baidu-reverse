# baidu-reverse

Reverse-engineered **async** Python client for [chat.baidu.com](https://chat.baidu.com) (Baidu Wenxin 文心) — text chat with streaming, real-time reasoning, deep search and internet search. Pure-algorithm implementation (no browser automation).

```bash
pip install -e .
```

---

## Features

- **Chat** — one-shot and streaming (SSE incremental) via `POST /aichat/api/conversation`
- **Real-time reasoning** — `thinking` events streamed as they arrive (`thinkingSteps`)
- **Models** — smartMode (智能模式), ERINE-5.1 (文心 5.1), DeepSeek-R1, DeepSeek-V4 Pro, DeepSeek-V4 Flash (each with `-thinking` / `-nothinking` variants where supported)
- **Deep search** / **internet search** toggles
- **Anti-bot** — scrapes the inline `aiTabFrameBaseData` JSON from the homepage for `token`/`lid` and lets the server set its cookies; auto-refreshes on 401/403
- **Cookies optional** — provide a browser cookie string to bypass the anti-bot check deterministically, or let the client fetch them automatically
- **SSE stall guard** — aborts after 8s without data instead of hanging

---

## Quickstart

### Basic text generation
```python
import asyncio
from baidu_reverse import BaiduChat

async def main():
    async with BaiduChat() as chat:
        reply = await chat.generate("Explain what a kernel is in 2 lines")
        print(reply)

asyncio.run(main())
```

### Streaming with real-time thinking
```python
import asyncio
from baidu_reverse import BaiduChat

COOKIES = "BAIDUID=xxx; ZFY=yyy; ..."  # from a logged-in browser

async def main():
    async with BaiduChat(cookies=COOKIES) as chat:
        async for chunk in chat.stream("Explain what an OS kernel is", model="deepseek-r1"):
            if chunk["type"] == "thinking":
                print(f"\r\033[90m{chunk['content']}\033[0m", end="", flush=True)
            elif chunk["type"] == "text":
                print(chunk["content"], end="", flush=True)

asyncio.run(main())
```

### Tool calling (emulated)
```python
import asyncio
from baidu_reverse import BaiduChat

COOKIES = "BAIDUID=xxx; ZFY=yyy; ..."

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}

async def main():
    async with BaiduChat(cookies=COOKIES) as chat:
        async for chunk in chat.chat_completions(
            [{"role": "user", "content": "Weather in Buenos Aires?"}],
            model="deepseek-r1",
            tools=[WEATHER_TOOL],
        ):
            if chunk["type"] == "tool_calls":
                print(chunk["tool_calls"])  # [{"function": {"name": "get_weather", ...}}]

asyncio.run(main())
```

Baidu has no native function API, so tools are emulated: the client injects a
system prompt asking for `<tool_calls>` XML and parses it back to OpenAI
`tool_calls`. Verified live: `deepseek-r1`, `smartmode` and `deepseek-v4-pro`
emit correct tool calls; the parser is tolerant of truncated XML.

```bash
$env:BAIDU_COOKIES="BAIDUID=...; ZFY=..."
python scripts/test_tools.py
```

### Testing all models (streaming + reasoning capture)
```bash
$env:BAIDU_COOKIES="BAIDUID=...; ZFY=...; BA_HECTOR=...; ppfuid=..."
python scripts/test_models.py
```

Verified results (23×47, streamed live):

| Model | Reasoning | Text chunks | 1st token | Answer |
|---|---|---|---|---|
| `smartmode` | – | 82 | ~1.5s | step-by-step |
| `smartmode-thinking` | – | 82 | ~0.34s | step-by-step |
| `ernie-5.1` | – | 1 | ~0.5s | 1081 |
| `deepseek-r1` | ✅ 179 chunks | 2 | ~0.35s | 1081 |
| `deepseek-v4-pro` | ✅ 20 chunks | 1 | ~0.33s | 1081 |
| `deepseek-v4-flash` | – | 1 | ~0.5s | 1081 |
| `deepseek-v4-flash-thinking` | – | 82 | ~0.34s | step-by-step |

`deepseek-r1` and `deepseek-v4-pro` emit incremental reasoning; the others return the answer in few chunks.

### Benchmarks

```bash
$env:BAIDU_COOKIES="BAIDUID=...; ZFY=..."
python scripts/benchmark.py                 # all models
python scripts/benchmark.py --model deepseek-v4-pro --repeats 3
```

Measured over live chat.baidu.com (Windows, residential IP). First token = latency
until the first text/reasoning token; cps = chars/sec of streaming.

| Model | 1st token | Total | text cps | think cps |
|---|---|---|---|---|
| `deepseek-v4-pro` | ~1.2-1.5s | ~2-3s | ~1100-1500 | ~1800-2700 |
| `deepseek-v4-pro-nothinking` | ~3-6s | ~7-13s | ~130-210 | – |
| `deepseek-v4-flash` | ~1.2-1.5s | ~1.5-2.8s | ~600-1500 | – |

Note: Baidu's `kunlun` rate limit (a few requests too close together on the same IP)
makes the stream return a hint with no text for a couple of minutes. Space benchmark
calls out, rotate proxies, or wait — the client surfaces it as an empty answer rather
than hanging.

### OpenAI-style chunks
```python
import asyncio
from baidu_reverse import BaiduChat

async def main():
    async with BaiduChat() as chat:
        async for chunk in chat.chat_to_openai_chunks("hello"):
            print(chunk)  # {"type": "content" | "reasoning_content" | "done", ...}

asyncio.run(main())
```

### Using browser cookies (deterministic anti-bot bypass)
```python
import asyncio
from baidu_reverse import BaiduChat

COOKIES = "BAIDUID=xxx; ZFY=yyy; BA_HECTOR=zzz; ..."  # from your logged-in browser

async def main():
    async with BaiduChat(cookies=COOKIES) as chat:
        print(await chat.generate("hello", model="ernie-5.1"))

asyncio.run(main())
```

### Proxy rotation (bypass IP rate limits)
```python
import asyncio
from baidu_reverse import BaiduChat
from baidu_reverse.proxies import get_working_proxies

async def main():
    proxies = await get_working_proxies()   # scrapes + tests free HTTP proxies
    async with BaiduChat(proxies=proxies) as chat:
        print(await chat.generate("hello"))

asyncio.run(main())
```

You can also pass a single static proxy with `BaiduChat(proxy="http://ip:port")`.

---

## CLI

```bash
baidu-reverse "Explain what a kernel is" --model smartmode
baidu-reverse "Deep question" --model deepseek-r1 --thinking --deep-search
```

---

## How it works

1. Fetch `token` + `lid` from `GET /aichat/api/aitabserver?ctl=baseData` (with a fallback to scraping the inline `aiTabFrameBaseData` script from the homepage).
2. Build the chat token: `base64("{token}|{MD5(query)}|{ms_timestamp}|{lid}")-{lid}-3`.
3. POST `/aichat/api/conversation` with the exact web payload: `enter_type=chat_url`, `anti_ext={"inputT":null}`, `usedModel` per model, and the `X-Chat-Message` header.
4. Stream the SSE events; `markdown-yiyan` chunks carry the answer, `thinkingSteps` carry the reasoning.

### Anti-bot notes

Baidu's `kunlun` check inspects the request's cookies and payload. To work reliably:

- Provide a **browser cookie string** (from `document.cookie` on chat.baidu.com). It is sent verbatim via the `Cookie` header — do not let aiohttp re-manage it from a cookie jar (the client sends it raw to mirror the web).
- Use `enter_type=chat_url` and a minimal `anti_ext={"inputT":null}` — earlier payloads with extra `ck1/ck9/ck10` fields get flagged.
- IP rate limits (kunlun) reset after a few minutes; rotating proxies (`proxies=[...]`) or a different network sidesteps them.

If the server responds with a hint (anti-bot / login) and no answer text, the client surfaces a descriptive error.

---

## Models

Primary models: **`deepseek-v4-pro`** (default) and **`deepseek-v4-flash`**.

| Model id | Baidu model | Thinking |
|---|---|---|
| `deepseek-v4-pro` (default) | DeepSeek-V4 | on |
| `deepseek-v4-pro-nothinking` | DeepSeek-V4 | off |
| `deepseek-v4-flash` | DeepSeek-V4-Flash | off |
| `deepseek-v4-flash-thinking` | DeepSeek-V4-Flash | on |
| `smartmode` / `smartmode-thinking` | 智能模式 smartMode | off / on |
| `ernie-5.1` | 文心 5.1 ERINE-5.1 | on |
| `deepseek-r1` | DeepSeek-R1 | forced on |

Aliases: `smart`, `wenxin`, `ernie`, `ernine`, `deepseek`, `ds-r1`, `ds-v4`, `ds-pro`, `v4-pro`, `ds-flash`, `v4-flash`, `flash`, etc.

---

## Disclaimer

For educational and research purposes. Not affiliated with or endorsed by Baidu. Endpoints and the token algorithm may change; the service may rate-limit or block unofficial clients. Use at your own risk. Reverse-engineering approach based on the [baidu2api2](https://github.com/dijiaozhibei-top/baidu2api2) project (pure-algorithm, no browser automation).
