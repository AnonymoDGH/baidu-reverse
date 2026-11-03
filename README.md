# baidu-reverse

Reverse-engineered **async** Python client for [chat.baidu.com](https://chat.baidu.com) / [wenxin.baidu.com](https://wenxin.baidu.com)
(Baidu Wenxin 文心). Pure-protocol chat (no browser needed to talk to the model), plus the pieces that make it
actually usable day to day:

| | |
|---|---|
| **Chat** | one-shot + streaming SSE, live reasoning (`thinking` events), deep/internet search |
| **Long chats** | native server-side memory (`Conversation`), `ChatSession` with rolling-summary compaction + persistence, SQLite response cache |
| **Tool calling** | emulated function calling with a tolerant parser + schema validation, an `@tool` decorator and an `Agent` loop |
| **OpenAI-compatible server** | `/v1/chat/completions` with streaming, `tools`, `reasoning_content` |
| **Captcha** | solver for Baidu's rotate-the-picture captcha, plugged into the client (`auto_solve=True`) |
| **Proxies** | scrape ~10k free proxies, verify them against Baidu, rotate requests through them |

```bash
pip install -e .                  # core: just aiohttp
pip install -e ".[captcha]"       # + captcha solver (playwright, torch, pillow, numpy, py7zr)
pip install -e ".[dev]"           # + pytest
```

Python ≥ 3.10.

---

## Quickstart

```python
import asyncio
from baidu_reverse import BaiduChat

async def main():
    async with BaiduChat() as chat:
        print(await chat.generate("Explain what a kernel is in 2 lines"))

        async for ev in chat.stream("Why is the sky blue?", model="deepseek-v4-pro"):
            if ev["type"] == "thinking":
                print(f"\x1b[90m{ev['content']}\x1b[0m", end="")   # live reasoning
            elif ev["type"] == "text":
                print(ev["content"], end="", flush=True)

asyncio.run(main())
```

Event types from `stream()`: `text`, `thinking`, `search_query`, `done`, `ping`, `metadata`.

If Baidu answers nothing because its anti-bot flagged you, `BaiduBlockedError` is raised (alias `BaiduCaptchaError`).
Fix it automatically with [`auto_solve=True`](#captcha-solver) and/or [`rotate_proxies=True`](#proxy-rotation).

---

## Models

Verified live against the backend (anonymous session):

| id | what actually answers | in the web UI | reasoning |
|---|---|---|---|
| `smartmode` | Baidu's own assistant (文心助手, "自动") | yes | off |
| `ernie-5.1` | ERNIE 5.1 (ERNIE Bot), web search on | yes | off (`-thinking` to enable) |
| `deepseek-v4-pro` *(default)* | **DeepSeek V4** | hidden | **on** |
| `deepseek-v4-flash` | **DeepSeek V4 Flash** | hidden | off (`-thinking` to enable) |

Append `-thinking` / `-nothinking` to any id to force reasoning on/off. Aliases: `smart`, `auto`, `wenxin`, `ernie`,
`deepseek`, `ds-v4`, `v4-pro`, `flash`, ...

> **Removed upstream:** `deepseek-r1`, `deepseek-v3` and `ernie-4.5` no longer exist. The server does **not** error: it
> silently answers with the generic assistant (no reasoning). The client maps them to the closest live model and emits a
> `DeprecationWarning`. The DeepSeek V4 ids still work but are hidden in the UI, so they could disappear without notice.

`baidu-reverse models` prints the catalog. The web UI also exposes a "work task" mode (`task`), image / PPT / music
generation and file upload; those use different flows and are **not** implemented.

---

## Long chats

### Native server-side sessions

Baidu keeps a conversation's context on its side. The first request sends `ori_lid=""` and the answer carries a
`sessionId`; follow-ups send that id plus an incremented `rank` and **only the new message**.

```python
from baidu_reverse import BaiduChat, Conversation

async with BaiduChat() as chat:
    conv = Conversation()
    await chat.generate("My name is Ana", conversation=conv)
    print(await chat.generate("What is my name?", conversation=conv))   # -> "Ana"
```

Calls **without** a `Conversation` are isolated (each starts a fresh session). Earlier versions leaked context between
independent calls because they all shared one `ori_lid`.

### `ChatSession`: memory that scales

```python
from baidu_reverse import BaiduChat, ChatSession, SessionStore

async with BaiduChat() as chat:
    s = ChatSession(chat, model="deepseek-v4-flash", system="Answer briefly.",
                    store=SessionStore())           # autosaves every turn
    await s.send("I live in Cusco and love cuy")
    ...
    print(await s.send("Where do I live?"))
    s.save("chat.json")                              # ChatSession.load(chat, "chat.json") later
```

* Each turn sends only the new message (server memory) and keeps the full transcript locally.
* When the server session grows past `max_context_chars` (or `max_turns_per_session` turns) the older part is folded into a
  **rolling summary** (written by the model itself, on a cheap model) and a **fresh server session** is bootstrapped from
  `system + summary + last N turns`.
* If the server answers nothing (it forgot the session) the session is rebuilt from the local transcript and the turn retried.
* A reloaded session replays its context in its first turn instead of trusting a possibly-expired server session.

### Caches

* `ResponseCache` (SQLite, TTL + size cap): identical *stateless* requests are answered locally. `BaiduChat(cache=True)`.
  Calls inside a `Conversation` are never cached.
* `PrefixSessionCache`: hash-chains an OpenAI-style `messages` list to a live server session so a client that resends the
  whole history each turn only costs the **new messages** upstream. Used by `chat_completions` and the server.

### Limits (measured)

* The JSON body accepted **30k–50k+ characters** with exact recall of a "needle" placed at the start.
* The `X-Chat-Message` header carries the URL-encoded query and Baidu flags requests whose header is long (≳600 encoded
  chars → empty answer). The client truncates the header copy; the body is untouched. (This is why large prompts used to fail.)

---

## Tool calling

Baidu has no function-calling API, so it is emulated and made reliable:

```python
from typing import Literal
from baidu_reverse import Agent, BaiduChat, tool

@tool
def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
    """Get the current weather.

    Args:
        city: City name, e.g. "Lima".
        unit: Temperature unit.
    """
    return f"17.3{unit.upper()}, drizzle in {city}"

async with BaiduChat() as chat:
    agent = Agent(chat, [get_weather], model="deepseek-v4-pro")
    result = await agent.run("What's the weather in Lima?")
    print(result.text, result.steps)          # steps: tool name, arguments, result, seconds
```

* **Schema from Python**: type hints + Google-style docstring → JSON schema. Sync and async tools, per-tool timeouts,
  parallel calls, tool errors are returned to the model so it can recover, loop/`max_steps` guards.
* **Prompt**: compact tool descriptions with an example generated from your schema; the history is rendered in the *same*
  XML dialect the model must produce.
* **Parser** (`extract_tool_calls`): survives truncation, markdown fences, CDATA, escaped markup, `<`/`&` inside arguments,
  `<invoke>` / JSON-body / bare-JSON styles, sloppy JSON (single quotes, trailing commas), fuzzy tool names
  (`functions.Get_Weather` → `get_weather`), and **validates against the schema** (coerces `"3"`→`3`, rejects missing
  required arguments with a reason).
* **Self-repair**: if a call was required (`tool_choice="required"`) or malformed, the model gets one short correction message in the same
  session ("missing required argument: city") instead of a replay of the whole prompt.
* **Streaming**: prose streams live, the XML is held back (`ToolStreamFilter`).
* With the built-in session cache, every agent step after the first sends **only the new tool results**.

Raw OpenAI-style usage: `chat.chat_completions(messages, tools=[...], tool_choice=...)` (streaming chunks) or
`await chat.complete(messages, tools=[...])`.

**Live results** (6 scenarios: single call, exact math, two-city comparison, a fact only a tool knows, no tool needed, forced
`tool_choice`): `deepseek-v4-pro` **6/6**, `deepseek-v4-flash` **5/6** (the miss was the model refusing to reveal a "secret
access code", not a protocol failure). `smartmode` and `ernie-5.1` were **not** measured: Baidu rate-limited the run before
they started.

---

## OpenAI-compatible server

```bash
baidu-reverse serve --port 8787 --auto-solve --cache --api-key sekret
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="sekret")
r = client.chat.completions.create(model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "hi"}], stream=True)
```

`GET /v1/models`, `POST /v1/chat/completions` (stream / non-stream, `tools`, `tool_choice`, `reasoning_content`),
`GET /health`. Extra body fields: `deep_search`, `internet_search`. Upstream blocks are reported as HTTP 503
(`baidu_blocked`) *before* a stream starts.

---

## Captcha solver

When Baidu's `kunlun` anti-bot flags a session the answer is empty and the web app shows a captcha: *drag the slider until the
picture is upright*.

```python
async with BaiduChat(auto_solve=True) as chat:      # needs: pip install -e ".[captcha]"
    print(await chat.generate("hello"))             # blocked -> browser -> captcha -> cookies -> retry
```

```bash
baidu-reverse solve-captcha                         # standalone: saves verified cookies
baidu-reverse solve-captcha --cdp-url http://127.0.0.1:9222   # attach to a browser you already run
```

**How it works.** Baidu's SDK encrypts the mouse trace, a PoW and a device fingerprint before submitting, so re-implementing
the wire protocol is brittle. Instead a real Brave/Chrome/Edge (auto-detected, or `BAIDU_REVERSE_BROWSER`) is driven over CDP
and only does what a human does:

1. The SDK rotates the picture with CSS `rotate(start + 360·percent deg)` and sends only `percent` (2 decimals → 3.6° steps).
2. A RotNet-style CNN ([lumina37/rotate-captcha-crack](https://github.com/lumina37/rotate-captcha-crack), MIT; weights
   downloaded once to `~/.cache/baidu_reverse`) predicts how far the picture is rotated, with 4-way rotation TTA.
3. The slider is dragged with a human-like trace (eased stroke, overshoot, jitter) and **corrected in closed loop** by reading the
   real rotation the SDK computed, until it equals the target.
4. On failure a fresh captcha is loaded and it retries; on success the browser's cookies become the client's cookies.

First-visit pop-ups that swallow clicks are dismissed automatically. Pass `solver_options={"save_dir": "captcha_samples"}` to
keep every picture with its predicted angle and the accept/reject result: accepted attempts are free *labelled* data to fine-tune
the model on Baidu's own pictures.

**Honest numbers.** Per-attempt accuracy is imperfect (in live runs: one first-try success, one failure at 97 % model
confidence followed by a success), so retries are part of the design. Solving takes ~15–60 s including the browser launch and a
one-time model load.

---

## Proxy rotation

`kunlun` rate-limits per IP, so bursts end in empty answers. A self-feeding pool spreads the load:

```python
async with BaiduChat(rotate_proxies=True) as chat:    # or proxies=["ip:port", ...] / proxy="http://ip:port"
    print(await chat.generate("hello"))
```

```bash
baidu-reverse proxies --count 15 --refresh            # scrape + verify, list the working ones
baidu-reverse "hello" --rotate-proxies
```

* **Scrape**: 10 public lists (~10k unique candidates).
* **Verify**: each candidate must reach `baseData` over HTTPS (certificate verified, so TLS-intercepting proxies fail) and return
  Baidu's real JSON. Checks run in waves and **stop as soon as `target_size` are healthy**. Measured: ~5 % of candidates pass; 13
  verified proxies in ~12 s.
* **Rotate**: latency-weighted random choice that avoids the exit just used; network failures rest a proxy (and drop one that never
  worked); a proxy that gets Baidu to answer "kunlun" rests 10+ minutes.
* **Fail over**: connection errors or proxy 407/429/5xx are retried through other exits (`proxy_retries`, default 4).
* **Block recovery order**: rotate the exit first, then (if `auto_solve`) solve the captcha.
* **Refill**: when few remain, a background refill runs; the verified set is cached (30 min) and re-verified on the next start.

> **Not verified yet:** that rotating IPs actually removes the `kunlun` block for an already-flagged cookie jar (it may be tied to
> the cookie, not only the IP). The mechanism and its unit tests are in; the live end-to-end check was not completed. Free
> proxies are slow (typically 2–6 s to reach Baidu) and see your traffic: do not send secrets through them.

---

## CLI

```text
baidu-reverse "question" [--model M] [--thinking] [--deep-search] [--internet-search]   one-shot, streams
baidu-reverse chat [--model M] [--system S] [--session ID]                              interactive, remembers, autosaves
baidu-reverse serve [--port 8787] [--api-key K] [--max-concurrency 2]                   OpenAI-compatible server
baidu-reverse solve-captcha [--cdp-url URL] [--exe PATH] [--save-captchas DIR]          get verified cookies
baidu-reverse proxies [--count N] [--refresh]                                           scrape + verify proxies
baidu-reverse models | sessions                                                          catalog / saved chats
```

Common flags: `--cookies`, `--cookie-file`, `--proxy`, `--rotate-proxies`, `--auto-solve`, `--cache`, `--debug`.

---

## How it works

1. `GET /aichat/api/aitabserver?ctl=baseData` → `token` + `lid` (falls back to the inline `aiTabFrameBaseData` of the homepage).
2. Chat token: `base64("{token}|{MD5(query)}|{ms}|{lid}")-{lid}-3`.
3. `POST /aichat/api/conversation` with the web payload (`enter_type=chat_url`, `anti_ext={"inputT":null}`, `usedModel`, ...) and
   the `X-Chat-Message` header (query URL-encoded **and truncated**). Cookies are sent verbatim and refreshed from `Set-Cookie`.
4. SSE: `markdown-yiyan` = answer, `thinkingSteps` = reasoning, a `kunlunHint` part / `chatHitKunlun` = anti-bot block.
   A stall guard ends a stream that goes silent.

Cookies are optional: without them the server mints anonymous ones. The client sends the first message of each conversation with
`ori_lid=""` and continues it with the returned `sessionId` and an incremented `rank`.

---

## Tests

```bash
pip install -e ".[dev]"
pytest            # 100+ offline tests (scripted fake of Baidu's server: sessions, forgetting, kunlun blocks, tools, server, proxies)
```

Live smoke scripts (need real cookies): `scripts/test_models.py`, `scripts/test_tools.py`, `scripts/benchmark.py`
(`$env:BAIDU_COOKIES="BAIDUID=...; ..."`).

---

## Known limitations

* Baidu changes this service often: models disappear (R1 did), payload/anti-bot details shift. Reverse-engineered, so no stability promise.
* A follow-up turn on a native session was seen once returning **empty** through the server (not reproduced afterwards; a later empty
  capture was a plain `kunlun` block). `ChatSession` recovers from empty replies by rebuilding the session; the OpenAI server does not
  retry that case yet.
* Per-IP rate limiting: a few dozen requests in a short time can get a session blocked (the solver / proxy rotation exist for this).
* Only HTTP(S) proxies; no SOCKS.
* Image generation, work-task mode, file upload and login-only features are not implemented.

## Disclaimer

For educational and research purposes. Not affiliated with or endorsed by Baidu. Endpoints and the token algorithm may change; the
service may rate-limit or block unofficial clients, and solving its captcha may breach its terms: use at your own risk. The
reverse-engineering approach for the chat protocol builds on [baidu2api2](https://github.com/dijiaozhibei-top/baidu2api2); the
captcha angle model is from [rotate-captcha-crack](https://github.com/lumina37/rotate-captcha-crack) (MIT).
