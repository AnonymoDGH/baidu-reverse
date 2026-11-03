"""OpenAI-compatible HTTP server on top of :class:`~baidu_reverse.BaiduChat`.

    baidu-reverse serve --port 8787 --auto-solve --cache

Endpoints: ``GET /v1/models``, ``POST /v1/chat/completions`` (streaming and
non-streaming, ``tools`` / ``tool_choice``, ``reasoning_content``), ``GET /health``.

Long chats are cheap: the history of every request is mapped (hash chain) to a live
Baidu server session, so a client that resends the whole conversation each turn only
costs one new message upstream (see :class:`~baidu_reverse.session.PrefixSessionCache`).
Extra body fields: ``deep_search``, ``internet_search``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Optional

from aiohttp import web

from .client import BaiduBlockedError, BaiduChat, BaiduError
from .models import DEFAULT_MODEL, describe_models, public_models
from .session import PrefixSessionCache


def _error(status: int, message: str, etype: str = "invalid_request_error") -> web.Response:
    return web.json_response({"error": {"message": message, "type": etype, "code": status}}, status=status)


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def _messages_chars(messages: list) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        n += len(c) if isinstance(c, str) else len(json.dumps(c, ensure_ascii=False)) if c else 0
    return n


def create_app(
    chat: BaiduChat,
    api_key: Optional[str] = None,
    session_cache: Optional[PrefixSessionCache] = None,
    max_concurrency: int = 2,
) -> web.Application:
    sessions = session_cache if session_cache is not None else PrefixSessionCache()
    sem = asyncio.Semaphore(max_concurrency)

    @web.middleware
    async def auth(request: web.Request, handler):
        if api_key and request.path.startswith("/v1/"):
            if request.headers.get("Authorization", "") != f"Bearer {api_key}":
                return _error(401, "invalid api key", "authentication_error")
        return await handler(request)

    async def health(_: web.Request) -> web.Response:
        out = {"status": "ok", "sessions": len(sessions), "session_hits": sessions.hits, "session_misses": sessions.misses}
        if chat.cache is not None:
            out["cache"] = chat.cache.stats()
        return web.json_response(out)

    async def models(_: web.Request) -> web.Response:
        meta = {m["id"]: m for m in describe_models()}
        data = []
        for mid in public_models():
            base = mid.rsplit("-thinking", 1)[0].rsplit("-nothinking", 1)[0]
            data.append({"id": mid, "object": "model", "created": 0, "owned_by": "baidu", **{"info": meta.get(base, {})}})
        return web.json_response({"object": "list", "data": data})

    def _kwargs(body: dict) -> dict:
        return {
            "model": body.get("model") or DEFAULT_MODEL,
            "tools": body.get("tools") or None,
            "tool_choice": body.get("tool_choice"),
            "deep_search": bool(body.get("deep_search")),
            "internet_search": bool(body.get("internet_search")),
            "session_cache": sessions,
        }

    async def completions(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except Exception:
            return _error(400, "body must be JSON")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return _error(400, "`messages` must be a non-empty list")
        kwargs = _kwargs(body)
        model = kwargs["model"]
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        prompt_tokens = _est_tokens("x" * _messages_chars(messages))

        async with sem:
            agen = chat.chat_completions(messages, **kwargs)
            try:
                if not body.get("stream"):
                    content: list[str] = []
                    reasoning: list[str] = []
                    calls: list[dict] = []
                    async for ch in agen:
                        if ch["type"] == "content":
                            content.append(ch["content"])
                        elif ch["type"] == "reasoning_content":
                            reasoning.append(ch["content"])
                        elif ch["type"] == "tool_calls":
                            calls.extend(ch["tool_calls"])
                    text = "".join(content).strip()
                    message: dict[str, Any] = {"role": "assistant", "content": text or None}
                    if reasoning:
                        message["reasoning_content"] = "".join(reasoning)
                    if calls:
                        message["tool_calls"] = calls
                    comp = _est_tokens(text + json.dumps(calls)) if (text or calls) else 0
                    return web.json_response(
                        {
                            "id": cid,
                            "object": "chat.completion",
                            "created": created,
                            "model": model,
                            "choices": [
                                {"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": comp,
                                "total_tokens": prompt_tokens + comp,
                            },
                        }
                    )

                # ---- streaming: fetch the first chunk before committing to a 200 ----
                first = None
                async for first in agen:
                    break
                resp = web.StreamResponse(
                    status=200,
                    headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
                await resp.prepare(request)

                async def send(delta: dict, finish: Optional[str] = None) -> None:
                    obj = {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    await resp.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")

                had_calls = False

                async def emit(ch: dict) -> None:
                    nonlocal had_calls
                    if ch["type"] == "content":
                        await send({"content": ch["content"]})
                    elif ch["type"] == "reasoning_content":
                        await send({"reasoning_content": ch["content"]})
                    elif ch["type"] == "tool_calls":
                        had_calls = True
                        await send(
                            {"tool_calls": [{"index": i, **c} for i, c in enumerate(ch["tool_calls"])]}
                        )

                await send({"role": "assistant", "content": ""})
                try:
                    if first is not None:
                        await emit(first)
                    async for ch in agen:
                        await emit(ch)
                except BaiduError as exc:  # already streaming: report in-band
                    await send({"content": f"\n[upstream error: {exc}]"})
                await send({}, "tool_calls" if had_calls else "stop")
                await resp.write(b"data: [DONE]\n\n")
                await resp.write_eof()
                return resp
            except BaiduBlockedError as exc:
                return _error(503, str(exc), "baidu_blocked")
            except BaiduError as exc:
                return _error(502, str(exc), "upstream_error")
            except (ConnectionResetError, asyncio.CancelledError):
                raise
            except Exception as exc:  # noqa: BLE001
                return _error(500, f"{type(exc).__name__}: {exc}", "server_error")
            finally:
                await agen.aclose()

    app = web.Application(middlewares=[auth], client_max_size=64 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", completions)
    return app


async def serve(
    host: str = "127.0.0.1",
    port: int = 8787,
    api_key: Optional[str] = None,
    max_concurrency: int = 2,
    **chat_kwargs: Any,
) -> None:
    """Run the server until cancelled."""
    chat = BaiduChat(**chat_kwargs)
    app = create_app(chat, api_key=api_key, max_concurrency=max_concurrency)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"baidu-reverse OpenAI-compatible server on http://{host}:{port}/v1  (Ctrl+C to stop)")
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        await chat.close()
