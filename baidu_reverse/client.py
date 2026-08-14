"""Async reverse-engineered client for Baidu Wenxin (chat.baidu.com).

Pure-algorithm implementation (no browser automation). Replicates what the
official web client does:

1. Fetch ``https://chat.baidu.com/`` and scrape the inline
   ``<script name="aiTabFrameBaseData">`` JSON to obtain ``token`` + ``lid``
   (and let the server set the anti-bot cookies).
2. Build the chat token: ``base64("{token}|{MD5(query)}|{ms}|{lid}")-{lid}-3``.
3. POST ``/aichat/api/conversation`` and stream the SSE events
   (``markdown-yiyan`` chunks = answer text, ``thinkingSteps`` = reasoning).
"""

import base64
import hashlib
import json
import re
import time
import urllib.parse
from typing import Any, AsyncIterator, Optional

import aiohttp

from .models import MODELS, MODEL_ALIASES, DEFAULT_MODEL

BASE_URL = "https://chat.baidu.com"
CONVERSATION_API = f"{BASE_URL}/aichat/api/conversation"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
)

# The inline script tag that carries the bootstrap data on the homepage.
_BASE_DATA_RE = re.compile(
    r'<script[^>]*name="aiTabFrameBaseData"[^>]*>(.*?)</script>',
    re.DOTALL,
)


class BaiduError(Exception):
    pass


class BaiduAuthError(BaiduError):
    pass


def _resolve_model(model: str) -> str:
    key = model.lower()
    if key in MODELS:
        return key
    return MODEL_ALIASES.get(key, DEFAULT_MODEL)


def _apply_model_suffix(spec: dict, model: str) -> dict:
    """Copy a spec and apply -thinking / -nothinking suffix overrides."""
    import copy

    spec = copy.deepcopy(spec)
    forced: Optional[bool] = None
    if model.endswith("-nothinking"):
        forced = False
    elif model.endswith("-thinking"):
        forced = True
    if forced is not None:
        spec["usedModel"]["modelFunction"]["thinkMode"] = "1" if forced else "0"
    return spec


class BaiduChat:
    """Async client for chat.baidu.com (Wenxin).

    Usage::

        async with BaiduChat() as chat:
            async for chunk in chat.stream("Explain what a kernel is"):
                if chunk["type"] == "text":
                    print(chunk["content"], end="", flush=True)

    ``cookies`` is optional: if omitted the client fetches them by visiting
    the homepage (which also yields ``token`` / ``lid``). Provide a browser
    cookie string to bypass the anti-bot check deterministically.
    """

    def __init__(
        self,
        cookies: Optional[str] = None,
        user_agent: Optional[str] = None,
        proxy: Optional[str] = None,
        proxies: Optional[list[str]] = None,
        timeout: int = 60,
        stall_seconds: float = 8.0,
    ):
        self._user_cookies = cookies or ""
        self.user_agent = user_agent or _USER_AGENT
        self.timeout = timeout
        self.stall_seconds = stall_seconds
        self.session: Optional[aiohttp.ClientSession] = None
        self._cookie_jar: Optional[aiohttp.CookieJar] = None
        self._token: Optional[str] = None
        self._lid: Optional[str] = None
        self._ori_lid: Optional[str] = None
        self._proxy_pool: Optional[ProxyPool] = None
        if proxies:
            from .proxies import ProxyPool

            self._proxy_pool = ProxyPool(proxies)
            proxy = None  # ignore static proxy when a pool is configured
        self.proxy = proxy

    def _current_proxy(self) -> Optional[str]:
        """Return the proxy to use for the next request (pool-aware)."""
        if self._proxy_pool is not None:
            p = self._proxy_pool.next()
            return f"http://{p}" if p else None
        return self.proxy

    def _report_proxy_result(self, ok: bool):
        if self._proxy_pool is None:
            return
        p = self._proxy_pool.proxies[(self._proxy_pool._idx - 1) % len(self._proxy_pool.proxies)] if self._proxy_pool.proxies else None
        if ok:
            self._proxy_pool.report_success(p)
        else:
            self._proxy_pool.report_failure(p)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def _base_headers(self) -> dict:
        return {
            "User-Agent": self.user_agent,
            "Accept": "text/event-stream",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": f"{BASE_URL}/",
            "Origin": BASE_URL,
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self._cookie_jar = aiohttp.CookieJar(unsafe=True)
            self.session = aiohttp.ClientSession(
                cookie_jar=self._cookie_jar,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
            # Do NOT inject user cookies into the jar: the exact cookie string
            # is sent via the Cookie header in _build_headers (mirrors the web).
        return self.session

    def _cookie_string(self) -> str:
        # prefer the user-provided cookie string verbatim
        if self._user_cookies:
            return self._user_cookies
        parts = []
        for morsel in self._cookie_jar:
            parts.append(f"{morsel.key}={morsel.value}")
        return "; ".join(parts)

    async def close(self):
        if self.session is not None:
            await self.session.close()
            self.session = None

    async def __aenter__(self) -> "BaiduChat":
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
        return False

    # ------------------------------------------------------------------
    # Bootstrap: token + lid (+ cookies)
    # ------------------------------------------------------------------
    async def _fetch_base_data(self) -> dict:
        session = await self._get_session()
        headers = {
            **self._base_headers(),
            "Accept": "application/json, text/plain, */*",
        }
        if self._user_cookies:
            headers["Cookie"] = self._user_cookies
        url = f"{BASE_URL}/aichat/api/aitabserver?ctl=baseData"
        async with session.get(
            url, headers=headers, proxy=self._current_proxy()
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        frame = (data.get("data") or {}).get("frameBaseData") or {}
        if not (frame.get("token") and frame.get("lid")):
            # fallback: try scraping the homepage inline script
            headers2 = {**self._base_headers(), "Accept": "text/html"}
            async with session.get(
                BASE_URL, headers=headers2, proxy=self._current_proxy()
            ) as resp2:
                resp2.raise_for_status()
                text = await resp2.text(encoding="utf-8", errors="replace")
            match = _BASE_DATA_RE.search(text)
            if not match:
                raise BaiduError(
                    "Could not find aiTabFrameBaseData in chat.baidu.com homepage — "
                    "Baidu may have changed the page structure"
                )
            frame = json.loads(match.group(1))
        self._token = frame.get("token", "")
        self._lid = frame.get("lid", "")
        self._ori_lid = (frame.get("logParams") or {}).get("applid") or self._lid
        return frame

    def _ensure_base_data(self):
        if not self._token or not self._lid:
            raise BaiduError("BaiduChat has no token/lid — call await chat.ready() first")

    async def ready(self):
        """Fetch homepage bootstrap data (token + lid + cookies). Idempotent."""
        if self._token and self._lid:
            return
        await self._fetch_base_data()

    async def _refresh(self):
        self._token = None
        self._lid = None
        self._ori_lid = None
        self._cookie_jar.clear()
        await self._fetch_base_data()

    # ------------------------------------------------------------------
    # Token + payload building
    # ------------------------------------------------------------------
    def _generate_chat_token(self, query: str) -> str:
        md5_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
        timestamp = int(time.time() * 1000)
        payload = f"{self._token}|{md5_hash}|{timestamp}|{self._lid}"
        b64 = base64.b64encode(payload.encode()).decode()
        return f"{b64}-{self._lid}-3"

    def _build_message_payload(
        self,
        query: str,
        model: str = DEFAULT_MODEL,
        deep_search: bool = False,
        internet_search: bool = False,
        rank: Optional[int] = None,
    ) -> dict:
        model_key = _resolve_model(model)
        spec = _apply_model_suffix(MODELS[model_key], model)
        model_cfg = dict(spec["usedModel"])
        if rank is None:
            rank = spec["rank"]
        if deep_search:
            model_cfg["modelFunction"]["deepSearch"] = "1"
        if internet_search:
            model_cfg["modelFunction"]["internetSearch"] = "1"
        chat_token = self._generate_chat_token(query)
        enter_type = spec["enter_type"]
        agt_sess_cnt = spec["agt_sess_cnt"]
        return {
            "message": {
                "inputMethod": "chat_search",
                "isRebuild": False,
                "content": {
                    "query": "",
                    "agentInfo": {"agent_id": [""], "params": f'{{"agt_rk":{rank},"agt_sess_cnt":{agt_sess_cnt}}}'},
                    "agentInfoList": [],
                    "qtype": 0,
                },
                "searchInfo": {
                    "srcid": "", "order": "", "tplname": "", "dqaKey": "",
                    "re_rank": str(rank), "ori_lid": self._ori_lid or "",
                    "sa": "bkb", "enter_type": enter_type,
                    "chatParams": {
                        "setype": "csaitab", "chat_samples": "WISE_NEW_CSAITAB",
                        "chat_token": chat_token, "scene": "",
                    },
                    "isPrivateChat": False, "usedModel": model_cfg,
                    "landingPageSwitch": "", "landingPage": "aitab",
                    "ecomFrom": "", "hasLocPermission": "", "isInnovate": 2,
                    "applid": "", "a_lid": "", "showMindMap": False,
                    "deepDecisionInfo": {"isDeepDecision": 0},
                },
                "from": "", "source": "pc_csaitab",
                "query": [{"type": "TEXT", "data": {"text": {"query": query, "extData": "{}", "text_type": ""}}}],
                "anti_ext": spec["anti_ext"],
            },
            "sa": "bkb", "setype": "csaitab", "rank": rank,
        }

    def _build_headers(self, query: str, model: str, rank: int) -> dict:
        spec = _apply_model_suffix(MODELS[_resolve_model(model)], model)
        anti_ext = json.dumps(spec["anti_ext"], separators=(",", ":"))
        model_name = spec["usedModel"]["modelName"]
        x_chat_msg = (
            f"query:{urllib.parse.quote(query)},"
            f"anti_ext:{urllib.parse.quote(anti_ext)},"
            f"enter_type:{spec['enter_type']},re_rank:{rank},modelName:{model_name},sa:bkb"
        )
        is_deepseek = "1" if spec.get("is_deepseek") else "0"
        headers = {
            **self._base_headers(),
            "Content-Type": "application/json",
            "X-Chat-Message": x_chat_msg,
            "isDeepseek": is_deepseek,
            "landingpageswitch": "",
            "personifiedSwitch": "0",
            "source": "pc_csaitab",
        }
        if self._user_cookies:
            # send the exact cookie string the user provided (mirrors the web)
            headers["Cookie"] = self._user_cookies
        return headers

    # ------------------------------------------------------------------
    # SSE parsing
    # ------------------------------------------------------------------
    async def _iter_sse(
        self, response: aiohttp.ClientResponse
    ) -> AsyncIterator[dict]:
        """Yield parsed SSE events {event, data}. Aborts after ``stall_seconds``."""
        buffer = ""
        last_data_time = time.time()
        async for raw_chunk in response.content:
            chunk = raw_chunk.decode("utf-8", errors="replace")
            if not chunk:
                if time.time() - last_data_time > self.stall_seconds:
                    return
                continue
            last_data_time = time.time()
            buffer += chunk.replace("\r\n", "\n")
            while "\n\n" in buffer:
                event_text, buffer = buffer.split("\n\n", 1)
                event_type = ""
                event_data = ""
                for line in event_text.strip().split("\n"):
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        event_data = line[5:].strip()
                if event_data:
                    try:
                        data = json.loads(event_data)
                    except json.JSONDecodeError:
                        data = {"raw": event_data}
                    yield {"event": event_type, "data": data}
        if buffer:
            event_text = buffer.strip()
            event_type = ""
            event_data = ""
            for line in event_text.split("\n"):
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    event_data = line[5:].strip()
            if event_data:
                try:
                    yield {"event": event_type, "data": json.loads(event_data)}
                except json.JSONDecodeError:
                    pass

    # ------------------------------------------------------------------
    # Public chat API (low level: raw SSE events)
    # ------------------------------------------------------------------
    async def chat(
        self,
        query: str,
        model: str = DEFAULT_MODEL,
        deep_search: bool = False,
        internet_search: bool = False,
        rank: Optional[int] = None,
    ) -> AsyncIterator[dict]:
        """Yield raw SSE events {event, data} for a single query."""
        await self.ready()
        self._ensure_base_data()

        payload = self._build_message_payload(query, model, deep_search, internet_search, rank)
        effective_rank = payload["rank"]
        headers = self._build_headers(query, model, effective_rank)
        session = await self._get_session()

        resp = await self._post_conversation(session, headers, payload)
        try:
            async for ev in self._iter_sse(resp):
                yield ev
        finally:
            resp.release()

    async def _post_conversation(self, session, headers, payload):
        resp = await session.post(
            CONVERSATION_API, headers=headers, json=payload, proxy=self._current_proxy()
        )
        if resp.status in (401, 403):
            body = await resp.text()
            resp.release()
            await self._refresh()
            resp = await session.post(
                CONVERSATION_API, headers=headers, json=payload, proxy=self._current_proxy()
            )
        if not 200 <= resp.status < 300:
            body = await resp.text(errors="replace")
            resp.release()
            self._report_proxy_result(False)
            raise BaiduError(f"Baidu API returned {resp.status}: {body[:500]}")
        self._report_proxy_result(True)
        return resp

    # ------------------------------------------------------------------
    # High level: normalized event stream
    # ------------------------------------------------------------------
    async def stream(
        self,
        query: str,
        model: str = DEFAULT_MODEL,
        deep_search: bool = False,
        internet_search: bool = False,
    ) -> AsyncIterator[dict]:
        """Yield normalized events: text, thinking, search_query, done, ping, metadata.

        Event shapes:
          {"type": "text", "content": "..."}
          {"type": "thinking", "content": "..."}
          {"type": "search_query", "content": "..."}
          {"type": "done", "content": ""}
          {"type": "ping", "content": ""}
          {"type": "metadata", "content": {...basedata...}}
        """
        msg_count = 0
        async for ev in self.chat(query, model, deep_search, internet_search):
            if ev["event"] == "message":
                msg_count += 1
                msg_data = ev["data"]
                data = msg_data.get("data", {})
                message = data.get("message", {})
                if message:
                    content = message.get("content", {})
                    meta = message.get("metaData", {})
                    state = meta.get("state", "")
                else:
                    content = data.get("content", {})
                    state = content.get("state", "")
                    meta = {}

                generator = content.get("generator")
                if isinstance(generator, dict):
                    comp = generator.get("component", "")
                    gen_data = generator.get("data", {})
                    if comp == "markdown-yiyan" and isinstance(gen_data, dict):
                        value = gen_data.get("value", "")
                        if value:
                            yield {"type": "text", "content": value}
                    elif comp == "thinkingSteps" and isinstance(gen_data, dict):
                        arr = gen_data.get("reasoningContentArr", [])
                        text = "".join(str(x) for x in arr if x) if isinstance(arr, list) else ""
                        if text:
                            yield {"type": "thinking", "content": text}
                    elif comp == "searchResult" and isinstance(gen_data, dict):
                        pass
                    elif comp == "questionClosely":
                        pass
                elif isinstance(generator, str) and generator:
                    yield {"type": "text", "content": generator}

                if "searchQuery" in content:
                    sq = content["searchQuery"]
                    if isinstance(sq, dict) and sq.get("query"):
                        yield {"type": "search_query", "content": sq["query"]}

                if state in ("completed", "success", "done", "generate-complete") or meta.get("endTurn"):
                    yield {"type": "done", "content": ""}

            elif ev["event"] == "basedata":
                yield {"type": "metadata", "content": ev["data"]}
            elif ev["event"] == "ping":
                yield {"type": "ping", "content": ""}

    # ------------------------------------------------------------------
    # Convenience: full text / OpenAI-style chunks / sync reply
    # ------------------------------------------------------------------
    async def generate(
        self,
        query: str,
        model: str = DEFAULT_MODEL,
        deep_search: bool = False,
        internet_search: bool = False,
    ) -> str:
        """Return the complete answer as a single string."""
        parts: list[str] = []
        async for chunk in self.stream(query, model, deep_search, internet_search):
            if chunk["type"] == "text":
                parts.append(chunk["content"])
        return "".join(parts)

    async def chat_to_openai_chunks(
        self,
        query: str,
        model: str = DEFAULT_MODEL,
        deep_search: bool = False,
        internet_search: bool = False,
    ) -> AsyncIterator[dict]:
        """Yield OpenAI-style chunks: content / reasoning_content / done."""
        async for chunk in self.stream(query, model, deep_search, internet_search):
            ct = chunk["type"]
            if ct == "text":
                yield {"type": "content", "content": chunk["content"]}
            elif ct == "thinking":
                yield {"type": "reasoning_content", "content": chunk["content"]}
            elif ct == "done":
                break

    async def chat_to_openai_sync(
        self,
        query: str,
        model: str = DEFAULT_MODEL,
        deep_search: bool = False,
        internet_search: bool = False,
    ) -> dict:
        """Return {"role", "content", "reasoning_content"} for one query."""
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        async for chunk in self.stream(query, model, deep_search, internet_search):
            ct = chunk["type"]
            if ct == "text":
                text_parts.append(chunk["content"])
            elif ct == "thinking":
                thinking_parts.append(chunk["content"])
        content = "".join(text_parts)
        reasoning = "".join(thinking_parts) if thinking_parts else None
        if not content:
            raise BaiduError("Baidu returned no answer (empty content)")
        return {"role": "assistant", "content": content, "reasoning_content": reasoning}

    # ------------------------------------------------------------------
    # OpenAI-style chat completions (messages + tools + streaming)
    # ------------------------------------------------------------------
    async def chat_completions(
        self,
        messages: list,
        model: str = DEFAULT_MODEL,
        tools: Optional[list] = None,
        tool_choice: Any = None,
        deep_search: bool = False,
        internet_search: bool = False,
        max_chars: int = 12000,
        max_messages: int = 16,
        max_message_chars: int = 2000,
    ) -> AsyncIterator[dict]:
        """Yield OpenAI-style chunks for a full messages array.

        Emulates tool calling: the tools are injected as a system prompt and
        ``<tool_calls>`` XML output is converted back to ``tool_calls``.
        Chunks: {"type": "content"|"reasoning_content"|"tool_calls"|"done",
        "content"|"tool_calls"}.
        """
        from .tool_calling import messages_to_prompt, parse_tool_calls

        query = messages_to_prompt(
            messages, tools, tool_choice,
            max_chars=max_chars, max_messages=max_messages,
            max_message_chars=max_message_chars,
        )
        if not query:
            raise BaiduError("No user message found in messages")

        has_tools = bool(tools)
        content_parts: list[str] = []
        async for chunk in self.stream(query, model, deep_search, internet_search):
            ct = chunk["type"]
            if ct == "text":
                content_parts.append(chunk["content"])
                if not has_tools:
                    yield {"type": "content", "content": chunk["content"]}
            elif ct == "thinking":
                yield {"type": "reasoning_content", "content": chunk["content"]}
            elif ct == "done":
                break

        full_content = "".join(content_parts)
        if has_tools:
            parsed_content, tool_calls = parse_tool_calls(full_content)
            if tool_calls:
                if parsed_content:
                    yield {"type": "content", "content": parsed_content}
                yield {"type": "tool_calls", "tool_calls": tool_calls}
            elif parsed_content:
                yield {"type": "content", "content": parsed_content}
        yield {"type": "done", "content": ""}
