"""Async reverse-engineered client for Baidu Wenxin (chat.baidu.com).

Pure-algorithm implementation (no browser automation for chatting). It does what
the official web client does:

1. ``GET /aichat/api/aitabserver?ctl=baseData`` -> ``token`` + ``lid`` (and the
   anti-bot cookies); falls back to scraping the inline ``aiTabFrameBaseData``.
2. Build the chat token: ``base64("{token}|{MD5(query)}|{ms}|{lid}")-{lid}-3``.
3. ``POST /aichat/api/conversation`` and stream the SSE events
   (``markdown-yiyan`` = answer text, ``thinkingSteps`` = reasoning).

Behaviour worth knowing (all verified live):

* Each call starts a **fresh server session** (``ori_lid=""``). Pass a
  :class:`~baidu_reverse.session.Conversation` to continue one: only the new
  message is sent, the server remembers the rest.
* The ``X-Chat-Message`` header carries the (URL-encoded) query and the server
  flags requests whose header is too long, so it is truncated; the JSON body
  itself accepts 50k+ characters.
* When Baidu flags the session (``kunlun``) the answer is empty. With
  ``auto_solve=True`` the client solves the rotate-captcha in a real browser,
  adopts the verified cookies and retries.
"""

from __future__ import annotations

import asyncio
import base64
import codecs
import contextvars
import hashlib
import json
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional, Union

import aiohttp

from . import tool_calling as tc
from .cache import ResponseCache
from .models import (
    DEFAULT_MODEL,
    MODEL_ALIASES,
    MODELS,
    apply_think_suffix,
    resolve_model,
)
from .session import Conversation, PrefixSessionCache

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

#: Max URL-encoded length of the query inside ``X-Chat-Message`` (500 ok, 775 flagged).
HEADER_QUERY_MAX = 400

_DONE_STATES = ("completed", "success", "done", "generate-complete")

#: proxy used by the request being served in the current task (concurrency-safe)
_proxy_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("baidu_proxy", default=None)

#: statuses that mean "this proxy is bad", not "Baidu said no"
_PROXY_BAD_STATUSES = (407, 429, 502, 503, 504)
_NET_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError, OSError)


class BaiduError(Exception):
    pass


class BaiduAuthError(BaiduError):
    pass


class BaiduBlockedError(BaiduError):
    """Baidu's ``kunlun`` anti-bot answered nothing (captcha / rate limit)."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


#: alias, the common case is a captcha being required
BaiduCaptchaError = BaiduBlockedError


def _resolve_model(model: str) -> str:
    return resolve_model(model)


def _apply_model_suffix(spec: dict, model: str) -> dict:
    """Copy a spec and apply -thinking / -nothinking suffix overrides."""
    return apply_think_suffix(spec, model)


def _clip_header_query(query: str, max_encoded: int = HEADER_QUERY_MAX) -> str:
    """Longest prefix of ``query`` whose URL-encoding fits ``max_encoded`` chars."""
    out: list[str] = []
    size = 0
    for ch in query:
        n = len(urllib.parse.quote(ch))
        if size + n > max_encoded:
            break
        out.append(ch)
        size += n
    return "".join(out)


def _parse_cookie_string(cookie_str: str) -> dict[str, str]:
    jar: dict[str, str] = {}
    for part in (cookie_str or "").split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip():
                jar[k.strip()] = v.strip()
    return jar


class BaiduChat:
    """Async client for chat.baidu.com (Wenxin).

    Usage::

        async with BaiduChat() as chat:
            async for chunk in chat.stream("Explain what a kernel is"):
                if chunk["type"] == "text":
                    print(chunk["content"], end="", flush=True)

    Args:
        cookies: browser cookie string; sent verbatim. Optional: without it the
            client lets the server mint anonymous ones.
        auto_solve: on a ``kunlun`` block, solve the captcha with a real browser
            (needs ``pip install baidu-reverse[captcha]``) and retry.
        cookie_file: JSON file to load/save the cookies (defaults to
            ``~/.cache/baidu_reverse/cookies.json`` when ``auto_solve`` is on).
        cache: ``True`` / a :class:`~baidu_reverse.cache.ResponseCache` to answer
            identical stateless requests locally.
        min_interval: minimum seconds between requests (kunlun rate limit).
        solver_options: kwargs for :func:`baidu_reverse.captcha.browser.solve_and_get_cookies`.
    """

    def __init__(
        self,
        cookies: Optional[str] = None,
        user_agent: Optional[str] = None,
        proxy: Optional[str] = None,
        proxies: Optional[list[str]] = None,
        timeout: int = 60,
        stall_seconds: float = 25.0,
        *,
        auto_solve: bool = False,
        cookie_file: Optional[str] = None,
        cache: Union[bool, ResponseCache, None] = None,
        min_interval: float = 1.0,
        max_block_retries: int = 2,
        solver_options: Optional[dict] = None,
        rotate_proxies: bool = False,
        proxy_pool: Any = None,
        proxy_retries: int = 4,
    ):
        self.user_agent = user_agent or _USER_AGENT
        self.timeout = timeout
        self.stall_seconds = stall_seconds
        self.auto_solve = auto_solve
        self.min_interval = min_interval
        self.max_block_retries = max_block_retries
        self.solver_options = dict(solver_options or {})
        self.session: Optional[aiohttp.ClientSession] = None
        self._token: Optional[str] = None
        self._lid: Optional[str] = None
        self._ori_lid: Optional[str] = None
        self._last_request = 0.0
        self._cookie_epoch = 0
        self._cookies_dirty = False
        self._solve_lock: Optional[asyncio.Lock] = None
        self._user_supplied_cookies = bool(cookies)

        # cookies: explicit string > cookie file
        self._cookies: dict[str, str] = _parse_cookie_string(cookies or "")
        if cookie_file is None and auto_solve:
            from .paths import default_cookie_file

            cookie_file = str(default_cookie_file())
        self.cookie_file = cookie_file
        if not self._cookies and cookie_file:
            self._load_cookie_file()

        if cache is True:
            cache = ResponseCache()
        # NB: never `cache or None` - an empty ResponseCache has len() == 0 (falsy)
        self.cache: Optional[ResponseCache] = cache if isinstance(cache, ResponseCache) else None

        self.proxy_retries = proxy_retries
        self._last_latency = 0.0
        self._proxy_pool: Any = None
        self._owns_pool = proxy_pool is None  # a pool passed in is the caller's to close
        if proxy_pool is not None:
            self._proxy_pool = proxy_pool
        elif rotate_proxies:
            from .proxies import ProxyManager

            self._proxy_pool = ProxyManager()  # scrapes + verifies + refills itself
        elif proxies:
            from .proxies import ProxyPool

            self._proxy_pool = ProxyPool(proxies)
        if self._proxy_pool is not None:
            proxy = None  # a pool replaces the static proxy
        self.proxy = proxy

    # ------------------------------------------------------------------
    # Proxies
    # ------------------------------------------------------------------
    @property
    def proxy_pool(self) -> Any:
        return self._proxy_pool

    def _current_proxy(self) -> Optional[str]:
        """Proxy URL for the next request (pool-aware); remembered per task."""
        if self._proxy_pool is not None:
            p = self._proxy_pool.next()
            _proxy_var.set(p)
            return f"http://{p}" if p else None
        return self.proxy

    def _report_proxy_result(self, ok: bool, blocked: bool = False, latency: Optional[float] = None) -> None:
        pool, addr = self._proxy_pool, _proxy_var.get()
        if pool is None or not addr:
            return
        if blocked:
            pool.report_blocked(addr)
        elif ok:
            pool.report_success(addr, latency)
        else:
            pool.report_failure(addr)

    async def _ensure_proxies(self) -> None:
        if self._proxy_pool is not None:
            await self._proxy_pool.ensure()

    async def _send(self, session: aiohttp.ClientSession, method: str, url: str, **kw: Any) -> aiohttp.ClientResponse:
        """One HTTP request; with a proxy pool, retries through other proxies.

        A proxy that fails to connect (or answers 407/429/5xx itself) is reported to
        the pool and the next one is tried, up to ``proxy_retries`` times.
        """
        if self._proxy_pool is None:
            return await session.request(method, url, proxy=self.proxy, **kw)
        last: Optional[BaseException] = None
        for _ in range(self.proxy_retries + 1):
            proxy = self._current_proxy()
            t0 = time.monotonic()
            try:
                resp = await session.request(method, url, proxy=proxy, **kw)
            except _NET_ERRORS as exc:
                last = exc
                self._report_proxy_result(False)
                continue
            if proxy and resp.status in _PROXY_BAD_STATUSES:
                resp.release()
                last = BaiduError(f"proxy answered {resp.status}")
                self._report_proxy_result(False)
                continue
            self._last_latency = time.monotonic() - t0
            return resp
        raise BaiduError(f"all {self.proxy_retries + 1} proxy attempts failed: {last!r}") from last

    # ------------------------------------------------------------------
    # Cookies
    # ------------------------------------------------------------------
    def _cookie_string(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self._cookies.items())

    def _set_cookies(self, cookie_str: str) -> None:
        self._cookies = _parse_cookie_string(cookie_str)
        self._cookie_epoch += 1
        self._cookies_dirty = True
        self._user_supplied_cookies = True

    def _absorb_cookies(self, resp: aiohttp.ClientResponse) -> None:
        try:
            for name, morsel in resp.cookies.items():
                if name and self._cookies.get(name) != morsel.value:
                    self._cookies[name] = morsel.value
                    self._cookies_dirty = True
        except Exception:
            pass

    def _load_cookie_file(self) -> None:
        try:
            data = json.loads(Path(self.cookie_file).read_text(encoding="utf-8"))
            self._cookies = _parse_cookie_string(data.get("cookies", ""))
            self._user_supplied_cookies = bool(self._cookies)
        except Exception:
            pass

    def _save_cookie_file(self) -> None:
        if not self.cookie_file or not self._cookies:
            return
        try:
            p = Path(self.cookie_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps({"cookies": self._cookie_string(), "saved_at": time.time()}),
                encoding="utf-8",
            )
            self._cookies_dirty = False
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def _base_headers(self) -> dict:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/event-stream",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": f"{BASE_URL}/",
            "Origin": BASE_URL,
        }
        if self._cookies:
            headers["Cookie"] = self._cookie_string()
        return headers

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            # We manage cookies ourselves (sent verbatim, like the web does).
            self.session = aiohttp.ClientSession(
                cookie_jar=aiohttp.DummyCookieJar(),
                timeout=aiohttp.ClientTimeout(total=None, connect=20, sock_read=self.timeout),
            )
        return self.session

    async def close(self) -> None:
        if self._cookies_dirty:
            self._save_cookie_file()
        if self._proxy_pool is not None and self._owns_pool:
            await self._proxy_pool.close()
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
        headers = {**self._base_headers(), "Accept": "application/json, text/plain, */*"}
        url = f"{BASE_URL}/aichat/api/aitabserver?ctl=baseData"
        resp = await self._send(session, "GET", url, headers=headers)
        try:
            resp.raise_for_status()
            self._absorb_cookies(resp)
            data = await resp.json(content_type=None)
        finally:
            resp.release()

        frame = (data.get("data") or {}).get("frameBaseData") or {}
        if not (frame.get("token") and frame.get("lid")):
            # fallback: scrape the homepage inline script
            headers2 = {**self._base_headers(), "Accept": "text/html"}
            resp2 = await self._send(session, "GET", BASE_URL, headers=headers2)
            try:
                resp2.raise_for_status()
                self._absorb_cookies(resp2)
                text = await resp2.text(encoding="utf-8", errors="replace")
            finally:
                resp2.release()
            match = _BASE_DATA_RE.search(text)
            if not match:
                raise BaiduError(
                    "Could not find aiTabFrameBaseData in chat.baidu.com homepage - "
                    "Baidu may have changed the page structure"
                )
            frame = json.loads(match.group(1))
        self._token = frame.get("token", "")
        self._lid = frame.get("lid", "")
        self._ori_lid = (frame.get("logParams") or {}).get("applid") or self._lid
        return frame

    def _ensure_base_data(self) -> None:
        if not self._token or not self._lid:
            raise BaiduError("BaiduChat has no token/lid - call await chat.ready() first")

    async def ready(self) -> None:
        """Fetch homepage bootstrap data (token + lid + cookies). Idempotent."""
        if self._token and self._lid:
            return
        await self._ensure_proxies()
        tries = self.proxy_retries + 1 if self._proxy_pool is not None else 1
        for i in range(tries):
            try:
                await self._fetch_base_data()
                self._report_proxy_result(True, latency=self._last_latency)
                return
            except (*_NET_ERRORS, ValueError, BaiduError):
                # garbage JSON / tampering proxy: try another exit
                if self._proxy_pool is None or i == tries - 1:
                    raise
                self._report_proxy_result(False)

    async def _refresh(self) -> None:
        self._token = None
        self._lid = None
        self._ori_lid = None
        if not self._user_supplied_cookies:
            self._cookies.clear()
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
        conversation: Optional[Conversation] = None,
    ) -> dict:
        model_key = _resolve_model(model)
        spec = _apply_model_suffix(MODELS[model_key], model)
        model_cfg = dict(spec["usedModel"])
        if rank is None:
            rank = spec["rank"] + (conversation.turn if conversation is not None else 0)
        if deep_search:
            model_cfg["modelFunction"]["deepSearch"] = "1"
        if internet_search:
            model_cfg["modelFunction"]["internetSearch"] = "1"
        chat_token = self._generate_chat_token(query)
        enter_type = spec["enter_type"]
        agt_sess_cnt = spec["agt_sess_cnt"]
        # "" starts a brand-new server session; a session id continues one
        ori_lid = conversation.session_id if conversation is not None else ""
        return {
            "message": {
                "inputMethod": "chat_search",
                "isRebuild": False,
                "content": {
                    "query": "",
                    "agentInfo": {
                        "agent_id": [""],
                        "params": f'{{"agt_rk":{rank},"agt_sess_cnt":{agt_sess_cnt}}}',
                    },
                    "agentInfoList": [],
                    "qtype": 0,
                    "extData": {},
                },
                "searchInfo": {
                    "srcid": "",
                    "order": "",
                    "tplname": "",
                    "dqaKey": "",
                    "re_rank": str(rank),
                    "ori_lid": ori_lid,
                    "sa": "bkb",
                    "enter_type": enter_type,
                    "chatParams": {
                        "setype": "csaitab",
                        "chat_samples": "WISE_NEW_CSAITAB",
                        "chat_token": chat_token,
                        "scene": "",
                    },
                    "isPrivateChat": False,
                    "usedModel": model_cfg,
                    "landingPageSwitch": "",
                    "landingPage": "aitab",
                    "ecomFrom": "",
                    "hasLocPermission": "",
                    "isInnovate": 2,
                    "applid": "",
                    "a_lid": "",
                    "capsuleSelectMode": "fast",
                    "showMindMap": False,
                    "deepDecisionInfo": {"isDeepDecision": 0},
                },
                "from": "",
                "source": "pc_csaitab",
                "query": [
                    {"type": "TEXT", "data": {"text": {"query": query, "extData": "{}", "text_type": ""}}}
                ],
                "anti_ext": spec["anti_ext"],
            },
            "sa": "bkb",
            "setype": "csaitab",
            "rank": rank,
        }

    def _build_headers(self, query: str, model: str, rank: int) -> dict:
        spec = _apply_model_suffix(MODELS[_resolve_model(model)], model)
        anti_ext = json.dumps(spec["anti_ext"], separators=(",", ":"))
        model_name = spec["usedModel"]["modelName"]
        x_chat_msg = (
            f"query:{urllib.parse.quote(_clip_header_query(query))},"
            f"anti_ext:{urllib.parse.quote(anti_ext)},"
            f"enter_type:{spec['enter_type']},re_rank:{rank},modelName:{model_name},sa:bkb"
        )
        return {
            **self._base_headers(),
            "Content-Type": "application/json",
            "X-Chat-Message": x_chat_msg,
            "isDeepseek": "1" if spec.get("is_deepseek") else "0",
            "landingpageswitch": "",
            "personifiedSwitch": "0",
            "source": "pc_csaitab",
        }

    # ------------------------------------------------------------------
    # SSE parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_sse_block(event_text: str) -> Optional[dict]:
        event_type = ""
        data_lines: list[str] = []
        for line in event_text.strip().split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        raw = "".join(data_lines)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"raw": raw}
        return {"event": event_type, "data": data}

    async def _iter_sse(self, response: aiohttp.ClientResponse) -> AsyncIterator[dict]:
        """Yield parsed SSE events ``{event, data}``.

        Ends the stream if no data arrives for ``stall_seconds`` (instead of hanging).
        """
        content = response.content
        source = content.iter_any() if hasattr(content, "iter_any") else content
        it = source.__aiter__()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        buffer = ""
        while True:
            try:
                raw_chunk = await asyncio.wait_for(it.__anext__(), timeout=self.stall_seconds)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                return
            buffer += decoder.decode(raw_chunk).replace("\r\n", "\n")
            while "\n\n" in buffer:
                event_text, buffer = buffer.split("\n\n", 1)
                ev = self._parse_sse_block(event_text)
                if ev is not None:
                    yield ev
        buffer += decoder.decode(b"", final=True)
        if buffer.strip():
            ev = self._parse_sse_block(buffer)
            if ev is not None:
                yield ev

    # ------------------------------------------------------------------
    # Public chat API (low level: raw SSE events)
    # ------------------------------------------------------------------
    async def _throttle(self) -> None:
        if self.min_interval > 0:
            wait = self._last_request + self.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
        self._last_request = time.monotonic()

    async def chat(
        self,
        query: str,
        model: str = DEFAULT_MODEL,
        deep_search: bool = False,
        internet_search: bool = False,
        rank: Optional[int] = None,
        conversation: Optional[Conversation] = None,
    ) -> AsyncIterator[dict]:
        """Yield raw SSE events ``{event, data}`` for a single query."""
        await self.ready()
        self._ensure_base_data()

        def build() -> tuple[dict, dict]:
            payload = self._build_message_payload(
                query, model, deep_search, internet_search, rank, conversation
            )
            return self._build_headers(query, model, payload["rank"]), payload

        headers, payload = build()
        session = await self._get_session()
        await self._ensure_proxies()
        await self._throttle()
        resp = await self._post_conversation(session, headers, payload, rebuild=build)
        try:
            async for ev in self._iter_sse(resp):
                if conversation is not None and ev["event"] == "message" and not conversation.session_id:
                    sid = ev["data"].get("sessionId") if isinstance(ev["data"], dict) else None
                    if sid:
                        conversation.session_id = str(sid)
                yield ev
        finally:
            resp.release()

    async def _post_conversation(
        self,
        session: aiohttp.ClientSession,
        headers: dict,
        payload: dict,
        rebuild: Optional[Callable[[], tuple[dict, dict]]] = None,
    ) -> aiohttp.ClientResponse:
        resp = await self._send(session, "POST", CONVERSATION_API, headers=headers, json=payload)
        if resp.status in (401, 403):
            resp.release()
            await self._refresh()
            if rebuild is not None:
                headers, payload = rebuild()  # fresh token/lid/cookies
            resp = await self._send(session, "POST", CONVERSATION_API, headers=headers, json=payload)
        self._absorb_cookies(resp)
        if not 200 <= resp.status < 300:
            body = await resp.text(errors="replace")
            resp.release()
            self._report_proxy_result(False)
            raise BaiduError(f"Baidu API returned {resp.status}: {body[:500]}")
        return resp

    # ------------------------------------------------------------------
    # Event normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_event(ev: dict) -> list[dict]:
        """Raw SSE event -> list of normalized events.

        Shapes: ``text``, ``thinking``, ``search_query``, ``done``, ``ping``,
        ``metadata`` and ``blocked`` (kunlun anti-bot hint; internal).
        """
        out: list[dict] = []
        kind = ev.get("event")
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if kind == "message":
            inner = data.get("data", {})
            message = inner.get("message", {})
            if message:
                content = message.get("content", {})
                meta = message.get("metaData", {})
                state = meta.get("state", "")
            else:
                content = inner.get("content", {})
                state = content.get("state", "")
                meta = {}

            for part in (content.get("hints") or {}).get("parts", []) or []:
                if isinstance(part, dict) and part.get("type") == "kunlunHint":
                    out.append({"type": "blocked", "content": part.get("text", "")})

            generator = content.get("generator")
            if isinstance(generator, dict):
                comp = generator.get("component", "")
                gen_data = generator.get("data", {})
                if comp == "markdown-yiyan" and isinstance(gen_data, dict):
                    value = gen_data.get("value", "")
                    if value:
                        out.append({"type": "text", "content": value})
                elif comp == "thinkingSteps" and isinstance(gen_data, dict):
                    arr = gen_data.get("reasoningContentArr", [])
                    text = "".join(str(x) for x in arr if x) if isinstance(arr, list) else ""
                    if text:
                        out.append({"type": "thinking", "content": text})
            elif isinstance(generator, str) and generator:
                out.append({"type": "text", "content": generator})

            sq = content.get("searchQuery")
            if isinstance(sq, dict) and sq.get("query"):
                out.append({"type": "search_query", "content": sq["query"]})

            if state in _DONE_STATES or meta.get("endTurn"):
                out.append({"type": "done", "content": ""})
        elif kind == "basedata":
            if data.get("chatHitKunlun"):
                out.append({"type": "blocked", "content": str(data.get("chatHitKunlun"))})
            out.append({"type": "metadata", "content": ev.get("data")})
        elif kind == "ping":
            out.append({"type": "ping", "content": ""})
        return out

    async def _stream_once(
        self,
        query: str,
        model: str,
        deep_search: bool,
        internet_search: bool,
        conversation: Optional[Conversation],
    ) -> AsyncIterator[dict]:
        kwargs = {"conversation": conversation} if conversation is not None else {}
        agen = self.chat(query, model, deep_search, internet_search, **kwargs)
        try:
            async for ev in agen:
                for out in self._normalize_event(ev):
                    yield out
        finally:
            aclose = getattr(agen, "aclose", None)
            if aclose is not None:
                await aclose()

    # ------------------------------------------------------------------
    # Captcha / block recovery
    # ------------------------------------------------------------------
    async def solve_captcha(self) -> str:
        """Solve Baidu's captcha in a real browser and adopt the verified cookies.

        Needs ``pip install baidu-reverse[captcha]`` and a local Brave/Chrome/Edge.
        Returns the new cookie string.
        """
        if self._solve_lock is None:
            self._solve_lock = asyncio.Lock()
        epoch = self._cookie_epoch
        async with self._solve_lock:
            if self._cookie_epoch != epoch:  # someone else just refreshed them
                return self._cookie_string()
            try:
                from .captcha.browser import solve_and_get_cookies
            except ImportError as exc:  # pragma: no cover
                raise BaiduError(
                    "captcha solving needs extras: pip install 'baidu-reverse[captcha]'"
                ) from exc
            cookies = await solve_and_get_cookies(seed_cookies=self._cookie_string(), **self.solver_options)
            self._set_cookies(cookies)
            self._token = self._lid = self._ori_lid = None
            self._save_cookie_file()
            return cookies

    async def _recover_from_block(self, attempt: int, hint: str) -> bool:
        """Decide whether to retry a blocked request.

        kunlun limits per IP, so with a proxy pool the first recovery is simply to
        leave through another exit (the blocked one is rested for a long while);
        only after that (or without a pool) is the captcha solved.
        """
        if self._proxy_pool is not None:
            self._report_proxy_result(False, blocked=True)
            if attempt < self.proxy_retries:
                await self._ensure_proxies()
                return True
            attempt -= self.proxy_retries
        if not self.auto_solve or attempt >= self.max_block_retries:
            return False
        await self.solve_captcha()
        return True

    # ------------------------------------------------------------------
    # High level: normalized event stream
    # ------------------------------------------------------------------
    async def stream(
        self,
        query: str,
        model: str = DEFAULT_MODEL,
        deep_search: bool = False,
        internet_search: bool = False,
        conversation: Optional[Conversation] = None,
        use_cache: bool = True,
    ) -> AsyncIterator[dict]:
        """Yield normalized events: text, thinking, search_query, done, ping, metadata.

        Event shapes::

          {"type": "text", "content": "..."}
          {"type": "thinking", "content": "..."}
          {"type": "search_query", "content": "..."}
          {"type": "done", "content": ""}
          {"type": "ping", "content": ""}
          {"type": "metadata", "content": {...basedata...}}

        Pass a :class:`Conversation` to continue a server-side session (only the
        new message is sent). Raises :class:`BaiduBlockedError` if Baidu answers
        nothing because of its anti-bot (unless ``auto_solve`` recovers it).
        """
        cache_key = None
        if self.cache is not None and conversation is None and use_cache:
            cache_key = self.cache.make_key(model, deep_search, internet_search, query)
            hit = self.cache.get(cache_key)
            if hit is not None:
                for ev in hit:
                    yield ev
                return

        attempt = 0
        while True:
            got_content = False
            blocked = ""
            recorded: list[dict] = []
            agen = self._stream_once(query, model, deep_search, internet_search, conversation)
            try:
                async for ev in agen:
                    t = ev["type"]
                    if t == "blocked":
                        blocked = blocked or ev["content"] or "blocked"
                        continue
                    if t == "done" and blocked and not got_content:
                        continue  # a blocked stream is not a finished answer
                    if t in ("text", "thinking"):
                        got_content = True
                    if t not in ("metadata", "ping"):
                        recorded.append(ev)
                    yield ev
            finally:
                await agen.aclose()
            if got_content or not blocked:
                break
            if not await self._recover_from_block(attempt, blocked):
                raise BaiduBlockedError(
                    f"Baidu blocked the request ({blocked}). Pass auto_solve=True to "
                    "solve the captcha automatically, or run `baidu-reverse solve-captcha`.",
                    hint=blocked,
                )
            attempt += 1

        if got_content:
            self._report_proxy_result(True, latency=self._last_latency)
        if conversation is not None and got_content:
            conversation.turn += 1
        if cache_key is not None and any(e["type"] == "text" for e in recorded):
            self.cache.put(cache_key, recorded, model)

    # ------------------------------------------------------------------
    # Convenience: full text / OpenAI-style chunks / sync reply
    # ------------------------------------------------------------------
    async def generate(
        self,
        query: str,
        model: str = DEFAULT_MODEL,
        deep_search: bool = False,
        internet_search: bool = False,
        conversation: Optional[Conversation] = None,
    ) -> str:
        """Return the complete answer as a single string."""
        parts: list[str] = []
        kwargs = {"conversation": conversation} if conversation is not None else {}
        async for chunk in self.stream(query, model, deep_search, internet_search, **kwargs):
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
        """Yield OpenAI-style chunks: content / reasoning_content."""
        async for chunk in self.stream(query, model, deep_search, internet_search):
            ct = chunk["type"]
            if ct == "text":
                yield {"type": "content", "content": chunk["content"]}
            elif ct == "thinking":
                yield {"type": "reasoning_content", "content": chunk["content"]}

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
        max_chars: int = 40000,
        max_messages: int = 60,
        max_message_chars: int = 12000,
        session_cache: Optional[PrefixSessionCache] = None,
        tool_retries: int = 1,
    ) -> AsyncIterator[dict]:
        """Yield OpenAI-style chunks for a full ``messages`` array.

        Chunks: ``{"type": "content"|"reasoning_content"|"tool_calls"|"done", ...}``.

        Tool calling is emulated (see :mod:`baidu_reverse.tool_calling`): tools are
        described in the prompt, the XML reply is parsed back to ``tool_calls`` and
        validated against the schema; if a call was required (or the XML was
        malformed) the model is asked once more to fix it.

        With a ``session_cache`` the history is mapped to a live server session so
        that follow-up requests only send the new messages.
        """
        if tool_choice == "none":
            tools = None
        tools = tools or None
        tsig = tc.tools_signature(tools)

        hit = session_cache.lookup(messages, model, tsig) if session_cache is not None else None
        chars_sent = 0
        if hit is not None:
            conversation = hit.conversation
            chars_sent = hit.chars
            query = tc.render_delta(
                messages[hit.consumed :], tools, tool_choice, id_names=tc._call_id_names(messages)
            )
        else:
            conversation = Conversation()
            query = tc.messages_to_prompt(
                messages,
                tools,
                tool_choice,
                max_chars=max_chars,
                max_messages=max_messages,
                max_message_chars=max_message_chars,
            )
        if not query:
            raise BaiduError("No user message found in messages")

        attempt = 0
        yielded_content: list[str] = []
        final_calls: list[dict] = []
        forced = tc.wants_tool_call(tool_choice)
        while True:
            filt = tc.ToolStreamFilter() if tools else None
            # Prose is shown live, except when a retry may follow and would repeat it:
            # forced calls (the reply should be XML only) and every retry attempt.
            quiet = forced or attempt > 0
            content_parts: list[str] = []
            async for chunk in self.stream(query, model, deep_search, internet_search, conversation=conversation):
                ct = chunk["type"]
                if ct == "text":
                    content_parts.append(chunk["content"])
                    if filt is None:
                        yielded_content.append(chunk["content"])
                        yield {"type": "content", "content": chunk["content"]}
                    else:
                        out = filt.feed(chunk["content"])
                        if out and not quiet:
                            yielded_content.append(out)
                            yield {"type": "content", "content": out}
                elif ct == "thinking":
                    yield {"type": "reasoning_content", "content": chunk["content"]}
            full = "".join(content_parts)
            chars_sent += len(query) + len(full)

            if filt is None:
                break

            tail = filt.finish()
            if tail and not quiet:
                yielded_content.append(tail)
                yield {"type": "content", "content": tail}
            res = tc.extract_tool_calls(full, tools)
            failed = (not res.calls) and (forced or (res.found_markup and bool(res.errors)))
            if failed and attempt < tool_retries and conversation.session_id:
                attempt += 1
                query = tc.build_retry_prompt(res.errors, tools, tool_choice)
                continue

            if res.calls:
                final_calls = res.calls
            cleaned = res.text.strip()
            if quiet:
                # nothing was streamed: only surface prose when there is no call
                residual = "" if final_calls or yielded_content else cleaned
            else:
                streamed = filt.emitted.strip()
                residual = cleaned[len(streamed) :].strip() if cleaned.startswith(streamed) else ""
                if not streamed and cleaned:
                    residual = cleaned
            if residual:
                yielded_content.append(residual)
                yield {"type": "content", "content": residual}
            if final_calls:
                yield {"type": "tool_calls", "tool_calls": final_calls}
            break

        if session_cache is not None and conversation.session_id:
            reply = {
                "role": "assistant",
                "content": "".join(yielded_content).strip(),
                "tool_calls": final_calls or None,
            }
            session_cache.store(list(messages) + [reply], model, tsig, conversation, chars_sent)
        yield {"type": "done", "content": ""}

    async def complete(self, messages: list, model: str = DEFAULT_MODEL, **kwargs: Any) -> dict:
        """Non-streaming :meth:`chat_completions`: ``{"content", "reasoning_content", "tool_calls"}``."""
        content: list[str] = []
        reasoning: list[str] = []
        calls: list[dict] = []
        async for chunk in self.chat_completions(messages, model=model, **kwargs):
            ct = chunk["type"]
            if ct == "content":
                content.append(chunk["content"])
            elif ct == "reasoning_content":
                reasoning.append(chunk["content"])
            elif ct == "tool_calls":
                calls.extend(chunk["tool_calls"])
        return {
            "role": "assistant",
            "content": "".join(content).strip(),
            "reasoning_content": "".join(reasoning) or None,
            "tool_calls": calls or None,
        }
