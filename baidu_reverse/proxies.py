"""Proxy scraping, health checks and rotation for baidu-reverse.

Baidu's ``kunlun`` anti-bot rate-limits per IP, so a burst of requests from one
address ends in empty answers. Rotating through a pool of working proxies spreads
the load.

* :class:`ProxyPool` -- the rotation brain: latency-weighted random choice, automatic
  bans (short after a network failure, long after Baidu blocked that IP), forgetting
  proxies that keep failing.
* :class:`ProxyManager` -- a pool that *feeds itself*: scrapes ~10 public lists
  (~10k candidates), checks each against the real ``baseData`` endpoint (so a proxy
  must reach Baidu over HTTPS CONNECT, return the right JSON and not tamper with TLS),
  stops as soon as enough healthy ones are found, refills in the background when the
  pool runs low and caches the verified set on disk.

Only HTTP(S) proxies are used (what aiohttp speaks natively). Free proxies are
unreliable by nature: expect ~5% of the scraped ones to pass.

    async with BaiduChat(rotate_proxies=True) as chat:
        print(await chat.generate("hello"))
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import aiohttp

DEFAULT_SOURCES = [
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/http/data.txt",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/http.txt",
    "https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS_RAW.txt",
    "https://raw.githubusercontent.com/vakhov/fresh-proxy-list/master/http.txt",
    "https://raw.githubusercontent.com/zloi-user/hideip.me/main/http.txt",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=8000&country=all&ssl=all&anonymity=all",
    "https://api.openproxylist.xyz/http.txt",
]

TEST_URL = "https://chat.baidu.com/aichat/api/aitabserver?ctl=baseData"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
)
_PROXY_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3}):(\d{2,5})(?!\d)")


def _extract_proxies(text: str) -> list[str]:
    found = set()
    for host, port in _PROXY_RE.findall(text):
        if all(0 <= int(p) <= 255 for p in host.split(".")) and 0 < int(port) < 65536:
            found.add(f"{host}:{port}")
    return sorted(found)


class _QuietProactor(logging.Filter):
    """Windows' proactor logs a traceback for every proxy that resets the connection."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "_call_connection_lost" not in record.getMessage()


# ----------------------------------------------------------------------------
# rotation
# ----------------------------------------------------------------------------
@dataclass
class Proxy:
    addr: str
    latency: float = 3.0  # EWMA of seconds, lower is better
    ok: int = 0
    fail: int = 0  # consecutive network failures
    blocked: int = 0  # times Baidu answered "kunlun" through it
    banned_until: float = 0.0
    last_used: float = 0.0

    @property
    def url(self) -> str:
        return f"http://{self.addr}"

    def usable(self, now: float) -> bool:
        return now >= self.banned_until


class ProxyPool:
    """Latency-weighted rotation with automatic bans.

    ``next()`` returns an ``ip:port`` (or ``None`` when the pool is empty -> go direct).
    Report what happened with :meth:`report_success`, :meth:`report_failure` and
    :meth:`report_blocked`; the pool adapts by itself.
    """

    def __init__(
        self,
        proxies: Optional[Iterable[str]] = None,
        *,
        fail_cooldown: float = 30.0,
        block_cooldown: float = 600.0,
        max_fails: int = 3,
        rng: Optional[random.Random] = None,
    ):
        self.fail_cooldown = fail_cooldown
        self.block_cooldown = block_cooldown
        self.max_fails = max_fails
        self._rng = rng or random.Random()
        self._proxies: dict[str, Proxy] = {}
        for p in proxies or []:
            self.add(p)

    # -- compat -----------------------------------------------------------------
    @property
    def proxies(self) -> list[str]:
        return list(self._proxies)

    def __len__(self) -> int:
        return len(self._proxies)

    def __contains__(self, addr: str) -> bool:
        return addr in self._proxies

    # -- membership ---------------------------------------------------------------
    def add(self, addr: str, latency: Optional[float] = None) -> Proxy:
        p = self._proxies.get(addr)
        if p is None:
            p = self._proxies[addr] = Proxy(addr)
        if latency is not None:
            p.latency = latency
        return p

    def remove(self, addr: str) -> None:
        self._proxies.pop(addr, None)

    def healthy(self) -> list[Proxy]:
        now = time.time()
        return [p for p in self._proxies.values() if p.usable(now)]

    # -- choosing ---------------------------------------------------------------------
    def _weight(self, p: Proxy, now: float) -> float:
        speed = 1.0 / (p.latency + 0.5)
        rest = min(1.0, (now - p.last_used) / 5.0) + 0.1  # prefer ones not just used
        trust = 1.0 / (1.0 + p.blocked)
        return speed * rest * trust

    def next(self) -> Optional[str]:
        now = time.time()
        live = [p for p in self._proxies.values() if p.usable(now)]
        if not live:
            return None
        chosen = self._rng.choices(live, weights=[self._weight(p, now) for p in live], k=1)[0]
        chosen.last_used = now
        return chosen.addr

    # -- feedback -----------------------------------------------------------------
    def report_success(self, addr: Optional[str], latency: Optional[float] = None) -> None:
        p = self._proxies.get(addr or "")
        if p is None:
            return
        p.ok += 1
        p.fail = 0
        p.banned_until = 0.0
        if latency is not None:
            p.latency = 0.7 * p.latency + 0.3 * latency

    def report_failure(self, addr: Optional[str]) -> None:
        """Network-level failure: ban for ``fail_cooldown * fails^2``; drop after ``max_fails``."""
        p = self._proxies.get(addr or "")
        if p is None:
            return
        p.fail += 1
        if p.fail >= self.max_fails and p.ok == 0:
            self.remove(p.addr)  # never worked for us and keeps failing
            return
        p.banned_until = time.time() + min(900.0, self.fail_cooldown * p.fail**2)

    def report_blocked(self, addr: Optional[str]) -> None:
        """Baidu's kunlun answered nothing through this IP: rest it for a long while."""
        p = self._proxies.get(addr or "")
        if p is None:
            return
        p.blocked += 1
        p.banned_until = time.time() + self.block_cooldown * min(p.blocked, 4)

    def stats(self) -> dict:
        now = time.time()
        live = [p for p in self._proxies.values() if p.usable(now)]
        return {
            "total": len(self._proxies),
            "healthy": len(live),
            "banned": len(self._proxies) - len(live),
            "avg_latency": round(sum(p.latency for p in live) / len(live), 2) if live else None,
        }

    # -- ensure() is a no-op for static pools -----------------------------------------
    async def ensure(self) -> None:
        return None

    async def close(self) -> None:
        return None


# ----------------------------------------------------------------------------
# self-feeding pool
# ----------------------------------------------------------------------------
class ProxyManager(ProxyPool):
    """A :class:`ProxyPool` that scrapes, verifies, caches and refills itself.

    Args:
        target_size: stop checking once this many healthy proxies are found.
        min_healthy: when fewer are left, a refill runs (in the background while
            others are still usable).
        max_latency: reject proxies slower than this to reach Baidu (seconds).
        cache_file: where the verified set is stored (default
            ``~/.cache/baidu_reverse/proxies.json``); ``False`` disables it.
        test_url: endpoint used for the health check. It must return JSON with
            ``data.frameBaseData.token`` and ``lid`` (Baidu's bootstrap).
    """

    def __init__(
        self,
        *,
        sources: Optional[list[str]] = None,
        target_size: int = 25,
        min_healthy: int = 6,
        max_latency: float = 6.0,
        max_candidates: int = 2500,
        check_concurrency: int = 150,
        check_timeout: float = 8.0,
        cache_file: "str | bool | None" = None,
        cache_ttl: float = 1800.0,
        test_url: str = TEST_URL,
        user_agent: str = _UA,
        **pool_kwargs,
    ):
        super().__init__(**pool_kwargs)
        self.sources = list(sources or DEFAULT_SOURCES)
        self.target_size = target_size
        self.min_healthy = min_healthy
        self.max_latency = max_latency
        self.max_candidates = max_candidates
        self.check_concurrency = check_concurrency
        self.check_timeout = check_timeout
        self.cache_ttl = cache_ttl
        self.test_url = test_url
        self.user_agent = user_agent
        if cache_file is False:
            self.cache_file: Optional[Path] = None
        elif cache_file:
            self.cache_file = Path(cache_file)
        else:
            from .paths import cache_dir

            self.cache_file = cache_dir() / "proxies.json"
        self.source_stats: dict[str, int] = {}
        self._dead: dict[str, float] = {}  # candidates that failed a check recently
        self._lock = asyncio.Lock()
        self._bg: Optional[asyncio.Task] = None
        self.last_refill = 0.0
        self.log = lambda msg: None

    # -- scraping --------------------------------------------------------------------
    async def scrape(self) -> list[str]:
        """Download every source in parallel; returns unique ``ip:port`` candidates."""
        found: set[str] = set()
        self.source_stats = {}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:

            async def one(url: str) -> None:
                try:
                    async with session.get(url, headers={"User-Agent": self.user_agent}) as resp:
                        if resp.status == 200:
                            got = _extract_proxies(await resp.text(errors="replace"))
                            self.source_stats[url] = len(got)
                            found.update(got)
                            return
                except Exception:
                    pass
                self.source_stats[url] = 0

            await asyncio.gather(*(one(u) for u in self.sources))
        return sorted(found)

    # -- checking --------------------------------------------------------------------
    async def check(self, addr: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[float]:
        """Latency in seconds if ``addr`` really reaches Baidu, else ``None``.

        The proxy must tunnel HTTPS (certificate is verified, so TLS-intercepting
        proxies fail) and return Baidu's real bootstrap JSON.
        """
        own = session is None
        if own:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.check_timeout),
                cookie_jar=aiohttp.DummyCookieJar(),
            )
        t0 = time.monotonic()
        try:
            async with session.get(
                self.test_url,
                proxy=f"http://{addr}",
                headers={"User-Agent": self.user_agent, "Accept": "application/json, text/plain, */*"},
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(content_type=None)
            frame = (data.get("data") or {}).get("frameBaseData") or {}
            if frame.get("token") and frame.get("lid"):
                return time.monotonic() - t0
            return None
        except Exception:
            return None
        finally:
            if own:
                await session.close()

    async def _check_many(self, candidates: list[str], want: int) -> int:
        """Check candidates concurrently; stop scheduling once ``want`` are healthy."""
        sem = asyncio.Semaphore(self.check_concurrency)
        added = 0
        timeout = aiohttp.ClientTimeout(total=self.check_timeout)
        connector = aiohttp.TCPConnector(limit=0, force_close=True)
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector, cookie_jar=aiohttp.DummyCookieJar()
        ) as session:

            async def one(addr: str) -> None:
                nonlocal added
                if added >= want:
                    return
                async with sem:
                    if added >= want:
                        return
                    latency = await self.check(addr, session)
                if latency is not None and latency <= self.max_latency:
                    self.add(addr, latency)
                    added += 1
                else:
                    self._dead[addr] = time.time()

            await asyncio.gather(*(one(a) for a in candidates))
        return added

    # -- cache -----------------------------------------------------------------------
    def _load_cache(self) -> list[str]:
        if not self.cache_file or not self.cache_file.exists():
            return []
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            if time.time() - float(data.get("saved_at", 0)) > self.cache_ttl:
                return []
            return [str(x["addr"]) for x in data.get("proxies", []) if x.get("addr")]
        except Exception:
            return []

    def save_cache(self) -> None:
        if not self.cache_file:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                {"addr": p.addr, "latency": round(p.latency, 3), "ok": p.ok}
                for p in self._proxies.values()
                if p.blocked == 0
            ]
            self.cache_file.write_text(
                json.dumps({"saved_at": time.time(), "proxies": rows}), encoding="utf-8"
            )
        except Exception:
            pass

    # -- refill ----------------------------------------------------------------------
    async def refill(self, force: bool = False) -> dict:
        """Make sure the pool has ``target_size`` verified proxies. Returns a summary."""
        async with self._lock:
            qfilter = _QuietProactor()
            logging.getLogger("asyncio").addFilter(qfilter)
            try:
                started = time.time()
                have = len(self.healthy())
                if have >= self.target_size and not force:
                    return {"added": 0, **self.stats()}

                # 1) cached proxies from a previous run (re-verified, they go stale fast)
                cached = [a for a in self._load_cache() if a not in self._proxies]
                if cached:
                    got = await self._check_many(cached, self.target_size - have)
                    self.log(f"cache: {got}/{len(cached)} still work")

                # 2) scrape fresh ones until we reach the target
                have = len(self.healthy())
                if have < self.target_size:
                    now = time.time()
                    self._dead = {a: t for a, t in self._dead.items() if now - t < 1800}
                    pool = [a for a in await self.scrape() if a not in self._proxies and a not in self._dead]
                    random.shuffle(pool)
                    self.log(f"scraped {len(pool)} fresh candidates from {len(self.sources)} sources")
                    # check in waves so we can stop early once enough are healthy
                    wave = max(self.check_concurrency * 2, 200)
                    for i in range(0, min(len(pool), self.max_candidates), wave):
                        need = self.target_size - len(self.healthy())
                        if need <= 0:
                            break
                        await self._check_many(pool[i : i + wave], need)
                        self.log(f"{len(self.healthy())}/{self.target_size} healthy after {min(i + wave, len(pool))} checked")
                self.last_refill = time.time()
                self.save_cache()
                return {"seconds": round(time.time() - started, 1), **self.stats()}
            finally:
                logging.getLogger("asyncio").removeFilter(qfilter)

    async def ensure(self) -> None:
        """Called before requests: refill now if empty, in the background if running low."""
        healthy = len(self.healthy())
        if healthy == 0 and not self._lock.locked():
            await self.refill()
        elif healthy < self.min_healthy and (self._bg is None or self._bg.done()):
            self._bg = asyncio.create_task(self.refill())
        elif healthy == 0:  # a refill is already running: wait for it
            async with self._lock:
                pass

    async def close(self) -> None:
        if self._bg is not None and not self._bg.done():
            self._bg.cancel()
            try:
                await self._bg
            except (asyncio.CancelledError, Exception):
                pass
        self.save_cache()


# ----------------------------------------------------------------------------
# backwards compatible helpers
# ----------------------------------------------------------------------------
async def scrape_proxies(sources: Optional[list[str]] = None, timeout: float = 15.0) -> list[str]:
    """Fetch proxy lists from the given sources and merge unique entries."""
    return await ProxyManager(sources=sources, cache_file=False).scrape()


async def test_proxy(proxy: str, target: str = TEST_URL, timeout: float = 12.0) -> bool:
    """True if ``proxy`` reaches Baidu's bootstrap endpoint and returns valid data."""
    mgr = ProxyManager(test_url=target, check_timeout=timeout, cache_file=False)
    return await mgr.check(proxy) is not None


async def get_working_proxies(
    sources: Optional[list[str]] = None,
    limit: int = 40,
    concurrency: int = 150,
    test_timeout: float = 8.0,
) -> list[str]:
    """Scrape + verify; returns up to ``limit`` working ``ip:port`` (fastest first)."""
    mgr = ProxyManager(
        sources=sources, target_size=limit, check_concurrency=concurrency,
        check_timeout=test_timeout, cache_file=False,
    )
    await mgr.refill()
    live = sorted(mgr.healthy(), key=lambda p: p.latency)
    return [p.addr for p in live][:limit]
