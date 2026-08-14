"""Proxy scraping + pool for baidu-reverse.

Scrapes free HTTP proxies from public lists, tests them against
chat.baidu.com, and rotates working proxies across requests to avoid
IP-level rate limits (the kunlun anti-bot).
"""

import asyncio
import random
import re
import time
from typing import Optional

import aiohttp

DEFAULT_SOURCES = [
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/http/data.txt",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
]

_PATTERNS = [
    re.compile(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{2,5})"),
]


def _extract_proxies(text: str) -> list[str]:
    found = set()
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            found.add(f"{m.group(1)}:{m.group(2)}")
    return sorted(found)


async def scrape_proxies(
    sources: Optional[list[str]] = None,
    timeout: float = 15.0,
) -> list[str]:
    """Fetch proxy lists from the given sources and merge unique entries."""
    sources = sources or DEFAULT_SOURCES
    proxies: set[str] = set()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:
        for url in sources:
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        text = await resp.text(errors="replace")
                        proxies.update(_extract_proxies(text))
            except Exception:
                continue
    return sorted(proxies)


async def test_proxy(
    proxy: str,
    target: str = "https://chat.baidu.com/aichat/api/aitabserver?ctl=baseData",
    timeout: float = 12.0,
) -> bool:
    """Check whether a proxy can reach chat.baidu.com AND pass the anti-bot.

    Merely reaching the host is not enough — Baidu's ``kunlun`` anti-bot
    returns an empty/invalid ``baseData`` for blocked IPs. This verifies the
    response parses as JSON with a usable ``token`` + ``lid``.
    """
    import json

    try:
        timeout_ctx = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_ctx) as session:
            async with session.get(
                target,
                proxy=f"http://{proxy}",
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/138.0.0.0 Safari/537.36"
                    ),
                    "accept": "application/json, text/plain, */*",
                },
            ) as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
        fb = (data.get("data") or {}).get("frameBaseData") or {}
        return bool(fb.get("token") and fb.get("lid"))
    except Exception:
        return False


async def get_working_proxies(
    sources: Optional[list[str]] = None,
    limit: int = 40,
    concurrency: int = 20,
    test_timeout: float = 10.0,
) -> list[str]:
    """Scrape proxies, test them concurrently, return working ones."""
    proxies = await scrape_proxies(sources)
    print(f"[proxies] scraped {len(proxies)} candidates")

    sem = asyncio.Semaphore(concurrency)
    working: list[str] = []
    lock = asyncio.Lock()

    async def worker(p: str):
        async with sem:
            if await test_proxy(p, timeout=test_timeout):
                async with lock:
                    working.append(p)

    await asyncio.gather(*(worker(p) for p in proxies[: limit * 10]))
    # keep the working ones, randomize order
    random.shuffle(working)
    print(f"[proxies] {len(working)} reach chat.baidu.com")
    return working[:limit]


class ProxyPool:
    """Round-robin pool of working proxies for BaiduChat."""

    def __init__(self, proxies: Optional[list[str]] = None):
        self.proxies = proxies or []
        self._idx = 0
        self._last_failures: dict[str, float] = {}

    def add(self, proxy: str):
        if proxy not in self.proxies:
            self.proxies.append(proxy)

    def next(self) -> Optional[str]:
        if not self.proxies:
            return None
        now = time.time()
        for _ in range(len(self.proxies)):
            p = self.proxies[self._idx % len(self.proxies)]
            self._idx += 1
            # skip recently-failed proxies (30s cooldown)
            if self._last_failures.get(p, 0) > now - 30:
                continue
            return p
        return self.proxies[(self._idx - 1) % len(self.proxies)]

    def report_failure(self, proxy: str):
        if proxy:
            self._last_failures[proxy] = time.time()

    def report_success(self, proxy: str):
        if proxy:
            self._last_failures.pop(proxy, None)
