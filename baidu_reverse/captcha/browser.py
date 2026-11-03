"""Drive a real browser to get Baidu cookies that pass the anti-bot check.

``solve_and_get_cookies`` opens Brave / Chrome / Edge (remote debugging), loads
the Wenxin web app with your current cookies, sends a short message so that Baidu
shows its rotate-captcha if it wants one, solves it with
:class:`~baidu_reverse.captcha.solver.SpinCaptchaSolver`, and returns the
resulting ``Cookie`` header string for :class:`~baidu_reverse.BaiduChat`.

Requires ``pip install baidu-reverse[captcha]``.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from ..paths import cache_dir
from .solver import CaptchaError, SpinCaptchaSolver

WEB_URL = "https://wenxin.baidu.com/?enter_type=chat_site"


def find_browser(preferred: Optional[str] = None) -> Optional[str]:
    """Locate a Chromium-based browser (Brave first, then Chrome, Edge, Chromium)."""
    if preferred and Path(preferred).exists():
        return preferred
    env = os.environ.get("BAIDU_REVERSE_BROWSER")
    if env and Path(env).exists():
        return env
    cands: list[str] = []
    if sys.platform.startswith("win"):
        roots = [os.environ.get(k, "") for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
        rels = [
            r"BraveSoftware\Brave-Browser\Application\brave.exe",
            r"Google\Chrome\Application\chrome.exe",
            r"Microsoft\Edge\Application\msedge.exe",
            r"Chromium\Application\chrome.exe",
        ]
        cands = [str(Path(r) / rel) for rel in rels for r in roots if r]
    elif sys.platform == "darwin":
        cands = [
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    else:
        for name in (
            "brave-browser", "brave", "google-chrome", "google-chrome-stable",
            "chromium", "chromium-browser", "microsoft-edge",
        ):
            p = shutil.which(name)
            if p:
                cands.append(p)
    for c in cands:
        if Path(c).exists():
            return c
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _probe(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2).read()
        return True
    except Exception:
        return False


async def launch_browser(
    exe: Optional[str] = None,
    port: Optional[int] = None,
    profile_dir: Optional[str] = None,
    headless: bool = False,
) -> tuple[subprocess.Popen, int]:
    """Start the browser with ``--remote-debugging-port`` and wait until it answers."""
    exe = find_browser(exe)
    if not exe:
        raise CaptchaError(
            "no Chromium-based browser found (Brave/Chrome/Edge). "
            "Set BAIDU_REVERSE_BROWSER to its executable path."
        )
    port = port or _free_port()
    profile = Path(profile_dir) if profile_dir else cache_dir() / "browser-profile"
    profile.mkdir(parents=True, exist_ok=True)
    args = [
        exe,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-features=Translate",
        "--window-size=1100,760",
    ]
    if headless:
        args.append("--headless=new")
    args.append("about:blank")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(80):
        if proc.poll() is not None:
            raise CaptchaError(f"browser exited immediately (code {proc.returncode}); is the profile already in use?")
        if await asyncio.to_thread(_probe, port):
            return proc, port
        await asyncio.sleep(0.25)
    proc.terminate()
    raise CaptchaError("browser did not open its debugging port in time")


def cookies_to_header(cookies: list[dict]) -> str:
    """Playwright cookie dicts -> ``Cookie`` header for chat.baidu.com."""
    jar: dict[str, str] = {}
    for c in cookies:
        dom = (c.get("domain") or "").lstrip(".")
        if not c.get("name") or not (dom == "baidu.com" or dom.endswith(".baidu.com")):
            continue
        jar[c["name"]] = c["value"]
    return "; ".join(f"{k}={v}" for k, v in jar.items())


def _parse_seed(seed: str) -> list[dict]:
    out = []
    for part in (seed or "").split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip():
                out.append({"name": k.strip(), "value": v.strip(), "domain": ".baidu.com", "path": "/"})
    return out


_DISMISS_JS = """() => {
  let n = 0;
  for (const el of document.querySelectorAll('.cos-dialog, [class*="dialog-mask"], [class*="guide-dialog"], [class*="modal-mask"]')) {
    el.remove(); n++;
  }
  return n;
}"""


async def _dismiss_overlays(page, log: Callable[[str], None]) -> None:
    """First-visit pop-ups (task-mode guide, login nudges) swallow clicks: get rid of them."""
    for _ in range(2):
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.2)
    try:
        close = page.locator("[class*='guide'] [class*='close'], .cos-dialog [class*='close']").first
        if await close.count() and await close.is_visible():
            await close.click(timeout=2000)
            await asyncio.sleep(0.3)
    except Exception:
        pass
    try:
        removed = await page.evaluate(_DISMISS_JS)
        if removed:
            log(f"removed {removed} blocking overlay element(s)")
    except Exception:
        pass


async def _trigger(page, query: str, log: Callable[[str], None] = lambda m: None) -> None:
    tb = page.get_by_role("textbox").first
    await tb.wait_for(timeout=25000)
    await _dismiss_overlays(page, log)
    await tb.focus()  # no pointer needed, so it can't be intercepted by a modal
    await page.keyboard.type(query, delay=random.randint(40, 90))
    await asyncio.sleep(random.uniform(0.3, 0.7))
    await page.keyboard.press("Enter")


def _on_captcha_url(url: str) -> bool:
    return "wappass.baidu.com" in url or "/captcha/" in url


async def solve_and_get_cookies(
    seed_cookies: str = "",
    *,
    cdp_url: Optional[str] = None,
    exe: Optional[str] = None,
    headless: bool = False,
    profile_dir: Optional[str] = None,
    trigger_query: str = "hello",
    timeout: float = 150.0,
    quiet_seconds: float = 8.0,
    max_attempts: int = 8,
    save_dir: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    keep_open: bool = False,
    predictor=None,
) -> str:
    """Return a ``Cookie`` string verified by Baidu (captcha solved if it was shown).

    Args:
        seed_cookies: cookies of the session you want verified (injected first).
        cdp_url: attach to an already running browser (``http://127.0.0.1:9222``)
            instead of launching one.
        quiet_seconds: how long the answer page must stay captcha-free before we
            conclude that no captcha is required.
    """
    from playwright.async_api import async_playwright

    log = log or (lambda m: None)
    proc: Optional[subprocess.Popen] = None
    async with async_playwright() as pw:
        if cdp_url:
            browser = await pw.chromium.connect_over_cdp(cdp_url)
        else:
            proc, port = await launch_browser(exe, None, profile_dir, headless)
            log(f"browser launched on port {port}")
            browser = await pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        page = None
        try:
            ctx = browser.contexts[0] if browser.contexts else await browser.new_context()
            seed = _parse_seed(seed_cookies)
            if seed:
                try:
                    await ctx.add_cookies(seed)
                except Exception as exc:  # odd cookie names/values
                    log(f"could not inject some seed cookies: {exc}")
            page = await ctx.new_page()
            await page.goto(WEB_URL, wait_until="domcontentloaded", timeout=45000)
            await asyncio.sleep(random.uniform(1.5, 2.5))
            await _trigger(page, trigger_query, log)

            solver = SpinCaptchaSolver(
                predictor=predictor, max_attempts=max_attempts, log=log, save_dir=save_dir
            )
            deadline = time.time() + timeout
            clean_since: Optional[float] = None
            solved = False
            while time.time() < deadline:
                if _on_captcha_url(page.url) or await solver.is_present(page):
                    clean_since = None
                    log("captcha shown -> solving")
                    if not await solver.solve(page):
                        raise CaptchaError("could not solve the captcha (all attempts failed)")
                    solved = True
                    end = time.time() + 25
                    while time.time() < end and _on_captcha_url(page.url):
                        await asyncio.sleep(0.5)
                    await asyncio.sleep(2.0)
                    break
                if "/search/" in page.url:
                    clean_since = clean_since or time.time()
                    if time.time() - clean_since >= quiet_seconds:
                        log("no captcha was required")
                        break
                await asyncio.sleep(0.5)
            else:
                raise CaptchaError("timed out waiting for the captcha / answer page")

            header = cookies_to_header(await ctx.cookies())
            if not header:
                raise CaptchaError("browser produced no baidu.com cookies")
            log(f"{'captcha solved; ' if solved else ''}harvested {header.count('=')} cookies")
            return header
        finally:
            try:
                if page is not None:
                    await page.close()
            except Exception:
                pass
            if proc is not None and not keep_open:
                try:
                    await browser.close()
                except Exception:
                    pass
                proc.terminate()
                try:
                    await asyncio.to_thread(proc.wait, 10)
                except Exception:
                    proc.kill()
