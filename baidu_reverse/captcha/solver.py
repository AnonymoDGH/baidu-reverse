"""Browser-driven solver for Baidu's rotate ("spin") captcha.

Baidu's captcha SDK (``mkd_v2.js``) encrypts the mouse trace / PoW / device
fingerprint before posting it, so re-implementing the wire protocol is brittle.
Instead we let the real page do that and only do what a human does:

1. read the captcha picture,
2. predict how much it is rotated (:mod:`baidu_reverse.captcha.angle`),
3. drag the slider so that ``percent = angle / 360`` with a human-like trace,
4. verify the page accepted it (retry with a fresh captcha otherwise).

Requires ``playwright`` (``pip install baidu-reverse[captcha]``). The browser is
driven through CDP; use :func:`baidu_reverse.captcha.browser.launch_browser`
to start a local Brave/Chrome/Edge with remote debugging.
"""

from __future__ import annotations

import asyncio
import base64
import math
import random
import re
import time
from typing import Any, Callable, Optional

from .angle import AnglePredictor

BG_SEL = ".passMod_spin-background"
BTN_SEL = ".passMod_slide-btn"
CONTROL_SEL = ".passMod_slide-control"
SUCCESS_SEL = ".passMod_verify-success"
FAIL_SEL = ".passMod_verify-fail"
REFRESH_SEL = ".passMod_dialog-footer-refresh"

_ROTATE_RE = re.compile(r"rotate\((-?[\d.]+)deg\)")

_FETCH_IMG_JS = """async (sel) => {
  const img = document.querySelector(sel);
  if (!img || !img.src) return null;
  const r = await fetch(img.src, {credentials: 'include'});
  const buf = new Uint8Array(await r.arrayBuffer());
  let s = '';
  for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  return {src: img.src, b64: btoa(s), style: img.getAttribute('style') || ''};
}"""


class CaptchaError(Exception):
    pass


def _parse_rotation(style: str) -> float:
    m = _ROTATE_RE.search(style or "")
    return float(m.group(1)) if m else 0.0


def percent_for(angle_ccw: float, start_deg: float = 0.0) -> float:
    """Slider fraction (0..1) that makes the picture upright.

    ``angle_ccw`` is how far the *raw* picture is rotated counter-clockwise; the page
    shows it rotated clockwise by ``start_deg + 360 * percent`` so we need
    ``start + 360 * percent == angle (mod 360)``. The SDK quantises to 2 decimals.
    """
    return round(((angle_ccw - start_deg) % 360.0) / 360.0, 2) % 1.0


def _human_curve(t: float) -> float:
    """Slow start, fast middle, slow end (smootherstep)."""
    return t * t * t * (t * (6 * t - 15) + 10)


class SpinCaptchaSolver:
    """Solve the spin captcha shown on a Playwright ``Page``."""

    def __init__(
        self,
        predictor: Optional[AnglePredictor] = None,
        max_attempts: int = 6,
        min_confidence: float = 0.0,
        log: Optional[Callable[[str], None]] = None,
        save_dir: Optional[str] = None,
    ):
        self._predictor = predictor
        self.max_attempts = max_attempts
        self.min_confidence = min_confidence
        self._log = log or (lambda m: None)
        self.save_dir = save_dir
        self.stats = {"attempts": 0, "solved": 0}

    @property
    def predictor(self) -> AnglePredictor:
        if self._predictor is None:
            self._predictor = AnglePredictor()
        return self._predictor

    # ------------------------------------------------------------------
    async def is_present(self, page) -> bool:
        try:
            return await page.locator(BG_SEL).count() > 0
        except Exception:
            return False

    async def wait_for_captcha(self, page, timeout: float = 15.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if await self.is_present(page):
                # image must have loaded
                ok = await page.evaluate(
                    "(s)=>{const i=document.querySelector(s);return !!(i&&i.complete&&i.naturalWidth>0)}",
                    BG_SEL,
                )
                if ok:
                    return True
            await asyncio.sleep(0.25)
        return False

    async def _grab(self, page) -> tuple[bytes, float, str]:
        res = await page.evaluate(_FETCH_IMG_JS, BG_SEL)
        if not res or not res.get("b64"):
            raise CaptchaError("could not read captcha image")
        data = base64.b64decode(res["b64"])
        return data, _parse_rotation(res.get("style", "")), res["src"]

    async def _geometry(self, page) -> dict:
        g = await page.evaluate(
            """([b, c]) => {
              const btn = document.querySelector(b), ctl = document.querySelector(c);
              if (!btn || !ctl) return null;
              const rb = btn.getBoundingClientRect(), rc = ctl.getBoundingClientRect();
              return {bx: rb.left + rb.width / 2, by: rb.top + rb.height / 2, bw: rb.width, cw: rc.width, cl: rc.left};
            }""",
            [BTN_SEL, CONTROL_SEL],
        )
        if not g:
            raise CaptchaError("slider not found")
        return g

    async def _current_rotation(self, page) -> float:
        style = await page.evaluate(
            "(s)=>{const i=document.querySelector(s);return i?(i.getAttribute('style')||''):''}",
            BG_SEL,
        )
        return _parse_rotation(style)

    # ------------------------------------------------------------------
    async def _drag(self, page, percent: float, start_deg: float) -> None:
        """Drag the slider until the picture's CSS rotation == start + 360*percent."""
        g = await self._geometry(page)
        span = g["cw"] - 52  # SDK: percent = distance / (controlWidth - 52)
        percent = round(percent, 2)
        target_dx = percent * span

        x0 = g["bx"] + random.uniform(-6, 6)
        y0 = g["by"] + random.uniform(-5, 5)
        mouse = page.mouse
        # approach the knob
        await mouse.move(x0 - random.uniform(60, 140), y0 + random.uniform(-40, 40), steps=random.randint(6, 12))
        await asyncio.sleep(random.uniform(0.15, 0.4))
        await mouse.move(x0, y0, steps=random.randint(5, 9))
        await asyncio.sleep(random.uniform(0.08, 0.25))
        await mouse.down()
        await asyncio.sleep(random.uniform(0.06, 0.18))

        # main stroke, slightly overshoot like a human, then settle
        overshoot = random.uniform(2, 7) if target_dx > 40 else 0.0
        n = random.randint(26, 40)
        dur = random.uniform(0.7, 1.4)
        y_wob = random.uniform(1.5, 4.0)
        phase = random.uniform(0, math.pi)
        for i in range(1, n + 1):
            t = i / n
            dx = (target_dx + overshoot) * _human_curve(t)
            y = y0 + math.sin(t * math.pi * 2 + phase) * y_wob + random.uniform(-0.6, 0.6)
            await mouse.move(x0 + dx, y)
            await asyncio.sleep(dur / n * random.uniform(0.6, 1.4))
        if overshoot:
            for i in range(1, 6):
                await mouse.move(x0 + target_dx + overshoot * (1 - i / 5), y0 + random.uniform(-1, 1))
                await asyncio.sleep(random.uniform(0.02, 0.06))

        # closed-loop correction: read the real rotation the SDK computed
        dx = target_dx
        for _ in range(8):
            cur = await self._current_rotation(page)
            got = round((cur - start_deg) / 360.0, 2)
            if abs(got - percent) < 0.005:
                break
            dx += (percent - got) * span * 0.9 + (0.6 if percent > got else -0.6)
            dx = max(0.0, min(span, dx))
            await mouse.move(x0 + dx, y0 + random.uniform(-1, 1), steps=2)
            await asyncio.sleep(random.uniform(0.05, 0.12))

        await asyncio.sleep(random.uniform(0.25, 0.6))
        await mouse.up()

    async def _outcome(self, page, old_src: str, timeout: float = 8.0) -> str:
        """Return 'success' | 'fail' | 'gone' | 'timeout'."""
        end = time.time() + timeout
        probe = """([s, f, b]) => {
          const vis = (q) => { const e = document.querySelector(q); return !!e && getComputedStyle(e).display !== 'none' && e.getBoundingClientRect().height > 0; };
          const img = document.querySelector(b);
          return {success: vis(s), fail: vis(f), present: !!img, src: img ? img.src : ''};
        }"""
        while time.time() < end:
            try:
                st = await page.evaluate(probe, [SUCCESS_SEL, FAIL_SEL, BG_SEL])
            except Exception:
                return "gone"  # navigation destroyed the execution context -> page moved on
            if st["success"]:
                return "success"
            if st["fail"]:
                return "fail"
            if not st["present"]:
                return "gone"
            if st["src"] and st["src"] != old_src:
                return "fail"  # a fresh captcha was loaded
            await asyncio.sleep(0.2)
        return "timeout"

    async def solve(self, page) -> bool:
        """Solve the captcha on ``page``. Returns True once Baidu accepted it."""
        for attempt in range(1, self.max_attempts + 1):
            if not await self.wait_for_captcha(page, timeout=20 if attempt == 1 else 10):
                # captcha disappeared => maybe already accepted
                if not await self.is_present(page):
                    return attempt > 1 or self.stats["attempts"] > 0
                raise CaptchaError("captcha image never finished loading")

            data, start_deg, src = await self._grab(page)
            angle, conf = self.predictor.predict_bytes(data)
            percent = percent_for(angle, start_deg)
            self.stats["attempts"] += 1
            self._log(
                f"attempt {attempt}: angle={angle:.1f}deg conf={conf:.2f} start={start_deg:.1f} percent={percent:.2f}"
            )
            record = self._save_attempt(data, angle, conf, start_deg, percent)

            if conf < self.min_confidence:
                self._log("low confidence -> refresh captcha")
                await self.refresh(page)
                continue

            await self._drag(page, percent, start_deg)
            result = await self._outcome(page, src)
            self._log(f"  -> {result}")
            self._finish_attempt(record, result)
            if result in ("success", "gone"):
                self.stats["solved"] += 1
                return True
            if result == "timeout":
                await self.refresh(page)
                continue
            # fail: SDK usually reloads a fresh image by itself; make sure
            await asyncio.sleep(random.uniform(0.8, 1.6))
            cur = await page.evaluate(
                "(s)=>{const i=document.querySelector(s);return i?i.src:''}", BG_SEL
            )
            if cur == src:
                await self.refresh(page)
        return False

    # -- dataset logging -----------------------------------------------------
    def _save_attempt(self, data: bytes, angle: float, conf: float, start: float, percent: float):
        """Keep the picture + what we predicted; the outcome is added afterwards.

        Accepted attempts are *labelled* samples (true angle ~= ``angle`` +- 2 deg), a free
        dataset for fine-tuning the predictor on Baidu's own pictures.
        """
        if not self.save_dir:
            return None
        import os

        os.makedirs(self.save_dir, exist_ok=True)
        stem = os.path.join(self.save_dir, f"{int(time.time() * 1000)}")
        with open(stem + ".jpg", "wb") as fh:
            fh.write(data)
        record = {"file": os.path.basename(stem) + ".jpg", "angle_ccw": round(angle, 2), "confidence": round(conf, 4),
                  "start": start, "percent": percent, "result": None}
        self._write_record(stem, record)
        return stem, record

    @staticmethod
    def _write_record(stem: str, record: dict) -> None:
        import json

        with open(stem + ".json", "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False)

    def _finish_attempt(self, saved, result: str) -> None:
        if not saved:
            return
        stem, record = saved
        record["result"] = "success" if result in ("success", "gone") else "fail"
        self._write_record(stem, record)

    async def refresh(self, page) -> None:
        try:
            el = page.locator(REFRESH_SEL).first
            if await el.count():
                box = await el.bounding_box()
                if box:
                    await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=8)
                    await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        except Exception:
            pass
        await asyncio.sleep(random.uniform(0.8, 1.4))
