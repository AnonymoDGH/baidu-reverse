"""Rotate-captcha ("拖动左侧滑块使图片为正") solver for Baidu.

Optional extra: ``pip install baidu-reverse[captcha]``.

* :class:`~baidu_reverse.captcha.angle.AnglePredictor` - predicts how far the picture is rotated.
* :class:`~baidu_reverse.captcha.solver.SpinCaptchaSolver` - drags the slider on a Playwright page.
* :func:`~baidu_reverse.captcha.browser.solve_and_get_cookies` - end to end: browser -> captcha -> cookies.

Nothing is imported eagerly, so the core package works without the heavy extras.
"""

from __future__ import annotations

__all__ = ["AnglePredictor", "SpinCaptchaSolver", "CaptchaError", "solve_and_get_cookies", "find_browser"]


def __getattr__(name: str):
    if name == "AnglePredictor":
        from .angle import AnglePredictor

        return AnglePredictor
    if name in ("SpinCaptchaSolver", "CaptchaError"):
        from . import solver

        return getattr(solver, name)
    if name in ("solve_and_get_cookies", "find_browser"):
        from . import browser

        return getattr(browser, name)
    raise AttributeError(name)
