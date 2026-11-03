"""Filesystem locations used by baidu-reverse."""

from __future__ import annotations

import os
from pathlib import Path


def cache_dir() -> Path:
    """``$BAIDU_REVERSE_HOME`` or ``~/.cache/baidu_reverse`` (created on demand)."""
    base = os.environ.get("BAIDU_REVERSE_HOME")
    root = Path(base) if base else Path.home() / ".cache" / "baidu_reverse"
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_cookie_file() -> Path:
    return cache_dir() / "cookies.json"
