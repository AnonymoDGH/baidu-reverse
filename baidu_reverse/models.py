"""Model catalog for chat.baidu.com / wenxin.baidu.com (Baidu Wenxin 文心).

Catalog verified live (anonymous session). What the backend actually serves:

==================  ===========================================  ============
id                  what answers                                 in web UI
==================  ===========================================  ============
smartmode           Baidu's own assistant (文心助手, "自动")       yes
ernie-5.1           ERNIE 5.1 (文心一言 / ERNIE Bot), web search   yes
deepseek-v4-pro     DeepSeek V4, reasoning on by default          hidden
deepseek-v4-flash   DeepSeek V4 Flash, reasoning off by default   hidden
==================  ===========================================  ============

Removed upstream (the server silently falls back to the generic assistant, so
they are mapped to the closest live model with a :class:`DeprecationWarning`):
``deepseek-r1``, ``deepseek-v3``, ``ernie-4.5``.

Thinking is controlled with ``thinkMode`` inside ``modelFunction``; every id
accepts a ``-thinking`` / ``-nothinking`` suffix to force it.
"""

from __future__ import annotations

import copy
import warnings
from typing import Optional

#: Canonical model specs keyed by internal id.
MODELS = {
    "smartmode": {
        "usedModel": {
            "modelName": "smartMode",
            "modelFunction": {"deepSearch": "0", "thinkMode": "0"},
            "showModelName": "smartMode",
        },
        "enter_type": "chat_url",
        "agt_sess_cnt": 1,
        "anti_ext": {"inputT": None},
        "rank": 1,
        "is_deepseek": True,
        "label": "自动 / 文心助手 (smartMode)",
        "ui_visible": True,
    },
    "ernie-5.1": {
        "usedModel": {
            "modelName": "ERINE-5.1",
            "modelFunction": {"deepSearch": "0", "internetSearch": "1", "thinkMode": "0"},
            "showModelName": "ERINE-5.1",
        },
        "enter_type": "chat_url",
        "agt_sess_cnt": 1,
        "anti_ext": {"inputT": None},
        "rank": 1,
        "is_deepseek": False,
        "label": "文心 5.1 (ERNIE 5.1)",
        "ui_visible": True,
    },
    "deepseek-v4-pro": {
        "usedModel": {
            "modelName": "DeepSeek-V4",
            "modelFunction": {"deepSearch": "0", "thinkMode": "1"},
            "showModelName": "DeepSeek-V4",
        },
        "enter_type": "chat_url",
        "agt_sess_cnt": 1,
        "anti_ext": {"inputT": None},
        "rank": 1,
        "is_deepseek": True,
        "label": "DeepSeek V4 (reasoning)",
        "ui_visible": False,
    },
    "deepseek-v4-flash": {
        "usedModel": {
            "modelName": "DeepSeek-V4-Flash",
            "modelFunction": {"deepSearch": "0", "thinkMode": "0"},
            "showModelName": "DeepSeek-V4-Flash",
        },
        "enter_type": "chat_url",
        "agt_sess_cnt": 1,
        "anti_ext": {"inputT": None},
        "rank": 1,
        "is_deepseek": True,
        "label": "DeepSeek V4 Flash",
        "ui_visible": False,
    },
}

DEFAULT_MODEL = "deepseek-v4-pro"

#: Models the upstream removed -> closest live replacement.
DEPRECATED_MODELS = {
    "deepseek-r1": "deepseek-v4-pro",
    "ds-r1": "deepseek-v4-pro",
    "deepseek-v3": "deepseek-v4-flash",
    "ernie-4.5": "ernie-5.1",
    "ernie-4": "ernie-5.1",
}

#: User-facing aliases -> canonical internal id.
MODEL_ALIASES = {
    "smart": "smartmode",
    "smart-mode": "smartmode",
    "auto": "smartmode",
    "wenxin": "ernie-5.1",
    "ernie": "ernie-5.1",
    "ernie-5": "ernie-5.1",
    "ernine": "ernie-5.1",
    "ernine-5": "ernie-5.1",
    "ernine-5.1": "ernie-5.1",
    "erine-5.1": "ernie-5.1",
    "deepseek": "deepseek-v4-pro",
    "deepseek-v4": "deepseek-v4-pro",
    "ds-v4": "deepseek-v4-pro",
    "ds-pro": "deepseek-v4-pro",
    "v4-pro": "deepseek-v4-pro",
    "dsv4pro": "deepseek-v4-pro",
    "ds-flash": "deepseek-v4-flash",
    "v4-flash": "deepseek-v4-flash",
    "flash": "deepseek-v4-flash",
    **DEPRECATED_MODELS,
}

_SUFFIXES = (("-nothinking", False), ("-thinking", True))


def split_suffix(model: str) -> tuple[str, Optional[bool]]:
    """``"ernie-5.1-thinking"`` -> ``("ernie-5.1", True)``."""
    low = model.strip().lower()
    for suffix, forced in _SUFFIXES:
        if low.endswith(suffix):
            return low[: -len(suffix)], forced
    return low, None


def resolve_model(model: str, warn: bool = True) -> str:
    """Resolve any id / alias / suffixed id to a canonical key in :data:`MODELS`.

    Unknown names fall back to :data:`DEFAULT_MODEL` (with a warning), because the
    server itself silently answers with a generic assistant for unknown models.
    """
    base, _ = split_suffix(model or "")
    if base in DEPRECATED_MODELS and warn:
        warnings.warn(
            f"model {base!r} was removed upstream; using {DEPRECATED_MODELS[base]!r}",
            DeprecationWarning,
            stacklevel=3,
        )
    if base in MODELS:
        return base
    if base in MODEL_ALIASES:
        return MODEL_ALIASES[base]
    if warn:
        warnings.warn(
            f"unknown model {model!r}; falling back to {DEFAULT_MODEL!r}",
            RuntimeWarning,
            stacklevel=3,
        )
    return DEFAULT_MODEL


def apply_think_suffix(spec: dict, model: str) -> dict:
    """Deep-copy ``spec`` and apply a ``-thinking`` / ``-nothinking`` suffix."""
    spec = copy.deepcopy(spec)
    _, forced = split_suffix(model or "")
    if forced is not None:
        spec["usedModel"]["modelFunction"]["thinkMode"] = "1" if forced else "0"
    return spec


def public_models() -> list[str]:
    """Model ids exposed through the public API (canonical ids + thinking variants)."""
    ids: list[str] = []
    for key in MODELS:
        ids.append(key)
        ids.append(f"{key}-thinking")
        ids.append(f"{key}-nothinking")
    return list(dict.fromkeys(ids))


def describe_models() -> list[dict]:
    """Human readable catalog (used by ``baidu-reverse models`` and ``/v1/models``)."""
    out = []
    for key, spec in MODELS.items():
        fn = spec["usedModel"]["modelFunction"]
        out.append(
            {
                "id": key,
                "baidu_name": spec["usedModel"]["modelName"],
                "label": spec.get("label", key),
                "thinking_default": fn.get("thinkMode") == "1",
                "web_search_default": fn.get("internetSearch") == "1",
                "ui_visible": spec.get("ui_visible", True),
            }
        )
    return out
