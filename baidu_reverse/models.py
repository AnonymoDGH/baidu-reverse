"""Model and chat-mode catalogs for chat.baidu.com (Wenxin).

Model specs mirror the ones used by the official web client. Each entry
defines the ``usedModel`` payload, the ``enter_type``, the ``anti_ext``
fingerprint values and the default rank.

Thinking behaviour is controlled by ``thinkMode`` inside ``modelFunction``:
  - ``smartmode``  : no thinking (deep search off)
  - ``smartmode-thinking`` : thinking on
  - ``ERINE-5.1``  : Wenxin 5.1, thinking on by default
  - ``DeepSeek-R1``: forced thinking (server-side, cannot be disabled)
  - ``DeepSeek-V4-Pro`` / ``DeepSeek-V4-Flash``: thinking on/off variants
"""

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
    },
    "deepseek-r1": {
        "usedModel": {
            "modelName": "DeepSeek-R1",
            "modelFunction": {"deepSearch": "0", "thinkMode": "1"},
            "showModelName": "DeepSeek-R1",
        },
        "enter_type": "chat_url",
        "agt_sess_cnt": 1,
        "anti_ext": {"inputT": None},
        "rank": 1,
        "is_deepseek": True,
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
    },
}

#: User-facing model ids and aliases -> canonical internal id.
MODEL_ALIASES = {
    "smart": "smartmode",
    "smart-mode": "smartmode",
    "smartmode": "smartmode",
    "smartmode-thinking": "smartmode",
    "wenxin": "ernie-5.1",
    "ernie": "ernie-5.1",
    "ernie-5": "ernie-5.1",
    "ernie-5.1": "ernie-5.1",
    "ernine": "ernie-5.1",
    "ernine-5": "ernie-5.1",
    "ernine-5.1": "ernie-5.1",
    "deepseek": "deepseek-r1",
    "ds-r1": "deepseek-r1",
    "deepseek-r1": "deepseek-r1",
    "deepseek-v4": "deepseek-v4-pro",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "deepseek-v4-pro-thinking": "deepseek-v4-pro",
    "deepseek-v4-pro-nothinking": "deepseek-v4-pro",
    "ds-v4": "deepseek-v4-pro",
    "ds-pro": "deepseek-v4-pro",
    "v4-pro": "deepseek-v4-pro",
    "dsv4pro": "deepseek-v4-pro",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-flash-thinking": "deepseek-v4-flash",
    "deepseek-v4-flash-nothinking": "deepseek-v4-flash",
    "ds-flash": "deepseek-v4-flash",
    "v4-flash": "deepseek-v4-flash",
    "flash": "deepseek-v4-flash",
}

#: Model ids exposed through the public API (canonical + aliases).
def public_models() -> list[str]:
    ids = []
    for key in MODELS:
        ids.append(key)
        if key in ("smartmode", "deepseek-v4-pro", "deepseek-v4-flash"):
            ids.append(f"{key}-thinking")
    ids.append("smartmode-thinking")
    ids.append("deepseek-v4-pro-nothinking")
    ids.append("deepseek-v4-flash-nothinking")
    # de-dupe preserving order
    return list(dict.fromkeys(ids))


DEFAULT_MODEL = "deepseek-v4-pro"
