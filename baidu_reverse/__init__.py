"""baidu-reverse: async reverse-engineered client for Baidu Wenxin.

Example::

    import asyncio
    from baidu_reverse import BaiduChat

    async def main():
        async with BaiduChat(auto_solve=True) as chat:
            print(await chat.generate("Explain what a kernel is in 2 lines"))

    asyncio.run(main())

Highlights: streaming + reasoning, native server-side multi-turn
(:class:`Conversation`, :class:`ChatSession`), tool calling (:class:`Agent`, ``@tool``),
response/session caches, an OpenAI-compatible server and a rotate-captcha solver.
"""

from .agent import Agent, AgentResult, Tool, tool
from .cache import ResponseCache
from .client import (
    BASE_URL,
    CONVERSATION_API,
    BaiduAuthError,
    BaiduBlockedError,
    BaiduCaptchaError,
    BaiduChat,
    BaiduError,
)
from .models import (
    DEFAULT_MODEL,
    DEPRECATED_MODELS,
    MODEL_ALIASES,
    MODELS,
    describe_models,
    public_models,
    resolve_model,
)
from .proxies import ProxyManager, ProxyPool, get_working_proxies, scrape_proxies, test_proxy
from .session import ChatSession, Conversation, PrefixSessionCache, SessionStore
from .tool_calling import (
    ToolStreamFilter,
    build_tool_system_prompt,
    compact_messages,
    extract_tool_calls,
    messages_to_prompt,
    parse_tool_calls,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "BaiduChat", "BaiduError", "BaiduAuthError", "BaiduBlockedError", "BaiduCaptchaError",
    "BASE_URL", "CONVERSATION_API",
    "MODELS", "MODEL_ALIASES", "DEPRECATED_MODELS", "DEFAULT_MODEL",
    "public_models", "describe_models", "resolve_model",
    "ProxyPool", "ProxyManager", "scrape_proxies", "test_proxy", "get_working_proxies",
    "Conversation", "ChatSession", "SessionStore", "PrefixSessionCache", "ResponseCache",
    "Agent", "AgentResult", "Tool", "tool",
    "build_tool_system_prompt", "messages_to_prompt", "compact_messages",
    "parse_tool_calls", "extract_tool_calls", "ToolStreamFilter",
]
