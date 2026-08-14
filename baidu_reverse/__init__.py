"""baidu-reverse: async reverse-engineered client for Baidu Wenxin.

Example::

    import asyncio
    from baidu_reverse import BaiduChat

    async def main():
        async with BaiduChat() as chat:
            reply = await chat.generate("Explain what a kernel is in 2 lines")
            print(reply)

    asyncio.run(main())
"""

from .client import (
    BaiduChat,
    BaiduError,
    BaiduAuthError,
    BASE_URL,
    CONVERSATION_API,
)
from .models import MODELS, MODEL_ALIASES, DEFAULT_MODEL, public_models
from .proxies import (
    ProxyPool,
    scrape_proxies,
    test_proxy,
    get_working_proxies,
)
from .tool_calling import (
    build_tool_system_prompt,
    messages_to_prompt,
    compact_messages,
    parse_tool_calls,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BaiduChat", "BaiduError", "BaiduAuthError",
    "BASE_URL", "CONVERSATION_API",
    "MODELS", "MODEL_ALIASES", "DEFAULT_MODEL", "public_models",
    "ProxyPool", "scrape_proxies", "test_proxy", "get_working_proxies",
    "build_tool_system_prompt", "messages_to_prompt", "compact_messages", "parse_tool_calls",
]
