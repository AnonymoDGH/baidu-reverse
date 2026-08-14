"""CLI for baidu-reverse."""

import argparse
import asyncio
import sys


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baidu-reverse", description="Baidu Wenxin (chat.baidu.com) async client")
    parser.add_argument("query", help="The prompt to send")
    parser.add_argument(
        "--model", default="deepseek-v4-pro",
        help="Model id (deepseek-v4-pro, deepseek-v4-flash, smartmode, ernie-5.1, deepseek-r1, ...)",
    )
    parser.add_argument("--deep-search", action="store_true", help="Enable deep search")
    parser.add_argument("--internet-search", action="store_true", help="Enable internet search")
    parser.add_argument("--thinking", action="store_true", help="Show thinking/reasoning events")
    parser.add_argument("--cookies", default="", help="Baidu cookie string (optional)")
    parser.add_argument("--proxy", default="", help="Static HTTP proxy (optional)")
    parser.add_argument(
        "--proxies", action="store_true",
        help="Scrape free HTTP proxies, test them against chat.baidu.com, and rotate",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    from .client import BaiduChat

    proxies = None
    if args.proxies:
        from .proxies import get_working_proxies

        proxies = await get_working_proxies()
        if not proxies:
            print("No working proxies found — falling back to direct connection", file=sys.stderr)

    async with BaiduChat(
        cookies=args.cookies or None,
        proxy=args.proxy or None,
        proxies=proxies,
    ) as chat:
        async for chunk in chat.stream(
            args.query,
            model=args.model,
            deep_search=args.deep_search,
            internet_search=args.internet_search,
        ):
            if chunk["type"] == "text":
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "thinking" and args.thinking:
                print(f"\n\x1b[90m{chunk['content']}\x1b[0m", end="", flush=True)
            elif chunk["type"] == "done":
                break
        print()
    return 0


def main():
    parser = _make_parser()
    args = parser.parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
