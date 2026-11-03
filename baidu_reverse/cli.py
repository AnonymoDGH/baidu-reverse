"""CLI for baidu-reverse.

    baidu-reverse "Explain what a kernel is" --model smartmode      # one-shot (default)
    baidu-reverse chat --model deepseek-v4-flash                    # interactive, remembers
    baidu-reverse serve --port 8787 --auto-solve --cache            # OpenAI-compatible API
    baidu-reverse solve-captcha                                     # get verified cookies
    baidu-reverse models                                            # what is available
"""

from __future__ import annotations

import argparse
import asyncio
import sys

COMMANDS = {"chat", "serve", "solve-captcha", "models", "sessions", "proxies"}


def _add_client_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cookies", default="", help="Baidu cookie string (optional)")
    p.add_argument("--cookie-file", default="", help="JSON cookie file to load/save (default with --auto-solve)")
    p.add_argument("--proxy", default="", help="Static HTTP proxy (optional)")
    p.add_argument(
        "--rotate-proxies", "--proxies", dest="rotate_proxies", action="store_true",
        help="Scrape free HTTP proxies, verify them against Baidu and rotate requests through them",
    )
    p.add_argument("--auto-solve", action="store_true", help="Solve Baidu's captcha in a real browser when blocked")
    p.add_argument("--cache", action="store_true", help="Cache identical stateless answers in SQLite")
    p.add_argument("--debug", action="store_true", help="Verbose captcha / browser logging")


def _legacy_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="baidu-reverse",
        description="Baidu Wenxin (chat.baidu.com) async client. "
        "Subcommands: chat, serve, solve-captcha, models, sessions.",
    )
    p.add_argument("query", help="The prompt to send")
    p.add_argument(
        "--model", default="deepseek-v4-pro",
        help="Model id (deepseek-v4-pro, deepseek-v4-flash, smartmode, ernie-5.1, + -thinking/-nothinking)",
    )
    p.add_argument("--deep-search", action="store_true", help="Enable deep search")
    p.add_argument("--internet-search", action="store_true", help="Enable internet search")
    p.add_argument("--thinking", action="store_true", help="Show thinking/reasoning events")
    _add_client_args(p)
    return p


def _solver_options(args: argparse.Namespace) -> dict:
    opts: dict = {}
    if getattr(args, "debug", False):
        opts["log"] = lambda m: print(f"[captcha] {m}", file=sys.stderr)
    return opts


async def _make_chat(args: argparse.Namespace):
    from .client import BaiduChat

    return BaiduChat(
        cookies=args.cookies or None,
        proxy=args.proxy or None,
        rotate_proxies=args.rotate_proxies,
        auto_solve=args.auto_solve,
        cookie_file=args.cookie_file or None,
        cache=True if args.cache else None,
        solver_options=_solver_options(args),
    )


async def _ask(args: argparse.Namespace) -> int:
    chat = await _make_chat(args)
    async with chat:
        async for chunk in chat.stream(
            args.query, model=args.model, deep_search=args.deep_search, internet_search=args.internet_search
        ):
            if chunk["type"] == "text":
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "thinking" and args.thinking:
                print(f"\x1b[90m{chunk['content']}\x1b[0m", end="", flush=True)
        print()
    return 0


# ----------------------------------------------------------------------------
async def _chat_repl(args: argparse.Namespace) -> int:
    from .session import ChatSession, SessionStore

    chat = await _make_chat(args)
    store = SessionStore(args.sessions_dir or None)
    async with chat:
        if args.session:
            session = store.load(chat, args.session)
            print(f"resumed {session.id} ({len(session.messages)} messages)")
        else:
            session = ChatSession(chat, model=args.model, system=args.system or None, store=store)
        session.deep_search = args.deep_search
        session.internet_search = args.internet_search
        print(f"session {session.id} - model {session.model} - /help for commands")
        while True:
            try:
                line = (await asyncio.to_thread(input, "you> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                cmd, _, rest = line.partition(" ")
                if cmd in ("/exit", "/quit"):
                    break
                if cmd == "/help":
                    print("/new  /model <id>  /summary  /compact  /save [file]  /sessions  /exit")
                elif cmd == "/new":
                    session = ChatSession(chat, model=session.model, system=session.system, store=store)
                    print(f"new session {session.id}")
                elif cmd == "/model":
                    session.model = rest.strip() or session.model
                    print(f"model -> {session.model}")
                elif cmd == "/summary":
                    print(session.summary or "(no summary yet)")
                elif cmd == "/compact":
                    print("compacted" if await session.compact(force=True) else "nothing to compact")
                elif cmd == "/save":
                    print("saved ->", session.save(rest.strip() or None))
                elif cmd == "/sessions":
                    for s in store.list()[:15]:
                        print(f"{s['id']}  {s['model']:<18} {s['messages']:>3} msgs  {s['title']}")
                else:
                    print("unknown command")
                continue
            print("bot> ", end="", flush=True)
            try:
                async for ev in session.stream(line):
                    if ev["type"] == "text":
                        print(ev["content"], end="", flush=True)
                    elif ev["type"] == "thinking" and args.thinking:
                        print(f"\x1b[90m{ev['content']}\x1b[0m", end="", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"\n[error] {exc}")
            print()
        session.save()
        print(f"saved session {session.id}")
    return 0


async def _serve(args: argparse.Namespace) -> int:
    from .server import serve

    await serve(
        host=args.host,
        port=args.port,
        api_key=args.api_key or None,
        max_concurrency=args.max_concurrency,
        cookies=args.cookies or None,
        proxy=args.proxy or None,
        rotate_proxies=args.rotate_proxies,
        auto_solve=args.auto_solve,
        cookie_file=args.cookie_file or None,
        cache=True if args.cache else None,
        solver_options=_solver_options(args),
    )
    return 0


async def _solve(args: argparse.Namespace) -> int:
    from .captcha.browser import solve_and_get_cookies
    from .paths import default_cookie_file

    log = lambda m: print(f"[captcha] {m}", file=sys.stderr)  # noqa: E731
    seed = args.cookies
    out_path = args.out or str(default_cookie_file())
    if not seed:
        try:
            import json
            from pathlib import Path

            seed = json.loads(Path(out_path).read_text(encoding="utf-8")).get("cookies", "")
        except Exception:
            seed = ""
    cookies = await solve_and_get_cookies(
        seed,
        cdp_url=args.cdp_url or None,
        exe=args.exe or None,
        headless=args.headless,
        save_dir=args.save_captchas or None,
        log=log,
    )
    import json
    import time
    from pathlib import Path

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"cookies": cookies, "saved_at": time.time()}), encoding="utf-8")
    print(f"ok: {cookies.count('=')} cookies saved to {p}")
    if args.print:
        print(cookies)
    return 0


async def _proxies(args: argparse.Namespace) -> int:
    from .proxies import ProxyManager

    mgr = ProxyManager(target_size=args.count, check_timeout=args.timeout, max_latency=args.timeout)
    mgr.log = lambda m: print(f"[proxies] {m}", file=sys.stderr)
    summary = await mgr.refill(force=args.refresh)
    live = sorted(mgr.healthy(), key=lambda p: p.latency)
    if mgr.source_stats:
        print("sources:")
        for url, n in mgr.source_stats.items():
            print(f"  {n:>6}  {url[:90]}")
    print(f"\n{len(live)} verified proxies (checked in {summary.get('seconds', 0)}s)")
    for p in live:
        print(f"  {p.addr:<22} {p.latency:5.2f}s")
    if args.print:
        print(",".join(p.addr for p in live))
    return 0 if live else 1


def _models(_: argparse.Namespace) -> int:
    from .models import DEPRECATED_MODELS, MODEL_ALIASES, describe_models

    print(f"{'id':<20} {'baidu name':<20} think  web  ui   label")
    for m in describe_models():
        print(
            f"{m['id']:<20} {m['baidu_name']:<20} {'on ' if m['thinking_default'] else 'off'}    "
            f"{'on ' if m['web_search_default'] else 'off'}  {'yes' if m['ui_visible'] else 'no '}  {m['label']}"
        )
    print("\nAppend -thinking / -nothinking to any id to force reasoning on/off.")
    print("Removed upstream (mapped with a warning): " + ", ".join(f"{k}->{v}" for k, v in DEPRECATED_MODELS.items()))
    print("Aliases: " + ", ".join(sorted(k for k in MODEL_ALIASES if k not in DEPRECATED_MODELS)))
    return 0


def _sessions(args: argparse.Namespace) -> int:
    from .session import SessionStore

    store = SessionStore(args.sessions_dir or None)
    rows = store.list()
    if not rows:
        print("(no saved sessions)")
    for s in rows:
        print(f"{s['id']}  {str(s['model']):<18} {s['messages']:>3} msgs  {s['title']}")
    return 0


# ----------------------------------------------------------------------------
def _command_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="baidu-reverse")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat", help="interactive chat with memory (server-side session + rolling summary)")
    c.add_argument("--model", default="deepseek-v4-pro")
    c.add_argument("--system", default="", help="System prompt")
    c.add_argument("--session", default="", help="Resume a saved session id")
    c.add_argument("--sessions-dir", default="")
    c.add_argument("--deep-search", action="store_true")
    c.add_argument("--internet-search", action="store_true")
    c.add_argument("--thinking", action="store_true", help="Show reasoning")
    _add_client_args(c)

    s = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.add_argument("--api-key", default="")
    s.add_argument("--max-concurrency", type=int, default=2)
    _add_client_args(s)

    z = sub.add_parser("solve-captcha", help="open a browser, solve Baidu's captcha, save verified cookies")
    z.add_argument("--cookies", default="", help="Seed cookies to verify (default: the saved cookie file)")
    z.add_argument("--out", default="", help="Cookie file to write (default ~/.cache/baidu_reverse/cookies.json)")
    z.add_argument("--cdp-url", default="", help="Attach to a running browser, e.g. http://127.0.0.1:9222")
    z.add_argument("--exe", default="", help="Browser executable (Brave/Chrome/Edge)")
    z.add_argument("--headless", action="store_true")
    z.add_argument("--save-captchas", default="", help="Directory to keep the captcha images")
    z.add_argument("--print", action="store_true", help="Also print the cookie string")

    px = sub.add_parser("proxies", help="scrape + verify free proxies against Baidu and list the working ones")
    px.add_argument("--count", type=int, default=15, help="stop once this many are verified")
    px.add_argument("--timeout", type=float, default=6.0, help="max seconds a proxy may take to reach Baidu")
    px.add_argument("--refresh", action="store_true", help="ignore the cached set and scrape again")
    px.add_argument("--print", action="store_true", help="also print them comma separated")

    sub.add_parser("models", help="list models and aliases")

    ss = sub.add_parser("sessions", help="list saved chat sessions")
    ss.add_argument("--sessions-dir", default="")
    return p


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    argv = sys.argv[1:] if argv is None else argv
    try:
        if argv and argv[0] in COMMANDS:
            args = _command_parser().parse_args(argv)
            if args.cmd == "models":
                sys.exit(_models(args))
            if args.cmd == "sessions":
                sys.exit(_sessions(args))
            run = {"chat": _chat_repl, "serve": _serve, "solve-captcha": _solve, "proxies": _proxies}[args.cmd]
        else:
            args = _legacy_parser().parse_args(argv)
            run = _ask
        asyncio.run(run(args))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
