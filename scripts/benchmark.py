"""Benchmark de velocidad de los modelos de chat.baidu.com.

Mide por modelo:
  - first_token_ms : latencia hasta el primer token (texto o razonamiento)
  - think_ms       : duracion total de la fase de razonamiento
  - text_ms        : duracion de la fase de texto
  - total_s        : tiempo total
  - text_chars     : chars de respuesta
  - think_chars    : chars de razonamiento
  - text_cps       : chars/segundo del streaming de texto
  - think_cps      : chars/segundo del razonamiento

Requiere BAIDU_COOKIES. Uso:
    $env:BAIDU_COOKIES="..." ; python scripts/benchmark.py [--model X] [--repeats N]
"""

import argparse
import asyncio
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from baidu_reverse import BaiduChat

COOKIES = os.environ.get("BAIDU_COOKIES", "")

QUERIES = [
    "Explica en 2 parrafos que es la teoria de la relatividad",
    "Escribe un poema corto sobre el otono en espanol",
]

ALL_MODELS = [
    "deepseek-v4-pro",
    "deepseek-v4-pro-nothinking",
    "deepseek-v4-flash",
    "deepseek-v4-flash-thinking",
    "smartmode",
    "ernie-5.1",
    "ernie-5.1-thinking",
]


async def bench_once(chat: BaiduChat, model: str, query: str) -> dict:
    started = False
    text_start = None
    think_start = None
    think_end = None
    first_token_ms = None
    t0 = time.time()
    text_chars = 0
    think_chars = 0
    text_cps_samples = []
    think_cps_samples = []
    last = {"t": None, "c": 0}

    async for chunk in chat.stream(query, model=model):
        ct = chunk["type"]
        if not started:
            started = True
        if ct == "thinking":
            if think_start is None:
                think_start = time.time()
            think_chars += len(chunk["content"])
            if first_token_ms is None:
                first_token_ms = int((time.time() - t0) * 1000)
            # cps muestreo
            now = time.time()
            if last["t"] and now - last["t"] >= 0.5:
                think_cps_samples.append((last["c"] - last["pc"]) / (now - last["t"]))
            last = {"t": now, "c": think_chars}
            last["pc"] = 0
        elif ct == "text":
            if text_start is None:
                text_start = time.time()
                last["pc"] = text_chars
            if first_token_ms is None:
                first_token_ms = int((time.time() - t0) * 1000)
            text_chars += len(chunk["content"])
            now = time.time()
            if last["t"] and now - last["t"] >= 0.5:
                text_cps_samples.append((last["c"] - last["pc"]) / (now - last["t"]))
            last = {"t": now, "c": text_chars}
        elif ct == "done":
            break

    total = time.time() - t0
    if think_start is not None:
        think_end = time.time() if text_start is None else text_start
    text_secs = (time.time() - text_start) if text_start is not None else 0
    think_secs = (think_end - think_start) if think_start is not None else 0

    return {
        "model": model,
        "first_token_ms": first_token_ms,
        "total_s": round(total, 2),
        "text_chars": text_chars,
        "think_chars": think_chars,
        "text_cps": round(text_chars / text_secs, 1) if text_secs > 0 else None,
        "think_cps": round(think_chars / think_secs, 1) if think_secs > 0 else None,
    }


def fmt(r: dict) -> str:
    return (
        f"  first={r['first_token_ms'] or '-':<6}ms total={r['total_s']:<6}s "
        f"text={r['text_chars']:<5}ch @ {r['text_cps'] or '-':<7}cps "
        f"think={r['think_chars']:<5}ch @ {r['think_cps'] or '-':<7}cps"
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="benchmark only this model")
    parser.add_argument("--repeats", type=int, default=2, help="repeats per model (default 2)")
    args = parser.parse_args()

    if not COOKIES:
        print("Set BAIDU_COOKIES env var", file=sys.stderr)
        return 1

    models = [args.model] if args.model else ALL_MODELS

    async with BaiduChat(cookies=COOKIES) as chat:
        for model in models:
            rows = []
            for q in QUERIES:
                for _ in range(args.repeats):
                    try:
                        r = await bench_once(chat, model, q)
                        rows.append(r)
                    except Exception as e:
                        rows.append({"model": model, "error": f"{type(e).__name__}: {str(e)[:80]}"})
                    await asyncio.sleep(1.5)

            ok = [r for r in rows if "error" not in r]
            print(f"\n### {model}  ({len(ok)}/{len(rows)} runs)")
            for r in rows:
                if "error" in r:
                    print(f"  ERROR {r['error']}")
                else:
                    print(fmt(r))
            if ok:
                n = len(ok)
                avg = lambda k: round(sum(r[k] for r in ok if r[k] is not None) / sum(1 for r in ok if r[k] is not None), 1) if any(r[k] is not None for r in ok) else "-"
                print(f"  AVG: first={avg('first_token_ms')}ms total={avg('total_s')}s text_cps={avg('text_cps')} think_cps={avg('think_cps')}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
