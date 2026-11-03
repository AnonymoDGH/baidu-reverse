"""Prueba de humo contra chat.baidu.com real.

Requiere ``BAIDU_COOKIES`` en el entorno (o edita COOKIES abajo) con las
cookies de un navegador logueado en chat.baidu.com.

Verifica, por cada modelo: streaming (text chunks), captura de razonamiento
(thinking) y latencia del primer token.

Uso:
    $env:BAIDU_COOKIES="BAIDUID=...; ZFY=..." ; python scripts/test_models.py
"""

import asyncio
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from baidu_reverse import BaiduChat
from baidu_reverse.models import MODELS

COOKIES = os.environ.get("BAIDU_COOKIES", "")

MODELS_TO_TEST = [
    "smartmode",
    "smartmode-thinking",
    "ernie-5.1",
    "ernie-5.1-thinking",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-v4-flash-thinking",
]

QUERY = "Calcula 23 * 47 y responde solo con el resultado en una linea"


async def test_model(chat: BaiduChat, model: str) -> dict:
    thinking_chunks = 0
    text_chunks = 0
    text = ""
    thinking = ""
    done = False
    first_token_ms = None
    start = time.time()
    try:
        async for chunk in chat.stream(QUERY, model=model):
            ct = chunk["type"]
            if ct == "thinking":
                thinking_chunks += 1
                thinking += chunk["content"]
                if first_token_ms is None:
                    first_token_ms = int((time.time() - start) * 1000)
            elif ct == "text":
                text_chunks += 1
                text += chunk["content"]
                if first_token_ms is None:
                    first_token_ms = int((time.time() - start) * 1000)
            elif ct == "done":
                done = True
    except Exception as e:
        return {"model": model, "error": f"{type(e).__name__}: {str(e)[:120]}"}
    return {
        "model": model,
        "ok": done and bool(text.strip()),
        "text_chunks": text_chunks,
        "thinking_chunks": thinking_chunks,
        "text": text.strip(),
        "thinking": thinking.strip(),
        "first_token_ms": first_token_ms,
        "elapsed_s": round(time.time() - start, 1),
    }


async def main():
    if not COOKIES:
        print("Set BAIDU_COOKIES env var with your chat.baidu.com cookies", file=sys.stderr)
        return 1

    print(f"Modelos internos ({len(MODELS)}): {list(MODELS.keys())}")
    print("=" * 70)

    async with BaiduChat(cookies=COOKIES) as chat:
        results = []
        for model in MODELS_TO_TEST:
            print(f"\n### MODELO: {model}")
            r = await test_model(chat, model)
            results.append(r)
            if "error" in r:
                print(f"  ERROR: {r['error']}")
                continue
            print(
                f"  OK={r['ok']} | text_chunks={r['text_chunks']} | "
                f"think_chunks={r['thinking_chunks']} | first_token={r['first_token_ms']}ms | "
                f"total={r['elapsed_s']}s"
            )
            if r.get("thinking"):
                print(f"  RAZONAMIENTO: {r['thinking'][:200]}")
            if r.get("text"):
                print(f"  TEXTO: {r['text'][:120]}")
            await asyncio.sleep(2)

        print("\n" + "=" * 70)
        print("RESUMEN:")
        for r in results:
            status = "OK" if r.get("ok") else ("FAIL" if "error" in r else "EMPTY")
            print(
                f"  {r['model']:<28} {status:<6} think={r.get('thinking_chunks', '-'):<4} "
                f"text={r.get('text_chunks', '-'):<4} first_token={r.get('first_token_ms', '-')}ms"
            )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
