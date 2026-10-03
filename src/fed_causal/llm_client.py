from __future__ import annotations

import os
import threading
import time
from typing import Any

API_KEY = os.environ.get("FED_CAUSAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
API_BASE_URL = os.environ.get("FED_CAUSAL_API_BASE_URL") or os.environ.get(
    "OPENAI_BASE_URL"
)
DEPLOYMENT_NAME = os.environ.get("FED_CAUSAL_MODEL", "gpt-5.4-mini")

PRICE_INPUT_PER_1M = 0.25
PRICE_OUTPUT_PER_1M = 2.00


_client = None
_lock = threading.Lock()
_counter = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "usd_estimated": 0.0,
    "errors": 0,
    "retries": 0,
}


def get_client():
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "Set FED_CAUSAL_API_KEY or OPENAI_API_KEY before API-backed runs."
            )
        from openai import OpenAI

        kwargs = {"api_key": API_KEY}
        if API_BASE_URL:
            kwargs["base_url"] = API_BASE_URL
        _client = OpenAI(**kwargs)
    return _client


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * PRICE_INPUT_PER_1M
            + completion_tokens * PRICE_OUTPUT_PER_1M) / 1_000_000.0


def chat(messages: list[dict[str, str]],
         max_tokens: int = 512,
         temperature: float = 0.0,
         model: str = DEPLOYMENT_NAME,
         max_retries: int = 4,
         timeout: float = 60.0,
         **kwargs) -> tuple[str, dict[str, Any]]:
    client = get_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    **kwargs,
                )
            except TypeError:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    **kwargs,
                )
            elapsed = time.time() - t0

            text = resp.choices[0].message.content or ""
            u = resp.usage
            pt = getattr(u, "prompt_tokens", 0) or 0
            ct = getattr(u, "completion_tokens", 0) or 0
            tt = getattr(u, "total_tokens", pt + ct) or (pt + ct)
            usd = _estimate_cost(pt, ct)
            usage = {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
                "usd_estimated": usd,
                "elapsed_s": elapsed,
            }
            with _lock:
                _counter["calls"] += 1
                _counter["prompt_tokens"] += pt
                _counter["completion_tokens"] += ct
                _counter["total_tokens"] += tt
                _counter["usd_estimated"] += usd
            return text, usage
        except Exception as e:
            last_err = e
            msg = str(e)
            with _lock:
                _counter["retries"] += 1
            if "429" in msg or "rate" in msg.lower():
                time.sleep(min(2 ** attempt, 10))
            else:
                time.sleep(0.5 * (attempt + 1))
            continue
    with _lock:
        _counter["errors"] += 1
    raise RuntimeError(f"LLM chat failed after {max_retries} retries: {last_err}")


def reset_counter():
    with _lock:
        for k in _counter:
            _counter[k] = 0 if isinstance(_counter[k], int) else 0.0


def get_counter() -> dict[str, Any]:
    with _lock:
        return dict(_counter)


if __name__ == "__main__":
    t, u = chat([{"role": "user", "content": "Reply with just OK."}], max_tokens=8)
    print("response:", repr(t))
    print("usage:", u)
    print("counter:", get_counter())
