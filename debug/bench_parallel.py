"""Client-side benchmark for parallelism strategies.

Measures prefill/decode throughput purely from streaming timestamps (the server
usage fields are unreliable / zero), so nothing here trusts server-reported
token counts. One request at a time => single-request long-context behaviour.

Usage:
  python bench_parallel.py <model_path> <url> <label> <lengths_csv> <decode_steps>
"""

import json
import random
import subprocess
import sys
import time

import requests
from transformers import AutoTokenizer

MODEL = sys.argv[1]
URL = sys.argv[2].rstrip("/")
LABEL = sys.argv[3]
LENGTHS = [int(x) for x in sys.argv[4].split(",")]
DECODE = int(sys.argv[5]) if len(sys.argv) > 5 else 64

tok = AutoTokenizer.from_pretrained(MODEL)
_BASE = tok(" the quick brown fox jumps over the lazy dog .", add_special_tokens=False).input_ids
_VOCAB = getattr(tok, "vocab_size", 100000)
# Seed uniquely per run so nonces never collide with a prior run's cached KV on
# a reused server (deterministic seeds caused phantom prefix-cache hits).
_rng = random.Random(int(time.time() * 1000) ^ id(object()))


def make_prompt(target_len: int):
    # A unique random prefix per request defeats the radix prefix cache, so every
    # request pays a full prefill (otherwise shorter prompts hit the cached KV of
    # a longer same-text prompt and report bogus, near-instant prefill).
    nonce = [_rng.randint(1000, min(_VOCAB, 150000) - 1) for _ in range(64)]
    body = (_BASE * (target_len // len(_BASE) + 1))[: max(0, target_len - len(nonce))]
    ids = nonce + body
    return tok.decode(ids), len(ids)


def gpu_mem_max_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode()
    return max(int(x) for x in out.split())


# Warm up: the very first request pays one-time lazy-init / allocation costs
# that otherwise corrupt the first measured length. Throw it away.
try:
    _wp, _ = make_prompt(512)
    _r = requests.post(
        URL + "/generate",
        json={"prompt": _wp, "max_tokens": 8, "ignore_eos": True},
        stream=True,
        timeout=300,
    )
    for _line in _r.iter_lines():
        if _line and b"[DONE]" in _line:
            break
    print("warmup done", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"warmup failed: {str(e)[:120]}", flush=True)

results = []
for L in LENGTHS:
    prompt, real_L = make_prompt(L)
    payload = {"prompt": prompt, "max_tokens": DECODE, "ignore_eos": True}
    t0 = time.perf_counter()
    tics = []
    err = None
    try:
        r = requests.post(URL + "/generate", json=payload, stream=True, timeout=2400)
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8", "ignore")
            if s.startswith("data:"):
                if "[DONE]" in s:
                    break
                tics.append(time.perf_counter())
    except Exception as e:  # noqa: BLE001 - benchmark, surface any failure
        err = str(e)[:160]

    if err is not None or not tics:
        rec = {"label": LABEL, "target_len": L, "input_len": real_L, "error": err or "no tokens"}
        print(json.dumps(rec), flush=True)
        results.append(rec)
        continue

    ttft = tics[0] - t0
    n = len(tics)
    decode_span = tics[-1] - tics[0] if n > 1 else 0.0
    rec = {
        "label": LABEL,
        "input_len": real_L,
        "ttft_s": round(ttft, 3),
        "prefill_tps": round(real_L / ttft, 1) if ttft > 0 else None,
        "decode_tokens": n,
        "decode_tps": round((n - 1) / decode_span, 2) if decode_span > 0 else None,
        "gpu_mem_mib": gpu_mem_max_mib(),
    }
    print(json.dumps(rec), flush=True)
    results.append(rec)

print("===SUMMARY===", flush=True)
print(json.dumps(results), flush=True)
