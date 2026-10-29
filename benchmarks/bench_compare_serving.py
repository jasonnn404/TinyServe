
import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import median
from urllib.request import Request, urlopen

from benchmarks.bench_http_serving import PROMPTS


URLS = {
    "tinyserve": "http://127.0.0.1:8000/generate/stream",
    "vllm": "http://127.0.0.1:8001/v1/completions",
}
MODEL_ID = "Qwen/Qwen3-0.6B-Base"


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def read_sse(response):
    event = "message"
    data_lines = []

    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")

        if not line:
            if data_lines:
                yield event, "\n".join(data_lines)
            event, data_lines = "message", []
            continue

        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))

    if data_lines:
        yield event, "\n".join(data_lines)


def generate(backend, prompt, max_tokens):
    if backend == "tinyserve":
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_tokens,
        }
    else:
        payload = {
            "model": MODEL_ID,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    request = Request(
        URLS[backend],
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_token = None
    end = None
    token_count = 0
    completed = False

    with urlopen(request, timeout=180) as response:
        for event, raw_data in read_sse(response):
            now = time.perf_counter()

            if backend == "vllm" and raw_data == "[DONE]":
                end = now
                completed = True
                break

            data = json.loads(raw_data)

            if backend == "tinyserve":
                if event == "error":
                    raise RuntimeError(data.get("detail", "Stream error"))

                if event == "token":
                    if first_token is None:
                        first_token = now
                    token_count += 1

                elif event == "done":
                    end = now
                    completed = True
                    break

            else:
                if "error" in data:
                    raise RuntimeError(str(data["error"]))

                usage = data.get("usage")
                if usage is not None:
                    token_count = usage["completion_tokens"]

                choices = data.get("choices") or []
                if choices:
                    piece = choices[0].get("text") or ""
                    if piece and first_token is None:
                        first_token = now

    if not completed or first_token is None or token_count < 1:
        raise RuntimeError(
            f"Incomplete {backend} response: "
            f"completed={completed}, tokens={token_count}"
        )

    latency = end - start
    ttft = first_token - start

    # Post-first-token time per remaining output token.
    # A cross-backend TPOT proxy, not exact per-token GPU latency.
    tpot_proxy = (
        (end - first_token) / (token_count - 1)
        if token_count > 1 else None
    )

    return {
        "tokens": token_count,
        "ttft": ttft,
        "latency": latency,
        "tpot_proxy": tpot_proxy,
    }


def run_load(backend, prompts, concurrency, max_tokens):
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(generate, backend, p, max_tokens)
            for p in prompts
        ]
        results = [f.result() for f in futures]

    wall_time = time.perf_counter() - start
    total_tokens = sum(r["tokens"] for r in results)
    tpot_values = [
        r["tpot_proxy"] for r in results
        if r["tpot_proxy"] is not None
    ]

    return {
        "throughput": total_tokens / wall_time,
        "ttft_p50": median(r["ttft"] for r in results),
        "ttft_p95": percentile([r["ttft"] for r in results], 95),
        "e2e_p50": median(r["latency"] for r in results),
        "tpot_p50": median(tpot_values) if tpot_values else 0,
        "tokens": total_tokens,
        "wall_time": wall_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=URLS, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    if not (1 <= args.concurrency <= args.requests):
        raise ValueError("Require 1 <= concurrency <= requests")
    if args.trials < 1 or args.max_tokens < 1:
        raise ValueError("Trials and max-tokens must be positive")

    prompts = [
        PROMPTS[i % len(PROMPTS)]
        for i in range(args.requests)
    ]

    print(f"Backend: {args.backend}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Requests: {args.requests}")
    print(f"Max output tokens: {args.max_tokens}")
    print("Warming up...")

    for _ in range(2):
        run_load(
            args.backend, prompts,
            args.concurrency, args.max_tokens
        )

    results = []

    for i in range(args.trials):
        r = run_load(
            args.backend, prompts,
            args.concurrency, args.max_tokens
        )
        results.append(r)

        print(
            f"Trial {i + 1}: "
            f"{r['throughput']:.1f} tok/s | "
            f"TTFT={r['ttft_p50'] * 1000:.1f}ms | "
            f"E2E={r['e2e_p50'] * 1000:.1f}ms | "
            f"tokens={r['tokens']}"
        )

    print("\nSummary (median across trials)")
    for name in ("throughput", "ttft_p50", "ttft_p95",
                 "e2e_p50", "tpot_p50"):
        values = [r[name] for r in results]
        value = median(values)

        if name == "throughput":
            print(f"{name}: {value:.1f} tok/s")
        else:
            print(f"{name}: {value * 1000:.1f} ms")

    print(
        "Throughput range:",
        f"{min(r['throughput'] for r in results):.1f}–"
        f"{max(r['throughput'] for r in results):.1f} tok/s"
    )
    print("Unified HTTP Benchmark: PASS")


if __name__ == "__main__":
    main()
