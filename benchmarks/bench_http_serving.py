
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import median
from urllib.request import Request, urlopen


URL = "http://127.0.0.1:8000/generate/stream"
MAX_NEW_TOKENS = 32

PROMPTS = [
    "The capital of Canada is",
    "The capital of France is",
    "Artificial intelligence is",
    "The largest ocean on Earth is",
    "Machine learning is a field of",
    "The capital of Germany is",
    "The largest planet in our solar system is",
    "Python is a programming language that",
]


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def generate(prompt):
    payload = json.dumps({
        "prompt": prompt,
        "max_new_tokens": MAX_NEW_TOKENS,
    }).encode()

    req = Request(
        URL, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_token = None
    last_token = None
    token_ids = []
    event_type = None

    with urlopen(req, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()

            if line.startswith("event:"):
                event_type = line[6:].strip()

            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())

                if event_type == "token":
                    now = time.perf_counter()

                    if first_token is None:
                        first_token = now

                    last_token = now
                    token_ids.append(data["token_id"])

                elif event_type == "error":
                    raise RuntimeError(data.get("detail", "Stream error"))

                elif event_type == "done":
                    end = time.perf_counter()

                    assert first_token is not None
                    assert len(token_ids) <= MAX_NEW_TOKENS

                    return {
                        "ttft": first_token - start,
                        "latency": end - start,
                        "tokens": len(token_ids),
                        "tpot": (
                            (last_token - first_token) / (len(token_ids) - 1)
                            if len(token_ids) > 1 else 0
                        ),
                    }

                event_type = None

    raise RuntimeError("Stream ended without a done event")


def run_load(prompts, concurrency):
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(generate, prompts))

    wall_time = time.perf_counter() - start
    total_tokens = sum(r["tokens"] for r in results)

    return {
        "concurrency": concurrency,
        "ttft_p50": median(r["ttft"] for r in results),
        "ttft_p95": percentile([r["ttft"] for r in results], 95),
        "latency_p50": median(r["latency"] for r in results),
        "tpot_p50": median(r["tpot"] for r in results),
        "throughput": total_tokens / wall_time,
        "total_tokens": total_tokens,
    }


def main():
    print("Warming up the model and CUDA Graphs...")

    for concurrency in [1, 2, 4]:
        run_load(PROMPTS[:concurrency], concurrency)

    print("\nTinyServe End-to-End HTTP Benchmark")
    print(f"Requests per run: {len(PROMPTS)}")
    print(f"Max output tokens: {MAX_NEW_TOKENS}\n")

    print(
        f"{'Concurrency':>11} {'TTFT p50':>11} "
        f"{'TTFT p95':>11} {'E2E p50':>11} "
        f"{'TPOT p50':>11} {'tok/s':>11}"
    )
    print("-" * 72)

    for concurrency in [1, 2, 4, 8]:
        r = run_load(PROMPTS, concurrency)

        print(
            f"{concurrency:>11} "
            f"{r['ttft_p50'] * 1000:>10.1f}ms "
            f"{r['ttft_p95'] * 1000:>10.1f}ms "
            f"{r['latency_p50'] * 1000:>10.1f}ms "
            f"{r['tpot_p50'] * 1000:>10.1f}ms "
            f"{r['throughput']:>11.1f}"
        )

    print("\nHTTP Serving Benchmark: PASS")


if __name__ == "__main__":
    main()
