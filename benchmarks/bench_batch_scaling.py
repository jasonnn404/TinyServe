import argparse
from statistics import median
from benchmarks.bench_http_serving import PROMPTS, run_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    if args.concurrency < 1 or args.requests < args.concurrency:
        raise ValueError("Require requests >= concurrency >= 1")

    prompts = [
        PROMPTS[i % len(PROMPTS)]
        for i in range(args.requests)
    ]

    print(f"Concurrency: {args.concurrency}")
    print(f"Total requests: {len(prompts)}")
    print("Warming up...")

    for _ in range(2):
        run_load(prompts, args.concurrency)

    results = []

    for i in range(args.trials):
        result = run_load(prompts, args.concurrency)
        results.append(result)

        print(
            f"Trial {i + 1}: "
            f"{result['throughput']:.1f} tok/s | "
            f"TTFT p50={result['ttft_p50'] * 1000:.1f}ms"
        )

    throughput = [r["throughput"] for r in results]
    ttft = [r["ttft_p50"] for r in results]
    latency = [r["latency_p50"] for r in results]

    print("\nSummary")
    print(f"Median throughput: {median(throughput):.1f} tok/s")
    print(f"Throughput range: {min(throughput):.1f}–{max(throughput):.1f} tok/s")
    print(f"Median TTFT p50: {median(ttft) * 1000:.1f}ms")
    print(f"Median E2E p50: {median(latency) * 1000:.1f}ms")


if __name__ == "__main__":
    main()