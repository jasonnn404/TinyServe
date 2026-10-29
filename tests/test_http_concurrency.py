
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from urllib.request import Request, urlopen


URL = "http://127.0.0.1:8000/generate"
MAX_NEW_TOKENS = 16

BASE_PROMPTS = [
    "The capital of France is",
    "The capital of Canada is",
    "Artificial intelligence is",
    "The largest planet in our solar system is",
    "Machine learning is a field of",
    "The capital of Germany is",
    "Python is a programming language that",
    "The largest ocean on Earth is",
]

PROMPTS = BASE_PROMPTS + [
    prompt + " " + "Please provide a detailed explanation. " * repeats
    for repeats in (1, 3, 5)
    for prompt in BASE_PROMPTS
]

assert len(PROMPTS) == 32


def generate(prompt, barrier=None):
    if barrier is not None:
        barrier.wait(timeout=30)

    payload = json.dumps({
        "prompt": prompt,
        "max_new_tokens": MAX_NEW_TOKENS,
    }).encode("utf-8")

    request = Request(
        URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=120) as response:
        assert response.status == 200
        return json.load(response)


def main():
    print("Step 1: Sequential reference generation")

    references = []
    for i, prompt in enumerate(PROMPTS):
        result = generate(prompt)
        references.append(result["generated_token_ids"])
        print(f"  Reference {i + 1}: PASS")

    print(f"\nStep 2: Send {len(PROMPTS)} concurrent requests")

    barrier = Barrier(len(PROMPTS))

    with ThreadPoolExecutor(max_workers=len(PROMPTS)) as executor:
        futures = [
            executor.submit(generate, prompt, barrier)
            for prompt in PROMPTS
        ]
        results = [future.result() for future in futures]

    print("\nStep 3: Verify responses")

    request_ids = [r["request_id"] for r in results]
    assert len(set(request_ids)) == len(PROMPTS)
    print("Unique request IDs: PASS")

    for i, result in enumerate(results):
        token_ids = result["generated_token_ids"]

        assert 1 <= len(token_ids) <= MAX_NEW_TOKENS
        assert token_ids == references[i], (
            f"Token mismatch for prompt {i + 1}"
        )

        print(
            f"  Request {i + 1}: "
            f"{len(token_ids)} tokens | PASS"
        )

    print("\nConcurrent HTTP Correctness: PASS")
    print("Check server logs to verify actual GPU batching.")


if __name__ == "__main__":
    main()
