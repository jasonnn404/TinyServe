import time
import torch
from engine.model_runner import ModelRunner


def measure(fn, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()

    result = fn(*args, **kwargs)

    torch.cuda.synchronize()
    end = time.perf_counter()

    return result, end - start


runner = ModelRunner()

prompt = "The capital of Canada is"
max_new_tokens = 100

# Warmup
runner.generate_naive(prompt, max_new_tokens=5)
runner.generate_cached(prompt, max_new_tokens=5)

naive_text, naive_time = measure(
    runner.generate_naive,
    prompt,
    max_new_tokens=max_new_tokens,
)

cached_text, cached_time = measure(
    runner.generate_cached,
    prompt,
    max_new_tokens=max_new_tokens,
)

print("Same output:", naive_text == cached_text)
print()
print(f"Naive:  {naive_time:.3f} s")
print(f"Cached: {cached_time:.3f} s")
print(f"Speedup: {naive_time / cached_time:.2f}x")