import time
import statistics
import torch
from engine.model_runner import ModelRunner

runner = ModelRunner()
model = runner.model
tokenizer = runner.tokenizer

PROMPT_LENGTHS = [32, 128, 512, 1024, 2048]
PREFILL_RUNS = 6
DECODE_STEPS = 50
DECODE_WARMUP = 5


def sync():
    torch.cuda.synchronize()


def make_input_ids(length):
    text = "Machine learning is transforming how computers understand and generate information. " * 500
    ids = tokenizer(text, return_tensors="pt").input_ids
    return ids[:, :length].to(model.device)


def run_prefill(input_ids):
    attention_mask = torch.ones_like(input_ids)

    sync()
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    sync()
    elapsed = time.perf_counter() - start

    return outputs, attention_mask, elapsed


def benchmark(length):
    input_ids = make_input_ids(length)

    # ---------- Prefill benchmark ----------
    prefill_times = []

    for i in range(PREFILL_RUNS):
        outputs, attention_mask, elapsed = run_prefill(input_ids)

        if i > 0:
            prefill_times.append(elapsed)

    prefill_time = statistics.median(prefill_times)

    # Use the final prefill result to start decode
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    cache_position = torch.tensor(
        [length],
        device=model.device,
    )

    # ---------- Decode benchmark ----------
    decode_times = []

    for step in range(DECODE_STEPS):
        attention_mask = torch.cat([
            attention_mask,
            attention_mask.new_ones((1, 1)),
        ], dim=1)

        sync()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )

        sync()
        elapsed = time.perf_counter() - start

        past_key_values = outputs.past_key_values
        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )
        cache_position += 1

        if step >= DECODE_WARMUP:
            decode_times.append(elapsed)

    decode_time = statistics.median(decode_times)

    return prefill_time, decode_time


# Global warmup
warmup_ids = make_input_ids(32)
with torch.no_grad():
    model(input_ids=warmup_ids, use_cache=True)

print(f"{'Prompt':<10}{'Prefill ms':<15}{'Decode ms':<15}")
print("-" * 40)

for length in PROMPT_LENGTHS:
    prefill, decode = benchmark(length)

    print(
        f"{length:<10}"
        f"{prefill * 1000:<15.2f}"
        f"{decode * 1000:<15.2f}"
    )