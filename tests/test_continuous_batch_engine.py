import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.request import Request
from engine.scheduler import Scheduler
from engine.kv_cache_manager import KVCacheManager
from engine.kv_cache_pool import KVCachePool
from engine.paged_kv_cache import PagedKVCache
from engine.continuous_batch_engine import ContinuousBatchEngine

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEVICE = "cuda"
BLOCK_SIZE = 16

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager"
).to(DEVICE).eval()


def make_request(request_id, prompt, max_new_tokens):
    return Request(
        request_id=request_id,
        prompt=prompt,
        prompt_ids=tokenizer(prompt)["input_ids"],
        max_new_tokens=max_new_tokens,
    )


requests = [
    make_request("A", "The capital of Canada is", 2),
    make_request("B", "Machine learning systems must", 5),
    make_request("C", "Paris is the capital of", 3),
]

# Enough memory for all test requests, including generated tokens.
num_blocks = sum(
    math.ceil((len(r.prompt_ids) + r.max_new_tokens) / BLOCK_SIZE)
    for r in requests
) + 1

manager = KVCacheManager(num_blocks=num_blocks, block_size=BLOCK_SIZE)
config = model.config

pool = KVCachePool(
    num_layers=config.num_hidden_layers,
    num_blocks=num_blocks,
    num_kv_heads=config.num_key_value_heads,
    block_size=BLOCK_SIZE,
    head_dim=config.head_dim,
    dtype=torch.float32,
    device=DEVICE,
)

cache = PagedKVCache(manager, pool)
scheduler = Scheduler(
    max_batch_size=2,
    total_kv_blocks=num_blocks,
    block_size=BLOCK_SIZE,
)

engine = ContinuousBatchEngine(
    model, scheduler, cache, pool, manager,
    eos_token_id=tokenizer.eos_token_id,
    attention_backend="cuda_graph",
)

# Initially, only A and B arrive.
scheduler.add_request(requests[0])
scheduler.add_request(requests[1])

tick = 0

while engine.has_work:
    tick += 1
    events = engine.step()

    print(f"\nTick {tick}")
    print("Decode batch:", engine.last_decode_batch)

    for request_id, phase, token in events:
        print(f"{request_id} | {phase} | {tokenizer.decode([token])!r}")

    # C arrives while the engine is running.
    if tick == 1:
        scheduler.add_request(requests[2])
        print("New request C arrived!")

    # Check that active requests never share physical blocks.
    tables = [
        set(manager.get_block_table(r))
        for r in engine.active
    ]
    for i in range(len(tables)):
        for j in range(i + 1, len(tables)):
            assert tables[i].isdisjoint(tables[j])

    assert tick < 20, "Engine did not terminate"

# All requests must finish.
assert set(engine.finished) == {"A", "B", "C"}

for req in requests:
    assert req.is_finished
    assert 1 <= len(req.generated_ids) <= req.max_new_tokens

    print(f"\nRequest {req.request_id}:")
    print(tokenizer.decode(req.generated_ids))

captured = engine.graph_dispatcher.captured_batch_sizes()

print("Captured graphs:", captured)
assert captured == [1, 2]

print("Continuous Batching Integration: PASS")