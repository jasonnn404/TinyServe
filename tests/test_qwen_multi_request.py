import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.kv_cache_manager import KVCacheManager
from engine.kv_cache_pool import KVCachePool
from engine.paged_kv_cache import PagedKVCache
from engine.qwen_paged_runner import qwen_paged_decode_step
from engine.qwen_paged_batch_runner import qwen_paged_decode_batch_step

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEVICE = "cuda"
BLOCK_SIZE = 16
DECODE_STEPS = 5

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager"
).to(DEVICE).eval()

prompts = {
    "A": "The capital of Canada is Ottawa. " * 4,
    "B": "Machine learning systems must efficiently manage memory.",
}

# 1. Prefill both requests using Hugging Face
states = {}

with torch.inference_mode():
    for request_id, prompt in prompts.items():
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        result = model(**inputs, use_cache=True, logits_to_keep=1)

        next_token = result.logits[:, -1, :].argmax(-1, keepdim=True)

        states[request_id] = {
            "seq_len": inputs.input_ids.shape[1],
            "hf_cache": result.past_key_values,
            "token_ids": next_token,
            "generated": [next_token.item()],
        }

# 2. Create ONE shared KV pool
num_blocks = sum(
    math.ceil((s["seq_len"] + DECODE_STEPS) / BLOCK_SIZE)
    for s in states.values()
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

# 3. Transfer each request's real K/V
with torch.inference_mode():
    for request_id, state in states.items():
        cache.add_request(request_id)
        hf_cache = state["hf_cache"]

        for layer in range(config.num_hidden_layers):
            if hasattr(hf_cache, "layers"):
                k = hf_cache.layers[layer].keys
                v = hf_cache.layers[layer].values
            else:
                k = hf_cache.key_cache[layer]
                v = hf_cache.value_cache[layer]

            for t in range(state["seq_len"]):
                cache.write_token(
                    request_id, layer, t,
                    k[0, :, t, :], v[0, :, t, :]
                )

# 4. Check physical block isolation
def check_isolation():
    a = set(manager.get_block_table("A"))
    b = set(manager.get_block_table("B"))
    assert a.isdisjoint(b), "Physical block collision!"

check_isolation()

print("Initial A blocks:", manager.get_block_table("A"))
print("Initial B blocks:", manager.get_block_table("B"))

# 5. Batched decoding
with torch.inference_mode():
    for step in range(DECODE_STEPS):
        request_ids = list(states.keys())

        # Combine requests into ONE batch
        batch_tokens = torch.cat(
            [states[r]["token_ids"] for r in request_ids], dim=0
        )

        positions = [
            states[r]["seq_len"] + step for r in request_ids
        ]

        # ONE batched model forward
        batch_logits = qwen_paged_decode_batch_step(
            model, batch_tokens, cache, pool, manager,
            request_ids=request_ids, positions=positions,
            attention_backend="triton",
        )

        print("Batch logits shape:", tuple(batch_logits.shape))

        # Check each request against Hugging Face
        for i, request_id in enumerate(request_ids):
            state = states[request_id]
            actual = batch_logits[i:i + 1]

            reference = model(
                input_ids=state["token_ids"],
                past_key_values=state["hf_cache"],
                use_cache=True,
                logits_to_keep=1,
            ).logits[:, -1, :]

            torch.testing.assert_close(
                actual, reference, atol=1e-3, rtol=1e-3
            )

            next_token = actual.argmax(-1, keepdim=True)

            assert torch.equal(
                next_token, reference.argmax(-1, keepdim=True)
            )

            state["generated"].append(next_token.item())
            state["token_ids"] = next_token

            error = (actual - reference).abs().max().item()
            print(f"Step {step + 1} | {request_id} | error={error:.2e}")

        check_isolation()

# 6. Show results
for request_id, state in states.items():
    print(f"\nRequest {request_id}:")
    print("Text:", tokenizer.decode(state["generated"]))
    print("Blocks:", manager.get_block_table(request_id))

print("\nMulti-Request KV Isolation: PASS")

cache.free_request("A")
cache.free_request("B")