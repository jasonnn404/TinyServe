import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.kv_cache_manager import KVCacheManager
from engine.kv_cache_pool import KVCachePool
from engine.paged_kv_cache import PagedKVCache
from engine.qwen_paged_runner import qwen_paged_decode_step

NUM_NEW_TOKENS = 8
MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 16

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager"
).to(DEVICE).eval()

inputs = tokenizer(
    "The capital of Canada is Ottawa. " * 4,
    return_tensors="pt"
).to(DEVICE)

seq_len = inputs.input_ids.shape[1]
config = model.config

# 1. Hugging Face prefill
with torch.inference_mode():
    prefill = model(**inputs, use_cache=True, logits_to_keep=1)

hf_cache = prefill.past_key_values
next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

# 2. Initialize TinyServe KV Cache
num_blocks = math.ceil((seq_len + NUM_NEW_TOKENS) / BLOCK_SIZE) + 1

manager = KVCacheManager(num_blocks=num_blocks, block_size=BLOCK_SIZE)

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
cache.add_request("A")

# 3. Transfer prefill K/V into TinyServe
with torch.inference_mode():
    for layer in range(config.num_hidden_layers):
        if hasattr(hf_cache, "layers"):
            k = hf_cache.layers[layer].keys
            v = hf_cache.layers[layer].values
        else:
            k = hf_cache.key_cache[layer]
            v = hf_cache.value_cache[layer]

        for t in range(seq_len):
            cache.write_token("A", layer, t, k[0, :, t, :], v[0, :, t, :])

    # 4. Generate tokens autoregressively
    generated_ids = [next_token.item()]
    token_ids = next_token

    for step in range(NUM_NEW_TOKENS - 1):
        position = seq_len + step

        # TinyServe decoding
        actual = qwen_paged_decode_step(
            model, token_ids, cache, pool, manager,
            request_id="A", position=position
        )

        # Hugging Face reference decoding
        reference = model(
            input_ids=token_ids,
            past_key_values=hf_cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits[:, -1, :]

        # Compare both implementations
        torch.testing.assert_close(
            actual, reference, atol=1e-3, rtol=1e-3
        )

        next_actual = actual.argmax(dim=-1, keepdim=True)
        next_reference = reference.argmax(dim=-1, keepdim=True)

        assert torch.equal(next_actual, next_reference)

        generated_ids.append(next_actual.item())
        token_ids = next_actual

        error = (actual - reference).abs().max().item()
        print(f"Step {step + 1}: max logits error = {error:.2e}")

print("Generated text:", tokenizer.decode(generated_ids))
print("Block table:", manager.get_block_table("A"))
print("Multi-token Paged Decoding: PASS")

cache.free_request("A")