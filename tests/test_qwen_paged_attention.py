import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from engine.kv_cache_manager import KVCacheManager
from engine.kv_cache_pool import KVCachePool
from engine.paged_kv_cache import PagedKVCache
from engine.paged_attention import paged_attention_gqa

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 16
LAYER = 0

# 1. Load Qwen
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager"
).to(DEVICE).eval()

prompt = "The capital of Canada is Ottawa. " * 4
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

seq_len = inputs["input_ids"].shape[1]
config = model.config
attn = model.model.layers[LAYER].self_attn

# 2. Capture Q and Hugging Face's attention output
captured = {}

def capture_q(_module, _inputs, output):
    captured["q_before_rope"] = output.detach()

def capture_attention(_module, inputs):
    captured["hf_attention"] = inputs[0].detach()

hook_q = attn.q_norm.register_forward_hook(capture_q)
hook_o = attn.o_proj.register_forward_pre_hook(capture_attention)

with torch.inference_mode():
    output = model(**inputs, use_cache=True, logits_to_keep=1)

hook_q.remove()
hook_o.remove()

# 3. Get real K/V from Hugging Face
hf_cache = output.past_key_values

if hasattr(hf_cache, "layers"):
    hf_k = hf_cache.layers[LAYER].keys
    hf_v = hf_cache.layers[LAYER].values
else:
    hf_k = hf_cache.key_cache[LAYER]
    hf_v = hf_cache.value_cache[LAYER]

# 4. Apply RoPE to the final token's Query
q_before = captured["q_before_rope"][:, -1:, :, :].transpose(1, 2)
position_ids = torch.tensor([[seq_len - 1]], device=DEVICE)

with torch.inference_mode():
    cos, sin = model.model.rotary_emb(q_before, position_ids)
    q_after, _ = apply_rotary_pos_emb(q_before, q_before, cos, sin)

q = q_after[0, :, 0, :]  # [16, 128]

# 5. Create our Paged KV Cache
num_blocks = math.ceil(seq_len / BLOCK_SIZE) + 1

manager = KVCacheManager(
    num_blocks=num_blocks, block_size=BLOCK_SIZE
)

pool = KVCachePool(
    num_layers=config.num_hidden_layers,
    num_blocks=num_blocks,
    num_kv_heads=config.num_key_value_heads,
    block_size=BLOCK_SIZE,
    head_dim=config.head_dim,
    dtype=hf_k.dtype,
    device=DEVICE,
)

cache = PagedKVCache(manager, pool)
cache.add_request("A")

# 6. Store real Qwen K/V in our physical blocks
with torch.inference_mode():
    for t in range(seq_len):
        cache.write_token(
            "A", LAYER, t,
            hf_k[0, :, t, :],
            hf_v[0, :, t, :]
        )

    # 7. Compute Attention using our implementation
    actual = paged_attention_gqa(
        q,
        pool.k_pool,
        pool.v_pool,
        manager.get_block_table("A"),
        layer=LAYER,
        seq_len=seq_len,
    )

    # 8. Hugging Face's output before o_proj
    expected = captured["hf_attention"][0, -1].reshape(
        config.num_attention_heads, config.head_dim
    )

    # 9. Compare
    max_error = (actual - expected).abs().max().item()

    torch.testing.assert_close(
        actual, expected, atol=1e-3, rtol=1e-3
    )

print("Query shape:", tuple(q.shape))
print("HF K shape:", tuple(hf_k.shape))
print("HF Attention shape:", tuple(expected.shape))
print("Paged Attention shape:", tuple(actual.shape))
print("Max absolute error:", max_error)
print("Real Qwen PagedAttention: PASS")

cache.free_request("A")