import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from engine.paged_attention import paged_attention_gqa


@torch.inference_mode()
def qwen_paged_decode_step(
    model, token_ids, cache, pool, manager, request_id, position
):
    """Decode one token using our Paged KV Cache and Attention."""
    assert token_ids.shape == (1, 1)

    backbone = model.model
    x = backbone.embed_tokens(token_ids)

    position_ids = torch.tensor([[position]], device=x.device)
    cos, sin = backbone.rotary_emb(x, position_ids)

    for layer_idx, layer in enumerate(backbone.layers):
        attn = layer.self_attn
        residual = x

        # 1. Prepare hidden states
        hidden = layer.input_layernorm(x)
        shape = (1, 1, -1, attn.head_dim)

        # 2. Compute Q/K/V using Qwen's trained weights
        q = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
        v = attn.v_proj(hidden).view(shape).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Store the new K/V in our Paged KV Cache
        cache.write_token(
            request_id, layer_idx, position,
            k[0, :, 0, :], v[0, :, 0, :]
        )

        # 4. Run OUR PagedAttention
        attention_output = paged_attention_gqa(
            q[0, :, 0, :],
            pool.k_pool,
            pool.v_pool,
            manager.get_block_table(request_id),
            layer=layer_idx,
            seq_len=position + 1,
        )

        # 5. Continue through Qwen's output projection and FFN
        attention_output = attention_output.reshape(1, 1, -1)
        attention_output = attn.o_proj(attention_output.to(hidden.dtype))

        x = residual + attention_output
        x = x + layer.mlp(layer.post_attention_layernorm(x))

    # 6. Predict the next token
    x = backbone.norm(x)
    logits = model.lm_head(x[:, -1, :]).float()

    return logits