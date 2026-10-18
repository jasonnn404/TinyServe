import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from engine.paged_attention import paged_attention_gqa


@torch.inference_mode()
def qwen_paged_decode_batch_step(
    model, token_ids, cache, pool, manager, request_ids, positions
):
    """
    Batched decode with paged KV storage.

    token_ids:   [B, 1]
    request_ids: list of B request IDs
    positions:   list of B token positions
    returns:     [B, vocab_size]
    """
    B = token_ids.shape[0]

    assert token_ids.shape == (B, 1)
    assert B > 0
    assert len(request_ids) == len(positions) == B
    assert len(set(request_ids)) == B

    backbone = model.model
    x = backbone.embed_tokens(token_ids)

    position_ids = torch.tensor(
        positions, device=x.device, dtype=torch.long
    ).unsqueeze(1)

    cos, sin = backbone.rotary_emb(x, position_ids)

    for layer_idx, layer in enumerate(backbone.layers):
        attn = layer.self_attn
        residual = x

        hidden = layer.input_layernorm(x)
        shape = (B, 1, -1, attn.head_dim)

        # Batched Q/K/V projections
        q = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
        v = attn.v_proj(hidden).view(shape).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        outputs = []

        for i, request_id in enumerate(request_ids):
            position = positions[i]

            # Write this request's K/V
            cache.write_token(
                request_id, layer_idx, position,
                k[i, :, 0, :], v[i, :, 0, :]
            )

            # Each request has its own block table
            out = paged_attention_gqa(
                q[i, :, 0, :],
                pool.k_pool,
                pool.v_pool,
                manager.get_block_table(request_id),
                layer=layer_idx,
                seq_len=position + 1,
            )
            outputs.append(out)

        attention_output = torch.stack(outputs, dim=0)
        attention_output = attention_output.reshape(B, 1, -1)
        attention_output = attn.o_proj(attention_output.to(hidden.dtype))

        x = residual + attention_output
        x = x + layer.mlp(layer.post_attention_layernorm(x))

    x = backbone.norm(x)
    return model.lm_head(x[:, -1, :]).float()