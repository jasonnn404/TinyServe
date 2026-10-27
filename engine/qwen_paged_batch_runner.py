
import torch
from torch.profiler import record_function
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from engine.paged_attention import paged_attention_gqa_batch
from engine.triton_batched_paged_attention import (
    prepare_paged_metadata,
    triton_paged_attention_batch,
)


@torch.inference_mode()
def qwen_paged_decode_batch_step(
    model, token_ids, cache, pool, manager, request_ids, positions,
    attention_backend="torch"
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

    if attention_backend not in ("torch", "triton"):
        raise ValueError("Unknown attention backend")

    backbone = model.model
    x = backbone.embed_tokens(token_ids)

    position_ids = torch.tensor(
        positions, device=x.device, dtype=torch.long
    ).unsqueeze(1)

    cos, sin = backbone.rotary_emb(x, position_ids)

    # 1. Prepare KV cache capacity and block tables
    seq_lens = [position + 1 for position in positions]

    for request_id, seq_len in zip(request_ids, seq_lens):
        manager.ensure_capacity(request_id, seq_len)

    block_tables = [
        manager.get_block_table(rid) for rid in request_ids
    ]

    # Prepare GPU metadata once per decode batch
    if attention_backend == "triton":
        tables_gpu, lengths_gpu = prepare_paged_metadata(
            block_tables,
            seq_lens,
            block_size=pool.k_pool.shape[3],
            n_physical=pool.k_pool.shape[1],
            device=token_ids.device,
        )

    # 2. Run all Qwen Transformer layers
    for layer_idx, layer in enumerate(backbone.layers):
        attn = layer.self_attn
        residual = x

        hidden = layer.input_layernorm(x)
        shape = (B, 1, -1, attn.head_dim)

        # 3. Batched Q/K/V projections
        q = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
        v = attn.v_proj(hidden).view(shape).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4. Write current K/V for every request
        for i, request_id in enumerate(request_ids):
            cache.write_token(
                request_id, layer_idx, positions[i],
                k[i, :, 0, :], v[i, :, 0, :]
            )

        # 5. Run our PagedAttention backend
        with record_function("TinyServe/PagedAttention"):
            if attention_backend == "triton":
                attention_output = triton_paged_attention_batch(
                    q[:, :, 0, :].contiguous(),
                    pool.k_pool,
                    pool.v_pool,
                    tables_gpu,
                    lengths_gpu,
                    layer=layer_idx,
                )
            else:
                attention_output = paged_attention_gqa_batch(
                    q[:, :, 0, :],
                    pool.k_pool,
                    pool.v_pool,
                    block_tables,
                    layer=layer_idx,
                    seq_lens=seq_lens,
                )

        # 6. Attention output projection + residual connection
        attention_output = attention_output.reshape(B, 1, -1)
        attention_output = attn.o_proj(attention_output.to(hidden.dtype))

        x = residual + attention_output

        # 7. Feed-forward network + residual connection
        x = x + layer.mlp(layer.post_attention_layernorm(x))

    # 8. Final normalization and next-token logits
    x = backbone.norm(x)
    logits = model.lm_head(x[:, -1, :]).float()

    return logits
