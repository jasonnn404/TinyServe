
import torch
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from engine.triton_batched_paged_attention import triton_paged_attention_batch
from engine.triton_kv_write import triton_write_kv_batch


class QwenCUDAGraphDecoder:
    def __init__(self, model, pool, batch_size, max_blocks):
        self.model = model
        self.pool = pool
        self.batch_size = batch_size
        self.max_blocks = max_blocks
        self.block_size = pool.k_pool.shape[3]
        self.device = pool.k_pool.device

        # Fixed GPU buffers: their addresses must stay unchanged
        self.tokens = torch.empty(
            (batch_size, 1), dtype=torch.long, device=self.device
        )
        self.tables = torch.full(
            (batch_size, max_blocks), -1,
            dtype=torch.int32, device=self.device
        )
        self.lengths = torch.empty(
            batch_size, dtype=torch.int32, device=self.device
        )

        # Reusable CPU staging buffers
        self.host_tables = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32
        )
        self.host_lengths = torch.empty(
            batch_size, dtype=torch.int32
        )

        self.graph = None
        self.output = None

    def prepare(self, token_ids, manager, request_ids, positions):
        """Update the static GPU buffers before a graph replay."""
        B = self.batch_size

        if token_ids.shape != (B, 1):
            raise ValueError("Invalid token shape")
        if len(request_ids) != B or len(positions) != B:
            raise ValueError("Batch metadata mismatch")

        self.host_tables.fill_(-1)

        for i, (rid, position) in enumerate(zip(request_ids, positions)):
            seq_len = position + 1
            needed = (seq_len + self.block_size - 1) // self.block_size

            if needed > self.max_blocks:
                raise ValueError("Sequence exceeds graph block capacity")

            # CPU allocation remains outside the graph
            manager.ensure_capacity(rid, seq_len)
            table = manager.get_block_table(rid)

            if len(table) < needed:
                raise ValueError("Incomplete block table")

            for j in range(needed):
                self.host_tables[i, j] = table[j]

            self.host_lengths[i] = seq_len

        # Copy contents into existing GPU allocations
        self.tokens.copy_(token_ids)
        self.tables.copy_(self.host_tables)
        self.lengths.copy_(self.host_lengths)

    @torch.inference_mode()
    def _forward(self):
        """GPU-only decode: no Python KV manager operations."""
        model = self.model
        backbone = model.model
        B = self.batch_size

        x = backbone.embed_tokens(self.tokens)

        # Derive positions from GPU lengths, not Python lists
        position_ids = (self.lengths.to(torch.long) - 1).unsqueeze(1)
        cos, sin = backbone.rotary_emb(x, position_ids)

        for layer_idx, layer in enumerate(backbone.layers):
            attn = layer.self_attn
            residual = x

            hidden = layer.input_layernorm(x)
            shape = (B, 1, -1, attn.head_dim)

            q = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
            v = attn.v_proj(hidden).view(shape).transpose(1, 2)

            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # GPU determines destination physical blocks
            triton_write_kv_batch(
                k[:, :, 0, :].contiguous(),
                v[:, :, 0, :].contiguous(),
                self.pool.k_pool,
                self.pool.v_pool,
                self.tables,
                self.lengths,
                layer=layer_idx,
            )

            # GPU reads the same updated metadata
            attention_output = triton_paged_attention_batch(
                q[:, :, 0, :].contiguous(),
                self.pool.k_pool,
                self.pool.v_pool,
                self.tables,
                self.lengths,
                layer=layer_idx,
            )

            attention_output = attention_output.reshape(B, 1, -1)
            attention_output = attn.o_proj(
                attention_output.to(hidden.dtype)
            )

            x = residual + attention_output
            x = x + layer.mlp(layer.post_attention_layernorm(x))

        x = backbone.norm(x)
        return model.lm_head(x[:, -1, :]).float()

    @torch.inference_mode()
    def capture(self):
        """Warm up CUDA kernels, then capture one decode step."""
        if self.graph is not None:
            raise RuntimeError("Graph already captured")

        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))

        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward()

        torch.cuda.current_stream(self.device).wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph):
            self.output = self._forward()

    @torch.inference_mode()
    def replay(self):
        if self.graph is None:
            raise RuntimeError("Call capture() first")

        self.graph.replay()
        return self.output
