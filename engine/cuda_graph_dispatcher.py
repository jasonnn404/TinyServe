
import torch

from engine.qwen_cuda_graph_runner import QwenCUDAGraphDecoder
from engine.qwen_paged_batch_runner import qwen_paged_decode_batch_step


class CUDAGraphDispatcher:
    def __init__(
        self, model, cache, pool, manager,
        max_seq_len=512, graph_batch_sizes=(1, 2, 4),
    ):
        self.model = model
        self.cache = cache
        self.pool = pool
        self.manager = manager

        block_size = pool.k_pool.shape[3]
        self.max_blocks = (max_seq_len + block_size - 1) // block_size
        self.graph_batch_sizes = set(graph_batch_sizes)

        # One captured decoder per supported batch size
        self.graphs = {}

    @torch.inference_mode()
    def decode(self, token_ids, request_ids, positions):
        batch_size = token_ids.shape[0]

        if batch_size == 0:
            raise ValueError("Empty decode batch")
        if token_ids.shape != (batch_size, 1):
            raise ValueError("Expected token_ids [B, 1]")
        if len(request_ids) != batch_size or len(positions) != batch_size:
            raise ValueError("Batch metadata mismatch")
        if len(set(request_ids)) != batch_size:
            raise ValueError("Duplicate request IDs")

        # Fallback: retain the existing, validated runner
        if (
            batch_size not in self.graph_batch_sizes
            or any((p + 1) > self.max_blocks * self.pool.k_pool.shape[3]
                   for p in positions)
        ):
            return qwen_paged_decode_batch_step(
                self.model, token_ids, self.cache,
                self.pool, self.manager,
                request_ids, positions,
                attention_backend="triton",
            )

        decoder = self.graphs.get(batch_size)

        if decoder is None:
            decoder = QwenCUDAGraphDecoder(
                self.model, self.pool,
                batch_size=batch_size,
                max_blocks=self.max_blocks,
            )

            # Prepare valid data before capture
            decoder.prepare(
                token_ids, self.manager, request_ids, positions
            )
            decoder.capture()
            self.graphs[batch_size] = decoder
        else:
            # Same graph, updated request metadata
            decoder.prepare(
                token_ids, self.manager, request_ids, positions
            )

        return decoder.replay()

    def captured_batch_sizes(self):
        return sorted(self.graphs.keys())
