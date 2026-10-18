from dataclasses import dataclass
import torch


@dataclass
class BatchState:
    requests: list
    past_key_values: object
    attention_mask: torch.Tensor
    next_token_ids: torch.Tensor

    # Number of REAL tokens currently stored for each request.
    sequence_lengths: torch.Tensor

    # Next PHYSICAL position in the shared KV cache.
    cache_position: torch.Tensor