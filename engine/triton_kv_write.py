
import torch
import triton
import triton.language as tl


@triton.jit
def _kv_write_kernel(
    K, V, K_POOL, V_POOL, TABLES, LENGTHS,
    LAYER,
    N_PHYSICAL: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    D: tl.constexpr,
):
    request_idx = tl.program_id(0)
    kv_head = tl.program_id(1)
    d = tl.arange(0, D)

    # Current token is at position seq_len - 1
    seq_len = tl.load(LENGTHS + request_idx)
    position = seq_len - 1

    logical_block = position // BLOCK_SIZE
    token_offset = position % BLOCK_SIZE

    # Read the updated physical block ID from GPU metadata
    physical = tl.load(
        TABLES + request_idx * MAX_BLOCKS + logical_block
    )

    # New K/V input: [B, H_kv, D]
    src = (request_idx * N_KV_HEADS + kv_head) * D + d

    # Paged pool: [layers, physical_blocks, H_kv, block_size, D]
    dst = (
        ((LAYER * N_PHYSICAL + physical) * N_KV_HEADS + kv_head)
        * BLOCK_SIZE + token_offset
    ) * D + d

    k = tl.load(K + src)
    v = tl.load(V + src)

    tl.store(K_POOL + dst, k)
    tl.store(V_POOL + dst, v)


def triton_write_kv_batch(
    k, v, k_pool, v_pool, tables_gpu, lengths_gpu, layer
):
    """Write one new token per request into the paged KV pool."""
    if k.ndim != 3 or k_pool.ndim != 5:
        raise ValueError("Invalid tensor dimensions")

    B, n_kv_heads, D = k.shape
    n_layers, n_physical, pool_heads, block_size, pool_dim = k_pool.shape

    if B < 1 or k.shape != v.shape:
        raise ValueError("Invalid K/V shapes")
    if n_kv_heads != pool_heads or D != pool_dim:
        raise ValueError("KV head dimensions do not match")
    if v_pool.shape != k_pool.shape:
        raise ValueError("K/V pool shape mismatch")
    if tables_gpu.ndim != 2 or tables_gpu.shape[0] != B:
        raise ValueError("Invalid block table dimensions")
    if lengths_gpu.shape != (B,):
        raise ValueError("Invalid sequence length dimensions")
    if not 0 <= layer < n_layers:
        raise ValueError("Invalid layer")

    tensors = (k, v, k_pool, v_pool, tables_gpu, lengths_gpu)
    if not all(x.is_cuda and x.is_contiguous() for x in tensors):
        raise ValueError("Expected contiguous CUDA tensors")
    if len({x.device for x in tensors}) != 1:
        raise ValueError("All tensors must be on the same GPU")
    if any(x.dtype != torch.float32 for x in (k, v, k_pool, v_pool)):
        raise ValueError("Expected FP32 K/V tensors")
    if tables_gpu.dtype != torch.int32 or lengths_gpu.dtype != torch.int32:
        raise ValueError("Expected int32 metadata")
    if triton.next_power_of_2(block_size) != block_size:
        raise ValueError("Block size must be a power of two")
    if triton.next_power_of_2(D) != D:
        raise ValueError("Head dimension must be a power of two")

    # Caller must ensure capacity and valid, non-conflicting block IDs.
    _kv_write_kernel[(B, n_kv_heads)](
        k, v, k_pool, v_pool, tables_gpu, lengths_gpu,
        layer,
        N_PHYSICAL=n_physical,
        N_KV_HEADS=n_kv_heads,
        BLOCK_SIZE=block_size,
        MAX_BLOCKS=tables_gpu.shape[1],
        D=D,
        num_warps=4,
    )
