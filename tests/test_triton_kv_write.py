
import torch

from engine.triton_kv_write import triton_write_kv_batch


@torch.inference_mode()
def main():
    torch.manual_seed(42)

    device = "cuda"
    B, H, D = 3, 8, 128
    BLOCK_SIZE = 16
    LAYER = 1

    k_pool = torch.full(
        (3, 12, H, BLOCK_SIZE, D), -999.0, device=device
    )
    v_pool = torch.full_like(k_pool, -999.0)

    # Fixed GPU memory addresses
    static_k = torch.empty(B, H, D, device=device)
    static_v = torch.empty_like(static_k)

    tables_gpu = torch.tensor([
        [7, 2, 5],
        [1, 4, 8],
        [0, 6, 9],
    ], device=device, dtype=torch.int32)

    lengths_gpu = torch.tensor(
        [16, 17, 32], device=device, dtype=torch.int32
    )

    def write():
        triton_write_kv_batch(
            static_k, static_v, k_pool, v_pool,
            tables_gpu, lengths_gpu, layer=LAYER
        )

    # Warm up the Triton kernel on a side stream
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        static_k.zero_()
        static_v.zero_()
        write()

    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    k_pool.fill_(-999.0)
    v_pool.fill_(-999.0)
    torch.cuda.synchronize()

    # Capture one GPU KV write
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write()

    def check(expected_k, expected_v, tables, lengths):
        for i in range(B):
            pos = lengths[i] - 1
            physical = tables[i][pos // BLOCK_SIZE]
            offset = pos % BLOCK_SIZE

            torch.testing.assert_close(
                k_pool[LAYER, physical, :, offset, :],
                expected_k[i], atol=0, rtol=0
            )
            torch.testing.assert_close(
                v_pool[LAYER, physical, :, offset, :],
                expected_v[i], atol=0, rtol=0
            )

    # First replay
    k1 = torch.randn_like(static_k)
    v1 = torch.randn_like(static_v)

    static_k.copy_(k1)
    static_v.copy_(v1)
    graph.replay()
    torch.cuda.synchronize()

    tables1 = [[7, 2, 5], [1, 4, 8], [0, 6, 9]]
    lengths1 = [16, 17, 32]
    check(k1, v1, tables1, lengths1)
    print("Replay 1: PASS")

    # Second replay: update the metadata in the SAME buffers
    tables2 = [[7, 10, 5], [1, 4, 8], [0, 6, 9]]
    lengths2 = [17, 18, 33]

    tables_gpu.copy_(torch.tensor(
        tables2, device=device, dtype=torch.int32
    ))
    lengths_gpu.copy_(torch.tensor(
        lengths2, device=device, dtype=torch.int32
    ))

    k2 = torch.randn_like(static_k)
    v2 = torch.randn_like(static_v)

    static_k.copy_(k2)
    static_v.copy_(v2)

    graph.replay()
    torch.cuda.synchronize()

    check(k2, v2, tables2, lengths2)

    # The previous KV writes must still be intact
    check(k1, v1, tables1, lengths1)

    # A now writes to physical block 10, not the old block 2
    assert torch.all(k_pool[LAYER, 2, :, 0, :] == -999.0)
    assert torch.all(v_pool[LAYER, 2, :, 0, :] == -999.0)

    print("Replay 2 (updated block tables): PASS")
    print("Previous KV preserved: PASS")
    print("CUDA Graph Paged KV Write: PASS")


if __name__ == "__main__":
    main()
