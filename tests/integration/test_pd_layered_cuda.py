"""Real CUDA pack/install ordering; this does not replace native RDMA acceptance."""

import minisgl.distributed.info as tp_info
import pytest
import torch
from minisgl.kvcache.mha_pool import MHAKVCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and real pinned memory")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_host_staging_round_trip_with_independent_transfer_and_install_streams(
    dtype, monkeypatch
):
    monkeypatch.setattr(tp_info, "_TP_INFO", tp_info.DistributedInfo(0, 1))
    device = torch.device("cuda:0")
    producer = MHAKVCache(2, 5, 8, 9, 4, dtype, device)
    receiver = MHAKVCache(2, 5, 8, 9, 4, dtype, device)
    slots = torch.tensor([28, 29, 30, 31, 8], device=device, dtype=torch.int64)
    target = torch.tensor([4, 5, 6, 7, 20], device=device, dtype=torch.int64)
    transfer, install = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    layer_bytes = 2 * len(slots) * 2 * 8 * dtype.itemsize
    scratch = torch.empty(2 * layer_bytes, dtype=torch.uint8, device=device)
    host = torch.empty(5 * layer_bytes, dtype=torch.uint8, pin_memory=True)
    with torch.cuda.stream(torch.cuda.current_stream(device)):
        producer._kv_buffer.copy_(
            torch.arange(producer._kv_buffer.numel(), device=device).reshape(
                producer._kv_buffer.shape
            )
        )
        receiver._kv_buffer.fill_(-1)
        ready = torch.cuda.Event()
        ready.record()
    with torch.cuda.stream(transfer):
        transfer.wait_event(ready)
        for start in range(0, 5, 2):
            end = min(start + 2, 5)
            nbytes = (end - start) * layer_bytes
            producer.export_layers_into(
                scratch[:nbytes], slots, start, end, num_pool_pages=8, trusted_indices=slots
            )
            host[start * layer_bytes : end * layer_bytes].copy_(scratch[:nbytes], non_blocking=True)
        copied = torch.cuda.Event()
        copied.record(transfer)
    copied.synchronize()  # Background sender's producer fence, without native RDMA in this test.
    with torch.cuda.stream(install):
        receiver.import_host_layered_tensor(host, target, workspace=scratch, layers_per_chunk=2)
        installed = torch.cuda.Event()
        installed.record(install)
    installed.synchronize()
    assert torch.equal(
        producer._kv_buffer.flatten(2, 3).index_select(2, slots).cpu(),
        receiver._kv_buffer.flatten(2, 3).index_select(2, target).cpu(),
    )
    assert torch.all(receiver._kv_buffer[:, :, 8] == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA PyTorch environment")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("group", [1, 2, 4])
def test_cuda_layer_group_pack_install(dtype, group, monkeypatch):
    monkeypatch.setattr(tp_info, "_TP_INFO", tp_info.DistributedInfo(0, 1))
    device = torch.device("cuda:0")
    compute = torch.cuda.Stream(device=device)
    transfer = torch.cuda.Stream(device=device)
    producer = MHAKVCache(2, 5, 8, 9, 4, dtype, device)
    receiver = MHAKVCache(2, 5, 8, 9, 4, dtype, device)
    slots = torch.tensor([28, 29, 30, 31, 8], device=device, dtype=torch.int64)
    target = torch.tensor([4, 5, 6, 7, 20], device=device, dtype=torch.int64)
    layer_bytes = 2 * len(slots) * 2 * 8 * dtype.itemsize
    staging = torch.empty(group * layer_bytes, dtype=torch.uint8, device=device)
    received = torch.empty(5 * layer_bytes, dtype=torch.uint8, device=device)
    compute.wait_stream(torch.cuda.current_stream(device))
    transfer.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(compute):
        producer._kv_buffer.copy_(
            torch.arange(producer._kv_buffer.numel(), device=device).reshape(
                producer._kv_buffer.shape
            )
        )
        ready = torch.cuda.Event()
        ready.record(compute)
    with torch.cuda.stream(transfer):
        transfer.wait_event(ready)
        for start in range(0, 5, group):
            end = min(start + group, 5)
            active = staging[: (end - start) * layer_bytes]
            producer.export_layers_into(active, slots, start, end, num_pool_pages=8)
            received[start * layer_bytes : end * layer_bytes].copy_(active)
        packed = torch.cuda.Event()
        packed.record(transfer)
    torch.cuda.current_stream(device).wait_event(packed)
    receiver._kv_buffer.fill_(-1)
    receiver.import_layered_tensor(received, target, num_pool_pages=8)
    assert torch.equal(
        producer._kv_buffer.flatten(2, 3).index_select(2, slots).cpu(),
        receiver._kv_buffer.flatten(2, 3).index_select(2, target).cpu(),
    )
    assert torch.all(receiver._kv_buffer[:, :, 8] == -1)
