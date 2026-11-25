"""Real pinned-host/CUDA stream ordering; native RDMA still needs separate acceptance."""

import time
from types import SimpleNamespace
from uuid import uuid4

import minisgl.distributed.info as tp_info
import pytest
import torch
from minisgl.disagg.control import StartRequest
from minisgl.disagg.dram_controller import DramController
from minisgl.disagg.dram_pool import DramPagePool
from minisgl.disagg.dram_protocol import DramConfig
from minisgl.disagg.prefix_hash import PrefixHashSpec, prepare_prefix
from minisgl.disagg.protocol import KVLayout
from minisgl.kvcache.mha_pool import MHAKVCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and pinned host memory")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("page_size", [1, 4])
def test_pinned_page_offload_and_restore_on_independent_cuda_streams(monkeypatch, dtype, page_size):
    monkeypatch.setattr(tp_info, "_TP_INFO", tp_info.DistributedInfo(0, 1))
    device = torch.device("cuda:0")
    layout = KVLayout("a" * 64, str(dtype).removeprefix("torch."), 2, 1, 8, page_size)
    from dataclasses import asdict

    spec = PrefixHashSpec.from_layout(asdict(layout))
    config = DramConfig(capacity_bytes=65536, staging_slots=2, batch_pages=1)
    pool = DramPagePool(page_size * layout.bytes_per_token, config.capacity_bytes)
    producer = MHAKVCache(1, 2, 8, 9, page_size, dtype, device)
    receiver = MHAKVCache(1, 2, 8, 9, page_size, dtype, device)
    table = torch.full((3, 8 * page_size), -1, dtype=torch.int32, device=device)
    source = torch.tensor(
        [6 * page_size + i for i in range(page_size)]
        + [2 * page_size + i for i in range(page_size)],
        device=device,
    )
    target = torch.tensor(
        [1 * page_size + i for i in range(page_size)]
        + [5 * page_size + i for i in range(page_size)],
        device=device,
    )
    table[0, : 2 * page_size] = source
    table[1, : 2 * page_size] = target
    compute = torch.cuda.current_stream(device)
    scheduler = SimpleNamespace(
        device=device,
        layout=layout,
        table_manager=SimpleNamespace(_max_running_reqs=2),
        engine=SimpleNamespace(stream=compute, kv_cache=producer, page_table=table),
    )
    controller = DramController(scheduler, pool, spec, config)
    request = StartRequest(str(uuid4()), 0, tuple(range(2 * page_size)), 2, True)
    pool.begin(request.transfer_id)
    context = SimpleNamespace(
        request=request,
        dram=controller.begin(request),
        req=SimpleNamespace(
            cached_len=2 * page_size,
            table_idx=0,
            input_ids=torch.tensor(request.prompt_ids, dtype=torch.int32),
        ),
    )
    address = pool.buffer.data_ptr()
    try:
        producer._kv_buffer.copy_(
            torch.arange(producer._kv_buffer.numel(), device=device).reshape(
                producer._kv_buffer.shape
            )
        )
        deadline = time.monotonic() + 5
        while context.dram.cursor < 2 * page_size and time.monotonic() < deadline:
            controller.offload(context)
            controller.progress()
            time.sleep(0.001)
        pool.finish(request.transfer_id, success=True)
        while context.dram.pending_writes and time.monotonic() < deadline:
            controller.progress()
            time.sleep(0.001)
        assert not context.dram.pending_writes and pool.health()["ready_pages"] == 2
        pages = pool.acquire(tuple(prepare_prefix(request.prompt_ids, spec).digests()))
        receiver._kv_buffer.fill_(-1)
        scheduler.engine.kv_cache = receiver
        task = controller.restore(1, 0, pages)
        while not task.done and time.monotonic() < deadline:
            controller.advance_restore(task)
            controller.progress()
            time.sleep(0.001)
        assert task.done and not task.error
        assert torch.equal(
            producer._kv_buffer.flatten(2, 3).index_select(2, source).cpu(),
            receiver._kv_buffer.flatten(2, 3).index_select(2, target).cpu(),
        )
        assert not pool.health()["readers"] and pool.buffer.data_ptr() == address
    finally:
        controller.offload_stream.synchronize()
        controller.restore_stream.synchronize()
        controller.progress()
        controller.close()
