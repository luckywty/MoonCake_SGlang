from unittest.mock import patch

import minisgl.core as core
import minisgl.distributed.info as tp_info
import pytest
import torch
from minisgl.core import Req, SamplingParams
from minisgl.distributed import set_tp_info
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.scheduler.cache import CacheManager


@pytest.fixture(autouse=True)
def reset_tp(monkeypatch):
    monkeypatch.setattr(tp_info, "_TP_INFO", None)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [1, 4])
def test_real_pool_round_trip_noncontiguous_pages(dtype, page_size):
    set_tp_info(rank=0, size=1)
    p = MHAKVCache(2, 3, 4, 9, page_size, dtype, torch.device("cpu"))
    d = MHAKVCache(2, 3, 4, 9, page_size, dtype, torch.device("cpu"))
    p._kv_buffer.copy_(torch.arange(p._kv_buffer.numel()).view(p._kv_buffer.shape).to(dtype))
    d._kv_buffer.fill_(-1)
    tokens = page_size + 1
    source = torch.tensor([7 * page_size + i for i in range(page_size)] + [3 * page_size])
    target = torch.tensor([2 * page_size + i for i in range(page_size)] + [6 * page_size])
    payload = p.export_tokens(source, num_pool_pages=8)
    assert len(payload) == 2 * 3 * tokens * 2 * 4 * dtype.itemsize
    d.import_tokens(payload, target, num_pool_pages=8)
    assert torch.equal(
        p._kv_buffer.flatten(2, 3).index_select(2, source),
        d._kv_buffer.flatten(2, 3).index_select(2, target),
    )
    assert torch.all(d._kv_buffer[:, :, 8] == -1)  # Graph dummy page never imported.
    assert torch.all(d._kv_buffer[:, :, 0] == -1)


def test_import_rejects_invalid_payload_before_writing():
    set_tp_info(rank=0, size=1)
    pool = MHAKVCache(1, 1, 2, 5, 4, torch.float32, torch.device("cpu"))
    pool._kv_buffer.fill_(-1)
    for slots in (
        torch.tensor([16]),
        torch.tensor([0, 0]),
        torch.tensor([-1]),
        torch.tensor([1.0]),
    ):
        with pytest.raises(ValueError):
            pool.export_tokens(slots, num_pool_pages=4)
    with pytest.raises(ValueError):
        pool.import_tokens(b"bad-length", torch.tensor([0, 1]), num_pool_pages=4)
    assert torch.all(pool._kv_buffer == -1)


@pytest.fixture
def manager():
    old = core._GLOBAL_CTX
    core._GLOBAL_CTX = core.Context(page_size=4)
    table = torch.full((3, 64), -1, dtype=torch.int32)
    cm = CacheManager(8, 4, table, "naive")
    yield cm
    core._GLOBAL_CTX = old


def test_import_capacity_and_discard_partial_forward(manager):
    cm = manager
    slots = cm.reserve_import(0, 5)
    assert len(slots) == 8
    assert torch.equal(cm.page_table[0, :8], slots)
    assert len(cm.free_slots) == 6
    handle = cm.prefix_cache.match_prefix(torch.empty(0, dtype=torch.int32)).cuda_handle
    req = Req(torch.tensor([1, 2, 3, 4, 5, 6]), 0, 5, 4, 0, SamplingParams(), handle)
    cm.discard_private(req, allocated_len=5)
    cm.check_integrity()
    assert len(set(cm.free_slots.tolist())) == cm.num_pages
    with pytest.raises(ValueError):
        cm.reserve_import(2, 5)  # Dummy table row.
    with pytest.raises(RuntimeError):
        cm.reserve_import(1, 33)


def test_allocator_rolls_back_failed_table_write(manager):
    cm = manager
    handle = cm.prefix_cache.match_prefix(torch.empty(0, dtype=torch.int32)).cuda_handle
    req = Req(torch.tensor([1, 2, 3, 4, 5]), 0, 0, 4, 0, SamplingParams(), handle)
    with patch("minisgl.scheduler.cache._write_page_table", side_effect=RuntimeError("injected")):
        with pytest.raises(RuntimeError, match="injected"):
            cm.allocate_paged([req])
    cm.check_integrity()
    assert len(set(cm.free_slots.tolist())) == cm.num_pages


def test_discard_keeps_borrowed_radix_prefix_pages():
    old = core._GLOBAL_CTX
    core._GLOBAL_CTX = core.Context(page_size=4)
    try:
        table = torch.full((3, 32), -1, dtype=torch.int32)
        cm = CacheManager(8, 4, table, "radix")
        prefix_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        prefix_slots = torch.arange(4, dtype=torch.int32)
        cm.free_slots = cm.free_slots[1:]
        handle = cm.prefix_cache.insert_prefix(prefix_ids, prefix_slots).handle
        cm.lock(handle)
        table[0, :4] = prefix_slots
        table[0, 4:8] = cm._page_to_token(cm._allocate(1))
        req = Req(torch.arange(8), 0, 4, 4, 0, SamplingParams(), handle)
        cm.discard_private(req, allocated_len=8)
        cm.check_integrity()
        assert len(cm.free_slots) == 7
        assert 0 not in cm.free_slots.tolist()
        assert cm.prefix_cache.size_info.protected_size == 0
        assert cm.prefix_cache.size_info.evictable_size == 4
    finally:
        core._GLOBAL_CTX = old
