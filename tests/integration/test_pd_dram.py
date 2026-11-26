"""Host-cache ownership and real CPU attention continuation; no network/model download."""

import ctypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from layered_reference import wait_for
from minisgl.disagg.control import StartRequest
from minisgl.disagg.dram_pool import DramPagePool
from minisgl.disagg.dram_protocol import DramConfig, RemotePullPlan
from minisgl.disagg.dram_transfer import DramTransfer, RemotePullJob
from minisgl.disagg.prefix_hash import PrefixHashSpec, prepare_prefix
from minisgl.disagg.routing_policy import RoutingConfig
from minisgl.disagg.service import Session
from minisgl.disagg.tier_index import TierIndex, TierPublisher
from minisgl.kvcache.radix_cache import RadixTreeNode
from minisgl.scheduler.cache import CacheManager
from pd_reference import reference_tokens
from test_pd_batching import cluster, drive, fake_cuda, receive  # noqa: F401
from test_pd_routing import running_cluster


@pytest.fixture(autouse=True)
def cpu_radix_compare(monkeypatch):
    def compare(node, ids):
        count = 0
        for a, b in zip(node._key.tolist(), ids.tolist()):
            if a != b:
                break
            count += 1
        return count

    monkeypatch.setattr(RadixTreeNode, "get_match_len", compare)


def identity(page=4):
    return PrefixHashSpec.from_layout(
        {
            "model_fingerprint": "a" * 64,
            "dtype": "float32",
            "layers": 2,
            "kv_heads": 1,
            "head_dim": 8,
            "page_size": page,
            "tp_rank": 0,
            "tp_size": 1,
            "packing": "kv-layer-token-head-dim-le",
            "quantization": "none",
        }
    )


def commit_page(pool, prepared, index=0, transaction=None):
    txn = transaction or str(uuid4())
    pool.begin(txn)
    hashes = tuple(prepared.digests())
    parent = identity().root if index == 0 else hashes[index - 1]
    record = pool.reserve(hashes[index], parent, index + 1, txn)
    pool.view(record).fill_(index + 1)
    pool.written(record)
    pool.finish(txn, success=True)
    return record


def test_commit_abort_and_expired_export_require_a_fence():
    now = [0.0]
    pool = DramPagePool(16, 512, pin_memory=False, clock=lambda: now[0], lease_ttl=2)
    prepared = prepare_prefix(tuple(range(12)), identity())
    record = commit_page(pool, prepared)
    lease = pool.pin_dram([record.digest], "peer")
    now[0] = 3
    assert pool.expire() == {"peer"}
    assert record.exports == 1
    assert not pool.unpin_dram(lease.id, "peer", drained=False)
    second = commit_page(pool, prepared, 1)
    assert pool._evict_one() and second.digest not in pool.pages
    assert record.digest in pool.pages
    pool.fence_complete({"peer"})
    assert record.exports == 0 and not pool.leases
    assert pool._evict_one()


def test_aborted_writer_cannot_publish_and_slot_generations_change():
    pool = DramPagePool(16, 256, pin_memory=False)
    prepared = prepare_prefix(tuple(range(8)), identity())
    hashes = tuple(prepared.digests())
    pool.begin("cancelled")
    record = pool.reserve(hashes[0], identity().root, 1, "cancelled")
    pool.finish("cancelled", success=False)
    assert not pool.free  # D2H still owns this slot.
    pool.written(record)
    assert pool.free and not pool.pages
    next_record = commit_page(pool, prepared, 1)
    assert next_record.slot == record.slot and next_record.generation > record.generation


def test_failed_copy_cannot_be_adopted_by_a_later_successful_transaction():
    pool = DramPagePool(16, 256, pin_memory=False)
    digest = next(prepare_prefix(tuple(range(4)), identity()).digests())
    for transaction in ("failed-copy", "replacement"):
        pool.begin(transaction)
    first = pool.reserve(digest, identity().root, 1, "failed-copy")
    first.readers += 1
    pool.written(first, success=False)
    assert pool.reserve(digest, identity().root, 1, "replacement") is None
    assert not first.transactions
    pool.release([first])
    replacement = pool.reserve(digest, identity().root, 1, "replacement")
    pool.written(replacement)
    pool.finish("failed-copy", success=True)
    assert not replacement.ready
    pool.finish("replacement", success=False)
    assert not pool.pages


def test_finalized_requests_do_not_exhaust_the_transaction_budget():
    pool = DramPagePool(16, 256, pin_memory=False, max_transactions=2)
    for i in range(20):
        prepared = prepare_prefix(tuple(range(i, i + 4)), identity())
        page = commit_page(pool, prepared, transaction=f"transaction-{i}")
        pool.end(f"transaction-{i}")
        assert page.ready
    assert not pool.transactions and len(pool.closed_transactions) == 2


@pytest.mark.parametrize("policy", ["lru", "lfu"])
def test_eviction_does_not_recycle_local_readers(policy):
    pool = DramPagePool(16, 512, pin_memory=False, policy=policy)
    prepared = prepare_prefix(tuple(range(12)), identity())
    first = commit_page(pool, prepared)
    second = commit_page(pool, prepared, 1)
    borrowed = pool.acquire([first.digest])
    commit_page(pool, prepared, 2)
    assert first.digest in pool.pages and second.digest not in pool.pages
    pool.release(borrowed)


@pytest.mark.parametrize("policy,victim", [("lru", 0), ("lfu", 1)])
def test_capacity_pressure_uses_the_configured_eviction_policy(policy, victim):
    now = [0.0]
    pool = DramPagePool(16, 3 * 256, pin_memory=False, policy=policy, clock=lambda: now[0])
    prepared = prepare_prefix(tuple(range(16)), identity())
    pages = [commit_page(pool, prepared)]
    # A was used often, but its last use precedes the creation of B and C.
    for _ in range(5):
        borrowed = pool.acquire([pages[0].digest])
        pool.release(borrowed)
    for i in range(1, 3):
        now[0] = float(i)
        pages.append(commit_page(pool, prepared, i))
    now[0] = 3.0
    replacement = commit_page(pool, prepared, 3)
    assert pages[victim].digest not in pool.pages
    assert all(page.digest in pool.pages for i, page in enumerate(pages) if i != victim)
    assert replacement.ready and pool.stats["evictions"] == 1
    assert not pool.free and not any(page.readers or page.exports for page in pool.pages.values())


def sync(index, node, publisher):
    index.observe(node.worker_id, node.health)
    for _ in range(20):
        state, request = index.request(node.worker_id)
        if not index.apply(node.worker_id, state, publisher.sync(request)):
            return
    raise AssertionError("directory did not catch up")


def test_tier_index_joins_local_prefix_with_one_remote_suffix():
    spec = identity()
    prepared = prepare_prefix(tuple(range(21)), spec)
    hashes = tuple(prepared.digests())
    nodes, publishers = [], []
    for role in ("prefill", "decode", "decode"):
        instance = str(uuid4())
        publisher = TierPublisher(spec, instance)
        node = SimpleNamespace(
            worker_id=f"{role}-{len(nodes)}",
            healthy=True,
            endpoint=f"127.0.0.1:{9000 + len(nodes)}",
            health={
                "role": role,
                "instance_id": instance,
                "layout": {
                    "model_fingerprint": "a" * 64,
                    "dtype": "float32",
                    "layers": 2,
                    "kv_heads": 1,
                    "head_dim": 8,
                    "page_size": 4,
                    "tp_rank": 0,
                    "tp_size": 1,
                    "packing": "kv-layer-token-head-dim-le",
                    "quantization": "none",
                },
                "dram_cache": {
                    "remote_ready": True,
                    "min_remote_tokens": 4,
                    "remote_max_bytes": 64 * 1024**2,
                },
            },
        )
        nodes.append(node)
        publishers.append(publisher)
    for i in range(2):
        publishers[0].update(
            "add", "DRAM", hashes[i], spec.root if not i else hashes[i - 1], i + 1, 1
        )
    # Neither donor holds the prompt ancestors. Two donors must not be stitched.
    publishers[1].update("add", "DRAM", hashes[2], hashes[1], 3, 1)
    publishers[2].update("add", "DRAM", hashes[3], hashes[2], 4, 1)
    index = TierIndex()
    for node, publisher in zip(nodes, publishers, strict=True):
        node.health["tier_index"] = publisher.health()
        sync(index, node, publisher)
    hit = index.hits(nodes[:1], nodes[1:], prepared, min_remote_tokens=4)[nodes[0].worker_id]
    assert (hit.hbm_tokens, hit.dram_tokens, hit.remote_tokens) == (0, 8, 4)
    assert hit.remote_plan.source_id == nodes[1].worker_id
    assert hit.total == 12
    for constraints in (
        {"min_remote_tokens": 8},
        {"min_remote_tokens": 4, "max_remote_bytes": 511},  # One KV page is 512 bytes.
        {"min_remote_tokens": 4, "prefill_ms_per_token": 1e-12},
    ):
        rejected = index.hits(nodes[:1], nodes[1:], prepared, **constraints)[nodes[0].worker_id]
        assert rejected.total == 8 and rejected.remote_plan is None
    nodes[1].healthy = False
    rejected = index.hits(nodes[:1], nodes[1:], prepared, min_remote_tokens=4)[nodes[0].worker_id]
    assert rejected.total == 8 and rejected.remote_plan is None
    nodes[1].healthy = True
    # Removing an ancestor on P must not remove a still-resident donor suffix.
    publishers[0].update("remove", "DRAM", hashes[0], spec.root, 1, 1)
    sync(index, nodes[0], publishers[0])
    assert ("DRAM", hashes[2]) in index.states[nodes[1].worker_id].entries
    assert (
        index.hits(nodes[:1], nodes[1:], prepared, min_remote_tokens=4)[nodes[0].worker_id].total
        == 0
    )


def enable(service, **kwargs):
    config = DramConfig(capacity_bytes=65536, **kwargs)
    service._init_dram({"dram": asdict(config), "batch_wait_ms": 0})


def run_request(p, d, prompt, tokens, *, plan=None, ready_out=None, wait_drain=True):
    req = StartRequest(str(uuid4()), 0, tuple(prompt), tokens, True)
    ds = Session()
    d.submit(ds, {"op": "reserve", "request": req.to_object(), "cache_transaction": True})
    until = time.monotonic() + 3
    messages = []
    while time.monotonic() < until and not messages:
        d.step()
        messages = receive(ds)
        time.sleep(0.001)
    assert messages and messages[0]["status"] == "ready", messages
    if ready_out is not None:
        ready_out.append(messages[0])
    ps = Session()
    message = {
        "op": "prefill",
        "request": req.to_object(),
        "ticket": messages[0]["ticket"],
        "decode_prefix_len": messages[0]["decode_prefix_len"],
        "cache_transaction": True,
    }
    if plan is not None:
        message["remote_pull_plan"] = plan.to_object()
    p.submit(ps, message)
    output = drive(p, [d], [(req, ds, messages[0]["ticket"])], [ps])[req.transfer_id]
    until = time.monotonic() + 3
    while wait_drain and (p.contexts or d.contexts) and time.monotonic() < until:
        p.step()
        d.step()
        time.sleep(0.001)
    if wait_drain:
        assert not p.contexts and not d.contexts
    return req, output


def test_decode_without_gpu_radix_retains_only_new_full_dram_pages():
    prompt = tuple(range(1, 11))
    with cluster(budget=32) as (p, decoders, _):
        d = decoders[0]
        enable(p)
        enable(d)
        request, output = run_request(p, d, prompt, 8)
        assert output == reference_tokens(prompt, 8)
        assert len(d.s.cache_manager.free_slots) == d.s.engine.num_pages
        assert d.dram_pool.health()["ready_pages"] == 0  # No premature publication.
        for service in (p, d):
            service._dram_rpc({"op": "cache_commit", "transfer_id": request.transfer_id})
        assert p.dram_pool.health()["ready_pages"] == 2
        assert d.dram_pool.health()["ready_pages"] == 2
        prepared = prepare_prefix(prompt + tuple(output[:-1]), d.tier_publisher.spec)
        hashes = tuple(prepared.digests())
        assert set(d.dram_pool.pages) == set(hashes[2:4])
        # The first D page is the prompt/generation seam, not an independent B fragment.
        assert d.dram_pool.pages[hashes[2]].parent == hashes[1]
        second, output2 = run_request(p, d, prompt + (20, 21), 3)
        assert output2 == reference_tokens(prompt + (20, 21), 3)
        assert p.dram_stats["local_hit_tokens"] >= 8
        for service in (p, d):
            service._dram_rpc({"op": "cache_abort", "transfer_id": second.transfer_id})


class HostTicket:
    def __init__(self, engine, source, target, lengths):
        self.engine, self.source, self.target, self.lengths = engine, source, target, lengths
        self.drained = False
        self.status = "unknown"

    def drain(self, timeout):
        if self.drained:
            return self.status
        if self.engine.gate is not None and not self.engine.gate.is_set():
            return "unknown"
        for source, target, length in zip(self.source, self.target, self.lengths, strict=True):
            ctypes.memmove(target, source, length)
        self.drained, self.status = True, "complete"
        return self.status


class HostEngine:
    registrations = {}
    instances = []

    def __init__(self):
        self.register_calls = 0
        self.fences = 0
        self.fence_result = 0
        self.gate = None
        self.submissions = []
        self.instances.append(self)

    def initialize(self, host, metadata, protocol, device):
        assert protocol == "rdma" and metadata == "P2PHANDSHAKE"
        self.port = int(os.environ.get("MC_MIN_RPC_PORT", 26000 + len(self.instances)))
        return 0

    def get_rpc_port(self):
        return self.port

    def initialize_with_rpc_port(self, host, metadata, protocol, device, port):
        self.initialize(host, metadata, protocol, device)
        self.port = port
        return 0

    def has_remote_reader_fence(self):
        return True

    def register_memory(self, address, size, location, remote):
        assert address not in self.registrations
        self.register_calls += 1
        self.registrations[address] = (size, remote)
        return 0

    def submit_scatter_read(
        self, session, local, sizes, remote, capacities, source_offsets, target_offsets, lengths
    ):
        assert sizes == capacities and source_offsets == target_offsets == [[0]] * len(local)
        for target, source, size in zip(local, remote, sizes, strict=True):
            assert any(
                a <= target and target + size <= a + n for a, (n, _) in self.registrations.items()
            )
            assert any(
                a <= source and source + size <= a + n and visible
                for a, (n, visible) in self.registrations.items()
            )
        ticket = HostTicket(self, remote, local, sizes)
        self.submissions.append(ticket)
        return ticket

    def fence_remote_readers(self):
        self.fences += 1
        return self.fence_result


@pytest.fixture
def host_rdma(monkeypatch):
    # Only the native transport is replaced. Pool leases, RPC validation,
    # transfer progress, GPU restore scheduling and CPU attention remain real.
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    module = SimpleNamespace(
        TransferEngine=HostEngine,
        ScatterTransferCompletionStatus=SimpleNamespace(
            COMPLETED="complete", FAILED_DRAINED="failed", COMPLETION_UNKNOWN="unknown"
        ),
    )
    yield module
    HostEngine.registrations.clear()
    HostEngine.instances.clear()


def attach_host(service, module, *, timeout=0.1):
    service.dram_config = DramConfig(
        capacity_bytes=65536,
        min_remote_tokens=4,
        batch_pages=1,
        remote_timeout=timeout,
        remote_enabled=True,
        port=26000 + len(HostEngine.instances),
    )
    service.dram_transfer = DramTransfer(service, service.dram_config, module=module)
    return service.dram_transfer


@pytest.mark.parametrize("inference", [False, True])
def test_reverse_promotion_and_dram_reserve_keep_original_suffix_boundary(
    host_rdma, monkeypatch, inference
):
    prompt = tuple(range(1, 11))
    monkeypatch.setenv("MC_MIN_RPC_PORT", "22501")
    monkeypatch.setenv("MC_MAX_RPC_PORT", "22501")
    with torch.inference_mode(inference), cluster(budget=32) as (p, decoders, _):
        d = decoders[0]
        d.decode_radix_cache = True
        d.s.cache_manager = CacheManager(32, 4, d.s.engine.page_table, "radix")
        enable(p, batch_pages=1)
        enable(d, batch_pages=1)
        receiver, donor = attach_host(p, host_rdma), attach_host(d, host_rdma)
        assert receiver.engine.get_rpc_port() == 26000 and donor.engine.get_rpc_port() == 26001
        for service in (p, d):
            health = service.health()["dram_cache"]
            assert health["remote_enabled"] and health["remote_ready"]
            assert health["min_remote_tokens"] == service.dram_config.min_remote_tokens
            assert health["remote_max_bytes"] == service.dram_config.remote_max_bytes
        assert os.environ["MC_MIN_RPC_PORT"] == os.environ["MC_MAX_RPC_PORT"] == "22501"
        receiver._rpc = lambda endpoint, message: donor.rpc(message)
        first, output = run_request(p, d, prompt, 8)
        for service in (p, d):
            service._dram_rpc({"op": "cache_commit", "transfer_id": first.transfer_id})
        assert (
            d.s.cache_manager.prefix_cache.peek_prefix_len(torch.tensor(prompt + tuple(output)))
            == 16
        )
        d.s.cache_manager._free(d.s.cache_manager.prefix_cache.evict(8))
        followup = prompt + tuple(output) + (22, 23)
        prepared = prepare_prefix(followup, p.tier_publisher.spec)
        hashes = tuple(prepared.digests())
        plan = RemotePullPlan(
            "d-0", d.instance_id, "donor:9000", prepared.domain_id, 2, hashes[2:4]
        )
        ready = []
        second, output2 = run_request(p, d, followup, 3, plan=plan, ready_out=ready)
        assert output2 == reference_tokens(followup, 3)
        assert ready[0]["decode_prefix_len"] == 16  # P -> D still sends only C.
        assert d.stats["received_kv_tokens"] == len(prompt) + len(followup) - 16
        assert p.dram_stats["remote_hit_tokens"] == 8
        assert d.dram_stats["local_hit_tokens"] == 8
        for service in (p, d):
            service._dram_rpc({"op": "cache_commit", "transfer_id": second.transfer_id})
        assert all(h in p.dram_pool.pages and p.dram_pool.pages[h].ready for h in hashes[2:4])
        assert not donor.pool.leases and all(
            page.exports == 0 for page in donor.pool.pages.values()
        )
        assert receiver.engine.register_calls == donor.engine.register_calls == 1
        assert len(receiver.engine.submissions) == 2  # Page reads pipeline with H2D.


def test_late_read_completion_quarantines_destination_and_releases_source(host_rdma):
    with cluster() as (p, decoders, _):
        d = decoders[0]
        enable(p)
        enable(d)
        receiver, donor = attach_host(p, host_rdma, timeout=0.01), attach_host(d, host_rdma)
        receiver._rpc = lambda endpoint, message: donor.rpc(message)
        prepared = prepare_prefix(tuple(range(8)), p.tier_publisher.spec)
        record = commit_page(d.dram_pool, prepared)
        receiver.pool.begin("pull")
        plan = RemotePullPlan(
            "d-0", d.instance_id, "donor:9000", prepared.domain_id, 0, (record.digest,)
        )
        gate = receiver.engine.gate = threading.Event()
        job = RemotePullJob()
        assert not receiver.pull("pull", plan, 0, (record.digest,), job)
        assert job.take() is None
        assert receiver.pool.health()["quarantined_pages"] == 1
        assert record.exports == 1
        receiver.pool.finish("pull", success=False)
        receiver.progress()
        assert len(receiver.operations) == 1  # A timeout cannot free DMA destinations.
        gate.set()
        receiver.progress()
        until = time.monotonic() + 1
        while donor.pool.leases and time.monotonic() < until:
            time.sleep(0.001)
        assert not receiver.operations and not receiver.pool.pages
        assert not receiver.pool.quarantined_slots and not donor.pool.leases


def test_chunked_prefill_offloads_aligned_pages_when_copy_ring_is_smaller_than_a_chunk():
    prompt = tuple(range(1, 30))
    with cluster(budget=7) as (p, decoders, _):
        d = decoders[0]
        enable(p, batch_pages=1, staging_slots=1)
        enable(d, batch_pages=1, staging_slots=1)
        request, output = run_request(p, d, prompt, 4)
        assert output == reference_tokens(prompt, 4)
        p._dram_rpc({"op": "cache_commit", "transfer_id": request.transfer_id})
        prepared = prepare_prefix(prompt, p.tier_publisher.spec)
        assert set(p.dram_pool.pages) == set(prepared.digests())
        assert p.dram_pool.health()["ready_pages"] == len(prompt) // 4


def test_hash_queue_pressure_falls_back_without_failing_the_request():
    with cluster() as (p, decoders, _):
        d = decoders[0]
        enable(p)
        enable(d)
        for _ in range(p.dram_controller.hash_capacity):
            assert p.dram_controller.hash_slots.acquire(blocking=False)
        try:
            request, output = run_request(p, d, tuple(range(1, 11)), 5)
            assert output == reference_tokens(request.prompt_ids, 5)
            assert p.dram_controller.stats["hash_skips"] == 1
            assert not p.dram_pool.pages
        finally:
            for _ in range(p.dram_controller.hash_capacity):
                p.dram_controller.hash_slots.release()


def test_finished_decode_waits_for_offload_without_computing_extra_tokens():
    gate = threading.Event()
    with cluster(budget=32) as (p, decoders, _):
        d = decoders[0]
        enable(p)
        enable(d)
        for i in range(d.dram_config.staging_slots):
            d.dram_controller.events["write"][i] = SimpleNamespace(query=gate.is_set)
        try:
            request, output = run_request(p, d, tuple(range(1, 11)), 8, wait_drain=False)
            assert output == reference_tokens(request.prompt_ids, 8)
            context = next(iter(d.contexts.values()))
            assert context.phase == "draining" and context.dram.pending_writes
            cached_len = context.req.cached_len
            assert not d.s.decode_manager.running_reqs
            assert len(d.s.cache_manager.free_slots) < d.s.engine.num_pages
            for _ in range(5):
                d.step()
            assert context.req.cached_len == cached_len
        finally:
            gate.set()
        for _ in range(10):
            d.step()
        assert not d.contexts and len(d.s.cache_manager.free_slots) == d.s.engine.num_pages


@pytest.mark.parametrize("failure", ["pin_miss", "timeout", "read_busy"])
def test_remote_failure_recomputes_and_completes_the_request(host_rdma, failure):
    with cluster(budget=32) as (p, decoders, _):
        d = decoders[0]
        enable(p, batch_pages=1)
        enable(d, batch_pages=1)
        receiver, donor = attach_host(p, host_rdma, timeout=0.01), attach_host(d, host_rdma)
        receiver._rpc = lambda endpoint, message: donor.rpc(message)
        first, output = run_request(p, d, tuple(range(1, 11)), 8)
        for service in (p, d):
            service._dram_rpc({"op": "cache_commit", "transfer_id": first.transfer_id})
        followup = first.prompt_ids + tuple(output) + (22, 23)
        prepared = prepare_prefix(followup, p.tier_publisher.spec)
        hashes = tuple(prepared.digests())
        plan = RemotePullPlan(
            "d-0", d.instance_id, "donor:9000", prepared.domain_id, 2, hashes[2:4]
        )
        gate = receiver.engine.gate = threading.Event()
        if failure == "pin_miss":
            receiver._rpc = lambda endpoint, message: {"status": "cache_miss"}
        if failure == "read_busy":
            assert receiver.read_slots.acquire(blocking=False)
            assert receiver.read_slots.acquire(blocking=False)
        try:
            _, second_output = run_request(p, d, followup, 3, plan=plan)
            assert second_output == reference_tokens(followup, 3)
            assert p.dram_stats["pull_fallbacks"] > 0 and not p.dram_stats["remote_hit_tokens"]
            assert len(p.s.cache_manager.free_slots) == p.s.engine.num_pages
            if failure == "timeout":
                assert len(receiver.operations) == 1 and donor.pool.leases
        finally:
            gate.set()
            receiver.progress()
            if failure == "read_busy":
                receiver.read_slots.release()
                receiver.read_slots.release()


def test_partial_remote_timeout_keeps_loaded_prefix_and_recomputes_the_remainder(
    host_rdma, monkeypatch
):
    with cluster(budget=32) as (p, decoders, _):
        d = decoders[0]
        enable(p, batch_pages=1)
        enable(d, batch_pages=1)
        receiver, donor = attach_host(p, host_rdma, timeout=0.01), attach_host(d, host_rdma)
        receiver._rpc = lambda endpoint, message: donor.rpc(message)
        first, output = run_request(p, d, tuple(range(1, 11)), 8)
        for service in (p, d):
            service._dram_rpc({"op": "cache_commit", "transfer_id": first.transfer_id})
        followup = first.prompt_ids + tuple(output) + (22, 23)
        prepared = prepare_prefix(followup, p.tier_publisher.spec)
        hashes = tuple(prepared.digests())
        plan = RemotePullPlan(
            "d-0", d.instance_id, "donor:9000", prepared.domain_id, 2, hashes[2:4]
        )
        submit = receiver.engine.submit_scatter_read
        gate = threading.Event()

        def block_second_page(*args):
            ticket = submit(*args)
            if len(receiver.engine.submissions) == 2:
                receiver.engine.gate = gate
            return ticket

        monkeypatch.setattr(receiver.engine, "submit_scatter_read", block_second_page)
        try:
            second, output2 = run_request(p, d, followup, 3, plan=plan)
            assert output2 == reference_tokens(followup, 3)
            assert p.dram_stats["remote_hit_tokens"] == 4
            assert p.dram_stats["pull_fallbacks"] > 0
            assert len(p.s.cache_manager.free_slots) == p.s.engine.num_pages
            assert len(receiver.operations) == 1 and receiver.pool.quarantined_slots
            for service in (p, d):
                service._dram_rpc({"op": "cache_commit", "transfer_id": second.transfer_id})
            assert p.dram_pool.pages[hashes[2]].ready
            assert not p.dram_pool.pages[hashes[3]].ready  # Late DMA cannot publish a page.
        finally:
            gate.set()
            receiver.progress()
        wait_for(lambda: not donor.pool.leases)
        assert not receiver.operations and not receiver.pool.quarantined_slots


@pytest.mark.parametrize("result", [0, -1])
def test_export_expiry_fences_only_the_host_engine_and_never_recycles_without_proof(
    host_rdma, result
):
    with cluster() as (p, decoders, _):
        d = decoders[0]
        enable(d)
        donor = attach_host(d, host_rdma)
        prepared = prepare_prefix(tuple(range(8)), d.tier_publisher.spec)
        record = commit_page(d.dram_pool, prepared)
        now = [0.0]
        donor.pool.clock = lambda: now[0]
        lease = donor.pool.pin_dram([record.digest], "abandoned-reader", ttl=1)
        donor.engine.fence_result = result
        now[0] = 2
        peers = donor.pool.expire()
        assert peers == {"abandoned-reader"}
        donor.request_fence(peers)
        until = time.monotonic() + 1
        while donor.fence_future is not None and time.monotonic() < until:
            donor.progress()
            time.sleep(0.001)
        donor.request_fence(peers)
        assert donor.engine.fences == 1 and not donor.ready
        assert d.health()["status"] == "ready"  # Essential GPU P/D TE is untouched.
        if result == 0:
            assert not donor.pool.leases and record.exports == 0
        else:
            assert donor.pool.leases and record.exports == 1
            assert not donor.pool._evict_one()
            # The controlled fake has no in-flight read; a late receipt is safe.
            donor.pool.unpin_dram(lease.id, lease.peer, drained=True)


def test_restart_and_broken_delta_clear_both_tiers():
    spec = identity()
    publisher = TierPublisher(spec, str(uuid4()))
    node = SimpleNamespace(
        worker_id="p-0",
        healthy=True,
        health={
            "instance_id": publisher.instance_id,
            "layout": {
                "model_fingerprint": "a" * 64,
                "dtype": "float32",
                "layers": 2,
                "kv_heads": 1,
                "head_dim": 8,
                "page_size": 4,
                "tp_rank": 0,
                "tp_size": 1,
                "packing": "kv-layer-token-head-dim-le",
                "quantization": "none",
            },
            "tier_index": publisher.health(),
        },
    )
    hashes = tuple(prepare_prefix(tuple(range(8)), spec).digests())
    for tier in ("HBM", "DRAM"):
        publisher.update("add", tier, hashes[0], spec.root, 1, 1)
    index = TierIndex()
    sync(index, node, publisher)
    assert len(index.states[node.worker_id].entries) == 2
    state, request = index.request(node.worker_id)
    reply = publisher.sync(request)
    reply["seq"] += 1
    index.apply(node.worker_id, state, reply)
    assert not index.states[node.worker_id].entries
    sync(index, node, publisher)
    restarted = TierPublisher(spec, str(uuid4()))
    node.health.update(instance_id=restarted.instance_id, tier_index=restarted.health())
    index.observe(node.worker_id, node.health)
    assert not index.states[node.worker_id].entries
    assert index.states[node.worker_id].instance_id == restarted.instance_id


def test_remote_plan_requires_a_strict_protocol_and_single_source():
    prepared = prepare_prefix(tuple(range(8)), identity())
    plan = RemotePullPlan(
        "d-0", str(uuid4()), "donor:9000", prepared.domain_id, 0, tuple(prepared.digests())
    )
    assert RemotePullPlan.from_object(plan.to_object()) == plan
    for change in (
        {"version": True},
        {"source_instance": "old-worker"},
        {"start_page": -1},
        {"page_hashes": []},
        {"sources": ["d-0", "d-1"]},
    ):
        with pytest.raises((ValueError, TypeError)):
            RemotePullPlan.from_object({**plan.to_object(), **change})


def test_routed_client_commits_dram_and_uses_tier_index_without_p_gpu_radix(monkeypatch):
    config = RoutingConfig(policy="cache_aware", cache_min_tokens=1, prefix_sync_interval=0.01)
    with running_cluster(
        monkeypatch, routing_config=config, dram_config=DramConfig(capacity_bytes=65536)
    ) as (client, services, counts, frames):
        prompt = tuple(range(1, 11))
        output = client.generate(prompt, 8, ignore_eos=True)
        assert output == reference_tokens(prompt, 8)
        wait_for(lambda: sum(s.dram_pool.health()["ready_pages"] for s in services[:2]) == 2)
        wait_for(lambda: sum(s.dram_pool.health()["ready_pages"] for s in services[2:]) == 2)
        owner = next(s for s in services[:2] if s.dram_pool.health()["ready_pages"])
        assert owner.prefix_publisher is None  # A DRAM-only P can still provide the hash domain.
        client.pool.probe_all()
        prepared = prepare_prefix(prompt + tuple(output), owner.tier_publisher.spec)
        wait_for(
            lambda: any(
                hit.total >= 8
                for hit in client.pool.tier_index.hits(
                    client.pool.healthy("prefill"), client.pool.healthy("decode"), prepared
                ).values()
            )
        )
        assert client.generate(prepared.token_ids, 3, ignore_eos=True) == reference_tokens(
            prepared.token_ids, 3
        )
        assert owner.dram_stats["local_hit_tokens"] >= 8
        assert set(counts.values()) == {1}  # Finalization reuses multiplexed control connections.
        assert any(frame.get("message", {}).get("op") == "cache_commit" for frame in frames)
        prompts = [prompt + (i,) for i in range(20, 24)]
        expected = [reference_tokens(ids, 4) for ids in prompts]
        with ThreadPoolExecutor(max_workers=4) as executor:
            assert (
                list(executor.map(lambda ids: client.generate(ids, 4, ignore_eos=True), prompts))
                == expected
            )
        wait_for(lambda: all(not s.contexts for s in services))
        assert owner.dram_stats["local_hit_tokens"] >= 40
        iterator = client.stream((30, 29, 28, 27, 26, 25, 24, 23, 22), 8, ignore_eos=True)
        next(iterator)
        iterator.close()
        wait_for(lambda: any(s.dram_pool.stats["aborted"] for s in services))
        assert any(frame.get("message", {}).get("op") == "cache_abort" for frame in frames)

        def broken_index(*args, **kwargs):
            raise ValueError("stale metadata")

        monkeypatch.setattr(client.pool.tier_index, "hits", broken_index)
        assert client.generate(prompt, 3, ignore_eos=True) == reference_tokens(prompt, 3)
