"""Actual SHM/socket/pump + real CPU KV/scatter; CUDA fences are explicitly simulated."""

import json
import multiprocessing
import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from minisgl.disagg import client as direct_client
from minisgl.disagg.cluster_config import host_staging_plan
from minisgl.disagg.control import StartRequest
from minisgl.disagg.layered_protocol import LAYERED_TICKET_VERSION, LayeredReceiveTicket
from minisgl.disagg.layered_sender import BatchTransferEntry, BatchTransferPlan
from minisgl.disagg.layered_shm import TRANSPORT, LayeredShmBackend
from minisgl.disagg.protocol import HANDOFF_VERSION, KVLayout, KVTransferMetadata
from minisgl.disagg.router import NoHealthyWorker, WorkerPool
from minisgl.disagg.routing_policy import PolicyRouter, RoutingBucket, RoutingConfig
from minisgl.disagg.service import WorkerService
from minisgl.disagg.shm_relay_pool import FREE, READY, WRITING, SharedRelayPool, build_manifest
from minisgl.disagg.transfer import TransferBackendUnavailable
from minisgl.distributed.info import set_tp_info, try_get_tp_info
from minisgl.kvcache.mha_pool import MHAKVCache
from pd_cache_reference import (
    cached_round_trip,
    cpu_radix_compare,  # noqa: F401
    start_cached_request,
)
from pd_reference import cpu_pinned_allocations, cpu_scheduler, reference_tokens
from test_pd_batching import drive, reserve, submit


class Stream:
    def __init__(self, kind="compute"):
        self.kind = kind

    def wait_event(self, event):
        pass

    def wait_stream(self, stream):
        pass


class Event:
    d2h = threading.Event()
    h2d = threading.Event()

    def record(self, stream=None):
        self.kind = getattr(stream, "kind", "compute")

    def synchronize(self):
        if self.kind == "d2h":
            assert threading.current_thread().name == "pd-shm-send"
            assert self.d2h.wait(5)

    def query(self):
        return self.h2d.is_set() if self.kind == "h2d" else True


@pytest.fixture(autouse=True)
def fake_cuda(monkeypatch):
    Event.d2h.set()
    Event.h2d.set()
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    with cpu_pinned_allocations():
        yield
    Event.d2h.set()
    Event.h2d.set()


def make_manifest(layout, *, nodes=(0, 1), group=1):
    plan = host_staging_plan(
        max_seq_len=64,
        bytes_per_token=layout.bytes_per_token,
        layers=layout.layers,
        layers_per_chunk=group,
        staging_slots=3,
        max_transfer_bytes=64 * layout.bytes_per_token,
        max_registered_bytes=65536,
        rows=8,
        prefill=True,
    )
    return build_manifest(
        f"minisgl_test_{uuid4().hex}",
        [("p-0", 0)],
        plan,
        bytes_per_token=layout.bytes_per_token,
        layers=layout.layers,
        layers_per_chunk=group,
        local_slots=2,
        overflow_slots=1,
        numa_nodes=nodes,
    )


@contextmanager
def backend_pair(p, d, *, nodes=(0, 1), d_numa=0, group=1, inference=True):
    manifest = make_manifest(p.layout, nodes=nodes, group=group)
    owner = SharedRelayPool(manifest, create=True, initialize=False)
    backends = []
    try:
        for scheduler, worker_id, node, bind in [
            (p, "p-0", 0, None),
            (d, "d-0", d_numa, ("127.0.0.1", 0)),
        ]:
            backend = LayeredShmBackend(
                bind,
                manifest=manifest,
                worker_id=worker_id,
                numa_id=node,
                host_id=manifest.host_id,
                layers_per_chunk=group,
                staging_slots=2,
                overflow_slots=1,
                timeout=2,
                max_transfer_bytes=64 * p.layout.bytes_per_token,
                max_registered_bytes=65536,
            )
            backends.append(backend)
            with torch.inference_mode() if inference else nullcontext():
                backend.configure_pipeline(scheduler)
        yield owner, *backends
    finally:
        Event.d2h.set()
        Event.h2d.set()
        for backend in backends:
            for entry in backend._receives.values():
                entry.cancelled = True
                entry.queued.clear()
                backend._poll_installs(entry)
        for backend in backends:
            backend.close()
        owner.close()


@pytest.mark.parametrize("nodes,d_numa,group", [((0,), 0, 1), ((0, 1), 0, 1), ((0, 1), 1, 2)])
def test_real_service_prefill_to_decode_and_slot_reuse_match_reference(nodes, d_numa, group):
    cases = [((1, 2, 3), 5), ((4, 5, 6, 7), 4), ((8, 9), 6)]
    expected = [reference_tokens(prompt, count) for prompt, count in cases]
    p = cpu_scheduler("prefill", max_running_req=8, prefill_budget=5)
    d = cpu_scheduler("decode", max_running_req=8)
    p.pd_service_config = d.pd_service_config = {"batch_wait_ms": 0}
    with backend_pair(p, d, nodes=nodes, d_numa=d_numa, group=group) as (owner, pb, db):
        ps, ds = WorkerService(p, pb, 2), WorkerService(d, db, 2)
        try:
            items = [reserve(ds, prompt, count) for prompt, count in cases]
            assert items[0][2]["transfer_mode"] == "shm_dram"
            outputs = drive(ps, [ds], items, [submit(ps, item) for item in items], deadline=5)
            assert [outputs[item[0].transfer_id] for item in items] == expected
            assert not pb.allocator.leases and not db._receives
            assert not ps.contexts and not ds.contexts
            assert ps.health()["host_id"] == ds.health()["host_id"] == owner.manifest.host_id
            assert ds.health()["numa_id"] == d_numa
            assert ps.s.table_manager.available_size == ds.s.table_manager.available_size == 8
        finally:
            ps.close()
            ds.close()


@pytest.mark.usefixtures("cpu_radix_compare")
def test_real_d_prefix_suffix_and_full_hit_skip_shared_writes():
    p = cpu_scheduler("prefill", page_size=1, max_running_req=8, prefill_budget=32)
    d = cpu_scheduler("decode", page_size=1, max_running_req=8, cache_type="radix")
    p.pd_service_config = {"batch_wait_ms": 0}
    d.pd_service_config = {"batch_wait_ms": 0, "decode_radix_cache": True}
    with backend_pair(p, d) as (_, pb, db):
        ps, ds = WorkerService(p, pb, 2), WorkerService(d, db, 2)
        try:
            prompt = (1, 2, 3, 4, 5)
            first = cached_round_trip(ps, ds, prompt, 4)
            received = ds.stats["received_kv_tokens"]
            headers = {
                node: [db.shared.header(node, i) for i in range(count)]
                for node, count in db.manifest.node_slots.items()
            }
            full = cached_round_trip(ps, ds, prompt, 3)
            assert full.ready["ticket"] is None
            assert full.handoff["kv"]["prefix_len"] == len(prompt)
            assert full.handoff["kv"]["kv_tokens"] == 0
            assert full.output == reference_tokens(prompt, 3)
            assert ds.stats["received_kv_tokens"] == received
            assert headers == {
                node: [db.shared.header(node, i) for i in range(count)]
                for node, count in db.manifest.node_slots.items()
            }
            longer = prompt + ((first.output[0] + 1) % 32, 21)
            tail = start_cached_request(ps, ds, longer, 3)
            suffix = LayeredReceiveTicket.from_wire(json.dumps(tail.ready["ticket"]).encode())
            assert suffix.prefix_len == len(prompt) and suffix.kv_tokens == 2
            output = drive(ps, [ds], [tail.item], [tail.ps])
            assert output[tail.request.transfer_id] == reference_tokens(longer, 3)
            assert ds.stats["received_kv_tokens"] == received + 2
        finally:
            ps.close()
            ds.close()


def direct_scheduler(layers=5):
    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    pool = MHAKVCache(1, layers, 2, 9, 1, torch.float32, torch.device("cpu"))
    pool._kv_buffer.zero_()
    return SimpleNamespace(
        device=torch.device("cpu"),
        layout=KVLayout("a" * 64, "float32", layers, 1, 2, 1),
        engine=SimpleNamespace(kv_cache=pool, num_pages=8, stream=Stream()),
    )


def start_direct(pb, db, p):
    request = StartRequest(str(uuid4()), 0, (1, 2, 3), 3, True)
    ticket = db.reserve_layers(request, eos_token_id=31)
    job = pb.begin_layer_send(ticket, request, p.engine.kv_cache, wait=False)
    slots = torch.tensor([1, 3, 6], dtype=torch.int32)
    plan = BatchTransferPlan(pb, [BatchTransferEntry(request.transfer_id, slots, True, job)])
    for layer in range(p.layout.layers):
        plan.layer_ready(layer)
    job.seal()
    return ticket, job, slots


def until(predicate, *, step=lambda: None, seconds=3):
    deadline = time.monotonic() + seconds
    while not predicate():
        step()
        assert time.monotonic() < deadline, "shared relay did not make progress"
        time.sleep(0.001)


def test_more_groups_than_ring_slots_install_before_commit_without_deadlock():
    p, d = direct_scheduler(), direct_scheduler()
    p.engine.kv_cache._kv_buffer.copy_(
        torch.arange(p.engine.kv_cache._kv_buffer.numel()).reshape_as(p.engine.kv_cache._kv_buffer)
    )
    with backend_pair(p, d, d_numa=1) as (owner, pb, db):
        ticket, job, source = start_direct(pb, db, p)
        destination = torch.tensor([5, 2, 1], dtype=torch.int32)
        # Five groups cycle through two local + one overflow slot before commit.
        until(job.done.is_set, step=lambda: db.progress_receive(ticket, destination))
        assert job.error is None and job.completed == set(range(5))
        assert db.finish_layers(ticket) is not None
        for layer in range(5):
            for getter in ("k_cache", "v_cache"):
                actual = getattr(d.engine.kv_cache, getter)(layer).flatten(0, 1)[destination.long()]
                expected = getattr(p.engine.kv_cache, getter)(layer).flatten(0, 1)[source.long()]
                assert torch.equal(actual, expected)
        assert not pb.allocator.leases
        assert db.release_layers(ticket, consumer_complete=True)


def test_double_barrier_cancel_retains_slot_and_gpu_pages_until_h2d_completion():
    p, d = direct_scheduler(), direct_scheduler()
    with backend_pair(p, d) as (owner, pb, db):
        pb.transfer_stream, db.install_stream = Stream("d2h"), Stream("h2d")
        Event.d2h.clear()
        Event.h2d.clear()
        try:
            ticket, job, _ = start_direct(pb, db, p)
            until(lambda: bool(pb.allocator.leases))
            lease = next(iter(pb.allocator.leases.values()))
            assert owner.header(lease.slot_numa_id, lease.slot_id) == (lease.generation, WRITING)
            assert not db._lookup_layer(ticket).queued  # No notification before the copied fence.
            Event.d2h.set()
            until(lambda: bool(db._lookup_layer(ticket).queued))
            slots = torch.tensor([5, 2, 1], dtype=torch.int32)
            assert not db.progress_receive(ticket, slots)
            assert owner.header(lease.slot_numa_id, lease.slot_id)[1] == READY
            job.seal(cancel=True)
            assert not db.progress_receive(ticket, slots, cancel=True)
            assert not db.release_layers(ticket)
            assert pb.allocator.leases and not job.done.is_set()
            Event.h2d.set()
            until(job.done.is_set, step=lambda: db.progress_receive(ticket, slots, cancel=True))
            assert job.safe_sources and job.sources_done.is_set()
            assert not pb.allocator.leases
            assert db.release_layers(ticket)
            assert owner.header(lease.slot_numa_id, lease.slot_id)[1] == FREE
        finally:
            Event.d2h.set()
            Event.h2d.set()


def test_stale_generation_owner_and_cross_host_notifications_are_rejected():
    p, d = direct_scheduler(), direct_scheduler()
    with backend_pair(p, d) as (owner, pb, db):
        request = StartRequest(str(uuid4()), 0, (1, 2), 2, True)
        ticket = db.reserve_layers(request, eos_token_id=31)
        lease = pb.allocator.acquire(request.transfer_id, 0)
        pb.allocator.ready(lease, copied=True)
        message = {
            "backend": TRANSPORT,
            "op": "shm_notify",
            "ticket": json.loads(ticket.to_wire()),
            "pool": owner.manifest.name,
            "host_id": owner.manifest.host_id,
            "owner": "p-0",
            "group": 0,
            "descriptor": {
                **asdict(lease),
                "layer_start": 0,
                "layer_end": 1,
                "nbytes": ticket.layer_bytes,
            },
        }
        for bad in (
            {**message, "host_id": "remote"},
            {
                **message,
                "descriptor": {**message["descriptor"], "generation": lease.generation + 1},
            },
            {**message, "owner": "p-9"},
        ):
            with pytest.raises(RuntimeError):
                db._dispatch(bad)
        assert db._dispatch(message) == {"status": "accepted"}
        assert db._dispatch(message) == {"status": "accepted"}
        assert len(db._lookup_layer(ticket).queued) == 1
        assert db.release_layers(ticket)
        pb.allocator.reclaim_request(request.transfer_id, consumer_complete=True)


def test_protocol_versions_and_real_d_suffix_boundary_are_unchanged():
    p, d = direct_scheduler(), direct_scheduler()
    with backend_pair(p, d) as (_, _, db):
        request = StartRequest(str(uuid4()), 0, (1, 2, 3, 4), 2, True)
        ticket = db.reserve_layers(request, eos_token_id=31, prefix_len=2)
        meta = KVTransferMetadata.from_slots(
            request.transfer_id,
            p.layout,
            (0, 1),
            num_pool_pages=8,
            prefix_len=2,
            transfer_mode="shm_dram",
        )
        assert ticket.version == LAYERED_TICKET_VERSION == 3
        assert meta.version == HANDOFF_VERSION == 2
        assert ticket.prefix_len == 2 and ticket.kv_tokens == 2
        assert LayeredReceiveTicket.from_wire(ticket.to_wire()) == ticket
        assert KVTransferMetadata.from_wire(meta.to_wire()) == meta
        db.release_layers(ticket)


def test_service_cancel_does_not_free_destination_pages_before_partial_h2d_drains():
    p = cpu_scheduler("prefill", max_running_req=8, prefill_budget=32)
    d = cpu_scheduler("decode", max_running_req=8)
    p.pd_service_config = d.pd_service_config = {"batch_wait_ms": 0}
    with backend_pair(p, d) as (_, pb, db):
        ps, ds = WorkerService(p, pb, 2), WorkerService(d, db, 2)
        db.install_stream = Stream("h2d")
        d.engine.stream = Stream()
        Event.h2d.clear()
        try:
            item = reserve(ds, (1, 2, 3), 5)
            session = submit(ps, item)
            until(
                lambda: any(e.pending for e in db._receives.values()),
                step=lambda: (ps.step(), ds.step()),
            )
            before = (d.table_manager.available_size, d.cache_manager.available_size)
            session.cancelled.set()
            item[1].cancelled.set()
            for _ in range(5):
                ps.step()
                ds.step()
            assert (d.table_manager.available_size, d.cache_manager.available_size) == before
            assert ds.contexts and db._receives and pb.allocator.leases
            Event.h2d.set()
            until(lambda: not ps.contexts and not ds.contexts, step=lambda: (ps.step(), ds.step()))
            assert d.table_manager.available_size == 8
            assert not pb.allocator.leases and not db._receives
        finally:
            Event.h2d.set()
            ps.close()
            ds.close()


def test_unknown_h2d_never_unmaps_or_returns_its_slot(monkeypatch):
    p, d = direct_scheduler(), direct_scheduler()
    with backend_pair(p, d) as (_, pb, db):
        db.install_stream = Stream("h2d")
        Event.h2d.clear()
        ticket, job, _ = start_direct(pb, db, p)
        slots = torch.tensor([5, 2, 1], dtype=torch.int32)
        until(lambda: bool(db._lookup_layer(ticket).queued))
        db.progress_receive(ticket, slots)
        entry = db._lookup_layer(ticket)
        event = entry.pending[0][1]
        original = event.query
        monkeypatch.setattr(event, "query", lambda: (_ for _ in ()).throw(RuntimeError("unknown")))
        try:
            with pytest.raises(TransferBackendUnavailable, match="H2D completion"):
                db.progress_receive(ticket, slots)
            assert pb.allocator.leases and entry.pending and db.shared.regions
            assert db._fatal
        finally:
            # Only the CPU test can prove this fake copy is drained and reset the
            # injected driver failure. Production must stop the Worker instead.
            monkeypatch.setattr(event, "query", original)
            Event.h2d.set()
            db._fatal = None
            job.seal(cancel=True)
            until(job.done.is_set, step=lambda: db.progress_receive(ticket, slots, cancel=True))
            db.release_layers(ticket)


def _decode_process(encoded, pipe):
    from minisgl.disagg.shm_relay_pool import RelayManifest

    scheduler = direct_scheduler()
    manifest = RelayManifest.from_json(encoded)
    backend = LayeredShmBackend(
        ("127.0.0.1", 0),
        manifest=manifest,
        worker_id="d-0",
        numa_id=1,
        host_id=manifest.host_id,
        staging_slots=2,
        timeout=3,
        max_transfer_bytes=64 * scheduler.layout.bytes_per_token,
        max_registered_bytes=65536,
    )
    try:
        backend.configure_pipeline(scheduler)
        request = StartRequest(str(uuid4()), 0, (1, 2, 3), 3, True)
        ticket = backend.reserve_layers(request, eos_token_id=31)
        pipe.send((request.to_object(), ticket.to_wire()))
        slots = torch.tensor([5, 2, 1], dtype=torch.int32)
        deadline = time.monotonic() + 8
        while backend.finish_layers(ticket) is None:
            backend.progress_receive(ticket, slots)
            if time.monotonic() > deadline:
                raise TimeoutError("independent D process did not finish installation")
            time.sleep(0.001)
        pipe.send(
            [
                getattr(scheduler.engine.kv_cache, key)(layer).flatten(0, 1)[slots.long()].tolist()
                for layer in range(5)
                for key in ("k_cache", "v_cache")
            ]
        )
        pipe.recv()
        backend.release_layers(ticket, consumer_complete=True)
    finally:
        backend.close()
        pipe.close()


def test_independent_processes_use_offsets_instead_of_virtual_addresses():
    p = direct_scheduler()
    p.engine.kv_cache._kv_buffer.fill_(17)
    manifest = make_manifest(p.layout)
    owner = SharedRelayPool(manifest, create=True, initialize=False)
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_decode_process, args=(manifest.to_json(), child))
    pb = None
    try:
        process.start()
        child.close()
        assert parent.poll(15)
        obj, wire = parent.recv()
        request, ticket = StartRequest.from_object(obj), LayeredReceiveTicket.from_wire(wire)
        pb = LayeredShmBackend(
            manifest=manifest,
            worker_id="p-0",
            numa_id=0,
            host_id=manifest.host_id,
            staging_slots=2,
            timeout=3,
            max_transfer_bytes=64 * p.layout.bytes_per_token,
            max_registered_bytes=65536,
        )
        pb.configure_pipeline(p)
        job = pb.begin_layer_send(ticket, request, p.engine.kv_cache, wait=False)
        plan = BatchTransferPlan(
            pb,
            [
                BatchTransferEntry(
                    request.transfer_id, torch.tensor([1, 3, 6], dtype=torch.int32), True, job
                )
            ],
        )
        for layer in range(5):
            plan.layer_ready(layer)
        job.seal()
        until(job.done.is_set, seconds=5)
        assert job.error is None and not pb.allocator.leases
        assert parent.poll(5)
        assert parent.recv() == [[[[17.0, 17.0]]] * 3] * 10
        pb.close()
        pb = None
        parent.send("stop")
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
        process.join()
        parent.close()
        if pb is not None:
            pb.close()
        owner.close()


@pytest.fixture
def routing_pool():
    pool = WorkerPool(["p0:1", "p1:1"], ["d0:1", "d1:1"])
    for i, node in enumerate(pool.nodes):
        pool.apply_health(
            node,
            {
                "status": "ready",
                "role": node.role,
                "multiplex_version": 1,
                "handoff_version": 2,
                "decode_prefix_reuse": True,
                "layered_transfer": True,
                "transfer_mode": "shm_dram",
                "numa_id": i % 2,
                "host_id": "host",
                "shm_pool": "relay",
                "layout": {"page_size": 1},
                "num_pages": 10000,
                "max_sequence_tokens": 10000,
                "transfer": {"mooncake": {"layered": {"layers_per_chunk": 1}}},
                "load": {"waiting_requests": 0, "running_requests": 0, "pending_prefill_tokens": 0},
            },
        )
        assert node.healthy
    yield pool
    pool.close()


@pytest.mark.parametrize(
    "policy,affinity",
    [("power_of_two", "prefer"), ("cache_aware", "prefer"), ("cache_aware", "balanced")],
)
def test_numa_routing_prefers_local_but_falls_back_for_load_or_cache(
    routing_pool, policy, affinity
):
    p0, p1 = routing_pool.prefill_pool
    d0, d1 = routing_pool.decode_pool
    router = PolicyRouter(
        routing_pool,
        config=RoutingConfig(
            policy=policy, affinity_mode=affinity, cache_min_tokens=1, pd_cross_numa_penalty=4
        ),
    )
    request = StartRequest(str(uuid4()), 0, tuple(range(1000)), 2, True)
    bucket = RoutingBucket()
    assert router.select_decode(request=request, bucket=bucket, prefill=p0) is d0
    d0.health["load"]["waiting_requests"] = 10
    assert router.select_decode(request=request, bucket=bucket, prefill=p0) is d1
    # P0 has no local D; a high P0 prefix hit still outweighs the cross-socket cost.
    d0.healthy = False
    if policy == "cache_aware":
        assert router.select_prefill(request=request, bucket=bucket, prefix_hits={"p-0": 900}) is p0
        assert router.select_prefill(request=request, bucket=bucket, prefix_hits={}) is p1
    else:
        assert router.select_prefill(request=request, bucket=bucket) is p1
    d1.health["host_id"] = "different-host"
    with pytest.raises(NoHealthyWorker, match="same-host"):
        router.select_decode(request=request, bucket=bucket, prefill=p0)


def test_single_socket_cache_and_seeded_load_selection_keep_existing_semantics(routing_pool):
    for node in routing_pool.nodes:
        node.health["numa_id"] = 0
    config = RoutingConfig(policy="cache_aware", cache_min_tokens=1, pd_cross_numa_penalty=999999)
    router = PolicyRouter(routing_pool, seed=10, config=config)
    request = StartRequest(str(uuid4()), 0, (1, 2, 3, 4, 5), 2, True)
    p0, p1 = routing_pool.prefill_pool
    p0.health["load"]["pending_prefill_tokens"] = 10000
    assert (
        router.select_prefill(request=request, bucket=RoutingBucket(), prefix_hits={"p-0": 4}) is p0
    )
    a = [
        router.select_decode(request=request, bucket=RoutingBucket(), prefill=p0).worker_id
        for _ in range(20)
    ]
    for node in routing_pool.nodes:
        node.health["transfer_mode"] = "gpu_direct"
    old = PolicyRouter(routing_pool, seed=10, config=config)
    b = [old.select_decode(request=request, bucket=RoutingBucket()).worker_id for _ in range(20)]
    assert a == b


def test_health_requires_host_identity_and_probe_rejects_different_shared_pools(
    routing_pool, monkeypatch
):
    d = routing_pool.decode_pool[0]
    saved = dict(d.health)
    routing_pool.apply_health(d, {k: v for k, v in saved.items() if k != "host_id"})
    assert not d.healthy
    routing_pool.apply_health(d, saved)
    monkeypatch.setattr(routing_pool, "probe", lambda node: None)
    routing_pool.probe_all()
    assert all(n.healthy for n in routing_pool.nodes)  # NUMA ids may differ.
    d.health["shm_pool"] = "another-pool"
    routing_pool.probe_all()
    assert not d.healthy and all(n.healthy for n in routing_pool.prefill_pool)


@pytest.mark.parametrize("mismatch", [None, "host_id", "shm_pool"])
def test_direct_client_shm_capabilities_validate_before_reserve(monkeypatch, mismatch):
    caps = {
        "status": "capabilities",
        "layered_transfer": True,
        "handoff_version": 2,
        "decode_prefix_reuse": True,
        "transfer_mode": "shm_dram",
        "host_id": "local-host",
        "shm_pool": "relay",
    }
    decode = {**caps}
    if mismatch:
        decode[mismatch] = "different"

    class Wire:
        def __init__(self, replies):
            self.replies, self.sent = list(replies), []
            self.socket = self

        def send(self, value):
            self.sent.append(value)

        def receive(self):
            return self.replies.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    p, d = Wire([caps]), Wire([decode, {"status": "ready"}])
    monkeypatch.setattr(
        direct_client.socket,
        "create_connection",
        lambda endpoint, **kwargs: p if endpoint[0] == "p" else d,
    )
    monkeypatch.setattr(direct_client, "Channel", lambda sock, *args: sock)
    calls = []

    def continuation(request, prefill, decoder, ready, **kwargs):
        calls.append((request, prefill, decoder, kwargs))
        yield 7

    monkeypatch.setattr(direct_client, "stream_reserved", continuation)
    client = direct_client.PDClient("p:1", "d:1")
    try:
        if mismatch:
            with pytest.raises(ValueError, match="same host and shared pool"):
                client.generate((1, 2), 2)
            assert not calls
            assert not any(msg.get("op") == "reserve" for msg in d.sent)
        else:
            assert client.generate((1, 2), 2) == [7]
            assert calls[0][3]["transfer_mode"] == "shm_dram"
            assert any(msg.get("op") == "reserve" for msg in d.sent)
    finally:
        client.close()
