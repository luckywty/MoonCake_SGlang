"""Real managers, tensor attention, sender pump and pool; no socket/CUDA required."""

import json
import queue
import threading
import time
from contextlib import contextmanager, nullcontext
from uuid import uuid4

import pytest
import torch
from layered_reference import CpuLayeredBackend, FakeEngine, FakeEvent, FakeStream, MemoryControl
from minisgl.disagg.control import StartRequest
from minisgl.disagg.layered_protocol import LayeredReceiveTicket
from minisgl.disagg.service import Session, WorkerService
from pd_reference import cpu_pinned_allocations, cpu_scheduler, reference_tokens


@pytest.fixture(autouse=True)
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: FakeStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    with cpu_pinned_allocations():
        yield
    assert not FakeEngine.registrations


@contextmanager
def cluster(
    group=1,
    budget=32,
    rows=8,
    decoders=1,
    inference_pools=False,
    staging_slots=4,
    page_size=4,
    prefill_cache_type="naive",
    decode_radix_cache=False,
):
    backends, services = [], []
    try:
        for i, role in enumerate(["prefill"] + ["decode"] * decoders):
            cache_type = (
                prefill_cache_type
                if role == "prefill"
                else ("radix" if decode_radix_cache else "naive")
            )
            scheduler = cpu_scheduler(
                role,
                page_size=page_size,
                max_running_req=rows,
                prefill_budget=budget,
                cache_type=cache_type,
            )
            scheduler.pd_service_config = {
                "batch_wait_ms": 0,
                "decode_radix_cache": decode_radix_cache,
            }
            backend = CpuLayeredBackend(
                None,
                timeout=2,
                max_transfer_bytes=16384,
                max_registered_bytes=32768,
                layers_per_chunk=group,
                staging_slots=staging_slots,
            )
            backends.append(backend)
            backend.endpoint = f"127.0.0.1:{1980 + i}"
            backend.bind_device(scheduler.device)
            with torch.inference_mode() if inference_pools else nullcontext():
                backend.configure_pipeline(scheduler)
            services.append(WorkerService(scheduler, backend, 2))
        targets = {b.endpoint: b for b in backends[1:]}
        connections = []

        @contextmanager
        def connect(endpoint):
            connections.append(endpoint)
            yield MemoryControl(targets[endpoint])

        backends[0]._connection = connect
        backends[0]._channel = lambda sock: sock
        yield services[0], services[1:], connections
    finally:
        # Drain native writes before releasing test backing storage, including
        # when an assertion failed while deliberately holding a native gate.
        for backend in backends:
            for gate in backend.fake_engine.destination_gates.values():
                gate.set()
        for service in services:
            service.close()
        for backend in reversed(backends):
            backend.cleanup_fake()


def receive(session):
    result = []
    while True:
        try:
            result.append(session.outbox.get_nowait())
        except queue.Empty:
            return result


def reserve(d, prompt, tokens, ignore_eos=True):
    request = StartRequest(str(uuid4()), 0, tuple(prompt), tokens, ignore_eos)
    session = Session()
    d.submit(session, {"op": "reserve", "request": request.to_object()})
    d.step()
    messages = receive(session)
    assert len(messages) == 1 and messages[0]["status"] == "ready", messages
    return request, session, messages[0]["ticket"]


def submit(p, item):
    request, _, ticket = item
    session = Session()
    p.submit(session, {"op": "prefill", "request": request.to_object(), "ticket": ticket})
    return session


def drive(p, decoders, items, psessions, *, deadline=3):
    outputs, handoffs, sent = {}, {}, set()
    for request, _, _ in items:
        outputs[request.transfer_id] = []
    until = time.monotonic() + deadline
    while time.monotonic() < until:
        p.step()
        for d in decoders:
            d.step()
        for item, ps in zip(items, psessions, strict=True):
            request, ds, _ = item
            tid = request.transfer_id
            for message in receive(ps):
                assert message["status"] != "error", message
                if message["status"] == "handoff":
                    handoffs[tid] = message["handoff"]
                    outputs[tid].append(handoffs[tid]["first_token_id"])
                if message["status"] in ("sent", "retired"):
                    sent.add(tid)
                    target = next(d for d in decoders if ds in d.contexts)
                    if message["status"] == "sent":
                        target.submit(
                            ds, {"op": "commit", "transfer_id": tid, "handoff": handoffs[tid]}
                        )
                    else:
                        target.submit(ds, {"op": "release", "transfer_id": tid})
            for message in receive(ds):
                assert message["status"] != "error", message
                if message["status"] == "token":
                    outputs[tid].append(message["token_id"])
        if all(ds.done.is_set() for _, ds, _ in items) and not p.contexts:
            return outputs
        time.sleep(0.002)
    raise AssertionError(f"unfinished contexts: P={p.snapshot}, D={[d.snapshot for d in decoders]}")


@pytest.mark.parametrize("group,budget", [(1, 32), (2, 5), (4, 7)])
def test_batched_prefill_and_continuous_decode_match_independent_reference(group, budget):
    cases = [((1, 2, 3), 7), ((4, 5, 6, 7), 4), ((8, 9), 9)]
    expected = [reference_tokens(prompt, n) for prompt, n in cases]
    with cluster(group=group, budget=budget, inference_pools=True) as (p, decoders, connections):
        d = decoders[0]
        items = [reserve(d, prompt, n) for prompt, n in cases]
        tickets = [LayeredReceiveTicket.from_wire(json.dumps(item[2]).encode()) for item in items]
        assert len({t.address for t in tickets}) == len(tickets)
        for a, b in zip(tickets, tickets[1:]):
            assert a.address + a.nbytes <= b.address
        sessions = [submit(p, item) for item in items]
        outputs = drive(p, decoders, items, sessions)
        for item, tokens in zip(items, expected):
            assert outputs[item[0].transfer_id] == tokens
        assert p.stats["max_prefill_batch"] > 1
        assert d.stats["max_decode_batch"] > 1
        assert connections == [d.backend.endpoint]
        for service in [p, d]:
            assert service.s.health()["free_pages"] == 32
            assert service.s.health()["free_rows"] == 8
            assert not service.contexts
        assert not d.backend._leases
        assert d.backend._allocator.available == d.backend._receive_buffer.nbytes
        assert d.backend.fake_engine.register_calls == 1
        assert p.backend.fake_engine.register_calls == 4


@pytest.mark.parametrize("group,slots", [(1, 2), (2, 4)])
def test_blocked_destination_does_not_delay_another_requests_sent_receipt(group, slots):
    with cluster(group=group, decoders=2, staging_slots=slots) as (p, decoders, connections):
        slow, fast = decoders
        gate = threading.Event()
        p.backend.fake_engine.destination_gates[slow.backend.runtime.session] = gate
        a = reserve(slow, (1, 2, 3), 5)
        b = reserve(fast, (4, 5), 5)
        sa, sb = submit(p, a), submit(p, b)
        until = time.monotonic() + 1
        fast_messages = []
        while time.monotonic() < until and not sb.done.is_set():
            p.step()
            fast_messages.extend(receive(sb))
            time.sleep(0.002)
        fast_messages.extend(receive(sb))
        assert [m["status"] for m in fast_messages] == ["handoff", "sent"]
        assert not sa.done.is_set()
        assert p.stats["max_prefill_batch"] == 2
        assert sorted(connections) == sorted(d.backend.endpoint for d in decoders)
        # Cancelling one request must retain only its exposed lease until drain.
        a[1].cancelled.set()
        sa.cancelled.set()
        slow.step()
        p.step()
        assert a[2]["locator"] in slow.backend._leases
        fast.submit(
            b[1],
            {
                "op": "commit",
                "transfer_id": b[0].transfer_id,
                "handoff": fast_messages[0]["handoff"],
            },
        )
        tokens = [fast_messages[0]["handoff"]["first_token_id"]]
        for _ in range(20):
            fast.step()
            tokens.extend(m["token_id"] for m in receive(b[1]) if m["status"] == "token")
            if b[1].done.is_set():
                break
        assert tokens == reference_tokens(b[0].prompt_ids, b[0].max_tokens)
        gate.set()
        until = time.monotonic() + 1
        while p.contexts and time.monotonic() < until:
            p.step()
            slow.step()
            time.sleep(0.002)
        assert not p.contexts and not slow.backend._leases


@pytest.mark.parametrize("cancel_op", ["disconnect", "release"])
def test_cancelled_install_retains_receive_slice_and_private_pages_until_event_query(cancel_op):
    with cluster(group=2) as (p, decoders, _):
        d = decoders[0]
        item = reserve(d, (1, 2, 3), 5)
        ps = submit(p, item)
        messages = []
        until = time.monotonic() + 1
        while not ps.done.is_set() and time.monotonic() < until:
            p.step()
            messages.extend(receive(ps))
            time.sleep(0.002)
        messages.extend(receive(ps))
        assert [m["status"] for m in messages] == ["handoff", "sent"]
        d.install_stream = FakeStream()
        d.s.engine.stream = FakeStream()
        d.submit(
            item[1],
            {"op": "commit", "transfer_id": item[0].transfer_id, "handoff": messages[0]["handoff"]},
        )
        ctx = d.contexts[item[1]]
        d._command(*d.commands.get_nowait())
        d._install(ctx)
        fence = threading.Event()
        ctx.install_event.query = fence.is_set
        rows = d.s.table_manager.available_size
        pages = len(d.s.cache_manager.free_slots)
        if cancel_op == "release":
            d.submit(item[1], {"op": "release", "transfer_id": item[0].transfer_id})
        else:
            item[1].cancelled.set()
        d.step()
        assert d.s.table_manager.available_size == rows
        assert len(d.s.cache_manager.free_slots) == pages
        assert item[2]["locator"] in d.backend._leases
        assert not item[1].done.is_set()
        assert receive(item[1]) == []
        fence.set()
        d.step()
        assert not d.contexts and not d.backend._leases
        assert d.s.health()["free_pages"] == 32
        if cancel_op == "release":
            assert receive(item[1])[0]["status"] == "released"


def test_failed_cpu_install_releases_the_completed_scatter_consumer(monkeypatch):
    with cluster(group=2) as (p, decoders, _):
        d = decoders[0]
        item = reserve(d, (1, 2, 3), 5)
        ps = submit(p, item)
        messages = []
        until = time.monotonic() + 1
        while not ps.done.is_set() and time.monotonic() < until:
            p.step()
            messages.extend(receive(ps))
            time.sleep(0.002)
        messages.extend(receive(ps))
        original = d.s.engine.kv_cache.import_layered_tensor

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected partial install failure")

        monkeypatch.setattr(d.s.engine.kv_cache, "import_layered_tensor", fail)
        d.submit(
            item[1],
            {"op": "commit", "transfer_id": item[0].transfer_id, "handoff": messages[0]["handoff"]},
        )
        d.step()
        assert receive(item[1])[0]["status"] == "error"
        d.step()
        assert not d.contexts and not d.backend._leases
        assert d.s.health()["free_pages"] == 32


def test_decode_reservations_include_future_output_pages_and_fail_without_leaking():
    with cluster(rows=8) as (_, decoders, _):
        d = decoders[0]
        first = reserve(d, (1, 2, 3), 61)
        second = reserve(d, (4, 5, 6), 61)
        rejected = Session()
        request = StartRequest(str(uuid4()), 0, (7,), 2, True)
        d.submit(rejected, {"op": "reserve", "request": request.to_object()})
        d.step()
        assert receive(rejected)[0]["code"] == "busy"
        d.step()
        assert len(d.backend._leases) == 2
        assert len(d.contexts) == 2
        first[1].cancelled.set()
        d.step()
        third = reserve(d, (7, 8), 3)
        assert len(d.backend._leases) == 2
        for item in [second, third]:
            item[1].cancelled.set()
        d.step()
        assert d.s.health()["free_pages"] == 32


def test_chunked_prefill_cancel_releases_only_that_requests_pages():
    with cluster(budget=3) as (p, decoders, _):
        d = decoders[0]
        cancelled = reserve(d, (1, 2, 3, 4, 5, 6, 7), 5)
        survivor = reserve(d, (8, 9), 5)
        a, b = submit(p, cancelled), submit(p, survivor)
        p.step()
        assert p.contexts[a].req.cached_len == 3
        a.cancelled.set()
        cancelled[1].cancelled.set()
        d.step()
        outputs = drive(p, decoders, [survivor], [b])
        assert outputs[survivor[0].transfer_id] == reference_tokens((8, 9), 5)
        assert not p.contexts and not d.contexts
        assert p.s.health()["free_pages"] == d.s.health()["free_pages"] == 32


def test_same_batch_eos_does_not_cancel_other_requests():
    eos_prompt = next((i,) for i in range(32) if reference_tokens((i,), 1) == [31])
    with cluster(group=1) as (p, decoders, _):
        d = decoders[0]
        eos = reserve(d, eos_prompt, 5, ignore_eos=False)
        other = reserve(d, (1, 2, 3), 5)
        outputs = drive(p, decoders, [eos, other], [submit(p, eos), submit(p, other)])
        assert outputs[eos[0].transfer_id] == [31]
        assert outputs[other[0].transfer_id] == reference_tokens((1, 2, 3), 5)
        assert p.stats["max_prefill_batch"] == 2
        assert not d.backend._leases


def test_page_aligned_prefill_admission_queues_extra_requests_instead_of_overallocating():
    with cluster(budget=32) as (p, decoders, _):
        d = decoders[0]
        items = [reserve(d, (i,), 2) for i in range(4)]
        # Leave exactly two physical pages available on P. Four one-token
        # prompts fit a token-only estimate, but require four distinct pages.
        held = p.s.cache_manager._allocate(30)
        try:
            outputs = drive(p, decoders, items, [submit(p, item) for item in items])
            assert p.stats["max_prefill_batch"] == 2
            assert all(len(tokens) == 2 for tokens in outputs.values())
            assert len(p.s.cache_manager.free_slots) == 2
        finally:
            p.s.cache_manager.free_slots = torch.cat([p.s.cache_manager.free_slots, held])
        assert p.s.health()["free_pages"] == 32


def test_whole_payload_control_path_batches_with_mock_transport():
    from minisgl.disagg.transfer import MockFabric, MockTransferBackend

    fabric = MockFabric()
    services = [
        WorkerService(
            cpu_scheduler(role, max_running_req=4, prefill_budget=32),
            MockTransferBackend(role, fabric),
            1,
        )
        for role in ("prefill", "decode")
    ]
    p, d = services
    p.batch_wait = 0
    requests = [StartRequest(str(uuid4()), 0, (i, i + 1), 5, True) for i in (1, 3, 5)]
    ps, ds = [Session() for _ in requests], [Session() for _ in requests]
    outputs = [[] for _ in requests]
    try:
        for request, session in zip(requests, ps):
            p.submit(session, {"op": "prefill", "request": request.to_object()})
        p.step()
        for i, session in enumerate(ps):
            handoff = receive(session)[0]["handoff"]
            outputs[i].append(handoff["first_token_id"])
            d.submit(ds[i], {"op": "decode", "handoff": handoff})
        d.step()
        for source, dest in zip(ps, ds):
            p.submit(source, {"op": "send", "ticket": receive(dest)[0]["ticket"]})
        until = time.monotonic() + 2
        while time.monotonic() < until:
            p.step()
            for request, source, dest in zip(requests, ps, ds):
                for message in receive(source):
                    assert message["status"] == "sent", message
                    d.submit(dest, {"op": "commit", "transfer_id": request.transfer_id})
            d.step()
            for tokens, dest in zip(outputs, ds):
                for message in receive(dest):
                    assert message["status"] != "error", message
                    if message["status"] == "token":
                        tokens.append(message["token_id"])
            if all(dest.done.is_set() for dest in ds):
                break
            time.sleep(0.002)
        assert outputs == [reference_tokens(r.prompt_ids, r.max_tokens) for r in requests]
        assert p.stats["max_prefill_batch"] == 3
        assert d.stats["max_decode_batch"] > 1
        assert not fabric._reservations
        assert all(s.s.health()["free_pages"] == 32 for s in services)
    finally:
        for service in services:
            service.close()
