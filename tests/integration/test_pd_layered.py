import json
import socket
import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import minisgl.core as core
import minisgl.distributed.info as tp_info
import pytest
import torch
from layered_reference import (
    CpuLayeredBackend,
    FakeEngine,
    FakeEvent,
    FakeStream,
    MemoryControl,
    wait_for,
)
from minisgl.disagg.client import PDClient
from minisgl.disagg.control import (
    CancelledError,
    Channel,
    StartRequest,
    endpoint_address,
    endpoint_name,
)
from minisgl.disagg.layered_protocol import LAYERED_TICKET_VERSION
from minisgl.disagg.protocol import KVLayout, KVTransferMetadata, RequestHandoff
from minisgl.disagg.role import WorkerRole
from minisgl.disagg.transfer import TransferError
from minisgl.disagg.worker import serve_worker
from minisgl.kvcache.mha_pool import MHAKVCache
from pd_reference import cpu_pinned_allocations, cpu_scheduler, reference_tokens


@pytest.fixture(autouse=True)
def fake_cuda(monkeypatch):
    monkeypatch.setattr(core, "_GLOBAL_CTX", core._GLOBAL_CTX)
    monkeypatch.setattr(tp_info, "_TP_INFO", tp_info.DistributedInfo(0, 1))
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: FakeStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    with cpu_pinned_allocations():
        yield
    assert not FakeEngine.registrations


@contextmanager
def pair(layers=5, group=2, gate=None, fail=False, timeout=1, lose_receipt=None):
    layout = KVLayout("a" * 64, "float32", layers, 1, 8, 1)
    backends = []
    try:
        for role in (WorkerRole.DECODE, WorkerRole.PREFILL):
            backend = CpuLayeredBackend(
                None,
                timeout=timeout,
                max_transfer_bytes=16384,
                max_registered_bytes=32768,
                layers_per_chunk=group,
                staging_slots=2,
                gate=gate if role is WorkerRole.PREFILL else None,
                fail=fail,
            )
            backends.append(backend)
            if role is WorkerRole.DECODE:
                backend.endpoint = "127.0.0.1:1931"
            backend.bind_device(torch.device("cpu"))
            backend.configure_pipeline(
                SimpleNamespace(
                    layout=layout,
                    device=torch.device("cpu"),
                    role=role,
                    engine=SimpleNamespace(max_seq_len=32),
                )
            )
        p, d = backends[1], backends[0]
        p._connection = lambda endpoint: nullcontext(MemoryControl(d, lose_receipt))
        with patch("minisgl.disagg.layered_mooncake.Channel", lambda sock, *args: sock):
            yield p, d, layout
    finally:
        for backend in reversed(backends):
            backend.cleanup_fake()


def start_job(p, d, layout, tokens=3):
    request = StartRequest(str(uuid4()), 0, tuple(range(tokens)), 5, True)
    target = d.reserve_layers(request, 31)
    pool = MHAKVCache(1, layout.layers, 8, 9, 1, torch.float32, torch.device("cpu"))
    pool._kv_buffer.copy_(torch.arange(pool._kv_buffer.numel()).reshape(pool._kv_buffer.shape))
    slots = torch.tensor([7, 2, 5][:tokens], dtype=torch.int64)
    job = p.begin_layer_send(target, request, pool, lambda: None)
    job.set_slots(slots)
    return job, pool, slots, target


@pytest.mark.parametrize("group", [1, 2, 4])
def test_ring_backpressure_and_byte_exact_layer_install(group):
    gate = threading.Event()
    with pair(group=group, gate=gate) as (p, d, layout):
        job, pool, slots, target = start_job(p, d, layout)
        first_group_end = target.groups[0][1]
        for layer in range(first_group_end):
            job.layer_ready(layer)
        # Transfer has already started while the later layers have not even
        # notified readiness. This is independent of Python thread timing.
        wait_for(lambda: len(p.fake_engine.submissions) == 1)
        assert job._last_layer < layout.layers - 1
        for layer in range(first_group_end, layout.layers):
            job.layer_ready(layer)
        wait_for(lambda: len(p.fake_engine.submissions) == 2)
        assert job._last_layer == 4  # All producer notifications returned with DMA blocked.
        assert not job.done.is_set()
        assert all(not ticket.drained for ticket in p.fake_engine.submissions)
        gate.set()
        job.finish()
        assert len(p.fake_engine.submissions) == len(target.groups)
        received = d.finish_layers(target)
        dest = MHAKVCache(1, 5, 8, 9, 1, torch.float32, torch.device("cpu"))
        dest._kv_buffer.fill_(-1)
        dest.import_layered_tensor(received, torch.tensor([1, 3, 6]), num_pool_pages=8)
        assert torch.equal(pool._kv_buffer[:, :, slots], dest._kv_buffer[:, :, [1, 3, 6]])
        assert torch.all(dest._kv_buffer[:, :, 8] == -1)
        d.release_layers(target, consumer_complete=True)
        assert p.fake_engine.register_calls == 2
        assert d.fake_engine.register_calls == 1
        # Another request reuses registrations, with a new attempt identity.
        request = StartRequest(str(uuid4()), 0, (1,), 2, True)
        second = d.reserve_layers(request, 31)
        assert second.address == target.address and second.locator != target.locator
        d.release_layers(second)
        assert d.fake_engine.register_calls == 1


def test_cancel_keeps_exposed_receive_and_slots_until_drain():
    gate = threading.Event()
    with pair(group=1, gate=gate) as (p, d, layout):
        job, _, _, target = start_job(p, d, layout)
        for layer in range(layout.layers):
            job.layer_ready(layer)
        wait_for(lambda: len(p.fake_engine.submissions) == 2)
        d.release_layers(target)  # Frontend disconnected while DMA is in flight.
        job.seal(cancel=True)
        assert not job.done.wait(0.03)
        assert d._lease is not None and not d._receive_buffer.safe_to_release
        other = d.reserve_layers(StartRequest(str(uuid4()), 0, (1,), 2, True), 31)
        assert other.address >= target.address + target.nbytes
        d.release_layers(other)
        assert target.locator in d._leases
        gate.set()
        job.finish(cancel=True)
        assert d._lease is None and d._receive_buffer.safe_to_release
        assert all(ticket.drained for ticket in p.fake_engine.submissions)


def test_receive_pool_waits_for_cuda_consumer_after_native_drain():
    with pair() as (p, d, layout):
        job, _, _, target = start_job(p, d, layout)
        for layer in range(layout.layers):
            job.layer_ready(layer)
        job.finish()
        d.finish_layers(target)
        assert d._lease.consuming and not d._receive_buffer.safe_to_release
        d.release_layers(target)  # Cancellation cannot discard an active CUDA reader.
        assert d._lease is not None
        d.release_layers(target, consumer_complete=True)
        assert d._lease is None and d._receive_buffer.safe_to_release


def test_terminal_state_without_drained_flag_does_not_allow_slot_reuse():
    with pair() as (p, d, _):
        slot = p._send_buffers[0]
        operation = p.runtime.submit_write(
            slot,
            {
                "session": d.runtime.session,
                "address": d._receive_buffer.address,
                "nbytes": 16,
            },
            nbytes=16,
        )
        ticket = operation.ticket
        drain = ticket.drain
        ticket.drain = lambda timeout_ms: "complete"
        try:
            with pytest.raises(TransferError, match="physically drained"):
                p.runtime.poll(operation, timeout_ms=0)
            with pytest.raises(TransferError, match="before native drain"):
                p.runtime.recycle_operation(operation)
            assert not slot.safe_to_release and operation.ticket is ticket
        finally:
            ticket.drain = drain


def test_lost_receipt_replays_identical_terminal_proof():
    lost = threading.Event()
    with pair(lose_receipt=lost) as (p, d, layout):
        job, _, _, target = start_job(p, d, layout)
        for layer in range(layout.layers):
            job.layer_ready(layer)
        job.finish()  # A lost receipt is replayed on a reconnected control channel.
        assert lost.is_set()
        assert d._lease.terminal["result"] == "complete"
        assert d._receive_buffer.safe_to_release
        d.release_layers(target)


@pytest.mark.parametrize(
    "backend", ["mooncake-layered-gpu-v2", "mooncake-layered-gpu-v4", None, True]
)
def test_invalid_completion_protocol_cannot_release_an_exposed_receive_buffer(backend):
    with pair() as (p, d, _):
        target = d.reserve_layers(StartRequest(str(uuid4()), 0, (1, 2), 3, True), 31)
        d.open_layers(target, p.runtime.session)
        proof = {
            "op": "layer_terminal",
            "backend": backend,
            "locator": target.locator,
            "transfer_id": target.transfer_id,
            "receiver": target.session,
            "source": p.runtime.session,
            "submitted": [],
            "completed": [],
            "result": "cancelled",
        }
        with pytest.raises(TransferError, match="incorrect layer completion message"):
            d._settle_layers(proof)
        assert d._lease.ticket == target and not d._receive_buffer.safe_to_release
        d._settle_layers({**proof, "backend": f"mooncake-layered-gpu-v{LAYERED_TICKET_VERSION}"})
        d.release_layers(target)
        assert d._lease is None and d._receive_buffer.safe_to_release


def test_stale_or_incomplete_certificate_never_releases_another_lease():
    with pair() as (p, d, layout):
        request = StartRequest(str(uuid4()), 0, (1, 2), 3, True)
        target = d.reserve_layers(request, 31)
        d.open_layers(target, p.runtime.session)
        proof = {
            "op": "layer_terminal",
            "backend": f"mooncake-layered-gpu-v{LAYERED_TICKET_VERSION}",
            "locator": target.locator,
            "transfer_id": target.transfer_id,
            "receiver": target.session,
            "source": p.runtime.session,
            "submitted": [0],
            "completed": [0],
            "result": "complete",
        }
        with pytest.raises(TransferError, match="quiescence"):
            d._settle_layers(proof)
        assert not d._receive_buffer.safe_to_release
        d.release_layers(target)
        proof["result"] = "cancelled"
        assert d._settle_layers(proof)["status"] == "retired"
        newer = d.reserve_layers(StartRequest(str(uuid4()), 0, (1,), 3, True), 31)
        assert d._settle_layers(proof)["status"] == "retired"
        assert d._lease.ticket == newer
        with pytest.raises(TransferError, match="changed"):
            d._settle_layers({**proof, "completed": []})
        d.release_layers(newer)


@pytest.mark.parametrize("failure", ["disconnect", "commit", "install", "busy"])
def test_layered_decode_failure_rolls_back_private_pages(failure, monkeypatch):
    with pair(layers=2, group=1) as (p, d, layout):
        scheduler = cpu_scheduler(WorkerRole.DECODE, page_size=1)
        request = StartRequest(str(uuid4()), 0, (1, 2, 3), 5, True)
        previous = (
            d.reserve_layers(StartRequest(str(uuid4()), 0, (1,), 2, True), 31)
            if failure == "busy"
            else None
        )

        class Control:
            def send(self, message):
                if failure != "install":
                    return
                target = d._lease.ticket
                d.open_layers(target, p.runtime.session)
                d._settle_layers(
                    {
                        "op": "layer_terminal",
                        "backend": f"mooncake-layered-gpu-v{LAYERED_TICKET_VERSION}",
                        "locator": target.locator,
                        "transfer_id": target.transfer_id,
                        "receiver": target.session,
                        "source": p.runtime.session,
                        "submitted": [0, 1],
                        "completed": [0, 1],
                        "result": "complete",
                    }
                )

            def receive(self):
                if failure == "disconnect":
                    raise CancelledError("injected control disconnect")
                if failure == "commit":
                    return {"op": "commit", "transfer_id": "stale"}
                metadata = KVTransferMetadata(request.transfer_id, layout, 3, (0, 1, 2))
                handoff = RequestHandoff(0, request.prompt_ids, 7, 5, True, 31, metadata)
                return {
                    "op": "commit",
                    "transfer_id": request.transfer_id,
                    "handoff": json.loads(handoff.to_wire()),
                }

        if failure == "install":

            fail_install = Mock(side_effect=ValueError("injected install failure"))
            monkeypatch.setattr(scheduler.engine.kv_cache, "import_layered_tensor", fail_install)
        with pytest.raises((ValueError, RuntimeError, CancelledError)) as raised:
            scheduler.decode_layered(request, Control(), d)
        if failure == "install":
            fail_install.assert_called_once()
            assert str(raised.value) == "injected install failure"
        assert scheduler._owned_req is None
        assert scheduler.table_manager.available_size == 2
        assert len(scheduler.cache_manager.free_slots) == 32
        if previous is not None:
            assert d._lease.ticket == previous
            d.release_layers(previous)
        assert d._lease is None


@pytest.mark.parametrize("failure", ["unknown", "failed_drained"])
def test_unproven_remote_completion_is_quarantined(failure):
    gate = threading.Event() if failure == "unknown" else None
    with pair(group=1, gate=gate, fail=failure == "failed_drained", timeout=0.1) as (p, d, layout):
        job, _, _, target = start_job(p, d, layout)
        for layer in range(layout.layers):
            job.layer_ready(layer)
        with pytest.raises(TransferError):
            job.finish()
        assert not d._receive_buffer.safe_to_release
        assert d._receive_buffer.quarantine_since is not None
        d.release_layers(target)
        assert d._lease is not None
        if failure == "unknown":
            assert any(not buffer.safe_to_release for buffer in p._send_buffers)
            assert p.runtime._operations  # Strong ticket references remain alive.


@contextmanager
def cpu_workers(group):
    backends, threads, listeners, schedulers = [], [], [], []
    stop = threading.Event()
    errors = []
    try:
        endpoints = []
        for role in (WorkerRole.PREFILL, WorkerRole.DECODE):
            scheduler = cpu_scheduler(role, page_size=4)
            schedulers.append(scheduler)
            backend = CpuLayeredBackend(
                ("127.0.0.1", 0) if role is WorkerRole.DECODE else None,
                timeout=2,
                max_transfer_bytes=16384,
                max_registered_bytes=32768,
                layers_per_chunk=group,
                staging_slots=2,
            )
            backends.append(backend)
            backend.bind_device(torch.device("cpu"))
            backend.configure_pipeline(scheduler)
            listener = socket.create_server(("127.0.0.1", 0))
            listeners.append(listener)
            endpoints.append(endpoint_name(*listener.getsockname()))

            def run(scheduler=scheduler, listener=listener, backend=backend):
                try:
                    serve_worker(scheduler, listener, backend, 2, stop)
                except BaseException as exc:
                    errors.append(exc)

            thread = threading.Thread(target=run, name=f"cpu-{role.value}", daemon=True)
            threads.append(thread)
            thread.start()
        yield PDClient(*endpoints, timeout=3), schedulers, backends
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=4)
            assert not thread.is_alive()
        for backend in backends:
            backend.cleanup_fake()
        for listener in listeners:
            listener.close()
        assert not errors


@pytest.mark.parametrize("group", [1, 2, 4])
def test_early_handshake_real_scheduler_continuation_and_eos(group):
    with cpu_workers(group) as (client, schedulers, backends):
        for prompt, budget, ignore_eos in (([1], 1, True), ([1, 2, 3, 4, 5, 6, 7], 5, True)):
            assert client.generate(prompt, budget, ignore_eos=ignore_eos) == reference_tokens(
                prompt, budget
            )
        eos_prompt = next([token] for token in range(32) if reference_tokens([token], 1) == [31])
        assert client.generate(eos_prompt, 5) == [31]
        wait_for(
            lambda: all(
                s.table_manager.available_size == 2 and len(s.cache_manager.free_slots) == 32
                for s in schedulers
            )
        )
        assert all(s.table_manager.available_size == 2 for s in schedulers)
        assert all(len(s.cache_manager.free_slots) == 32 for s in schedulers)
        assert backends[0].fake_engine.register_calls == 2
        assert backends[1].fake_engine.register_calls == 1
        assert backends[1]._lease is None


def test_client_disconnect_after_first_token_reclaims_layer_reservation():
    with cpu_workers(1) as (client, schedulers, backends):
        iterator = client.stream([1, 2, 3, 4, 5], 10, ignore_eos=True)
        next(iterator)
        iterator.close()
        wait_for(lambda: all(s.table_manager.available_size == 2 for s in schedulers))
        wait_for(lambda: backends[1]._lease is None)
        assert client.generate([1, 2], 3, ignore_eos=True) == reference_tokens([1, 2], 3)


def test_reservation_disconnect_before_prefill_reclaims_private_pages():
    with cpu_workers(1) as (client, schedulers, backends):
        request = StartRequest(str(uuid4()), 0, (1, 2, 3), 5, True)
        with socket.create_connection(endpoint_address(client.decode)) as sock:
            channel = Channel(sock, 2)
            channel.send({"op": "reserve", "request": request.to_object()})
            assert channel.receive()["status"] == "ready"
        wait_for(
            lambda: schedulers[1].table_manager.available_size == 2 and backends[1]._lease is None
        )
        assert backends[1]._lease is None
        assert schedulers[1].table_manager.available_size == 2
        assert len(schedulers[1].cache_manager.free_slots) == 32


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("page_size", [1, 4])
def test_group_pack_into_reused_slot(dtype, page_size):
    pool = MHAKVCache(2, 5, 4, 9, page_size, dtype, torch.device("cpu"))
    pool._kv_buffer.copy_(torch.arange(pool._kv_buffer.numel()).reshape(pool._kv_buffer.shape))
    source = torch.tensor([7 * page_size + i for i in range(page_size)] + [2 * page_size])
    layer_bytes = 2 * len(source) * 2 * 4 * dtype.itemsize
    packed = torch.empty(5 * layer_bytes, dtype=torch.uint8)
    for start, end in ((0, 2), (2, 4), (4, 5)):
        pool.export_layers_into(
            packed[start * layer_bytes : end * layer_bytes], source, start, end, num_pool_pages=8
        )
    dest = MHAKVCache(2, 5, 4, 9, page_size, dtype, torch.device("cpu"))
    dest._kv_buffer.fill_(-1)
    dest.import_layered_tensor(packed, source, num_pool_pages=8)
    assert torch.equal(
        pool._kv_buffer.flatten(2, 3).index_select(2, source),
        dest._kv_buffer.flatten(2, 3).index_select(2, source),
    )
    assert torch.all(dest._kv_buffer[:, :, 8] == -1)
