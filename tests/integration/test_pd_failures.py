import socket
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace
from uuid import uuid4

import pytest
from minisgl.disagg.control import Channel, StartRequest
from minisgl.disagg.role import WorkerRole
from minisgl.disagg.transfer import MockFabric, MockTransferBackend, TransferError
from minisgl.disagg.worker import checkpoint_fingerprint, serve_worker
from pd_reference import cpu_pinned_allocations, cpu_scheduler


def request():
    return StartRequest(str(uuid4()), 0, (1, 2, 3, 4, 5), 5, True)


def idle_channel():
    return SimpleNamespace(check_connected=lambda: None)


def assert_released(worker):
    assert worker.health()["free_pages"] == 32
    assert worker.health()["free_rows"] == 2
    assert worker._owned_req is None
    assert not worker.decode_manager.runnable
    assert not worker.prefill_manager.runnable


@pytest.mark.parametrize("failure_point", ["forward", "metadata"])
def test_prefill_failure_releases_allocated_but_unpopulated_pages(monkeypatch, failure_point):
    with cpu_pinned_allocations():
        worker = cpu_scheduler("prefill")

        def fail(*_args):
            raise RuntimeError("injected failure after allocation")

        if failure_point == "forward":
            monkeypatch.setattr(worker.engine, "forward_batch", fail)
        else:
            monkeypatch.setattr(worker, "_prepare_batch_metadata", fail)
        with pytest.raises(RuntimeError, match="injected"):
            worker.prefill(request(), idle_channel())
        assert_released(worker)


@pytest.mark.parametrize("failure_point", ["prepare", "install"])
def test_decode_failure_rolls_back_private_import(monkeypatch, failure_point):
    with cpu_pinned_allocations():
        p = cpu_scheduler("prefill")
        handoff, payload = p.prefill(request(), idle_channel())
        d = cpu_scheduler("decode")
        fabric = MockFabric()
        receiver = MockTransferBackend("decode", fabric)
        sender = MockTransferBackend("prefill", fabric)
        original_prepare = receiver.prepare_receive

        def prepare(meta, layout, destination):
            if failure_point == "prepare":
                raise TransferError("injected registration failure")
            ticket = original_prepare(meta, layout, destination)
            sender.send(meta, payload, ticket)
            return ticket

        original_import = d.engine.kv_cache.import_tokens

        def install(*args, **kwargs):
            original_import(*args, **kwargs)
            raise RuntimeError("injected install failure after writes")

        monkeypatch.setattr(receiver, "prepare_receive", prepare)
        monkeypatch.setattr(d.engine.kv_cache, "import_tokens", install)
        channel = SimpleNamespace(
            check_connected=lambda: None,
            send=lambda message: None,
            receive=lambda: {"op": "commit", "transfer_id": handoff.kv.transfer_id},
        )
        with pytest.raises((TransferError, RuntimeError), match="injected"):
            d.decode(handoff, channel, receiver)
        assert_released(d)


def test_layout_mismatch_fails_before_page_allocation():
    with cpu_pinned_allocations():
        p = cpu_scheduler("prefill")
        handoff, _ = p.prefill(request(), idle_channel())
        d = cpu_scheduler("decode")
        d.layout = replace(d.layout, model_fingerprint="b" * 64)
        with pytest.raises(ValueError, match="incompatible"):
            d._reserve_decode(handoff)
        assert_released(d)


def test_checkpoint_fingerprint_binds_effective_config_and_weight_content(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"hash-fixture-a")
    initial = checkpoint_fingerprint(str(tmp_path), {"rope_theta": 10000})
    assert initial == checkpoint_fingerprint(str(tmp_path), {"rope_theta": 10000})
    assert initial != checkpoint_fingerprint(str(tmp_path), {"rope_theta": 20000})
    weights.write_bytes(b"hash-fixture-b")
    assert initial != checkpoint_fingerprint(str(tmp_path), {"rope_theta": 10000})


def test_worker_stops_when_cleanup_cannot_restore_ownership():
    class BrokenScheduler:
        role = WorkerRole.PREFILL
        _owned_req = object()

        def prefill(self, *args):
            raise RuntimeError("injected unrecoverable cleanup failure")

    errors = []
    stop = Event()
    with socket.create_server(("127.0.0.1", 0)) as listener:

        def serve():
            try:
                serve_worker(BrokenScheduler(), listener, None, 2, stop)
            except RuntimeError as exc:
                errors.append(str(exc))

        thread = Thread(target=serve, daemon=True)
        thread.start()
        try:
            with socket.create_connection(listener.getsockname()) as sock:
                channel = Channel(sock, 2)
                channel.send({"op": "prefill", "request": request().to_object()})
                assert channel.receive()["status"] == "error"
            thread.join(timeout=3)
            assert not thread.is_alive()
            assert errors == ["split worker could not release its request resources"]
        finally:
            stop.set()
            thread.join(timeout=3)
