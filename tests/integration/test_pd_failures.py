from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from minisgl.disagg.control import StartRequest
from minisgl.disagg.transfer import MockFabric, MockTransferBackend, TransferError
from minisgl.disagg.worker import checkpoint_fingerprint
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


def test_worker_stops_when_cleanup_cannot_restore_ownership(monkeypatch):
    from minisgl.disagg.service import Session, WorkerService
    from minisgl.disagg.transfer import TransferBackendUnavailable

    with cpu_pinned_allocations():
        scheduler = cpu_scheduler("prefill", prefill_budget=32)
        scheduler.pd_service_config = {"batch_wait_ms": 0}
        service = WorkerService(scheduler, MockTransferBackend("p", MockFabric()), 0.1)
        original = scheduler.cache_manager.discard_private
        session = Session()

        def fail_forward(*args):
            raise RuntimeError("injected forward failure")

        def fail_cleanup(*args):
            raise RuntimeError("injected unrecoverable cleanup failure")

        monkeypatch.setattr(scheduler.engine, "forward_batch", fail_forward)
        monkeypatch.setattr(scheduler.cache_manager, "discard_private", fail_cleanup)
        service.submit(session, {"op": "prefill", "request": request().to_object()})
        try:
            service.step()
            assert session.outbox.get_nowait()["status"] == "error"
            with pytest.raises(TransferBackendUnavailable, match="could not release"):
                service.step()
        finally:
            monkeypatch.setattr(scheduler.cache_manager, "discard_private", original)
            service.close()
        assert_released(scheduler)


def test_failed_batch_keeps_other_pending_requests_runnable(monkeypatch):
    from minisgl.disagg.service import Session, WorkerService

    with cpu_pinned_allocations():
        scheduler = cpu_scheduler("prefill", prefill_budget=32)
        scheduler.pd_service_config = {"batch_wait_ms": 0}
        service = WorkerService(scheduler, MockTransferBackend("p", MockFabric()), 0.1)
        original = scheduler.engine.forward_batch
        sessions = [Session() for _ in range(3)]
        for session in sessions:
            req = StartRequest(str(uuid4()), 0, (1, 2), 1, True)
            service.submit(session, {"op": "prefill", "request": req.to_object()})

        def fail(*args):
            raise RuntimeError("injected failed batch")

        monkeypatch.setattr(scheduler.engine, "forward_batch", fail)
        try:
            service.step()
            assert [session.outbox.get_nowait()["status"] for session in sessions[:2]] == [
                "error",
                "error",
            ]
            assert len(scheduler.prefill_manager.pending_list) == 1
            monkeypatch.setattr(scheduler.engine, "forward_batch", original)
            service.step()
            assert sessions[2].outbox.get_nowait()["status"] == "handoff"
        finally:
            service.close()
        assert_released(scheduler)


def test_export_failure_after_sampling_does_not_cancel_other_batch_members(monkeypatch):
    from minisgl.disagg.service import Session, WorkerService

    with cpu_pinned_allocations():
        scheduler = cpu_scheduler("prefill", prefill_budget=32)
        scheduler.pd_service_config = {"batch_wait_ms": 0}
        service = WorkerService(scheduler, MockTransferBackend("p", MockFabric()), 0.1)
        original = scheduler.engine.kv_cache.export_tokens
        calls = 0

        def export(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected first request export failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(scheduler.engine.kv_cache, "export_tokens", export)
        failed, healthy = Session(), Session()
        for session in (failed, healthy):
            service.submit(session, {"op": "prefill", "request": request().to_object()})
        try:
            service.step()
            assert failed.outbox.get_nowait()["status"] == "handoff"
            assert failed.outbox.get_nowait()["status"] == "error"
            assert healthy.outbox.get_nowait()["status"] == "handoff"
            service.step()
            assert not healthy.cancelled.is_set()
            assert list(service.contexts) == [healthy]
            assert service.contexts[healthy].staging
        finally:
            service.close()
        assert_released(scheduler)
