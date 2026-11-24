"""Multi-P/multi-D Workers, real routing/multiplexing, CPU tensors, no OS sockets."""

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from layered_reference import CpuLayeredBackend, MemoryControl, wait_for
from minisgl.disagg import multiplex, mux_server
from minisgl.disagg import service as service_module
from minisgl.disagg.client import RemoteCapacityError
from minisgl.disagg.multiplex import WorkerConnection
from minisgl.disagg.routed_client import RoutedPDClient
from minisgl.disagg.router import WorkerPool
from minisgl.disagg.routing_policy import RoutingConfig
from minisgl.disagg.service import WorkerService
from minisgl.disagg.transfer import MockFabric, MockTransferBackend
from pd_reference import cpu_scheduler, reference_tokens
from test_pd_batching import fake_cuda  # noqa: F401
from test_pd_control_loop import MemoryChannel, MemoryListener, readable


@contextmanager
def running_cluster(monkeypatch, mode="layered", seed=12, routing_config=None):
    services, backends, threads, errors = [], [], [], []
    stop, fabric = threading.Event(), MockFabric()
    counts, frames = {}, []
    listeners, targets = {}, {}
    try:
        for index, role in enumerate(["prefill", "prefill", "decode", "decode"]):
            scheduler = cpu_scheduler(role, max_running_req=8, prefill_budget=32)
            scheduler.worker_id = f"{role[0]}-{index % 2}"
            scheduler.pd_service_config = {"batch_wait_ms": 15}
            scheduler.engine.delay = 0.005
            if mode == "layered":
                backend = CpuLayeredBackend(
                    None,
                    timeout=2,
                    max_transfer_bytes=16384,
                    max_registered_bytes=32768,
                    layers_per_chunk=1,
                    staging_slots=2,
                )
                backend.endpoint = f"127.0.0.1:{22100 + index}"
                backend.bind_device(scheduler.device)
                backend.configure_pipeline(scheduler)
            else:
                backend = MockTransferBackend(f"worker-data-{index}", fabric)
            backends.append(backend)
            service = WorkerService(scheduler, backend, 2)
            services.append(service)
            listeners[("127.0.0.1", 20100 + index)] = MemoryListener()
            if role == "decode":
                targets[backend.endpoint] = backend

        @contextmanager
        def data_connection(endpoint):
            yield MemoryControl(targets[endpoint])

        if mode == "layered":
            for backend in backends[:2]:
                backend._connection = data_connection
                backend._channel = lambda sock: sock

        class TracedChannel(MemoryChannel):
            def send(self, message):
                frames.append(message)
                return super().send(message)

        monkeypatch.setattr(multiplex, "Channel", TracedChannel)
        monkeypatch.setattr(service_module, "Channel", TracedChannel)
        for module in (multiplex, mux_server, service_module):
            monkeypatch.setattr(module, "select", SimpleNamespace(select=readable))

        def connect(address, timeout):
            counts[address] = counts.get(address, 0) + 1
            return listeners[address].connect()

        def run(service, listener):
            try:
                service.run(listener, stop)
            except BaseException as exc:
                errors.append(exc)

        for service, listener in zip(services, listeners.values()):
            thread = threading.Thread(target=run, args=(service, listener), daemon=True)
            threads.append(thread)
            thread.start()
        prefill = [f"127.0.0.1:{20100 + i}" for i in range(2)]
        decode = [f"127.0.0.1:{20102 + i}" for i in range(2)]
        pool = WorkerPool(
            prefill,
            decode,
            timeout=2,
            health_interval=0.1,
            health_timeout=0.5,
            connection_factory=lambda endpoint, **kwargs: WorkerConnection(
                endpoint, connect=connect, **kwargs
            ),
        )
        client = RoutedPDClient(
            prefill,
            decode,
            pool=pool,
            routing_seed=seed,
            ready_timeout=2,
            routing_config=routing_config or RoutingConfig(policy="random", decode_policy="random"),
        )
        try:
            client.start()
            yield client, services, counts, frames
        finally:
            client.close()
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=8)
        assert all(not thread.is_alive() for thread in threads)
        for service in services:
            service.close()
        for backend in reversed(backends):
            if mode == "layered":
                backend.cleanup_fake()
        assert not errors


@pytest.mark.parametrize("mode", ["layered", "tcp"])
def test_random_many_to_many_batching_uses_one_connection_per_worker(monkeypatch, caplog, mode):
    caplog.set_level("INFO", logger="minisgl.disagg.router")
    prompts = [(i, i + 1) for i in range(1, 9)]
    expected = [reference_tokens(prompt, 6) for prompt in prompts]
    with running_cluster(monkeypatch, mode) as (client, services, counts, frames):
        for source in services[:2]:
            source.batch_wait = 0.05
        barrier = threading.Barrier(len(prompts))

        def generate(index):
            barrier.wait(timeout=2)
            return client.generate(prompts[index], 6, ignore_eos=True)

        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            assert list(executor.map(generate, range(len(prompts)))) == expected
        wait_for(lambda: all(not s.contexts for s in services))
        client.pool.probe_all()
        assert len(counts) == 4 and set(counts.values()) == {1}
        assert max(s.stats["max_prefill_batch"] for s in services[:2]) > 1
        assert max(s.stats["max_decode_batch"] for s in services[2:]) > 1
        routes = [
            json.loads(record.message.split("pd_route ", 1)[1])
            for record in caplog.records
            if "pd_route " in record.message and '"event":"route"' in record.message
        ]
        assert len(routes) == len(prompts)
        assert len({item["request_id"] for item in routes}) == len(prompts)
        assert {item["prefill_id"] for item in routes} <= {"p-0", "p-1"}
        assert {item["decode_id"] for item in routes} <= {"d-0", "d-1"}
        # Every wire frame, including connection negotiation, carries request_id.
        assert frames and all("request_id" in frame for frame in frames)
        assert all(
            frame.get("op") == "multiplex"
            or frame.get("status") == "multiplex"
            or set(frame) == {"request_id", "message", "terminal"}
            for frame in frames
        )
        assert all(s.s.health()["free_pages"] == 32 for s in services)


def _seed_with_first_decode_zero():
    for seed in range(32):
        rng = random.Random(seed)
        rng.choice([0, 1])
        if rng.choice([0, 1]) == 0:
            return seed
    raise AssertionError("missing routing seed")


@pytest.mark.parametrize("mode", ["layered", "tcp"])
def test_busy_reserve_reselects_another_d_before_prefill_once(monkeypatch, caplog, mode):
    caplog.set_level("INFO", logger="minisgl.disagg.router")
    with running_cluster(monkeypatch, mode, _seed_with_first_decode_zero()) as (
        client,
        services,
        _,
        _,
    ):
        busy = services[2]
        held = busy.s.cache_manager._allocate(31)
        calls = []
        for service in services[:2]:
            original = service.s.engine.forward_batch

            def forward(*args, _original=original, _service=service):
                calls.append(_service.s.worker_id)
                return _original(*args)

            monkeypatch.setattr(service.s.engine, "forward_batch", forward)
        request_id = str(uuid4())
        try:
            actual = client.generate((1, 2, 3), 6, ignore_eos=True, request_id=request_id)
            assert actual == reference_tokens((1, 2, 3), 6)
            assert len(calls) == 1
            routes = [
                json.loads(record.message.split("pd_route ", 1)[1])
                for record in caplog.records
                if "pd_route " in record.message
            ]
            assert [(r["event"], r["decode_id"]) for r in routes] == [
                ("reserve_retry", "d-0"),
                ("route", "d-1"),
            ]
            assert {r["request_id"] for r in routes} == {request_id}
            wait_for(lambda: all(not s.contexts for s in services))
        finally:
            import torch

            busy.s.cache_manager.free_slots = torch.cat([busy.s.cache_manager.free_slots, held])


def test_all_busy_decoders_fail_without_running_any_prefill(monkeypatch):
    with running_cluster(monkeypatch) as (client, services, _, _):
        held = [d.s.cache_manager._allocate(32) for d in services[2:]]
        try:
            with pytest.raises(RemoteCapacityError):
                client.generate((1, 2), 6, ignore_eos=True)
            assert all(s.stats["max_prefill_batch"] == 0 for s in services[:2])
            wait_for(lambda: all(not s.contexts for s in services))
        finally:
            for d, pages in zip(services[2:], held):
                d.s.cache_manager.free_slots = pages


def test_health_monitor_removes_retired_or_quarantined_worker_and_readds_recovered(monkeypatch):
    with running_cluster(monkeypatch) as (client, services, _, _):
        d = services[2]
        original = d.health
        status = {"retired": False, "quarantined": False}

        def health():
            result = original()
            if status["retired"]:
                result["transfer"]["status"] = result["status"] = "unavailable"
            if status["quarantined"]:
                result["transfer"]["mooncake"]["quarantined_bytes"] = 1
            return result

        monkeypatch.setattr(d, "health", health)
        for key in ("retired", "quarantined"):
            status[key] = True
            wait_for(lambda: [node.worker_id for node in client.pool.healthy("decode")] == ["d-1"])
            assert client.generate((1, 2), 4, ignore_eos=True) == reference_tokens((1, 2), 4)
            status[key] = False
            wait_for(lambda: len(client.pool.healthy("decode")) == 2)


def test_cancel_one_multiplexed_request_does_not_close_peer_channels(monkeypatch):
    with running_cluster(monkeypatch) as (client, services, counts, _):
        iterator = client.stream((1, 2), 20, ignore_eos=True)
        next(iterator)
        iterator.close()
        assert client.generate((3, 4), 6, ignore_eos=True) == reference_tokens((3, 4), 6)
        wait_for(lambda: all(not s.contexts for s in services))
        assert set(counts.values()) == {1}
        assert all(not b._leases for b in (s.backend for s in services[2:]))


def test_request_id_mismatch_is_rejected_without_contaminating_another_stream(monkeypatch):
    with running_cluster(monkeypatch, "tcp") as (client, services, _, _):
        node = client.pool.decode_pool[0]
        channel = client.pool.connect(node).open(str(uuid4()))
        try:
            from minisgl.disagg.control import StartRequest

            request = StartRequest(str(uuid4()), 0, (1, 2), 3, True)
            channel.send({"op": "reserve", "request": request.to_object()})
            error = channel.receive()
            assert error["status"] == "error" and "request_id" in error["error"]
        finally:
            channel.close()
        assert client.generate((1, 2), 3, ignore_eos=True) == reference_tokens((1, 2), 3)


def test_two_prefill_workers_concurrently_fan_out_to_both_decode_workers(monkeypatch):
    # Fix only node selection here so every required data-plane edge is exercised;
    # seeded random selection itself is checked separately and in the tests above.
    with running_cluster(monkeypatch) as (client, services, counts, _):
        for source in services[:2]:
            original = source._prefill

            def wait_for_batch(_source=source, _original=original):
                if len(_source.s.prefill_manager.pending_list) >= 2:
                    _original()

            monkeypatch.setattr(source, "_prefill", wait_for_batch)
        route = threading.local()
        monkeypatch.setattr(
            client.router, "select_prefill", lambda **kwargs: client.pool.prefill_pool[route.p]
        )
        monkeypatch.setattr(
            client.router,
            "select_decode",
            lambda exclude=(), **kwargs: client.pool.decode_pool[route.d],
        )
        barrier = threading.Barrier(4)

        def generate(pair):
            route.p, route.d = pair
            barrier.wait(timeout=2)
            prompt = (route.p + 1, route.d + 3)
            return client.generate(prompt, 8, ignore_eos=True), reference_tokens(prompt, 8)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(generate, [(0, 0), (0, 1), (1, 0), (1, 1)]))
        assert all(actual == expected for actual, expected in results)
        assert all(s.stats["max_prefill_batch"] == 2 for s in services[:2])
        wait_for(lambda: all(not s.contexts for s in services))
        assert all(not s.backend._leases for s in services[2:])
        assert set(counts.values()) == {1}


def test_connection_failure_after_first_token_does_not_replay_prefill(monkeypatch):
    with running_cluster(monkeypatch, "tcp") as (client, services, _, _):
        calls = []
        for service in services[:2]:
            original = service.s.engine.forward_batch

            def forward(*args, _original=original):
                calls.append(True)
                return _original(*args)

            monkeypatch.setattr(service.s.engine, "forward_batch", forward)
        iterator = client.stream((1, 2, 3), 6, ignore_eos=True)
        next(iterator)
        active = next(node for node in client.pool.prefill_pool if node.connection.requests)
        active.connection.close()
        with pytest.raises(ConnectionError):
            next(iterator)
        iterator.close()
        wait_for(lambda: all(not s.contexts for s in services))
        assert len(calls) == 1


def test_cancel_is_queued_behind_a_frame_writer_and_health_has_a_deadline(monkeypatch):
    from minisgl.disagg.control import StartRequest

    with running_cluster(monkeypatch, "tcp") as (client, services, _, _):
        connection = client.pool.connect(client.pool.decode_pool[0])
        request = StartRequest(str(uuid4()), 0, (1, 2), 6, True)
        channel = connection.open(request.transfer_id)
        channel.send({"op": "reserve", "request": request.to_object()})
        assert channel.receive()["status"] == "reserved"
        connection.write_lock.acquire()
        try:
            channel.close()
            assert request.transfer_id in connection.pending_cancels
            health = connection.open(str(uuid4()))
            try:
                with pytest.raises(TimeoutError, match="writer"):
                    health.send({"op": "health"}, timeout=0.01)
            finally:
                health.close()
            assert not connection.stopped.is_set()
        finally:
            connection.write_lock.release()
        wait_for(lambda: not services[2].contexts and not connection.pending_cancels)
        assert client.generate((1, 2), 3, ignore_eos=True) == reference_tokens((1, 2), 3)


def test_unserializable_request_does_not_damage_the_shared_connection(monkeypatch):
    with running_cluster(monkeypatch, "tcp") as (client, _, _, _):
        connection = client.pool.connect(client.pool.prefill_pool[0])
        channel = connection.open(str(uuid4()))
        try:
            with pytest.raises(ValueError):
                channel.send({"op": "prefill", "invalid": float("nan")})
            assert not connection.stopped.is_set()
        finally:
            channel.close()
        assert client.generate((1, 2), 3, ignore_eos=True) == reference_tokens((1, 2), 3)


def test_router_rejects_duplicate_active_request_id_across_the_entire_pool(monkeypatch):
    with running_cluster(monkeypatch, "tcp") as (client, services, _, _):
        request_id = str(uuid4())
        iterator = client.stream((1, 2), 6, ignore_eos=True, request_id=request_id)
        next(iterator)
        try:
            assert client.active_routes[request_id] is not None
            with pytest.raises(ValueError, match="already active"):
                client.generate((3, 4), 6, ignore_eos=True, request_id=request_id)
        finally:
            iterator.close()
        wait_for(lambda: all(not s.contexts for s in services))
        assert not client.active_routes


def test_health_round_pins_model_identity_across_worker_changes(monkeypatch):
    with running_cluster(monkeypatch, "tcp") as (client, services, _, _):
        original = services[0].health

        def changed_model():
            result = original()
            result["layout"] = {**result["layout"], "model_fingerprint": "b" * 64}
            return result

        monkeypatch.setattr(services[0], "health", changed_model)
        client.pool.probe_all()
        assert [node.worker_id for node in client.pool.healthy("prefill")] == ["p-1"]
        assert len(client.pool.healthy("decode")) == 2
        assert client.generate((1, 2), 3, ignore_eos=True) == reference_tokens((1, 2), 3)
