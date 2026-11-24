"""Load admission and bucket selection plus the unchanged CPU P/D data paths."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import torch
from layered_reference import wait_for
from minisgl.disagg.control import StartRequest
from minisgl.disagg.router import NoHealthyWorker, WorkerPool
from minisgl.disagg.routing_policy import (
    AdmissionLimits,
    PolicyRouter,
    RoutingBucket,
    RoutingConfig,
    WorkerGroup,
    load_buckets,
    policy_cli_args,
    policy_config,
)
from pd_reference import reference_tokens
from test_pd_batching import fake_cuda  # noqa: F401
from test_pd_routing import running_cluster


def request(n=8, output=4):
    return StartRequest(str(uuid4()), 0, tuple(range(n)), output, True)


@pytest.fixture
def pool():
    pool = WorkerPool(["p0:1", "p1:1"], ["d0:1", "d1:1"])
    for node in pool.nodes:
        pool.apply_health(
            node,
            {
                "status": "ready",
                "role": node.role,
                "multiplex_version": 1,
                "layered_transfer": False,
                "layout": {"page_size": 4},
                "num_pages": 32,
                "max_sequence_tokens": 128,
                "load": {
                    "waiting_requests": 0,
                    "running_requests": 0,
                    "pending_prefill_tokens": 0,
                    "kv_tokens": 0,
                    "kv_capacity": 128,
                    "running_capacity": 8,
                },
            },
        )
    yield pool
    pool.close()


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_two_choices_prefers_less_loaded_worker_and_stale_reports_fall_back(pool, role):
    router = PolicyRouter(pool, seed=4)
    nodes = getattr(pool, role + "_pool")
    busy, idle = nodes
    busy.health["load"].update(waiting_requests=4, pending_prefill_tokens=20)
    select = router.select_prefill if role == "prefill" else router.select_decode
    for _ in range(20):
        assert select(request=request(), bucket=RoutingBucket()) is idle
    busy.load_updated_at = idle.load_updated_at = time.monotonic() - 20
    router.inflight[idle.worker_id] = 2
    assert select(request=request(), bucket=RoutingBucket()) is busy


def test_prefill_estimated_queue_precedes_token_count_and_decode_compares_kv_share(pool):
    router = PolicyRouter(pool, seed=4)
    p0, p1 = pool.prefill_pool
    p0.health["load"].update(pending_prefill_tokens=20, estimated_prefill_queue_ms=2)
    p1.health["load"].update(pending_prefill_tokens=10, estimated_prefill_queue_ms=5)
    assert router.select_prefill(request=request(), bucket=RoutingBucket()) is p0
    d0, d1 = pool.decode_pool
    d0.health["load"].update(kv_tokens=32, kv_capacity=64)
    d1.health["load"].update(kv_tokens=48, kv_capacity=128)
    assert router.select_decode(request=request(), bucket=RoutingBucket()) is d1
    d1.health["load"].pop("kv_capacity")
    assert router.select_decode(request=request(), bucket=RoutingBucket()) is d0


@pytest.mark.parametrize(
    "limits,load",
    [
        (AdmissionLimits(max_waiting_requests=2), {"waiting_requests": 2}),
        (AdmissionLimits(max_pending_prefill_tokens=16), {"pending_prefill_tokens": 16}),
        (AdmissionLimits(max_running_usage=0.5), {"running_requests": 4}),
        (AdmissionLimits(max_kv_usage=0.5), {"kv_tokens": 64}),
    ],
)
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_admission_caps_reject_at_boundary_for_both_roles(pool, limits, load, role):
    router = PolicyRouter(pool, config=RoutingConfig(admission=limits))
    nodes = getattr(pool, role + "_pool")
    nodes[0].health["load"].update(load)
    assert router.candidates(role, request(), RoutingBucket()) == [nodes[1]]
    nodes[0].health["load"] = {}  # Unknown metrics fail open, never become a fake zero.
    assert router.candidates(role, request(), RoutingBucket()) == nodes


def test_inflight_caps_are_atomic_and_per_worker_and_release_is_idempotence_checked(pool):
    router = PolicyRouter(
        pool, config=RoutingConfig(admission=AdmissionLimits(max_inflight_requests=1))
    )

    def acquire(_):
        try:
            return router.acquire_prefill(request=request(), bucket=RoutingBucket())
        except NoHealthyWorker:
            return None

    with ThreadPoolExecutor(max_workers=12) as executor:
        acquired = [n for n in executor.map(acquire, range(12)) if n is not None]
    assert len(acquired) == 2 and len({n.worker_id for n in acquired}) == 2
    for node in acquired:
        router.release(node)
    assert not router.inflight
    with pytest.raises(RuntimeError, match="twice"):
        router.release(acquired[0])


def test_bucket_context_bounds_membership_and_group_specific_caps(pool):
    short = RoutingBucket(
        "short",
        max_input_tokens=8,
        max_context_tokens=12,
        prefill=WorkerGroup(("p-0",), admission=AdmissionLimits(max_waiting_requests=1)),
    )
    long = RoutingBucket("long", max_input_tokens=64, max_context_tokens=128)
    router = PolicyRouter(pool, config=RoutingConfig(buckets=(long, short)))
    assert [b.id for b in router.buckets_for(request())] == ["short", "long"]
    assert router.buckets_for(request(output=5)) == [long]
    pool.prefill_pool[0].health["load"]["waiting_requests"] = 1
    assert router.candidates("prefill", request(), short) == []
    assert len(router.candidates("prefill", request(), long)) == 2
    with pytest.raises(ValueError, match="bucket"):
        router.buckets_for(request(129))
    with pytest.raises(ValueError, match="unknown"):
        PolicyRouter(
            pool, config=RoutingConfig(buckets=(RoutingBucket(prefill=WorkerGroup(("p-9",))),))
        )


def test_cache_affinity_and_balancing_count_this_requests_uncached_tokens(pool):
    config = RoutingConfig(policy="cache_aware", cache_min_tokens=1)
    router = PolicyRouter(pool, seed=9, config=config)
    p0, p1 = pool.prefill_pool
    assert (
        router.select_prefill(request=request(), bucket=RoutingBucket(), prefix_hits={"p-1": 4})
        is p1
    )
    p1.health["load"]["pending_prefill_tokens"] = 100
    assert (
        router.select_prefill(request=request(), bucket=RoutingBucket(), prefix_hits={"p-1": 4})
        is p1
    )
    router.config = replace(config, affinity_mode="balanced", affinity_load_gap=4)
    assert (
        router.select_prefill(request=request(), bucket=RoutingBucket(), prefix_hits={"p-1": 4})
        is p0
    )


def test_config_round_trip_and_invalid_values_fail_before_gpu_start(tmp_path):
    from minisgl.disagg.launcher import parser

    config_path = tmp_path / "buckets.json"
    config_path.write_text(
        json.dumps(
            {
                "buckets": [
                    {
                        "id": "short",
                        "max_input_tokens": 32,
                        "prefill": {
                            "worker_ids": ["p-0"],
                            "policy": "cache_aware",
                            "admission": {"max_kv_usage": 0.8},
                        },
                        "decode": {"worker_ids": ["d-0"]},
                    }
                ]
            }
        )
    )
    args = parser().parse_args(
        [
            "--model",
            "model",
            "--bucket-config",
            str(config_path),
            "--max-worker-inflight-requests",
            "4",
            "--max-waiting-requests",
            "8",
        ]
    )
    assert policy_config(args) == policy_config(
        parser().parse_args(["--model", "model", *policy_cli_args(args)])
    )
    assert load_buckets(config_path)[0].prefill.admission.max_kv_usage == 0.8
    for invalid in (0, -1, float("nan"), True):
        with pytest.raises(ValueError):
            AdmissionLimits(max_kv_usage=invalid)
    config_path.write_text('{"buckets":[{"id":"bad","decode":{"policy":"cache_aware"}}]}')
    with pytest.raises(ValueError):
        load_buckets(config_path)
    for field in ("min_input_tokens", "rank"):
        config_path.write_text(json.dumps({"buckets": [{field: None}]}))
        with pytest.raises(ValueError, match="null"):
            load_buckets(config_path)


def test_launcher_rejects_unknown_bucket_workers_before_spawning(monkeypatch, tmp_path, capsys):
    from minisgl.disagg import launcher

    path = tmp_path / "buckets.json"
    path.write_text('{"buckets":[{"prefill":{"worker_ids":["p-9"]}}]}')
    monkeypatch.setattr(launcher, "visible_gpus", lambda: ["0", "1"])
    popen = Mock(side_effect=AssertionError("must not spawn"))
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    with pytest.raises(SystemExit) as error:
        launcher.main(["--model", "/missing", "--bucket-config", str(path)])
    assert error.value.code == 2 and "unknown prefill" in capsys.readouterr().err
    popen.assert_not_called()


@pytest.mark.parametrize("mode", ["layered", "tcp"])
def test_default_load_routing_preserves_batching_tokens_and_cleans_occupancy(monkeypatch, mode):
    with running_cluster(monkeypatch, mode, routing_config=RoutingConfig()) as (
        client,
        services,
        _,
        _,
    ):
        barrier = threading.Barrier(6)

        def generate(i):
            barrier.wait(timeout=3)
            return client.generate((i + 1, 2, 3), 6, ignore_eos=True)

        with ThreadPoolExecutor(max_workers=6) as executor:
            actual = list(executor.map(generate, range(6)))
        assert actual == [reference_tokens((i + 1, 2, 3), 6) for i in range(6)]
        wait_for(lambda: all(not s.contexts for s in services))
        assert not client.router.inflight
        assert max(s.stats["max_prefill_batch"] for s in services[:2]) > 1
        assert max(s.stats["max_decode_batch"] for s in services[2:]) > 1
        for service in services:
            load = service.health()["load"]
            assert load["kv_tokens"] == 0 and load["kv_capacity"] == 128


@pytest.mark.parametrize("mode", ["layered", "tcp"])
def test_busy_decode_bucket_falls_back_before_prefill_and_refunds_reservations(monkeypatch, mode):
    buckets = (
        RoutingBucket(
            "short", max_input_tokens=8, prefill=WorkerGroup(("p-0",)), decode=WorkerGroup(("d-0",))
        ),
        RoutingBucket(
            "fallback",
            max_input_tokens=32,
            prefill=WorkerGroup(("p-1",)),
            decode=WorkerGroup(("d-0", "d-1")),
        ),
    )
    # Bucket overlap must not reserve d-0 twice with an already-used request id.
    config = RoutingConfig(buckets=buckets, decode_policy="round_robin")
    with running_cluster(monkeypatch, mode, routing_config=config) as (
        client,
        services,
        _,
        _,
    ):
        held = services[2].s.cache_manager._allocate(31)
        try:
            assert client.generate((1, 2, 3), 6, ignore_eos=True) == reference_tokens((1, 2, 3), 6)
            assert services[0].stats["max_prefill_batch"] == 0
            assert services[1].stats["max_prefill_batch"] > 0
            assert not client.router.inflight
        finally:
            services[2].s.cache_manager._free(held)


def test_prefix_query_runs_on_gpu_owner_without_reserving_a_model_row(monkeypatch):
    config = RoutingConfig(policy="cache_aware", cache_min_tokens=1)
    with running_cluster(monkeypatch, routing_config=config) as (client, services, _, _):
        source = services[1]
        seen = []

        def peek(ids):
            seen.append(threading.current_thread().ident)
            assert not source.contexts  # Query occurs before D reserve and P start.
            assert source.s.table_manager.available_size == 8
            return len(ids)

        monkeypatch.setattr(source.s.cache_manager.prefix_cache, "peek_prefix_len", peek)
        # Start-time measurement may occur after context creation, so restore after the query.
        original_start = source._start

        def start(*args):
            monkeypatch.setattr(
                source.s.cache_manager.prefix_cache, "peek_prefix_len", lambda ids: 0
            )
            return original_start(*args)

        monkeypatch.setattr(source, "_start", start)
        assert client.generate((1, 2, 3, 4), 5, ignore_eos=True) == reference_tokens(
            (1, 2, 3, 4), 5
        )
        assert seen and all(t != threading.current_thread().ident for t in seen)
        assert services[0].stats["max_prefill_batch"] == 0
        assert source.stats["max_prefill_batch"] > 0
        assert not client.router.inflight


def test_cancel_after_first_token_releases_router_counts_and_layer_leases(monkeypatch):
    with running_cluster(monkeypatch, routing_config=RoutingConfig()) as (client, services, _, _):
        iterator = client.stream((1, 2, 3), 8, ignore_eos=True)
        next(iterator)
        iterator.close()
        wait_for(lambda: all(not s.contexts for s in services))
        assert not client.router.inflight
        assert all(not s.backend._leases for s in services[2:])


def test_prefill_occupancy_is_released_before_client_finishes_reading_decode(monkeypatch):
    with running_cluster(monkeypatch, routing_config=RoutingConfig()) as (client, services, _, _):
        iterator = client.stream((1, 2, 3), 8, ignore_eos=True)
        try:
            first, second = next(iterator), next(iterator)
            assert [first, second] == reference_tokens((1, 2, 3), 8)[:2]
            assert not any(key.startswith("p-") for key in client.router.inflight)
            assert sum(client.router.inflight.values()) == 1
        finally:
            iterator.close()
        wait_for(lambda: all(not s.contexts for s in services))
        assert not client.router.inflight


def test_radix_routing_peek_does_not_split_nodes_or_touch_lru(monkeypatch):
    import minisgl.core as core
    import minisgl.kernel as kernel
    from minisgl.kvcache.radix_cache import RadixPrefixCache

    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1))

    def compare(a, b):
        for i, (x, y) in enumerate(zip(a.tolist(), b.tolist())):
            if x != y:
                return i
        return min(len(a), len(b))

    monkeypatch.setattr(kernel, "fast_compare_key", compare)
    cache = RadixPrefixCache(torch.device("cpu"))
    cache.insert_prefix(torch.tensor([1, 2, 3, 4]), torch.arange(4))
    node = next(iter(cache.root_node.children.values()))
    timestamp = node.timestamp
    size = cache.size_info
    assert cache.peek_prefix_len(torch.tensor([1, 2, 9])) == 2
    assert cache.peek_prefix_len(torch.tensor([1])) == 1
    assert node.length == 4 and not node.children and node.timestamp == timestamp
    assert cache.size_info == size
