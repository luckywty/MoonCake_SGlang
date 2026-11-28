from dataclasses import asdict, replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from minisgl.disagg.cluster_config import (
    host_staging_plan,
    main_registered_budget,
    validate_shared_history,
)
from minisgl.disagg.control import StartRequest
from minisgl.disagg.dram_protocol import DramConfig
from minisgl.disagg.numa import host_id
from minisgl.disagg.prefix_hash import PrefixHashSpec, prepare_prefix
from minisgl.disagg.protocol import KVLayout
from minisgl.disagg.role import WorkerRole
from minisgl.disagg.router import NoHealthyWorker, WorkerPool
from minisgl.disagg.routing_policy import PolicyRouter, RoutingBucket, RoutingConfig
from minisgl.disagg.shared_dram_pool import SharedDramManifest, SharedPage, build_shared_manifest
from minisgl.disagg.shm_relay_pool import build_manifest
from minisgl.disagg.worker import PDWorkerScheduler
from minisgl.distributed.info import DistributedInfo
from minisgl.server.args import ServerArgs, parse_args
from test_pd_shared_history import history


@pytest.fixture
def configuration(monkeypatch, tmp_path):
    model = SimpleNamespace(num_layers=2, num_kv_heads=1, head_dim=8)
    monkeypatch.setattr(ServerArgs, "model_config", property(lambda self: model))
    manifest = SharedDramManifest(
        f"history_{uuid4().hex}",
        host_id(),
        512,
        4,
        "default",
        {0: 1, 1: 1},
        {"p-0": 0, "d-0": 1},
        "/tmp/shared-history-test.sock",
        uuid4().hex,
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.to_json())
    config = ServerArgs(
        model_path="/missing",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.float32,
        worker_role=WorkerRole.PREFILL,
        worker_id="p-0",
        max_seq_len_override=64,
        pd_dram=DramConfig(capacity_bytes=8192),
        pd_dram_cache_mode="shared_shm",
        pd_shared_dram_manifest=str(path),
        pd_numa_id=0,
        pd_host_id=host_id(),
        page_size=4,
        pd_max_transfer_bytes=4096,
        pd_max_registered_bytes=8192,
    )
    return config, manifest


@pytest.mark.parametrize(
    "change",
    [
        {"pd_max_registered_bytes": 8191},
        {"pd_dram": DramConfig(capacity_bytes=4096)},
        {"pd_host_id": "other"},
        {"worker_id": "p-unknown"},
        {"page_size": 1},
        {"pd_cache_namespace": "other"},
        {"pd_shared_dram_manifest": None},
    ],
)
def test_bad_history_settings_fail_before_weights_or_engine(configuration, monkeypatch, change):
    from minisgl.disagg import worker

    config, _ = configuration

    def forbidden(*a, **k):
        raise AssertionError("must fail before weights or Engine")

    monkeypatch.setattr(worker, "checkpoint_fingerprint", forbidden)
    monkeypatch.setattr(worker.Scheduler, "__init__", forbidden)
    with pytest.raises(ValueError):
        PDWorkerScheduler(replace(config, **change))


def test_combined_registration_budget_counts_relay_and_history(configuration, tmp_path):
    config, history_manifest = configuration
    plan = host_staging_plan(
        max_seq_len=64,
        bytes_per_token=128,
        layers=2,
        layers_per_chunk=1,
        staging_slots=3,
        max_transfer_bytes=4096,
        max_registered_bytes=65536,
        rows=8,
        prefill=True,
    )
    relay = build_manifest(
        f"relay_{uuid4().hex}",
        [("p-0", 0)],
        plan,
        bytes_per_token=128,
        layers=2,
        layers_per_chunk=1,
        local_slots=2,
        overflow_slots=1,
        numa_nodes=(0, 1),
    )
    path = tmp_path / "relay.json"
    path.write_text(relay.to_json())
    total = relay.registered_bytes + history_manifest.registered_bytes
    config = replace(
        config,
        pd_transfer_backend="shm_dram",
        pd_shm_manifest=str(path),
        pd_staging_slots=2,
        pd_max_registered_bytes=total,
    )
    assert validate_shared_history(config, check_segments=False) == history_manifest
    assert main_registered_budget(config) == relay.registered_bytes
    with pytest.raises(ValueError, match="budget"):
        validate_shared_history(
            replace(config, pd_max_registered_bytes=total - 1), check_segments=False
        )


def test_combined_mooncake_host_sender_ring_is_checked_before_spawning(configuration):
    config, manifest = configuration
    config = replace(
        config,
        pd_transfer_backend="mooncake",
        pd_mooncake_transport_mode="host_dram",
        pd_staging_slots=3,
        pd_layers_per_chunk=1,
        pd_max_registered_bytes=manifest.registered_bytes + 4096,
    )
    with pytest.raises(ValueError, match="sender ring"):
        validate_shared_history(config, check_segments=False)
    validate_shared_history(
        replace(config, pd_max_registered_bytes=manifest.registered_bytes + 6144),
        check_segments=False,
    )


def test_dynamic_numa_budgets_and_dev_shm_space(monkeypatch):
    m = build_shared_manifest(
        name="history_dynamic",
        page_bytes=512,
        page_size=4,
        capacity_bytes=12288,
        numa_nodes=(0, 2, 4),
        worker_numa={"p-0": 0, "d-0": 2},
        directory_socket="/tmp/dynamic.sock",
        secret="a" * 32,
        node_bytes={"0": 4096, "2": 4096, "4": 4096},
    )
    assert m.registered_bytes == 12288 and set(m.node_slots) == {0, 2, 4}
    monkeypatch.setattr(
        "minisgl.disagg.shared_dram_pool.os.statvfs",
        lambda _: SimpleNamespace(f_bavail=12287, f_frsize=1),
    )
    with pytest.raises(ValueError, match="/dev/shm"):
        m.check_space()


def test_shared_mode_requires_capacity_and_keeps_private_host_te_separate():
    for extra in ([], ["--pd-dram-cache-bytes", "8192", "--pd-dram-remote-pull"]):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--model",
                    "tiny",
                    "--worker-role",
                    "prefill",
                    "--dtype",
                    "float32",
                    "--pd-dram-cache-mode",
                    "shared_shm",
                    *extra,
                ]
            )


@pytest.mark.parametrize("affinity", ["prefer", "balanced"])
def test_router_cost_prefers_p_local_to_shared_prefix_and_keeps_hints_non_authoritative(affinity):
    layout = KVLayout("a" * 64, "float32", 2, 1, 8, 4)
    spec = PrefixHashSpec.from_layout(asdict(layout))
    tokens = tuple(range(1, 14))
    prepared = prepare_prefix(tokens, spec)
    with history(layout) as (manifest, path, _, directory, _):
        for digest in tuple(prepared.digests()):
            p = SharedPage.from_object(directory.allocate_write(digest, 1, "seed")["page"])
            directory.commit_write(p.numa_id, p.slot_id, p.generation, p.page_hash)
        pool = WorkerPool(["p0:1", "p1:1"], ["d0:1", "d1:1"])
        try:
            config = RoutingConfig(
                policy="cache_aware",
                cache_min_tokens=1,
                affinity_mode=affinity,
                pd_shared_dram_remote_h2d_factor=2,
            )
            pool.configure_prefix_index(config)
            for i, node in enumerate(pool.nodes):
                pool.apply_health(
                    node,
                    {
                        "status": "ready",
                        "role": node.role,
                        "multiplex_version": 1,
                        "handoff_version": 2,
                        "decode_prefix_reuse": True,
                        "layered_transfer": False,
                        "transfer_mode": "tcp",
                        "layout": asdict(layout),
                        "num_pages": 64,
                        "numa_id": i % 2,
                        "host_id": host_id(),
                        "shared_dram": {
                            "mode": "shared_shm",
                            "host_id": host_id(),
                            "pool": manifest.name,
                            "manifest": str(path),
                            "spec": spec.manifest,
                        },
                        "load": {
                            "waiting_requests": 0,
                            "running_requests": 0,
                            "pending_prefill_tokens": 0,
                        },
                    },
                )
                assert node.healthy
            hints = pool.shared_history.hits(pool.prefill_pool, prepared, {}, config)
            assert hints["p-0"].total == hints["p-1"].total == 12
            assert hints["p-0"].restore_cost_ms == hints["p-1"].restore_cost_ms * 2
            assert not directory.leases
            router = PolicyRouter(pool, config=config)
            request = StartRequest(str(uuid4()), 0, tokens, 2, True)
            assert (
                router.select_prefill(
                    request=request, bucket=RoutingBucket(), prefix_hits=hints
                ).worker_id
                == "p-1"
            )
            # Disconnected/different shared pools cannot be paired even on TCP.
            for node in pool.decode_pool:
                node.health["shared_dram"]["pool"] = "other-pool"
            with pytest.raises(NoHealthyWorker):
                router.select_prefill(request=request, bucket=RoutingBucket(), prefix_hits=hints)
        finally:
            pool.close()
