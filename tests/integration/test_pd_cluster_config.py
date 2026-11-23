import os
import random
import signal
import subprocess
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from minisgl.disagg import launcher
from minisgl.disagg.cluster_config import local_plan, remote_addresses
from minisgl.disagg.mooncake_runtime import MooncakeRuntime
from minisgl.disagg.router import RandomRouter, WorkerPool
from minisgl.disagg.transfer import TransferBackendUnavailable


def arguments(*extra):
    args, rest = launcher.parser().parse_known_args(["--model", "/models/tiny", *extra])
    assert not rest
    return args


def test_local_gpu_identity_and_all_ports_are_isolated():
    args = arguments(
        "--num-prefill-workers", "2", "--num-decode-workers", "2", "--prefill-gpus", "3,1"
    )
    plan = local_plan(args, ["0", "1", "2", "3"])
    assert [s.worker_id for s in plan] == ["p-0", "p-1", "d-0", "d-1"]
    assert [s.gpu for s in plan] == ["3", "1", "0", "2"]
    ports = [
        p for s in plan for p in (s.control_port, s.control_port + 1, s.mooncake_port, s.data_port)
    ]
    assert len(set(ports)) == 16 and args.port not in ports
    commands = [launcher.worker_command(s, args, []) for s in plan]
    assert [env["CUDA_VISIBLE_DEVICES"] for _, env in commands] == [s.gpu for s in plan]


@pytest.mark.parametrize(
    "flags,visible",
    [
        (["--num-prefill-workers", "0"], ["0", "1"]),
        (["--num-decode-workers", "-1"], ["0", "1"]),
        (["--num-prefill-workers", "2"], ["0", "1"]),
        (["--prefill-gpus", "0", "--decode-gpus", "0"], ["0", "1"]),
        (["--prefill-gpus", "9"], ["0", "1"]),
        (["--prefill-gpus", "0,1"], ["0", "1", "2"]),
        (["--base-mooncake-port", "1919"], ["0", "1"]),
        (["--pd-transfer-backend", "mooncake", "--base-mooncake-port", "40000"], ["0", "1"]),
        (["--base-control-port", "65534"], ["0", "1"]),
        (["--worker-host", "0.0.0.0"], ["0", "1"]),
        (["--prefill-numa-nodes", "-1"], ["0", "1"]),
        (["--prefill-rdma-devices", "mlx5_0,mlx5_1"], ["0", "1"]),
    ],
)
def test_invalid_cluster_resources_fail_before_spawning(flags, visible):
    with pytest.raises(ValueError):
        local_plan(arguments(*flags), visible)


def test_remote_address_counts_validate_without_a_gpu_inventory():
    args = arguments("--prefill-addrs", "p1:1919,p2:1919", "--decode-addrs", "d1:1919")
    assert remote_addresses(args) == (["p1:1919", "p2:1919"], ["d1:1919"])
    args.num_decode_workers = 2
    with pytest.raises(ValueError, match="count"):
        remote_addresses(args)
    args.num_decode_workers = 1
    args.decode_addrs = ["p1:1919"]
    with pytest.raises(ValueError, match="unique"):
        remote_addresses(args)


def test_numa_hca_and_fixed_native_port_are_passed_per_child(monkeypatch):
    args = arguments(
        "--pd-transfer-backend",
        "mooncake",
        "--prefill-numa-nodes",
        "0",
        "--decode-numa-nodes",
        "1",
        "--prefill-rdma-devices",
        "mlx5_0+mlx5_1",
        "--decode-rdma-devices",
        "mlx5_2",
    )
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/numactl")
    for spec in local_plan(args, ["0", "1"]):
        command, env = launcher.worker_command(spec, args, ["--max-running-requests", "8"])
        assert command[:3] == [
            "/usr/bin/numactl",
            f"--cpunodebind={spec.numa_node}",
            f"--membind={spec.numa_node}",
        ]
        assert env["MC_MIN_RPC_PORT"] == env["MC_MAX_RPC_PORT"] == str(spec.mooncake_port)
        assert command[command.index("--pd-mooncake-port") + 1] == str(spec.mooncake_port)
        assert command[command.index("--pd-mooncake-device") + 1] == spec.rdma_device


def test_bad_tp_is_rejected_before_any_child_is_created(monkeypatch):
    monkeypatch.setattr(launcher, "visible_gpus", lambda: ["0", "1", "2", "3"])
    popen = Mock(side_effect=AssertionError("must not spawn"))
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    with pytest.raises(SystemExit) as error:
        launcher.main(["--model", "/missing", "--num-prefill-workers", "1", "--tp-size", "2"])
    assert error.value.code == 2
    popen.assert_not_called()


def test_startup_failure_reaps_children_already_started(monkeypatch):
    args = arguments("--no-gateway")
    specs = local_plan(args, ["0", "1"])
    child = Mock(pid=1234567)
    child.poll.return_value = None
    popen = Mock(side_effect=[child, RuntimeError("injected failed spawn")])
    killed = []
    monkeypatch.setattr(launcher.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    supervisor = launcher.ClusterSupervisor(
        args, specs, [specs[0].endpoint], [specs[1].endpoint], [], popen=popen
    )
    with pytest.raises(RuntimeError, match="failed spawn"):
        supervisor.run()
    assert killed == [(child.pid, signal.SIGTERM)]
    child.wait.assert_called_once()


def test_shutdown_kills_only_owned_process_groups_after_deadline(monkeypatch):
    args = arguments("--no-gateway", "--shutdown-timeout", "0.01")
    child = Mock(pid=1234567)
    child.poll.return_value = None
    child.wait.side_effect = [subprocess.TimeoutExpired("worker", 0.01), 0]
    signals = []
    monkeypatch.setattr(launcher.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    supervisor = launcher.ClusterSupervisor(args, [], [], [], [])
    supervisor.children = [("p-0", child, "prefill")]
    supervisor.close()
    assert signals == [(child.pid, signal.SIGTERM), (child.pid, signal.SIGKILL)]


def test_seeded_router_draws_from_current_healthy_candidates_only():
    pool = WorkerPool(["p0:1", "p1:1"], ["d0:1", "d1:1"])
    try:
        for node in pool.nodes:
            pool.apply_health(
                node,
                {
                    "status": "ready",
                    "role": node.role,
                    "multiplex_version": 1,
                    "layered_transfer": False,
                    "layout": {},
                },
            )
        router, rng = RandomRouter(pool, 57), random.Random(57)
        for _ in range(12):
            assert router.select_prefill() is rng.choice(pool.prefill_pool)
            assert router.select_decode() is rng.choice(pool.decode_pool)
        pool.mark_unhealthy(pool.decode_pool[0], "retired")
        assert all(router.select_decode() is pool.decode_pool[1] for _ in range(8))
    finally:
        pool.close()


@pytest.mark.parametrize("actual", [22501, 22502])
def test_native_rpc_port_range_is_configured_and_verified(monkeypatch, actual):
    from minisgl.disagg import mooncake_runtime as module

    calls = []

    class Engine:
        def initialize(self, *args):
            calls.append((os.environ["MC_MIN_RPC_PORT"], os.environ["MC_MAX_RPC_PORT"]))
            return 0

        def get_rpc_port(self):
            return actual

        def register_memory(self, *args):
            return 0

        def unregister_memory(self, *args):
            return 0

        def submit_scatter_write(self, *args):
            pass

    native = ModuleType("fake_multi_pd_engine")
    native.SUPPORT_CUDA = True
    native.TransferEngine = Engine
    native.ScatterTransferCompletionStatus = SimpleNamespace(
        COMPLETED=1, FAILED_DRAINED=2, COMPLETION_UNKNOWN=3
    )
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(
        module, "CudaTransfer", lambda device: SimpleNamespace(activate=nullcontext, info={})
    )
    monkeypatch.setattr(
        MooncakeRuntime, "_configure_rdma", staticmethod(lambda device: ["fake_hca"])
    )
    monkeypatch.setenv("MC_MIN_RPC_PORT", "15000")
    monkeypatch.setenv("MC_MAX_RPC_PORT", "17000")
    runtime = MooncakeRuntime("127.0.0.1", rpc_port=22501, engine_module=native.__name__)
    try:
        if actual == 22501:
            runtime.bind_device(torch.device("cuda:0"))
            assert runtime.session == "127.0.0.1:22501"
        else:
            with pytest.raises(TransferBackendUnavailable, match="ignored"):
                runtime.bind_device(torch.device("cuda:0"))
        assert calls == [("22501", "22501")]
    finally:
        runtime.close()
