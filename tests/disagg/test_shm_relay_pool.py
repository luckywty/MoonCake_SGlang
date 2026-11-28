import ctypes
import json
import multiprocessing
import signal
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from minisgl.disagg import launcher, numa
from minisgl.disagg import shm_relay_pool as shm
from minisgl.disagg.cluster_config import host_staging_plan, local_plan, shm_staging_plan
from minisgl.disagg.role import WorkerRole
from minisgl.disagg.worker import PDWorkerScheduler


def manifest(nodes=(0, 1), workers=(("p-0", 0), ("p-1", 1)), local=2, overflow=1):
    plan = host_staging_plan(
        max_seq_len=8,
        bytes_per_token=64,
        layers=4,
        layers_per_chunk=1,
        staging_slots=local + overflow,
        max_transfer_bytes=512,
        max_registered_bytes=65536,
        rows=4,
        prefill=True,
    )
    return shm.build_manifest(
        f"minisgl_test_{uuid4().hex}",
        workers,
        plan,
        bytes_per_token=64,
        layers=4,
        layers_per_chunk=1,
        local_slots=local,
        overflow_slots=overflow,
        numa_nodes=nodes,
    )


def _partition_writer(encoded, worker_id, pipe):
    pool = shm.SharedRelayPool(shm.RelayManifest.from_json(encoded))
    try:
        allocator = shm.RelayAllocator(pool, worker_id)
        p = allocator.partition
        for i in range(len(p.local_slots) + len(p.overflow_slots)):
            lease = allocator.acquire(worker_id, p.overflow_numa, elapsed_ms=100)
            pool.tensor(lease.slot_numa_id, lease.slot_id, 256).fill_(i + 1)
            allocator.ready(lease, copied=True)
        pipe.send([shm.asdict(lease) for lease in allocator.leases.values()])
    finally:
        pool.close()  # A child mapper must never unlink another process's SHM.
        pipe.close()


def _signal_owner(encoded, manifest_path, pipe):
    m = shm.RelayManifest.from_json(encoded)
    args, _ = launcher.parser().parse_known_args(["--model", "tiny"])
    args.shm_manifest_path = manifest_path
    supervisor = launcher.ClusterSupervisor(args, [], [], [], [])
    try:
        supervisor.relay = shm.SharedRelayPool(m, create=True, initialize=False)
        signal.signal(signal.SIGTERM, lambda *_: supervisor.stop.set())
        pipe.send("ready")
        supervisor.stop.wait(15)
    finally:
        supervisor.close()
        pipe.close()


@pytest.mark.parametrize(
    "n,m,topology,counts",
    [
        (2, 6, {str(i): i // 4 for i in range(8)}, {0: (1, 3), 1: (1, 3)}),
        (2, 2, {str(i): i // 2 for i in range(4)}, {0: (1, 1), 1: (1, 1)}),
        (1, 3, {str(i): i // 2 for i in range(4)}, {0: (1, 1), 1: (0, 2)}),
        (1, 3, {str(i): -1 for i in range(4)}, {0: (1, 3)}),
    ],
)
def test_simulated_topologies_mix_roles_symmetrically(monkeypatch, n, m, topology, counts):
    monkeypatch.setenv("MINISGL_GPU_NUMA_MAP", json.dumps(topology))
    args, _ = launcher.parser().parse_known_args(
        [
            "--model",
            "tiny",
            "--num-prefill-workers",
            str(n),
            "--num-decode-workers",
            str(m),
        ]
    )
    specs = local_plan(args, list(topology))
    for node, (p, d) in counts.items():
        assert sum(s.numa_id == node and s.role == "prefill" for s in specs) == p
        assert sum(s.numa_id == node and s.role == "decode" for s in specs) == d
    assert len({s.host_id for s in specs}) == 1
    assert all(s.host_id == numa.host_id() for s in specs)


def test_explicit_gpu_mapping_takes_precedence(monkeypatch):
    monkeypatch.setenv("MINISGL_GPU_NUMA_MAP", '{"0":0,"1":0,"2":1,"3":1}')
    args, _ = launcher.parser().parse_known_args(
        [
            "--model",
            "tiny",
            "--prefill-gpus",
            "3",
            "--num-decode-workers",
            "3",
        ]
    )
    specs = local_plan(args, ["0", "1", "2", "3"])
    assert [s.gpu for s in specs] == ["3", "0", "1", "2"]
    assert specs[0].numa_id == 1


def test_sysfs_discovery_unknown_node_and_missing_inventory(monkeypatch, tmp_path):
    path = tmp_path / "0000:41:00.0"
    path.mkdir()
    (path / "numa_node").write_text("1")
    monkeypatch.delenv("MINISGL_GPU_NUMA_MAP", raising=False)
    monkeypatch.setattr(numa.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        numa.subprocess,
        "check_output",
        lambda *a, **k: "0, GPU-abc, 00000000:41:00.0\n1, GPU-def, 00000000:42:00.0\n",
    )
    assert numa.gpu_numa_nodes(["GPU-abc", "1"], sysfs=tmp_path) == {"GPU-abc": 1, "1": 0}
    (path / "numa_node").write_text("-1")
    assert numa.gpu_numa_nodes(["0"], sysfs=tmp_path) == {"0": 0}
    with pytest.raises(ValueError, match="every visible GPU"):
        numa.gpu_numa_nodes(["0", "1"], override={"0": 0})


def test_local_wait_cross_socket_immediate_overflow_and_aba():
    m = manifest(workers=(("p-0", 0),))
    pool = shm.SharedRelayPool(m, create=True, initialize=False)
    try:
        a = shm.RelayAllocator(pool, "p-0")
        x, y = [a.acquire("r", 0) for _ in range(2)]
        assert x.slot_numa_id == y.slot_numa_id == 0
        assert a.acquire("r", 0, elapsed_ms=0.9, local_wait_ms=1) is None
        z = a.acquire("r", 0, elapsed_ms=1, local_wait_ms=1)
        assert z.slot_numa_id == 1
        with pytest.raises(ValueError, match="D2H"):
            a.ready(x, copied=False)
        a.ready(x, copied=True)
        with pytest.raises(ValueError, match="H2D"):
            a.release(x, consumer_complete=False)
        a.release(x, consumer_complete=True)
        new = a.acquire("next", 1)
        assert new.slot_id == x.slot_id and new.generation == x.generation + 1
        with pytest.raises(ValueError, match="ABA"):
            a.release(x, consumer_complete=True)
        a.release(z, consumer_complete=True)
        immediate = a.acquire("cross", 1, elapsed_ms=0, local_wait_ms=100)
        assert immediate.slot_numa_id == 1
        a.reclaim_request("r")
        assert pool.header(y.slot_numa_id, y.slot_id)[1] == shm.QUARANTINED
        a.reclaim_request("r", consumer_complete=True)
        assert pool.header(y.slot_numa_id, y.slot_id)[1] == shm.FREE
        assert len(a.leases) == 2  # Other requests' main and overflow leases are untouched.
    finally:
        pool.close()


def test_one_socket_has_no_overflow_and_never_overwrites():
    m = manifest(nodes=(0,), workers=(("p-0", 0),), overflow=8)
    assert m.partition("p-0").overflow_slots == ()
    pool = shm.SharedRelayPool(m, create=True, initialize=False)
    try:
        a = shm.RelayAllocator(pool, "p-0")
        for _ in range(2):
            assert a.acquire("r", 0) is not None
        assert a.acquire("r", 0, elapsed_ms=99999) is None
    finally:
        pool.close()


def test_multiple_processes_only_write_their_own_static_segments_and_owner_unlinks():
    m = manifest(workers=(("p-0", 0), ("p-1", 0), ("p-2", 1)))
    owner = shm.SharedRelayPool(m, create=True, initialize=False)
    children = []
    try:
        ctx = multiprocessing.get_context("spawn")
        for p in m.partitions:
            parent, child = ctx.Pipe(duplex=False)
            process = ctx.Process(target=_partition_writer, args=(m.to_json(), p.worker_id, child))
            process.start()
            child.close()
            children.append((process, parent))
        addresses = set()
        for process, parent in children:
            assert parent.poll(20)
            leases = parent.recv()
            for i, lease in enumerate(leases):
                key = lease["slot_numa_id"], lease["slot_id"]
                assert key not in addresses
                addresses.add(key)
                assert owner.header(*key) == (1, shm.READY)
                assert owner.tensor(*key, 256).tolist() == [i + 1] * 256
            process.join(10)
            assert process.exitcode == 0
            parent.close()
        mapper = shm.SharedRelayPool(m)
        mapper.close()
    finally:
        for process, parent in children:
            if process.is_alive():
                process.kill()
            process.join()
            parent.close()
        owner.close()
    assert all(not Path("/dev/shm", m.segment_name(n)[1:]).exists() for n in m.node_slots)
    with pytest.raises(OSError):
        shm.SharedRelayPool(m)


def test_registration_budget_includes_all_partitions_headers_and_page_padding():
    m = manifest()
    m.validate(budget=m.registered_bytes)
    with pytest.raises(ValueError, match="local/overflow"):
        m.validate(budget=m.registered_bytes - 1)
    model = SimpleNamespace(num_layers=4, num_kv_heads=1, head_dim=2)
    config = SimpleNamespace(
        model_config=model,
        dtype=torch.float32,
        max_seq_len=8,
        pd_max_transfer_bytes=512,
        pd_max_registered_bytes=m.registered_bytes,
        pd_layers_per_chunk=1,
        pd_staging_slots=2,
        pd_shm_overflow_slots=1,
        max_running_req=4,
        pd_host_id=m.host_id,
        pd_numa_id=0,
        worker_role=SimpleNamespace(value="prefill"),
        worker_id="p-0",
    )
    assert shm_staging_plan(config, manifest=m).group_bytes == m.group_bytes
    config.pd_host_id = "other-host"
    with pytest.raises(ValueError, match="identity"):
        shm_staging_plan(config, manifest=m)
    bad = replace(m, partitions=(replace(m.partitions[0], local_slots=(0, 0)), *m.partitions[1:]))
    with pytest.raises(ValueError, match="overlap"):
        bad.validate()


def test_mbind_precedes_first_touch(monkeypatch):
    m = manifest(nodes=(0,), workers=(("p-0", 0),))
    owner = shm.SharedRelayPool(m, create=True, initialize=False)
    calls, sent = [], []
    monkeypatch.setattr(shm, "bind_cpu", lambda node: calls.append("cpu"))
    monkeypatch.setattr(shm, "bind_memory", lambda *args: calls.append("mbind"))
    memset = ctypes.memset
    monkeypatch.setattr(
        shm.ctypes, "memset", lambda *args: (calls.append("touch"), memset(*args))[1]
    )
    try:
        shm._initialize(
            m.segment_name(0),
            m.segment_bytes(0),
            0,
            SimpleNamespace(send=sent.append, close=lambda: None),
        )
        assert calls == ["cpu", "mbind", "touch"] and sent == [None]
    finally:
        owner.close()


def test_pin_failure_is_explicit_and_rolls_back_previous_registration(monkeypatch):
    m = manifest()
    owner = shm.SharedRelayPool(m, create=True, initialize=False)
    calls = []

    def register(address, size, flags):
        calls.append((address, size, flags))
        return 0 if len(calls) == 1 else 2

    removed = []
    monkeypatch.setattr(
        torch.cuda,
        "cudart",
        lambda: SimpleNamespace(
            cudaHostRegister=register, cudaHostUnregister=lambda addr: (removed.append(addr), 0)[1]
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="RLIMIT_MEMLOCK"):
            owner.register_cuda()
        assert removed == [calls[0][0]] and not owner._registered
        assert calls[0][2] == 0
    finally:
        owner.close()


def test_supervisor_cleanup_unlinks_after_children_exit(tmp_path):
    m = manifest(nodes=(0,), workers=(("p-0", 0),))
    args, _ = launcher.parser().parse_known_args(["--model", "tiny"])
    args.shm_manifest_path = str(tmp_path / "manifest.json")
    Path(args.shm_manifest_path).write_text(m.to_json())
    supervisor = launcher.ClusterSupervisor(args, [], [], [], [])
    supervisor.relay = shm.SharedRelayPool(m, create=True, initialize=False)
    supervisor.close()
    supervisor.close()
    assert not Path(args.shm_manifest_path).exists()
    assert not Path("/dev/shm", m.segment_name(0)[1:]).exists()


@pytest.mark.parametrize("bad", ["budget", "host", "partition"])
def test_shm_budget_and_same_host_errors_precede_weights_and_engine(tmp_path, monkeypatch, bad):
    from minisgl.disagg import worker

    m = manifest()
    path = tmp_path / "pool.json"
    path.write_text(m.to_json())
    config = SimpleNamespace(
        worker_role=WorkerRole.PREFILL,
        tp_info=SimpleNamespace(size=1),
        use_dummy_weight=False,
        dtype=torch.float32,
        max_extend_tokens=8,
        max_running_req=4,
        max_seq_len=8,
        model_config=SimpleNamespace(num_layers=4, num_kv_heads=1, head_dim=2),
        pd_transfer_backend="shm_dram",
        pd_mooncake_transport_mode="gpu_direct",
        pd_shm_manifest=str(path),
        pd_host_id=m.host_id,
        pd_numa_id=0,
        worker_id="p-0",
        pd_max_transfer_bytes=512,
        pd_max_registered_bytes=m.registered_bytes,
        pd_layers_per_chunk=1,
        pd_staging_slots=2,
        pd_shm_overflow_slots=1,
        pd_shm_local_wait_ms=1,
        pd_cross_numa_penalty=4,
    )
    if bad == "budget":
        config.pd_max_registered_bytes = m.registered_bytes - 1
    elif bad == "host":
        config.pd_host_id = "other"
    else:
        config.worker_id = "p-99"

    def forbidden(*args, **kwargs):
        raise AssertionError("must fail before loading weights/constructing Engine")

    monkeypatch.setattr(worker, "checkpoint_fingerprint", forbidden)
    monkeypatch.setattr(worker.Scheduler, "__init__", forbidden)
    with pytest.raises(ValueError):
        PDWorkerScheduler(config)


def test_host_inventory_includes_numa_without_a_gpu():
    assert numa.host_numa_nodes(override="0,1") == [0, 1]
    assert numa.gpu_numa_nodes(["0", "1"], override={"0": 0, "1": 0}) == {"0": 0, "1": 0}
    m = manifest(nodes=(0, 1), workers=(("p-0", 0),))
    assert m.partition("p-0").overflow_numa == 1
    assert m.node_slots[1] == 1


def test_identical_slot_numbers_on_different_numa_segments_are_distinct():
    m = manifest(nodes=(0, 1), workers=(("p-0", 0),), local=2, overflow=2)
    p = m.partitions[0]
    p = replace(p, overflow_slots=p.local_slots)
    assert replace(m, partitions=(p,)).validate().node_slots == {0: 2, 1: 2}


def test_sigterm_owner_unlinks_segments_and_manifest(tmp_path):
    m = manifest()
    path = tmp_path / "relay.json"
    path.write_text(m.to_json())
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_signal_owner, args=(m.to_json(), str(path), child))
    try:
        process.start()
        child.close()
        assert parent.poll(15) and parent.recv() == "ready"
        process.terminate()
        process.join(15)
        assert process.exitcode == 0
        assert not path.exists()
        assert all(not Path("/dev/shm", m.segment_name(n)[1:]).exists() for n in m.node_slots)
    finally:
        if process.is_alive():
            process.kill()
        process.join()
        parent.close()
