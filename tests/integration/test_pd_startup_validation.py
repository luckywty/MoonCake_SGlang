"""Model/health/port preflight checks, without a model download, GPU or OS sockets."""

import copy
import logging
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import torch
from minisgl.disagg import launcher
from minisgl.disagg.cluster_config import local_plan, validate_cluster_ports, validate_port_bindings
from minisgl.disagg.dram_protocol import MAX_DRAM_PULL_PAGES, DramConfig
from minisgl.disagg.dram_service import DramServiceMixin
from minisgl.disagg.prefix_hash import PrefixHashSpec, prepare_prefix
from minisgl.disagg.protocol import KVLayout
from minisgl.disagg.role import WorkerRole
from minisgl.disagg.router import WorkerPool
from minisgl.disagg.routing_policy import RoutingConfig
from minisgl.disagg.tier_index import TierIndex, TierPublisher
from minisgl.disagg.worker import PDWorkerScheduler
from minisgl.scheduler import Scheduler
from minisgl.server.args import parse_args


@pytest.mark.parametrize(
    "backend,mode,valid",
    [
        ("tcp", "host_dram", False),
        ("mooncake", "host_dram", True),
        ("mooncake", "gpu_direct", True),
    ],
)
def test_main_transport_is_explicit_and_host_is_not_a_tcp_fallback(
    monkeypatch, backend, mode, valid
):
    load = Mock(side_effect=AssertionError("model load during flag validation"))
    monkeypatch.setattr("minisgl.utils.cached_load_hf_config", load)
    flags = [
        "--model",
        "/unused",
        "--dtype",
        "bfloat16",
        "--worker-role",
        "prefill",
        "--pd-transfer-backend",
        backend,
        "--pd-mooncake-transport-mode",
        mode,
    ]
    if valid:
        args, _ = parse_args(flags)
        assert args.pd_mooncake_transport_mode == mode
        if mode == "host_dram":
            assert args.pd_transfer_reserve_bytes == 0  # Model-sized scratch reserved by Worker.
    else:
        with pytest.raises(SystemExit) as error:
            parse_args(flags)
        assert error.value.code == 2
    load.assert_not_called()


@pytest.mark.parametrize("prefill", [False, True])
def test_host_staging_plan_counts_alignment_and_bounds_only_the_main_te(prefill):
    from minisgl.disagg.cluster_config import host_staging_plan

    options = {
        "max_seq_len": 32,
        "bytes_per_token": 128,
        "layers": 2,
        "layers_per_chunk": 1,
        "staging_slots": 2,
        "max_transfer_bytes": 4096,
        "max_registered_bytes": 4096,
        "rows": 8,
        "prefill": prefill,
    }
    plan = host_staging_plan(**options)
    assert (plan.payload_bytes, plan.group_bytes, plan.registered_bytes, plan.gpu_bytes) == (
        4096,
        2048,
        4096,
        2304,
    )
    options.update(
        max_transfer_bytes=1000, max_registered_bytes=1000, bytes_per_token=240, layers=3
    )
    with pytest.raises(ValueError, match="--pd-max-registered-bytes"):
        host_staging_plan(**options)


@dataclass
class StagingModel:
    num_layers: int = 3
    num_kv_heads: int = 1
    head_dim: int = 20


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("bad_budget", [False, True])
def test_host_model_budget_is_validated_before_weights_and_reserves_only_gpu_scratch(
    monkeypatch, tmp_path, role, bad_budget
):
    from minisgl.distributed import DistributedInfo
    from minisgl.engine.config import EngineConfig
    from minisgl.server.args import ServerArgs

    model = StagingModel() if bad_budget else StagingModel(2, 1, 8)
    monkeypatch.setattr(EngineConfig, "model_config", property(lambda _: model))
    config = ServerArgs(
        str(tmp_path),
        DistributedInfo(0, 1),
        torch.float16,
        worker_role=role,
        pd_transfer_backend="mooncake",
        pd_mooncake_transport_mode="host_dram",
        max_seq_len_override=32,
        pd_max_transfer_bytes=1000 if bad_budget else 1024,
        pd_max_registered_bytes=1000 if bad_budget else 4096,
        pd_layers_per_chunk=1,
        pd_staging_slots=2,
    )
    captured = []

    class EngineBoundary(Exception):
        pass

    def stop_before_engine(self, actual, **kwargs):
        captured.append(actual)
        raise EngineBoundary()

    fingerprint = Mock(return_value="a" * 64)
    monkeypatch.setattr("minisgl.disagg.worker.checkpoint_fingerprint", fingerprint)
    monkeypatch.setattr(Scheduler, "__init__", stop_before_engine)
    if bad_budget:
        with pytest.raises(ValueError, match="--pd-max-registered-bytes"):
            PDWorkerScheduler(config)
        assert not captured
        fingerprint.assert_not_called()
    else:
        with pytest.raises(EngineBoundary):
            PDWorkerScheduler(config)
        assert captured[0].pd_transfer_reserve_bytes == 640  # 512-byte group + 16 int64 slots.


@pytest.mark.parametrize("mode", ["gpu_direct", "host_dram"])
def test_main_te_host_registration_does_not_require_cuda_enabled_mooncake(monkeypatch, mode):
    import sys
    from types import ModuleType

    from layered_reference import FakeCuda
    from minisgl.disagg import mooncake_runtime as module
    from minisgl.disagg.mooncake_runtime import MooncakeRuntime
    from minisgl.disagg.transfer import TransferBackendUnavailable
    from pd_reference import cpu_pinned_allocations

    registrations = []

    class Engine:
        def initialize(self, *args):
            return 0

        def get_rpc_port(self):
            return 22501

        def register_memory(self, address, length, location, remote):
            registrations.append((address, length, location, remote))
            return 0

        def unregister_memory(self, address):
            return 0

        def submit_scatter_write(self, *args):
            raise AssertionError("no request submitted in startup test")

    native = ModuleType("host_staging_test_engine")
    native.SUPPORT_CUDA = False
    native.TransferEngine = Engine
    native.ScatterTransferCompletionStatus = SimpleNamespace(
        COMPLETED=1, FAILED_DRAINED=2, COMPLETION_UNKNOWN=3
    )
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(
        module,
        "CudaTransfer",
        lambda device, **kwargs: FakeCuda(kwargs.get("transfer_mode", "gpu_direct")),
    )
    monkeypatch.setattr(MooncakeRuntime, "_configure_rdma", staticmethod(lambda _: ["fake_hca"]))
    monkeypatch.setenv("MC_MIN_RPC_PORT", "16000")
    monkeypatch.setenv("MC_MAX_RPC_PORT", "16000")
    runtime = MooncakeRuntime(
        "127.0.0.1",
        engine_module=native.__name__,
        rpc_port=22501,
        transfer_mode=mode,
        max_transfer_bytes=512,
        max_registered_bytes=1024,
    )
    try:
        if mode == "gpu_direct":
            with pytest.raises(TransferBackendUnavailable, match="lacks CUDA support"):
                runtime.bind_device(torch.device("cuda:0"))
            assert runtime.transfer_mode == "gpu_direct" and not registrations
        else:
            with cpu_pinned_allocations():
                runtime.bind_device(torch.device("cuda:0"))
                buffer = runtime.allocate_buffer(512, "host-main", pool=True)
            assert buffer.tensor.device.type == "cpu" and buffer.address % 256 == 0
            assert registrations == [(buffer.address, 512, "*", True)]
            assert runtime.info["memory"] == "host_pinned"
            assert runtime.info["transfer_mode"] == "host_dram"
            runtime.make_visible(buffer)
            assert runtime.cuda.visible == 0
    finally:
        runtime.close()


def test_explicit_host_cuda_adapter_skips_gdr_and_rejects_unpinned_memory(monkeypatch):
    from contextlib import nullcontext

    from minisgl.disagg import cuda_transfer
    from minisgl.disagg.transfer import TransferBackendUnavailable, TransferError

    driver = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        cuInit=lambda _: (0,),
        cuCtxGetCurrent=lambda: (0, 1),
        cuDeviceGet=lambda _: (0, 0),
        CUdevice_attribute=SimpleNamespace(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED=1),
        cuDeviceGetAttribute=lambda *args: (0, 0),
    )
    original = cuda_transfer.importlib.import_module
    monkeypatch.setattr(
        cuda_transfer.importlib,
        "import_module",
        lambda name: driver if name == "cuda.bindings.driver" else original(name),
    )
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    adapter = cuda_transfer.CudaTransfer(torch.device("cuda:0"), transfer_mode="host_dram")
    backing = torch.empty(767, dtype=torch.uint8)
    start = -backing.data_ptr() % 256
    tensor = backing[start : start + 512]
    with pytest.raises(TransferError, match="pinned CPU"):
        adapter.prepare_buffer(tensor)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda _: True)
    adapter.prepare_buffer(tensor)
    adapter.make_visible()  # Driver has no cuCtxSynchronize or GDR flush entrypoints.
    assert not adapter.info["gpudirect_rdma"]
    with pytest.raises(TransferBackendUnavailable, match="does not support GPUDirect"):
        cuda_transfer.CudaTransfer(torch.device("cuda:0"))


def test_full_prefix_reservation_still_rejects_a_different_transport_mode():
    from minisgl.disagg.client import reservation_prefix
    from minisgl.disagg.control import StartRequest

    request = StartRequest(str(uuid4()), 0, (1, 2, 3, 4), 3, True)
    ready = {
        "handoff_version": 2,
        "transfer_id": request.transfer_id,
        "prompt_tokens": 4,
        "decode_prefix_len": 4,
        "ticket": None,
        "transfer_mode": "host_dram",
    }
    assert reservation_prefix(ready, request, layered=True, transfer_mode="host_dram") == 4
    with pytest.raises(ValueError, match="transport mode"):
        reservation_prefix(ready, request, layered=True, transfer_mode="gpu_direct")


def dram_config(**kwargs):
    return DramConfig(capacity_bytes=1024**2, remote_enabled=True, **kwargs)


def test_default_remote_budget_accepts_exact_64kib_per_token_but_rejects_a_larger_model():
    config = dram_config()
    config.validate_layout(64 * 1024, 16)
    with pytest.raises(ValueError, match="model_bytes_per_token=65537") as error:
        config.validate_layout(64 * 1024 + 1, 16)
    assert "--pd-dram-min-remote-tokens=1024" in str(error.value)
    assert "--pd-dram-max-pull-bytes=67108864" in str(error.value)
    replace(config, remote_max_bytes=128 * 1024**2).validate_layout(128 * 1024, 16)


def test_non_aligned_token_threshold_requires_enough_bytes_for_a_whole_page():
    config = dram_config(min_remote_tokens=5, remote_max_bytes=5 * 128)
    with pytest.raises(ValueError, match="8 aligned tokens / 1024 bytes"):
        config.validate_layout(128, 4)
    replace(config, remote_max_bytes=8 * 128).validate_layout(128, 4)


def test_disabled_remote_mode_does_not_apply_the_remote_budget_to_local_cache():
    DramConfig().validate_layout(1024**2, 16)
    DramConfig(capacity_bytes=1024**2).validate_layout(1024**2, 16)


def test_descriptor_limit_cannot_make_an_enabled_remote_threshold_unreachable():
    config = dram_config(min_remote_tokens=MAX_DRAM_PULL_PAGES + 1)
    with pytest.raises(ValueError, match=f"at most {MAX_DRAM_PULL_PAGES} pages"):
        config.validate_layout(1, 1)


@pytest.mark.parametrize("role", [WorkerRole.PREFILL, WorkerRole.DECODE])
def test_bad_model_budget_fails_before_weights_engine_or_pinned_allocation(monkeypatch, role):
    config = SimpleNamespace(
        worker_role=role,
        tp_info=SimpleNamespace(size=1),
        use_dummy_weight=False,
        dtype=torch.float16,
        max_extend_tokens=32,
        max_running_req=8,
        page_size=16,
        pd_dram=dram_config(),
        model_config=SimpleNamespace(num_layers=32, num_kv_heads=8, head_dim=128),
    )
    engine = Mock(side_effect=AssertionError("must not construct the engine"))
    fingerprint = Mock(side_effect=AssertionError("must not read weights"))
    allocate = Mock(side_effect=AssertionError("must not allocate pinned memory"))
    monkeypatch.setattr(Scheduler, "__init__", engine)
    monkeypatch.setattr("minisgl.disagg.worker.checkpoint_fingerprint", fingerprint)
    monkeypatch.setattr("minisgl.disagg.dram_service.DramPagePool", allocate)
    with pytest.raises(ValueError, match="model_bytes_per_token=131072"):
        PDWorkerScheduler(config)
    service = SimpleNamespace(
        s=SimpleNamespace(layout=KVLayout("a" * 64, "float16", 32, 8, 128, 16))
    )
    with pytest.raises(ValueError, match="model_bytes_per_token=131072"):
        DramServiceMixin._init_dram(service, {"dram": asdict(config.pd_dram)})
    engine.assert_not_called()
    fingerprint.assert_not_called()
    allocate.assert_not_called()


LAYOUT = asdict(KVLayout("a" * 64, "float32", 2, 1, 8, 4))


def worker_health(node, minimum=4, maximum=4096):
    return {
        "status": "ready",
        "role": node.role,
        "instance_id": str(uuid4()),
        "multiplex_version": 1,
        "handoff_version": 2,
        "decode_prefix_reuse": True,
        "layered_transfer": False,
        "layout": dict(LAYOUT),
        "dram_cache": {
            "enabled": True,
            "remote_enabled": True,
            "remote_ready": True,
            "min_remote_tokens": minimum,
            "remote_max_bytes": maximum,
        },
    }


@pytest.fixture
def cache_pool():
    pool = WorkerPool(["p0:1919"], ["d0:1919"])
    pool.configure_prefix_index(
        RoutingConfig(policy="cache_aware", min_remote_pull_tokens=4, dram_max_pull_bytes=4096)
    )
    try:
        yield pool
    finally:
        pool.close()


@pytest.mark.parametrize("minimum,maximum", [(8, 4096), (4, 2048)])
@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_health_mismatch_warns_once_per_configuration_or_instance_without_retiring_worker(
    cache_pool, caplog, minimum, maximum, role
):
    caplog.set_level(logging.WARNING, logger="minisgl.disagg.router")
    node = next(n for n in cache_pool.nodes if n.role == role)
    health = worker_health(node, minimum, maximum)
    for _ in range(3):
        cache_pool.apply_health(node, copy.deepcopy(health))
    assert node.healthy and len(caplog.records) == 1
    assert "--min-remote-pull-tokens=4" in caplog.text
    assert f"--pd-dram-min-remote-tokens={minimum}" in caplog.text
    assert f"--pd-dram-max-pull-bytes={maximum}" in caplog.text
    assert "intersection" in caplog.text
    health["instance_id"] = str(uuid4())
    cache_pool.apply_health(node, health)
    assert node.healthy and len(caplog.records) == 2


def test_aligned_limits_have_no_warning_and_local_only_workers_are_not_mismatches(
    cache_pool, caplog
):
    for node in cache_pool.nodes:
        cache_pool.apply_health(node, worker_health(node))
        health = worker_health(node)
        health["dram_cache"].update(remote_enabled=False, remote_ready=False)
        cache_pool.apply_health(node, health)
        assert node.healthy
    assert not caplog.records


def test_health_warns_when_router_and_worker_intersection_cannot_fit_a_page(cache_pool, caplog):
    cache_pool.routing_config = replace(cache_pool.routing_config, min_remote_pull_tokens=5)
    node = cache_pool.nodes[0]
    cache_pool.apply_health(node, worker_health(node, 4, 512))
    assert node.healthy
    assert "no feasible pull" in caplog.text
    assert "1024 page-aligned bytes" in caplog.text
    assert "512-byte limit" in caplog.text


@pytest.mark.parametrize("limits", [{}, {"min_remote_tokens": True, "remote_max_bytes": 4096}])
def test_unknown_limits_warn_and_do_not_make_inference_worker_unhealthy(cache_pool, caplog, limits):
    node = cache_pool.nodes[0]
    health = worker_health(node)
    health["dram_cache"] = {"remote_ready": True, **limits}
    cache_pool.apply_health(node, health)
    assert node.healthy and "remote hints disabled" in caplog.text


def test_requested_but_unavailable_host_transport_has_an_explicit_health_warning(
    cache_pool, caplog
):
    node = cache_pool.nodes[0]
    health = worker_health(node)
    health["dram_cache"].update(remote_ready=False, error="missing reader fence API")
    cache_pool.apply_health(node, health)
    assert node.healthy and "missing reader fence API" in caplog.text


def tier_directory():
    spec = PrefixHashSpec.from_layout(LAYOUT)
    prepared = prepare_prefix(tuple(range(25)), spec)
    hashes = tuple(prepared.digests())
    index, nodes = TierIndex(), []
    for worker_id, role, depths in (("p-0", "prefill", range(2)), ("d-0", "decode", range(2, 6))):
        node = SimpleNamespace(
            worker_id=worker_id, role=role, endpoint=f"{worker_id}:1919", healthy=True
        )
        node.health = worker_health(node)
        publisher = TierPublisher(spec, node.health["instance_id"])
        for i in depths:
            tier = "HBM" if role == "prefill" and i == 0 else "DRAM"
            publisher.update(
                "add", tier, hashes[i], spec.root if i == 0 else hashes[i - 1], i + 1, 1
            )
        node.health["tier_index"] = publisher.health()
        index.observe(worker_id, node.health)
        for _ in range(10):
            state, request = index.request(worker_id)
            index.apply(worker_id, state, publisher.sync(request))
            if state.ready:
                break
        assert state.ready
        nodes.append(node)
    return index, nodes, prepared


@pytest.mark.parametrize(
    "p_changes,d_changes,router,remote",
    [
        ({}, {}, {}, 16),
        ({"min_remote_tokens": 20}, {}, {}, 0),
        ({}, {"min_remote_tokens": 20}, {}, 0),
        ({"remote_max_bytes": 1024}, {}, {}, 8),
        ({}, {"remote_max_bytes": 512}, {}, 4),
        ({}, {}, {"max_remote_bytes": 512}, 4),
        ({}, {}, {"min_remote_tokens": 20}, 0),
        ({"remote_max_bytes": 512}, {"min_remote_tokens": 8}, {}, 0),
        ({"min_remote_tokens": None}, {}, {}, 0),
        ({}, {"remote_max_bytes": True}, {}, 0),
    ],
)
def test_tier_plans_obey_both_workers_and_router_while_preserving_local_tiers(
    p_changes, d_changes, router, remote
):
    index, (p, d), prepared = tier_directory()
    p.health["dram_cache"].update(p_changes)
    d.health["dram_cache"].update(d_changes)
    options = {"min_remote_tokens": 4, "max_remote_bytes": 4096, **router}
    hit = index.hits([p], [d], prepared, **options)[p.worker_id]
    assert (hit.hbm_tokens, hit.dram_tokens, hit.remote_tokens) == (4, 4, remote)
    assert (hit.remote_plan is not None) == bool(remote)
    if remote:
        assert hit.remote_plan.source_id == d.worker_id
        assert hit.remote_plan.start_page == 2 and len(hit.remote_plan.page_hashes) == remote // 4


def cluster_args(*flags):
    args = launcher.parser().parse_args(["--model", "/unused/model", *flags])
    args.dram_remote_enabled = True
    return args


@pytest.mark.parametrize("source", ["control", "distributed", "data", "main TE", "Host TE"])
@pytest.mark.parametrize("target", ["control_port", "data_port", "mooncake_port", "dram_port"])
def test_every_cross_worker_port_category_is_checked_with_both_owners_in_the_error(source, target):
    args = cluster_args("--pd-transfer-backend", "mooncake")
    p, d = local_plan(args, ["0", "1"])
    port = {
        "control": p.control_port,
        "distributed": p.control_port + 1,
        "data": p.data_port,
        "main TE": p.mooncake_port,
        "Host TE": p.dram_port,
    }[source]
    bad = replace(d, **{target: port})
    with pytest.raises(ValueError, match=f"port {port} overlaps") as error:
        validate_cluster_ports([p, bad], gateway_port=args.port, backend="mooncake")
    assert f"p-0 {source}" in str(error.value) and "d-0" in str(error.value)


@pytest.mark.parametrize("label,port", [("GPU TE", 32768), ("Host TE", 60999)])
def test_native_ephemeral_port_rejection_names_the_engine_and_the_excluded_range(label, port):
    with pytest.raises(ValueError, match=f"{label}:.*ephemeral port range"):
        validate_port_bindings([(label, port, True)])


@pytest.mark.parametrize("port", [1024, 32767, 61000, 65535])
def test_native_fixed_port_allowed_range_boundaries(port):
    validate_port_bindings([("GPU TE", port, True)])


def test_control_ephemeral_range_warns_but_is_allowed(caplog):
    local_plan(cluster_args("--base-control-port", "40000"), ["0", "1"])
    assert "ephemeral range 32768..60999" in caplog.text
    assert "p-0 control=40000" in caplog.text


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("host_port", [1919, 1920, 1921, 22500])
def test_standalone_host_te_conflicts_fail_before_model_loading(
    monkeypatch, capsys, role, host_port
):
    load = Mock(side_effect=AssertionError("must not load the model"))
    monkeypatch.setattr("minisgl.utils.cached_load_hf_config", load)
    with pytest.raises(SystemExit) as error:
        parse_args(
            [
                "--model",
                "/unused",
                "--worker-role",
                role,
                "--pd-transfer-backend",
                "mooncake",
                "--pd-mooncake-port",
                "22500",
                "--pd-dram-remote-pull",
                "--pd-dram-cache-bytes",
                "65536",
                "--pd-dram-port",
                str(host_port),
            ]
        )
    assert error.value.code == 2
    assert "Host TE" in capsys.readouterr().err
    load.assert_not_called()


def test_launcher_rejects_host_te_colliding_with_next_worker_before_spawning(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "visible_gpus", lambda: ["0", "1"])
    popen = Mock(side_effect=AssertionError("must not spawn"))
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    with pytest.raises(SystemExit) as error:
        launcher.main(
            [
                "--model",
                "/unused",
                "--pd-transfer-backend",
                "mooncake",
                "--base-dram-port",
                "1921",
                "--pd-dram-remote-pull",
                "--pd-dram-cache-bytes",
                "65536",
            ]
        )
    assert error.value.code == 2
    diagnostic = capsys.readouterr().err
    assert "p-0 Host TE" in diagnostic and "d-0 control" in diagnostic
    popen.assert_not_called()


def test_programmatic_supervisor_rechecks_ports_before_starting_even_the_first_child():
    args = cluster_args("--no-gateway")
    p, d = local_plan(args, ["0", "1"])
    popen = Mock(side_effect=AssertionError("must not spawn"))
    supervisor = launcher.ClusterSupervisor(
        args, [p, replace(d, dram_port=p.data_port)], [p.endpoint], [d.endpoint], [], popen=popen
    )
    with pytest.raises(ValueError, match="overlaps"):
        supervisor.start()
    popen.assert_not_called()


def test_gateway_port_is_part_of_the_same_collision_domain():
    args = cluster_args()
    p, d = local_plan(args, ["0", "1"])
    with pytest.raises(ValueError, match="gateway HTTP.*p-0 Host TE"):
        validate_cluster_ports([p, d], gateway_port=p.dram_port)
