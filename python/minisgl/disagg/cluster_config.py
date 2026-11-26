"""Pure configuration validation for local P/D clusters and remote pools."""

import logging
from dataclasses import dataclass

from .control import endpoint_address, endpoint_name

logger = logging.getLogger(__name__)


def values(items):
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    result = [part.strip() for item in items for part in item.split(",")]
    if any(not part for part in result):
        raise ValueError("empty list entry")
    return result


def validate_mooncake_port(port):
    # Match classic TE's ValidatePortRange; otherwise it silently falls back to
    # its default range, which would defeat per-process fixed-port isolation.
    if type(port) is not int or not 1024 <= port <= 65535 or 32768 <= port <= 60999:
        raise ValueError(
            "Mooncake RPC port must be 1024..32767 or 61000..65535; "
            "32768..60999 is the excluded ephemeral port range"
        )


def worker_port_bindings(
    worker_id, control, data=None, gpu_te=None, host_te=None, *, native_gpu=True
):
    if type(control) is not int:
        raise ValueError(f"{worker_id} control port must be an integer")
    result = [
        (f"{worker_id} control", control, False),
        (f"{worker_id} distributed", control + 1, False),
    ]
    for label, port, native in (
        ("data", data, False),
        ("GPU TE", gpu_te, native_gpu),
        ("Host TE", host_te, True),
    ):
        if port is not None:
            result.append((f"{worker_id} {label}", port, native))
    return result


def validate_port_bindings(bindings):
    """Validate the whole known plan before any child/listener/native engine starts."""
    occupied, ephemeral = {}, []
    for label, port, native in bindings:
        if type(port) is not int or not 0 < port < 65536:
            raise ValueError(f"invalid {label} port {port!r}; expected 1..65535")
        if port in occupied:
            raise ValueError(f"port {port} overlaps: {occupied[port]} and {label} must be distinct")
        occupied[port] = label
        if native:
            try:
                validate_mooncake_port(port)
            except ValueError as exc:
                raise ValueError(f"{label}: {exc}") from exc
        elif 32768 <= port <= 60999:
            ephemeral.append(f"{label}={port}")
    if ephemeral:
        logger.warning(
            "Ports in ephemeral range 32768..60999 may collide with outgoing connections: %s; "
            "prefer ports outside that range (fixed TE ports reject it)",
            ", ".join(ephemeral),
        )


def validate_cluster_ports(specs, *, gateway_port=None, backend="tcp"):
    bindings = [] if gateway_port is None else [("gateway HTTP", gateway_port, False)]
    for spec in specs:
        bindings.extend(
            worker_port_bindings(
                spec.worker_id,
                spec.control_port,
                spec.data_port,
                spec.mooncake_port,
                spec.dram_port,
                native_gpu=backend == "mooncake",
            )
        )
    validate_port_bindings(bindings)


def add_routing_args(parser):
    from .routing_policy import add_policy_args

    add_policy_args(parser)
    parser.add_argument("--num-prefill-workers", type=int, default=None)
    parser.add_argument("--num-decode-workers", type=int, default=None)
    parser.add_argument("--prefill-addrs", nargs="+", default=None)
    parser.add_argument("--decode-addrs", nargs="+", default=None)
    parser.add_argument("--routing-seed", type=int, default=None)
    parser.add_argument("--health-interval", type=float, default=2.0)
    parser.add_argument("--health-timeout", type=float, default=3.0)
    parser.add_argument("--worker-ready-timeout", type=float, default=300.0)
    parser.add_argument("--max-quarantined-bytes", type=int, default=0)


def remote_addresses(args):
    p, d = values(args.prefill_addrs), values(args.decode_addrs)
    if bool(p) != bool(d):
        raise ValueError("--prefill-addrs and --decode-addrs must be supplied together")
    for count in (args.num_prefill_workers, args.num_decode_workers):
        if count is not None and count < 1:
            raise ValueError("N and M must both be at least 1")
    if p:
        for count, endpoints in ((args.num_prefill_workers, p), (args.num_decode_workers, d)):
            if count is not None and count != len(endpoints):
                raise ValueError("Worker count differs from its address list length")
        parsed = [endpoint_address(address) for address in p + d]
        if len(set(parsed)) != len(parsed):
            raise ValueError("Worker endpoints must be unique across roles")
    return p, d


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    role: str
    gpu: str
    host: str
    control_port: int
    mooncake_port: int
    data_port: int
    numa_node: int | None = None
    rdma_device: str = ""
    dram_port: int | None = None

    @property
    def endpoint(self):
        return endpoint_name(self.host, self.control_port)


def local_plan(args, visible_gpus):
    n = 1 if args.num_prefill_workers is None else args.num_prefill_workers
    m = 1 if args.num_decode_workers is None else args.num_decode_workers
    if any(type(count) is not int or count < 1 for count in (n, m)):
        raise ValueError("N and M must both be at least 1")
    if n + m > len(visible_gpus):
        raise ValueError(
            f"TP=1 needs {n + m} distinct GPUs, but only {len(visible_gpus)} are visible"
        )
    pg, dg = values(args.prefill_gpus), values(args.decode_gpus)
    if pg and len(pg) != n or dg and len(dg) != m:
        raise ValueError("GPU mapping length must match its Worker count")
    explicit = pg + dg
    if len(set(explicit)) != len(explicit) or any(gpu not in visible_gpus for gpu in explicit):
        raise ValueError("GPU mappings must be distinct members of CUDA_VISIBLE_DEVICES")
    remaining = [gpu for gpu in visible_gpus if gpu not in explicit]
    if not pg:
        pg, remaining = remaining[:n], remaining[n:]
    if not dg:
        dg = remaining[:m]
    host = args.worker_advertise_host or args.worker_host
    if host in ("0.0.0.0", "::"):
        raise ValueError("wildcard Worker bind requires --worker-advertise-host")
    endpoint_address(endpoint_name(host, args.base_control_port))
    specs = []
    for role, count, gpus, numa, hcas in (
        ("prefill", n, pg, values(args.prefill_numa_nodes), values(args.prefill_rdma_devices)),
        ("decode", m, dg, values(args.decode_numa_nodes), values(args.decode_rdma_devices)),
    ):
        if numa and len(numa) != count or hcas and len(hcas) != count:
            raise ValueError("NUMA/HCA mappings must have one entry per Worker")
        for i, gpu in enumerate(gpus):
            index = len(specs)
            control = args.base_control_port + 2 * index
            te = args.base_mooncake_port + 2 * index
            dram_port = None
            if getattr(args, "dram_remote_enabled", False):
                dram_port = getattr(args, "base_dram_port", 25000) + index
            node = int(numa[i]) if numa else None
            if node is not None and node < 0:
                raise ValueError("NUMA nodes must be nonnegative")
            hca = hcas[i].replace("+", ",") if hcas else ""
            if any(not token or "/" in token for token in hca.split(",")) and hca:
                raise ValueError("invalid HCA mapping")
            specs.append(
                WorkerSpec(
                    f"{role[0]}-{i}", role, gpu, host, control, te, te + 1, node, hca, dram_port
                )
            )
    validate_cluster_ports(
        specs, gateway_port=None if args.no_gateway else args.port, backend=args.pd_transfer_backend
    )
    return specs
