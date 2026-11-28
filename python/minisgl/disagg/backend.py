"""Lazy backend selection; using the reference TCP path never imports Mooncake."""

from __future__ import annotations

from .role import WorkerRole
from .tcp_backend import TcpTransferBackend


# 根据启动参数 args.pd_transfer_backend，
# 决定 P Worker 和 D Worker 使用 TCP 还是 Mooncake Transfer Engine 进行 KV Cache 传输。
def create_transfer_backend(args):
    name = args.pd_transfer_backend
    bind = (
        (args.server_host, args.server_port + 2 if args.pd_data_port is None else args.pd_data_port)
        if args.worker_role is WorkerRole.DECODE
        else None
    )
    kwargs = {"advertise_host": args.pd_advertise_host, "timeout": args.pd_timeout}
    from .cluster_config import main_registered_budget

    registered_budget = main_registered_budget(args)
    if name == "tcp":
        return TcpTransferBackend(bind, max_transfer_bytes=args.pd_max_transfer_bytes, **kwargs)
    if name == "mooncake":
        from .layered_mooncake import LayeredMooncakeBackend

        return LayeredMooncakeBackend(
            bind,
            local_host=args.pd_mooncake_host or args.pd_advertise_host or args.server_host,
            max_transfer_bytes=args.pd_max_transfer_bytes,
            max_registered_bytes=registered_budget,
            max_pending_transfers=getattr(args, "pd_max_pending_transfers", 32),
            max_quarantine_seconds=getattr(args, "pd_max_quarantine_seconds", 300.0),
            device_name=getattr(args, "pd_mooncake_device", ""),
            rpc_port=getattr(args, "pd_mooncake_port", None),
            layers_per_chunk=getattr(args, "pd_layers_per_chunk", 1),
            staging_slots=getattr(args, "pd_staging_slots", 3),
            transfer_mode=getattr(args, "pd_mooncake_transport_mode", "gpu_direct"),
            **kwargs,
        )
    if name == "shm_dram":
        from .layered_shm import LayeredShmBackend

        return LayeredShmBackend(
            bind,
            manifest=args.pd_shm_manifest,
            worker_id=args.worker_id,
            numa_id=args.pd_numa_id,
            host_id=args.pd_host_id,
            layers_per_chunk=args.pd_layers_per_chunk,
            staging_slots=args.pd_staging_slots,
            overflow_slots=args.pd_shm_overflow_slots,
            local_wait_ms=args.pd_shm_local_wait_ms,
            max_transfer_bytes=args.pd_max_transfer_bytes,
            max_registered_bytes=registered_budget,
            max_pending_transfers=args.pd_max_pending_transfers,
            **kwargs,
        )
    raise ValueError(f"unsupported P/D transfer backend: {name}")
