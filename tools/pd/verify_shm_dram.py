#!/usr/bin/env python3
"""Hardware preflight only: strict NUMA SHM, two CUDA registrations, DMA bandwidth.

PYTHONPATH=python python tools/pd/verify_shm_dram.py --prefill-gpu 0 --decode-gpu 1
Reports raw copy throughput, not model TTFT/TPOT or gather/scatter performance.
No Mooncake runtime or CUDA IPC event is used.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import signal
import statistics
import time
from pathlib import Path
from uuid import uuid4


def _gpu_process(encoded, gpu, numa_id, pipe, iterations):
    # Spawn, rather than fork, and initialize CUDA only after GPU/CPU affinity.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    from minisgl.disagg.numa import bind_worker
    from minisgl.disagg.shm_relay_pool import RelayManifest, SharedRelayPool

    pool = None
    try:
        bind_worker(numa_id)
        import torch

        torch.cuda.set_device(0)
        manifest = RelayManifest.from_json(encoded)
        pool = SharedRelayPool(manifest)
        pool.register_cuda()
        device = torch.full((manifest.group_bytes,), 37, dtype=torch.uint8, device="cuda")
        stream = torch.cuda.Stream()
        pipe.send(
            {
                "status": "registered",
                "gpu": gpu,
                "numa_id": numa_id,
                "registered_segments": len(pool._registered),
                "registered_bytes": manifest.registered_bytes,
                "device": torch.cuda.get_device_name(0),
            }
        )
        while True:
            command = pipe.recv()
            if command[0] == "stop":
                break
            direction, node = command
            host = pool.tensor(node, 0, manifest.group_bytes)
            if not host.is_pinned():
                raise RuntimeError("registered POSIX SHM is not recognized as CUDA pinned memory")
            samples, wall = [], []
            for i in range(iterations + 2):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                begin = time.perf_counter()
                with torch.cuda.stream(stream):
                    start.record(stream)
                    if direction == "d2h":
                        host.copy_(device, non_blocking=True)
                    else:
                        device.copy_(host, non_blocking=True)
                    end.record(stream)
                end.synchronize()  # Local barrier before IPC notification or reuse.
                elapsed = time.perf_counter() - begin
                if i >= 2:
                    samples.append(start.elapsed_time(end))
                    wall.append(elapsed)
            if direction == "h2d" and not bool(torch.all(device == 37).item()):
                raise RuntimeError("shared-memory payload validation failed")
            pipe.send(
                {
                    "direction": direction,
                    "gpu": gpu,
                    "gpu_numa_id": numa_id,
                    "slot_numa_id": node,
                    "local": node == numa_id,
                    "bytes": host.numel(),
                    "iterations": iterations,
                    "cuda_ms_median": statistics.median(samples),
                    "cuda_gbps": host.numel() / (statistics.median(samples) * 1e6),
                    "wall_gbps": host.numel() / (statistics.median(wall) * 1e9),
                }
            )
            del host
        torch.cuda.synchronize()
        pool.close()
        pool = None
    except BaseException as exc:
        pipe.send({"status": "error", "gpu": gpu, "error": f"{type(exc).__name__}: {exc}"})
        # Exit the process before the parent unlinks SHM if CUDA drain is unknown.
    finally:
        pipe.close()


def _reply(pipe, process, timeout):
    if not pipe.poll(timeout):
        raise RuntimeError(f"GPU child pid={process.pid} timed out (exit={process.exitcode})")
    result = pipe.recv()
    if result.get("status") == "error":
        raise RuntimeError(json.dumps(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-gpu", default="0", help="Physical GPU id/UUID")
    parser.add_argument("--decode-gpu", default="1", help="Physical GPU id/UUID")
    parser.add_argument("--numa-nodes", help="Comma-separated OS node ids; default: discover sysfs")
    parser.add_argument("--bytes", type=int, default=4 * 1024**2, help="Bytes per test slot")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    if args.bytes < 256 or args.bytes % 256 or args.iterations < 1 or args.timeout <= 0:
        parser.error(
            "bytes must be a positive multiple of 256; iterations/timeout must be positive"
        )
    from minisgl.disagg.cluster_config import host_staging_plan
    from minisgl.disagg.numa import gpu_numa_nodes
    from minisgl.disagg.shm_relay_pool import SharedRelayPool, build_manifest

    gpus = [args.prefill_gpu, args.decode_gpu]
    mapping = gpu_numa_nodes(gpus)
    nodes = (
        sorted({int(n) for n in args.numa_nodes.split(",")})
        if args.numa_nodes
        else sorted(int(p.name[4:]) for p in Path("/sys/devices/system/node").glob("node[0-9]*"))
    )
    nodes = nodes or [0]
    plan = host_staging_plan(
        max_seq_len=args.bytes // 2,
        bytes_per_token=2,
        layers=1,
        layers_per_chunk=1,
        staging_slots=3,
        max_transfer_bytes=args.bytes,
        max_registered_bytes=4 * args.bytes + 65536,
        rows=1,
        prefill=True,
    )
    manifest = build_manifest(
        f"minisgl_preflight_{uuid4().hex}",
        [("p-0", mapping[gpus[0]])],
        plan,
        bytes_per_token=2,
        layers=1,
        layers_per_chunk=1,
        local_slots=2,
        overflow_slots=1,
        numa_nodes=nodes,
    )
    owner, children = None, []
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    try:
        owner = SharedRelayPool(manifest, create=True)  # strict mbind -> first touch
        ctx = multiprocessing.get_context("spawn")
        for gpu in gpus:
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=_gpu_process,
                args=(manifest.to_json(), gpu, mapping[gpu], child, args.iterations),
            )
            process.start()
            child.close()
            children.append((process, parent))
        # Both registrations are alive before the first copy is submitted.
        registrations = [_reply(pipe, process, args.timeout) for process, pipe in children]
        print(
            json.dumps(
                {
                    "host_id": manifest.host_id,
                    "pool": manifest.name,
                    "nodes": nodes,
                    "registrations": registrations,
                }
            ),
            flush=True,
        )
        for node in nodes:
            for direction, (process, pipe) in zip(("d2h", "h2d"), children):
                pipe.send((direction, node))
                print(json.dumps(_reply(pipe, process, args.timeout)), flush=True)
    finally:
        for process, pipe in children:
            if process.is_alive():
                try:
                    pipe.send(("stop",))
                except (OSError, EOFError):
                    pass
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            pipe.close()
        if owner is not None:
            owner.close()  # Process-exit boundary also covers unknown CUDA operations.
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
