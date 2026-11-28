#!/usr/bin/env python3
"""Strict two-process shared historical KV preflight, without model weights.

PYTHONPATH=python python tools/pd/verify_shared_dram_history.py \
    --prefill-gpu 0 --decode-gpu 4

Choose GPUs attached to different sockets to verify cross-NUMA H2D. Both
processes register ALL segments before D writes. Each NUMA segment is tested:
D HBM -> resident shared page -> P GPU workspace -> P formal KV page. Reports
raw DMA bandwidth; gather/scatter correctness is checked separately. This does
not measure model TTFT/TPOT. No CUDA IPC event or Host-to-Host copy is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import signal
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4


def _copy(destination, source, stream, iterations):
    import torch

    samples, wall = [], []
    for i in range(iterations + 2):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        with torch.cuda.stream(stream):
            start.record(stream)
            destination.copy_(source, non_blocking=True)
            end.record(stream)
        end.synchronize()  # D2H-before-commit / H2D-before-unpin, in this process.
        elapsed = time.perf_counter() - begin
        if i >= 2:
            samples.append(start.elapsed_time(end))
            wall.append(elapsed)
    nbytes = destination.numel() * destination.element_size()
    ms = statistics.median(samples)
    return {
        "bytes": nbytes,
        "iterations": iterations,
        "cuda_ms_median": ms,
        "cuda_gbps": nbytes / (ms * 1e6),
        "wall_gbps": nbytes / (statistics.median(wall) * 1e9),
    }


def _gpu_process(encoded, layout_data, gpu, numa_id, writer, pipe, iterations):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    pool = client = None
    try:
        from minisgl.disagg.numa import bind_worker

        bind_worker(numa_id)  # Before CUDA initialization and GPU workspace allocation.
        import torch
        from minisgl.disagg.prefix_hash import PrefixHashSpec
        from minisgl.disagg.shared_dram_directory import SharedDramClient
        from minisgl.disagg.shared_dram_pool import (
            READY,
            SharedDramManifest,
            SharedDramPool,
            SharedPage,
        )
        from minisgl.distributed import set_tp_info
        from minisgl.kvcache.mha_pool import MHAKVCache

        torch.cuda.set_device(0)
        set_tp_info(0, 1)
        manifest = SharedDramManifest.from_json(encoded)
        spec = PrefixHashSpec.from_layout(layout_data)
        pool = SharedDramPool(manifest)
        pool.register_cuda()  # Explicit error on memlock/driver failure, no fallback.
        client = SharedDramClient(manifest, domain_id=spec.domain_id)
        cache = MHAKVCache(
            num_kv_heads=layout_data["kv_heads"],
            num_layers=layout_data["layers"],
            head_dim=layout_data["head_dim"],
            num_pages=1,
            page_size=manifest.page_size,
            dtype=getattr(torch, layout_data["dtype"]),
            device=torch.device("cuda:0"),
        )
        stage = torch.empty(manifest.page_bytes, dtype=torch.uint8, device="cuda")
        indices = torch.arange(manifest.page_size, dtype=torch.int64, device="cuda")
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
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
            _, node, key = command
            if writer:
                result = client.allocate([key], node)[0]
                if result["status"] != "allocated":
                    raise RuntimeError(f"preflight allocation failed: {result}")
                page = SharedPage.from_object(result["page"])
                if page.numa_id != node:
                    raise RuntimeError(
                        "preflight page was not allocated on its requested NUMA node"
                    )
                with torch.cuda.stream(stream):
                    cache._kv_buffer.fill_(37)
                    cache.export_page_into(stage, indices)
                host = pool.view(page)
                if not host.is_pinned():
                    raise RuntimeError("shared page is not recognized as CUDA pinned memory")
                measurements = _copy(host, stage, stream, iterations)
                client.written([page])  # All D2H events completed before READY publication.
                direction = "d2h"
            else:
                pages = client.pin_prefix([key])
                if len(pages) != 1 or pages[0].numa_id != node:
                    raise RuntimeError("P cannot pin the page published by D")
                page = pages[0]
                host = pool.view(page)  # Actual remote segment, no local Host payload buffer.
                if not host.is_pinned():
                    raise RuntimeError("shared page is not recognized as CUDA pinned memory")
                measurements = _copy(stage, host, stream, iterations)
                with torch.cuda.stream(stream):
                    cache.import_page_from(stage, indices)
                    installed = torch.cuda.Event()
                    installed.record(stream)
                while not installed.query():
                    time.sleep(0.001)
                if pool.header(node, page.slot_id) != (page.generation, READY):
                    raise RuntimeError("shared history generation/state changed while pinned")
                if not bool(torch.all(cache._kv_buffer == 37).item()):
                    raise RuntimeError("P formal KV page differs from D's original all-layer KV")
                client.unpin(pages)
                direction = "h2d"
            del host
            pipe.send(
                {
                    "status": "ok",
                    "direction": direction,
                    "gpu": gpu,
                    "gpu_numa_id": numa_id,
                    "page_numa_id": node,
                    "local": node == numa_id,
                    "generation": page.generation,
                    "correct": True,
                    **measurements,
                }
            )
        torch.cuda.synchronize()
        client.close(drained=True)
        client = None
        pool.close()
        pool = None
    except BaseException as exc:
        try:
            pipe.send({"status": "error", "gpu": gpu, "error": f"{type(exc).__name__}: {exc}"})
        except (OSError, EOFError):
            pass
        # Unknown DMA ownership is reclaimed only after this process exits.
    finally:
        if client is not None:
            client.disconnect()
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
    parser.add_argument("--numa-nodes", help="OS node ids; default: all discovered host nodes")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args(argv)
    if (
        any(
            v < 1
            for v in (args.layers, args.kv_heads, args.head_dim, args.page_size, args.iterations)
        )
        or args.timeout <= 0
        or args.prefill_gpu == args.decode_gpu
    ):
        parser.error("dimensions/iterations/timeout must be positive; use two distinct GPUs")
    from minisgl.disagg.numa import gpu_numa_nodes, host_numa_nodes
    from minisgl.disagg.protocol import KVLayout
    from minisgl.disagg.shared_dram_directory import SharedDramDirectory, SharedDramServer
    from minisgl.disagg.shared_dram_pool import SharedDramPool, build_shared_manifest
    from minisgl.disagg.shm_relay_pool import align

    gpus = [args.prefill_gpu, args.decode_gpu]
    mapping = gpu_numa_nodes(gpus)
    nodes = host_numa_nodes(override=args.numa_nodes)
    if any(mapping[g] not in nodes for g in gpus):
        parser.error("test NUMA inventory must contain both GPUs' NUMA nodes")
    layout = KVLayout(
        "f" * 64, "float16", args.layers, args.kv_heads, args.head_dim, args.page_size
    )
    page_bytes = layout.bytes_per_token * layout.page_size
    per_node = align(256 + align(page_bytes), 4096)
    root = tempfile.TemporaryDirectory(prefix="minisgl-cache-preflight-")
    manifest = build_shared_manifest(
        name=f"minisgl_history_preflight_{uuid4().hex}",
        page_bytes=page_bytes,
        page_size=args.page_size,
        capacity_bytes=per_node * len(nodes),
        numa_nodes=nodes,
        worker_numa={"p-0": mapping[gpus[0]], "d-0": mapping[gpus[1]]},
        directory_socket=str(Path(root.name) / "directory.sock"),
        secret=uuid4().hex,
    )
    owner = server = None
    children = []
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    try:
        owner = SharedDramPool(manifest, create=True)  # strict mbind -> first touch
        server = SharedDramServer(SharedDramDirectory(owner))
        ctx = multiprocessing.get_context("spawn")
        for gpu, writer in ((gpus[1], True), (gpus[0], False)):
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=_gpu_process,
                args=(
                    manifest.to_json(),
                    asdict(layout),
                    gpu,
                    mapping[gpu],
                    writer,
                    child,
                    args.iterations,
                ),
            )
            process.start()
            child.close()
            children.append((process, parent))
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
            key = hashlib.sha256(f"shared-history-preflight-{node}".encode()).hexdigest()
            for process, pipe in children:  # D publishes before P pins/restores.
                pipe.send(("copy", node, key))
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
        if server is not None:
            server.close()
        if owner is not None:
            owner.close()  # Unlink only AFTER every potential DMA owner has exited.
        root.cleanup()
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
