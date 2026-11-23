"""Launch isolated TP=1 P/D Workers and a health-aware random-routing gateway."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict

from .cluster_config import add_routing_args, local_plan, remote_addresses

CLUSTER_FLAGS = {
    "--num-prefill-workers",
    "--num-decode-workers",
    "--prefill-gpus",
    "--decode-gpus",
    "--base-control-port",
    "--base-mooncake-port",
    "--prefill-addrs",
    "--decode-addrs",
    "--routing-seed",
}


def is_cluster_command(argv):
    return any(arg.split("=", 1)[0] in CLUSTER_FLAGS for arg in argv)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_routing_args(p)
    p.add_argument("--model", "--model-path", required=True)
    p.add_argument("--prefill-gpus", nargs="+")
    p.add_argument("--decode-gpus", nargs="+")
    p.add_argument("--prefill-numa-nodes", nargs="+")
    p.add_argument("--decode-numa-nodes", nargs="+")
    p.add_argument("--prefill-rdma-devices", nargs="+")
    p.add_argument("--decode-rdma-devices", nargs="+")
    p.add_argument("--base-control-port", type=int, default=1919)
    p.add_argument("--base-mooncake-port", type=int, default=19500)
    p.add_argument("--worker-host", default="127.0.0.1")
    p.add_argument("--worker-advertise-host", default=None)
    p.add_argument("--pd-mooncake-host", default=None)
    p.add_argument("--pd-transfer-backend", choices=("tcp", "mooncake"), default="tcp")
    p.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    p.add_argument("--host", default="127.0.0.1", help="Gateway HTTP bind address")
    p.add_argument("--port", type=int, default=1940, help="Gateway HTTP port")
    p.add_argument("--no-gateway", action="store_true")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-inflight-requests", type=int, default=128)
    p.add_argument("--max-inflight-tokens", type=int, default=1048576)
    p.add_argument("--shutdown-timeout", type=float, default=10.0)
    return p


def visible_gpus():
    import torch

    count = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [str(i) for i in range(count)]
    devices = [item.strip() for item in visible.split(",") if item.strip() and item.strip() != "-1"]
    if len(devices) != count:
        raise ValueError("CUDA_VISIBLE_DEVICES does not match the accessible GPU inventory")
    return devices


def worker_command(spec, args, worker_args):
    command = [
        sys.executable,
        "-m",
        "minisgl",
        "--model",
        args.model,
        "--worker-role",
        spec.role,
        "--worker-id",
        spec.worker_id,
        "--host",
        args.worker_host,
        "--port",
        str(spec.control_port),
        "--pd-data-port",
        str(spec.data_port),
        "--pd-advertise-host",
        spec.host,
        "--pd-transfer-backend",
        args.pd_transfer_backend,
        "--dtype",
        args.dtype,
        "--pd-timeout",
        str(args.timeout),
        *worker_args,
    ]
    if args.pd_transfer_backend == "mooncake":
        command += [
            "--pd-mooncake-host",
            args.pd_mooncake_host or spec.host,
            "--pd-mooncake-port",
            str(spec.mooncake_port),
        ]
        if spec.rdma_device:
            command += ["--pd-mooncake-device", spec.rdma_device]
    if spec.numa_node is not None:
        numactl = shutil.which("numactl")
        if numactl is None:
            raise ValueError("NUMA placement requires numactl")
        command = [
            numactl,
            f"--cpunodebind={spec.numa_node}",
            f"--membind={spec.numa_node}",
            *command,
        ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=spec.gpu, MINISGL_WORKER_ID=spec.worker_id)
    if args.pd_transfer_backend == "mooncake":
        env.update(MC_MIN_RPC_PORT=str(spec.mooncake_port), MC_MAX_RPC_PORT=str(spec.mooncake_port))
    return command, env


class ClusterSupervisor:
    def __init__(self, args, specs, prefill, decode, worker_args, *, popen=None):
        self.args, self.specs = args, specs
        self.prefill, self.decode, self.worker_args = prefill, decode, worker_args
        self.popen, self.children = popen or subprocess.Popen, []
        self.stop = threading.Event()

    def start(self):
        # Build every command before starting any child, including affinity validation.
        commands = [worker_command(spec, self.args, self.worker_args) for spec in self.specs]
        for spec, (command, env) in zip(self.specs, commands):
            child = self.popen(command, env=env, start_new_session=True)
            self.children.append((spec.worker_id, child, spec.role))
        if self.specs:
            from .router import WorkerPool

            pool = WorkerPool(
                self.prefill,
                self.decode,
                timeout=self.args.timeout,
                health_timeout=self.args.health_timeout,
                health_interval=self.args.health_interval,
                max_quarantined_bytes=self.args.max_quarantined_bytes,
            )
            try:
                deadline = time.monotonic() + self.args.worker_ready_timeout
                while not self.stop.is_set():
                    if any(child.poll() is not None for _, child, _ in self.children):
                        raise RuntimeError("Worker exited during cluster startup")
                    pool.probe_all()
                    if all(node.healthy for node in pool.nodes):
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("cluster Worker readiness timed out")
                    self.stop.wait(self.args.health_interval)
            finally:
                pool.close()
        if self.stop.is_set() or self.args.no_gateway:
            return
        command = [
            sys.executable,
            "-m",
            "minisgl.disagg.gateway",
            "--model",
            self.args.model,
            "--host",
            self.args.host,
            "--port",
            str(self.args.port),
            "--prefill-addrs",
            *self.prefill,
            "--decode-addrs",
            *self.decode,
            "--timeout",
            str(self.args.timeout),
            "--health-interval",
            str(self.args.health_interval),
            "--health-timeout",
            str(self.args.health_timeout),
            "--worker-ready-timeout",
            str(self.args.worker_ready_timeout),
            "--max-quarantined-bytes",
            str(self.args.max_quarantined_bytes),
            "--max-inflight-requests",
            str(self.args.max_inflight_requests),
            "--max-inflight-tokens",
            str(self.args.max_inflight_tokens),
        ]
        if self.args.routing_seed is not None:
            command += ["--routing-seed", str(self.args.routing_seed)]
        self.children.append(
            (
                "gateway",
                self.popen(
                    command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), start_new_session=True
                ),
                "gateway",
            )
        )

    def run(self):
        try:
            self.start()
            announced = set()
            while not self.stop.wait(0.2):
                running = set()
                for name, child, role in self.children:
                    code = child.poll()
                    if code is None:
                        running.add(role)
                    elif name not in announced:
                        announced.add(name)
                        print(
                            json.dumps({"worker_exit": name, "returncode": code}),
                            file=sys.stderr,
                            flush=True,
                        )
                    if code is not None and role == "gateway":
                        raise RuntimeError("gateway exited")
                if self.specs and not {"prefill", "decode"} <= running:
                    raise RuntimeError("cluster lost every Worker of one role")
        finally:
            self.close()

    def close(self):
        self.stop.set()
        for _, child, _ in self.children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + self.args.shutdown_timeout
        for _, child, _ in reversed(self.children):
            try:
                child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()


def main(argv=None):
    p = parser()
    args, worker_args = p.parse_known_args(argv)
    try:
        for value in (
            args.timeout,
            args.health_timeout,
            args.health_interval,
            args.worker_ready_timeout,
            args.shutdown_timeout,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("timeouts and intervals must be finite and positive")
        if (
            not 0 < args.port < 65536
            or args.max_inflight_requests < 1
            or args.max_inflight_tokens < 1
            or args.max_quarantined_bytes < 0
        ):
            raise ValueError("invalid gateway port or admission/quarantine budget")
        prefill, decode = remote_addresses(args)
        if prefill:
            if (
                any(
                    (
                        args.prefill_gpus,
                        args.decode_gpus,
                        args.prefill_numa_nodes,
                        args.decode_numa_nodes,
                        args.prefill_rdma_devices,
                        args.decode_rdma_devices,
                    )
                )
                or worker_args
                or args.no_gateway
            ):
                raise ValueError(
                    "remote-address mode only starts a gateway; local GPU/Worker arguments do not apply"
                )
            specs = []
        else:
            # Worker-owned identity and resource parameters cannot be overridden by passthrough.
            blocked = {
                "--worker-role",
                "--worker-id",
                "--pd-data-port",
                "--pd-mooncake-port",
                "--pd-advertise-host",
            }
            if any(arg.split("=", 1)[0] in blocked for arg in worker_args):
                raise ValueError("cluster launcher owns Worker roles, identities and ports")
            # Validate split flags, TP=1 and model-independent options before spawning.
            from minisgl.server.args import parse_args

            specs = local_plan(args, visible_gpus())
            parse_args(
                [
                    "--model",
                    args.model,
                    "--worker-role",
                    "prefill",
                    *worker_args,
                    "--dtype",
                    "bfloat16",
                    "--model-source",
                    "huggingface",
                ]
            )
            prefill = [spec.endpoint for spec in specs if spec.role == "prefill"]
            decode = [spec.endpoint for spec in specs if spec.role == "decode"]
    except (ValueError, RuntimeError) as exc:
        p.error(str(exc))
    supervisor = ClusterSupervisor(args, specs, prefill, decode, worker_args)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, lambda *_: supervisor.stop.set())
    try:
        print(
            json.dumps(
                {
                    "cluster": [asdict(spec) for spec in specs],
                    "prefill_addrs": prefill,
                    "decode_addrs": decode,
                    "routing_seed": args.routing_seed,
                }
            ),
            flush=True,
        )
        supervisor.run()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
