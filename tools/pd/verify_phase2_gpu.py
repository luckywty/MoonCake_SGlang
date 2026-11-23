"""Compare actual GPU Unified serving against two independent P/D worker processes."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def cases():
    return [
        {"prompt_ids": ids, "max_tokens": count, "ignore_eos": ignore}
        for ids in ([1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7])
        for count, ignore in ((1, False), (6, True))
    ]


def baseline(args):
    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU validation requires a CUDA GPU and the full mini-SGLang dependencies"
        )

    class RawTokenLLM(LLM):
        def offline_send_result(self, reply):
            # Record the actual sampled EOS token as well, matching the handoff protocol.
            for item in reply:
                self.status_map[item.uid].output_ids.append(item.next_token)

    llm = RawTokenLLM(
        args.model,
        dtype=torch.bfloat16,
        num_page_override=512,
        max_running_req=4,
        max_seq_len_override=64,
        cuda_graph_max_bs=0,
        page_size=args.page_size,
        max_extend_tokens=3,
        attention_backend="fi",
        cache_type=args.cache_type,
    )
    records = []
    try:
        for case in cases():
            result = llm.generate(
                [case["prompt_ids"]],
                SamplingParams(max_tokens=case["max_tokens"], ignore_eos=case["ignore_eos"]),
            )[0]
            records.append({**case, "token_ids": result["token_ids"]})
    finally:
        llm.shutdown()
    args.baseline_output.write_text(json.dumps(records, indent=2) + "\n")


def free_port_triplet():
    for _ in range(100):
        with socket.create_server(("127.0.0.1", 0)) as probe:
            port = probe.getsockname()[1]
            if port > 65533:
                continue
            try:
                with socket.create_server(("127.0.0.1", port + 1)):
                    with socket.create_server(("127.0.0.1", port + 2)):
                        return port
            except OSError:
                pass
    raise RuntimeError("no free worker/distributed/data port triplet found")


def wait_ready(process, log: Path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log.read_text(errors="replace")
        if process.poll() is not None:
            raise RuntimeError(f"worker exited with {process.returncode}: {text[-4000:]}")
        if '"pd_worker":' in text and '"status": "ready"' in text:
            return
        time.sleep(0.2)
    raise TimeoutError(f"worker startup timed out; inspect {log}")


def run(args):
    from minisgl.disagg.client import PDClient
    from minisgl.disagg.control import Channel

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "gpu_results.json").unlink(missing_ok=True)
    records_path = output / "unified_tokens.json"
    env = {**os.environ, "MINISGL_DISABLE_OVERLAP_SCHEDULING": "1"}
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--page-size",
        str(args.page_size),
        "--cache-type",
        args.cache_type,
        "--baseline-output",
        str(records_path),
    ]
    with (output / "unified.log").open("w") as log:
        result = subprocess.run(
            command,
            env={**env, "CUDA_VISIBLE_DEVICES": args.prefill_device},
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=args.startup_timeout,
        )
    if result.returncode:
        raise RuntimeError(f"Unified baseline failed; inspect {output / 'unified.log'}")
    ports = []
    while len(ports) < 2:
        port = free_port_triplet()
        if not any(abs(port - existing) < 3 for existing in ports):
            ports.append(port)
    processes, logs = [], []
    try:
        for role, device, port in zip(
            ("prefill", "decode"), (args.prefill_device, args.decode_device), ports
        ):
            path = output / f"{role}.log"
            log = path.open("w")
            logs.append(log)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "minisgl",
                    "--model",
                    args.model,
                    "--worker-role",
                    role,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--dtype",
                    "bfloat16",
                    "--attention-backend",
                    "fi",
                    "--num-pages",
                    "512",
                    "--page-size",
                    str(args.page_size),
                    "--cache-type",
                    args.cache_type,
                    "--max-running-requests",
                    "4",
                    "--max-seq-len-override",
                    "64",
                    "--max-prefill-length",
                    "3",
                    "--cuda-graph-max-bs",
                    "0",
                    "--pd-transfer-backend", args.transfer_backend,
                    "--pd-layers-per-chunk", str(args.layers_per_chunk),
                    "--pd-staging-slots", str(args.staging_slots),
                    "--pd-mooncake-host", args.mooncake_host,
                    "--pd-mooncake-device", args.mooncake_device,
                    "--pd-max-transfer-bytes", str(args.max_transfer_bytes),
                    "--pd-max-registered-bytes", str(2 * args.max_transfer_bytes),
                ],
                env={**env, "CUDA_VISIBLE_DEVICES": device},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            processes.append(process)
        for role, process in zip(("prefill", "decode"), processes):
            wait_ready(process, output / f"{role}.log", args.startup_timeout)
        client = PDClient(*(f"127.0.0.1:{port}" for port in ports))
        results = []
        def health(port):
            with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
                channel = Channel(sock, 10)
                channel.send({"op": "health"})
                return channel.receive()
        startup_health = [health(port) for port in ports]
        for expected in json.loads(records_path.read_text()):
            actual = client.generate(
                expected["prompt_ids"], expected["max_tokens"], ignore_eos=expected["ignore_eos"]
            )
            results.append({**expected, "pd_token_ids": actual})
            if actual != expected["token_ids"]:
                raise AssertionError(f"greedy token mismatch: {results[-1]}")
        final_health = [health(port) for port in ports]
        for initial, state in zip(startup_health, final_health):
            if (state.get("status") != "ready" or state["free_rows"] != 4
                    or state["protected_tokens"] != 0):
                raise AssertionError(f"worker resource leak: {state}")
            if args.transfer_backend == "mooncake":
                before = initial["transfer"]["mooncake"]
                after = state["transfer"]["mooncake"]
                if (after["registered_bytes"] != before["registered_bytes"]
                        or after["registered_buffers"] != before["registered_buffers"]
                        or after["pending_operations"] or after["quarantined_buffers"]
                        or after["layered"]["receive_reserved"]):
                    raise AssertionError(f"layer transfer resource leak: {state}")
        report = {
            "result": "PASS",
            "page_size": args.page_size,
            "cache_type": args.cache_type,
            "prefill_device": args.prefill_device,
            "decode_device": args.decode_device,
            "transfer_backend": args.transfer_backend,
            "layers_per_chunk": args.layers_per_chunk,
            "startup_health": startup_health,
            "final_health": final_health,
            "cases": results,
        }
        (output / "gpu_results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefill-device", default="0")
    parser.add_argument("--decode-device", default="0")
    parser.add_argument("--page-size", type=int, choices=(1, 4), default=1)
    parser.add_argument("--cache-type", choices=("radix", "naive"), default="radix")
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2"))
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--transfer-backend", choices=("tcp", "mooncake"), default="tcp")
    parser.add_argument("--layers-per-chunk", type=int, default=1)
    parser.add_argument("--staging-slots", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--mooncake-host", default="127.0.0.1")
    parser.add_argument("--mooncake-device", default="")
    parser.add_argument("--max-transfer-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--baseline-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.baseline_output:
        baseline(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
