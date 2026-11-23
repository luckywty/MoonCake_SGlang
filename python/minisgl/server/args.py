from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from minisgl.disagg.role import WorkerRole
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False
    pd_timeout: float = 120.0
    pd_data_port: int | None = None
    pd_advertise_host: str | None = None
    pd_transfer_backend: str = "tcp"
    pd_mooncake_host: str | None = None
    pd_max_transfer_bytes: int = 1024**3
    pd_max_registered_bytes: int | None = None
    pd_max_pending_transfers: int = 32
    pd_max_quarantine_seconds: float = 300.0
    pd_mooncake_device: str = ""
    pd_layers_per_chunk: int = 1
    pd_staging_slots: int = 3
    pd_max_queued_requests: int = 128
    pd_max_queued_tokens: int | None = None
    pd_batch_wait_ms: float = 2.0
    pd_output_queue_size: int = 128

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from minisgl.attention import validate_attn_backend
    from minisgl.kvcache import SUPPORTED_CACHE_MANAGER
    from minisgl.moe import SUPPORTED_MOE_BACKENDS

    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    parser.add_argument(
        "--worker-role",
        choices=[role.value for role in WorkerRole],
        default=WorkerRole.UNIFIED.value,
        help="Worker role. Split roles expose the P/D control protocol on --host/--port.",
    )
    parser.add_argument("--pd-timeout", type=float, default=ServerArgs.pd_timeout)
    parser.add_argument("--pd-data-port", type=int, default=None)
    parser.add_argument("--pd-advertise-host", default=None)
    parser.add_argument("--pd-transfer-backend", choices=("tcp", "mooncake"), default="tcp")
    parser.add_argument("--pd-mooncake-host", default=None)
    parser.add_argument("--pd-layers-per-chunk", type=int, default=ServerArgs.pd_layers_per_chunk)
    parser.add_argument("--pd-staging-slots", type=int, default=ServerArgs.pd_staging_slots)
    parser.add_argument(
        "--pd-max-queued-requests", type=int, default=ServerArgs.pd_max_queued_requests
    )
    parser.add_argument("--pd-max-queued-tokens", type=int, default=None)
    parser.add_argument("--pd-batch-wait-ms", type=float, default=ServerArgs.pd_batch_wait_ms)
    parser.add_argument("--pd-output-queue-size", type=int, default=ServerArgs.pd_output_queue_size)
    parser.add_argument(
        "--pd-max-transfer-bytes", type=int, default=ServerArgs.pd_max_transfer_bytes
    )
    parser.add_argument(
        "--pd-max-registered-bytes",
        type=int,
        default=None,
        help="GPU transfer allocation budget per Mooncake worker; defaults to twice the payload limit.",
    )
    parser.add_argument(
        "--pd-max-pending-transfers", type=int, default=ServerArgs.pd_max_pending_transfers
    )
    parser.add_argument(
        "--pd-max-quarantine-seconds", type=float, default=ServerArgs.pd_max_quarantine_seconds
    )
    parser.add_argument(
        "--pd-mooncake-device",
        default=ServerArgs.pd_mooncake_device,
        help="Comma-separated RDMA HCA names, such as mlx5_0; not a CUDA device ordinal.",
    )

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()
    if kwargs["worker_role"] == WorkerRole.UNIFIED.value and kwargs["pd_transfer_backend"] != "tcp":
        parser.error("--pd-transfer-backend mooncake requires a split worker role")
    if kwargs["worker_role"] != WorkerRole.UNIFIED.value:
        if kwargs["tensor_parallel_size"] != 1 or kwargs["use_dummy_weight"]:
            parser.error("Phase 2 split workers require TP=1 and real model weights")
        if kwargs["shell_mode"] or run_shell:
            parser.error("split workers do not support shell mode")
        if (
            not math.isfinite(kwargs["pd_timeout"])
            or kwargs["pd_timeout"] <= 0
            or not 0 < kwargs["server_port"] < 65535
        ):
            parser.error("invalid split worker timeout or port")
        if kwargs["pd_max_transfer_bytes"] <= 0:
            parser.error("--pd-max-transfer-bytes must be positive")
        if (
            kwargs["pd_max_registered_bytes"] is not None
            and kwargs["pd_max_registered_bytes"] < kwargs["pd_max_transfer_bytes"]
        ):
            parser.error("--pd-max-registered-bytes must cover --pd-max-transfer-bytes")
        if kwargs["pd_max_pending_transfers"] <= 0:
            parser.error("--pd-max-pending-transfers must be positive")
        if kwargs["pd_layers_per_chunk"] <= 0:
            parser.error("--pd-layers-per-chunk must be positive")
        if not 2 <= kwargs["pd_staging_slots"] <= 4:
            parser.error("--pd-staging-slots must be between 2 and 4")
        if kwargs["pd_max_queued_requests"] < 1 or kwargs["pd_output_queue_size"] < 1:
            parser.error("P/D request and output queue limits must be positive")
        if kwargs["pd_max_queued_tokens"] is not None and kwargs["pd_max_queued_tokens"] < 1:
            parser.error("--pd-max-queued-tokens must be positive")
        if not math.isfinite(kwargs["pd_batch_wait_ms"]) or kwargs["pd_batch_wait_ms"] < 0:
            parser.error("--pd-batch-wait-ms must be finite and nonnegative")
        if (
            not math.isfinite(kwargs["pd_max_quarantine_seconds"])
            or kwargs["pd_max_quarantine_seconds"] <= 0
        ):
            parser.error("--pd-max-quarantine-seconds must be finite and positive")
        kwargs["pd_mooncake_device"] = kwargs["pd_mooncake_device"].strip()
        if kwargs["pd_mooncake_device"] and any(
            not name.strip() for name in kwargs["pd_mooncake_device"].split(",")
        ):
            parser.error("--pd-mooncake-device must contain nonempty RDMA HCA names")
        kwargs["pd_mooncake_device"] = ",".join(
            name.strip() for name in kwargs["pd_mooncake_device"].split(",")
        )
        if kwargs["pd_transfer_backend"] == "mooncake":
            local_host = (
                kwargs["pd_mooncake_host"] or kwargs["pd_advertise_host"] or kwargs["server_host"]
            )
            if not local_host.strip() or local_host in ("0.0.0.0", "::"):
                parser.error("Mooncake requires --pd-mooncake-host for a wildcard bind")
            if kwargs["pd_max_registered_bytes"] is None:
                kwargs["pd_max_registered_bytes"] = 2 * kwargs["pd_max_transfer_bytes"]
            kwargs["pd_transfer_reserve_bytes"] = kwargs["pd_max_registered_bytes"]
        data_port = (
            kwargs["server_port"] + 2 if kwargs["pd_data_port"] is None else kwargs["pd_data_port"]
        )
        if kwargs["worker_role"] == WorkerRole.DECODE.value:
            if not 0 < data_port < 65536 or data_port in (
                kwargs["server_port"],
                kwargs["server_port"] + 1,
            ):
                parser.error("KV data port must differ from the worker and distributed ports")
            if kwargs["server_host"] in ("0.0.0.0", "::") and not kwargs["pd_advertise_host"]:
                parser.error("wildcard decode bind requires --pd-advertise-host")

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    if (dtype_str := kwargs["dtype"]) == "auto":
        from minisgl.utils import cached_load_hf_config

        dtype_str = cached_load_hf_config(kwargs["model_path"]).dtype

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell
