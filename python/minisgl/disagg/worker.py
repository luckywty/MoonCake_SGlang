"""Serial P/D workers with TCP host payloads or direct Mooncake GPU payloads."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event
from typing import NoReturn

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.scheduler import Scheduler
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.scheduler.utils import PendingReq

from .backend import create_transfer_backend
from .control import CancelledError, Channel, StartRequest
from .lifecycle import RequestLifecycle
from .protocol import KVLayout, KVTransferMetadata, ReceiveTicket, RequestHandoff
from .role import WorkerRole
from .transfer import TransferBackendUnavailable


def checkpoint_fingerprint(directory: str, model_config: dict) -> str:
    root = Path(directory)
    files = sorted(root.glob("*.safetensors"))
    consolidated = [f for f in files if f.name.endswith("consolidated.safetensors")]
    files = consolidated or files
    if not files:
        raise ValueError("split workers require safetensors model weights")
    digest = hashlib.sha256(json.dumps(model_config, sort_keys=True).encode())
    for file in files:
        digest.update(json.dumps([file.name, file.stat().st_size]).encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_split_config(config) -> None:
    if config.worker_role is WorkerRole.UNIFIED:
        raise ValueError("split worker requires prefill or decode role")
    if config.tp_info.size != 1:
        raise ValueError("Phase 2 split workers require TP=1")
    if config.use_dummy_weight or config.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("Phase 2 requires real weights and unquantized KV")
    if config.max_extend_tokens < 1 or config.max_running_req < 1:
        raise ValueError("split worker requires positive prefill and request capacity")


class PDWorkerScheduler(Scheduler):
    def __init__(self, config):
        validate_split_config(config)
        self._gpu_transfer = getattr(config, "pd_transfer_backend", "tcp") == "mooncake"
        if self._gpu_transfer:
            maximum = getattr(config, "pd_max_transfer_bytes", 1024**3)
            reserved = getattr(config, "pd_max_registered_bytes", None)
            reserved = 2 * maximum if reserved is None else reserved
            if type(reserved) is not int or reserved < maximum:
                raise ValueError("GPU transfer reserve must cover the maximum KV payload")
            config = replace(config, pd_transfer_reserve_bytes=reserved)
        model_path = config.model_path
        if not Path(model_path).is_dir():
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                model_path,
                allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken"],
            )
        config = replace(
            config, model_path=model_path, offline_mode=True, cuda_graph_bs=[], cuda_graph_max_bs=0
        )
        fingerprint = checkpoint_fingerprint(model_path, asdict(config.model_config))
        super().__init__(config, _split_worker=True)
        if type(self.engine.kv_cache) is not MHAKVCache or self.engine.kv_cache.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            self.shutdown()
            raise ValueError("Phase 2 requires the MHAKVCache transfer layout")
        self.role = config.worker_role
        pool = self.engine.kv_cache
        self.layout = KVLayout(
            fingerprint,
            str(pool.dtype).removeprefix("torch."),
            pool.num_layers,
            pool.k_cache(0).shape[-2],
            pool.k_cache(0).shape[-1],
            config.page_size,
        )
        self.vocab_size = config.model_config.vocab_size
        self._owned_req = None
        self._allocated_len = 0

    @contextmanager
    def execution(self):
        with self.engine_stream_ctx:
            if self.device.type == "cuda":
                self.engine.stream.wait_stream(self.stream)
            try:
                yield
            finally:
                self.synchronize()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.engine.stream.synchronize()

    def _validate_request(self, ids: tuple[int, ...], max_tokens: int) -> None:
        if len(ids) + max_tokens > self.engine.max_seq_len:
            raise ValueError("prompt and output budget exceed max_seq_len")
        p = self.layout.page_size
        if ((len(ids) + max_tokens + p - 1) // p) * p > self.engine.page_table.shape[1]:
            raise ValueError("request exceeds page-aligned table capacity")
        if any(token >= self.vocab_size for token in ids):
            raise ValueError("input token is outside the model vocabulary")

    def _prepare_owned_batch(self, batch: Batch):
        self._owned_req = batch.reqs[0]
        self.engine.graph_runner.pad_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        self._allocated_len = max(self._allocated_len, self._owned_req.device_len)
        return self._prepare_batch_metadata(batch)

    def _release(self, *, cache: bool) -> None:
        self.synchronize()
        req = self._owned_req
        if req is None:
            return
        self.decode_manager.remove_req(req)
        self.prefill_manager.pending_list.clear()
        if cache:
            self.cache_manager.cache_req(req, finished=True)
        else:
            self.cache_manager.discard_private(req, self._allocated_len)
        self.table_manager.free(req.table_idx)
        self._owned_req = None
        self._allocated_len = 0

    @torch.inference_mode()
    def prefill(
        self, request: StartRequest, channel: Channel, *, backend=None
    ) -> tuple[RequestHandoff, bytes | torch.Tensor]:
        self._validate_request(request.prompt_ids, request.max_tokens)
        params = SamplingParams(max_tokens=request.max_tokens, ignore_eos=request.ignore_eos)
        pending = PendingReq(
            request.uid, torch.tensor(request.prompt_ids, dtype=torch.int32), params
        )
        success = False
        with self.execution():
            self.prefill_manager.pending_list.append(pending)
            try:
                while True:
                    channel.check_connected()
                    batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
                    if batch is None:
                        raise RuntimeError("insufficient KV or request capacity for prefill")
                    self._owned_req = req = batch.reqs[0]
                    self._allocated_len = max(self._allocated_len, req.cache_handle.cached_len)
                    prepared = self._prepare_owned_batch(batch)
                    output = self._forward(prepared)
                    self.decode_manager.remove_req(req)
                    output.copy_done_event.synchronize()
                    if isinstance(req, ChunkedReq):
                        continue
                    req.append_host(output.next_tokens_cpu[:1])
                    slots = self.engine.page_table[req.table_idx, : req.cached_len]
                    meta = KVTransferMetadata.from_slots(
                        request.transfer_id,
                        self.layout,
                        slots.tolist(),
                        num_pool_pages=self.engine.num_pages,
                    )
                    handoff = RequestHandoff(
                        req.uid,
                        request.prompt_ids,
                        int(output.next_tokens_cpu[0]),
                        request.max_tokens,
                        request.ignore_eos,
                        self.eos_token_id,
                        meta,
                    )
                    payload = b""
                    if handoff.remaining_tokens:
                        if getattr(backend, "gpu_payload", getattr(self, "_gpu_transfer", False)):
                            if backend is None or not getattr(backend, "gpu_payload", False):
                                raise ValueError("GPU prefill requires the Mooncake GPU backend")
                            backend.check_capacity(meta.nbytes)
                            payload = self.engine.kv_cache.export_tensor(
                                slots, num_pool_pages=self.engine.num_pages
                            )
                        else:
                            payload = self.engine.kv_cache.export_tokens(
                                slots, num_pool_pages=self.engine.num_pages
                            )
                    success = True
                    return handoff, payload
            finally:
                # _release synchronizes the engine stream. The independent GPU
                # transfer tensor is fully populated before the source pages are
                # cached/freed; its lifetime subsequently belongs to the backend.
                self._release(cache=success)
                self.prefill_manager.pending_list.clear()

    def _reserve_decode(self, handoff: RequestHandoff) -> torch.Tensor:
        if handoff.kv.layout != self.layout or handoff.eos_token_id != self.eos_token_id:
            raise ValueError("incompatible model, KV layout or EOS configuration")
        self._validate_request(handoff.prompt_ids, handoff.max_tokens)
        if handoff.first_token_id >= self.vocab_size or not handoff.remaining_tokens:
            raise ValueError("invalid first token or completed handoff")
        p = self.layout.page_size
        needed = ((len(handoff.prompt_ids) + handoff.max_tokens + p - 1) // p) * p
        if not self.table_manager.available_size or needed > self.cache_manager.available_size:
            raise RuntimeError("insufficient decode capacity")
        table_idx = self.table_manager.allocate()
        slots = None
        try:
            slots = self.cache_manager.reserve_import(table_idx, handoff.kv.kv_tokens)
            handle = self.cache_manager.prefix_cache.match_prefix(
                torch.empty(0, dtype=torch.int32)
            ).cuda_handle
            self.cache_manager.lock(handle)
            req = Req(
                input_ids=torch.tensor(
                    (*handoff.prompt_ids, handoff.first_token_id), dtype=torch.int32
                ),
                table_idx=table_idx,
                cached_len=handoff.kv.kv_tokens,
                output_len=handoff.remaining_tokens,
                uid=handoff.uid,
                sampling_params=SamplingParams(
                    max_tokens=handoff.max_tokens, ignore_eos=handoff.ignore_eos
                ),
                cache_handle=handle,
            )
            self._owned_req = req
            self._allocated_len = handoff.kv.kv_tokens
            return slots[: handoff.kv.kv_tokens]
        except Exception:
            if slots is not None:
                self.cache_manager._free(slots)
            self.table_manager.free(table_idx)
            raise

    @torch.inference_mode()
    def decode(self, handoff: RequestHandoff, channel: Channel, backend) -> None:
        ticket = None
        success = False
        state = RequestLifecycle(handoff.uid, handoff.kv.transfer_id)
        state.begin_prefill()
        state.complete_prefill(handoff)
        with self.execution():
            try:
                limit = getattr(backend, "max_transfer_bytes", None)
                if limit is not None and handoff.kv.nbytes > limit:
                    raise ValueError("KV payload exceeds the backend transfer limit")
                gpu_transfer = getattr(backend, "gpu_payload", False)
                if gpu_transfer:
                    if self.device.type != "cuda":
                        raise ValueError("Mooncake GPU transfer requires a CUDA scheduler")
                    backend.check_capacity(handoff.kv.nbytes)
                slots = self._reserve_decode(handoff)
                staging = (
                    torch.empty(handoff.kv.nbytes, dtype=torch.uint8, device=self.device)
                    if gpu_transfer
                    else bytearray(handoff.kv.nbytes)
                )
                ticket = backend.prepare_receive(handoff.kv, self.layout, staging)
                state.begin_transfer()
                channel.send({"status": "ready", "ticket": json.loads(ticket.to_wire())})
                if channel.receive() != {"op": "commit", "transfer_id": handoff.kv.transfer_id}:
                    raise ValueError("invalid transfer commit")
                backend.finish_receive(ticket)
                if gpu_transfer:
                    self.engine.kv_cache.import_tensor(
                        staging, slots, num_pool_pages=self.engine.num_pages
                    )
                else:
                    self.engine.kv_cache.import_tokens(
                        staging, slots, num_pool_pages=self.engine.num_pages
                    )
                req = self._owned_req
                self.token_pool[req.table_idx, : req.device_len] = req.input_ids.to(self.device)
                self.synchronize()
                # The GPU scatter has finished. Keep no unnecessary copy of the
                # prompt KV during the subsequent token-generation loop.
                staging = None
                state.accept_receive(handoff.kv.transfer_id)
                self.decode_manager.filter_reqs([req])
                state.begin_decode()
                channel.send({"status": "installed", "transfer_id": handoff.kv.transfer_id})
                for index in range(1, handoff.max_tokens):
                    channel.check_connected()
                    batch = self.decode_manager.schedule_next_batch()
                    output = self._forward(self._prepare_owned_batch(batch))
                    output.copy_done_event.synchronize()
                    req.append_host(output.next_tokens_cpu[:1])
                    token = int(output.next_tokens_cpu[0])
                    finished = not req.can_decode or (
                        not handoff.ignore_eos and token == self.eos_token_id
                    )
                    channel.send(
                        {
                            "status": "token",
                            "transfer_id": handoff.kv.transfer_id,
                            "index": index,
                            "token_id": token,
                            "finished": finished,
                        }
                    )
                    if finished:
                        state.finish_decode()
                        success = True
                        break
            except CancelledError:
                state.cancel()
                raise
            except Exception:
                state.fail()
                raise
            finally:
                try:
                    if ticket is not None:
                        backend.discard_receive(ticket)
                finally:
                    self._release(cache=success)

    def health(self) -> dict:
        self.cache_manager.check_integrity()
        return {
            "status": "ready",
            "role": self.role.value,
            "layout": asdict(self.layout),
            "free_rows": self.table_manager.available_size,
            "free_pages": len(self.cache_manager.free_slots),
            "num_pages": self.engine.num_pages,
            "protected_tokens": self.cache_manager.prefix_cache.size_info.protected_size,
            "evictable_tokens": self.cache_manager.prefix_cache.size_info.evictable_size,
        }


def serve_worker(
    scheduler, listener: socket.socket, backend, timeout: float, stop: Event | None = None
):
    listener.settimeout(0.2)
    recent = deque(maxlen=1024)
    while stop is None or not stop.is_set():
        try:
            sock, _ = listener.accept()
        except socket.timeout:
            check_available = getattr(backend, "check_available", None)
            if check_available is not None:
                check_available()
            continue
        with sock:
            channel = Channel(sock, timeout)
            try:
                message = channel.receive()
                if message == {"op": "health"}:
                    native_health = getattr(backend, "health", lambda: {})()
                    status = native_health.get("status", "ready")
                    channel.send(
                        {
                            **scheduler.health(),
                            "status": status,
                            "transfer_backend": getattr(backend, "name", "tcp"),
                            "transfer": native_health,
                        }
                    )
                    continue
                check_available = getattr(backend, "check_available", None)
                if check_available is not None:
                    check_available()
                if (
                    scheduler.role is WorkerRole.PREFILL
                    and set(message) == {"op", "request"}
                    and message["op"] == "prefill"
                ):
                    request = StartRequest.from_object(message["request"])
                    transfer_id = request.transfer_id
                    if transfer_id in recent:
                        raise ValueError("duplicate transfer attempt")
                    recent.append(transfer_id)
                    if getattr(backend, "gpu_payload", False):
                        handoff, payload = scheduler.prefill(request, channel, backend=backend)
                    else:
                        handoff, payload = scheduler.prefill(request, channel)
                    channel.send({"status": "handoff", "handoff": json.loads(handoff.to_wire())})
                    if handoff.remaining_tokens:
                        target = channel.receive()
                        if set(target) != {"op", "ticket"} or target["op"] != "send":
                            raise ValueError("expected a receive ticket")
                        ticket = ReceiveTicket.from_wire(json.dumps(target["ticket"]).encode())

                        # payload is bytes for TCP and an independent CUDA Tensor
                        # for Mooncake; ticket identifies D's receive allocation.
                        # `ticket` 包含 D 端的接收地址与接收凭据。
                        #使用普通 TCP 时，KV 直接通过两个 Worker 之间的 TCP socket 传输，
                        # P Worker ───── TCP：KV 字节 ─────→ D Worker
                        backend.send(
                            handoff.kv, payload, ticket, cancel_check=channel.check_connected
                        )
                        channel.send({"status": "sent", "transfer_id": transfer_id})
                elif (
                    scheduler.role is WorkerRole.DECODE
                    and set(message) == {"op", "handoff"}
                    and message["op"] == "decode"
                ):
                    handoff = RequestHandoff.from_wire(json.dumps(message["handoff"]).encode())
                    transfer_id = handoff.kv.transfer_id
                    if transfer_id in recent:
                        raise ValueError("duplicate transfer attempt")
                    recent.append(transfer_id)
                    scheduler.decode(handoff, channel, backend)
                else:
                    raise ValueError("operation does not match worker role")
            except TransferBackendUnavailable as exc:
                try:
                    channel.send({"status": "error", "error": str(exc)})
                except OSError:
                    pass
                raise
            except (OSError, RuntimeError, ValueError) as exc:
                try:
                    channel.send({"status": "error", "error": str(exc)})
                except OSError:
                    pass
                if scheduler._owned_req is not None:
                    raise RuntimeError(
                        "split worker could not release its request resources"
                    ) from exc
            finally:
                payload = None


def _exit_unavailable_worker(exc: BaseException) -> NoReturn:
    # Unknown native work retains CUDA Tensor/ticket references in the runtime.
    # Python destruction can block with the GIL held. Process exit is the final
    # boundary; do not run normal object teardown on that failure path.
    print(f"P/D transfer backend requires Worker restart: {exc}", file=sys.stderr, flush=True)
    os._exit(1)


def launch_split_worker(args) -> None:
    backend = None
    scheduler = None
    try:
        with socket.create_server((args.server_host, args.server_port)) as listener:
            backend = create_transfer_backend(args)
            with torch.inference_mode():
                scheduler = PDWorkerScheduler(args)
                bind_device = getattr(backend, "bind_device", None)
                if bind_device is not None:
                    # Engine must initialize CUDA first: Engine.__init__ rejects
                    # an already initialized torch.cuda context.
                    bind_device(scheduler.device)
                try:
                    print(
                        json.dumps(
                            {
                                "pd_worker": args.worker_role.value,
                                "status": "ready",
                                "port": args.server_port,
                                "data_endpoint": backend.endpoint,
                                "transfer_backend": getattr(backend, "name", "tcp"),
                                "worker_pid": os.getpid(),
                                "kv_memory": (
                                    "cuda" if getattr(backend, "gpu_payload", False) else "host"
                                ),
                                "transfer": getattr(backend, "health", lambda: {})(),
                                "cuda_graph": False,
                                "overlap": False,
                            }
                        ),
                        flush=True,
                    )
                    serve_worker(scheduler, listener, backend, args.pd_timeout)
                except KeyboardInterrupt:
                    pass
    except TransferBackendUnavailable as exc:
        _exit_unavailable_worker(exc)
    finally:
        try:
            if backend is not None:
                backend.close()
        except TransferBackendUnavailable as exc:
            _exit_unavailable_worker(exc)
        finally:
            if scheduler is not None:
                scheduler.shutdown()
