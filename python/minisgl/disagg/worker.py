"""P/D scheduler adapters and the concurrent Worker service entrypoint."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from contextlib import contextmanager, nullcontext
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
from .protocol import MOONCAKE_TRANSPORT_MODES, KVLayout, KVTransferMetadata, RequestHandoff
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
    mode = getattr(config, "pd_mooncake_transport_mode", "gpu_direct")
    if mode not in MOONCAKE_TRANSPORT_MODES or (
        mode == "host_dram" and getattr(config, "pd_transfer_backend", "tcp") != "mooncake"
    ):
        raise ValueError("host_dram requires explicit Mooncake transport configuration")
    if getattr(config, "pd_transfer_backend", "tcp") == "shm_dram":
        import math

        from .cluster_config import shm_staging_plan
        from .shm_relay_pool import RelayManifest

        if (
            not getattr(config, "pd_shm_manifest", None)
            or not math.isfinite(config.pd_shm_local_wait_ms)
            or config.pd_shm_local_wait_ms < 0
            or not math.isfinite(config.pd_cross_numa_penalty)
            or config.pd_cross_numa_penalty < 0
            or type(config.pd_shm_overflow_slots) is not int
            or config.pd_shm_overflow_slots < 0
            or type(config.pd_staging_slots) is not int
            or not 2 <= config.pd_staging_slots <= 4
            or type(config.pd_layers_per_chunk) is not int
            or config.pd_layers_per_chunk < 1
        ):
            raise ValueError(
                "shm_dram requires a launcher manifest and valid NUMA/overflow options"
            )
        manifest = RelayManifest.from_json(Path(config.pd_shm_manifest).read_text())
        shm_staging_plan(config, manifest=manifest)  # Before weights/CUDA/SHM registration.


class PDWorkerScheduler(Scheduler):
    def __init__(self, config):
        validate_split_config(config)
        from .dram_protocol import DramConfig

        dram = getattr(config, "pd_dram", DramConfig())
        if dram.remote_enabled:
            model = config.model_config
            # Split Workers require TP=1. Check before loading weights or constructing CUDA pools.
            dram.validate_layout(
                2 * model.num_layers * model.num_kv_heads * model.head_dim * config.dtype.itemsize,
                config.page_size,
            )
        if dram.enabled:
            config = replace(
                config, pd_dram_staging_pages=2 * dram.staging_slots * dram.batch_pages
            )
        self.decode_radix_cache = bool(
            getattr(config, "disaggregation_decode_enable_radix_cache", False)
        )
        if config.worker_role is WorkerRole.DECODE:
            config = replace(config, cache_type="radix" if self.decode_radix_cache else "naive")
        elif self.decode_radix_cache:
            raise ValueError("decode Radix caching requires a decode Worker")
        shm = getattr(config, "pd_transfer_backend", "tcp") == "shm_dram"
        self._gpu_transfer = getattr(config, "pd_transfer_backend", "tcp") in (
            "mooncake",
            "shm_dram",
        )
        self.transfer_mode = (
            "shm_dram" if shm else getattr(config, "pd_mooncake_transport_mode", "gpu_direct")
        )
        if self._gpu_transfer:
            maximum = getattr(config, "pd_max_transfer_bytes", 1024**3)
            reserved = getattr(config, "pd_max_registered_bytes", None)
            reserved = 2 * maximum if reserved is None else reserved
            if type(reserved) is not int or reserved < (1 if shm else maximum):
                raise ValueError("GPU transfer reserve must cover the maximum KV payload")
            if shm:
                from .cluster_config import shm_staging_plan
                from .shm_relay_pool import RelayManifest

                manifest = RelayManifest.from_json(Path(config.pd_shm_manifest).read_text())
                plan = shm_staging_plan(config, manifest=manifest)
                config = replace(config, pd_transfer_reserve_bytes=plan.gpu_bytes)
            elif self.transfer_mode == "host_dram":
                from .cluster_config import host_staging_plan

                model = config.model_config
                plan = host_staging_plan(
                    max_seq_len=config.max_seq_len,
                    bytes_per_token=2
                    * model.num_layers
                    * model.num_kv_heads
                    * model.head_dim
                    * config.dtype.itemsize,
                    layers=model.num_layers,
                    layers_per_chunk=getattr(config, "pd_layers_per_chunk", 1),
                    staging_slots=getattr(config, "pd_staging_slots", 3),
                    max_transfer_bytes=maximum,
                    max_registered_bytes=reserved,
                    rows=config.max_running_req,
                    prefill=config.worker_role is WorkerRole.PREFILL,
                )
                config = replace(config, pd_transfer_reserve_bytes=plan.gpu_bytes)
            else:
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
        self.worker_id = getattr(config, "worker_id", "")
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
        self.pd_service_config = {
            "numa_id": getattr(config, "pd_numa_id", 0),
            "host_id": getattr(config, "pd_host_id", ""),
            "dram": asdict(dram),
            "cache_namespace": getattr(config, "pd_cache_namespace", "default"),
            "prefix_max_pages": getattr(config, "pd_prefix_max_pages", 65536),
            "prefix_event_limit": getattr(config, "pd_prefix_event_limit", 32768),
            "decode_radix_cache": self.decode_radix_cache,
            "max_queued_requests": getattr(config, "pd_max_queued_requests", 128),
            "max_queued_tokens": getattr(config, "pd_max_queued_tokens", None),
            "batch_wait_ms": getattr(config, "pd_batch_wait_ms", 2.0),
            "output_queue_size": getattr(config, "pd_output_queue_size", 128),
        }

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
        self, request: StartRequest, channel: Channel, *, backend=None, layered=None
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
                    pool = self.engine.kv_cache
                    if layered is not None and not isinstance(req, ChunkedReq):
                        # Earlier prompt chunks and borrowed prefix KV already
                        # exist. In the final chunk each layer becomes complete
                        # for the full prompt, so send it exactly once.
                        layered.set_slots(
                            self.engine.page_table[req.table_idx, : req.device_len].to(torch.int64)
                        )
                        pool.layer_ready = layered.layer_ready
                    try:
                        output = self._forward(prepared)
                    finally:
                        pool.layer_ready = None
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
                        transfer_mode=getattr(self, "transfer_mode", "gpu_direct"),
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
                    if handoff.remaining_tokens and layered is None:
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
                # Layer jobs may still gather from the source pool. Their owner
                # drains/cancels them before releasing the request's pages.
                if layered is None:
                    self._release(cache=success)
                self.prefill_manager.pending_list.clear()

    def reserve_layered_request(self, request: StartRequest):
        """Reserve D's private KV pages before P has produced its first token."""
        self._validate_request(request.prompt_ids, request.max_tokens)
        p = self.layout.page_size
        needed = ((len(request.prompt_ids) + request.max_tokens + p - 1) // p) * p
        if not self.table_manager.available_size or needed > self.cache_manager.available_size:
            raise RuntimeError("insufficient decode capacity")
        table_idx = self.table_manager.allocate()
        slots = None
        try:
            slots = self.cache_manager.reserve_import(table_idx, len(request.prompt_ids))
            handle = self.cache_manager.prefix_cache.match_prefix(
                torch.empty(0, dtype=torch.int32)
            ).cuda_handle
            self.cache_manager.lock(handle)
            self._owned_req = Req(
                input_ids=torch.tensor((*request.prompt_ids, 0), dtype=torch.int32),
                table_idx=table_idx,
                cached_len=len(request.prompt_ids),
                output_len=request.max_tokens - 1,
                uid=request.uid,
                sampling_params=SamplingParams(
                    max_tokens=request.max_tokens, ignore_eos=request.ignore_eos
                ),
                cache_handle=handle,
            )
            self._allocated_len = len(request.prompt_ids)
            return slots[: len(request.prompt_ids)]
        except Exception:
            if slots is not None:
                self.cache_manager._free(slots)
            self.table_manager.free(table_idx)
            raise

    @torch.inference_mode()
    def decode_layered(self, request, channel, backend):
        ticket = None
        success = False
        with self.execution():
            try:
                slots = self.reserve_layered_request(request)
                ticket = backend.reserve_layers(request, self.eos_token_id)
                channel.send({"status": "ready", "ticket": json.loads(ticket.to_wire())})
                message = channel.receive()
                if message == {"op": "release", "transfer_id": request.transfer_id}:
                    backend.release_layers(ticket)
                    ticket = None
                    channel.send({"status": "released", "transfer_id": request.transfer_id})
                    return
                if set(message) != {"op", "transfer_id", "handoff"} or (
                    message["op"] != "commit" or message["transfer_id"] != request.transfer_id
                ):
                    raise ValueError("invalid layered commit")
                handoff = RequestHandoff.from_wire(json.dumps(message["handoff"]).encode())
                if (
                    handoff.uid != request.uid
                    or handoff.prompt_ids != request.prompt_ids
                    or handoff.max_tokens != request.max_tokens
                    or handoff.ignore_eos != request.ignore_eos
                    or handoff.kv.transfer_id != request.transfer_id
                    or handoff.kv.layout != self.layout
                    or handoff.kv.transfer_mode != backend.transfer_mode
                    or handoff.eos_token_id != self.eos_token_id
                    or not handoff.remaining_tokens
                    or handoff.first_token_id >= self.vocab_size
                ):
                    raise ValueError("layered handoff does not match its reservation")
                staging = backend.finish_layers(ticket)
                install_stream = (
                    backend.install_stream
                    if backend.transfer_mode == "host_dram" and self.device.type == "cuda"
                    else None
                )
                try:
                    if install_stream is not None:
                        install_stream.wait_stream(self.engine.stream)
                    with (
                        torch.cuda.stream(install_stream)
                        if install_stream is not None
                        else nullcontext()
                    ):
                        backend.install_layers(
                            self.engine.kv_cache,
                            staging,
                            slots,
                            num_pool_pages=self.engine.num_pages,
                        )
                finally:
                    if install_stream is not None:
                        self.engine.stream.wait_stream(install_stream)
                req = self._owned_req
                req.input_ids[-1] = handoff.first_token_id
                self.token_pool[req.table_idx, : req.device_len] = req.input_ids.to(self.device)
                self.synchronize()
                # Pool reuse is legal only after the scatter consumer finished.
                staging = None
                backend.release_layers(ticket, consumer_complete=True)
                ticket = None
                self.decode_manager.filter_reqs([req])
                channel.send({"status": "installed", "transfer_id": request.transfer_id})
                self._decode_tokens(handoff, channel)
                success = True
            finally:
                if ticket is not None:
                    # An import may raise after enqueueing some CUDA copies.
                    # Prove that those readers stopped before returning the pool.
                    try:
                        self.synchronize()
                    except BaseException as exc:
                        backend.release_layers(ticket)
                        backend.runtime.retire("CUDA layer installation completion is unknown")
                        raise TransferBackendUnavailable(
                            "CUDA layer installation did not synchronize"
                        ) from exc
                try:
                    if ticket is not None:
                        backend.release_layers(ticket, consumer_complete=True)
                finally:
                    self._release(cache=success)

    def _decode_tokens(self, handoff, channel):
        req = self._owned_req
        for index in range(1, handoff.max_tokens):
            channel.check_connected()
            batch = self.decode_manager.schedule_next_batch()
            output = self._forward(self._prepare_owned_batch(batch))
            output.copy_done_event.synchronize()
            req.append_host(output.next_tokens_cpu[:1])
            token = int(output.next_tokens_cpu[0])
            finished = not req.can_decode or (
                not handoff.ignore_eos and token == handoff.eos_token_id
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
                return

    def _reserve_decode(self, handoff: RequestHandoff) -> torch.Tensor:
        if handoff.kv.prefix_len:
            raise ValueError("suffix handoffs require the concurrent WorkerService")
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


def _serve_layered_prefill(scheduler, request, ticket, channel, backend):
    scheduler._validate_request(request.prompt_ids, request.max_tokens)
    if ticket.eos_token_id != scheduler.eos_token_id:
        raise ValueError("incompatible EOS configuration")
    job = backend.begin_layer_send(
        ticket, request, scheduler.engine.kv_cache, channel.check_connected
    )
    prefill_ok = False
    try:
        handoff, _ = scheduler.prefill(request, channel, backend=backend, layered=job)
        prefill_ok = True
        # Return token zero before waiting for the sender's tail convergence.
        channel.send({"status": "handoff", "handoff": json.loads(handoff.to_wire())})
        job.finish(cancel=not handoff.remaining_tokens)
        channel.send(
            {
                "status": "sent" if handoff.remaining_tokens else "retired",
                "transfer_id": request.transfer_id,
            }
        )
    finally:
        try:
            if not job.done.is_set():
                job.finish(cancel=True)
        finally:
            if not job.done.is_set() or not job.safe_sources:
                raise TransferBackendUnavailable(
                    "layer packing may still access the request's KV pages"
                )
            scheduler._release(cache=prefill_ok)


def serve_worker(
    scheduler, listener: socket.socket, backend, timeout: float, stop: Event | None = None
):
    from .service import WorkerService

    WorkerService(scheduler, backend, timeout).run(listener, stop)


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
        if args.pd_transfer_backend == "shm_dram":
            from .numa import bind_worker

            validate_split_config(args)
            bind_worker(args.pd_numa_id)
        with socket.create_server((args.server_host, args.server_port)) as listener:
            backend = create_transfer_backend(args)
            with torch.inference_mode():
                scheduler = PDWorkerScheduler(args)
                bind_device = getattr(backend, "bind_device", None)
                if bind_device is not None:
                    # Engine must initialize CUDA first: Engine.__init__ rejects
                    # an already initialized torch.cuda context.
                    bind_device(scheduler.device)
                configure_pipeline = getattr(backend, "configure_pipeline", None)
                if configure_pipeline is not None:
                    configure_pipeline(scheduler)
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
                                "worker_id": getattr(args, "worker_id", ""),
                                "decode_radix_cache": scheduler.decode_radix_cache,
                                "kv_memory": (
                                    "cuda" if getattr(backend, "gpu_payload", False) else "host"
                                ),
                                "transfer": getattr(backend, "health", lambda: {})(),
                                "cuda_graph": False,
                                "overlap": False,
                                "kv_transfer_overlap": bool(
                                    getattr(backend, "layered_transfer", False)
                                ),
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
