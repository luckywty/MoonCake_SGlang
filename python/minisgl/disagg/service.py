"""Concurrent control connections and one owner of GPU scheduling resources.

Connection handlers only exchange messages and set cancellation flags. The
calling thread owns request tables, KV pages, batch construction and forwards.
Native transfers and receiver visibility receipts run outside that thread.
"""

from __future__ import annotations

import json
import queue
import select
import socket
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from uuid import uuid4

import torch
from minisgl.core import Req, SamplingParams
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.scheduler.utils import PendingReq

from .control import CancelledError, Channel, StartRequest
from .dram_service import DramServiceMixin
from .layered_protocol import LayeredReceiveTicket
from .layered_sender import BatchTransferEntry, BatchTransferPlan
from .prefix_events import PrefixPublisher
from .prefix_hash import PREFIX_INDEX_VERSION, PrefixHashSpec
from .protocol import HANDOFF_VERSION, KVTransferMetadata, ReceiveTicket, RequestHandoff
from .role import WorkerRole
from .transfer import TransferBackendUnavailable, TransferCapacityError


@dataclass(eq=False)
class Session:
    output_limit: int = 128
    request_id: str | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        self.outbox = queue.Queue(self.output_limit)

    def emit(self, message, *, final=False):
        try:
            self.outbox.put_nowait(message)
        except queue.Full:
            self.cancelled.set()  # a slow reader only cancels its own request
        if final:
            self.done.set()

    def check_cancel(self):
        if self.cancelled.is_set():
            raise CancelledError("request cancelled")


@dataclass(eq=False)
class RequestContext:
    session: Session
    request: StartRequest
    uid: int
    phase: str
    req: Req | None = None
    allocated: int = 0
    prefix_hit_tokens: int = 0
    decode_prefix_len: int = 0
    ticket: object = None
    job: object = None
    handoff: RequestHandoff | None = None
    staging: object = None
    future: object = None
    install_event: object = None
    index: int = 1
    success: bool = False
    release_requested: bool = False
    dram: object = None
    remote_plan: object = None
    restore_task: object = None
    restore_kind: str | None = None
    restore_ready_len: int = 0
    pull_future: object = None
    pull_job: object = None
    drain_complete: bool = True
    shared_pin_future: object = None
    enqueued_at: float = field(default_factory=time.monotonic)

    @property
    def transfer_tokens(self):
        return len(self.request.prompt_ids) - self.decode_prefix_len


class WorkerService(DramServiceMixin):
    def __init__(self, scheduler, backend, timeout):
        self.s, self.backend, self.timeout = scheduler, backend, timeout
        config = getattr(scheduler, "pd_service_config", {})
        self.queue_limit = config.get("max_queued_requests", 128)
        self.token_limit = config.get("max_queued_tokens") or (
            scheduler.engine.num_pages * scheduler.layout.page_size * 4
        )
        self.batch_wait = config.get("batch_wait_ms", 2.0) / 1000
        self.output_limit = config.get("output_queue_size", 128)
        self.commands = queue.Queue(self.queue_limit * 4 + 32)
        self.contexts: dict[Session, RequestContext] = {}
        self.by_uid = {}
        self.recent = deque(maxlen=4096)
        self.next_uid = 0
        self.layered = bool(getattr(backend, "layered_transfer", False))
        self.transfer_mode = getattr(backend, "transfer_mode", "gpu_direct")
        self.is_prefill = scheduler.role is WorkerRole.PREFILL
        self.instance_id = str(uuid4())
        self.numa_id = getattr(backend, "numa_id", config.get("numa_id", 0))
        self.host_id = getattr(backend, "host_id", config.get("host_id", ""))
        self.cache_namespace = config.get("cache_namespace", "default")
        self.prefix_publisher = None
        cache = scheduler.cache_manager.prefix_cache
        if self.is_prefill and hasattr(cache, "page_observer"):
            spec = PrefixHashSpec.from_layout(asdict(scheduler.layout), self.cache_namespace)
            if not (
                spec.page_size
                == scheduler.cache_manager.page_size
                == scheduler.engine.kv_cache.page_size
            ):
                raise ValueError("prefix hash and MHA KV page sizes differ")
            self.prefix_publisher = PrefixPublisher(
                spec,
                self.instance_id,
                min(scheduler.engine.num_pages, config.get("prefix_max_pages", 65536)),
                config.get("prefix_event_limit", 32768),
            )
            cache.page_observer = self.prefix_publisher
        self.decode_radix_cache = not self.is_prefill and bool(
            config.get("decode_radix_cache", False)
        )
        if self.is_prefill:
            scheduler.prefill_manager.page_aligned_budget = True
        self.io = ThreadPoolExecutor(max_workers=8, thread_name_prefix="pd-transfer")
        self.install_stream = (
            (
                backend.install_stream
                if self.transfer_mode in ("host_dram", "shm_dram")
                else torch.cuda.Stream(device=scheduler.device)
            )
            if scheduler.device.type == "cuda" and not self.is_prefill
            else None
        )
        self.stats = {
            "max_prefill_batch": 0,
            "max_decode_batch": 0,
            "decode_cache_hit_requests": 0,
            "decode_cache_hit_tokens": 0,
            "received_kv_tokens": 0,
        }
        self.prefill_ms_per_token = None
        self.snapshot = {}
        self._init_dram(config)
        self._update_snapshot()

    def submit(self, session, message):
        try:
            self.commands.put_nowait((session, message))
        except queue.Full:
            session.emit(
                {"status": "error", "code": "busy", "error": "control queue is full"}, final=True
            )
            session.cancelled.set()

    def _update_snapshot(self):
        s = self.s
        capacity = s.engine.num_pages * s.layout.page_size
        waiting = sum(
            c.phase == "prefill" if self.is_prefill else c.phase != "decode"
            for c in self.contexts.values()
        )
        running = (
            sum(c.req is not None for c in self.contexts.values())
            if self.is_prefill
            else len(s.decode_manager.running_reqs)
        )
        pending_tokens = sum(
            max(0, len(c.request.prompt_ids) - max(c.allocated, c.prefix_hit_tokens))
            for c in self.contexts.values()
            if c.phase == "prefill"
        )
        self.snapshot = {
            "status": "ready",
            "role": s.role.value,
            "instance_id": self.instance_id,
            "cache_namespace": self.cache_namespace,
            "layout": asdict(s.layout),
            "free_rows": s.table_manager.available_size,
            "free_pages": len(s.cache_manager.free_slots),
            "num_pages": s.engine.num_pages,
            "protected_tokens": s.cache_manager.prefix_cache.size_info.protected_size,
            "evictable_tokens": s.cache_manager.prefix_cache.size_info.evictable_size,
            "active_requests": len(self.contexts),
            "queued_requests": sum(c.phase == "prefill" for c in self.contexts.values()),
            "installing_requests": sum(c.phase == "installing" for c in self.contexts.values()),
            "routing_version": 1,
            "handoff_version": HANDOFF_VERSION,
            "transfer_mode": self.transfer_mode if self.layered else "tcp",
            "numa_id": self.numa_id,
            "host_id": self.host_id,
            "shm_pool": getattr(getattr(self.backend, "manifest", None), "name", None),
            "shared_dram": self._shared_identity(),
            "decode_prefix_reuse": True,
            "decode_radix_cache": self.decode_radix_cache,
            "max_sequence_tokens": s.engine.max_seq_len,
            "load": {
                "waiting_requests": waiting,
                "running_requests": running,
                "running_capacity": s.table_manager._max_running_reqs,
                "pending_prefill_tokens": pending_tokens,
                # Evictable prefixes can make room; count private/protected KV pressure.
                "kv_tokens": max(0, capacity - s.cache_manager.available_size),
                "kv_capacity": capacity,
            },
            **self.stats,
        }
        if self.prefill_ms_per_token is not None:
            self.snapshot["load"]["estimated_prefill_queue_ms"] = (
                pending_tokens * self.prefill_ms_per_token
            )

    def capabilities(self):
        return {
            "status": "capabilities",
            "transfer_mode": self.transfer_mode if self.layered else "tcp",
            "layered_transfer": self.layered,
            "concurrent_requests": True,
            "multiplex_version": 1,
            "handoff_version": HANDOFF_VERSION,
            "decode_prefix_reuse": True,
            "prefix_index_version": PREFIX_INDEX_VERSION if self.prefix_publisher else None,
            "dram_cache_version": 1 if self.dram_pool is not None else None,
            "shared_dram": self._shared_identity(),
            **(
                {
                    "host_id": self.host_id,
                    "numa_id": self.numa_id,
                    "shm_pool": self.backend.manifest.name,
                }
                if self.transfer_mode == "shm_dram"
                else {}
            ),
        }

    def health(self):
        native = getattr(self.backend, "health", lambda: {})()
        return {
            **self.snapshot,
            "status": native.get("status", "ready"),
            "worker_id": getattr(self.s, "worker_id", ""),
            "transfer_backend": getattr(self.backend, "name", "tcp"),
            "layered_transfer": self.layered,
            "multiplex_version": 1,
            "transfer": native,
            "prefix_index": self.prefix_publisher.health() if self.prefix_publisher else None,
            "tier_index": self.tier_publisher.health() if self.tier_publisher else None,
            "dram_cache": self._dram_health(),
        }

    def prefix_sync(self, message):
        if self.prefix_publisher is None:
            raise ValueError("page prefix synchronization requires a Radix Prefill Worker")
        return self.prefix_publisher.sync(message)

    def _aligned(self, n):
        p = self.s.layout.page_size
        return (n + p - 1) // p * p

    def _reserve_pages(self, ctx, *, reuse_prefix=True):
        s, request = self.s, ctx.request
        row = None
        slots = handle = None
        locked = False
        try:
            handle = s.cache_manager.prefix_cache.match_prefix(
                torch.tensor(
                    request.prompt_ids if self.decode_radix_cache and reuse_prefix else (),
                    dtype=torch.int32,
                )
            ).cuda_handle
            s.cache_manager.lock(handle)
            locked = True
            prefix_len = ctx.decode_prefix_len = handle.cached_len
            # A matched page used to be evictable. Lock it before pricing the
            # remaining private/output pages against available_size.
            needed = self._aligned(len(request.prompt_ids) + request.max_tokens) - prefix_len
            future_tokens = sum(
                max(
                    0,
                    self._aligned(len(c.request.prompt_ids) + c.request.max_tokens)
                    - self._aligned(c.allocated),
                )
                for c in self.contexts.values()
                if c.req is not None
            )
            if (
                not s.table_manager.available_size
                or needed + future_tokens > s.cache_manager.available_size
            ):
                raise TransferCapacityError("decode KV page or output-token budget is busy")
            row = s.table_manager.allocate()
            if prefix_len:
                s.engine.page_table[row, :prefix_len] = handle.get_matched_indices()
            slots = (
                s.cache_manager.reserve_import(row, ctx.transfer_tokens, start=prefix_len)
                if ctx.transfer_tokens
                else s.engine.page_table[row, prefix_len:prefix_len]
            )
            ctx.req = Req(
                torch.tensor((*request.prompt_ids, 0), dtype=torch.int32),
                row,
                len(request.prompt_ids),
                request.max_tokens - 1,
                ctx.uid,
                SamplingParams(max_tokens=request.max_tokens, ignore_eos=request.ignore_eos),
                handle,
            )
            ctx.allocated = len(request.prompt_ids)
        except BaseException:
            if locked:
                s.cache_manager.unlock(handle)
            if slots is not None:
                s.cache_manager._free(slots)
            if row is not None:
                s.table_manager.free(row)
            raise

    def _reservation_reply(self, ctx, status):
        reply = {
            "status": status,
            "transfer_id": ctx.request.transfer_id,
            "decode_prefix_len": ctx.decode_prefix_len,
            "prompt_tokens": len(ctx.request.prompt_ids),
            "handoff_version": HANDOFF_VERSION,
            "transfer_mode": self.transfer_mode if self.layered else "tcp",
        }
        if self.dram_pool is not None:
            reply["dram_cache_version"] = 1
        if status == "ready":
            reply["ticket"] = json.loads(ctx.ticket.to_wire()) if ctx.ticket is not None else None
        return reply

    def _start(self, session, message):
        op = message.get("op")
        if op == "prefill" and self.is_prefill:
            allowed = {"op", "request", "ticket"} if self.layered else {"op", "request"}
            if "decode_prefix_len" in message:
                allowed.add("decode_prefix_len")
            allowed.update(k for k in ("cache_transaction", "remote_pull_plan") if k in message)
            if set(message) != allowed:
                raise ValueError("prefill requires the worker's transfer protocol")
            request = StartRequest.from_object(message["request"])
        elif (
            op == "reserve"
            and not self.is_prefill
            and set(message) <= {"op", "request", "cache_transaction"}
            and "request" in message
        ):
            request = StartRequest.from_object(message["request"])
        elif (
            op == "decode"
            and not self.is_prefill
            and not self.layered
            and set(message) == {"op", "handoff"}
        ):
            handoff = RequestHandoff.from_wire(json.dumps(message["handoff"]).encode())
            request = StartRequest(
                handoff.kv.transfer_id,
                handoff.uid,
                handoff.prompt_ids,
                handoff.max_tokens,
                handoff.ignore_eos,
            )
            self._validate_handoff(request, handoff)
        else:
            raise ValueError("operation does not match worker role")
        self.s._validate_request(request.prompt_ids, request.max_tokens)
        prefix_len = message.get("decode_prefix_len", 0) if self.is_prefill else 0
        if (
            type(prefix_len) is not int
            or not 0 <= prefix_len <= len(request.prompt_ids)
            or prefix_len % self.s.layout.page_size
        ):
            raise ValueError("invalid decode prefix length")
        if session.request_id is not None and session.request_id != request.transfer_id:
            raise ValueError("request_id does not match transfer_id")
        if request.transfer_id in self.recent:
            raise ValueError("duplicate transfer attempt")
        if (
            len(self.contexts) >= self.queue_limit
            or sum(len(c.request.prompt_ids) + c.request.max_tokens for c in self.contexts.values())
            + len(request.prompt_ids)
            + request.max_tokens
            > self.token_limit
        ):
            raise TransferCapacityError("worker request or token admission budget is busy")
        ctx = RequestContext(
            session, request, self.next_uid, "prefill" if self.is_prefill else "reserved"
        )
        ctx.decode_prefix_len = prefix_len
        self.next_uid += 1
        self.contexts[session] = ctx
        self.by_uid[ctx.uid] = ctx
        self.recent.append(request.transfer_id)
        if self.is_prefill:
            self._check_transfer_size(ctx)
            ctx.prefix_hit_tokens = self.s.cache_manager.prefix_cache.peek_prefix_len(
                torch.tensor(request.prompt_ids[:-1], dtype=torch.int32)
            )
            if self.layered and ctx.transfer_tokens:
                ticket = LayeredReceiveTicket.from_wire(json.dumps(message["ticket"]).encode())
                if (
                    ticket.eos_token_id != self.s.eos_token_id
                    or ticket.prefix_len != ctx.decode_prefix_len
                ):
                    raise ValueError("incompatible EOS configuration or decode prefix")
                ctx.ticket = ticket
                ctx.job = self.backend.begin_layer_send(
                    ticket, request, self.s.engine.kv_cache, session.check_cancel, wait=False
                )
            elif self.layered and message["ticket"] is not None:
                raise ValueError("a full decode cache hit must not expose a receive ticket")
            if self._begin_dram_request(ctx, message):
                return
            self.s.prefill_manager.pending_list.append(
                PendingReq(
                    ctx.uid,
                    torch.tensor(request.prompt_ids, dtype=torch.int32),
                    SamplingParams(max_tokens=1, ignore_eos=True),
                )
            )
        else:
            self._begin_dram_request(ctx, message)
            # Legacy post-prefill TCP handoffs already contain a full payload;
            # only the reserve-first protocol may promise a reusable prefix.
            self._reserve_pages(ctx, reuse_prefix=op != "decode")
            if op == "reserve" and self._dram_decode_restore(ctx):
                return
            self._complete_reserve(ctx, legacy_handoff=handoff if op == "decode" else None)

    def _complete_reserve(self, ctx, *, legacy_handoff=None):
        request, session = ctx.request, ctx.session
        ctx.phase = "reserved"
        self._check_transfer_size(ctx)
        if self.layered and ctx.transfer_tokens:
            ctx.ticket = self.backend.reserve_layers(
                request, self.s.eos_token_id, prefix_len=ctx.decode_prefix_len
            )
        elif not self.layered:
            # Host payloads are admission-bounded too, independently of KV pages.
            size = ctx.transfer_tokens * self.s.layout.bytes_per_token
            if size:
                self._check_host_budget(size)
                ctx.staging = bytearray(size)
            if legacy_handoff is None:
                if ctx.decode_prefix_len:
                    self.stats["decode_cache_hit_requests"] += 1
                    self.stats["decode_cache_hit_tokens"] += ctx.decode_prefix_len
                session.emit(self._reservation_reply(ctx, "reserved"))
                return
            ctx.handoff = legacy_handoff
            ctx.ticket = self.backend.prepare_receive(legacy_handoff.kv, self.s.layout, ctx.staging)
        if ctx.decode_prefix_len:
            self.stats["decode_cache_hit_requests"] += 1
            self.stats["decode_cache_hit_tokens"] += ctx.decode_prefix_len
        session.emit(self._reservation_reply(ctx, "ready"))

    def _check_transfer_size(self, ctx):
        limit = getattr(self.backend, "max_transfer_bytes", 1024**3)
        if ctx.transfer_tokens * self.s.layout.bytes_per_token > limit:
            raise ValueError("KV suffix exceeds the backend transfer limit")

    def _check_host_budget(self, nbytes):
        used = sum(
            len(c.staging)
            for c in self.contexts.values()
            if isinstance(c.staging, (bytes, bytearray))
        )
        limit = getattr(self.backend, "max_transfer_bytes", 1024**3) * 2
        if used + nbytes > limit:
            raise TransferCapacityError("host KV staging budget is busy")

    def _validate_handoff(self, request, handoff, *, prefix_len=0):
        if (
            handoff.uid != request.uid
            or handoff.kv.transfer_id != request.transfer_id
            or handoff.prompt_ids != request.prompt_ids
            or handoff.max_tokens != request.max_tokens
            or handoff.ignore_eos != request.ignore_eos
            or handoff.kv.layout != self.s.layout
            or handoff.kv.transfer_mode != self.transfer_mode
            or handoff.kv.prefix_len != prefix_len
            or handoff.eos_token_id != self.s.eos_token_id
            or handoff.first_token_id >= self.s.vocab_size
            or not handoff.remaining_tokens
        ):
            raise ValueError("incompatible model, layout, or decode handoff")

    def _command(self, session, message):
        if message.get("op") in (
            "tier_sync",
            "cache_commit",
            "cache_abort",
            "dram_pin",
            "dram_unpin",
        ):
            session.emit(self._dram_rpc(message), final=True)
            return
        ctx = self.contexts.get(session)
        if ctx is None:
            if session.done.is_set() or session.cancelled.is_set():
                return
            if message.get("op") == "prefix_match":
                if not self.is_prefill or set(message) != {"op", "prompt_ids"}:
                    raise ValueError("prefix_match requires a Prefill Worker and prompt_ids")
                ids = message["prompt_ids"]
                self.s._validate_request(ids, 1)
                matched = self.s.cache_manager.prefix_cache.peek_prefix_len(
                    torch.tensor(ids[:-1], dtype=torch.int32)
                )
                session.emit({"status": "prefix_match", "matched_tokens": matched}, final=True)
                return
            return self._start(session, message)
        op = message.get("op")
        if op == "prepare" and not self.layered and ctx.phase == "reserved" and ctx.handoff is None:
            if (
                set(message) != {"op", "transfer_id", "handoff"}
                or message["transfer_id"] != ctx.request.transfer_id
            ):
                raise ValueError("invalid reserved payload preparation")
            ctx.handoff = RequestHandoff.from_wire(json.dumps(message["handoff"]).encode())
            self._validate_handoff(ctx.request, ctx.handoff, prefix_len=ctx.decode_prefix_len)
            if ctx.transfer_tokens:
                ctx.ticket = self.backend.prepare_receive(
                    ctx.handoff.kv, self.s.layout, ctx.staging
                )
            session.emit(self._reservation_reply(ctx, "ready"))
            return
        if op in ("cancel", "release") and message == {
            "op": op,
            "transfer_id": ctx.request.transfer_id,
        }:
            if ctx.session.done.is_set() and ctx.success:
                return
            session.cancelled.set()
            if op == "release":
                ctx.release_requested = True
            return
        if op == "commit" and ctx.phase == "reserved":
            fields = {"op", "transfer_id", "handoff"} if self.layered else {"op", "transfer_id"}
            if set(message) != fields or message["transfer_id"] != ctx.request.transfer_id:
                raise ValueError("invalid transfer commit")
            if self.layered:
                ctx.handoff = RequestHandoff.from_wire(json.dumps(message["handoff"]).encode())
                self._validate_handoff(ctx.request, ctx.handoff, prefix_len=ctx.decode_prefix_len)
            elif ctx.handoff is None or (ctx.transfer_tokens and ctx.ticket is None):
                raise ValueError("payload must be prepared before commit")
            ctx.phase = "committed"
            return
        if (
            op == "send"
            and ctx.phase == "handoff"
            and not self.layered
            and set(message) == {"op", "ticket"}
        ):
            target = ReceiveTicket.from_wire(json.dumps(message["ticket"]).encode())
            ctx.future = self.io.submit(
                self.backend.send,
                ctx.handoff.kv,
                ctx.staging,
                target,
                cancel_check=session.check_cancel,
            )
            ctx.phase = "sending"
            return
        raise ValueError("unexpected control message in request phase " + ctx.phase)

    def _free_pages(self, ctx):
        if ctx.req is None:
            return True
        if not self._dram_can_free(ctx):
            return False
        try:
            self.s.decode_manager.remove_req(ctx.req)
            if ctx.success and (self.is_prefill or self.decode_radix_cache):
                self.s.cache_manager.cache_req(ctx.req, finished=True)
            else:
                self.s.cache_manager.discard_private(ctx.req, ctx.allocated)
            self.s.table_manager.free(ctx.req.table_idx)
        except Exception as exc:
            raise TransferBackendUnavailable(
                "split worker could not release its request resources"
            ) from exc
        ctx.req = None
        return True

    def _remove(self, ctx, *, complete=True):
        if not self._free_pages(ctx):
            ctx.phase = "draining"
            ctx.drain_complete = complete
            return False
        ctx.staging = None
        if self.dram_pool is not None:
            self.dram_pool.end(ctx.request.transfer_id)
        if complete:
            ctx.session.done.set()
        self.contexts.pop(ctx.session, None)
        self.by_uid.pop(ctx.uid, None)
        return True

    def _cancel(self, ctx):
        self.s.prefill_manager.abort_req(ctx.uid)
        if ctx.req is not None:
            self.s.decode_manager.remove_req(ctx.req)
        if not self._dram_can_free(ctx):
            return
        if (
            not self.is_prefill
            and ctx.ticket is not None
            and getattr(self.backend, "incremental_install", False)
        ):
            slots = self.s.engine.page_table[
                ctx.req.table_idx, ctx.decode_prefix_len : len(ctx.request.prompt_ids)
            ]
            if not self.backend.progress_receive(ctx.ticket, slots, cancel=True):
                return  # A partial H2D still owns private GPU pages and its SHM reader.
        if ctx.job is not None:
            ctx.job.seal(cancel=True)
            if not ctx.job.sources_done.is_set():
                if ctx.job.done.is_set() and not ctx.job.safe_sources:
                    raise TransferBackendUnavailable("layer gather may still read private KV pages")
                return
            self._free_pages(ctx)
            if not ctx.job.done.is_set():
                return
        if ctx.phase == "installing":
            if not self._installation_ready(ctx):
                return
            ctx.install_event = None
            if self.layered and ctx.ticket is not None:
                self.backend.release_layers(ctx.ticket, consumer_complete=True)
                ctx.ticket = None
            ctx.phase = "cancelled"
        if ctx.ticket is not None and not self.is_prefill:
            if self.layered:
                self.backend.release_layers(ctx.ticket)
            else:
                self.backend.discard_receive(ctx.ticket)
            ctx.ticket = None
        if ctx.future is not None and not ctx.future.done():
            return
        self._remove(ctx, complete=not ctx.release_requested)
        if ctx.release_requested:
            ctx.session.emit(
                {"status": "released", "transfer_id": ctx.request.transfer_id}, final=True
            )

    def _fail(self, session, exc):
        session.emit(
            {
                "status": "error",
                "code": "busy" if isinstance(exc, TransferCapacityError) else "failed",
                "error": str(exc),
            },
            final=True,
        )
        session.cancelled.set()

    def _prepare(self, batch):
        s = self.s
        for req in batch.reqs:
            ctx = self.by_uid[req.uid]
            ctx.req = req
            ctx.allocated = max(ctx.allocated, req.cache_handle.cached_len)
            if self.is_prefill and ctx.dram is not None and not ctx.dram.started:
                ctx.dram.cursor = req.cached_len
                ctx.dram.started = True
        s.engine.graph_runner.pad_batch(batch)
        s.cache_manager.allocate_paged(batch.reqs)
        for req in batch.reqs:
            ctx = self.by_uid[req.uid]
            ctx.allocated = max(ctx.allocated, req.device_len)
        return s._prepare_batch_metadata(batch)

    def _prefill(self):
        s = self.s
        pending = s.prefill_manager.pending_list
        if not pending:
            return
        if (
            all(p.chunked_req is None for p in pending)
            and time.monotonic() - min(self.by_uid[p.uid].enqueued_at for p in pending)
            < self.batch_wait
        ):
            return
        batch = s.prefill_manager.schedule_next_batch(s.prefill_budget)
        if batch is None:
            return
        self.stats["max_prefill_batch"] = max(self.stats["max_prefill_batch"], batch.size)
        compute_started = time.monotonic()
        new_tokens = sum(req.extend_len for req in batch.reqs)
        try:
            prepared = self._prepare(batch)
            if self.layered:
                entries = [
                    BatchTransferEntry(
                        self.by_uid[r.uid].request.transfer_id,
                        s.engine.page_table[
                            r.table_idx, self.by_uid[r.uid].decode_prefix_len : r.device_len
                        ].to(torch.int64),
                        not isinstance(r, ChunkedReq),
                        self.by_uid[r.uid].job,
                    )
                    for r in batch.reqs
                    if self.by_uid[r.uid].job is not None
                ]
                if entries:
                    s.engine.kv_cache.layer_ready = BatchTransferPlan(
                        self.backend, entries
                    ).layer_ready
            try:
                output = s._forward(prepared)
            finally:
                s.engine.kv_cache.layer_ready = None
            # One batch-tail sampling fence; no transfer or packing wait here.
            output.copy_done_event.synchronize()
            measured = (time.monotonic() - compute_started) * 1000 / max(1, new_tokens)
            self.prefill_ms_per_token = (
                measured
                if self.prefill_ms_per_token is None
                else 0.8 * self.prefill_ms_per_token + 0.2 * measured
            )
        except TransferBackendUnavailable:
            raise
        except Exception as exc:
            self._fence_failed_compute()
            # Resources for every row must be owned even if preparation failed.
            for req in batch.reqs:
                ctx = self.by_uid.get(req.uid)
                if ctx is not None and ctx.req is not None:
                    ctx.success = False
                    self._fail(ctx.session, exc)

            return
        for i, req in enumerate(batch.reqs):
            s.decode_manager.remove_req(req)
            if isinstance(req, ChunkedReq):
                self._stage_dram_live(self.by_uid[req.uid])
                continue
            ctx = self.by_uid[req.uid]
            try:
                self._prefill_result(ctx, req, output.next_tokens_cpu[i : i + 1])
            except TransferBackendUnavailable:
                raise
            except Exception as exc:
                self._fence_failed_compute()
                ctx.success = False
                self._fail(ctx.session, exc)

    def _prefill_result(self, ctx, req, token):
        s, request = self.s, ctx.request
        req.append_host(token)
        slots = s.engine.page_table[req.table_idx, ctx.decode_prefix_len : req.cached_len]
        meta = KVTransferMetadata.from_slots(
            request.transfer_id,
            s.layout,
            slots.tolist(),
            num_pool_pages=s.engine.num_pages,
            prefix_len=ctx.decode_prefix_len,
            transfer_mode=self.transfer_mode,
        )
        ctx.handoff = RequestHandoff(
            request.uid,
            request.prompt_ids,
            int(token[0]),
            request.max_tokens,
            request.ignore_eos,
            s.eos_token_id,
            meta,
        )
        ctx.success = True
        self._stage_dram_live(ctx)
        ctx.phase = "handoff"
        ctx.session.emit({"status": "handoff", "handoff": json.loads(ctx.handoff.to_wire())})
        if not ctx.transfer_tokens:
            # No staging allocation, registration, gather or native operation.
            # P still computed its own logits and sampled the first token.
            self._free_pages(ctx)
            ctx.session.emit(
                {
                    "status": "sent" if ctx.handoff.remaining_tokens else "retired",
                    "transfer_id": request.transfer_id,
                },
                final=True,
            )
            self._remove(ctx)
            return
        if self.layered:
            ctx.job.seal(cancel=not ctx.handoff.remaining_tokens)
        else:
            if ctx.handoff.remaining_tokens:
                self._check_host_budget(meta.nbytes)
                ctx.staging = s.engine.kv_cache.export_tokens(
                    slots, num_pool_pages=s.engine.num_pages
                )
            self._free_pages(ctx)
            if not ctx.handoff.remaining_tokens:
                self._remove(ctx)

    def _fence_failed_compute(self):
        # Error recovery only: never recycle pages while a failed forward may
        # still read them. Normal layer callbacks and transfer progress do not wait.
        try:
            self.s.synchronize()
        except Exception as exc:
            raise TransferBackendUnavailable(
                "CUDA compute completion is unknown; retain KV pages"
            ) from exc

    def _install(self, ctx):
        if self.layered and ctx.transfer_tokens:
            staging = self.backend.finish_layers(ctx.ticket, wait=False)
            if staging is None:
                return
            ctx.staging = staging
        elif ctx.transfer_tokens:
            self.backend.finish_receive(ctx.ticket)
        req, s = ctx.req, self.s
        stream = self.install_stream
        ctx.phase = "installing"
        try:
            req.input_ids[-1] = ctx.handoff.first_token_id
            slots = s.engine.page_table[
                req.table_idx, ctx.decode_prefix_len : len(ctx.request.prompt_ids)
            ]
            if stream is not None:
                stream.wait_stream(s.engine.stream)
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                if self.layered and ctx.transfer_tokens:
                    self.backend.install_layers(
                        s.engine.kv_cache, ctx.staging, slots, num_pool_pages=s.engine.num_pages
                    )
                elif ctx.transfer_tokens:
                    s.engine.kv_cache.import_tokens(
                        ctx.staging, slots, num_pool_pages=s.engine.num_pages
                    )
                s.token_pool[req.table_idx, : req.device_len] = req.input_ids.to(
                    s.device, non_blocking=True
                )
        finally:
            # Also fence a partial scatter which raises after enqueueing copies.
            if stream is not None:
                try:
                    ctx.install_event = torch.cuda.Event()
                    ctx.install_event.record(stream)
                except Exception as exc:
                    raise TransferBackendUnavailable(
                        "cannot fence CUDA installation; retain receive lease"
                    ) from exc

    def _installation_ready(self, ctx):
        try:
            return ctx.install_event is None or ctx.install_event.query()
        except Exception as exc:
            raise TransferBackendUnavailable(
                "CUDA installation completion is unknown; retain its pages"
            ) from exc

    def _progress(self):
        self._dram_progress()
        for ctx in list(self.contexts.values()):
            try:
                if ctx.phase == "draining":
                    continue
                if ctx.session.cancelled.is_set():
                    self._cancel(ctx)
                    continue
                if (
                    not self.is_prefill
                    and ctx.ticket is not None
                    and getattr(self.backend, "incremental_install", False)
                ):
                    slots = self.s.engine.page_table[
                        ctx.req.table_idx, ctx.decode_prefix_len : len(ctx.request.prompt_ids)
                    ]
                    self.backend.progress_receive(ctx.ticket, slots)
                if ctx.job is not None:
                    if ctx.job.sources_done.is_set():
                        self._free_pages(ctx)
                    if ctx.job.done.is_set():
                        if ctx.job.error is not None:
                            raise ctx.job.error
                        ctx.session.emit(
                            {
                                "status": "sent" if ctx.handoff.remaining_tokens else "retired",
                                "transfer_id": ctx.request.transfer_id,
                            },
                            final=True,
                        )
                        self._remove(ctx)
                elif ctx.phase == "sending" and ctx.future.done():
                    ctx.future.result()
                    ctx.session.emit(
                        {"status": "sent", "transfer_id": ctx.request.transfer_id}, final=True
                    )
                    self._remove(ctx)
                elif ctx.phase == "committed":
                    self._install(ctx)
                if ctx.phase == "installing" and self._installation_ready(ctx):
                    if self.layered and ctx.ticket is not None:
                        self.backend.release_layers(ctx.ticket, consumer_complete=True)
                    ctx.ticket = ctx.staging = ctx.install_event = None
                    if self.decode_radix_cache:
                        try:
                            self.s.cache_manager.cache_req(ctx.req, finished=False, rebind=True)
                        except Exception as exc:
                            # Insertion may already have adopted/freed pages.
                            # Never run private-page cleanup on ambiguous ownership.
                            raise TransferBackendUnavailable(
                                "cannot publish installed decode KV safely"
                            ) from exc
                    self.stats["received_kv_tokens"] += ctx.transfer_tokens
                    if getattr(self, "shared_history", False):
                        self._stage_dram_live(ctx)
                    self.s.decode_manager.filter_reqs([ctx.req])
                    ctx.phase = "decode"
                    ctx.session.emit(
                        {"status": "installed", "transfer_id": ctx.request.transfer_id}
                    )
            except TransferBackendUnavailable:
                raise
            except Exception as exc:
                self._fail(ctx.session, exc)

    def _decode(self):
        batch = self.s.decode_manager.schedule_next_batch()
        if batch is None:
            return
        self.stats["max_decode_batch"] = max(self.stats["max_decode_batch"], batch.size)
        try:
            output = self.s._forward(self._prepare(batch))
            output.copy_done_event.synchronize()
            for i, req in enumerate(batch.reqs):
                ctx = self.by_uid[req.uid]
                req.append_host(output.next_tokens_cpu[i : i + 1])
                self._stage_dram_live(ctx)
                token = int(output.next_tokens_cpu[i])
                finished = not req.can_decode or (
                    not ctx.request.ignore_eos and token == self.s.eos_token_id
                )
                message = {
                    "status": "token",
                    "transfer_id": ctx.request.transfer_id,
                    "index": ctx.index,
                    "token_id": token,
                    "finished": finished,
                }
                ctx.index += 1
                if finished:
                    ctx.success = True
                    self.s.decode_manager.remove_req(req)
                    self._remove(ctx, complete=False)
                ctx.session.emit(message, final=finished)
        except TransferBackendUnavailable:
            raise
        except Exception as exc:
            self._fence_failed_compute()
            for req in batch.reqs:
                ctx = self.by_uid.get(req.uid)
                if ctx is not None:
                    self._fail(ctx.session, exc)

    def step(self):
        for _ in range(64):
            try:
                session, message = self.commands.get_nowait()
            except queue.Empty:
                break
            try:
                self._command(session, message)
            except TransferBackendUnavailable:
                raise
            except Exception as exc:
                self._fail(session, exc)
        self._progress()
        if self.is_prefill:
            self._prefill()
        else:
            self._decode()
        self._update_snapshot()

    def close(self):
        if self.prefix_publisher is not None:
            self.prefix_publisher.disable("Worker service is closing")
        for ctx in self.contexts.values():
            ctx.session.cancelled.set()
        deadline = time.monotonic() + 3 * self.timeout + 1
        while self.contexts and time.monotonic() < deadline:
            self._progress()
            time.sleep(0.002)
        self.io.shutdown(wait=False, cancel_futures=True)
        if self.contexts:
            raise TransferBackendUnavailable("request consumers did not stop before shutdown")
        if self.dram_transfer is not None:
            self.dram_transfer.close()
        if self.dram_controller is not None:
            self.dram_controller.close()

    def run(self, listener, stop=None):
        stopping = threading.Event()
        sockets, lock = set(), threading.Lock()
        slots = threading.BoundedSemaphore(self.queue_limit + 16)
        handlers = ThreadPoolExecutor(
            max_workers=self.queue_limit + 16, thread_name_prefix="pd-control"
        )
        listener.settimeout(0.2)

        def handle(sock):
            session = Session(self.output_limit)
            channel = Channel(sock, self.timeout, stopping)
            last_activity = time.monotonic()
            try:
                with sock:
                    message = channel.receive()
                    if message.get("op") == "multiplex":
                        from .mux_server import serve_multiplex

                        serve_multiplex(self, sock, channel, message, stopping)
                        return
                    if message == {"op": "capabilities"}:
                        channel.send(self.capabilities())
                        message = channel.receive()
                    if message == {"op": "health"}:
                        channel.send(self.health())
                        return
                    if message.get("op") in (
                        "prefix_sync",
                        "tier_sync",
                        "cache_commit",
                        "cache_abort",
                        "dram_pin",
                        "dram_unpin",
                    ):
                        channel.send(
                            self.prefix_sync(message)
                            if message["op"] == "prefix_sync"
                            else self._dram_rpc(message)
                        )
                        return
                    self.submit(session, message)
                    while not stopping.is_set():
                        try:
                            item = session.outbox.get_nowait()
                        except queue.Empty:
                            if session.done.is_set():
                                return
                        else:
                            channel.send(item)
                            last_activity = time.monotonic()
                            continue
                        readable, _, _ = select.select([sock], [], [], 0.01)
                        if readable:
                            self.submit(session, channel.receive())
                            last_activity = time.monotonic()
                        if time.monotonic() - last_activity > self.timeout:
                            raise TimeoutError("request control connection timed out")
            except (OSError, RuntimeError, ValueError):
                pass
            finally:
                if not session.done.is_set():
                    session.cancelled.set()
                with lock:
                    sockets.discard(sock)
                slots.release()

        def accept():
            while not stopping.is_set():
                try:
                    sock, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    return
                if not slots.acquire(blocking=False):
                    sock.close()
                    continue
                with lock:
                    sockets.add(sock)
                handlers.submit(handle, sock)

        acceptor = threading.Thread(target=accept, name="pd-control-listener", daemon=True)
        acceptor.start()
        try:
            with torch.inference_mode(), self.s.execution():
                while stop is None or not stop.is_set():
                    getattr(self.backend, "check_available", lambda: None)()
                    self.step()
                    time.sleep(0.001)
        finally:
            stopping.set()
            acceptor.join(timeout=1)
            with lock:
                for sock in sockets:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
            handlers.shutdown(wait=True)
            with self.s.execution():
                self.close()
