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

import torch
from minisgl.core import Req, SamplingParams
from minisgl.scheduler.prefill import ChunkedReq
from minisgl.scheduler.utils import PendingReq

from .control import CancelledError, Channel, StartRequest
from .layered_protocol import LayeredReceiveTicket
from .layered_sender import BatchTransferEntry, BatchTransferPlan
from .protocol import KVTransferMetadata, ReceiveTicket, RequestHandoff
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
    ticket: object = None
    job: object = None
    handoff: RequestHandoff | None = None
    staging: object = None
    future: object = None
    install_event: object = None
    index: int = 1
    success: bool = False
    release_requested: bool = False
    enqueued_at: float = field(default_factory=time.monotonic)


class WorkerService:
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
        self.is_prefill = scheduler.role is WorkerRole.PREFILL
        if self.is_prefill:
            scheduler.prefill_manager.page_aligned_budget = True
        self.io = ThreadPoolExecutor(max_workers=8, thread_name_prefix="pd-transfer")
        self.install_stream = (
            torch.cuda.Stream(device=scheduler.device) if scheduler.device.type == "cuda" else None
        )
        self.stats = {"max_prefill_batch": 0, "max_decode_batch": 0}
        self.snapshot = {}
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
        self.snapshot = {
            "status": "ready",
            "role": s.role.value,
            "layout": asdict(s.layout),
            "free_rows": s.table_manager.available_size,
            "free_pages": len(s.cache_manager.free_slots),
            "num_pages": s.engine.num_pages,
            "protected_tokens": s.cache_manager.prefix_cache.size_info.protected_size,
            "evictable_tokens": s.cache_manager.prefix_cache.size_info.evictable_size,
            "active_requests": len(self.contexts),
            "queued_requests": sum(c.phase == "prefill" for c in self.contexts.values()),
            "installing_requests": sum(c.phase == "installing" for c in self.contexts.values()),
            **self.stats,
        }

    def capabilities(self):
        return {
            "status": "capabilities",
            "layered_transfer": self.layered,
            "concurrent_requests": True,
            "multiplex_version": 1,
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
        }

    def _aligned(self, n):
        p = self.s.layout.page_size
        return (n + p - 1) // p * p

    def _reserve_pages(self, ctx):
        s, request = self.s, ctx.request
        needed = self._aligned(len(request.prompt_ids) + request.max_tokens)
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
        slots = handle = None
        try:
            slots = s.cache_manager.reserve_import(row, len(request.prompt_ids))
            handle = s.cache_manager.prefix_cache.match_prefix(
                torch.empty(0, dtype=torch.int32)
            ).cuda_handle
            s.cache_manager.lock(handle)
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
            if handle is not None:
                s.cache_manager.unlock(handle)
            if slots is not None:
                s.cache_manager._free(slots)
            s.table_manager.free(row)
            raise

    def _start(self, session, message):
        op = message.get("op")
        if op == "prefill" and self.is_prefill:
            allowed = {"op", "request", "ticket"} if self.layered else {"op", "request"}
            if set(message) != allowed:
                raise ValueError("prefill requires the worker's transfer protocol")
            request = StartRequest.from_object(message["request"])
        elif op == "reserve" and not self.is_prefill and set(message) == {"op", "request"}:
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
        limit = getattr(self.backend, "max_transfer_bytes", 1024**3)
        if len(request.prompt_ids) * self.s.layout.bytes_per_token > limit:
            raise ValueError("prompt KV exceeds the backend transfer limit")
        ctx = RequestContext(
            session, request, self.next_uid, "prefill" if self.is_prefill else "reserved"
        )
        self.next_uid += 1
        self.contexts[session] = ctx
        self.by_uid[ctx.uid] = ctx
        self.recent.append(request.transfer_id)
        if self.is_prefill:
            if self.layered:
                ticket = LayeredReceiveTicket.from_wire(json.dumps(message["ticket"]).encode())
                if ticket.eos_token_id != self.s.eos_token_id:
                    raise ValueError("incompatible EOS configuration")
                ctx.ticket = ticket
                ctx.job = self.backend.begin_layer_send(
                    ticket, request, self.s.engine.kv_cache, session.check_cancel, wait=False
                )
            self.s.prefill_manager.pending_list.append(
                PendingReq(
                    ctx.uid,
                    torch.tensor(request.prompt_ids, dtype=torch.int32),
                    SamplingParams(max_tokens=1, ignore_eos=True),
                )
            )
        else:
            self._reserve_pages(ctx)
            if self.layered:
                ctx.ticket = self.backend.reserve_layers(request, self.s.eos_token_id)
            else:
                # Host payloads are admission-bounded too, independently of KV pages.
                size = len(request.prompt_ids) * self.s.layout.bytes_per_token
                self._check_host_budget(size)
                ctx.staging = bytearray(size)
                if op == "reserve":
                    session.emit({"status": "reserved", "transfer_id": request.transfer_id})
                    return
                ctx.handoff = handoff
                ctx.ticket = self.backend.prepare_receive(handoff.kv, self.s.layout, ctx.staging)
            session.emit({"status": "ready", "ticket": json.loads(ctx.ticket.to_wire())})

    def _check_host_budget(self, nbytes):
        used = sum(
            len(c.staging)
            for c in self.contexts.values()
            if isinstance(c.staging, (bytes, bytearray))
        )
        limit = getattr(self.backend, "max_transfer_bytes", 1024**3) * 2
        if used + nbytes > limit:
            raise TransferCapacityError("host KV staging budget is busy")

    def _validate_handoff(self, request, handoff):
        if (
            handoff.uid != request.uid
            or handoff.kv.transfer_id != request.transfer_id
            or handoff.prompt_ids != request.prompt_ids
            or handoff.max_tokens != request.max_tokens
            or handoff.ignore_eos != request.ignore_eos
            or handoff.kv.layout != self.s.layout
            or handoff.eos_token_id != self.s.eos_token_id
            or handoff.first_token_id >= self.s.vocab_size
            or not handoff.remaining_tokens
        ):
            raise ValueError("incompatible model, layout, or decode handoff")

    def _command(self, session, message):
        ctx = self.contexts.get(session)
        if ctx is None:
            if session.done.is_set() or session.cancelled.is_set():
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
            self._validate_handoff(ctx.request, ctx.handoff)
            ctx.ticket = self.backend.prepare_receive(ctx.handoff.kv, self.s.layout, ctx.staging)
            session.emit({"status": "ready", "ticket": json.loads(ctx.ticket.to_wire())})
            return
        if op in ("cancel", "release") and message == {
            "op": op,
            "transfer_id": ctx.request.transfer_id,
        }:
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
                self._validate_handoff(ctx.request, ctx.handoff)
            elif ctx.handoff is None or ctx.ticket is None:
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
            return
        try:
            self.s.decode_manager.remove_req(ctx.req)
            if ctx.success:
                self.s.cache_manager.cache_req(ctx.req, finished=True)
            else:
                self.s.cache_manager.discard_private(ctx.req, ctx.allocated)
            self.s.table_manager.free(ctx.req.table_idx)
        except Exception as exc:
            raise TransferBackendUnavailable(
                "split worker could not release its request resources"
            ) from exc
        ctx.req = None

    def _remove(self, ctx, *, complete=True):
        self._free_pages(ctx)
        ctx.staging = None
        if complete:
            ctx.session.done.set()
        self.contexts.pop(ctx.session, None)
        self.by_uid.pop(ctx.uid, None)

    def _cancel(self, ctx):
        self.s.prefill_manager.abort_req(ctx.uid)
        if ctx.req is not None:
            self.s.decode_manager.remove_req(ctx.req)
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
            if self.layered:
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
        try:
            prepared = self._prepare(batch)
            if self.layered:
                entries = [
                    BatchTransferEntry(
                        self.by_uid[r.uid].request.transfer_id,
                        s.engine.page_table[r.table_idx, : r.device_len].to(torch.int64),
                        not isinstance(r, ChunkedReq),
                        self.by_uid[r.uid].job,
                    )
                    for r in batch.reqs
                ]
                s.engine.kv_cache.layer_ready = BatchTransferPlan(self.backend, entries).layer_ready
            try:
                output = s._forward(prepared)
            finally:
                s.engine.kv_cache.layer_ready = None
            # One batch-tail sampling fence; no transfer or packing wait here.
            output.copy_done_event.synchronize()
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
        slots = s.engine.page_table[req.table_idx, : req.cached_len]
        meta = KVTransferMetadata.from_slots(
            request.transfer_id, s.layout, slots.tolist(), num_pool_pages=s.engine.num_pages
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
        ctx.phase = "handoff"
        ctx.session.emit({"status": "handoff", "handoff": json.loads(ctx.handoff.to_wire())})
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
        if self.layered:
            staging = self.backend.finish_layers(ctx.ticket, wait=False)
            if staging is None:
                return
            ctx.staging = staging
        else:
            self.backend.finish_receive(ctx.ticket)
        req, s = ctx.req, self.s
        stream = self.install_stream
        ctx.phase = "installing"
        try:
            req.input_ids[-1] = ctx.handoff.first_token_id
            slots = s.engine.page_table[req.table_idx, : len(ctx.request.prompt_ids)]
            if stream is not None:
                stream.wait_stream(s.engine.stream)
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                if self.layered:
                    s.engine.kv_cache.import_layered_tensor(
                        ctx.staging, slots, num_pool_pages=s.engine.num_pages
                    )
                else:
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
        for ctx in list(self.contexts.values()):
            try:
                if ctx.session.cancelled.is_set():
                    self._cancel(ctx)
                    continue
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
                    if self.layered:
                        self.backend.release_layers(ctx.ticket, consumer_complete=True)
                    ctx.ticket = ctx.staging = ctx.install_event = None
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
        for ctx in self.contexts.values():
            ctx.session.cancelled.set()
        deadline = time.monotonic() + 3 * self.timeout + 1
        while self.contexts and time.monotonic() < deadline:
            self._progress()
            time.sleep(0.002)
        self.io.shutdown(wait=False, cancel_futures=True)
        if self.contexts:
            raise TransferBackendUnavailable("request consumers did not stop before shutdown")

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
