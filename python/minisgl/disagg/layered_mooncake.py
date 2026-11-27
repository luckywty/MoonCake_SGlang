"""Startup-registered GPU/host pools and a layer-ready compute/communication pipeline.

Only the sender thread waits for packing or native completion. A staging slot
is reused only after a checked terminal drain; D's published pool is retained
until the sender proves that every submitted write completed successfully.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from uuid import uuid4

import torch

from .allocator import ReceiveAllocator
from .cluster_config import host_staging_plan
from .control import CancelledError, Channel, StartRequest
from .layered_protocol import LayeredReceiveTicket
from .layered_sender import ControlSession, LayeredSend, sender_loop
from .mooncake_backend import MooncakeTransferBackend
from .role import WorkerRole
from .transfer import TransferBackendUnavailable, TransferCapacityError, TransferError

_TRANSPORT = "mooncake-layered-gpu-v3"


@dataclass
class _Lease:
    request: StartRequest
    ticket: LayeredReceiveTicket
    offset: int = 0
    safe: bool = True
    source: str | None = None
    exposed: bool = False
    cancelled: bool = False
    owner_released: bool = False
    delivered: bool = False
    consuming: bool = False
    finalizing: bool = False
    quarantine_since: float | None = None
    terminal: dict | None = None
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


class LayeredMooncakeBackend(MooncakeTransferBackend):
    """Multi-lease receive pool and a fair sender sharing startup registrations."""

    layered_transfer = True

    def __init__(self, *args, layers_per_chunk=1, staging_slots=3, **kwargs):
        if type(layers_per_chunk) is not int or layers_per_chunk < 1:
            raise ValueError("layers_per_chunk must be positive")
        if type(staging_slots) is not int or not 2 <= staging_slots <= 4:
            raise ValueError("staging_slots must be between 2 and 4")
        self.layers_per_chunk, self.staging_slots = layers_per_chunk, staging_slots
        self._layout = None
        self._send_buffers = []
        self._receive_buffer = None
        self._leases = {}
        self._lease_attempts = {}
        self._layer_tombstones = OrderedDict()
        self._jobs = queue.Queue()
        self._batch_tasks = queue.Queue()
        self._send_jobs = {}
        self._control_sessions = {}
        self._sender_thread = None
        self._pipeline_closed = False
        super().__init__(*args, **kwargs)
        self.transport = (
            _TRANSPORT if self.transfer_mode == "gpu_direct" else "mooncake-layered-host-dram-v3"
        )
        self._gpu_workspace = self._gpu_indices = None

    def configure_pipeline(self, scheduler) -> None:
        """After Engine and bind_device, before the worker reports readiness."""
        if self._layout is not None:
            raise TransferError("layer pools are already configured")
        self._layout = scheduler.layout
        self._max_tokens = min(
            scheduler.engine.max_seq_len,
            self.max_transfer_bytes // self._layout.bytes_per_token,
        )
        if self._max_tokens < 1:
            raise TransferError("transfer limit cannot hold one prompt token")
        self._pool_bytes = self._max_tokens * self._layout.bytes_per_token
        self.transfer_stream = torch.cuda.Stream(device=scheduler.device)
        host_plan = None
        if self.transfer_mode == "host_dram":
            table = getattr(scheduler.engine, "page_table", None)
            rows = table.shape[0] - 1 if table is not None else 1
            host_plan = host_staging_plan(
                max_seq_len=scheduler.engine.max_seq_len,
                bytes_per_token=self._layout.bytes_per_token,
                layers=self._layout.layers,
                layers_per_chunk=self.layers_per_chunk,
                staging_slots=self.staging_slots,
                max_transfer_bytes=self.max_transfer_bytes,
                max_registered_bytes=self.runtime.max_registered_bytes,
                rows=rows,
                prefill=scheduler.role is WorkerRole.PREFILL,
            )
            self._gpu_workspace = torch.empty(
                host_plan.group_bytes, dtype=torch.uint8, device=scheduler.device
            )
            self._gpu_indices = torch.empty(
                host_plan.max_tokens, dtype=torch.int64, device=scheduler.device
            )
            self.runtime.staging_tensors = (self._gpu_workspace, self._gpu_indices)
            allocated = torch.cuda.Event()
            allocated.record(torch.cuda.current_stream(scheduler.device))
            self.transfer_stream.wait_event(allocated)
            if scheduler.role is WorkerRole.DECODE:
                self.install_stream = torch.cuda.Stream(device=scheduler.device)
                self.install_stream.wait_event(allocated)
        if scheduler.role is WorkerRole.PREFILL:
            size = (
                self._pool_bytes
                // self._layout.layers
                * min(self.layers_per_chunk, self._layout.layers)
            )
            if host_plan is not None:
                size = host_plan.group_bytes
            if size * self.staging_slots > self.runtime.max_registered_bytes:
                raise TransferError("sender ring exceeds --pd-max-registered-bytes")
            if self.staging_slots > self.runtime.max_pending_transfers:
                raise TransferError("sender ring exceeds --pd-max-pending-transfers")
            for slot in range(self.staging_slots):
                self._send_buffers.append(
                    self.runtime.allocate_buffer(
                        size, f"layer-slot:{slot}", pool=host_plan is not None
                    )
                )
            self._sender_thread = threading.Thread(
                target=self._sender_loop, name="pd-layer-sender", daemon=True
            )
            self._sender_thread.start()
        else:
            # Memory was excluded from Engine's KV capacity at startup. Keep
            # each request bounded by max_transfer_bytes, but share a larger MR.
            table = getattr(scheduler.engine, "page_table", None)
            rows = table.shape[0] - 1 if table is not None else 1
            capacity = min(self.runtime.max_registered_bytes, rows * (self._pool_bytes + 255))
            if host_plan is not None:
                capacity = host_plan.registered_bytes
            self._receive_buffer = self.runtime.allocate_buffer(
                capacity, "layer-receive-pool", pool=True
            )
            self._allocator = ReceiveAllocator(capacity)

    def check_capacity(self, nbytes):
        # Reservations are slices of startup registrations, not new allocations.
        self.check_available()
        if type(nbytes) is not int or not 0 < nbytes <= self._pool_bytes:
            raise TransferError("prompt KV exceeds the startup transfer pool")

    def reserve_layers(
        self, request: StartRequest, eos_token_id, *, prefix_len: int = 0
    ) -> LayeredReceiveTicket:
        if (
            type(prefix_len) is not int
            or not 0 <= prefix_len < len(request.prompt_ids)
            or prefix_len % self._layout.page_size
        ):
            raise TransferError("invalid layered transfer suffix")
        transfer_tokens = len(request.prompt_ids) - prefix_len
        nbytes = transfer_tokens * self._layout.bytes_per_token
        self.check_capacity(nbytes)
        with self._lock:
            if request.transfer_id in self._lease_attempts:
                raise TransferError("receive attempt is already reserved")
            if len(self._leases) >= self.runtime.max_pending_transfers:
                raise TransferCapacityError("receive lease budget is busy")
            locator = str(uuid4())
            offset = self._allocator.allocate(locator, nbytes)
            layer_bytes = nbytes // self._layout.layers
            try:
                ticket = LayeredReceiveTicket(
                    request.transfer_id,
                    self.endpoint,
                    locator,
                    self.runtime.session,
                    self._receive_buffer.address + offset,
                    self._layout,
                    transfer_tokens,
                    self.layers_per_chunk,
                    tuple(i * layer_bytes for i in range(self._layout.layers)),
                    eos_token_id,
                    prefix_len=prefix_len,
                    transfer_mode=self.transfer_mode,
                )
            except Exception:
                self._allocator.release(locator)
                raise
            self._leases[locator] = _Lease(request, ticket, offset=offset)
            self._lease_attempts[request.transfer_id] = locator
            return ticket

    def begin_layer_send(self, ticket, request, pool, cancel_check=None, *, wait=True):
        self.check_capacity(ticket.nbytes)
        if (
            ticket.layout != self._layout
            or ticket.transfer_id != request.transfer_id
            or ticket.prompt_tokens != len(request.prompt_ids)
            or ticket.layers_per_chunk != self.layers_per_chunk
            or ticket.transfer_mode != self.transfer_mode
        ):
            raise TransferError("incompatible early receive ticket")
        with self._lock:
            if ticket.transfer_id in self._send_jobs:
                raise TransferError("duplicate layer send attempt")
            if len(self._send_jobs) >= self.runtime.max_pending_transfers:
                raise TransferCapacityError("sender request budget is busy")
            job = LayeredSend(self, ticket, pool, cancel_check)
            self._send_jobs[ticket.transfer_id] = job
            self._jobs.put_nowait(job)
        if wait:
            if not job.opened.wait(self.timeout + 1):
                job.seal(cancel=True)
                raise TransferError("layer sender did not open its receive reservation")
            if job.error is not None:
                job.seal(cancel=True)
                raise job.error
        return job

    def _session(self, endpoint):
        with self._lock:
            if endpoint not in self._control_sessions:
                if len(self._control_sessions) >= self.runtime.max_pending_transfers:
                    active = {job.ticket.endpoint for job in self._send_jobs.values()}
                    idle = next((key for key in self._control_sessions if key not in active), None)
                    if idle is None:
                        raise TransferCapacityError("destination connection budget is busy")
                    self._control_sessions.pop(idle).close()
                self._control_sessions[endpoint] = ControlSession(self, endpoint)
            return self._control_sessions[endpoint]

    def _channel(self, sock):
        return Channel(sock, self.timeout)

    def _sender_loop(self):
        try:
            # Inference mode is thread-local. Startup pools may themselves be
            # inference tensors, so the packing writer needs it on this thread.
            with torch.inference_mode():
                sender_loop(self)
        except BaseException as exc:
            self.runtime.retire(f"layer sender failed; retain its buffers: {exc}")
            self._record_fatal(exc)

    @property
    def _lease(self):
        # Compatibility for ownership diagnostics written for a single lease.
        return next(iter(self._leases.values()), None)

    def _pack(self, job, task, buffer):
        group, slots, ready = task
        start, end = job.ticket.groups[group]
        nbytes = (end - start) * job.ticket.layer_bytes
        with self.runtime.cuda.activate():
            with torch.cuda.stream(self.transfer_stream):
                self.transfer_stream.wait_event(ready)
                if self.transfer_mode == "host_dram":
                    indices = self._gpu_indices[: len(slots)]
                    indices.copy_(slots, non_blocking=True)
                    job.pool.export_layers_into(
                        self._gpu_workspace[:nbytes],
                        slots,
                        start,
                        end,
                        num_pool_pages=job.pool._kv_buffer.shape[2] - 1,
                        trusted_indices=indices,
                    )
                    buffer.tensor[:nbytes].copy_(self._gpu_workspace[:nbytes], non_blocking=True)
                else:
                    job.pool.export_layers_into(
                        buffer.tensor[:nbytes],
                        slots,
                        start,
                        end,
                        num_pool_pages=job.pool._kv_buffer.shape[2] - 1,
                    )
                copied = torch.cuda.Event()
                copied.record(self.transfer_stream)
            # SYNC_MEMOPS is not a cross-stream producer fence. Both this host-side
            # packing fence and wait_event are required before native DMA. Keep
            # the Worker's CUDA context active also while waiting for the event.
            copied.synchronize()
        return nbytes

    @torch.inference_mode()
    def install_layers(self, pool, packed, slots, *, num_pool_pages):
        """Enqueue on the caller's install stream; no allocation, sync, or DRAM-cache insertion."""
        if self.transfer_mode == "gpu_direct":
            return pool.import_layered_tensor(packed, slots, num_pool_pages=num_pool_pages)
        indices = self._gpu_indices[: len(slots)]
        indices.copy_(slots, non_blocking=True)
        pool.import_host_layered_tensor(
            packed,
            indices,
            workspace=self._gpu_workspace,
            layers_per_chunk=self.layers_per_chunk,
        )

    def _send_terminal(self, ticket, submitted, completed, result, channel=None):
        message = {
            "op": "layer_terminal",
            "backend": self.transport,
            "locator": ticket.locator,
            "transfer_id": ticket.transfer_id,
            "receiver": ticket.session,
            "source": self.runtime.session,
            "submitted": sorted(submitted),
            "completed": sorted(completed),
            "result": result,
        }

        def exchange(ch):
            ch.send(message)
            reply = ch.receive()
            if (
                set(reply) != {"status", "locator"}
                or reply["locator"] != ticket.locator
                or reply["status"] not in ("complete", "retired", "quarantined")
            ):
                raise TransferError("incorrect layer completion receipt")
            if result == "complete" and reply["status"] != "complete":
                raise CancelledError("layer receiver no longer accepts the request")

        if channel is not None:
            exchange(channel)
        else:
            with self._connection(ticket.endpoint) as sock:
                exchange(Channel(sock, self.timeout))

    def _receive(self, sock: socket.socket) -> None:
        channel = Channel(sock, self.timeout, self._stop)
        try:
            while not self._stop.is_set():
                message = channel.receive()
                if message.get("backend") != self.transport:
                    raise TransferError("incompatible layered transfer backend")
                if message.get("op") == "layer_terminal":
                    channel.send(self._settle_layers(message))
                    continue
                self.check_available()
                if (
                    set(message) != {"op", "backend", "ticket", "source"}
                    or message["op"] != "layer_open"
                ):
                    raise TransferError("incorrect layer transfer open")
                ticket = LayeredReceiveTicket.from_wire(json.dumps(message["ticket"]).encode())
                channel.send(self.open_layers(ticket, message["source"]))
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            try:
                channel.send({"status": "error", "error": str(exc)})
            except (OSError, ValueError):
                pass

    def open_layers(self, ticket, source):
        from .control import endpoint_address

        endpoint_address(source)
        with self._lock:
            entry = self._get_lease(ticket)
            if (
                entry.cancelled
                or entry.finalizing
                or entry.terminal is not None
                or (entry.exposed and entry.source != source)
            ):
                raise TransferError("layer reservation was revoked or already opened")
            entry.source = source
            # Mark unsafe before granting permission. A partial ready reply
            # must never let cancellation reuse an address that P may write.
            self.runtime.expose(self._receive_buffer)
            entry.exposed = True
            entry.safe = False
            return {"status": "ready", "locator": ticket.locator}

    def _get_lease(self, ticket):
        entry = self._leases.get(ticket.locator)
        if entry is None or entry.ticket != ticket:
            raise TransferError("unknown or stale layer reservation")
        return entry

    def _refresh_pool_safety(self):
        entries = list(self._leases.values())
        quarantines = [
            e.quarantine_since for e in entries if e.quarantine_since is not None and not e.safe
        ]
        self.runtime.update_pool_safety(
            self._receive_buffer,
            safe=all(e.safe and not e.consuming for e in entries),
            quarantine_since=min(quarantines, default=None),
        )

    def _settle_layers(self, message):
        if not isinstance(message, dict) or not isinstance(message.get("locator"), str):
            raise TransferError("incorrect layer completion message")
        with self._lock:
            entry = self._leases.get(message["locator"])
        # Serialize duplicate receipts for this lease while letting other leases
        # reserve/release during the potentially slow CUDA visibility fence.
        with entry.lock if entry is not None else nullcontext():
            return self._settle_serial(message)

    def _settle_serial(self, message):
        if (
            not isinstance(message, dict)
            or set(message)
            != {
                "op",
                "backend",
                "locator",
                "transfer_id",
                "receiver",
                "source",
                "submitted",
                "completed",
                "result",
            }
            or message["op"] != "layer_terminal"
            or message["backend"] != self.transport
        ):
            raise TransferError("incorrect layer completion message")
        with self._lock:
            old = self._layer_tombstones.get(message["locator"])
            if old is not None:
                if old[0] != message:
                    raise TransferError("layer completion changed after retirement")
                return dict(old[1])
            entry = self._leases.get(message["locator"])
            if (
                entry is None
                or not entry.exposed
                or message["locator"] != entry.ticket.locator
                or message["transfer_id"] != entry.request.transfer_id
                or message["receiver"] != entry.ticket.session
                or message["source"] != entry.source
            ):
                raise TransferError("layer completion belongs to another reservation")
            if entry.terminal is not None and entry.terminal != message:
                raise TransferError("layer completion changed")
            groups = set(range(len(entry.ticket.groups)))
            submitted, completed = message["submitted"], message["completed"]
            if any(
                not isinstance(ids, list)
                or any(type(i) is not int for i in ids)
                or ids != sorted(set(ids))
                or not set(ids) <= groups
                for ids in (submitted, completed)
            ):
                raise TransferError("invalid layer completion bitmap")
            if not set(completed) <= set(submitted):
                raise TransferError("completed layers were not submitted")
            result = message["result"]
            if (
                result not in ("complete", "cancelled", "quarantined")
                or (result == "complete" and set(completed) != groups)
                or (result != "quarantined" and completed != submitted)
            ):
                raise TransferError("layer completion does not prove quiescence")
            visible = entry.terminal is None and result == "complete" and not entry.cancelled
            entry.finalizing = True
        if visible:
            self.runtime.make_visible(self._receive_buffer)
        with self._lock:
            if entry.terminal is None:
                if result == "quarantined":
                    entry.quarantine_since = time.monotonic()
                    entry.cancelled = True
                else:
                    if result == "complete" and not entry.cancelled:
                        entry.delivered = True
                    entry.safe = True
                    entry.quarantine_since = None
                    entry.cancelled |= result == "cancelled"
                entry.terminal = dict(message)
                entry.done.set()
            reply = {
                "status": (
                    "quarantined"
                    if result == "quarantined"
                    else ("retired" if entry.cancelled else "complete")
                ),
                "locator": entry.ticket.locator,
            }
            if entry.owner_released and entry.safe and not entry.consuming:
                self._forget_lease(entry, reply)
            self._refresh_pool_safety()
            return reply

    def finish_layers(self, ticket, *, wait=True):
        with self._lock:
            entry = self._get_lease(ticket)
        if not wait and not entry.done.is_set():
            return None
        if wait and not entry.done.wait(self.timeout):
            raise TransferError("layer receive completion timed out")
        with self._lock:
            if entry.cancelled or not entry.delivered or not entry.safe or entry.consuming:
                raise TransferError("layer receive is cancelled or incomplete")
            # Native writes are drained, but CUDA scatter is a separate owner.
            # Keep the pool unavailable until that consumer's fence is proved.
            self.runtime.expose(self._receive_buffer)
            entry.consuming = True
            return self._receive_buffer.tensor[entry.offset : entry.offset + ticket.nbytes]

    def release_layers(self, ticket, *, consumer_complete=False):
        with self._lock:
            entry = self._get_lease(ticket)
            entry.cancelled = True
            entry.owner_released = True
            if entry.consuming and consumer_complete:
                entry.consuming = False
            if entry.safe and not entry.consuming:
                self._forget_lease(entry, {"status": "retired", "locator": ticket.locator})
            elif not entry.safe and entry.quarantine_since is None:
                entry.quarantine_since = time.monotonic()
            self._refresh_pool_safety()

    def _forget_lease(self, entry, reply):
        if entry.terminal is not None:
            self._layer_tombstones[entry.ticket.locator] = (entry.terminal, dict(reply))
            while len(self._layer_tombstones) > 1024:
                self._layer_tombstones.popitem(last=False)
        self._leases.pop(entry.ticket.locator)
        self._lease_attempts.pop(entry.request.transfer_id)
        self._allocator.release(entry.ticket.locator)

    def health(self):
        result = super().health()
        with self._lock:
            result["mooncake"]["layered"] = {
                "transfer_mode": self.transfer_mode,
                "gpu_workspace_bytes": (
                    self._gpu_workspace.numel() if self._gpu_workspace is not None else 0
                ),
                "layers_per_chunk": self.layers_per_chunk,
                "staging_slots": self.staging_slots,
                "pool_bytes": getattr(self, "_pool_bytes", 0),
                "receive_reserved": bool(self._leases),
                "receive_leases": len(self._leases),
                "receive_pool_bytes": self._receive_buffer.nbytes if self._receive_buffer else 0,
                "receive_available_bytes": self._allocator.available if self._receive_buffer else 0,
                "quarantined_lease_bytes": sum(
                    e.ticket.nbytes
                    for e in self._leases.values()
                    if e.quarantine_since is not None and not e.safe
                ),
                "send_requests": len(self._send_jobs),
            }
        return result

    def close(self):
        if not self._pipeline_closed:
            for job in list(self._send_jobs.values()):
                job.seal(cancel=True)
            if self._sender_thread is not None:
                self._jobs.put_nowait(None)
                self._sender_thread.join(timeout=3 * self.timeout + 1)
                if self._sender_thread.is_alive():
                    self.runtime.retire("layer sender remains active during shutdown")
                    raise TransferBackendUnavailable("layer sender did not shut down")
            with self._lock:
                for entry in list(self._leases.values()):
                    self.release_layers(entry.ticket)
            for session in self._control_sessions.values():
                session.close()
            for buffer in self._send_buffers + (
                [self._receive_buffer] if self._receive_buffer else []
            ):
                if buffer.safe_to_release:
                    try:
                        self.runtime.release(buffer)
                    except TransferError as exc:
                        # Let runtime.close retain failed registrations and
                        # report the required process-exit boundary.
                        self._record_fatal(exc)
            self._pipeline_closed = True
        super().close()
