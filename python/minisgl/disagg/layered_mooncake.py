"""Startup-registered GPU pools and a layer-ready compute/communication pipeline.

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
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from uuid import uuid4

import torch

from .control import CancelledError, Channel, StartRequest
from .layered_protocol import LayeredReceiveTicket
from .mooncake_backend import MooncakeTransferBackend
from .role import WorkerRole
from .transfer import TransferBackendUnavailable, TransferError

_TRANSPORT = "mooncake-layered-gpu-v2"
_END = object()


@dataclass
class _Lease:
    request: StartRequest
    ticket: LayeredReceiveTicket
    source: str | None = None
    exposed: bool = False
    cancelled: bool = False
    owner_released: bool = False
    delivered: bool = False
    consuming: bool = False
    terminal: dict | None = None
    done: threading.Event = field(default_factory=threading.Event)


class LayeredSend:
    def __init__(self, backend, ticket, pool, cancel_check):
        self.backend, self.ticket, self.pool = backend, ticket, pool
        self.cancel_check = cancel_check
        self.tasks = queue.Queue()
        self.opened = threading.Event()
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.error = None
        self.safe_sources = True
        self._sealed = False
        self._last_layer = -1
        self._slots = None

    def set_slots(self, slots):
        # The snapshot is created before forward on the compute stream. Every
        # layer event below also orders its production against the transfer stream.
        self._slots = slots

    def layer_ready(self, layer: int) -> None:
        """Model callback: record an event and enqueue; never wait or pack here."""
        if self.error is not None:
            raise self.error
        if layer != self._last_layer + 1:
            raise TransferError("layer-ready notifications are not in model order")
        self._last_layer = layer
        if self.cancelled.is_set():
            return
        if (layer + 1) % self.ticket.layers_per_chunk and layer + 1 != self.ticket.layout.layers:
            return
        if self._slots is None:
            raise TransferError("layer transfer is missing its source slots")
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.backend.runtime.device))
        self.tasks.put_nowait((layer // self.ticket.layers_per_chunk, self._slots, event))

    def seal(self, *, cancel: bool = False) -> None:
        if cancel:
            self.cancelled.set()
        if not self._sealed:
            self._sealed = True
            self.tasks.put_nowait(_END)

    def finish(self, *, cancel: bool = False) -> None:
        self.seal(cancel=cancel)
        # Native drain and GPU packing fences remain on the sender thread.
        if not self.done.wait(3 * self.backend.timeout + 1):
            self.backend.runtime.retire("layer sender did not stop; its GPU pools must be retained")
            raise TransferBackendUnavailable("layer sender did not stop")
        if self.error is not None:
            raise self.error

    def _check_cancel(self):
        if self.cancel_check is not None and not self.cancelled.is_set():
            try:
                self.cancel_check()
            except (OSError, RuntimeError, ValueError) as exc:
                self.error = exc
                self.cancelled.set()

    def run(self):
        pending = deque()
        free = deque(self.backend._send_buffers)
        submitted, completed = set(), set()
        uncertain = False
        terminal_result = None

        def poll_one():
            nonlocal uncertain
            buffer, operation, group, deadline = pending[0]
            try:
                status = self.backend.runtime.poll(operation, timeout_ms=0)
            except (RuntimeError, ValueError):
                status = "unknown"
            if status == "unknown":
                if time.monotonic() >= deadline:
                    self.backend.runtime.quarantine(buffer, "layer write completion is unknown")
                    uncertain = True
                    pending.popleft()
                    self.error = TransferError("layer write did not drain before its deadline")
                    self.cancelled.set()
                    return True
                return False
            # This checks ticket.drained again before dropping the operation.
            self.backend.runtime.recycle_operation(operation)
            pending.popleft()
            free.append(buffer)
            if status == "completed":
                completed.add(group)
            else:
                uncertain = True  # FAILED_DRAINED cannot authorize a D-side reuse.
                self.error = TransferError("layer write failed after local drain")
                self.cancelled.set()
            return True

        try:
            with self.backend._connection(self.ticket.endpoint) as sock:
                channel = Channel(sock, self.backend.timeout)
                channel.send(
                    {
                        "op": "layer_open",
                        "backend": _TRANSPORT,
                        "ticket": json.loads(self.ticket.to_wire()),
                        "source": self.backend.runtime.session,
                    }
                )
                if channel.receive() != {"status": "ready", "locator": self.ticket.locator}:
                    raise TransferError("layer receive reservation was revoked or rejected")
                self.opened.set()
                expected = 0
                while True:
                    self._check_cancel()
                    try:
                        task = self.tasks.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    if task is _END:
                        break
                    if self.cancelled.is_set():
                        continue
                    group, slots, event = task
                    if group != expected:
                        raise TransferError("missing or repeated layer group")
                    expected += 1
                    while not free and not self.cancelled.is_set():
                        self._check_cancel()
                        if not poll_one():
                            time.sleep(0.002)  # Release the GIL; never Python busy-poll.
                    if self.cancelled.is_set():
                        continue
                    buffer = free.popleft()
                    self.safe_sources = False
                    nbytes = self.backend._pack(self, task, buffer)
                    self.safe_sources = True
                    self._check_cancel()
                    if self.cancelled.is_set():
                        free.append(buffer)
                        continue
                    start, _ = self.ticket.groups[group]
                    remote = {
                        "session": self.ticket.session,
                        "address": self.ticket.address + self.ticket.layer_offsets[start],
                        "nbytes": nbytes,
                    }
                    # A partially failed native submission can still access the slot.
                    try:
                        operation = self.backend.runtime.submit_write(buffer, remote, nbytes=nbytes)
                    except BaseException:
                        uncertain = True
                        raise
                    submitted.add(group)
                    pending.append(
                        (buffer, operation, group, time.monotonic() + self.backend.timeout)
                    )
                if not self.cancelled.is_set() and expected != len(self.ticket.groups):
                    raise TransferError("prefill did not produce every layer group")
                while pending:
                    self._check_cancel()
                    if not poll_one():
                        time.sleep(0.002)
                result = (
                    "quarantined"
                    if uncertain
                    else ("cancelled" if self.cancelled.is_set() else "complete")
                )
                terminal_result = result
                self.backend._send_terminal(self.ticket, submitted, completed, result, channel)
        except BaseException as exc:
            self.error = exc
            self.cancelled.set()
            while pending:
                if not poll_one():
                    time.sleep(0.002)
            # Re-send the exact certificate after a lost receipt. Changing a
            # previously recorded success into cancellation breaks idempotency.
            result = terminal_result or ("quarantined" if uncertain else "cancelled")
            # An independent connection is needed when the original control
            # channel failed. If both receipts are lost, D keeps its reservation.
            try:
                self.backend._send_terminal(self.ticket, submitted, completed, result)
            except (OSError, RuntimeError, ValueError):
                pass
        finally:
            if uncertain or not self.safe_sources:
                self.backend.runtime.retire("layer transfer could not prove safe completion")
            self.opened.set()
            self.done.set()


class LayeredMooncakeBackend(MooncakeTransferBackend):
    """One fixed receive region or a fixed sender ring per serial split Worker."""

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
        self._lease = None
        self._layer_tombstones = OrderedDict()
        self._jobs = queue.Queue()
        self._sender_thread = None
        self._job = None
        self._pipeline_closed = False
        super().__init__(*args, **kwargs)

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
        if scheduler.role is WorkerRole.PREFILL:
            size = (
                self._pool_bytes
                // self._layout.layers
                * min(self.layers_per_chunk, self._layout.layers)
            )
            if size * self.staging_slots > self.runtime.max_registered_bytes:
                raise TransferError("sender ring exceeds --pd-max-registered-bytes")
            if self.staging_slots > self.runtime.max_pending_transfers:
                raise TransferError("sender ring exceeds --pd-max-pending-transfers")
            for slot in range(self.staging_slots):
                self._send_buffers.append(self.runtime.allocate_buffer(size, f"layer-slot:{slot}"))
            self._sender_thread = threading.Thread(
                target=self._sender_loop, name="pd-layer-sender", daemon=True
            )
            self._sender_thread.start()
        else:
            self._receive_buffer = self.runtime.allocate_buffer(
                self._pool_bytes, "layer-receive-pool"
            )

    def check_capacity(self, nbytes):
        # Reservations are slices of startup registrations, not new allocations.
        self.check_available()
        if type(nbytes) is not int or not 0 < nbytes <= self._pool_bytes:
            raise TransferError("prompt KV exceeds the startup transfer pool")

    def reserve_layers(self, request: StartRequest, eos_token_id) -> LayeredReceiveTicket:
        nbytes = len(request.prompt_ids) * self._layout.bytes_per_token
        self.check_capacity(nbytes)
        with self._lock:
            if self._lease is not None:
                raise TransferError("the receive pool is busy or quarantined")
            layer_bytes = nbytes // self._layout.layers
            ticket = LayeredReceiveTicket(
                request.transfer_id,
                self.endpoint,
                str(uuid4()),
                self.runtime.session,
                self._receive_buffer.address,
                self._layout,
                len(request.prompt_ids),
                self.layers_per_chunk,
                tuple(i * layer_bytes for i in range(self._layout.layers)),
                eos_token_id,
            )
            self._lease = _Lease(request, ticket)
            return ticket

    def begin_layer_send(self, ticket, request, pool, cancel_check):
        self.check_capacity(ticket.nbytes)
        if (
            ticket.layout != self._layout
            or ticket.transfer_id != request.transfer_id
            or ticket.kv_tokens != len(request.prompt_ids)
            or ticket.layers_per_chunk != self.layers_per_chunk
        ):
            raise TransferError("incompatible early receive ticket")
        with self._lock:
            if self._job is not None and not self._job.done.is_set():
                raise TransferError("the layer sender is busy")
            job = self._job = LayeredSend(self, ticket, pool, cancel_check)
            self._jobs.put_nowait(job)
        if not job.opened.wait(self.timeout + 1):
            job.seal(cancel=True)
            raise TransferError("layer sender did not open its receive reservation")
        if job.error is not None:
            raise job.error
        return job

    def _sender_loop(self):
        while True:
            job = self._jobs.get()
            if job is _END:
                return
            job.run()

    def _pack(self, job, task, buffer):
        group, slots, ready = task
        start, end = job.ticket.groups[group]
        nbytes = (end - start) * job.ticket.layer_bytes
        with self.runtime.cuda.activate():
            with torch.cuda.stream(self.transfer_stream):
                self.transfer_stream.wait_event(ready)
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

    def _send_terminal(self, ticket, submitted, completed, result, channel=None):
        message = {
            "op": "layer_terminal",
            "backend": _TRANSPORT,
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
        channel = Channel(sock, self.timeout)
        try:
            message = channel.receive()
            if message.get("backend") != _TRANSPORT:
                raise TransferError("incompatible layered transfer backend")
            if message.get("op") == "layer_terminal":
                channel.send(self._settle_layers(message))
                return
            self.check_available()
            if (
                set(message) != {"op", "backend", "ticket", "source"}
                or message["op"] != "layer_open"
            ):
                raise TransferError("incorrect layer transfer open")
            ticket = LayeredReceiveTicket.from_wire(json.dumps(message["ticket"]).encode())
            channel.send(self.open_layers(ticket, message["source"]))
            terminal = channel.receive()
            channel.send(self._settle_layers(terminal))
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            # The reservation owner handles revocation. A disconnected sender
            # never constitutes evidence that an exposed pool is safe to reuse.
            try:
                channel.send({"status": "error", "error": str(exc)})
            except (OSError, ValueError):
                pass

    def open_layers(self, ticket, source):
        from .control import endpoint_address

        endpoint_address(source)
        with self._lock:
            entry = self._get_lease(ticket)
            if entry.cancelled or entry.exposed:
                raise TransferError("layer reservation was revoked or already opened")
            entry.source = source
            # Mark unsafe before granting permission. A partial ready reply
            # must never let cancellation reuse an address that P may write.
            self.runtime.expose(self._receive_buffer)
            entry.exposed = True
            return {"status": "ready", "locator": ticket.locator}

    def _get_lease(self, ticket):
        if self._lease is None or self._lease.ticket != ticket:
            raise TransferError("unknown or stale layer reservation")
        return self._lease

    def _settle_layers(self, message):
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
            or message["backend"] != _TRANSPORT
        ):
            raise TransferError("incorrect layer completion message")
        with self._lock:
            old = self._layer_tombstones.get(message["locator"])
            if old is not None:
                if old[0] != message:
                    raise TransferError("layer completion changed after retirement")
                return dict(old[1])
            entry = self._lease
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
            if entry.terminal is None:
                if result == "quarantined":
                    self.runtime.quarantine(
                        self._receive_buffer, "peer could not prove every layer write"
                    )
                    entry.cancelled = True
                else:
                    if result == "complete" and not entry.cancelled:
                        self.runtime.make_visible(self._receive_buffer)
                        entry.delivered = True
                    self.runtime.mark_safe(self._receive_buffer)
                    entry.cancelled |= result == "cancelled"
                entry.terminal = dict(message)
                entry.done.set()
            reply = {
                "status": "quarantined"
                if result == "quarantined"
                else ("retired" if entry.cancelled else "complete"),
                "locator": entry.ticket.locator,
            }
            if entry.owner_released and self._receive_buffer.safe_to_release:
                self._forget_lease(entry, reply)
            return reply

    def finish_layers(self, ticket):
        with self._lock:
            entry = self._get_lease(ticket)
        if not entry.done.wait(self.timeout):
            raise TransferError("layer receive completion timed out")
        with self._lock:
            if entry.cancelled or not entry.delivered or not self._receive_buffer.safe_to_release:
                raise TransferError("layer receive is cancelled or incomplete")
            # Native writes are drained, but CUDA scatter is a separate owner.
            # Keep the pool unavailable until that consumer's fence is proved.
            self.runtime.expose(self._receive_buffer)
            entry.consuming = True
            return self._receive_buffer.tensor[: ticket.nbytes]

    def release_layers(self, ticket, *, consumer_complete=False):
        with self._lock:
            entry = self._get_lease(ticket)
            entry.cancelled = True
            entry.owner_released = True
            if entry.consuming and consumer_complete:
                self.runtime.mark_safe(self._receive_buffer)
                entry.consuming = False
            if self._receive_buffer.safe_to_release:
                self._forget_lease(entry, {"status": "retired", "locator": ticket.locator})
            else:
                self.runtime.quarantine(
                    self._receive_buffer, "revoked layer reservation awaits sender drain"
                )

    def _forget_lease(self, entry, reply):
        if entry.terminal is not None:
            self._layer_tombstones[entry.ticket.locator] = (entry.terminal, dict(reply))
            while len(self._layer_tombstones) > 1024:
                self._layer_tombstones.popitem(last=False)
        self._lease = None

    def health(self):
        result = super().health()
        result["mooncake"]["layered"] = {
            "layers_per_chunk": self.layers_per_chunk,
            "staging_slots": self.staging_slots,
            "pool_bytes": getattr(self, "_pool_bytes", 0),
            "receive_reserved": self._lease is not None,
        }
        return result

    def close(self):
        if not self._pipeline_closed:
            job = self._job
            if job is not None and not job.done.is_set():
                job.seal(cancel=True)
            if self._sender_thread is not None:
                self._jobs.put_nowait(_END)
                self._sender_thread.join(timeout=3 * self.timeout + 1)
                if self._sender_thread.is_alive():
                    self.runtime.retire("layer sender remains active during shutdown")
                    raise TransferBackendUnavailable("layer sender did not shut down")
            with self._lock:
                if self._lease is not None:
                    self.release_layers(self._lease.ticket)
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
