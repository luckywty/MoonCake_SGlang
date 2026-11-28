"""Same-host layered relay: HBM -> pinned POSIX SHM -> HBM.

The socket carries descriptors/receipts only. D's scheduler installs groups as
they arrive, even before commit; only commit + ALL installed admits Decode.
CUDA events stay process-local. No cudaIpcEvent and no CPU payload memcpy.
"""

from __future__ import annotations

import json
import math
import queue
import socket
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import torch

from .control import Channel, endpoint_address
from .layered_protocol import LayeredReceiveTicket
from .layered_sender import ControlSession, LayeredSend
from .shm_relay_pool import READY, RelayAllocator, RelayManifest, SharedRelayPool
from .tcp_backend import TcpTransferBackend
from .transfer import TransferBackendUnavailable, TransferCapacityError, TransferError

TRANSPORT = "shm-layered-dram-v3"


@dataclass
class ShmReceive:
    ticket: LayeredReceiveTicket
    owner: str | None = None
    queued: deque = field(default_factory=deque)
    descriptors: dict = field(default_factory=dict)
    pending: list = field(default_factory=list)
    completed: set = field(default_factory=set)
    cancelled: bool = False
    terminal: bool = False
    error: Exception | None = None

    @property
    def drained(self):
        return not self.pending and not self.queued


class LayeredShmBackend(TcpTransferBackend):
    name = "shm_dram"
    transport = TRANSPORT
    transfer_mode = "shm_dram"
    layered_transfer = True
    incremental_install = True

    def __init__(
        self,
        bind=None,
        *,
        manifest,
        worker_id,
        numa_id,
        host_id,
        layers_per_chunk=1,
        staging_slots=3,
        overflow_slots=1,
        local_wait_ms=1.0,
        max_registered_bytes=None,
        max_pending_transfers=32,
        **kwargs,
    ):
        self.manifest = (
            manifest
            if isinstance(manifest, RelayManifest)
            else RelayManifest.from_json(Path(manifest).read_text())
        )
        self.manifest.validate(budget=max_registered_bytes)
        if host_id != self.manifest.host_id or numa_id not in self.manifest.node_slots:
            raise ValueError("shm_dram Worker host/NUMA does not match the shared pool")
        if (
            layers_per_chunk != self.manifest.layers_per_chunk
            or type(staging_slots) is not int
            or not 2 <= staging_slots <= 4
            or type(overflow_slots) is not int
            or overflow_slots < 0
            or local_wait_ms < 0
            or not math.isfinite(local_wait_ms)
            or type(max_pending_transfers) is not int
            or max_pending_transfers < 1
        ):
            raise ValueError("invalid shared relay layer, slot or wait configuration")
        self.worker_id, self.numa_id, self.host_id = worker_id, numa_id, host_id
        self.layers_per_chunk, self.local_wait_ms = layers_per_chunk, local_wait_ms
        self.max_pending_transfers = max_pending_transfers
        self.shared = SharedRelayPool(self.manifest)
        self.allocator = None if bind is not None else RelayAllocator(self.shared, worker_id)
        self._lock = threading.RLock()
        self._receives, self._recent, self._send_jobs, self._sessions = {}, OrderedDict(), {}, {}
        self._batch_tasks, self._jobs = queue.Queue(), queue.Queue()
        self._sender = None
        self._fatal = None
        self._layout = None
        self.runtime = self  # BatchTransferPlan/LayeredSend only need device and retire().
        self.transfer_stream = self.install_stream = None
        try:
            super().__init__(bind, **kwargs)
        except BaseException:
            self.shared.close()
            raise

    def retire(self, reason):
        self._fatal = str(reason)

    def bind_device(self, device):
        self.device = torch.device(device)

    def configure_pipeline(self, scheduler):
        self.scheduler, self._layout = scheduler, scheduler.layout
        self.device = scheduler.device
        layout, manifest = self._layout, self.manifest
        if (layout.layers, layout.bytes_per_token, self.layers_per_chunk) != (
            manifest.layers,
            manifest.bytes_per_token,
            manifest.layers_per_chunk,
        ):
            raise ValueError("shared relay layout differs from Worker KV geometry")
        if self.device.type == "cuda":
            self.shared.register_cuda()  # Mandatory: fail explicitly, never use pageable SHM.
            self.transfer_stream = torch.cuda.Stream(device=self.device)
            self.install_stream = torch.cuda.Stream(device=self.device)
        self.workspace = torch.empty(manifest.group_bytes, dtype=torch.uint8, device=self.device)
        self.indices = torch.empty(manifest.max_tokens, dtype=torch.int64, device=self.device)
        if self.allocator is not None:
            self._sender = threading.Thread(target=self._pump, name="pd-shm-send", daemon=True)
            self._sender.start()

    @contextmanager
    def _connection(self, endpoint):
        with socket.create_connection(endpoint_address(endpoint), timeout=self.timeout) as sock:
            yield sock

    def _channel(self, sock):
        return Channel(sock, self.timeout)

    def _session(self, endpoint):
        if endpoint not in self._sessions:
            self._sessions[endpoint] = ControlSession(self, endpoint)
        return self._sessions[endpoint]

    def _submit(self, job, op, **extra):
        def exchange(channel):
            channel.send(
                {
                    "backend": TRANSPORT,
                    "op": op,
                    "ticket": json.loads(job.ticket.to_wire()),
                    "pool": self.manifest.name,
                    "host_id": self.host_id,
                    "owner": self.worker_id,
                    **extra,
                }
            )
            result = channel.receive()
            if result.get("status") == "error":
                raise TransferError(result.get("error", "shared relay RPC rejected"))
            return result

        return self._session(job.ticket.endpoint).submit(exchange)

    def reserve_layers(self, request, eos_token_id, *, prefix_len=0):
        tokens = len(request.prompt_ids) - prefix_len
        if not 0 < tokens <= self.manifest.max_tokens:
            raise TransferCapacityError("shared relay prompt suffix exceeds startup capacity")
        if tokens * self._layout.bytes_per_token > self.max_transfer_bytes:
            raise TransferCapacityError("shared relay transfer exceeds byte limit")
        layer_bytes = tokens * self._layout.bytes_per_token // self._layout.layers
        ticket = LayeredReceiveTicket(
            transfer_id=request.transfer_id,
            endpoint=self.endpoint,
            locator=str(uuid4()),
            session=self.endpoint,
            address=1,
            layout=self._layout,
            kv_tokens=tokens,
            layers_per_chunk=self.layers_per_chunk,
            layer_offsets=tuple(i * layer_bytes for i in range(self._layout.layers)),
            eos_token_id=eos_token_id,
            prefix_len=prefix_len,
            transfer_mode="shm_dram",
        )
        with self._lock:
            if len(self._receives) >= self.max_pending_transfers:
                raise TransferCapacityError("shared relay receive reservations are busy")
            self._receives[ticket.locator] = ShmReceive(ticket)
        return ticket

    def _lookup_layer(self, ticket):
        entry = self._receives.get(ticket.locator) or self._recent.get(ticket.locator)
        if entry is None or entry.ticket != ticket:
            raise TransferError("shared relay ticket was revoked or belongs to another attempt")
        return entry

    def _receive(self, sock):
        channel = Channel(sock, self.timeout, self._stop)
        while not self._stop.is_set():
            try:
                message = channel.receive()
                with self._lock:
                    result = self._dispatch(message)
                channel.send(result)
            except (OSError, RuntimeError, ValueError) as exc:
                try:
                    channel.send({"status": "error", "error": str(exc)})
                except (OSError, RuntimeError):
                    pass
                return

    def _dispatch(self, message):
        if (
            message.get("backend") != TRANSPORT
            or message.get("pool") != self.manifest.name
            or message.get("host_id") != self.host_id
        ):
            raise TransferError("shm_dram requires matching host and shared relay identity")
        ticket = LayeredReceiveTicket.from_wire(json.dumps(message["ticket"]).encode())
        entry = self._lookup_layer(ticket)
        owner = message.get("owner")
        try:
            partition = self.manifest.partition(owner)
        except StopIteration as exc:
            raise TransferError("unknown shared relay partition owner") from exc
        if entry.owner is not None and entry.owner != owner:
            raise TransferError("shared relay reservation owner changed")
        entry.owner = owner
        op = message.get("op")
        if op == "shm_open":
            return {"status": "ready", "numa_id": self.numa_id}
        if op == "shm_notify":
            group, descriptor = message["group"], message["descriptor"]
            if type(group) is not int or not 0 <= group < len(ticket.groups):
                raise TransferError("invalid shared relay layer group")
            expected = {
                "slot_numa_id",
                "slot_id",
                "generation",
                "request_id",
                "layer_start",
                "layer_end",
                "nbytes",
            }
            if not isinstance(descriptor, dict) or set(descriptor) != expected:
                raise TransferError("invalid shared relay descriptor")
            if any(type(descriptor[k]) is not int for k in expected - {"request_id"}):
                raise TransferError("shared relay descriptor dimensions must be integers")
            node, slot, generation = (
                descriptor[k] for k in ("slot_numa_id", "slot_id", "generation")
            )
            owned = (node == partition.numa_id and slot in partition.local_slots) or (
                node == partition.overflow_numa and slot in partition.overflow_slots
            )
            start, end = ticket.groups[group]
            if (
                not owned
                or generation < 1
                or descriptor["request_id"] != ticket.transfer_id
                or (descriptor["layer_start"], descriptor["layer_end"], descriptor["nbytes"])
                != (start, end, (end - start) * ticket.layer_bytes)
            ):
                raise TransferError("shared relay descriptor exceeds its owner or layer range")
            if group in entry.descriptors:
                if entry.descriptors[group] != descriptor:
                    raise TransferError(
                        "shared relay group was repeated with a different generation"
                    )
                return {"status": "accepted"}  # Idempotent even after that slot has been reused.
            if entry.cancelled or entry.terminal:
                raise TransferError("shared relay request is closed")
            if self.shared.header(node, slot) != (generation, READY):
                raise TransferError("shared relay slot is stale or D2H has not published READY")
            if any(
                d["slot_numa_id"] == node and d["slot_id"] == slot and g not in entry.completed
                for g, d in entry.descriptors.items()
            ):
                raise TransferError("shared relay slot is still in use")
            entry.descriptors[group] = descriptor
            entry.queued.append(group)
            return {"status": "accepted"}
        if op == "shm_cancel":
            entry.cancelled = True
            entry.queued.clear()  # Not yet submitted to CUDA; these readers never started.
        elif op == "shm_complete":
            if entry.cancelled or len(entry.completed) != len(ticket.groups):
                raise TransferError("not all shared relay layers have finished installation")
            entry.terminal = True
        elif op != "shm_status":
            raise TransferError("unknown shared relay operation")
        return {
            "status": "ok",
            "completed": sorted(entry.completed),
            "cancelled": entry.cancelled,
            "drained": entry.drained,
            "error": str(entry.error) if entry.error else None,
        }

    def begin_layer_send(self, ticket, request, pool, cancel_check=None, *, wait=True):
        if (
            ticket.transfer_mode != "shm_dram"
            or ticket.layout != self._layout
            or ticket.prompt_tokens != len(request.prompt_ids)
            or ticket.transfer_id != request.transfer_id
            or self.allocator is None
            or ticket.layers_per_chunk != self.layers_per_chunk
            or ticket.kv_tokens > self.manifest.max_tokens
            or ticket.nbytes > self.max_transfer_bytes
        ):
            raise TransferError("incorrect shared relay send ticket")
        with self._lock:
            if self._fatal:
                raise TransferBackendUnavailable(self._fatal)
            if (
                ticket.transfer_id in self._send_jobs
                or len(self._send_jobs) >= self.max_pending_transfers
            ):
                raise TransferCapacityError("shared relay sender is busy")
            job = LayeredSend(self, ticket, pool, cancel_check)
            job.leases, job.wait_started, job.last_poll = {}, None, 0.0
            job.started, job.target_numa = time.monotonic(), None
            job.action, job.cancelling = "open", False
            job.control = self._submit(job, "shm_open")
            self._send_jobs[ticket.transfer_id] = job
            self._jobs.put_nowait(job)
        if wait and not job.opened.wait(self.timeout):
            job.seal(cancel=True)
            raise TransferError("shared relay receiver did not open")
        return job

    def _pack(self, job, task, lease):
        group, slots, ready = task
        start, end = job.ticket.groups[group]
        nbytes = (end - start) * job.ticket.layer_bytes
        destination = self.shared.tensor(lease.slot_numa_id, lease.slot_id, nbytes)
        stream = self.transfer_stream
        copied = None
        try:
            if stream is not None:
                stream.wait_event(ready)
            elif ready is not None:
                ready.synchronize()
            with torch.cuda.stream(stream) if stream is not None else nullcontext():
                indices = self.indices[: len(slots)]
                indices.copy_(slots, non_blocking=True)
                workspace = self.workspace[:nbytes]
                job.pool.export_layers_into(
                    workspace,
                    slots,
                    start,
                    end,
                    num_pool_pages=self.scheduler.engine.num_pages,
                    trusted_indices=indices,
                )
                destination.copy_(workspace, non_blocking=True)
        finally:
            if stream is not None:
                try:
                    copied = torch.cuda.Event()
                    copied.record(stream)
                    copied.synchronize()  # Barrier 1; BACKGROUND sender only.
                except BaseException as exc:
                    self.retire("D2H completion is unknown; retain SHM and source KV")
                    raise TransferBackendUnavailable(self._fatal) from exc
        self.allocator.ready(lease, copied=True)
        return {**asdict(lease), "layer_start": start, "layer_end": end, "nbytes": nbytes}

    def _consume_reply(self, job, result):
        if job.action == "open":
            node = result.get("numa_id")
            if type(node) is not int or node not in self.manifest.node_slots:
                raise TransferError("invalid receiver NUMA identity")
            job.target_numa = node
            job.opened.set()
        elif job.action in ("status", "cancel", "complete"):
            if result.get("error"):
                job.error = TransferError(result["error"])
                job.cancelled.set()
            for group in result.get("completed", []):
                lease = job.leases.pop(group, None)
                if lease is not None:
                    self.allocator.release(lease, consumer_complete=True)  # Barrier 2 receipt.
                    job.completed.add(group)
            if result.get("cancelled"):
                job.cancelled.set()
                if result.get("drained"):
                    self.allocator.reclaim_request(job.ticket.transfer_id, consumer_complete=True)
                    job.leases.clear()
                    job.cancel_drained = True
            if job.action == "complete":
                job.terminal_started = True
        job.control = None

    @torch.inference_mode()
    def _pump(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        active = deque()
        while not self._stop.is_set() or active or not self._jobs.empty():
            while not self._jobs.empty():
                active.append(self._jobs.get_nowait())
            while True:
                try:
                    group, plan, event = self._batch_tasks.get_nowait()
                except queue.Empty:
                    break
                if group is None:
                    plan._producer_done = True
                else:
                    for entry in plan.entries:
                        if entry.is_last_chunk:
                            entry.job.tasks.append((group, entry.slots, event))
            for _ in range(len(active)):
                job = active.popleft()
                try:
                    if time.monotonic() - job.started > 3 * self.timeout:
                        raise TransferBackendUnavailable(
                            "shared relay peer did not prove H2D drain"
                        )
                    job._check_cancel()
                    if self._stop.is_set():
                        job.cancelled.set()
                    if job.control is not None:
                        if not job.control.done():
                            active.append(job)
                            continue
                        self._consume_reply(job, job.control.result())
                    if job.cancelled.is_set():
                        job.tasks.clear()
                        if job._producer_done and job.safe_sources:
                            job.sources_done.set()
                        if not getattr(job, "cancel_drained", False):
                            job.action, job.control = "cancel", self._submit(job, "shm_cancel")
                        elif job._producer_done:
                            job.done.set()
                    elif job.terminal_started:
                        job.done.set()
                    else:
                        now = time.monotonic()
                        if job.leases and now - job.last_poll >= 0.002:
                            job.action, job.control = "status", self._submit(job, "shm_status")
                            job.last_poll = now
                        elif job.tasks:
                            if job.wait_started is None:
                                job.wait_started = now
                            lease = self.allocator.acquire(
                                job.ticket.transfer_id,
                                job.target_numa,
                                elapsed_ms=(now - job.wait_started) * 1000,
                                local_wait_ms=self.local_wait_ms,
                            )
                            if lease is not None:
                                task = job.tasks.popleft()
                                group = task[0]
                                if group != job.expected:
                                    self.allocator.release(lease, consumer_complete=True)
                                    raise TransferError(
                                        "missing or repeated shared relay layer group"
                                    )
                                job.expected += 1
                                job.safe_sources = False
                                job.leases[group] = lease
                                try:
                                    descriptor = self._pack(job, task, lease)
                                finally:
                                    job.safe_sources = self._fatal is None
                                job.submitted.add(group)
                                job.action = "notify"
                                job.control = self._submit(
                                    job, "shm_notify", group=group, descriptor=descriptor
                                )
                                job.wait_started = None
                            elif now - job.wait_started > self.timeout:
                                raise TransferCapacityError(
                                    "shared relay slots did not drain before timeout"
                                )
                        if job._producer_done and not job.tasks and job.safe_sources:
                            job.sources_done.set()
                            if not job.leases and job.control is None:
                                if job.expected != len(job.ticket.groups):
                                    raise TransferError(
                                        "prefill did not produce all shared relay groups"
                                    )
                                job.action, job.control = "complete", self._submit(
                                    job, "shm_complete"
                                )
                except BaseException as exc:
                    job.error = exc
                    job.cancelled.set()
                    if (
                        isinstance(exc, TransferBackendUnavailable)
                        or time.monotonic() - job.started > 3 * self.timeout
                    ):
                        # Failed RPCs first try a cancellation/drain handshake. If
                        # that cannot prove safety either, preserve every address.
                        self.allocator.reclaim_request(job.ticket.transfer_id)
                        self.retire(str(exc))
                        job.error = TransferBackendUnavailable(str(exc))
                        job.uncertain = True
                        job.done.set()
                    job.control = None
                if job.done.is_set():
                    if job._producer_done and job.safe_sources:
                        job.sources_done.set()
                    with self._lock:
                        self._send_jobs.pop(job.ticket.transfer_id, None)
                else:
                    active.append(job)
            time.sleep(0.001)

    def _poll_installs(self, entry):
        for item in list(entry.pending):
            group, event, tensor = item
            try:
                ready = event is None or event.query()
            except BaseException as exc:
                self.retire("H2D completion is unknown; retain SHM and destination KV")
                raise TransferBackendUnavailable(self._fatal) from exc
            if ready:
                descriptor = entry.descriptors[group]
                node, slot = descriptor["slot_numa_id"], descriptor["slot_id"]
                if self.shared.header(node, slot) != (descriptor["generation"], READY):
                    self.retire("shared relay generation changed before H2D completion")
                    raise TransferBackendUnavailable(self._fatal)
                entry.pending.remove(item)
                entry.completed.add(group)

    @torch.inference_mode()
    def progress_receive(self, ticket, slots, *, cancel=False):
        """Scheduler thread only. No host waits; each completed group returns its slot."""
        with self._lock:
            entry = self._lookup_layer(ticket)
            if cancel:
                entry.cancelled = True
                entry.queued.clear()
            self._poll_installs(entry)
            if entry.cancelled:
                return entry.drained
            stream = self.install_stream
            if stream is not None:
                stream.wait_stream(self.scheduler.engine.stream)
            while entry.queued:
                group = entry.queued.popleft()
                descriptor = entry.descriptors[group]
                node, slot, generation = (
                    descriptor[k] for k in ("slot_numa_id", "slot_id", "generation")
                )
                if self.shared.header(node, slot) != (generation, READY):
                    raise TransferError("shared relay slot changed before installation")
                tensor = self.shared.tensor(node, slot, descriptor["nbytes"])
                event = None
                try:
                    with torch.cuda.stream(stream) if stream is not None else nullcontext():
                        indices = self.indices[: len(slots)]
                        indices.copy_(slots, non_blocking=True)
                        self.scheduler.engine.kv_cache.import_host_layered_tensor(
                            tensor,
                            indices,
                            workspace=self.workspace,
                            layers_per_chunk=self.layers_per_chunk,
                            layer_start=descriptor["layer_start"],
                            layer_end=descriptor["layer_end"],
                        )
                except BaseException as exc:
                    entry.error, entry.cancelled = exc, True
                    entry.queued.clear()
                finally:
                    if stream is not None:
                        try:
                            event = torch.cuda.Event()
                            event.record(stream)
                        except BaseException as exc:
                            self.retire("cannot fence H2D installation")
                            raise TransferBackendUnavailable(self._fatal) from exc
                    entry.pending.append((group, event, tensor))  # Keep mapping/tensor alive.
                if entry.cancelled:
                    break
            self._poll_installs(entry)
            if entry.error is not None:
                raise entry.error
            return entry.drained

    def finish_layers(self, ticket, *, wait=False):
        with self._lock:
            entry = self._lookup_layer(ticket)
            if entry.cancelled:
                raise TransferError("shared relay request was cancelled")
            return entry if entry.terminal and len(entry.completed) == len(ticket.groups) else None

    def install_layers(self, pool, staging, slots, *, num_pool_pages):
        if not staging.terminal or len(staging.completed) != len(staging.ticket.groups):
            raise TransferError("shared relay installation is incomplete")
        # Each layer has already been installed by progress_receive, before commit.

    def release_layers(self, ticket, *, consumer_complete=False):
        with self._lock:
            entry = self._lookup_layer(ticket)
            if not consumer_complete:
                entry.cancelled = True
                entry.queued.clear()
            self._poll_installs(entry)
            if not entry.drained:
                return False
            self._receives.pop(ticket.locator, None)
            self._recent[ticket.locator] = entry
            while len(self._recent) > 4096:
                self._recent.popitem(last=False)
            return True

    def health(self):
        # Keep the existing generic layered health shape; no Mooncake TE is constructed.
        return {
            "status": "error" if self._fatal else "ready",
            "reason": self._fatal,
            "shm": {
                "name": self.manifest.name,
                "registered_bytes": self.manifest.registered_bytes,
                "numa_id": self.numa_id,
            },
            "mooncake": {"layered": {"layers_per_chunk": self.layers_per_chunk}},
        }

    def close(self):
        self._stop.set()
        for job in list(self._send_jobs.values()):
            job.seal(cancel=True)
        if self._sender is not None:
            self._sender.join(3 * self.timeout + 1)
            if self._sender.is_alive():
                raise TransferBackendUnavailable(
                    "shared relay sender did not stop; retain mappings"
                )
        for session in self._sessions.values():
            session.close()
        with self._lock:
            for entry in self._receives.values():
                self._poll_installs(entry)
                if entry.pending:
                    raise TransferBackendUnavailable(
                        "shared relay H2D is still active; retain mappings"
                    )
        if self._fatal:
            raise TransferBackendUnavailable(self._fatal)
        super().close()
        self.shared.close()
