"""One fair sender pump, shared registered slots, independent request receipts."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from .transfer import TransferBackendUnavailable, TransferError

TRANSPORT = "mooncake-layered-gpu-v3"


class ControlSession:
    """Reuse a short-RPC control connection per destination, never hold it for DMA."""

    def __init__(self, backend, endpoint):
        self.backend, self.endpoint = backend, endpoint
        self.lock = threading.Lock()
        self.context = self.channel = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pd-layer-control")

    def disconnect(self):
        if self.context is not None:
            self.context.__exit__(None, None, None)
        self.context = self.channel = None

    def close(self):
        self.executor.shutdown(wait=True)
        self.disconnect()

    def submit(self, action):
        return self.executor.submit(self.exchange, action)

    def exchange(self, action):
        with self.lock:
            for attempt in range(2):
                try:
                    if self.channel is None:
                        self.context = self.backend._connection(self.endpoint)
                        sock = self.context.__enter__()
                        self.channel = self.backend._channel(sock)
                    return action(self.channel)
                except (OSError, RuntimeError, ValueError):
                    self.disconnect()
                    if attempt:
                        raise


@dataclass(frozen=True)
class BatchTransferEntry:
    request_id: str
    slots: torch.Tensor
    is_last_chunk: bool
    job: LayeredSend


class BatchTransferPlan:
    """Immutable source-slot snapshots and per-request destination tickets."""

    def __init__(self, backend, entries):
        self.backend, self.entries = backend, tuple(entries)
        self.last_layer = -1

    def layer_ready(self, layer):
        if layer != self.last_layer + 1:
            raise TransferError("layer-ready notifications are not in model order")
        self.last_layer = layer
        size = self.backend.layers_per_chunk
        if (layer + 1) % size and layer + 1 != self.backend._layout.layers:
            return
        # One producer fence for the entire batch/layer group. No copies, RPCs,
        # native polling, or host waits are allowed on this callback's thread.
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.backend.runtime.device))
        self.backend._batch_tasks.put_nowait((layer // size, self, event))


class LayeredSend:
    def __init__(self, backend, ticket, pool, cancel_check):
        self.backend, self.ticket, self.pool = backend, ticket, pool
        self.cancel_check = cancel_check
        self.tasks = deque()  # exclusively consumed by the sender pump
        self.opened = threading.Event()
        self.done = threading.Event()
        self.sources_done = threading.Event()
        self.cancelled = threading.Event()
        self.error = None
        self.safe_sources = True
        self._sealed = False
        self._producer_done = False
        self._last_layer = -1
        self._slots = None
        self.submitted, self.completed = set(), set()
        self.expected = 0
        self.uncertain = False
        self.control = None
        self.terminal_started = False

    def set_slots(self, slots):
        self._slots = slots

    def layer_ready(self, layer):
        # Compatibility for callers with a one-request forward.
        self._last_layer = layer
        if layer == 0:
            self._plan = BatchTransferPlan(
                self.backend, [BatchTransferEntry(self.ticket.transfer_id, self._slots, True, self)]
            )
        self._plan.layer_ready(layer)

    def seal(self, *, cancel=False):
        if cancel:
            self.cancelled.set()
        # FIFO with batch tasks: a seal cannot overtake previously recorded groups.
        if not self._sealed:
            self._sealed = True
            self.backend._batch_tasks.put_nowait((None, self, None))

    def finish(self, *, cancel=False):
        self.seal(cancel=cancel)
        if not self.done.wait(3 * self.backend.timeout + 1):
            self.backend.runtime.retire("layer sender did not stop; retain GPU pools")
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


def sender_loop(backend):
    active, pending = deque(), []
    free = deque(backend._send_buffers)
    closing = False

    def open_job(job):
        def exchange(channel):
            channel.send(
                {
                    "op": "layer_open",
                    "backend": TRANSPORT,
                    "ticket": json.loads(job.ticket.to_wire()),
                    "source": backend.runtime.session,
                }
            )
            if channel.receive() != {"status": "ready", "locator": job.ticket.locator}:
                raise TransferError("layer receive reservation was revoked or rejected")

        return backend._session(job.ticket.endpoint).submit(exchange)

    def terminal(job):
        result = (
            "quarantined"
            if job.uncertain
            else ("cancelled" if job.cancelled.is_set() else "complete")
        )
        return backend._session(job.ticket.endpoint).submit(
            lambda ch: backend._send_terminal(job.ticket, job.submitted, job.completed, result, ch)
        )

    while not closing or active or pending:
        while True:
            try:
                job = backend._jobs.get_nowait()
            except queue.Empty:
                break
            if job is None:
                closing = True
                continue
            job.control = open_job(job)
            active.append(job)
        while True:
            try:
                group, plan, event = backend._batch_tasks.get_nowait()
            except queue.Empty:
                break
            if group is None:
                plan._producer_done = True
            else:
                for entry in plan.entries:
                    if entry.is_last_chunk:
                        entry.job.tasks.append((group, entry.slots, event))

        # Check every operation. A slow request or target cannot block another
        # request's receipt merely by occupying the head of an in-flight queue.
        for item in list(pending):
            job, buffer, operation, group, deadline = item
            try:
                status = backend.runtime.poll(operation, timeout_ms=0)
            except (RuntimeError, ValueError):
                status = "unknown"
            if status == "unknown" and time.monotonic() < deadline:
                continue
            pending.remove(item)
            if status == "unknown":
                backend.runtime.quarantine(buffer, "layer write completion is unknown")
                job.uncertain = True
                job.error = TransferError("layer write did not drain before its deadline")
                job.cancelled.set()
            else:
                backend.runtime.recycle_operation(operation)  # verifies ticket.drained
                free.append(buffer)
                if status == "completed":
                    job.completed.add(group)
                else:
                    job.uncertain = True
                    job.error = TransferError("layer write failed after local drain")
                    job.cancelled.set()

        competing = sum(not job.cancelled.is_set() and not job.terminal_started for job in active)
        quota = max(1, len(backend._send_buffers) // max(1, competing))
        last_packed = None
        for _ in range(len(active)):
            job = active.popleft()
            job._check_cancel()
            if not job.opened.is_set() and job.control.done():
                try:
                    job.control.result()
                except BaseException as exc:
                    job.error = exc
                    job.cancelled.set()
                job.opened.set()
            owned_slots = sum(item[0] is job for item in pending)
            if (
                job.opened.is_set()
                and job.tasks
                and ((free and owned_slots < quota) or job.cancelled.is_set())
            ):
                task = job.tasks.popleft()
                if not job.cancelled.is_set():
                    last_packed = job
                    buffer = free.popleft()
                    try:
                        group, _, _ = task
                        if group != job.expected:
                            raise TransferError("missing or repeated layer group")
                        job.expected += 1
                        job.safe_sources = False
                        backend.runtime.expose(buffer)  # a CUDA packing writer also owns this slot
                        nbytes = backend._pack(job, task, buffer)
                        job.safe_sources = True
                        backend.runtime.mark_safe(buffer)
                        if job.cancelled.is_set():
                            free.append(buffer)
                        else:
                            start, _ = job.ticket.groups[group]
                            remote = {
                                "session": job.ticket.session,
                                "address": job.ticket.address + job.ticket.layer_offsets[start],
                                "nbytes": nbytes,
                            }
                            # Even a failed submission may have exposed the slot.
                            operation = backend.runtime.submit_write(buffer, remote, nbytes=nbytes)
                            job.submitted.add(group)
                            pending.append(
                                (job, buffer, operation, group, time.monotonic() + backend.timeout)
                            )
                    except BaseException as exc:
                        job.error = exc
                        job.cancelled.set()
                        if buffer.safe_to_release and job.safe_sources:
                            free.append(buffer)
                        else:
                            job.uncertain = True
                            backend.runtime.quarantine(
                                buffer, "packing or native submission did not finish"
                            )
            if job._producer_done and not job.tasks and job.safe_sources:
                job.sources_done.set()
            if (
                job._producer_done
                and not job.tasks
                and job.opened.is_set()
                and not any(item[0] is job for item in pending)
            ):
                if not job.terminal_started:
                    if not job.cancelled.is_set() and job.expected != len(job.ticket.groups):
                        job.error = TransferError("prefill did not produce every layer group")
                        job.cancelled.set()
                    job.terminal_started = True
                    job.control = terminal(job)
                elif job.control.done():
                    try:
                        job.control.result()
                    except BaseException as exc:
                        job.error = exc
                    if job.uncertain or not job.safe_sources:
                        backend.runtime.retire("layer transfer could not prove safe completion")
                    job.done.set()
                    with backend._lock:
                        backend._send_jobs.pop(job.ticket.transfer_id, None)
                    continue
            active.append(job)
        # Resume after the last served request, and keep a per-request share of
        # the ring while peers compete. A slow target must not capture a peer's
        # freshly drained slot and prevent that peer's remaining layers finishing.
        if last_packed in active:
            active.rotate(-active.index(last_packed) - 1)
        # Native drain(0) is bounded; release the GIL between polling rounds.
        time.sleep(0.002)
