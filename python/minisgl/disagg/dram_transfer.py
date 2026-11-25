"""Optional host-only RDMA reads, isolated from the essential GPU P/D engine."""

from __future__ import annotations

import importlib
import logging
import queue
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from .control import Channel, endpoint_address, endpoint_name
from .dram_protocol import MAX_DRAM_PULL_PAGES, digest_bytes

logger = logging.getLogger(__name__)
_LIVE_HOST_ENGINES = {}


class RemotePullJob:
    """Bounded handoff from host RDMA progress to the GPU scheduler owner."""

    def __init__(self, slots=2, *, parent=None):
        self.chunks = queue.Queue(slots)
        self.cancelled = threading.Event()
        self.parent = parent

    def take(self):
        try:
            return self.chunks.get_nowait()
        except queue.Empty:
            return None

    def publish(self, pages, deadline):
        while not self.cancelled.is_set() and time.monotonic() < deadline:
            try:
                self.chunks.put(pages, timeout=0.005)
                return True
            except queue.Full:
                continue
        return False


class DramTransfer:
    def __init__(self, service, config, *, module=None):
        if module is None:
            from .mooncake_runtime import MooncakeRuntime

            # A QP fence cannot revoke TCP/SHM readers. Enforce the same
            # classic, one-sided, HCA-only transport as the essential TE.
            MooncakeRuntime._configure_rdma(config.device_name)
        self.service, self.pool, self.config = service, service.dram_pool, config
        self.ready = False
        self.reason = None
        self.lock = threading.RLock()
        self.epoch = str(uuid4())
        self.engine = None
        self.operations = []
        self.fence_future = None
        self.fence_attempted = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dram-fence")
        self.read_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dram-read")
        self.read_slots = threading.BoundedSemaphore(2)
        self.credit, self.credit_at = config.remote_max_bytes, time.monotonic()
        module = module or importlib.import_module("mooncake.engine")
        kind = module.TransferEngine
        if any(
            not callable(getattr(kind, method, None))
            for method in (
                "initialize",
                "initialize_with_rpc_port",
                "register_memory",
                "submit_scatter_read",
                "fence_remote_readers",
            )
        ):
            self.executor.shutdown(wait=False)
            self.read_executor.shutdown(wait=False)
            raise RuntimeError(
                "remote DRAM disabled: Mooncake needs submit_scatter_read and a checked fence_remote_readers API"
            )
        self.module = module
        if config.host in ("0.0.0.0", "::"):
            raise ValueError("DRAM TE requires a reachable advertised host")
        # A distinct TE instance owns the host MR. Never fence the GPU P/D TE.
        self.engine = kind()
        _LIVE_HOST_ENGINES[id(self)] = self
        if config.port is None:
            raise ValueError("DRAM TE requires its dedicated RPC port")
        if (
            self.engine.initialize_with_rpc_port(
                config.host, "P2PHANDSHAKE", "rdma", config.device_name, config.port
            )
            != 0
        ):
            raise RuntimeError("DRAM TE initialization failed")
        port = self.engine.get_rpc_port()
        if config.port is not None and port != config.port:
            raise RuntimeError("DRAM TE ignored its dedicated RPC port")
        self.session = endpoint_name(config.host, port)
        if (
            not callable(getattr(self.engine, "has_remote_reader_fence", None))
            or not self.engine.has_remote_reader_fence()
        ):
            raise RuntimeError(
                "remote DRAM requires a classic RDMA engine with checked reader revocation"
            )
        if not self.pool.buffer.is_pinned() or self.pool.buffer.device.type != "cpu":
            raise ValueError("RDMA host cache requires the startup pinned CPU allocation")
        if (
            self.engine.register_memory(
                self.pool.buffer.data_ptr(), self.pool.registered_bytes, "*", not service.is_prefill
            )
            != 0
        ):
            raise RuntimeError("DRAM MR registration failed; retain the allocation")
        self.ready = True

    def _spend(self, size):
        now = time.monotonic()
        self.credit = min(
            self.config.remote_max_bytes,
            self.credit + (now - self.credit_at) * self.config.remote_bytes_per_second,
        )
        self.credit_at = now
        if size > self.credit:
            return False
        self.credit -= size
        return True

    def source_healthy(self):
        try:
            health = getattr(self.service.backend, "health", lambda: {})()
            native = health.get("mooncake", {})
            return (
                health.get("status", "ready") in ("ready", "degraded")
                and native.get("state", "ready") in ("ready", "degraded")
                and not native.get("reason")
            )
        except Exception:
            return False

    def rpc(self, message):
        if message.get("instance_id") != self.service.instance_id:
            return {"status": "cache_miss"}
        peer = message.get("peer")
        if not isinstance(peer, str) or not 1 <= len(peer) <= 128:
            return {"status": "cache_miss"}
        if message["op"] == "dram_unpin":
            if message.get("success") is True:
                with self.pool.lock:
                    lease = self.pool.leases.get(message.get("lease_id"))
                    if lease is not None and lease.peer == peer:
                        for page in lease.pages:
                            self.pool.touch(page)
            released = self.pool.unpin_dram(
                message.get("lease_id"), peer, drained=message.get("drained") is True
            )
            return {"status": "dram_unpinned" if released else "cache_miss"}
        with self.lock:
            if (
                not self.ready
                or self.service.is_prefill
                or not self.source_healthy()
                or message.get("domain_id") != self.service.tier_publisher.spec.domain_id
            ):
                return {"status": "cache_miss"}
            values = message.get("page_hashes")
            if (
                not isinstance(values, list)
                or not values
                or len(values) > MAX_DRAM_PULL_PAGES
                or len(values) * self.service.s.layout.page_size < self.config.min_remote_tokens
                or len(values) * self.pool.page_bytes > self.config.remote_max_bytes
                or len(self.pool.leases) >= 2
            ):
                return {"status": "cache_miss"}
            hashes = tuple(digest_bytes(h) for h in values)
            if not self._spend(len(hashes) * self.pool.page_bytes):
                return {"status": "cache_miss"}
            lease = self.pool.pin_dram(hashes, peer)
            if lease is None:
                return {"status": "cache_miss"}
            if len(lease.pages) * self.service.s.layout.page_size < self.config.min_remote_tokens:
                self.pool.unpin_dram(lease.id, peer, drained=True)
                return {"status": "cache_miss"}
            return {
                "status": "dram_pinned",
                "instance_id": self.service.instance_id,
                "domain_id": self.service.tier_publisher.spec.domain_id,
                "session": self.session,
                "epoch": self.epoch,
                "memory": "host_pinned",
                "lease_id": lease.id,
                "expires_in_ms": int(self.config.lease_ttl * 1000),
                "pages": [
                    [
                        p.digest.hex(),
                        p.parent.hex(),
                        p.depth,
                        p.generation,
                        self.pool.buffer.data_ptr() + p.slot * self.pool.stride,
                        self.pool.page_bytes,
                    ]
                    for p in lease.pages
                ],
            }

    def _rpc(self, endpoint, message):
        with socket.create_connection(
            endpoint_address(endpoint), timeout=self.config.remote_timeout
        ) as sock:
            channel = Channel(sock, self.config.remote_timeout)
            channel.send(message)
            return channel.receive()

    def _receipt(self, plan, lease, peer, *, success=False):
        try:
            self._rpc(
                plan.source_endpoint,
                {
                    "op": "dram_unpin",
                    "instance_id": plan.source_instance,
                    "lease_id": lease["lease_id"],
                    "peer": peer,
                    "drained": True,
                    "success": success,
                },
            )
        except Exception:
            pass  # Source TTL performs checked revocation if the receipt is lost.

    def start_pull(self, transaction, plan, start, hashes, job):
        if not self.ready or not self.read_slots.acquire(blocking=False):
            return None
        try:
            return self.read_executor.submit(
                self.pull, transaction, plan, start, hashes, job, slot_owned=True
            )
        except RuntimeError:
            self.read_slots.release()
            return None

    def pull(self, transaction, plan, start, hashes, job, *, slot_owned=False):
        if not slot_owned and (not self.ready or not self.read_slots.acquire(blocking=False)):
            return False
        lease = None
        pages = []
        ticket = None
        drained, retained, read_complete = True, False, False
        peer = self.service.instance_id + "/" + self.epoch
        try:
            if not self.ready or job.cancelled.is_set():
                return False
            with self.lock:
                if not self._spend(len(hashes) * self.pool.page_bytes):
                    return False
            lease = self._rpc(
                plan.source_endpoint,
                {
                    "op": "dram_pin",
                    "instance_id": plan.source_instance,
                    "domain_id": plan.domain_id,
                    "peer": peer,
                    "page_hashes": [h.hex() for h in hashes],
                },
            )
            if (
                lease.get("status") != "dram_pinned"
                or lease.get("instance_id") != plan.source_instance
                or lease.get("domain_id") != plan.domain_id
                or lease.get("memory") != "host_pinned"
            ):
                return False
            records = lease.get("pages")
            if (
                not isinstance(records, list)
                or len(records) > len(hashes)
                or len(records) * self.service.s.layout.page_size < self.config.min_remote_tokens
            ):
                return False
            addresses, parents = [], []
            for i, record in enumerate(records):
                if (
                    not isinstance(record, list)
                    or len(record) != 6
                    or digest_bytes(record[0]) != hashes[i]
                    or record[2] != start + i + 1
                    or type(record[3]) is not int
                    or record[3] < 1
                    or type(record[4]) is not int
                    or record[4] <= 0
                    or record[5] != self.pool.page_bytes
                ):
                    raise ValueError("invalid remote DRAM page descriptor")
                parent = digest_bytes(record[1])
                if i and parent != hashes[i - 1]:
                    raise ValueError("remote DRAM ancestry mismatch")
                if i == 0 and job.parent is not None and parent != job.parent:
                    raise ValueError("remote DRAM seam ancestry mismatch")
                addresses.append(record[4])
                parents.append(parent)
            until = time.monotonic() + self.config.remote_timeout
            for offset in range(0, len(records), self.config.batch_pages):
                if job.cancelled.is_set() or time.monotonic() >= until:
                    return False
                end = min(len(records), offset + self.config.batch_pages)
                for i in range(offset, end):
                    page = self.pool.reserve(hashes[i], parents[i], start + i + 1, transaction)
                    if page is None:
                        return False
                    with self.pool.lock:
                        page.readers += 1
                    pages.append(page)
                local = [self.pool.buffer.data_ptr() + p.slot * self.pool.stride for p in pages]
                sizes = [self.pool.page_bytes] * len(pages)
                drained = False  # Failed submission can already have exposed memory.
                ticket = self.engine.submit_scatter_read(
                    lease["session"],
                    local,
                    sizes,
                    addresses[offset:end],
                    sizes,
                    [[0] for _ in pages],
                    [[0] for _ in pages],
                    [[self.pool.page_bytes] for _ in pages],
                )
                while time.monotonic() < until:
                    status = ticket.drain(0)
                    if ticket.drained:
                        drained = True
                        if status != self.module.ScatterTransferCompletionStatus.COMPLETED:
                            return False
                        for page in pages:
                            self.pool.written(page)
                        if not job.publish(tuple(pages), until):
                            return False
                        pages = []  # H2D owns these host readers while the next read starts.
                        break
                    time.sleep(0.001)
                if not drained:
                    return False
            read_complete = True
            return len(records) == len(hashes)
        except Exception as exc:
            self.reason = str(exc)
            return False
        finally:
            if not drained:
                with self.pool.lock:
                    self.pool.quarantined_slots.update(p.slot for p in pages)
                with self.lock:
                    self.operations.append((ticket, tuple(pages), lease, plan, peer))
                retained = True
                pages = []  # Reaper retains all destination allocations/tickets.
            for page in pages:
                if page.writing:
                    self.pool.written(page, success=False)
            if pages:
                self.pool.release(pages)
            if drained and lease is not None and lease.get("status") == "dram_pinned":
                self._receipt(plan, lease, peer, success=read_complete)
            if not retained:
                self.read_slots.release()

    def request_fence(self, peers):
        with self.lock:
            if self.fence_attempted:
                return
            self.ready = False
            self.fence_attempted = True
            self.fence_future = self.executor.submit(self.engine.fence_remote_readers)

    def progress(self):
        if self.ready and not self.service.is_prefill and not self.source_healthy():
            self.request_fence(set())
        with self.lock:
            future = self.fence_future
            if future is not None and future.done():
                try:
                    result = future.result()
                    if result != 0:
                        raise RuntimeError("native host-reader fence did not prove completion")
                    with self.pool.lock:
                        self.pool.fence_complete({l.peer for l in self.pool.leases.values()})
                    self.reason = "host export engine fenced after lease expiry; remote cache disabled until restart"
                except Exception as exc:
                    self.reason = str(exc)
                self.fence_future = None
            for operation in list(self.operations):
                ticket, pages, lease, plan, peer = operation
                if ticket is None:
                    continue
                try:
                    ticket.drain(0)
                    if not ticket.drained:
                        continue
                    with self.pool.lock:
                        self.pool.quarantined_slots.difference_update(p.slot for p in pages)
                    for page in pages:
                        if page.writing:
                            self.pool.written(page, success=False)
                    self.pool.release(pages)
                    self.operations.remove(operation)
                    self.read_slots.release()
                    if lease is not None:
                        self.executor.submit(self._receipt, plan, lease, peer)
                except Exception:
                    continue

    def close(self):
        self.ready = False
        with self.pool.lock:
            peers = {l.peer for l in self.pool.leases.values()}
        if peers and self.fence_future is None:
            self.request_fence(peers)
        deadline = time.monotonic() + min(self.config.remote_timeout, 0.5)
        while True:
            self.progress()
            if (not self.operations and self.fence_future is None) or time.monotonic() >= deadline:
                break
            time.sleep(0.001)
        self.executor.shutdown(wait=False, cancel_futures=False)
        self.read_executor.shutdown(wait=False, cancel_futures=False)
        if not self.operations and not self.pool.leases and self.fence_future is None:
            # Startup registrations remain alive until this dedicated engine closes.
            self.engine = None
            _LIVE_HOST_ENGINES.pop(id(self), None)
