"""Persistent health-filtered Worker pools; RandomRouter remains a legacy API."""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from uuid import uuid4

from .client import RemoteCapacityError
from .control import endpoint_address
from .multiplex import MUX_VERSION, WorkerConnection
from .protocol import HANDOFF_VERSION

logger = logging.getLogger(__name__)


class NoHealthyWorker(RemoteCapacityError):
    pass


@dataclass(eq=False)
class WorkerNode:
    worker_id: str
    role: str
    endpoint: str
    healthy: bool = False
    health: dict = field(default_factory=dict)
    load_updated_at: float = 0.0
    reason: str | None = "not probed"
    connection: object = None
    connection_lock: threading.Lock = field(default_factory=threading.Lock)


class WorkerPool:
    def __init__(
        self,
        prefill_addrs,
        decode_addrs,
        *,
        timeout=30.0,
        health_interval=2.0,
        health_timeout=3.0,
        max_quarantined_bytes=0,
        max_requests=144,
        connection_factory=WorkerConnection,
    ):
        if any(not math.isfinite(v) or v <= 0 for v in (timeout, health_interval, health_timeout)):
            raise ValueError("pool timeouts and health interval must be positive and finite")
        if type(max_quarantined_bytes) is not int or max_quarantined_bytes < 0:
            raise ValueError("max_quarantined_bytes must be nonnegative")
        if not prefill_addrs or not decode_addrs:
            raise ValueError("both prefill and decode address lists are required")
        endpoints = [*prefill_addrs, *decode_addrs]
        addresses = [endpoint_address(endpoint) for endpoint in endpoints]
        if len(set(addresses)) != len(addresses):
            raise ValueError("Worker addresses must be distinct across both roles")
        self.prefill_pool = [
            WorkerNode(f"p-{i}", "prefill", addr) for i, addr in enumerate(prefill_addrs)
        ]
        self.decode_pool = [
            WorkerNode(f"d-{i}", "decode", addr) for i, addr in enumerate(decode_addrs)
        ]
        self.nodes = self.prefill_pool + self.decode_pool
        self.timeout, self.health_interval, self.health_timeout = (
            timeout,
            health_interval,
            health_timeout,
        )
        self.max_quarantined_bytes, self.max_requests = max_quarantined_bytes, max_requests
        self.factory = connection_factory
        self.lock = threading.Lock()
        self.lifecycle_lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.closed = False
        self.expected_identity = None
        self.probes = ThreadPoolExecutor(
            max_workers=min(16, len(self.nodes)), thread_name_prefix="pd-health"
        )

    def connect(self, node, *, for_probe=False):
        # Serialize initial connection establishment only for this destination.
        with node.connection_lock:
            if self.stop.is_set():
                raise ConnectionError("Worker pool is closed")
            if not for_probe and not node.healthy:
                raise ConnectionError("Worker is no longer healthy")
            if node.connection is None or node.connection.stopped.is_set():
                if node.connection is not None:
                    self.mark_unhealthy(
                        node, "Worker connection failed; awaiting a fresh health probe"
                    )
                    node.connection.close()
                    node.connection = None
                    if not for_probe:
                        raise ConnectionError(
                            "Worker connection failed; awaiting a fresh health probe"
                        )
                node.connection = self.factory(
                    node.endpoint,
                    timeout=self.timeout,
                    max_requests=self.max_requests,
                    connect_timeout=self.health_timeout,
                )
            return node.connection

    def healthy(self, role, exclude=()):
        excluded = set(exclude)
        with self.lock:
            if self.stop.is_set():
                return []
            return [
                node
                for node in self.nodes
                if node.role == role
                and node.healthy
                and node.worker_id not in excluded
                and (node.connection is None or not node.connection.stopped.is_set())
            ]

    def mark_unhealthy(self, node, reason):
        with self.lock:
            node.healthy = False
            node.reason = str(reason)

    def apply_health(self, node, health):
        transfer = health.get("transfer", {})
        if not isinstance(transfer, dict):
            self.mark_unhealthy(node, "invalid transfer health")
            return
        native = transfer.get("mooncake", {})
        if (
            not isinstance(native, dict)
            or type(native.get("quarantined_bytes", 0)) is not int
            or native.get("quarantined_bytes", 0) < 0
        ):
            self.mark_unhealthy(node, "invalid native health")
            return
        layered = native.get("layered", {})
        if not isinstance(layered, dict) or (
            health.get("layered_transfer") is True
            and (
                type(layered.get("layers_per_chunk")) is not int or layered["layers_per_chunk"] < 1
            )
        ):
            self.mark_unhealthy(node, "invalid layer transfer health")
            return
        valid = (
            health.get("status") in ("ready", "degraded")
            and health.get("role") == node.role
            and type(health.get("multiplex_version")) is int
            and health.get("multiplex_version") == MUX_VERSION
            and health.get("handoff_version") == HANDOFF_VERSION
            and health.get("decode_prefix_reuse") is True
            and type(health.get("layered_transfer")) is bool
            and isinstance(health.get("layout"), dict)
            and transfer.get("status", "ready") in ("ready", "degraded")
            and native.get("state", "ready") in ("ready", "degraded")
            and not native.get("reason")
            and native.get("quarantined_bytes", 0) <= self.max_quarantined_bytes
        )
        with self.lock:
            node.health = health
            node.load_updated_at = time.monotonic()
            node.healthy = bool(valid)
            node.reason = (
                None if valid else "Worker health, protocol, or quarantine budget rejected"
            )

    def probe(self, node):
        request = None
        try:
            connection = self.connect(node, for_probe=True)
            request = connection.open(str(uuid4()), self.stop)
            request.send({"op": "health"}, timeout=self.health_timeout)
            health = request.receive(timeout=self.health_timeout)
            if not isinstance(health, dict):
                raise ValueError("invalid Worker health response")
            self.apply_health(node, health)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.mark_unhealthy(node, exc)
        finally:
            if request is not None:
                request.close()

    def probe_all(self):
        futures = [self.probes.submit(self.probe, node) for node in self.nodes]
        for future in futures:
            future.result()
        healthy_p = self.healthy("prefill")

        def identity(health):
            group = (
                health.get("transfer", {})
                .get("mooncake", {})
                .get("layered", {})
                .get("layers_per_chunk")
            )
            return (
                health.get("layout"),
                health.get("layered_transfer"),
                health.get("handoff_version"),
                group if health.get("layered_transfer") else None,
            )

        with self.lock:
            # Pin model identity for this gateway's lifetime, including reconnects.
            if self.expected_identity is None and healthy_p:
                self.expected_identity = identity(healthy_p[0].health)
            if self.expected_identity is not None:
                for node in self.nodes:
                    if node.healthy and identity(node.health) != self.expected_identity:
                        node.healthy = False
                        node.reason = "P/D model layouts or transfer modes differ"

    def start(self, ready_timeout=30.0, *, require_all=False):
        if not math.isfinite(ready_timeout) or ready_timeout <= 0:
            raise ValueError("Worker readiness timeout must be positive")
        with self.lifecycle_lock:
            if self.closed:
                raise RuntimeError("Worker pool is closed")
            if self.thread is not None:
                return
            deadline = time.monotonic() + ready_timeout
            while not self.stop.is_set():
                self.probe_all()
                ready = (
                    all(node.healthy for node in self.nodes)
                    if require_all
                    else bool(self.healthy("prefill") and self.healthy("decode"))
                )
                if ready:
                    break
                if time.monotonic() >= deadline:
                    raise NoHealthyWorker(
                        "Worker pool readiness timed out: "
                        + "; ".join(
                            f"{node.worker_id} {node.endpoint}: {node.reason}"
                            for node in self.nodes
                            if not node.healthy
                        )
                    )
                self.stop.wait(min(self.health_interval, max(0, deadline - time.monotonic())))
            if self.stop.is_set():
                raise ConnectionError("Worker pool closed during startup")
            self.thread = threading.Thread(
                target=self._monitor, name="pd-health-monitor", daemon=True
            )
            self.thread.start()

    def _monitor(self):
        while not self.stop.wait(self.health_interval):
            try:
                self.probe_all()
            except Exception as exc:
                if self.stop.is_set():
                    return
                for node in self.nodes:
                    self.mark_unhealthy(node, exc)
                logger.exception("Worker health round failed")

    def close(self):
        self.stop.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=self.timeout + self.health_timeout + 1)
        self.probes.shutdown(wait=True, cancel_futures=True)
        for node in self.nodes:
            with node.connection_lock:
                if node.connection is not None:
                    node.connection.close()
        self.closed = True


class RandomRouter:
    def __init__(self, pool, seed=None):
        self.pool, self.random = pool, random.Random(seed)
        self.lock = threading.Lock()

    def _select(self, role, exclude):
        candidates = self.pool.healthy(role, exclude)
        if not candidates:
            raise NoHealthyWorker(f"no healthy {role} Worker is available")
        with self.lock:
            return self.random.choice(candidates)

    def select_prefill(self, exclude=()):
        return self._select("prefill", exclude)

    def select_decode(self, exclude=()):
        return self._select("decode", exclude)

    def record(self, request_id, prefill, decode, attempt, event="route"):
        logger.info(
            "pd_route %s",
            json.dumps(
                {
                    "event": event,
                    "request_id": request_id,
                    "prefill_id": prefill.worker_id,
                    "decode_id": decode.worker_id,
                    "attempt": attempt,
                },
                separators=(",", ":"),
            ),
        )
