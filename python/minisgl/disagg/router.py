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
from .dram_protocol import minimum_pull_bytes, remote_pull_limits
from .multiplex import MUX_VERSION, WorkerConnection
from .prefix_index import PrefixIndex
from .protocol import HANDOFF_VERSION, LAYERED_TRANSFER_MODES, KVLayout, worker_transfer_mode
from .tier_index import TierIndex

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
    probe_lock: threading.Lock = field(default_factory=threading.Lock)
    prefix_sync_lock: threading.Lock = field(default_factory=threading.Lock)
    instance_id: str | None = None
    dram_warning: tuple | None = None


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
        self.prefix_index = PrefixIndex()
        self.tier_index = TierIndex()
        self.prefix_enabled = False
        self.routing_config = None
        self.prefix_sync_interval = 0.2
        self.prefix_thread = None
        self.prefix_probes = ThreadPoolExecutor(
            max_workers=min(8, len(self.prefill_pool)), thread_name_prefix="pd-prefix"
        )
        self.probes = ThreadPoolExecutor(
            max_workers=min(16, len(self.nodes)), thread_name_prefix="pd-health"
        )

    def configure_prefix_index(self, config):
        if self.thread is not None:
            raise RuntimeError("prefix index must be configured before WorkerPool.start")
        self.routing_config = config
        self.prefix_enabled = config.policy == "cache_aware" or any(
            bucket.prefill.policy == "cache_aware" for bucket in config.buckets
        )
        self.prefix_sync_interval = config.prefix_sync_interval
        self.prefix_index = PrefixIndex(
            config.prefix_index_max_entries,
            config.prefix_index_max_worker_entries,
            config.prefix_index_max_age,
        )
        self.tier_index = TierIndex(
            config.prefix_index_max_entries,
            config.prefix_index_max_worker_entries,
            config.prefix_index_max_age,
        )

    def _sync_tier(self, node):
        for _ in range(16):
            if self.stop.is_set() or not node.healthy:
                return
            task = self.tier_index.request(node.worker_id)
            if task is None:
                return
            state, message = task
            channel = None
            try:
                channel = self.connect(node).open(str(uuid4()), self.stop)
                channel.send(message, timeout=self.health_timeout)
                reply = channel.receive(timeout=self.health_timeout)
                if not self.tier_index.apply(node.worker_id, state, reply):
                    return
            except (OSError, RuntimeError, TypeError, ValueError):
                self.tier_index.failed(node.worker_id, state)
                return
            finally:
                if channel is not None:
                    channel.close()

    def _sync_prefix(self, node):
        with node.prefix_sync_lock:
            # Bound each round so one large snapshot cannot monopolize the pool.
            for _ in range(16):
                if self.stop.is_set() or not node.healthy:
                    return
                task = self.prefix_index.request(node.worker_id)
                if task is None:
                    return
                state, message = task
                channel = None
                try:
                    connection = self.connect(node)
                    channel = connection.open(str(uuid4()), self.stop)
                    channel.send(message, timeout=self.health_timeout)
                    reply = channel.receive(timeout=self.health_timeout)
                    if not isinstance(reply, dict):
                        raise ValueError("invalid prefix synchronization response")
                    if not self.prefix_index.apply(node.worker_id, state, reply):
                        return
                except (OSError, RuntimeError, TypeError, ValueError):
                    # This optional directory does not retire an inference Worker.
                    self.prefix_index.failed(node.worker_id, state)
                    return
                finally:
                    if channel is not None:
                        channel.close()

    def _prefix_monitor(self):
        while not self.stop.is_set():
            try:
                futures = [
                    self.prefix_probes.submit(self._sync_prefix, node)
                    for node in self.healthy("prefill")
                ]
                futures.extend(
                    self.prefix_probes.submit(self._sync_tier, node)
                    for role in ("prefill", "decode")
                    for node in self.healthy(role)
                )
                for future in futures:
                    future.result()
            except Exception:
                if not self.stop.is_set():
                    logger.exception("Prefix directory synchronization failed")
            if self.stop.wait(self.prefix_sync_interval):
                return

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
            self.prefix_index.forget(node.worker_id)
            self.tier_index.forget(node.worker_id)

    def _dram_config_warnings(self, health):
        if not self.prefix_enabled or self.routing_config is None:
            return ()
        dram = health.get("dram_cache", {})
        if not isinstance(dram, dict) or not (
            dram.get("remote_enabled") is True or dram.get("remote_ready") is True
        ):
            return ()
        if dram.get("remote_ready") is not True:
            return (f"remote DRAM requested but unavailable: {dram.get('error') or 'not ready'}",)
        limits = remote_pull_limits(dram)
        if limits is None:
            return (
                "missing or invalid DRAM pull limits in health; remote hints disabled; upgrade Worker",
            )
        minimum, maximum = limits
        config = self.routing_config
        warnings = []
        if minimum != config.min_remote_pull_tokens or maximum != config.dram_max_pull_bytes:
            warnings.append(
                "DRAM pull limits differ: "
                f"Router --min-remote-pull-tokens={config.min_remote_pull_tokens}, "
                f"--dram-max-pull-bytes={config.dram_max_pull_bytes}; "
                f"Worker --pd-dram-min-remote-tokens={minimum}, "
                f"--pd-dram-max-pull-bytes={maximum}; plans use the P/D/Router intersection"
            )
        try:
            layout = KVLayout(**health["layout"])
            tokens = max(minimum, config.min_remote_pull_tokens)
            budget = min(maximum, config.dram_max_pull_bytes)
            required = minimum_pull_bytes(layout.bytes_per_token, layout.page_size, tokens)
            if required > budget:
                warnings.append(
                    f"remote DRAM has no feasible pull: {tokens} tokens require {required} "
                    f"page-aligned bytes, exceeding the effective {budget}-byte limit; "
                    "increase the byte limits or lower the token thresholds"
                )
        except (KeyError, TypeError, ValueError):
            warnings.append("cannot validate remote DRAM byte budget: invalid Worker KV layout")
        return tuple(warnings)

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
            and worker_transfer_mode(health)
            in (LAYERED_TRANSFER_MODES if health.get("layered_transfer") else ("tcp",))
            and isinstance(health.get("layout"), dict)
            and transfer.get("status", "ready") in ("ready", "degraded")
            and native.get("state", "ready") in ("ready", "degraded")
            and not native.get("reason")
            and native.get("quarantined_bytes", 0) <= self.max_quarantined_bytes
        )
        if worker_transfer_mode(health) == "shm_dram":
            valid = bool(
                valid
                and type(health.get("numa_id")) is int
                and health["numa_id"] >= 0
                and isinstance(health.get("host_id"), str)
                and health["host_id"]
                and isinstance(health.get("shm_pool"), str)
                and health["shm_pool"]
            )
        warnings = self._dram_config_warnings(health) if valid else ()
        with self.lock:
            instance = health.get("instance_id")
            if instance != node.instance_id:
                self.prefix_index.forget(node.worker_id)
                self.tier_index.forget(node.worker_id)
                node.instance_id = instance
            node.health = health
            node.load_updated_at = time.monotonic()
            node.healthy = bool(valid)
            node.reason = (
                None if valid else "Worker health, protocol, or quarantine budget rejected"
            )
            signature = (instance, warnings)
            changed = signature != node.dram_warning
            node.dram_warning = signature
            if valid and self.prefix_enabled:
                self.prefix_index.observe(node.worker_id, health)
                self.tier_index.observe(node.worker_id, health)
            else:
                self.prefix_index.forget(node.worker_id)
                self.tier_index.forget(node.worker_id)
        if changed:
            for message in warnings:
                logger.warning("Worker %s (%s): %s", node.worker_id, node.endpoint, message)

    def probe(self, node):
        # A late probe cannot roll a newly discovered instance back to an old one.
        with node.probe_lock:
            self._probe(node)

    def _probe(self, node):
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
                health.get("cache_namespace", "default"),
                health.get("layered_transfer"),
                worker_transfer_mode(health),
                health.get("handoff_version"),
                group if health.get("layered_transfer") else None,
                (
                    (health.get("host_id"), health.get("shm_pool"))
                    if worker_transfer_mode(health) == "shm_dram"
                    else None
                ),
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
                        self.prefix_index.forget(node.worker_id)
                        self.tier_index.forget(node.worker_id)

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
            if self.prefix_enabled:
                self.prefix_thread = threading.Thread(
                    target=self._prefix_monitor, name="pd-prefix-monitor", daemon=True
                )
                self.prefix_thread.start()

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
        if self.prefix_thread is not None and self.prefix_thread is not threading.current_thread():
            self.prefix_thread.join(timeout=self.health_timeout + 1)
        self.prefix_probes.shutdown(wait=True, cancel_futures=True)
        self.probes.shutdown(wait=True, cancel_futures=True)
        for node in self.nodes:
            with node.connection_lock:
                if node.connection is not None:
                    node.connection.close()
            self.prefix_index.forget(node.worker_id)
            self.tier_index.forget(node.worker_id)
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
