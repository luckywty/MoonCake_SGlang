"""Bind admitted P/D Workers before Prefill, retaining the layered data plane."""

import math
import threading
from uuid import uuid4

from .client import (
    PDClient,
    RemoteCapacityError,
    RemoteUnavailableError,
    reservation_prefix,
    response,
    stream_reserved,
)
from .control import StartRequest
from .prefix_hash import PreparedPrefix, prepare_prefix
from .protocol import worker_transfer_mode
from .router import NoHealthyWorker, WorkerPool
from .routing_policy import PolicyRouter


class RoutedPDClient(PDClient):
    def __init__(
        self,
        prefill_addrs,
        decode_addrs,
        *,
        routing_seed=None,
        timeout=120.0,
        max_inflight_requests=128,
        max_inflight_tokens=1048576,
        health_interval=2.0,
        health_timeout=3.0,
        ready_timeout=30.0,
        max_quarantined_bytes=0,
        pool=None,
        routing_config=None,
    ):
        if not prefill_addrs or not decode_addrs:
            raise ValueError("nonempty P and D address lists are required")
        if not math.isfinite(ready_timeout) or ready_timeout <= 0:
            raise ValueError("Worker readiness timeout must be positive and finite")
        super().__init__(
            prefill_addrs[0],
            decode_addrs[0],
            timeout=timeout,
            max_inflight_requests=max_inflight_requests,
            max_inflight_tokens=max_inflight_tokens,
        )
        self.pool = pool or WorkerPool(
            prefill_addrs,
            decode_addrs,
            timeout=timeout,
            health_interval=health_interval,
            health_timeout=health_timeout,
            max_quarantined_bytes=max_quarantined_bytes,
            max_requests=max_inflight_requests + 16,
        )
        self.router = PolicyRouter(self.pool, routing_seed, routing_config)
        if hasattr(self.pool, "configure_prefix_index"):
            self.pool.configure_prefix_index(self.router.config)
        self.ready_timeout = ready_timeout
        self.active_lock = threading.Lock()
        self.active_routes = {}
        self.pull_plans = {}

    def start(self):
        self.pool.start(self.ready_timeout)

    def close(self):
        super().close()
        self.pool.close()

    def _cache_rpc(self, endpoint, message):
        node = next(n for n in self.pool.nodes if n.endpoint == endpoint)
        channel = self.pool.connect(node).open(str(uuid4()))
        try:
            channel.send(message, timeout=0.2)
            return channel.receive(timeout=0.2)
        finally:
            channel.close()

    def prefix_spec(self):
        if not getattr(self.pool, "prefix_enabled", False):
            return None
        nodes = self.pool.healthy("prefill")
        tier_index = getattr(self.pool, "tier_index", None)
        return self.pool.prefix_index.spec_for(nodes) or (
            tier_index.spec_for(nodes) if tier_index is not None else None
        )

    def _prefix_matches(self, request, bucket, cancel, prepared_prefix=None):
        if (bucket.prefill.policy or self.router.config.policy) != "cache_aware":
            return {}

        if cancel is not None and cancel.is_set():
            raise InterruptedError("request cancelled")
        if prepared_prefix is None or not hasattr(self.pool, "prefix_index"):
            return {}  # Cold/incompatible directories use the ordinary load policy.
        nodes = self.router.candidates("prefill", request, bucket)
        result = self.pool.prefix_index.hits(nodes, prepared_prefix)
        if hasattr(self.pool, "tier_index"):
            config = self.router.config
            try:
                tier_hits = self.pool.tier_index.hits(
                    nodes,
                    self.pool.healthy("decode"),
                    prepared_prefix,
                    min_remote_tokens=config.min_remote_pull_tokens,
                    h2d_gbps=config.dram_h2d_gbps,
                    rdma_gbps=config.dram_rdma_gbps,
                    prefill_ms_per_token=config.dram_prefill_ms_per_token,
                    max_remote_bytes=config.dram_max_pull_bytes,
                )
                result.update(tier_hits)
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                pass  # An optional directory cannot fail an inference request.
        return result

    def _route_and_reserve(self, request, cancel, prepared_prefix=None):
        last_error = None
        attempted_decode = set()
        for bucket in self.router.buckets_for(request):
            prefill = None
            try:
                hits = self._prefix_matches(request, bucket, cancel, prepared_prefix)
                prefill = self.router.acquire_prefill(
                    request=request, bucket=bucket, prefix_hits=hits
                )
                decode, channel, ready = self._reserve(
                    request, prefill, cancel, bucket=bucket, excluded=attempted_decode
                )
                with self.active_lock:
                    self.pull_plans[request.transfer_id] = getattr(
                        hits.get(prefill.worker_id), "remote_plan", None
                    )
                return prefill, decode, channel, ready, bucket
            except InterruptedError:
                if prefill is not None:
                    self.router.release(prefill)
                raise
            except (RemoteCapacityError, RemoteUnavailableError, OSError) as exc:
                if prefill is not None:
                    self.router.release(prefill)
                last_error = exc
            except BaseException:
                if prefill is not None:
                    self.router.release(prefill)
                raise
        raise last_error or NoHealthyWorker("no routing bucket can supply a P/D pair")

    def _reserve(self, request, prefill, cancel, *, bucket, excluded):
        last_error = None
        layered = prefill.health["layered_transfer"]
        for _ in range(len(self.pool.decode_pool)):
            try:
                node = self.router.acquire_decode(
                    excluded, request=request, bucket=bucket, prefill=prefill
                )
            except NoHealthyWorker:
                if last_error is not None:
                    raise last_error
                raise
            attempt = len(excluded)
            excluded.add(node.worker_id)
            channel = None
            try:
                if node.health["layered_transfer"] != layered or worker_transfer_mode(
                    node.health
                ) != worker_transfer_mode(prefill.health):
                    raise RemoteUnavailableError("P/D transfer modes differ")
                if worker_transfer_mode(prefill.health) == "shm_dram" and not self.router._shm_pair(
                    prefill, node
                ):
                    raise RemoteUnavailableError("shm_dram requires the same host and shared pool")
                channel = self.pool.connect(node).open(request.transfer_id, cancel)
                message = {"op": "reserve", "request": request.to_object()}
                if node.health.get("dram_cache", {}).get("enabled", False):
                    message["cache_transaction"] = True
                channel.send(message)
                ready = response(channel, "ready" if layered else "reserved")
                reservation_prefix(
                    ready,
                    request,
                    layered=layered,
                    transfer_mode=worker_transfer_mode(prefill.health) if layered else "gpu_direct",
                )
                self.router.record(request.transfer_id, prefill, node, attempt, bucket=bucket)
                return node, channel, ready
            except InterruptedError:
                self.router.release(node)
                if channel is not None:
                    channel.close()
                raise
            except (RemoteCapacityError, RemoteUnavailableError, OSError) as exc:
                self.router.release(node)
                last_error = exc
                if not isinstance(exc, RemoteCapacityError):
                    self.pool.mark_unhealthy(node, exc)
                self.router.record(
                    request.transfer_id, prefill, node, attempt, "reserve_retry", bucket=bucket
                )
                if channel is not None:
                    channel.close()
            except BaseException:
                self.router.release(node)
                if channel is not None:
                    channel.close()
                raise
        raise last_error or NoHealthyWorker("no decode reservation succeeded")

    def _stream(
        self,
        prompt_ids,
        max_tokens,
        *,
        ignore_eos=False,
        cancel=None,
        request_id=None,
        prepared_prefix=None,
    ):
        request_id = request_id or str(uuid4())
        with self.active_lock:
            if request_id in self.active_routes:
                raise ValueError("request_id is already active in this router")
            self.active_routes[request_id] = None
        try:
            yield from self._stream_bound(
                prompt_ids,
                max_tokens,
                ignore_eos=ignore_eos,
                cancel=cancel,
                request_id=request_id,
                prepared_prefix=prepared_prefix,
            )
        finally:
            with self.active_lock:
                self.active_routes.pop(request_id, None)
                self.pull_plans.pop(request_id, None)

    def _stream_bound(
        self,
        prompt_ids,
        max_tokens,
        *,
        ignore_eos=False,
        cancel=None,
        request_id=None,
        prepared_prefix=None,
    ):
        self.start()
        request = StartRequest(
            request_id or str(uuid4()), 0, tuple(prompt_ids), max_tokens, ignore_eos
        )
        if prepared_prefix is not None and (
            not isinstance(prepared_prefix, PreparedPrefix)
            or prepared_prefix.token_ids != request.prompt_ids
        ):
            raise ValueError("prepared prefix does not belong to these tokens")
        spec = self.prefix_spec()
        if spec is not None and (
            prepared_prefix is None or prepared_prefix.domain_id != spec.domain_id
        ):
            # CLI/Token-ID callers run here on their ordinary blocking client thread.
            prepared_prefix = prepare_prefix(request.prompt_ids, spec)
        if cancel is not None and cancel.is_set():
            raise InterruptedError("request cancelled")
        p = d = None
        held_prefill = held_decode = None
        cache_endpoints = []
        completed = False
        try:
            prefill, decode, d, ready, bucket = self._route_and_reserve(
                request, cancel, prepared_prefix
            )
            held_prefill, held_decode = prefill, decode
            cache_prefill = prefill.health.get("dram_cache", {}).get("enabled", False)
            if cache_prefill:
                cache_endpoints.append(prefill.endpoint)
            if decode.health.get("dram_cache", {}).get("enabled", False):
                cache_endpoints.append(decode.endpoint)
            with self.active_lock:
                self.active_routes[request.transfer_id] = (prefill.worker_id, decode.worker_id)
            p = self.pool.connect(prefill).open(request.transfer_id, cancel)
            layered = prefill.health["layered_transfer"]

            def release_prefill():
                nonlocal held_prefill
                self.router.release(held_prefill)
                held_prefill = None

            yield from stream_reserved(
                request,
                p,
                d,
                ready,
                layered=layered,
                transfer_mode=worker_transfer_mode(prefill.health) if layered else "gpu_direct",
                on_prefill_done=release_prefill,
                cache_transaction=cache_prefill,
                remote_pull_plan=self.pull_plans.get(request.transfer_id),
            )
            completed = True
        finally:
            self._cache_finalize(
                cache_endpoints,
                request.transfer_id,
                completed and (cancel is None or not cancel.is_set()),
            )
            try:
                if p is not None:
                    p.close()
            finally:
                try:
                    if d is not None:
                        d.close()
                finally:
                    self.router.release(held_prefill)
                    self.router.release(held_decode)
