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

    def start(self):
        self.pool.start(self.ready_timeout)

    def close(self):
        self.pool.close()

    def prefix_spec(self):
        if not getattr(self.pool, "prefix_enabled", False):
            return None
        return self.pool.prefix_index.spec_for(self.pool.healthy("prefill"))

    def _prefix_matches(self, request, bucket, cancel, prepared_prefix=None):
        if (bucket.prefill.policy or self.router.config.policy) != "cache_aware":
            return {}

        if cancel is not None and cancel.is_set():
            raise InterruptedError("request cancelled")
        if prepared_prefix is None or not hasattr(self.pool, "prefix_index"):
            return {}  # Cold/incompatible directories use the ordinary load policy.
        nodes = self.router.candidates("prefill", request, bucket)
        return self.pool.prefix_index.hits(nodes, prepared_prefix)

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
                node = self.router.acquire_decode(excluded, request=request, bucket=bucket)
            except NoHealthyWorker:
                if last_error is not None:
                    raise last_error
                raise
            attempt = len(excluded)
            excluded.add(node.worker_id)
            channel = None
            try:
                if node.health["layered_transfer"] != layered:
                    raise RemoteUnavailableError("P/D transfer modes differ")
                channel = self.pool.connect(node).open(request.transfer_id, cancel)
                channel.send({"op": "reserve", "request": request.to_object()})
                ready = response(channel, "ready" if layered else "reserved")
                reservation_prefix(ready, request, layered=layered)
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

    def _stream(self, prompt_ids, max_tokens, *, ignore_eos=False, cancel=None, request_id=None,
                prepared_prefix=None):
        request_id = request_id or str(uuid4())
        with self.active_lock:
            if request_id in self.active_routes:
                raise ValueError("request_id is already active in this router")
            self.active_routes[request_id] = None
        try:
            yield from self._stream_bound(
                prompt_ids, max_tokens, ignore_eos=ignore_eos, cancel=cancel, request_id=request_id,
                prepared_prefix=prepared_prefix,
            )
        finally:
            with self.active_lock:
                self.active_routes.pop(request_id, None)

    def _stream_bound(
        self, prompt_ids, max_tokens, *, ignore_eos=False, cancel=None, request_id=None,
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
        try:
            prefill, decode, d, ready, bucket = self._route_and_reserve(
                request, cancel, prepared_prefix
            )
            held_prefill, held_decode = prefill, decode
            with self.active_lock:
                self.active_routes[request.transfer_id] = (prefill.worker_id, decode.worker_id)
            p = self.pool.connect(prefill).open(request.transfer_id, cancel)
            layered = prefill.health["layered_transfer"]

            def release_prefill():
                nonlocal held_prefill
                self.router.release(held_prefill)
                held_prefill = None

            yield from stream_reserved(
                request, p, d, ready, layered=layered, on_prefill_done=release_prefill
            )
        finally:
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
