"""Randomly bind one request to healthy P/D Workers and persistent channels."""

import json
import math
import threading
from uuid import uuid4

from .client import (
    PDClient,
    RemoteCapacityError,
    RemoteUnavailableError,
    response,
)
from .control import StartRequest
from .layered_protocol import LayeredReceiveTicket
from .protocol import RequestHandoff, _token
from .router import NoHealthyWorker, RandomRouter, WorkerPool


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
        self.router = RandomRouter(self.pool, routing_seed)
        self.ready_timeout = ready_timeout
        self.active_lock = threading.Lock()
        self.active_routes = {}

    def start(self):
        self.pool.start(self.ready_timeout)

    def close(self):
        self.pool.close()

    def _reserve(self, request, prefill, cancel):
        excluded, last_error = set(), None
        layered = prefill.health["layered_transfer"]
        for attempt in range(len(self.pool.decode_pool)):
            try:
                node = self.router.select_decode(excluded)
            except NoHealthyWorker:
                if last_error is not None:
                    raise last_error
                raise
            excluded.add(node.worker_id)
            channel = None
            try:
                if node.health["layered_transfer"] != layered:
                    raise RemoteUnavailableError("P/D transfer modes differ")
                channel = self.pool.connect(node).open(request.transfer_id, cancel)
                channel.send({"op": "reserve", "request": request.to_object()})
                ready = response(channel, "ready" if layered else "reserved")
                if layered:
                    ticket = LayeredReceiveTicket.from_wire(json.dumps(ready["ticket"]).encode())
                    if ticket.transfer_id != request.transfer_id or ticket.kv_tokens != len(
                        request.prompt_ids
                    ):
                        raise ValueError("receive ticket belongs to another request")
                elif ready.get("transfer_id") != request.transfer_id:
                    raise ValueError("reservation belongs to another request")
                self.router.record(request.transfer_id, prefill, node, attempt)
                return node, channel, ready
            except InterruptedError:
                if channel is not None:
                    channel.close()
                raise
            except (RemoteCapacityError, RemoteUnavailableError, OSError) as exc:
                last_error = exc
                if not isinstance(exc, RemoteCapacityError):
                    self.pool.mark_unhealthy(node, exc)
                self.router.record(request.transfer_id, prefill, node, attempt, "reserve_retry")
                if channel is not None:
                    channel.close()
            except BaseException:
                if channel is not None:
                    channel.close()
                raise
        raise last_error or NoHealthyWorker("no decode reservation succeeded")

    def _stream(self, prompt_ids, max_tokens, *, ignore_eos=False, cancel=None, request_id=None):
        request_id = request_id or str(uuid4())
        with self.active_lock:
            if request_id in self.active_routes:
                raise ValueError("request_id is already active in this router")
            self.active_routes[request_id] = None
        try:
            yield from self._stream_bound(
                prompt_ids, max_tokens, ignore_eos=ignore_eos, cancel=cancel, request_id=request_id
            )
        finally:
            with self.active_lock:
                self.active_routes.pop(request_id, None)

    def _stream_bound(
        self, prompt_ids, max_tokens, *, ignore_eos=False, cancel=None, request_id=None
    ):
        self.start()
        request = StartRequest(
            request_id or str(uuid4()), 0, tuple(prompt_ids), max_tokens, ignore_eos
        )
        if cancel is not None and cancel.is_set():
            raise InterruptedError("request cancelled")
        prefill = self.router.select_prefill()
        p = d = None
        try:
            decode, d, ready = self._reserve(request, prefill, cancel)
            with self.active_lock:
                self.active_routes[request.transfer_id] = (prefill.worker_id, decode.worker_id)
            p = self.pool.connect(prefill).open(request.transfer_id, cancel)
            layered = prefill.health["layered_transfer"]
            start = {"op": "prefill", "request": request.to_object()}
            if layered:
                start["ticket"] = ready["ticket"]
            p.send(start)
            wire = response(p, "handoff")["handoff"]
            handoff = RequestHandoff.from_wire(json.dumps(wire).encode())
            if (
                handoff.uid != request.uid
                or handoff.kv.transfer_id != request.transfer_id
                or handoff.prompt_ids != request.prompt_ids
                or handoff.max_tokens != max_tokens
                or handoff.ignore_eos != ignore_eos
            ):
                raise ValueError("prefill handoff does not match the routed request")
            yield handoff.first_token_id
            if not handoff.remaining_tokens:
                if layered:
                    if response(p, "retired").get("transfer_id") != request.transfer_id:
                        raise ValueError("incorrect retirement acknowledgment")
                d.send({"op": "release", "transfer_id": request.transfer_id})
                if response(d, "released").get("transfer_id") != request.transfer_id:
                    raise ValueError("incorrect release acknowledgment")
                return
            if layered:
                if response(p, "sent").get("transfer_id") != request.transfer_id:
                    raise ValueError("incorrect send acknowledgment")
                d.send({"op": "commit", "transfer_id": request.transfer_id, "handoff": wire})
            else:
                d.send({"op": "prepare", "transfer_id": request.transfer_id, "handoff": wire})
                ticket = response(d, "ready")["ticket"]
                p.send({"op": "send", "ticket": ticket})
                if response(p, "sent").get("transfer_id") != request.transfer_id:
                    raise ValueError("incorrect send acknowledgment")
                d.send({"op": "commit", "transfer_id": request.transfer_id})
            if response(d, "installed").get("transfer_id") != request.transfer_id:
                raise ValueError("incorrect installation acknowledgment")
            for index in range(1, max_tokens):
                message = response(d, "token")
                token = message.get("token_id")
                _token(token)
                finished = index + 1 == max_tokens or (
                    not ignore_eos and token == handoff.eos_token_id
                )
                if (
                    message.get("transfer_id") != request.transfer_id
                    or message.get("index") != index
                    or type(message.get("finished")) is not bool
                    or message["finished"] != finished
                ):
                    raise ValueError("invalid or out-of-order routed decode token")
                yield token
                if finished:
                    return
        finally:
            if p is not None:
                p.close()
            if d is not None:
                d.close()
