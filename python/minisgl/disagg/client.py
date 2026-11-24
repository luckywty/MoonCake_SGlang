"""Route one greedy request through P, direct P-to-D KV transfer, and D decode."""

from __future__ import annotations

import argparse
import json
import math
import socket
from contextlib import ExitStack
from threading import Event
from uuid import uuid4

from .admission import AdmissionBudget, AdmissionError
from .control import Channel, StartRequest, endpoint_address
from .layered_protocol import LayeredReceiveTicket
from .lifecycle import RequestLifecycle
from .protocol import HANDOFF_VERSION, ReceiveTicket, RequestHandoff, _token

"""协调一次请求从 P 到 D 的全过程
PDClient 本质上是一个 P/D Orchestrator（协调器）。"""


class RemoteWorkerError(RuntimeError):
    pass


class RemoteCapacityError(RemoteWorkerError, AdmissionError):
    pass


class RemoteUnavailableError(RemoteWorkerError):
    pass


def response(channel: Channel, status: str) -> dict:
    message = channel.receive()
    if message.get("status") == "error":
        if message.get("code") == "busy":
            raise RemoteCapacityError(str(message.get("error", "worker capacity is busy")))
        if message.get("code") == "unavailable":
            raise RemoteUnavailableError(str(message.get("error", "worker is retired")))
        raise RemoteWorkerError(str(message.get("error", "remote worker failed")))
    if message.get("status") != status:
        raise ValueError(f"expected worker response {status}")
    return message


def reservation_prefix(message, request, *, layered):
    """Validate D's immutable reservation before any Prefill computation."""
    prefix = message.get("decode_prefix_len")
    if (
        message.get("handoff_version") != HANDOFF_VERSION
        or message.get("transfer_id") != request.transfer_id
        or type(message.get("prompt_tokens")) is not int
        or message["prompt_tokens"] != len(request.prompt_ids)
        or type(prefix) is not int
        or not 0 <= prefix <= len(request.prompt_ids)
    ):
        raise ValueError("invalid decode prefix reservation; upgrade Workers and clients together")
    if layered:
        if "ticket" not in message:
            raise ValueError("layered reservation must include a ticket or explicit null")
        wire_ticket = message.get("ticket")
        if prefix == len(request.prompt_ids):
            if wire_ticket is not None:
                raise ValueError("a full decode cache hit must not expose a receive ticket")
        else:
            ticket = LayeredReceiveTicket.from_wire(json.dumps(wire_ticket).encode())
            if (
                ticket.transfer_id != request.transfer_id
                or ticket.prompt_tokens != len(request.prompt_ids)
                or ticket.prefix_len != prefix
            ):
                raise ValueError("receive ticket belongs to another request or suffix")
    return prefix


def stream_reserved(request, p, d, ready, *, layered, on_prefill_done=None):
    """Common reserve-first handshake for direct sockets and routed channels."""
    prefix = reservation_prefix(ready, request, layered=layered)
    transfer_tokens = len(request.prompt_ids) - prefix
    state = RequestLifecycle(request.uid, request.transfer_id)
    state.begin_prefill()
    if layered:
        state.begin_prefill_transfer()
    start = {
        "op": "prefill",
        "request": request.to_object(),
        "decode_prefix_len": prefix,
    }
    if layered:
        start["ticket"] = ready["ticket"]
    try:
        p.send(start)
        wire = response(p, "handoff")["handoff"]
        handoff = RequestHandoff.from_wire(json.dumps(wire).encode())
        if (
            handoff.version != HANDOFF_VERSION
            or handoff.uid != request.uid
            or handoff.kv.transfer_id != request.transfer_id
            or handoff.prompt_ids != request.prompt_ids
            or handoff.max_tokens != request.max_tokens
            or handoff.ignore_eos != request.ignore_eos
            or handoff.kv.prefix_len != prefix
        ):
            raise ValueError("prefill handoff does not match the decode reservation")
        state.complete_prefill(handoff)
        yield handoff.first_token_id
        if not handoff.remaining_tokens:
            if layered or not transfer_tokens:
                if response(p, "retired").get("transfer_id") != request.transfer_id:
                    raise ValueError("incorrect retirement acknowledgment")
            if on_prefill_done is not None:
                on_prefill_done()
            d.send({"op": "release", "transfer_id": request.transfer_id})
            if response(d, "released").get("transfer_id") != request.transfer_id:
                raise ValueError("incorrect release acknowledgment")
            if layered:
                state.finish_without_decode()
            return
        if not layered:
            state.begin_transfer()
            d.send({"op": "prepare", "transfer_id": request.transfer_id, "handoff": wire})
            prepared = response(d, "ready")
            if reservation_prefix(prepared, request, layered=False) != prefix:
                raise ValueError("decode prefix changed after reservation")
            if transfer_tokens:
                ticket = ReceiveTicket.from_wire(json.dumps(prepared["ticket"]).encode())
                if ticket.transfer_id != request.transfer_id or ticket.nbytes != handoff.kv.nbytes:
                    raise ValueError("incorrect suffix receive ticket")
                p.send({"op": "send", "ticket": prepared["ticket"]})
            elif prepared.get("ticket") is not None:
                raise ValueError("a full decode cache hit must not allocate a receive buffer")
        if response(p, "sent").get("transfer_id") != request.transfer_id:
            raise ValueError("incorrect send acknowledgment")
        if on_prefill_done is not None:
            on_prefill_done()
        commit = {"op": "commit", "transfer_id": request.transfer_id}
        if layered:
            commit["handoff"] = wire
        d.send(commit)
        if response(d, "installed").get("transfer_id") != request.transfer_id:
            raise ValueError("incorrect installation acknowledgment")
        state.accept_receive(request.transfer_id)
        state.begin_decode()
        for index in range(1, request.max_tokens):
            item = response(d, "token")
            token = item.get("token_id")
            _token(token)
            finished = index + 1 == request.max_tokens or (
                not request.ignore_eos and token == handoff.eos_token_id
            )
            if (
                item.get("transfer_id") != request.transfer_id
                or item.get("index") != index
                or type(item.get("finished")) is not bool
                or item["finished"] != finished
            ):
                raise ValueError("invalid or out-of-order decode token")
            yield token
            if finished:
                state.finish_decode()
                return
    except InterruptedError:
        state.cancel()
        raise
    except Exception:
        state.fail()
        raise


class PDClient:
    # 没有实现多 P Worker、多 D Worker 的负载均衡
    def __init__(
        self,
        prefill: str,
        decode: str,
        *,
        timeout: float = 120.0,
        max_inflight_requests: int = 128,
        max_inflight_tokens: int = 1048576,
    ):
        endpoint_address(prefill)  # 主要用于验证传入的端点格式能否被解析。
        endpoint_address(decode)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.prefill = prefill
        self.decode = decode
        self.timeout = timeout
        self.admission = AdmissionBudget(max_inflight_requests, max_inflight_tokens)

    def stream(self, prompt_ids, max_tokens, *, ignore_eos=False, cancel=None, request_id=None):
        # Validate before acquiring a shared budget. Each admitted stream owns
        # its own sockets; only accounting updates hold the condition lock.
        request_id = str(uuid4()) if request_id is None else request_id
        StartRequest(request_id, 0, tuple(prompt_ids), max_tokens, ignore_eos)
        with self.admission.reserve(len(prompt_ids) + max_tokens, self.timeout, cancel):
            yield from self._stream(
                prompt_ids, max_tokens, ignore_eos=ignore_eos, cancel=cancel, request_id=request_id
            )

    def _stream(
        self,
        prompt_ids: tuple[int, ...] | list[int],
        max_tokens: int,
        *,
        ignore_eos: bool = False,
        cancel: Event | None = None,
        request_id=None,
    ):
        # 构造一个请求
        request = StartRequest(
            request_id or str(uuid4()), 0, tuple(prompt_ids), max_tokens, ignore_eos
        )
        with ExitStack() as connections:
            channels = []

            # 连接的起点是运行 PDClient 的进程所在机器，分别连接到 P、D Worker
            def connect(endpoint):
                # 检查请求是否已经取消
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("request cancelled")
                sock = connections.enter_context(  # 建立到指定 Worker 的 TCP 连接
                    socket.create_connection(endpoint_address(endpoint), timeout=self.timeout)
                )  # 解析 Worker 地址，并建立 TCP 连接。
                channel = Channel(sock, self.timeout, cancel)
                channels.append(channel)
                return channel

            try:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("request cancelled")
                p = connect(self.prefill)  # 与prefillWorker建立连接
                p.send({"op": "capabilities"})
                capabilities = response(p, "capabilities")
                layered = capabilities.get("layered_transfer")
                if (
                    type(layered) is not bool
                    or capabilities.get("handoff_version") != HANDOFF_VERSION
                    or capabilities.get("decode_prefix_reuse") is not True
                ):
                    raise ValueError(
                        "incompatible Prefill protocol; upgrade Workers and clients together"
                    )
                d = connect(self.decode)
                d.send({"op": "capabilities"})
                decode_caps = response(d, "capabilities")
                if (
                    decode_caps.get("layered_transfer") is not layered
                    or decode_caps.get("handoff_version") != HANDOFF_VERSION
                    or decode_caps.get("decode_prefix_reuse") is not True
                ):
                    raise ValueError("incompatible Decode protocol or transfer mode")
                d.send({"op": "reserve", "request": request.to_object()})
                ready = response(d, "ready" if layered else "reserved")
                yield from stream_reserved(request, p, d, ready, layered=layered)
            finally:
                # Explicit cancellation also covers closing a generator directly
                # after its first token. Socket closure remains the fallback.
                for channel in channels:
                    try:
                        Channel(channel.socket, min(self.timeout, 0.2)).send(
                            {"op": "cancel", "transfer_id": request.transfer_id}
                        )
                    except (OSError, ValueError):
                        pass

    # 这个 generate() 是 stream() 的非流式封装：把逐个产生的 token 编号收集成列表，等生成结束后一次性返回。
    def generate(
        self, prompt_ids, max_tokens, *, ignore_eos=False, cancel=None, request_id=None
    ) -> list[int]:
        return list(
            self.stream(
                prompt_ids, max_tokens, ignore_eos=ignore_eos, cancel=cancel, request_id=request_id
            )
        )


def main() -> None:
    from .cluster_config import add_routing_args, remote_addresses

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill")
    parser.add_argument("--decode")
    add_routing_args(parser)
    parser.add_argument("--input-ids", required=True, help="JSON array of input token IDs")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    try:
        prefill, decode = remote_addresses(args)
        if prefill:
            if args.prefill or args.decode:
                raise ValueError("use either single endpoints or address lists")
            from .routed_client import RoutedPDClient
            from .routing_policy import policy_config

            client = RoutedPDClient(
                prefill,
                decode,
                timeout=args.timeout,
                routing_seed=args.routing_seed,
                health_interval=args.health_interval,
                health_timeout=args.health_timeout,
                ready_timeout=args.worker_ready_timeout,
                max_quarantined_bytes=args.max_quarantined_bytes,
                routing_config=policy_config(args),
            )
        elif args.prefill and args.decode:
            client = PDClient(args.prefill, args.decode, timeout=args.timeout)
        else:
            raise ValueError("both P and D endpoints or address lists are required")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    try:
        tokens = client.generate(
            json.loads(args.input_ids), args.max_tokens, ignore_eos=args.ignore_eos
        )
        print(json.dumps({"token_ids": tokens, "completion_tokens": len(tokens)}))
    finally:
        if hasattr(client, "close"):
            client.close()


if __name__ == "__main__":
    main()
