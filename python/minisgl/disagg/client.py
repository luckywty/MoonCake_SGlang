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
from .protocol import RequestHandoff, _token

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
        state = RequestLifecycle(
            request.uid, request.transfer_id
        )  # 它用于跟踪请求经过的执行阶段，它记录和检查请求的阶段变化
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
                state.begin_prefill()  # 向prefillWorker发送prefill请求，更新客户端维护的请求状态。
                p.send({"op": "capabilities"})
                capabilities = response(p, "capabilities")
                layered = capabilities.get("layered_transfer")
                if type(layered) is not bool:
                    raise ValueError("invalid worker transfer capabilities")
                start = {"op": "prefill", "request": request.to_object()}
                if layered:
                    # This is before P starts any prompt computation. D reserves
                    # final KV pages and a slice of its startup-registered pool.
                    d = connect(self.decode)
                    d.send({"op": "reserve", "request": request.to_object()})
                    wire_ticket = response(d, "ready")["ticket"]
                    ticket = LayeredReceiveTicket.from_wire(json.dumps(wire_ticket).encode())
                    if ticket.transfer_id != request.transfer_id or ticket.kv_tokens != len(
                        prompt_ids
                    ):
                        raise ValueError("early reservation belongs to another request")
                    start["ticket"] = wire_ticket
                    state.begin_prefill_transfer()
                p.send(start)
                envelope = response(p, "handoff")["handoff"]  # 等待 Prefill Worker 返回 Handoff
                handoff = RequestHandoff.from_wire(json.dumps(envelope).encode())
                if (
                    handoff.uid != request.uid
                    or handoff.kv.transfer_id != request.transfer_id
                    or handoff.prompt_ids != request.prompt_ids
                    or handoff.max_tokens != request.max_tokens
                    or handoff.ignore_eos != request.ignore_eos
                ):
                    raise ValueError("prefill handoff does not match the request")

                # 提前返回首 Token
                state.complete_prefill(handoff)
                yield handoff.first_token_id

                if layered:
                    status = "sent" if handoff.remaining_tokens else "retired"
                    if response(p, status).get("transfer_id") != request.transfer_id:
                        raise ValueError("incorrect layer send acknowledgment")
                    if not handoff.remaining_tokens:
                        d.send({"op": "release", "transfer_id": request.transfer_id})
                        if response(d, "released").get("transfer_id") != request.transfer_id:
                            raise ValueError("incorrect reservation release acknowledgment")
                        state.finish_without_decode()
                        return
                    d.send(
                        {"op": "commit", "transfer_id": request.transfer_id, "handoff": envelope}
                    )
                elif not handoff.remaining_tokens:
                    return
                else:
                    d = connect(self.decode)
                    d.send({"op": "decode", "handoff": envelope})
                    ticket = response(d, "ready")["ticket"]
                    state.begin_transfer()
                    p.send({"op": "send", "ticket": ticket})
                    if response(p, "sent").get("transfer_id") != request.transfer_id:
                        raise ValueError("incorrect send acknowledgment")
                    d.send({"op": "commit", "transfer_id": request.transfer_id})
                if response(d, "installed").get("transfer_id") != request.transfer_id:
                    raise ValueError("incorrect installation acknowledgment")
                state.accept_receive(request.transfer_id)
                state.begin_decode()
                for index in range(1, max_tokens):
                    item = response(d, "token")  # 负责从 Decode Worker 接收消息。
                    if item.get("transfer_id") != request.transfer_id or item.get("index") != index:
                        raise ValueError("out-of-order decode output")
                    token = item.get("token_id")  # 从 Decode Worker 的响应消息中提取 Token ID
                    _token(token)

                    # 判断 Decode 是否应该结束
                    finished = index + 1 == max_tokens or (
                        not ignore_eos and token == handoff.eos_token_id
                    )
                    if type(item.get("finished")) is not bool or item["finished"] != finished:
                        raise ValueError("incorrect decode completion flag")
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

            client = RoutedPDClient(
                prefill,
                decode,
                timeout=args.timeout,
                routing_seed=args.routing_seed,
                health_interval=args.health_interval,
                health_timeout=args.health_timeout,
                ready_timeout=args.worker_ready_timeout,
                max_quarantined_bytes=args.max_quarantined_bytes,
            )
        elif args.prefill and args.decode:
            client = PDClient(args.prefill, args.decode, timeout=args.timeout)
        else:
            raise ValueError("both P and D endpoints or address lists are required")
    except ValueError as exc:
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
