"""Route one greedy request through P, direct P-to-D KV transfer, and D decode."""

from __future__ import annotations

import argparse
import json
import math
import socket
from contextlib import ExitStack
from threading import Event
from uuid import uuid4

from .control import Channel, StartRequest, endpoint_address
from .lifecycle import RequestLifecycle
from .protocol import RequestHandoff, _token

'''协调一次请求从 P 到 D 的全过程
PDClient 本质上是一个 P/D Orchestrator（协调器）。'''
class RemoteWorkerError(RuntimeError):
    pass


def response(channel: Channel, status: str) -> dict:
    message = channel.receive()
    if message.get("status") == "error":
        raise RemoteWorkerError(str(message.get("error", "remote worker failed")))
    if message.get("status") != status:
        raise ValueError(f"expected worker response {status}")
    return message


class PDClient:
    #没有实现多 P Worker、多 D Worker 的负载均衡
    def __init__(self, prefill: str, decode: str, *, timeout: float = 120.0):
        endpoint_address(prefill)   #主要用于验证传入的端点格式能否被解析。
        endpoint_address(decode)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.prefill = prefill
        self.decode = decode
        self.timeout = timeout

    def stream(
        self,
        prompt_ids: tuple[int, ...] | list[int],
        max_tokens: int,
        *,
        ignore_eos: bool = False,
        cancel: Event | None = None,
    ):
        #构造一个请求
        request = StartRequest(str(uuid4()), 0, tuple(prompt_ids), max_tokens, ignore_eos)
        state = RequestLifecycle(request.uid, request.transfer_id)  #它用于跟踪请求经过的执行阶段，它记录和检查请求的阶段变化
        with ExitStack() as connections:

            #连接的起点是运行 PDClient 的进程所在机器，分别连接到 P、D Worker
            def connect(endpoint):

                #检查请求是否已经取消
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("request cancelled")
                sock = connections.enter_context(   #建立到指定 Worker 的 TCP 连接
                    socket.create_connection(endpoint_address(endpoint), timeout=self.timeout)
                )   #解析 Worker 地址，并建立 TCP 连接。
                return Channel(sock, self.timeout, cancel)  #把 socket 包装成控制消息通道。

            try:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("request cancelled")
                p = connect(self.prefill)   #与prefillWorker建立连接
                state.begin_prefill()       #向prefillWorker发送prefill请求，更新客户端维护的请求状态。
                p.send({"op": "prefill", "request": request.to_object()})   #Prefill Worker 接收请求，执行对应的模型计算。
                envelope = response(p, "handoff")["handoff"]    #等待 Prefill Worker 返回 Handoff
                handoff = RequestHandoff.from_wire(json.dumps(envelope).encode())
                if (
                    handoff.uid != request.uid
                    or handoff.kv.transfer_id != request.transfer_id
                    or handoff.prompt_ids != request.prompt_ids
                    or handoff.max_tokens != request.max_tokens
                    or handoff.ignore_eos != request.ignore_eos
                ):
                    raise ValueError("prefill handoff does not match the request")

                #提前返回首 Token
                state.complete_prefill(handoff)
                yield handoff.first_token_id

                if not handoff.remaining_tokens:
                    return
                
                d = connect(self.decode)
                d.send({"op": "decode", "handoff": envelope})   #PDClient 向 D Worker 发送控制请求，要求它接续 P 已完成 Prefill 的这个请求。
                ticket = response(d, "ready")["ticket"] #取得 D Worker 为本次传输生成的 Ticket
                state.begin_transfer()
                p.send({"op": "send", "ticket": ticket})    #PDClient 通知 P Worker，按照 D 提供的接收凭据，将 KV 数据直接发送给 D。
                if response(p, "sent").get("transfer_id") != request.transfer_id:
                    raise ValueError("incorrect send acknowledgment")
                d.send({"op": "commit", "transfer_id": request.transfer_id})    #PDClient 向 D Worker 发送“提交本次 KV 交接”。
                if response(d, "installed").get("transfer_id") != request.transfer_id:
                    raise ValueError("incorrect installation acknowledgment")
                state.accept_receive(request.transfer_id)
                state.begin_decode()
                for index in range(1, max_tokens):
                    item = response(d, "token") #负责从 Decode Worker 接收消息。
                    if item.get("transfer_id") != request.transfer_id or item.get("index") != index:
                        raise ValueError("out-of-order decode output")
                    token = item.get("token_id")    #从 Decode Worker 的响应消息中提取 Token ID
                    _token(token)

                    #判断 Decode 是否应该结束
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
    #这个 generate() 是 stream() 的非流式封装：把逐个产生的 token 编号收集成列表，等生成结束后一次性返回。
    def generate(self, prompt_ids, max_tokens, *, ignore_eos=False, cancel=None) -> list[int]:
        return list(self.stream(prompt_ids, max_tokens, ignore_eos=ignore_eos, cancel=cancel))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--input-ids", required=True, help="JSON array of input token IDs")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    client = PDClient(args.prefill, args.decode, timeout=args.timeout)
    tokens = client.generate(
        json.loads(args.input_ids), args.max_tokens, ignore_eos=args.ignore_eos
    )
    print(json.dumps({"token_ids": tokens, "completion_tokens": len(tokens)}))


if __name__ == "__main__":
    main()
