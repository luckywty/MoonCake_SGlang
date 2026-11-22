"""Reference TCP transfer using private host staging and attempt-scoped receive tickets."""

from __future__ import annotations

import hashlib
import json
import math
import socket
import threading
import time
from collections.abc import Callable

from .control import Channel, endpoint_address, endpoint_name, read_bytes, write_bytes
from .protocol import KVLayout, KVTransferMetadata, ReceiveTicket
from .transfer import MockFabric, MockTransferBackend, TransferError

'''
P Worker
    |
    | 1. Prefill 计算得到 KV Cache
    v
GPU KV Cache
    |
    | 2. export_tokens()
    v
CPU bytes
    |
    | 3. TCP send()
    v
======================= 网络 =======================
    |
    | 4. TCP receive()
    v
D Worker CPU staging
    |
    | 5. import_tokens()
    v
D Worker GPU KV Cache
    |
    | 6. 继续执行 Decode
    v
生成后续 Token

'''
class TcpTransferBackend(MockTransferBackend):  #继承MockTransferBackend
    def __init__(   #初始化 TCP Backend
        self,
        bind: tuple[str, int] | None = None,
        *,
        advertise_host: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self._stop = threading.Event()
        self._listener = None
        self._connection = None
        self._connection_lock = threading.Lock()
        self._thread = None
        self._cancellations = {}
        endpoint = "tcp-sender"
        if bind is not None:    #bind 参数决定是否开启接收服务
            ''' 对于 P Worker，bind=None，不创建接收监听 Socket。
                对于 D Worker，会在指定地址和端口创建 TCP 监听 Socket。'''

            #创建 TCP 监听 socket。bind 通常是 (host, port)
            listener = socket.create_server(bind)
            listener.settimeout(0.2)    #设置监听操作的等待时间
            host, port = listener.getsockname()[:2]

            #这行只修改准备公布的地址字符串，不会改变 socket 已经绑定的地址。
            host = advertise_host or host   #如果提供了非空的 advertise_host，就使用它；否则使用实际绑定得到的 host。
            if host in ("0.0.0.0", "::"):   #阻止公布通配监听地址
                listener.close()
                raise ValueError("wildcard bind requires an advertised host")
            endpoint = endpoint_name(host, port)
            self._listener = listener
        super().__init__(endpoint, MockFabric())
        if self._listener is not None:
            self._thread = threading.Thread(target=self._serve, name="pd-kv-receiver", daemon=True)
            self._thread.start()

    def prepare_receive(
        self, meta: KVTransferMetadata, local_layout: KVLayout, destination: bytearray
    ) -> ReceiveTicket:
        if self._listener is None or self._stop.is_set():
            raise TransferError("TCP receive endpoint is unavailable")
        target = super().prepare_receive(meta, local_layout, destination)
        with self._fabric._lock:
            self._cancellations[target.locator] = threading.Event()
        return target

    def finish_receive(self, target: ReceiveTicket) -> None:
        super().finish_receive(target)
        with self._fabric._lock:
            self._cancellations.pop(target.locator, None)

    def discard_receive(self, target: ReceiveTicket) -> None:
        if target.endpoint != self.endpoint:
            raise TransferError("only the receiver can discard a ticket")
        with self._fabric._lock:
            cancelled = self._cancellations.pop(target.locator, None)
            if cancelled is not None:
                cancelled.set()
        super().discard_receive(target)

    def send(
        self,
        meta: KVTransferMetadata,
        source: bytes | bytearray,
        target: ReceiveTicket,
        *,
        cancel_check: Callable[[], None] | None = None,
    ) -> None:
        if cancel_check is not None:
            cancel_check()
        if not isinstance(source, (bytes, bytearray)) or len(source) != meta.nbytes:
            raise TransferError("incorrect source buffer")
        if target.transfer_id != meta.transfer_id or target.nbytes != meta.nbytes:
            raise TransferError("ticket metadata mismatch")
        payload = bytes(source)
        try:
            with socket.create_connection(
                endpoint_address(target.endpoint), timeout=self.timeout
            ) as sock:
                channel = Channel(sock, self.timeout)
                channel.send(
                    {
                        "metadata": json.loads(meta.to_wire()),
                        "ticket": json.loads(target.to_wire()),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                ready = channel.receive()
                if ready != {"status": "ready"}:
                    raise TransferError(str(ready.get("error", "receive ticket rejected")))
                if cancel_check is not None:
                    cancel_check()
                write_bytes(sock, payload, time.monotonic() + self.timeout)
                if channel.receive() != {"status": "complete"}:
                    raise TransferError("TCP transfer was not acknowledged")
                if cancel_check is not None:
                    cancel_check()
        except (OSError, ValueError) as exc:
            raise TransferError(f"TCP transfer failed: {exc}") from exc

    def _receive(self, sock: socket.socket) -> None:
        channel = Channel(sock, self.timeout)
        try:
            header = channel.receive()
            if set(header) != {"metadata", "ticket", "sha256"}:
                raise TransferError("incorrect transfer header")
            meta = KVTransferMetadata.from_wire(json.dumps(header["metadata"]).encode())
            target = ReceiveTicket.from_wire(json.dumps(header["ticket"]).encode())
            with self._fabric._lock:
                entry = self._lookup(target)
                if entry.meta != meta or entry.delivered:
                    raise TransferError("incorrect or consumed transfer ticket")
                cancelled = self._cancellations.get(target.locator)
                if cancelled is None or cancelled.is_set():
                    raise TransferError("receive ticket was cancelled")
            channel.send({"status": "ready"})
            payload = read_bytes(sock, target.nbytes, time.monotonic() + self.timeout, cancelled)
            if hashlib.sha256(payload).hexdigest() != header["sha256"]:
                raise TransferError("KV payload checksum mismatch")
            # This lock-protected host copy revalidates the ticket after the network receive.
            # A discarded reservation can never be written by a late socket operation.
            super().send(meta, payload, target)
            channel.send({"status": "complete"})
        except (OSError, ValueError, TransferError) as exc:
            try:
                channel.send({"status": "error", "error": str(exc)})
            except OSError:
                pass

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                sock, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with sock:
                with self._connection_lock:
                    if self._stop.is_set():
                        return
                    self._connection = sock
                try:
                    self._receive(sock)
                finally:
                    with self._connection_lock:
                        self._connection = None

    def close(self) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
        with self._connection_lock:
            if self._connection is not None:
                try:
                    self._connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=self.timeout + 1)
            if self._thread.is_alive():
                raise TransferError("TCP receive thread did not stop")
        with self._fabric._lock:
            for cancelled in self._cancellations.values():
                cancelled.set()
            self._cancellations.clear()
            for entry in self._fabric._reservations.values():
                entry.destination.release()
            self._fabric._reservations.clear()

    def __enter__(self) -> TcpTransferBackend:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
