"""GPU-only Mooncake RDMA, owned directly by the P/D Worker processes.

TCP carries descriptors and completion messages only. KV stays in CUDA memory;
there is no helper process, host payload snapshot, or TCP data-plane fallback.
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

import torch

from .control import CancelledError, Channel, endpoint_address, endpoint_name
from .mooncake_runtime import MooncakeRuntime
from .protocol import KVLayout, KVTransferMetadata, ReceiveTicket
from .transfer import TransferBackendUnavailable, TransferError

if TYPE_CHECKING:
    from .mooncake_runtime import RegisteredBuffer

MAX_TRANSFER_BYTES = 1024**3
_TRANSPORT = "mooncake-rdma-gpu-v1"
_TERMINAL_STATES = {"completed", "failed_drained", "not_submitted"}


@dataclass
class _Receive:
    meta: KVTransferMetadata
    target: ReceiveTicket
    buffer: RegisteredBuffer
    lock: threading.RLock = field(default_factory=threading.RLock)
    done: threading.Event = field(default_factory=threading.Event)
    exposed: bool = False
    cancelled: bool = False
    delivered: bool = False
    sender_session: str | None = None
    terminal: str | None = None
    error: str | None = None


class MooncakeTransferBackend:
    """Persistent Worker TE with GPU registrations and bounded control handlers.

    P Worker 的 send() 直接提交本进程 GPU Tensor，不再把 KV 交给 helper。
    D Worker 的 _receive() 只交换地址和状态，KV 由 RDMA 写入 GPU。
    业务取消只撤销交付；注册区必须活到能够证明原生访问已经结束。
    """

    name = "mooncake"
    gpu_payload = True

    def __init__(
        self,
        bind: tuple[str, int] | None = None,
        *,
        local_host: str = "127.0.0.1",
        advertise_host: str | None = None,
        timeout: float = 30.0,
        max_transfer_bytes: int = MAX_TRANSFER_BYTES,
        max_registered_bytes: int | None = None,
        max_pending_transfers: int = 32,
        max_quarantine_seconds: float = 300.0,
        device_name: str = "",
        rpc_port: int | None = None,
        _engine_module: str = "mooncake.engine",
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if type(max_pending_transfers) is not int or max_pending_transfers <= 0:
            raise ValueError("max_pending_transfers must be positive")
        self.timeout = timeout
        self.max_transfer_bytes = max_transfer_bytes
        self._stop = threading.Event()
        self._bound = threading.Event()
        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self._fatal_error: BaseException | None = None
        self._receives: dict[str, _Receive] = {}
        self._attempts: dict[str, str] = {}
        self._preparing: set[str] = set()
        self._tombstones: OrderedDict[tuple, tuple[str, str]] = OrderedDict()
        self._connections: set[socket.socket] = set()
        self._handler_futures: set[Future] = set()
        self._cleanup_futures: set[Future] = set()
        self._finalizing: set[tuple[str, str]] = set()
        self._cleanup_slots = threading.BoundedSemaphore(max_pending_transfers)
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._handler_slots = threading.BoundedSemaphore(max(2, max_pending_transfers))
        self._handlers = ThreadPoolExecutor(
            max_workers=max(2, max_pending_transfers), thread_name_prefix="pd-rdma-control"
        )
        self._cleanup = ThreadPoolExecutor(
            max_workers=min(4, max_pending_transfers), thread_name_prefix="pd-rdma-finalize"
        )
        self.runtime = MooncakeRuntime(
            local_host,
            timeout=timeout,
            max_transfer_bytes=max_transfer_bytes,
            max_registered_bytes=(
                2 * max_transfer_bytes if max_registered_bytes is None else max_registered_bytes
            ),
            max_pending_transfers=max_pending_transfers,
            max_quarantine_seconds=max_quarantine_seconds,
            device_name=device_name,
            engine_module=_engine_module,
            rpc_port=rpc_port,
        )
        self.endpoint = "rdma-sender"
        if bind is not None:
            listener = socket.create_server(bind)
            listener.settimeout(0.2)
            host, port = listener.getsockname()[:2]
            host = advertise_host or host
            if host in ("0.0.0.0", "::"):
                listener.close()
                raise ValueError("wildcard bind requires an advertised host")
            self.endpoint = endpoint_name(host, port)
            self._listener = listener
            self._thread = threading.Thread(
                target=self._serve, name="pd-rdma-listener", daemon=True
            )
            self._thread.start()

    @property
    def runtime_info(self) -> dict:
        return self.runtime.info

    def bind_device(self, device: torch.device) -> None:
        # Engine 必须先初始化 CUDA，再让 TE 绑定这个 Worker 的 CUDA context。
        self.runtime.bind_device(device)
        self._bound.set()

    def check_available(self) -> None:
        if self._stop.is_set() or self._closing:
            raise TransferBackendUnavailable("Mooncake backend is closing")
        with self._lock:
            error = self._fatal_error
        if error is not None:
            raise TransferBackendUnavailable(f"Mooncake control backend failed: {error}")
        if not self._bound.is_set():
            raise TransferBackendUnavailable("Mooncake backend is not bound to its GPU")
        self.runtime.check_available()

    def check_capacity(self, nbytes: int) -> None:
        self.check_available()
        self.runtime.check_capacity(nbytes)

    def health(self) -> dict:
        native = self.runtime.health()
        with self._lock:
            count = len(self._receives)
            connections = len(self._connections)
            error = str(self._fatal_error) if self._fatal_error is not None else None
        status = native.get("state", "unavailable")
        if error is not None or self._stop.is_set() or self._closing or not self._bound.is_set():
            status = "unavailable"
        elif status not in ("ready", "degraded"):
            status = "unavailable"
        return {
            "status": status,
            "mooncake": {
                **native,
                "control_endpoint": self.endpoint,
                "receive_records": count,
                "control_connections": connections,
                "control_error": error,
            },
        }

    def _record_fatal(self, error: BaseException) -> None:
        with self._lock:
            if self._fatal_error is None:
                self._fatal_error = error

    @contextmanager
    def _connection(self, endpoint: str):
        if self._stop.is_set():
            raise TransferBackendUnavailable("Mooncake backend is closing")
        sock = socket.create_connection(endpoint_address(endpoint), timeout=self.timeout)
        with self._lock:
            self._connections.add(sock)
        try:
            if self._stop.is_set():
                raise TransferBackendUnavailable("Mooncake backend is closing")
            yield sock
        finally:
            with self._lock:
                self._connections.discard(sock)
            sock.close()

    def prepare_receive(
        self, meta: KVTransferMetadata, local_layout: KVLayout, destination: torch.Tensor
    ) -> ReceiveTicket:
        self.check_capacity(meta.nbytes)
        if self._listener is None:
            raise TransferError("Mooncake receive endpoint is unavailable")
        if meta.layout != local_layout:
            raise TransferError("incompatible KV layout or model")
        self._validate_tensor(destination, meta.nbytes)
        target = ReceiveTicket(meta.transfer_id, self.endpoint, str(uuid4()), meta.nbytes)
        with self._lock:
            if meta.transfer_id in self._attempts or meta.transfer_id in self._preparing:
                raise TransferError("receive attempt is already reserved")
            self._preparing.add(meta.transfer_id)
        try:
            buffer = self.runtime.register_tensor(destination, target.locator)
            with self._lock:
                self._receives[target.locator] = _Receive(meta, target, buffer)
                self._attempts[meta.transfer_id] = target.locator
            return target
        finally:
            with self._lock:
                self._preparing.discard(meta.transfer_id)

    @staticmethod
    def _validate_tensor(tensor: torch.Tensor, nbytes: int) -> None:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cuda"
            or not tensor.is_contiguous()
            or tensor.numel() * tensor.element_size() != nbytes
        ):
            raise TransferError("Mooncake payload must be a contiguous CUDA Tensor of exact size")

    def _lookup(self, target: ReceiveTicket) -> _Receive:
        with self._lock:
            entry = self._receives.get(target.locator)
            if target.endpoint != self.endpoint or entry is None or entry.target != target:
                raise TransferError("unknown or stale receive ticket")
            return entry

    def finish_receive(self, target: ReceiveTicket) -> None:
        entry = self._lookup(target)
        if not entry.done.wait(self.timeout):
            self.discard_receive(target)
            raise TransferError("Mooncake receive completion timed out")
        with entry.lock:
            if not entry.delivered or entry.cancelled:
                raise TransferError(entry.error or "Mooncake transfer is incomplete")
            self.runtime.release(entry.buffer)
            self._forget(entry, "complete")

    def discard_receive(self, target: ReceiveTicket) -> None:
        if target.endpoint != self.endpoint:
            raise TransferError("only the receiver can discard a ticket")
        with self._lock:
            entry = self._receives.get(target.locator)
        if entry is None or entry.target != target:
            return
        with entry.lock:
            entry.cancelled = True
            entry.error = entry.error or "request cancelled"
            entry.done.set()
            if entry.buffer.safe_to_release:
                self.runtime.release(entry.buffer)
                self._forget(entry, "retired")
            else:
                # 已公布的 GPU 地址仍可能被原生传输访问，不能立即注销或复用。
                self.runtime.quarantine(entry.buffer, entry.error)

    def _identity(self, entry: _Receive) -> tuple:
        return (
            entry.target.transfer_id,
            entry.target.locator,
            self.runtime.info["session"],
            entry.target.nbytes,
            entry.sender_session,
        )

    def _forget(self, entry: _Receive, result: str) -> None:
        with self._lock:
            if self._receives.get(entry.target.locator) is entry:
                self._receives.pop(entry.target.locator)
            if self._attempts.get(entry.target.transfer_id) == entry.target.locator:
                self._attempts.pop(entry.target.transfer_id)
            if entry.sender_session is not None:
                self._tombstones[self._identity(entry)] = (
                    entry.terminal or "not_submitted",
                    result,
                )
                while len(self._tombstones) > 1024:
                    self._tombstones.popitem(last=False)

    def send(
        self,
        meta: KVTransferMetadata,
        source: torch.Tensor,
        target: ReceiveTicket,
        *,
        cancel_check: Callable[[], None] | None = None,
    ) -> None:
        self.check_capacity(meta.nbytes)
        self._validate_tensor(source, meta.nbytes)
        if target.transfer_id != meta.transfer_id or target.nbytes != meta.nbytes:
            raise TransferError("ticket metadata mismatch")
        if cancel_check is not None:
            cancel_check()
        buffer = operation = remote = terminal = None
        submitted = deferred = False
        try:
            with self._connection(target.endpoint) as sock:
                channel = Channel(sock, self.timeout, self._stop)
                channel.send(
                    {
                        "op": "prepare",
                        "backend": _TRANSPORT,
                        "metadata": json.loads(meta.to_wire()),
                        "ticket": json.loads(target.to_wire()),
                        "source_session": self.runtime.info["session"],
                    }
                )
                ready = channel.receive()
                if set(ready) != {"status", "backend", "target"} or (
                    ready["status"] != "ready" or ready["backend"] != _TRANSPORT
                ):
                    raise TransferError(str(ready.get("error", "incompatible transfer backend")))
                candidate = ready["target"]
                if not isinstance(candidate, dict) or set(candidate) != {
                    "session",
                    "address",
                    "nbytes",
                }:
                    raise TransferError("incorrect Mooncake remote descriptor")
                if not isinstance(candidate["session"], str):
                    raise TransferError("incorrect Mooncake remote session")
                endpoint_address(candidate["session"])
                if (
                    type(candidate["address"]) is not int
                    or candidate["address"] <= 0
                    or type(candidate["nbytes"]) is not int
                    or candidate["nbytes"] != meta.nbytes
                ):
                    raise TransferError("incorrect Mooncake remote allocation")
                remote = candidate
                if cancel_check is not None:
                    cancel_check()
                buffer = self.runtime.register_tensor(source, "send:" + target.locator)
                if cancel_check is not None:
                    cancel_check()
                # 提交接口开始执行后，异常也不能等同于“肯定未提交”。
                submitted = True
                operation = self.runtime.submit_write(buffer, remote)
                deadline = time.monotonic() + self.timeout
                while True:
                    self.check_available()
                    if cancel_check is not None:
                        cancel_check()
                    terminal = self.runtime.poll(operation, timeout_ms=50)
                    if terminal != "unknown":
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Mooncake transfer completion timed out")
                channel.send(self._terminal_message(target, remote, terminal, "terminal"))
                result = self._check_receipt(channel.receive(), target)
                if terminal != "completed":
                    raise TransferError("Mooncake native transfer failed after local drain")
                if result != "complete":
                    raise CancelledError("receiver no longer accepts this request")
        except BaseException as exc:
            if operation is not None and terminal in (None, "unknown"):
                self._defer_source(operation, buffer, target, remote)
                deferred = True
            elif remote is not None and terminal in _TERMINAL_STATES:
                self._schedule_finalize(target, remote, terminal)
            elif remote is not None and not submitted:
                self._schedule_finalize(target, remote, "not_submitted")
            if isinstance(
                exc, (CancelledError, TransferBackendUnavailable, KeyboardInterrupt, SystemExit)
            ):
                raise
            raise TransferError(f"Mooncake GPU transfer failed: {exc}") from exc
        finally:
            if buffer is not None and buffer.registered and not deferred:
                if buffer.safe_to_release:
                    self.runtime.release(buffer)
                else:
                    self.runtime.quarantine(buffer, "native submission outcome is unknown")

    def _defer_source(self, operation, buffer, target: ReceiveTicket, remote: dict) -> None:
        self.runtime.quarantine(buffer, "request ended before native completion")

        def completed(status: str) -> None:
            try:
                self._schedule_finalize(target, remote, status)
            finally:
                try:
                    if buffer.registered and buffer.safe_to_release:
                        self.runtime.release(buffer)
                except BaseException as exc:
                    self._record_fatal(exc)

        self.runtime.defer(operation, completed)

    def _terminal_message(
        self, target: ReceiveTicket, remote: dict, status: str, operation: str
    ) -> dict:
        return {
            "op": operation,
            "backend": _TRANSPORT,
            "transfer_id": target.transfer_id,
            "locator": target.locator,
            "receiver_session": remote["session"],
            "source_session": self.runtime.info["session"],
            "nbytes": target.nbytes,
            "native_status": status,
        }

    @staticmethod
    def _check_receipt(reply: dict, target: ReceiveTicket) -> str:
        if (
            set(reply) != {"status", "transfer_id", "locator"}
            or reply["status"] not in ("complete", "retired", "quarantined")
            or reply["transfer_id"] != target.transfer_id
            or reply["locator"] != target.locator
        ):
            raise TransferError(str(reply.get("error", "incorrect Mooncake completion receipt")))
        return reply["status"]

    def _schedule_finalize(self, target: ReceiveTicket, remote: dict, status: str) -> None:
        key = (target.transfer_id, target.locator)
        with self._lock:
            if self._stop.is_set() or key in self._finalizing:
                return
            if not self._cleanup_slots.acquire(blocking=False):
                # No completion receipt is also safe: D retains its isolated allocation.
                return
            try:
                future = self._cleanup.submit(self._late_finalize, target, dict(remote), status)
            except RuntimeError:
                self._cleanup_slots.release()
                if not self._stop.is_set():
                    raise
                return
            self._finalizing.add(key)
            self._cleanup_futures.add(future)
            future.add_done_callback(lambda finished: self._finalize_done(finished, key))

    def _finalize_done(self, future: Future, key: tuple[str, str]) -> None:
        with self._lock:
            self._cleanup_futures.discard(future)
            self._finalizing.discard(key)
        self._cleanup_slots.release()
        if not future.cancelled():
            error = future.exception()
            if error is not None:
                self._record_fatal(error)

    def _handler_done(self, future: Future) -> None:
        with self._lock:
            self._handler_futures.discard(future)
        if not future.cancelled():
            error = future.exception()
            if error is not None:
                self._record_fatal(error)

    def _late_finalize(self, target: ReceiveTicket, remote: dict, status: str) -> None:
        # 此连接只传完成证明，不传 KV 字节；无法送达时 D 继续保留隔离区。
        for _ in range(2):
            if self._stop.is_set():
                return
            try:
                with self._connection(target.endpoint) as sock:
                    channel = Channel(sock, self.timeout, self._stop)
                    channel.send(self._terminal_message(target, remote, status, "finalize"))
                    self._check_receipt(channel.receive(), target)
                    return
            except (OSError, ValueError, TransferError):
                continue

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                sock, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    self._record_fatal(exc)
                return
            if not self._handler_slots.acquire(blocking=False):
                sock.close()
                continue
            try:
                with self._lock:
                    if self._stop.is_set():
                        self._handler_slots.release()
                        sock.close()
                        return
                    self._connections.add(sock)
                    future = self._handlers.submit(self._handle_connection, sock)
                    self._handler_futures.add(future)
                    future.add_done_callback(self._handler_done)
            except RuntimeError as exc:
                with self._lock:
                    self._connections.discard(sock)
                self._handler_slots.release()
                sock.close()
                if not self._stop.is_set():
                    self._record_fatal(exc)
                return

    def _handle_connection(self, sock: socket.socket) -> None:
        try:
            with sock:
                self._receive(sock)
        finally:
            with self._lock:
                self._connections.discard(sock)
            self._handler_slots.release()

    def _receive(self, sock: socket.socket) -> None:
        """Exchange GPU descriptors and completion; never receive KV bytes here."""
        channel = Channel(sock, self.timeout, self._stop)
        entry = None
        owns_receive = False
        try:
            header = channel.receive()
            if header.get("backend") != _TRANSPORT:
                raise TransferError("incompatible Mooncake GPU transfer backend")
            if header.get("op") == "finalize":
                channel.send(self._settle_receive(header))
                return
            self.check_available()
            if set(header) != {"op", "backend", "metadata", "ticket", "source_session"} or (
                header["op"] != "prepare"
            ):
                raise TransferError("incorrect Mooncake GPU transfer header")
            meta = KVTransferMetadata.from_wire(json.dumps(header["metadata"]).encode())
            target = ReceiveTicket.from_wire(json.dumps(header["ticket"]).encode())
            source_session = header["source_session"]
            if not isinstance(source_session, str):
                raise TransferError("incorrect Mooncake source session")
            endpoint_address(source_session)
            entry = self._lookup(target)
            with entry.lock:
                if entry.meta != meta or entry.cancelled or entry.exposed:
                    raise TransferError("incorrect or consumed receive ticket")
                owns_receive = True
                entry.sender_session = source_session
                # ready 发送可能只完成一部分，所以公布地址前先标记exposed。
                self.runtime.expose(entry.buffer)
                entry.exposed = True
                remote = {
                    "session": self.runtime.info["session"],
                    "address": entry.buffer.address,
                    "nbytes": entry.buffer.nbytes,
                }
                channel.send({"status": "ready", "backend": _TRANSPORT, "target": remote})
            terminal = channel.receive()
            if terminal.get("op") != "terminal":
                raise TransferError("expected native transfer completion")
            channel.send(self._settle_receive(terminal))
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            if isinstance(exc, TransferBackendUnavailable):
                self._record_fatal(exc)
            if entry is not None and owns_receive:
                try:
                    self.discard_receive(entry.target)
                except (RuntimeError, ValueError) as cleanup_error:
                    self._record_fatal(cleanup_error)
            try:
                channel.send({"status": "error", "error": str(exc)})
            except (OSError, ValueError):
                pass

    def _settle_receive(self, message: dict) -> dict:
        if set(message) != {
            "op",
            "backend",
            "transfer_id",
            "locator",
            "receiver_session",
            "source_session",
            "nbytes",
            "native_status",
        } or (
            message["op"] not in ("terminal", "finalize")
            or message["backend"] != _TRANSPORT
            or not isinstance(message["native_status"], str)
            or message["native_status"] not in _TERMINAL_STATES
            or type(message["nbytes"]) is not int
            or any(
                not isinstance(message[key], str)
                for key in ("transfer_id", "locator", "receiver_session", "source_session")
            )
        ):
            raise TransferError("incorrect native completion message")
        identity = (
            message["transfer_id"],
            message["locator"],
            message["receiver_session"],
            message["nbytes"],
            message["source_session"],
        )
        with self._lock:
            entry = self._receives.get(message["locator"])
            tombstone = self._tombstones.get(identity)
        status = message["native_status"]
        if entry is None:
            if tombstone is None or tombstone[0] != status:
                raise TransferError("unknown or stale native completion")
            result = tombstone[1]
        else:
            with entry.lock:
                if not entry.exposed or identity != self._identity(entry):
                    raise TransferError("native completion belongs to another receive")
                if entry.terminal is not None and entry.terminal != status:
                    raise TransferError("native completion status changed")
                if status == "completed":
                    if entry.terminal != status:
                        self.runtime.make_visible(entry.buffer)
                        self.runtime.mark_safe(entry.buffer)
                        entry.terminal = status
                    if entry.cancelled:
                        if entry.buffer.registered:
                            self.runtime.release(entry.buffer)
                        self._forget(entry, "retired")
                        result = "retired"
                    else:
                        entry.delivered = True
                        result = "complete"
                elif status == "not_submitted":
                    self.runtime.mark_safe(entry.buffer)
                    entry.terminal = status
                    entry.cancelled = True
                    entry.error = "sender did not submit the transfer"
                    if entry.buffer.registered:
                        self.runtime.release(entry.buffer)
                    self._forget(entry, "retired")
                    result = "retired"
                else:
                    # P本地排空不代表D已接纳的远端写入也都结束。
                    entry.terminal = status
                    entry.cancelled = True
                    entry.error = "sender failed; remote receive completion is unconfirmed"
                    self.runtime.quarantine(entry.buffer, entry.error)
                    result = "quarantined"
                entry.done.set()
        return {
            "status": result,
            "transfer_id": message["transfer_id"],
            "locator": message["locator"],
        }

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closing = True
            # Keep completion-only connections alive while rejecting new data work.
            # A peer's late success can still release a cancelled GPU registration.
            with self._lock:
                receives = list(self._receives.values())
            for entry in receives:
                try:
                    self.discard_receive(entry.target)
                except TransferError as exc:
                    self._record_fatal(exc)
            deadline = time.monotonic() + self.timeout
            while self._bound.is_set() and time.monotonic() < deadline:
                native = self.runtime.health()
                with self._lock:
                    finishing = bool(self._cleanup_futures)
                if native.get("registered_buffers", 0) == 0 and not finishing:
                    break
                time.sleep(0.01)
            self._stop.set()
            if self._listener is not None:
                self._listener.close()
            with self._lock:
                connections = list(self._connections)
                receives = list(self._receives.values())
            for sock in connections:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if self._thread is not None:
                self._thread.join(timeout=self.timeout + 1)
                if self._thread.is_alive():
                    raise TransferBackendUnavailable("Mooncake listener did not stop")
            with self._lock:
                pending = list(self._handler_futures | self._cleanup_futures)
            _, unfinished = wait(pending, timeout=self.timeout)
            self._handlers.shutdown(wait=False, cancel_futures=bool(unfinished))
            self._cleanup.shutdown(wait=False, cancel_futures=bool(unfinished))
            if unfinished:
                raise TransferBackendUnavailable("Mooncake control handlers did not stop")
            for entry in receives:
                try:
                    self.discard_receive(entry.target)
                except TransferError as exc:
                    self._record_fatal(exc)
            # 未知原生状态必须继续保留强引用；runtime.close会明确报告未排空。
            self.runtime.close()
            self._closed = True

    def __enter__(self) -> MooncakeTransferBackend:
        return self

    def __exit__(self, *_args) -> None:
        self.close()
