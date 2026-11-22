"""Synchronous P/D transfer contract and a byte-exact, in-process reference backend."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable, Protocol, TypeVar
from uuid import uuid4

from .protocol import KVLayout, KVTransferMetadata, ReceiveTicket

Source = TypeVar("Source", contravariant=True)
Destination = TypeVar("Destination", contravariant=True)


class KVTransferBackend(Protocol[Source, Destination]):
    '''规定所有传输后端必须提供的接口'''
    def prepare_receive(
        self, meta: KVTransferMetadata, local_layout: KVLayout, destination: Destination
    ) -> ReceiveTicket:
        """D validates compatibility and registers its local, exclusively owned destination."""
        ...

    def send(
        self,
        meta: KVTransferMetadata,
        source: Source,
        target: ReceiveTicket,
        *,
        cancel_check: Callable[[], None] | None = None,
    ) -> None:
        """P pushes exactly the specified bytes. Success means the transfer has completed."""
        ...

    def finish_receive(self, target: ReceiveTicket) -> None:
        """D verifies transfer completion and releases the registration before consuming the data."""
        ...

    def discard_receive(self, target: ReceiveTicket) -> None:
        """Revoke delivery; retain private native buffers until late I/O has ended.

        A native backend may keep its cancelled allocation registered. It must never
        reassign that allocation to a new request while the old I/O can still use it.
        """
        ...


class TransferError(RuntimeError):
    """A transfer failed validation or completion; the caller must not admit it for decode."""


class TransferBackendUnavailable(TransferError):
    """The backend cannot safely accept work; its Worker must stop serving.

    Unknown native operations must retain their buffers and operation handles until
    process exit. Catching this as an ordinary per-request error is not sufficient.
    """


@dataclass
class _Reservation:
    meta: KVTransferMetadata
    target: ReceiveTicket
    destination: memoryview
    delivered: bool = False


class MockFabric:
    """One instance per test transport, shared explicitly by the two mock endpoints."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._reservations: dict[tuple[str, str], _Reservation] = {}


class MockTransferBackend:
    """真正实现模拟传输;Host bytes only. A ticket contains neither a Python object nor the receiver's table index."""

    def __init__(self, endpoint: str, fabric: MockFabric) -> None:
        '''创建传输端点,模拟 Worker 的数据通信接口'''
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("endpoint must be a nonempty string")
        self.endpoint = endpoint
        self._fabric = fabric

    def prepare_receive(
        self, meta: KVTransferMetadata, local_layout: KVLayout, destination: bytearray
    ) -> ReceiveTicket:
        #准备接受kv
        '''
        meta：Prefill 告诉 Decode 的 KV 传输描述。
        local_layout：Decode 本地允许接收的 KV Layout。
        destination：Decode 已经分配好的目标 Buffer。
        '''
        if meta.layout != local_layout:#检查布局一致性
            raise TransferError("incompatible KV layout or model")
        if not isinstance(destination, bytearray):#检查类型一致
            raise TransferError("mock destination must be a bytearray")
        target = ReceiveTicket(meta.transfer_id, self.endpoint, str(uuid4()), meta.nbytes)
        key = (self.endpoint, meta.transfer_id)#注册目标 Buffer
        with self._fabric._lock:
            reservations = self._fabric._reservations
            if key in reservations or any(
                r.destination.obj is destination for r in reservations.values()
            ):
                '''同一个 endpoint + transfer_id 已经被注册。
                    同一个 bytearray 已经被其他接收任务占用。'''
                raise TransferError("attempt or destination already reserved")
            view = memoryview(destination)
            if view.nbytes != target.nbytes:
                view.release()
                raise TransferError("incorrect destination size")
            reservations[key] = _Reservation(meta, target, view)
        return target

    def _lookup(self, target: ReceiveTicket) -> _Reservation:
        # The caller holds the fabric lock.
        entry = self._fabric._reservations.get((target.endpoint, target.transfer_id))
        if entry is None or entry.target != target:
            raise TransferError("unknown or stale receive ticket")
        return entry

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
        if not isinstance(source, (bytes, bytearray)):
            raise TransferError("mock source must be bytes or bytearray")
        with self._fabric._lock, memoryview(source) as view:
            entry = self._lookup(target)
            if meta != entry.meta or view.nbytes != target.nbytes:
                raise TransferError("source metadata or length mismatch")
            if entry.delivered:
                raise TransferError("ticket already received a payload")
            entry.destination[:] = view
            entry.delivered = True

    def _release(self, target: ReceiveTicket, *, discard: bool) -> None:
        if target.endpoint != self.endpoint:
            raise TransferError("only the receiver can release its ticket")
        key = (self.endpoint, target.transfer_id)
        with self._fabric._lock:
            entry = self._fabric._reservations.get(key)
            if entry is None or entry.target != target:
                if discard:
                    return
                raise TransferError("unknown or stale receive ticket")
            if not discard and not entry.delivered:
                raise TransferError("transfer is incomplete")
            entry.destination.release()
            del self._fabric._reservations[key]

    def finish_receive(self, target: ReceiveTicket) -> None:
        self._release(target, discard=False)

    def discard_receive(self, target: ReceiveTicket) -> None:
        self._release(target, discard=True)
