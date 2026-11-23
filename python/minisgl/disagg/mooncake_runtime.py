"""Worker-owned TE and CUDA registrations; no helper or host KV staging.

Cancelling a business request does not cancel DMA. Keep Tensor, registration,
ticket and engine references until native completion is proved. An unresolved
close requires the Worker's process-exit boundary, never an unsafe buffer free.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from .cluster_config import validate_mooncake_port
from .control import endpoint_address, endpoint_name
from .cuda_transfer import CudaTransfer
from .transfer import TransferBackendUnavailable, TransferError

# Unknown tickets must not lose their last reference: their native destructor
# waits for physical completion with the GIL held. The launcher handles a failed
# close with a nonzero process exit while this registry retains all allocations.
_LIVE_RUNTIMES: dict[int, MooncakeRuntime] = {}
_LIVE_RUNTIMES_LOCK = threading.Lock()


@dataclass(eq=False)
class RegisteredBuffer:
    key: str
    tensor: torch.Tensor | None
    address: int
    nbytes: int
    safe_to_release: bool = True
    registered: bool = False
    exposed: bool = False
    quarantine_since: float | None = None
    last_error: str | None = None
    release_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(eq=False)
class TransferOperation:
    buffer: RegisteredBuffer
    ticket: object | None = None
    status: str = "unknown"
    deferred: bool = False
    callback: Callable[[str], None] | None = None
    callback_done: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class MooncakeRuntime:
    def __init__(
        self,
        local_host: str,
        *,
        timeout: float = 30.0,
        max_transfer_bytes: int = 1024**3,
        max_registered_bytes: int | None = None,
        max_pending_transfers: int = 32,
        max_quarantine_seconds: float = 300.0,
        device_name: str = "",
        engine_module: str = "mooncake.engine",
        rpc_port: int | None = None,
    ):
        if (
            not isinstance(local_host, str)
            or not local_host.strip()
            or local_host.strip() in ("0.0.0.0", "::")
        ):
            raise ValueError("Mooncake requires a reachable local host, not a wildcard")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Mooncake timeout must be finite and positive")
        if rpc_port is not None:
            validate_mooncake_port(rpc_port)
        self.rpc_port = rpc_port
        if type(max_transfer_bytes) is not int or max_transfer_bytes <= 0:
            raise ValueError("max_transfer_bytes must be positive")
        if max_registered_bytes is None:
            max_registered_bytes = max_transfer_bytes * 2
        if type(max_registered_bytes) is not int or max_registered_bytes < max_transfer_bytes:
            raise ValueError("registered GPU budget must cover one maximum-sized transfer")
        if type(max_pending_transfers) is not int or max_pending_transfers < 1:
            raise ValueError("max_pending_transfers must be positive")
        if not math.isfinite(max_quarantine_seconds) or max_quarantine_seconds <= 0:
            raise ValueError("max_quarantine_seconds must be finite and positive")
        if not isinstance(device_name, str):
            raise ValueError("Mooncake device filter must be a comma-separated HCA list")
        filters = [name.strip() for name in device_name.split(",")] if device_name else []
        if any(not name or "/" in name or name in (".", "..") for name in filters):
            raise ValueError("incorrect Mooncake HCA device filter")
        self.local_host = local_host.strip()
        self.timeout = timeout
        self.max_transfer_bytes = max_transfer_bytes
        self.max_registered_bytes = max_registered_bytes
        self.max_pending_transfers = max_pending_transfers
        self.max_quarantine_seconds = max_quarantine_seconds
        self.device_name = ",".join(filters)
        self.engine_module = engine_module
        self.owner_pid = os.getpid()
        self.device: torch.device | None = None
        self.engine = None
        self.cuda: CudaTransfer | None = None
        self.module = None
        self.session: str | None = None
        self._info: dict = {}
        self._lock = threading.RLock()
        self._buffers: dict[str, RegisteredBuffer] = {}
        self._operations: dict[str, TransferOperation] = {}
        self._registered_bytes = 0
        self._closing = False
        self._closed = False
        self._retire_reason: str | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._reaper: threading.Thread | None = None

    def _check_pid(self) -> None:
        if os.getpid() != self.owner_pid:
            raise TransferBackendUnavailable(
                "Mooncake runtimes cannot be shared across forked Workers"
            )

    @staticmethod
    def _configure_rdma(device_name: str) -> list[str]:
        incompatible = (
            "MC_FORCE_TCP",
            "MC_USE_TENT",
            "MC_USE_TEV1",
            "MC_FORCE_SHM",
            "MC_FORCE_MNNVL",
            "MC_INTRANODE_NVLINK",
            "MC_FORCE_MUSA",
            "MC_ENABLE_HYLINK",
            "MC_USE_RDMA_TWOSIDED",
            "USE_BAREX",
        )
        present = [name for name in incompatible if name in os.environ]
        if present:
            raise TransferBackendUnavailable(
                "GPU Mooncake requires classic one-sided RDMA; unset " + ", ".join(present)
            )
        root = Path("/sys/class/infiniband")
        devices = sorted(path.name for path in root.iterdir()) if root.is_dir() else []
        selected = device_name.split(",") if device_name else devices
        if not selected or any(name not in devices for name in selected):
            raise TransferBackendUnavailable("the selected RDMA HCA devices are not available")
        # Classic auto-discovery otherwise installs TCP on a host without HCAs.
        os.environ["MC_FORCE_HCA"] = "1"
        return selected

    @staticmethod
    def _native_ok(operation: str, result) -> None:
        if type(result) is not int or result != 0:
            raise TransferError(f"Mooncake {operation} failed (return={result!r})")

    def bind_device(self, device: torch.device) -> None:
        """Called after Scheduler/Engine: constructor must not initialize CUDA."""
        self._check_pid()
        with self._lock:
            if self.engine is not None or self._closing:
                raise TransferBackendUnavailable("Mooncake runtime is already bound or closing")
            device = torch.device(device)
            if device.type != "cuda" or device.index is None:
                raise TransferBackendUnavailable("GPU Mooncake needs a CUDA Worker")
            devices = self._configure_rdma(self.device_name)
            if self.rpc_port is not None:
                # Classic P2PHANDSHAKE chooses within this range, including
                # when initialize(local_host:port) would ignore a requested port.
                os.environ["MC_MIN_RPC_PORT"] = os.environ["MC_MAX_RPC_PORT"] = str(self.rpc_port)
            cuda = CudaTransfer(device)
            try:
                module = importlib.import_module(self.engine_module)
            except ImportError as exc:
                raise TransferBackendUnavailable("mooncake.engine cannot be loaded") from exc
            if getattr(module, "SUPPORT_CUDA", False) is not True:
                raise TransferBackendUnavailable("the Mooncake extension lacks CUDA support")
            engine_type = getattr(module, "TransferEngine", None)
            methods = (
                "initialize",
                "get_rpc_port",
                "register_memory",
                "unregister_memory",
                "submit_scatter_write",
            )
            if engine_type is None or any(
                not callable(getattr(engine_type, name, None)) for name in methods
            ):
                raise TransferBackendUnavailable("Mooncake lacks the required GPU scatter API")
            states = getattr(module, "ScatterTransferCompletionStatus", None)
            if states is None or any(
                not hasattr(states, name)
                for name in ("COMPLETED", "FAILED_DRAINED", "COMPLETION_UNKNOWN")
            ):
                raise TransferBackendUnavailable(
                    "Mooncake lacks explicit scatter completion states"
                )
            self.module, self.cuda, self.device = module, cuda, device
            with cuda.activate():
                self.engine = engine_type()
                with _LIVE_RUNTIMES_LOCK:
                    _LIVE_RUNTIMES[id(self)] = self
                try:
                    self._native_ok(
                        "initialize",
                        self.engine.initialize(
                            self.local_host, "P2PHANDSHAKE", "rdma", self.device_name
                        ),
                    )
                    port = self.engine.get_rpc_port()
                    if type(port) is not int or not 0 < port < 65536:
                        raise TransferError("Mooncake returned an invalid RPC port")
                    if self.rpc_port is not None and port != self.rpc_port:
                        raise TransferError("Mooncake ignored its configured RPC port range")
                    self.session = endpoint_name(self.local_host, port)
                except Exception as exc:
                    self._retire_reason = f"Mooncake initialization failed: {exc}"
                    raise TransferBackendUnavailable(self._retire_reason) from exc
            try:
                version = importlib.metadata.version("mooncake-transfer-engine")
            except importlib.metadata.PackageNotFoundError:
                version = "source-build"
            self._info = {
                "owner_pid": self.owner_pid,
                "session": self.session,
                "transport": "rdma",
                "memory": "cuda",
                "helpers": False,
                "engine_file": str(getattr(module, "__file__", "unknown")),
                "package_version": version,
                "hca_devices": devices,
                "cuda": cuda.info,
            }
            self._reaper = threading.Thread(
                target=self._reap_loop, name="pd-mooncake-reaper", daemon=True
            )
            self._reaper.start()

    @property
    def info(self) -> dict:
        with self._lock:
            return {
                **self._info,
                "owner_pid": self.owner_pid,
                "initialized": self.session is not None and not self._closed,
            }

    def _age_check_locked(self) -> None:
        now = time.monotonic()
        if any(
            buffer.quarantine_since is not None
            and now - buffer.quarantine_since >= self.max_quarantine_seconds
            for buffer in self._buffers.values()
        ):
            self._retire_reason = "an unresolved GPU registration exceeded its quarantine deadline"

    def check_available(self) -> None:
        self._check_pid()
        with self._lock:
            self._age_check_locked()
            if self._retire_reason:
                raise TransferBackendUnavailable(self._retire_reason)
            if self._closing or self._closed or self.session is None:
                raise TransferBackendUnavailable("the GPU Mooncake runtime is not ready")

    def check_capacity(self, nbytes: int, *, pool: bool = False) -> None:
        self.check_available()
        limit = self.max_registered_bytes if pool else self.max_transfer_bytes
        if type(nbytes) is not int or not 0 < nbytes <= limit:
            raise TransferError("KV payload exceeds the configured GPU transfer limit")
        with self._lock:
            if (
                self._registered_bytes + nbytes > self.max_registered_bytes
                or len(self._buffers) >= self.max_pending_transfers
            ):
                if any(buffer.quarantine_since is not None for buffer in self._buffers.values()):
                    self._retire_reason = (
                        "unresolved GPU registrations exhausted the transfer budget"
                    )
                    raise TransferBackendUnavailable(self._retire_reason)
                raise TransferError("the registered GPU transfer budget is busy")

    def _owned_locked(self, buffer: RegisteredBuffer) -> None:
        if self._buffers.get(buffer.key) is not buffer:
            raise TransferError("unknown or released GPU registration")

    def register_tensor(self, tensor: torch.Tensor, key: str, *, pool=False) -> RegisteredBuffer:
        if not isinstance(tensor, torch.Tensor) or tensor.device != self.device:
            raise TransferError("KV payload must be a tensor on the Worker's CUDA device")
        if not isinstance(key, str) or not key:
            raise TransferError("GPU registrations need an attempt-scoped key")
        if not tensor.is_contiguous() or tensor.dtype not in (
            torch.uint8,
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise TransferError("GPU KV registration requires contiguous, unquantized storage")
        nbytes = tensor.numel() * tensor.element_size()
        address = tensor.data_ptr()
        with self._lock:
            # A startup pool can hold several individually bounded transfers.
            self.check_capacity(nbytes, pool=pool)
            if key in self._buffers:
                raise TransferError("GPU registration key is already in use")
            if any(
                address < entry.address + entry.nbytes and entry.address < address + nbytes
                for entry in self._buffers.values()
            ):
                raise TransferError("GPU allocation overlaps an existing registration")
            # Registration itself is native work. close() must retain this
            # Tensor even before register_memory() has returned.
            buffer = RegisteredBuffer(key, tensor, address, nbytes, safe_to_release=False)
            self._buffers[key] = buffer
            self._registered_bytes += nbytes
        assert self.cuda is not None
        with buffer.release_lock:
            try:
                # CUDA's allocator can recycle a block according to stream
                # ordering even for torch.empty(). External RDMA does not obey
                # that stream, so finish prior allocation/producer work first.
                with self.cuda.activate():
                    torch.cuda.current_stream(self.device).synchronize()
                self.cuda.prepare_buffer(tensor)
            except Exception:
                with self._lock:
                    self._buffers.pop(key)
                    self._registered_bytes -= nbytes
                    buffer.tensor = None
                raise
            # Preserve the allocation on an uncertain partial native failure.
            buffer.registered = True
            try:
                with self.cuda.activate():
                    self._native_ok(
                        "register_memory",
                        self.engine.register_memory(
                            address, nbytes, f"cuda:{self.device.index}", True
                        ),
                    )
            except Exception as exc:
                self.quarantine(buffer, f"memory registration failed: {exc}")
                raise TransferError(f"GPU memory registration failed: {exc}") from exc
            with self._lock:
                buffer.safe_to_release = True
        return buffer

    def allocate_buffer(self, nbytes: int, key: str, *, pool=False) -> RegisteredBuffer:
        self.check_capacity(nbytes, pool=pool)
        assert self.cuda is not None
        with self.cuda.activate():
            tensor = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        return self.register_tensor(tensor, key, pool=pool)

    def expose(self, buffer: RegisteredBuffer) -> None:
        with self._lock:
            self._owned_locked(buffer)
            buffer.exposed = True
            buffer.safe_to_release = False

    def mark_safe(self, buffer: RegisteredBuffer) -> None:
        with self._lock:
            self._owned_locked(buffer)
            buffer.safe_to_release = True
            buffer.quarantine_since = None
            buffer.last_error = None

    def update_pool_safety(self, buffer, *, safe: bool, quarantine_since):
        """Atomically aggregate slice ownership without clearing another lease's quarantine."""
        with self._lock:
            self._owned_locked(buffer)
            buffer.safe_to_release = safe
            buffer.quarantine_since = quarantine_since
            buffer.last_error = (
                "receive slice awaits native completion" if quarantine_since is not None else None
            )

    def quarantine(self, buffer: RegisteredBuffer, reason: str) -> None:
        with self._lock:
            self._owned_locked(buffer)
            if buffer.quarantine_since is None:
                buffer.quarantine_since = time.monotonic()
            buffer.last_error = str(reason)

    def make_visible(self, buffer: RegisteredBuffer) -> None:
        """Caller must have validated the peer's successful native completion.

        Keep the allocation unsafe until this visibility step succeeds. That
        permits the receiver to retain it if CUDA reports an ordering error.
        """
        with self._lock:
            self._owned_locked(buffer)
        assert self.cuda is not None
        self.cuda.make_visible()

    def release(self, buffer: RegisteredBuffer) -> None:
        with buffer.release_lock:
            with self._lock:
                if self._buffers.get(buffer.key) is not buffer:
                    return
                if not buffer.safe_to_release:
                    self.quarantine(buffer, "release requested before native completion was proved")
                    raise TransferError("cannot release GPU memory with unresolved native access")
            assert self.cuda is not None
            try:
                if buffer.registered:
                    with self.cuda.activate():
                        self._native_ok(
                            "unregister_memory", self.engine.unregister_memory(buffer.address)
                        )
            except Exception as exc:
                self.quarantine(buffer, f"memory unregistration failed: {exc}")
                raise TransferError(f"GPU memory unregistration failed: {exc}") from exc
            with self._lock:
                buffer.registered = False
                self._buffers.pop(buffer.key)
                self._registered_bytes -= buffer.nbytes
                operation = self._operations.pop(buffer.key, None)
                if operation is not None:
                    operation.ticket = None  # terminal native operation only
                buffer.tensor = None

    def recycle_operation(self, operation: TransferOperation) -> None:
        """Retire a drained operation while retaining its startup registration."""
        with operation.lock, self._lock:
            self._owned_locked(operation.buffer)
            if (
                operation.status not in ("completed", "failed_drained")
                or operation.ticket is None
                or not operation.ticket.drained
                or not operation.buffer.safe_to_release
            ):
                raise TransferError("cannot reuse a staging slot before native drain")
            if self._operations.get(operation.buffer.key) is not operation:
                raise TransferError("stale staging operation")
            self._operations.pop(operation.buffer.key)
            operation.ticket = None

    def retire(self, reason: str) -> None:
        with self._lock:
            self._retire_reason = str(reason)

    def submit_write(
        self, buffer: RegisteredBuffer, remote: dict, *, nbytes: int | None = None
    ) -> TransferOperation:
        self.check_available()
        if not isinstance(remote, dict) or set(remote) != {"session", "address", "nbytes"}:
            raise TransferError("incorrect Mooncake GPU target descriptor")
        endpoint_address(remote["session"])
        if type(remote["address"]) is not int or remote["address"] <= 0:
            raise TransferError("incorrect remote GPU address")
        nbytes = buffer.nbytes if nbytes is None else nbytes
        if type(nbytes) is not int or not 0 < nbytes <= buffer.nbytes:
            raise TransferError("write exceeds its registered staging slot")
        if type(remote["nbytes"]) is not int or remote["nbytes"] != nbytes:
            raise TransferError("remote GPU capacity does not match the payload")
        with self._lock:
            self._owned_locked(buffer)
            if buffer.key in self._operations or not buffer.safe_to_release:
                raise TransferError("GPU source already has an in-flight operation")
            operation = TransferOperation(buffer)
            self._operations[buffer.key] = operation
            buffer.safe_to_release = False
        try:
            assert self.cuda is not None
            with self.cuda.activate():
                operation.ticket = self.engine.submit_scatter_write(
                    remote["session"],
                    [buffer.address],
                    [nbytes],
                    [remote["address"]],
                    [remote["nbytes"]],
                    [[0]],
                    [[0]],
                    [[nbytes]],
                )
            if operation.ticket is None or not callable(getattr(operation.ticket, "drain", None)):
                raise TransferError("Mooncake did not return a drainable native operation")
        except Exception as exc:
            self.quarantine(buffer, f"native submission completion is unknown: {exc}")
            raise TransferError(f"Mooncake GPU submission failed: {exc}") from exc
        return operation

    def poll(self, operation: TransferOperation, timeout_ms: int = 50) -> str:
        if type(timeout_ms) is not int or not 0 <= timeout_ms <= 1000:
            raise ValueError("native drain polling interval must be 0..1000 milliseconds")
        with operation.lock:
            if operation.status != "unknown":
                return operation.status
            if operation.ticket is None:
                return "unknown"
            try:
                assert self.cuda is not None
                with self.cuda.activate():
                    status = operation.ticket.drain(timeout_ms)
                states = self.module.ScatterTransferCompletionStatus
                if status == states.COMPLETED:
                    result = "completed"
                elif status == states.FAILED_DRAINED:
                    result = "failed_drained"
                elif status == states.COMPLETION_UNKNOWN:
                    return "unknown"
                else:
                    raise TransferError(f"incorrect native completion state: {status!r}")
                if not operation.ticket.drained:
                    raise TransferError("a terminal native result was not physically drained")
                operation.status = result
                self.mark_safe(operation.buffer)
                return result
            except Exception as exc:
                self.quarantine(operation.buffer, f"native completion query failed: {exc}")
                raise TransferError(f"cannot determine Mooncake completion: {exc}") from exc

    def defer(self, operation: TransferOperation, on_terminal: Callable[[str], None]) -> None:
        if not callable(on_terminal):
            raise ValueError("a deferred operation needs a completion callback")
        with self._lock:
            self._owned_locked(operation.buffer)
            if operation.callback is not None:
                raise TransferError("native operation already has a deferred consumer")
            operation.callback = on_terminal
            operation.deferred = True
            if operation.status == "unknown":
                self.quarantine(operation.buffer, "business request ended before native completion")
        self._wake.set()

    def _reap_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                operations = [entry for entry in self._operations.values() if entry.deferred]
                self._age_check_locked()
            for operation in operations:
                if self._stop.is_set():
                    break
                try:
                    status = self.poll(operation, timeout_ms=0)
                except (RuntimeError, ValueError):
                    continue
                if status == "unknown":
                    continue
                with self._lock:
                    if operation.callback_done:
                        continue
                    operation.callback_done = True
                    operation.deferred = False
                    callback = operation.callback
                try:
                    if callback is not None:
                        callback(status)
                except Exception as exc:
                    with self._lock:
                        if self._buffers.get(operation.buffer.key) is operation.buffer:
                            self.quarantine(operation.buffer, f"terminal receipt failed: {exc}")
            self._wake.wait(0.05)
            self._wake.clear()

    def health(self) -> dict:
        with self._lock:
            self._age_check_locked()
            quarantined = [
                entry for entry in self._buffers.values() if entry.quarantine_since is not None
            ]
            oldest = max(
                (time.monotonic() - entry.quarantine_since for entry in quarantined), default=0.0
            )
            return {
                "state": "unavailable"
                if self._retire_reason or self._closing
                else ("degraded" if quarantined else ("ready" if self.session else "unbound")),
                "reason": self._retire_reason,
                "registered_bytes": self._registered_bytes,
                "registered_buffers": len(self._buffers),
                "pending_operations": sum(
                    entry.status == "unknown" for entry in self._operations.values()
                ),
                "quarantined_bytes": sum(entry.nbytes for entry in quarantined),
                "quarantined_buffers": len(quarantined),
                "oldest_quarantine_seconds": oldest,
                "max_registered_bytes": self.max_registered_bytes,
                "max_pending_transfers": self.max_pending_transfers,
                "runtime": self.info,
            }

    def close(self) -> None:
        self._check_pid()
        with self._lock:
            if self._closed:
                return
            self._closing = True
        deadline = time.monotonic() + self.timeout
        while True:
            with self._lock:
                operations = [
                    entry for entry in self._operations.values() if entry.status == "unknown"
                ]
            if not operations or time.monotonic() >= deadline:
                break
            for operation in operations:
                try:
                    self.poll(operation, timeout_ms=0)
                except (RuntimeError, ValueError):
                    pass
            self._wake.set()
            self._stop.wait(0.01)
        self._stop.set()
        self._wake.set()
        if self._reaper is not None and self._reaper is not threading.current_thread():
            self._reaper.join(timeout=self.timeout)
        with self._lock:
            buffers = list(self._buffers.values())
        for buffer in buffers:
            if buffer.safe_to_release:
                try:
                    self.release(buffer)
                except TransferError:
                    pass
        with self._lock:
            if self._buffers or (self._reaper is not None and self._reaper.is_alive()):
                self._retire_reason = (
                    "Mooncake close could not drain every GPU registration; the Worker must exit"
                )
                raise TransferBackendUnavailable(self._retire_reason)
            self.engine = None
            self.module = None
            self._closed = True
            with _LIVE_RUNTIMES_LOCK:
                _LIVE_RUNTIMES.pop(id(self), None)
