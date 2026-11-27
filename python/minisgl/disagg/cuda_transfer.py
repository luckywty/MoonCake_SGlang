"""CUDA ordering for GPUDirect RDMA. This module never copies KV to host memory.

The driver bindings are loaded only after the model Engine initialized CUDA.
RDMA completion and visibility to a CUDA consumer are separate conditions.
"""

from __future__ import annotations

import ctypes
import importlib
from contextlib import contextmanager

import torch

from .protocol import MOONCAKE_TRANSPORT_MODES
from .transfer import TransferBackendUnavailable, TransferError


class CudaTransfer:
    def __init__(self, device: torch.device, *, transfer_mode="gpu_direct"):
        if transfer_mode not in MOONCAKE_TRANSPORT_MODES:
            raise ValueError("unsupported Mooncake transfer mode")
        self.transfer_mode = transfer_mode
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise TransferBackendUnavailable(
                "Mooncake GPU transfer requires an indexed CUDA device"
            )
        if not torch.cuda.is_initialized():
            raise TransferBackendUnavailable(
                "bind Mooncake after the model Engine initializes CUDA"
            )
        try:
            self.driver = importlib.import_module("cuda.bindings.driver")
        except ImportError as exc:
            raise TransferBackendUnavailable(
                "GPU Mooncake requires cuda-bindings>=12.9; install the pd-mooncake extra"
            ) from exc
        with torch.cuda.device(self.device):
            self._call(self.driver.cuInit(0), "cuInit")
            self.context = self._one(self.driver.cuCtxGetCurrent(), "cuCtxGetCurrent")
            if not int(self.context):
                raise TransferBackendUnavailable("the model's CUDA context is not current")
            self.cuda_device = self._one(self.driver.cuDeviceGet(self.device.index), "cuDeviceGet")
            if transfer_mode == "host_dram":
                self.ordering = self.flush_options = 0
                self.needs_flush = False
                return  # No GDR capability query, device-pointer check or GDR flush required.
            attrs = self.driver.CUdevice_attribute
            supported = self._attribute(attrs.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED)
            if not supported:
                raise TransferBackendUnavailable("this CUDA device does not support GPUDirect RDMA")
            self.ordering = self._attribute(
                attrs.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING
            )
            self.flush_options = self._attribute(
                attrs.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS
            )
        self.scope = (
            self.driver.CUflushGPUDirectRDMAWritesScope.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER
        )
        self.needs_flush = self.ordering < int(self.scope)
        host_flush = int(
            self.driver.CUflushGPUDirectRDMAWritesOptions.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST
        )
        if self.needs_flush and not (self.flush_options & host_flush):
            raise TransferBackendUnavailable(
                "GPUDirect writes are not naturally ordered and host-side RDMA flush is unavailable"
            )

    def _call(self, result: tuple, operation: str) -> tuple:
        if not isinstance(result, tuple) or not result:
            raise TransferError(f"CUDA {operation} returned an invalid result")
        if result[0] != self.driver.CUresult.CUDA_SUCCESS:
            raise TransferError(f"CUDA {operation} failed: {result[0]!s}")
        return result[1:]

    def _one(self, result: tuple, operation: str):
        values = self._call(result, operation)
        if len(values) != 1:
            raise TransferError(f"CUDA {operation} returned an invalid value count")
        return values[0]

    def _attribute(self, attribute) -> int:
        return int(
            self._one(
                self.driver.cuDeviceGetAttribute(attribute, self.cuda_device),
                "cuDeviceGetAttribute",
            )
        )

    @contextmanager
    def activate(self):
        """Use the model's context also from receiver and cleanup threads."""
        with torch.cuda.device(self.device):
            previous = self._one(self.driver.cuCtxGetCurrent(), "cuCtxGetCurrent")
            changed = int(previous) != int(self.context)
            if changed:
                self._call(self.driver.cuCtxSetCurrent(self.context), "cuCtxSetCurrent")
            try:
                yield
            finally:
                if changed:
                    self._call(self.driver.cuCtxSetCurrent(previous), "cuCtxSetCurrent")

    def prepare_buffer(self, tensor: torch.Tensor) -> None:
        """Check real device memory, then enable CUDA's registration consistency mode."""
        if self.transfer_mode == "host_dram":
            if (
                tensor.device.type != "cpu"
                or tensor.dtype != torch.uint8
                or not tensor.is_contiguous()
                or not tensor.numel()
                or not tensor.is_pinned()
                or tensor.data_ptr() % 256
            ):
                raise TransferError(
                    "host_dram requires nonempty 256-byte-aligned pinned CPU uint8 storage"
                )
            return
        if tensor.device != self.device or not tensor.is_contiguous() or not tensor.numel():
            raise TransferError(
                "registered KV must be a nonempty contiguous tensor on the Worker GPU"
            )
        address = tensor.data_ptr()
        attrs = self.driver.CUpointer_attribute
        with self.activate():
            memory_type = self._one(
                self.driver.cuPointerGetAttribute(attrs.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, address),
                "cuPointerGetAttribute(memory type)",
            )
            if int(memory_type) != int(self.driver.CUmemorytype.CU_MEMORYTYPE_DEVICE):
                raise TransferError("the KV transfer pointer is not CUDA device memory")
            managed = self._one(
                self.driver.cuPointerGetAttribute(attrs.CU_POINTER_ATTRIBUTE_IS_MANAGED, address),
                "cuPointerGetAttribute(managed)",
            )
            if managed:
                raise TransferError("managed/unified memory is not accepted for GPU KV transfer")
            ordinal = self._one(
                self.driver.cuPointerGetAttribute(
                    attrs.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, address
                ),
                "cuPointerGetAttribute(device ordinal)",
            )
            if int(ordinal) != self.device.index:
                raise TransferError("the KV transfer allocation belongs to a different CUDA device")
            enabled = ctypes.c_uint(1)
            self._call(
                self.driver.cuPointerSetAttribute(
                    ctypes.addressof(enabled), attrs.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, address
                ),
                "cuPointerSetAttribute(SYNC_MEMOPS)",
            )

    def make_visible(self) -> None:
        """Run only after remote completion, before enqueueing the CUDA consumer."""
        if self.transfer_mode == "host_dram":
            return
        with self.activate():
            if self.needs_flush:
                self._call(
                    self.driver.cuFlushGPUDirectRDMAWrites(
                        self.driver.CUflushGPUDirectRDMAWritesTarget.CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                        self.scope,
                    ),
                    "cuFlushGPUDirectRDMAWrites",
                )
            # The first version uses an explicit context boundary. No GPU consumer
            # is launched before this point, including checksum/install kernels.
            self._call(self.driver.cuCtxSynchronize(), "cuCtxSynchronize")

    @property
    def info(self) -> dict:
        return {
            "device": str(self.device),
            "gpudirect_rdma": self.transfer_mode == "gpu_direct",
            "transfer_mode": self.transfer_mode,
            "rdma_writes_ordering": self.ordering,
            "rdma_flush_options": self.flush_options,
            "host_flush_required": self.needs_flush,
        }
