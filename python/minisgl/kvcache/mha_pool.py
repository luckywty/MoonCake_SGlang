from __future__ import annotations

import sys

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._page_size = page_size
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        # Installed only for the final prompt chunk by the split P scheduler.
        self.layer_ready = None

    @property
    def page_size(self) -> int:
        return self._page_size

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )
        if self.layer_ready is not None:
            self.layer_ready(layer_id)

    @torch.inference_mode()
    def export_layers_into(
        self,
        destination: torch.Tensor,
        slots: torch.Tensor,
        start: int,
        end: int,
        *,
        num_pool_pages: int,
        trusted_indices: torch.Tensor | None = None,
    ) -> None:
        """Gather into a preallocated layer-major staging slot, without a KV temporary.

        Called on the transfer stream, after it waits for the producer event.
        Index validation can synchronize metadata here; it never runs in the
        model's layer-ready callback or copies the KV payload to the CPU.
        """
        if not 0 <= start < end <= self.num_layers:
            raise ValueError("invalid layer range")
        indices = (
            self._transfer_indices(slots, num_pool_pages)
            if trusted_indices is None
            else trusted_indices
        )
        if (
            indices.device != self.device
            or indices.dtype != torch.int64
            or len(indices) != len(slots)
        ):
            raise ValueError("invalid trusted staging indices")
        heads, dim = self._storage_shape[1:]
        shape = (end - start, 2, len(slots), heads, dim)
        nbytes = (end - start) * 2 * len(slots) * heads * dim * self.dtype.itemsize
        if (
            destination.device != self.device
            or destination.dtype != torch.uint8
            or destination.ndim != 1
            or not destination.is_contiguous()
            or destination.numel() != nbytes
        ):
            raise ValueError("incorrect layer staging region")
        typed = destination.view(self.dtype).view(shape)
        for i, layer in enumerate(range(start, end)):
            source = self._kv_buffer[:, layer].flatten(1, 2)
            torch.index_select(source, 1, indices, out=typed[i])

    @torch.inference_mode()
    def import_layered_tensor(
        self, packed: torch.Tensor, slots: torch.Tensor, *, num_pool_pages: int
    ) -> None:
        """Install a completed [layer, K/V, token, head, dim] receive region."""
        indices = self._transfer_indices(slots, num_pool_pages)
        heads, dim = self._storage_shape[1:]
        shape = (self.num_layers, 2, len(slots), heads, dim)
        expected = 2 * self.num_layers * len(slots) * heads * dim * self.dtype.itemsize
        if (
            packed.device != self.device
            or packed.dtype != torch.uint8
            or packed.ndim != 1
            or not packed.is_contiguous()
            or packed.numel() != expected
        ):
            raise ValueError("incorrect layered receive region")
        typed = packed.view(self.dtype).view(shape)
        for layer in range(self.num_layers):
            self._kv_buffer[:, layer].flatten(1, 2).index_copy_(1, indices, typed[layer])

    @torch.inference_mode()
    def import_host_layered_tensor(self, packed, indices, *, workspace, layers_per_chunk):
        """Pinned receive region -> fixed GPU group buffer -> private KV, on install_stream.

        Indices are scheduler-owned and copied into a startup int64 workspace.
        No GPU-to-host metadata validation or allocation is allowed on this path.
        """
        if (
            packed.device.type != "cpu"
            or packed.dtype != torch.uint8
            or not packed.is_contiguous()
            or indices.device != self.device
            or (self.device.type == "cuda" and not packed.is_pinned())
            or indices.dtype != torch.int64
            or indices.ndim != 1
            or workspace.device != self.device
            or workspace.dtype != torch.uint8
        ):
            raise ValueError("invalid host layered staging or scheduler indices")
        heads, dim = self._storage_shape[1:]
        layer_bytes = 2 * len(indices) * heads * dim * self.dtype.itemsize
        if packed.numel() != self.num_layers * layer_bytes:
            raise ValueError("incorrect host layered receive length")
        for start in range(0, self.num_layers, layers_per_chunk):
            end = min(start + layers_per_chunk, self.num_layers)
            nbytes = (end - start) * layer_bytes
            if workspace.numel() < nbytes:
                raise ValueError("startup scatter workspace is too small")
            group = workspace[:nbytes]
            group.copy_(packed[start * layer_bytes : end * layer_bytes], non_blocking=True)
            typed = group.view(self.dtype).view(end - start, 2, len(indices), heads, dim)
            for offset, layer in enumerate(range(start, end)):
                self._kv_buffer[:, layer].flatten(1, 2).index_copy_(1, indices, typed[offset])

    def _transfer_indices(self, slots: torch.Tensor, num_pool_pages: int) -> torch.Tensor:
        """验证传入的 KV Cache Slot 索引是否合法，避免访问非法显存位置、重复 Slot 或不支持的数据格式。
        将 Slot 索引转换为 GPU 上的 torch.int64 Tensor，供后续 KV Cache 数据读取或写入使用。"""
        if sys.byteorder != "little" or self.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("unsupported transfer dtype or byte order")
        if (
            not isinstance(slots, torch.Tensor)
            or slots.ndim != 1
            or slots.dtype not in (torch.int32, torch.int64)
            or not len(slots)
        ):
            raise ValueError("slots must be a nonempty int32/int64 vector")
        if not 0 < num_pool_pages <= self._kv_buffer.shape[2]:
            raise ValueError("invalid usable pool page count")
        host_slots = slots.tolist()
        if (
            min(host_slots) < 0
            or max(host_slots) >= num_pool_pages * self._kv_buffer.shape[3]
            or len(set(host_slots)) != len(host_slots)
        ):
            raise ValueError("duplicate or out-of-pool KV slot")
        return slots.to(device=self.device, dtype=torch.int64)

    @torch.inference_mode()
    def export_page_into(self, destination: torch.Tensor, indices: torch.Tensor) -> None:
        """Trusted scheduler-owned slots -> preallocated [layer,K/V,token,head,dim]."""
        heads, dim = self._storage_shape[1:]
        shape = (self.num_layers, 2, len(indices), heads, dim)
        if (
            destination.device != self.device
            or destination.dtype != torch.uint8
            or destination.numel()
            != self.num_layers * 2 * len(indices) * heads * dim * self.dtype.itemsize
            or indices.dtype != torch.int64
            or indices.device != self.device
        ):
            raise ValueError("invalid preallocated host-cache export")
        typed = destination.view(self.dtype).view(shape)
        for layer in range(self.num_layers):
            torch.index_select(
                self._kv_buffer[:, layer].flatten(1, 2), 1, indices, out=typed[layer]
            )

    @torch.inference_mode()
    def import_page_from(self, source: torch.Tensor, indices: torch.Tensor) -> None:
        """Preallocated layer-major page -> private, scheduler-owned GPU slots."""
        heads, dim = self._storage_shape[1:]
        shape = (self.num_layers, 2, len(indices), heads, dim)
        if (
            source.device != self.device
            or source.dtype != torch.uint8
            or source.numel()
            != self.num_layers * 2 * len(indices) * heads * dim * self.dtype.itemsize
            or indices.dtype != torch.int64
            or indices.device != self.device
        ):
            raise ValueError("invalid preallocated host-cache import")
        typed = source.view(self.dtype).view(shape)
        for layer in range(self.num_layers):
            self._kv_buffer[:, layer].flatten(1, 2).index_copy_(1, indices, typed[layer])

    @torch.inference_mode()
    def export_tensor(self, slots: torch.Tensor, *, num_pool_pages: int) -> torch.Tensor:
        """Pack valid KV on its original device, without a host payload copy.

        The returned allocation is independent of the pool. Its owner must keep it
        alive until both the producing CUDA work and the transfer have completed.
        打包副本留在 GPU；完成打包后，原请求页与传输副本可以分别管理生命周期。
        """
        indices = self._transfer_indices(slots, num_pool_pages)
        return self._kv_buffer.flatten(2, 3).index_select(2, indices).contiguous()

    @torch.inference_mode()
    def import_tensor(
        self, packed: torch.Tensor, slots: torch.Tensor, *, num_pool_pages: int
    ) -> None:
        """Scatter a same-device KV tensor into receiver-owned slots.

        Accept either export_tensor's typed 5-D layout or a flat uint8 RDMA
        receive allocation. The latter is reinterpreted, never numerically cast.
        调用方必须先确认 RDMA 完成以及写入对 CUDA 可见；这里不进行 CPU 暂存。
        """
        indices = self._transfer_indices(slots, num_pool_pages)
        heads, dim = self._storage_shape[1:]
        shape = (2, self.num_layers, len(slots), heads, dim)
        expected = 2 * self.num_layers * len(slots) * heads * dim * self.dtype.itemsize
        if (
            not isinstance(packed, torch.Tensor)
            or packed.device != self._kv_buffer.device
            or not packed.is_contiguous()
            or packed.numel() * packed.element_size() != expected
        ):
            raise ValueError("packed KV must be contiguous on the pool device with matching length")
        if packed.dtype == torch.uint8:
            if packed.ndim != 1:
                raise ValueError("byte-packed KV must be a flat uint8 tensor")
            typed = packed.view(self.dtype).view(shape)
        elif packed.dtype == self.dtype and tuple(packed.shape) == shape:
            typed = packed
        else:
            raise ValueError("incorrect packed KV dtype or shape")
        self._kv_buffer.flatten(2, 3).index_copy_(2, indices, typed)

    @torch.inference_mode()
    def export_tokens(self, slots: torch.Tensor, *, num_pool_pages: int) -> bytes:
        """它只提取 slots 指定的有效 token，不导出整个 KV 池。 返回字节串之前，GPU→CPU 复制已经完成。"""
        """Pack valid tokens as [K/V, layer, logical token, local head, head dim]. Blocks for D2H."""
        packed = self.export_tensor(slots, num_pool_pages=num_pool_pages).cpu()
        return packed.view(torch.uint8).numpy().tobytes()

    @torch.inference_mode()
    def import_tokens(
        self, payload: bytes | bytearray, slots: torch.Tensor, *, num_pool_pages: int
    ) -> None:
        """将 Decode Worker CPU 内存中的 KV Cache 数据复制到 GPU 显存，并写入 GPU KV Cache 中指定的物理 Slot。"""
        "实现 D 端 KV 安装"
        """Scatter packed bytes to receiver-owned slots on the current stream."""
        indices = self._transfer_indices(slots, num_pool_pages)
        heads, dim = self._storage_shape[1:]
        shape = (2, self.num_layers, len(slots), heads, dim)
        expected = 2 * self.num_layers * len(slots) * heads * dim * self.dtype.itemsize
        if len(payload) != expected:
            raise ValueError("incorrect packed KV length")
        writable = bytearray(payload) if isinstance(payload, bytes) else payload
        packed_cpu = torch.frombuffer(writable, dtype=self.dtype).view(shape)
        packed = packed_cpu.to(self.device)
        self._kv_buffer.flatten(2, 3).index_copy_(2, indices, packed)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
