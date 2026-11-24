from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        # if needed_pages > 0:
        #     allocated = self._page_to_token(self._allocate(needed_pages))
        #     _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
        '''新写法保留 pages，捕获上述异常后归还到 free_slots，再继续抛出异常。'''
        if needed_pages > 0:
            pages = self._allocate(needed_pages)
            try:
                allocated = self._page_to_token(pages)
                _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
            except Exception:
                self.free_slots = torch.cat([self.free_slots, pages])
                raise

    def reserve_import(self, table_idx: int, num_tokens: int, *, start: int = 0) -> torch.Tensor:
        '''为 D 端导入 KV 准备存储位置,它先检查参数类型、token 数、表行范围、页表宽度和可用容量，然后按页分配'''
        """Reserve private pages without registering uninitialized KV in the prefix cache."""
        if type(num_tokens) is not int or type(table_idx) is not int or type(start) is not int:
            raise ValueError("import token count and table row must be integers")
        pages = div_ceil(num_tokens, self.page_size)
        if (
            num_tokens <= 0
            or start < 0
            or start % self.page_size
            or not 0 <= table_idx < self.page_table.shape[0] - 1
        ):
            raise ValueError("invalid import token count or request table row")
        if start + pages * self.page_size > self.page_table.shape[1]:
            raise ValueError("import exceeds page table capacity")
        if pages * self.page_size > self.available_size:
            raise RuntimeError("insufficient KV capacity for import")
        allocated_pages = self._allocate(pages)
        try:
            allocated = self._page_to_token(allocated_pages)
            self.page_table[table_idx, start : start + len(allocated)] = allocated
        except Exception:
            self.free_slots = torch.cat([self.free_slots, allocated_pages])
            raise
        return allocated

    def discard_private(self, req: Req, allocated_len: int) -> None:
        '''用于释放请求私有 KV Cache 资源的函数'''
        """Release a split request, including allocated KV that a failed forward never populated."""
        end = div_ceil(allocated_len, self.page_size) * self.page_size
        start = req.cache_handle.cached_len
        if not 0 <= start <= end <= self.page_table.shape[1]:
            raise ValueError("invalid private KV allocation boundary")
        self.unlock(req.cache_handle)
        self._free(self.page_table[req.table_idx, start:end])


    def cache_req(self, req: Req, *, finished: bool, rebind: bool = False) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        if rebind and not finished:
            page_indices = page_indices.clone()
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        if rebind and not finished and new_handle.cached_len:
            # Another received request may have published the same prefix.
            # Adopt the canonical pages before returning duplicate private pages.
            indices = new_handle.get_matched_indices()
            self.page_table[req.table_idx, : new_handle.cached_len] = indices
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # Return original private pages replaced by an already-cached prefix.
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)

    def check_integrity(self) -> None:
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
