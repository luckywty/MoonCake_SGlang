"""Aligned, coalescing suballocator for a startup-registered receive pool.

The caller owns synchronization and may free a range only after both native
writers and CUDA consumers have stopped accessing it.
"""

from .transfer import TransferCapacityError, TransferError


class ReceiveAllocator:
    def __init__(self, capacity: int, alignment: int = 256):
        if capacity <= 0 or alignment <= 0:
            raise ValueError("pool capacity and alignment must be positive")
        self.capacity, self.alignment = capacity, alignment
        self.free_ranges = [(0, capacity)]
        self.allocations: dict[str, tuple[int, int]] = {}

    @property
    def available(self):
        return sum(size for _, size in self.free_ranges)

    def allocate(self, key: str, nbytes: int) -> int:
        if key in self.allocations or nbytes <= 0:
            raise TransferError("invalid or duplicate receive lease")
        size = (nbytes + self.alignment - 1) // self.alignment * self.alignment
        for i, (offset, length) in enumerate(self.free_ranges):
            if length >= size:
                self.allocations[key] = (offset, size)
                self.free_ranges[i : i + 1] = (
                    [(offset + size, length - size)] if length > size else []
                )
                return offset
        raise TransferCapacityError("receive pool has insufficient contiguous space")

    def release(self, key: str):
        allocation = self.allocations.pop(key)
        ranges = sorted([*self.free_ranges, allocation])
        merged = []
        for offset, length in ranges:
            if merged and sum(merged[-1]) == offset:
                start, size = merged[-1]
                merged[-1] = (start, size + length)
            else:
                merged.append((offset, length))
        self.free_ranges = merged
