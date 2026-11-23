"""Early receive reservation for layer-major, grouped GPU KV transfers."""

from __future__ import annotations

from dataclasses import dataclass

from .control import endpoint_address
from .protocol import KVLayout, _array, _int, _load, _object, _token, _wire, check_transfer_id


@dataclass(frozen=True)
class LayeredReceiveTicket:
    transfer_id: str
    endpoint: str
    locator: str
    session: str
    address: int
    layout: KVLayout
    kv_tokens: int
    layers_per_chunk: int
    layer_offsets: tuple[int, ...]
    eos_token_id: int | None
    packing: str = "layer-kv-token-head-dim-le"
    version: int = 2

    def __post_init__(self) -> None:
        check_transfer_id(self.transfer_id)
        check_transfer_id(self.locator)
        endpoint_address(self.endpoint)
        endpoint_address(self.session)
        _int("address", self.address, 1)
        _int("kv_tokens", self.kv_tokens, 1)
        _int("layers_per_chunk", self.layers_per_chunk, 1)
        if not isinstance(self.layout, KVLayout):
            raise ValueError("missing layer transfer layout")
        if self.eos_token_id is not None:
            _token(self.eos_token_id)
        if type(self.version) is not int or self.version != 2:
            raise ValueError("unsupported layer transfer version")
        if self.packing != "layer-kv-token-head-dim-le":
            raise ValueError("unsupported layer packing")
        expected = tuple(i * self.layer_bytes for i in range(self.layout.layers))
        if (
            not isinstance(self.layer_offsets, tuple)
            or any(type(offset) is not int for offset in self.layer_offsets)
            or self.layer_offsets != expected
        ):
            raise ValueError("layer offsets must cover the contiguous receive region exactly")

    @property
    def layer_bytes(self) -> int:
        return self.kv_tokens * self.layout.bytes_per_token // self.layout.layers

    @property
    def nbytes(self) -> int:
        return self.layer_bytes * self.layout.layers

    @property
    def groups(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (start, min(start + self.layers_per_chunk, self.layout.layers))
            for start in range(0, self.layout.layers, self.layers_per_chunk)
        )

    def to_wire(self) -> bytes:
        return _wire(self)

    @classmethod
    def from_wire(cls, payload: bytes) -> LayeredReceiveTicket:
        data = _object(cls, _load(payload))
        data["layout"] = KVLayout(**_object(KVLayout, data["layout"]))
        data["layer_offsets"] = _array(data["layer_offsets"])
        return cls(**data)
