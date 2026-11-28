"""P/D handoff and suffix metadata; wire data never contains local tensor handles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable
from uuid import UUID

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}
MAX_INT32 = 2**31 - 1
HANDOFF_VERSION = 2
MOONCAKE_TRANSPORT_MODES = ("gpu_direct", "host_dram")
LAYERED_TRANSFER_MODES = (*MOONCAKE_TRANSPORT_MODES, "shm_dram")


def worker_transfer_mode(info):
    """Missing metadata is the legacy direct-GPU protocol, never a host fallback."""
    return info.get("transfer_mode", "gpu_direct" if info.get("layered_transfer") else "tcp")


def _int(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _token(value: int) -> None:
    _int("token ID", value)
    if value > MAX_INT32:
        raise ValueError("token ID exceeds the int32 token pool")


def check_transfer_id(value: str) -> None:
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("transfer_id must be a canonical UUID string")


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate JSON field")
    return obj


def _wire(record: Any) -> bytes:
    obj = asdict(record)
    # Preserve the existing gpu_direct wire representation. Host mode must be explicit.
    for value in (obj, obj.get("kv", {})):
        if value.get("transfer_mode") == "gpu_direct":
            value.pop("transfer_mode")
    return json.dumps(obj, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _load(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)


def _object(cls: type, obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict) or obj.keys() != {field.name for field in fields(cls)}:
        raise ValueError(f"incorrect fields for {cls.__name__}")
    return dict(obj)


def _array(value: Any) -> tuple:
    if not isinstance(value, list):
        raise ValueError("expected a JSON array")
    return tuple(value)


def _v1(version: int) -> None:
    if type(version) is not int or version != 1:
        raise ValueError("unsupported wire version")


@dataclass(frozen=True)
class KVLayout:
    """定义 KV Cache 的数据类型、张量维度、打包格式和 TP 信息"""

    """Per-rank packed payload: K/V, layer, logical token, local KV head, head dimension."""

    model_fingerprint: str
    dtype: str
    layers: int
    kv_heads: int
    head_dim: int
    page_size: int
    tp_rank: int = 0
    tp_size: int = 1
    packing: str = "kv-layer-token-head-dim-le"
    quantization: str = "none"

    def __post_init__(self) -> None:
        # The producer hashes a deployment manifest (weights + effective model/adapter config).
        if not isinstance(self.model_fingerprint, str) or len(self.model_fingerprint) != 64:
            raise ValueError("model_fingerprint must be a lowercase SHA-256 hex digest")
        if any(c not in "0123456789abcdef" for c in self.model_fingerprint):
            raise ValueError("invalid model_fingerprint")
        if not isinstance(self.dtype, str) or self.dtype not in DTYPE_BYTES:
            raise ValueError("unsupported KV dtype")
        if self.quantization != "none" or self.packing != "kv-layer-token-head-dim-le":
            raise ValueError("unsupported KV quantization or packing")
        for name in ("layers", "kv_heads", "head_dim", "page_size", "tp_size"):
            _int(name, getattr(self, name), 1)
        _int("tp_rank", self.tp_rank)
        if self.tp_rank >= self.tp_size:
            raise ValueError("tp_rank out of range")

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class KVTransferMetadata:
    """某一次传输需要搬运哪些 KV Cache

    KVLayout 描述一类 KV Cache 的格式。
    KVTransferMetadata 描述某次具体传输涉及的 KV Cache。
    """

    """Source pages cover the suffix after prefix_len; kv_tokens may be zero on a full hit."""

    transfer_id: str
    layout: KVLayout
    kv_tokens: int
    source_pages: tuple[int, ...]
    version: int = HANDOFF_VERSION
    prefix_len: int = 0
    transfer_mode: str = "gpu_direct"

    def __post_init__(self) -> None:
        if self.transfer_mode not in LAYERED_TRANSFER_MODES:
            raise ValueError("unsupported KV transfer mode")
        if self.version == 1 and self.transfer_mode != "gpu_direct":
            raise ValueError("v1 cannot describe host DRAM transport")
        if type(self.version) is not int or self.version not in (1, HANDOFF_VERSION):
            raise ValueError("unsupported KV metadata version")
        check_transfer_id(self.transfer_id)
        if not isinstance(self.layout, KVLayout):
            raise ValueError("missing KV layout")
        _int("kv_tokens", self.kv_tokens)
        _int("prefix_len", self.prefix_len)
        if not self.prefix_len + self.kv_tokens:
            raise ValueError("KV metadata must describe a nonempty prompt")
        if self.version == 1 and (self.prefix_len or not self.kv_tokens):
            raise ValueError("v1 requires a nonempty full-prompt transfer")
        p = self.layout.page_size
        if self.prefix_len % p:
            raise ValueError("decode prefix must contain whole KV pages")
        if (
            not isinstance(self.source_pages, tuple)
            or len(self.source_pages) != (self.kv_tokens + p - 1) // p
        ):
            raise ValueError("source_pages must cover exactly the valid token range")
        for page in self.source_pages:
            _int("source page", page)
        if len(set(self.source_pages)) != len(self.source_pages):
            raise ValueError("repeated physical page within one request")

    @property
    def nbytes(self) -> int:
        return self.kv_tokens * self.layout.bytes_per_token

    @property
    def prompt_tokens(self) -> int:
        return self.prefix_len + self.kv_tokens

    @classmethod
    def from_slots(
        cls,
        transfer_id: str,
        layout: KVLayout,
        slots: Iterable[int],
        *,
        num_pool_pages: int,
        prefix_len: int = 0,
        transfer_mode: str = "gpu_direct",
    ) -> KVTransferMetadata:
        """Consume only the valid CPU slot snapshot. num_pool_pages excludes the graph dummy."""
        _int("num_pool_pages", num_pool_pages, 1)
        pages: list[int] = []
        count = 0
        for pos, slot in enumerate(slots):
            _int("physical token slot", slot)
            page, offset = divmod(slot, layout.page_size)
            if page >= num_pool_pages or offset != pos % layout.page_size:
                raise ValueError("out-of-pool or misaligned token slot")
            if offset == 0:
                pages.append(page)
            elif pages[-1] != page:
                raise ValueError("physical page changed within a logical page")
            count += 1
        return cls(
            transfer_id,
            layout,
            count,
            tuple(pages),
            prefix_len=prefix_len,
            transfer_mode=transfer_mode,
        )

    def to_wire(self) -> bytes:
        return _wire(self)

    @classmethod
    def _from_object(cls, obj: Any) -> KVTransferMetadata:
        if isinstance(obj, dict):
            obj = {"transfer_mode": "gpu_direct", **obj}
        if isinstance(obj, dict) and obj.get("version") == 1:
            obj = {"prefix_len": 0, **obj}
        data = _object(cls, obj)
        data["layout"] = KVLayout(**_object(KVLayout, data["layout"]))
        data["source_pages"] = _array(data["source_pages"])
        return cls(**data)

    @classmethod
    def from_wire(cls, payload: bytes) -> KVTransferMetadata:
        return cls._from_object(_load(payload))


@dataclass(frozen=True)
class RequestHandoff:
    """Prefill 完成后，Decode 应该如何接续这个推理请求

    Decode 收到 KV Cache 后，如何知道自己应该继续执行哪个请求，以及接下来需要生成什么？
    """

    """After final prefill. P sampled token zero; the sole egress publishes it once."""

    uid: int
    prompt_ids: tuple[int, ...]
    first_token_id: int
    max_tokens: int
    ignore_eos: bool
    eos_token_id: int | None
    kv: KVTransferMetadata
    sampling: str = "greedy"
    version: int = HANDOFF_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version not in (1, HANDOFF_VERSION):
            raise ValueError("unsupported handoff version")
        _int("uid", self.uid)
        _int("max_tokens", self.max_tokens, 1)
        if not isinstance(self.kv, KVTransferMetadata):
            raise ValueError("missing KV metadata")
        if self.version == 1 and (
            self.kv.prefix_len or not self.kv.kv_tokens or self.kv.transfer_mode != "gpu_direct"
        ):
            raise ValueError("v1 requires a nonempty full-prompt transfer")
        if not isinstance(self.prompt_ids, tuple) or len(self.prompt_ids) != self.kv.prompt_tokens:
            raise ValueError("decode prefix and transferred KV must cover the complete prompt")
        for token in self.prompt_ids:
            _token(token)
        _token(self.first_token_id)
        if self.eos_token_id is not None:
            _token(self.eos_token_id)
        if type(self.ignore_eos) is not bool or self.sampling != "greedy":
            raise ValueError("handoff requires a boolean ignore_eos and greedy sampling")

    @property
    def remaining_tokens(self) -> int:
        if not self.ignore_eos and self.first_token_id == self.eos_token_id:
            return 0
        return self.max_tokens - 1

    def to_wire(self) -> bytes:
        return _wire(self)

    @classmethod
    def from_wire(cls, payload: bytes) -> RequestHandoff:
        data = _object(cls, _load(payload))
        data["kv"] = KVTransferMetadata._from_object(data["kv"])
        data["prompt_ids"] = _array(data["prompt_ids"])
        return cls(**data)


@dataclass(frozen=True)
class ReceiveTicket:
    """Decode 已经准备好的接收任务，供 Prefill 定位传输目标

    让Prefill 知道自己需要发送什么 KV Cache，但是应该发送到 Decode 的哪个接收任务
    """

    """Attempt-scoped receiver locator, interpreted only by the selected transfer backend."""

    transfer_id: str
    endpoint: str
    locator: str
    nbytes: int
    version: int = 1

    def __post_init__(self) -> None:
        _v1(self.version)
        check_transfer_id(self.transfer_id)
        _text("endpoint", self.endpoint)
        _text("locator", self.locator)
        _int("nbytes", self.nbytes, 1)

    def to_wire(self) -> bytes:
        return _wire(self)

    @classmethod
    def from_wire(cls, payload: bytes) -> ReceiveTicket:
        return cls(**_object(cls, _load(payload)))
