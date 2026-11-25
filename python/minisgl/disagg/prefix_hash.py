"""Canonical, CPU-only identities for full pages of a P's HBM prefix cache."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

PREFIX_INDEX_VERSION = 1
_ROOT_TAG = b"minisgl-prefix-root-v1\0"
_PAGE_TAG = b"minisgl-prefix-page-v1\0"


@dataclass(frozen=True)
class PrefixHashSpec:
    # Serialized canonical manifest, rather than a mutable layout dictionary.
    manifest: str

    def __post_init__(self):
        if not isinstance(self.manifest, str) or len(self.manifest.encode("utf-8")) > 8192:
            raise ValueError("prefix hash manifest exceeds its metadata bound")
        value = json.loads(self.manifest)
        if set(value) != {"hash_version", "model", "version_group", "page_size", "cache_namespace"}:
            raise ValueError("invalid prefix hash manifest")
        if type(value["hash_version"]) is not int or value["hash_version"] != PREFIX_INDEX_VERSION:
            raise ValueError("unsupported prefix hash version")
        if type(value["page_size"]) is not int or value["page_size"] < 1:
            raise ValueError("invalid prefix page size")
        if (
            not isinstance(value["cache_namespace"], str)
            or not 1 <= len(value["cache_namespace"].encode("utf-8")) <= 1024
        ):
            raise ValueError("invalid cache namespace")
        model = value["model"]
        if not isinstance(model, str) or len(model) != 64:
            raise ValueError("prefix model must be a checkpoint fingerprint")
        if bytes.fromhex(model).hex() != model:
            raise ValueError("prefix fingerprint is not canonical SHA256")
        group = value["version_group"]
        if not isinstance(group, dict) or group.get("kv_layout") != "mha-kv-layer-page-token-head-dim/v1":
            raise ValueError("unsupported prefix KV layout")
        if group.get("dtype") not in ("float16", "bfloat16", "float32"):
            raise ValueError("unsupported prefix dtype")
        if set(group) != {
            "kv_layout", "dtype", "layers", "kv_heads", "head_dim",
            "tp_rank", "tp_size", "packing", "quantization",
        }:
            raise ValueError("invalid prefix version group")
        for name in ("layers", "kv_heads", "head_dim", "tp_size"):
            if type(group[name]) is not int or group[name] < 1:
                raise ValueError("invalid prefix KV geometry")
        if (
            type(group["tp_rank"]) is not int
            or not 0 <= group["tp_rank"] < group["tp_size"]
            or group["packing"] != "kv-layer-token-head-dim-le"
            or group["quantization"] != "none"
        ):
            raise ValueError("unsupported prefix version group")
        if self.manifest != _canonical(value):
            raise ValueError("prefix hash manifest is not canonical")

    @classmethod
    def from_layout(cls, layout: dict, cache_namespace: str = "default") -> PrefixHashSpec:
        group = {
            name: layout[name]
            for name in (
                "dtype", "layers", "kv_heads", "head_dim", "tp_rank", "tp_size",
                "packing", "quantization",
            )
        }
        group["kv_layout"] = "mha-kv-layer-page-token-head-dim/v1"
        return cls(
            _canonical({
                "hash_version": PREFIX_INDEX_VERSION,
                "model": layout["model_fingerprint"],
                "version_group": group,
                "page_size": layout["page_size"],
                "cache_namespace": cache_namespace,
            })
        )

    @cached_property
    def page_size(self) -> int:
        return json.loads(self.manifest)["page_size"]

    @cached_property
    def root(self) -> bytes:
        raw = self.manifest.encode("utf-8")
        return hashlib.sha256(_ROOT_TAG + struct.pack("<I", len(raw)) + raw).digest()

    @cached_property
    def domain_id(self) -> str:
        return self.root.hex()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def hash_page(parent: bytes, tokens: Iterable[int], page_size: int) -> bytes:
    ids = tuple(tokens)
    if len(parent) != 32 or len(ids) != page_size:
        raise ValueError("prefix hashing requires a parent digest and one full page")
    if any(type(token) is not int or not 0 <= token < 2**31 for token in ids):
        raise ValueError("prefix tokens must fit nonnegative int32")
    return hashlib.sha256(_PAGE_TAG + parent + struct.pack(f"<{page_size}i", *ids)).digest()


@dataclass(frozen=True)
class PreparedPrefix:
    """Internal request preparation result; never a proof that KV exists."""

    token_ids: tuple[int, ...]
    domain_id: str
    # Packed digests avoid one Python object/hex string per token at page_size=1.
    page_hashes: bytes

    def digests(self):
        for offset in range(0, len(self.page_hashes), 32):
            yield self.page_hashes[offset:offset + 32]


def prepare_prefix(token_ids, spec: PrefixHashSpec) -> PreparedPrefix:
    ids = tuple(token_ids)
    if any(type(token) is not int or not 0 <= token < 2**31 for token in ids):
        raise ValueError("prefix tokens must fit nonnegative int32")
    page = spec.page_size
    parent = spec.root
    result = bytearray()
    for offset in range(0, len(ids) // page * page, page):
        parent = hash_page(parent, ids[offset:offset + page], page)
        result.extend(parent)
    return PreparedPrefix(ids, spec.domain_id, bytes(result))
