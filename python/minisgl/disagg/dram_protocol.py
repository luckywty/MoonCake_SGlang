"""Optional host-cache protocol; does not change P/D handoff v2 or ticket v3."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .control import endpoint_address
from .protocol import check_transfer_id

DRAM_PROTOCOL_VERSION = 1
TIER_INDEX_VERSION = 2
MAX_DRAM_PULL_PAGES = 16384  # Keep pin plans/descriptors below the control-frame limit.


def digest_bytes(value):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("invalid page digest")
    raw = bytes.fromhex(value)
    if raw.hex() != value:
        raise ValueError("noncanonical page digest")
    return raw


@dataclass(frozen=True)
class RemotePullPlan:
    source_id: str
    source_instance: str
    source_endpoint: str
    domain_id: str
    start_page: int
    page_hashes: tuple[bytes, ...]

    def to_object(self):
        return {
            "version": DRAM_PROTOCOL_VERSION,
            "source_id": self.source_id,
            "source_instance": self.source_instance,
            "source_endpoint": self.source_endpoint,
            "domain_id": self.domain_id,
            "start_page": self.start_page,
            "page_hashes": [h.hex() for h in self.page_hashes],
        }

    @classmethod
    def from_object(cls, value):
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "version",
                "source_id",
                "source_instance",
                "source_endpoint",
                "domain_id",
                "start_page",
                "page_hashes",
            }
            or type(value["version"]) is not int
            or value["version"] != DRAM_PROTOCOL_VERSION
            or type(value["start_page"]) is not int
            or value["start_page"] < 0
            or not isinstance(value["page_hashes"], list)
            or not 1 <= len(value["page_hashes"]) <= MAX_DRAM_PULL_PAGES
        ):
            raise ValueError("invalid remote pull plan")
        check_transfer_id(value["source_instance"])
        endpoint_address(value["source_endpoint"])
        digest_bytes(value["domain_id"])
        if not isinstance(value["source_id"], str) or not value["source_id"]:
            raise ValueError("missing remote source identity")
        return cls(
            value["source_id"],
            value["source_instance"],
            value["source_endpoint"],
            value["domain_id"],
            value["start_page"],
            tuple(digest_bytes(h) for h in value["page_hashes"]),
        )


@dataclass(frozen=True)
class DramConfig:
    capacity_bytes: int = 0
    policy: str = "lru"
    staging_slots: int = 2
    batch_pages: int = 8
    max_leases: int = 32
    lease_ttl: float = 30.0
    transaction_ttl: float = 300.0
    remote_enabled: bool = False
    min_remote_tokens: int = 1024
    remote_timeout: float = 1.0
    remote_max_bytes: int = 64 * 1024**2
    remote_bytes_per_second: int = 256 * 1024**2
    host: str = "127.0.0.1"
    port: int | None = None
    device_name: str = ""

    def __post_init__(self):
        if (
            type(self.capacity_bytes) is not int
            or self.capacity_bytes < 0
            or self.policy not in ("lru", "lfu")
            or type(self.remote_enabled) is not bool
        ):
            raise ValueError("invalid DRAM cache configuration")
        for name in (
            "staging_slots",
            "batch_pages",
            "max_leases",
            "min_remote_tokens",
            "remote_max_bytes",
            "remote_bytes_per_second",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid DRAM {name}")
        for name in ("lease_ttl", "transaction_ttl", "remote_timeout"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid DRAM {name}")
        if self.remote_enabled and (
            not self.capacity_bytes or self.lease_ttl <= self.remote_timeout
        ):
            raise ValueError("remote DRAM requires capacity and a TTL longer than its read timeout")

    @property
    def enabled(self):
        return bool(self.capacity_bytes)

    def gpu_staging_bytes(self, page_bytes):
        return 2 * self.staging_slots * self.batch_pages * page_bytes if self.enabled else 0


_DRAM_FLAGS = (
    ("cache-bytes", "capacity_bytes", int),
    ("staging-slots", "staging_slots", int),
    ("batch-pages", "batch_pages", int),
    ("max-leases", "max_leases", int),
    ("lease-ttl", "lease_ttl", float),
    ("transaction-ttl", "transaction_ttl", float),
    ("min-remote-tokens", "min_remote_tokens", int),
    ("pull-timeout", "remote_timeout", float),
    ("max-pull-bytes", "remote_max_bytes", int),
    ("read-bandwidth", "remote_bytes_per_second", int),
)


def add_dram_args(parser):
    defaults = DramConfig()
    for flag, name, kind in _DRAM_FLAGS:
        parser.add_argument(
            "--pd-dram-" + flag, dest="dram_" + name, type=kind, default=getattr(defaults, name)
        )
    parser.add_argument(
        "--pd-dram-cache-policy", dest="dram_policy", choices=("lru", "lfu"), default="lru"
    )
    parser.add_argument("--pd-dram-remote-pull", dest="dram_remote_enabled", action="store_true")
    parser.add_argument("--pd-dram-host", dest="dram_host")
    parser.add_argument("--pd-dram-port", dest="dram_port", type=int)
    parser.add_argument("--pd-dram-rdma-device", dest="dram_device_name")


def pop_dram_config(kwargs):
    values = {name: kwargs.pop("dram_" + name) for _, name, _ in _DRAM_FLAGS}
    for name in ("policy", "remote_enabled", "host", "port", "device_name"):
        values[name] = kwargs.pop("dram_" + name)
    values["host"] = (
        values["host"]
        or kwargs.get("pd_mooncake_host")
        or kwargs.get("pd_advertise_host")
        or kwargs["server_host"]
    )
    values["device_name"] = values["device_name"] or kwargs.get("pd_mooncake_device", "")
    if values["remote_enabled"] and values["port"] is None:
        values["port"] = kwargs["server_port"] + 3
    result = DramConfig(**values)
    if result.enabled and kwargs["worker_role"] == "unified":
        raise ValueError("DRAM KV caching requires a split Worker")
    if result.remote_enabled:
        if result.host in ("0.0.0.0", "::"):
            raise ValueError("remote DRAM needs --pd-dram-host or an advertised host")
        if result.port is not None:
            from .cluster_config import validate_mooncake_port

            validate_mooncake_port(result.port)
            ports = {
                kwargs["server_port"],
                kwargs["server_port"] + 1,
                kwargs.get("pd_data_port") or kwargs["server_port"] + 2,
                kwargs.get("pd_mooncake_port"),
            }
            if result.port in ports:
                raise ValueError("DRAM TE port conflicts with a Worker/GPU transfer port")
    return result
