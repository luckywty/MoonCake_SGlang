"""Bucket-first P/D selection; routing observations never allocate GPU pages."""

from __future__ import annotations

import json
import math
import random
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path

from .router import NoHealthyWorker, logger

POLICIES = ("power_of_two", "cache_aware", "round_robin", "random")


@dataclass(frozen=True)
class AdmissionLimits:
    max_inflight_requests: int | None = None
    max_waiting_requests: int | None = None
    max_pending_prefill_tokens: int | None = None
    max_running_usage: float | None = None
    max_kv_usage: float | None = None

    def __post_init__(self):
        for name, value in vars(self).items():
            if value is None:
                continue
            if name.endswith("usage"):
                valid = type(value) in (int, float) and math.isfinite(value) and 0 < value <= 1
            else:
                valid = type(value) is int and value > 0
            if not valid:
                raise ValueError(f"invalid routing admission limit {name}")

    def merged(self, defaults):
        return AdmissionLimits(
            **{k: v if v is not None else getattr(defaults, k) for k, v in vars(self).items()}
        )


@dataclass(frozen=True)
class WorkerGroup:
    worker_ids: tuple[str, ...] | None = None
    policy: str | None = None
    admission: AdmissionLimits = field(default_factory=AdmissionLimits)

    def __post_init__(self):
        if self.policy is not None and self.policy not in POLICIES:
            raise ValueError("unknown Worker routing policy")
        if self.worker_ids is not None and (
            not self.worker_ids
            or any(not isinstance(v, str) or not v for v in self.worker_ids)
            or len(set(self.worker_ids)) != len(self.worker_ids)
        ):
            raise ValueError("Worker membership must be a nonempty unique list")


@dataclass(frozen=True)
class RoutingBucket:
    id: str = "default"
    min_input_tokens: int = 1
    max_input_tokens: int | None = None
    max_context_tokens: int | None = None
    rank: int = 0
    prefill: WorkerGroup = field(default_factory=WorkerGroup)
    decode: WorkerGroup = field(default_factory=WorkerGroup)

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("routing bucket needs an id")
        for name in ("min_input_tokens", "max_input_tokens", "max_context_tokens", "rank"):
            value = getattr(self, name)
            if value is None and name in ("min_input_tokens", "rank"):
                raise ValueError(f"bucket {name} cannot be null")
            if value is not None and (
                type(value) is not int or value < (0 if name == "rank" else 1)
            ):
                raise ValueError(f"invalid bucket {name}")
        if self.max_input_tokens is not None and self.min_input_tokens > self.max_input_tokens:
            raise ValueError("bucket input range is reversed")
        if self.max_context_tokens is not None and self.max_context_tokens < self.min_input_tokens:
            raise ValueError("bucket context limit is smaller than its minimum input")
        if self.decode.policy == "cache_aware":
            raise ValueError("cache_aware is a Prefill policy")

    def fits(self, request):
        n = len(request.prompt_ids)
        return (
            n >= self.min_input_tokens
            and (self.max_input_tokens is None or n <= self.max_input_tokens)
            and (
                self.max_context_tokens is None or n + request.max_tokens <= self.max_context_tokens
            )
        )


def _object(value, cls):
    if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
        raise ValueError(f"invalid {cls.__name__} fields")
    return dict(value)


def load_buckets(path):
    root = json.loads(Path(path).read_text())
    if (
        not isinstance(root, dict)
        or set(root) != {"buckets"}
        or not isinstance(root["buckets"], list)
    ):
        raise ValueError("bucket config must contain a buckets list")
    result = []
    for value in root["buckets"]:
        args = _object(value, RoutingBucket)
        for role in ("prefill", "decode"):
            group = _object(args.get(role, {}), WorkerGroup)
            group["admission"] = AdmissionLimits(
                **_object(group.get("admission", {}), AdmissionLimits)
            )
            if "worker_ids" in group:
                if not isinstance(group["worker_ids"], list):
                    raise ValueError("worker_ids must be a list")
                group["worker_ids"] = tuple(group["worker_ids"])
            args[role] = WorkerGroup(**group)
        result.append(RoutingBucket(**args))
    return tuple(result)


@dataclass(frozen=True)
class RoutingConfig:
    policy: str = "power_of_two"
    decode_policy: str = "power_of_two"
    admission: AdmissionLimits = field(default_factory=AdmissionLimits)
    buckets: tuple[RoutingBucket, ...] = field(default_factory=lambda: (RoutingBucket(),))
    load_max_age: float = 6.0
    cache_min_tokens: int = 1024
    cache_min_ratio: float = 0.0
    affinity_mode: str = "prefer"
    affinity_load_factor: float = 2.0
    affinity_load_gap: int = 1024
    prefix_index_max_entries: int = 262144
    prefix_index_max_worker_entries: int = 65536
    prefix_index_max_age: float = 6.0
    prefix_sync_interval: float = 0.2
    min_remote_pull_tokens: int = 1024
    dram_h2d_gbps: float = 12.0
    dram_rdma_gbps: float = 10.0
    dram_prefill_ms_per_token: float = 0.01
    dram_max_pull_bytes: int = 64 * 1024**2
    pd_cross_numa_penalty: float = 4.0
    pd_shared_dram_remote_h2d_factor: float = 1.5

    def __post_init__(self):
        if not math.isfinite(self.pd_cross_numa_penalty) or self.pd_cross_numa_penalty < 0:
            raise ValueError("cross NUMA routing penalty must be finite and nonnegative")
        if (
            not math.isfinite(self.pd_shared_dram_remote_h2d_factor)
            or self.pd_shared_dram_remote_h2d_factor < 1
        ):
            raise ValueError("shared history cross-NUMA H2D factor must be finite and >= 1")
        if (
            type(self.min_remote_pull_tokens) is not int
            or self.min_remote_pull_tokens < 1
            or type(self.dram_max_pull_bytes) is not int
            or self.dram_max_pull_bytes < 1
            or any(
                not math.isfinite(v) or v <= 0
                for v in (self.dram_h2d_gbps, self.dram_rdma_gbps, self.dram_prefill_ms_per_token)
            )
        ):
            raise ValueError("invalid tier-aware routing costs")
        if any(
            type(value) is not int or value < 1
            for value in (self.prefix_index_max_entries, self.prefix_index_max_worker_entries)
        ) or any(
            not math.isfinite(value) or value <= 0
            for value in (self.prefix_index_max_age, self.prefix_sync_interval)
        ):
            raise ValueError("prefix index bounds/intervals must be positive")
        if (
            self.policy not in POLICIES
            or self.decode_policy not in POLICIES
            or self.decode_policy == "cache_aware"
        ):
            raise ValueError("invalid P/D routing policy")
        if not self.buckets or len({b.id for b in self.buckets}) != len(self.buckets):
            raise ValueError("routing buckets must be nonempty and have unique ids")
        if not math.isfinite(self.load_max_age) or self.load_max_age <= 0:
            raise ValueError("routing load age must be positive and finite")
        if type(self.cache_min_tokens) is not int or self.cache_min_tokens < 1:
            raise ValueError("cache token threshold must be positive")
        if not math.isfinite(self.cache_min_ratio) or not 0 <= self.cache_min_ratio <= 1:
            raise ValueError("cache ratio must be in [0, 1]")
        if self.affinity_mode not in ("prefer", "balanced"):
            raise ValueError("invalid affinity mode")
        if not math.isfinite(self.affinity_load_factor) or self.affinity_load_factor < 1:
            raise ValueError("affinity factor must be at least 1")
        if type(self.affinity_load_gap) is not int or self.affinity_load_gap < 0:
            raise ValueError("affinity gap must be nonnegative")

    def validate_worker_ids(self, prefill_ids, decode_ids):
        allowed = {"prefill": set(prefill_ids), "decode": set(decode_ids)}
        for bucket in self.buckets:
            for role in ("prefill", "decode"):
                members = getattr(bucket, role).worker_ids
                if members is not None and not set(members) <= allowed[role]:
                    raise ValueError(f"bucket {bucket.id} contains unknown {role} Worker ids")


class PolicyRouter:
    def __init__(self, pool, seed=None, config=None):
        self.pool, self.random = pool, random.Random(seed)
        self.config = config or RoutingConfig()
        self.lock = threading.RLock()
        self.inflight, self.cursors = {}, {}
        self.config.validate_worker_ids(
            [n.worker_id for n in pool.prefill_pool], [n.worker_id for n in pool.decode_pool]
        )

    def buckets_for(self, request):
        result = [b for b in self.config.buckets if b.fits(request)]
        if not result:
            raise ValueError("no routing bucket supports the requested token length")
        return sorted(
            result,
            key=lambda b: (
                min(b.max_input_tokens or math.inf, b.max_context_tokens or math.inf),
                b.rank,
                b.id,
            ),
        )

    def _loads(self):
        with self.pool.lock:
            now, result = time.monotonic(), {}
            for node in self.pool.nodes:
                load = node.health.get("load", {})
                if now - node.load_updated_at > self.config.load_max_age or not isinstance(
                    load, dict
                ):
                    load = {}
                result[node.worker_id] = {
                    k: v
                    for k, v in load.items()
                    if type(v) in (int, float) and math.isfinite(v) and v >= 0
                }
            return result

    def candidates(self, role, request, bucket, exclude=(), *, loads=None):
        loads = loads if loads is not None else self._loads()
        group = getattr(bucket, role)
        limits = group.admission.merged(self.config.admission)
        result = []
        for node in self.pool.healthy(role, exclude):
            if group.worker_ids is not None and node.worker_id not in group.worker_ids:
                continue
            h = node.health
            max_seq = h.get("max_sequence_tokens")
            if type(max_seq) is int and len(request.prompt_ids) + request.max_tokens > max_seq:
                continue
            capacity, page = h.get("num_pages"), h.get("layout", {}).get("page_size")
            needed = len(request.prompt_ids) + (request.max_tokens if role == "decode" else 0)
            if type(capacity) is int and type(page) is int and needed > capacity * page:
                continue
            load = loads[node.worker_id]
            counts = (
                ("max_inflight_requests", self.inflight.get(node.worker_id, 0)),
                ("max_waiting_requests", load.get("waiting_requests")),
                ("max_pending_prefill_tokens", load.get("pending_prefill_tokens")),
            )
            if any(
                getattr(limits, k) is not None and v is not None and v >= getattr(limits, k)
                for k, v in counts
            ):
                continue
            if any(
                self._at_share(load, key, used, cap, limits)
                for key, used, cap in (
                    ("max_running_usage", "running_requests", "running_capacity"),
                    ("max_kv_usage", "kv_tokens", "kv_capacity"),
                )
            ):
                continue
            result.append(node)
        return sorted(result, key=lambda n: n.worker_id)

    @staticmethod
    def _at_share(load, key, used, cap, limits):
        threshold = getattr(limits, key)
        return (
            threshold is not None
            and used in load
            and load.get(cap, 0) > 0
            and load[used] / load[cap] >= threshold
        )

    def _better(self, left, right, role, loads):
        a, b = loads[left.worker_id], loads[right.worker_id]
        required = ("waiting_requests", "running_requests")
        if role == "prefill":
            required = ("pending_prefill_tokens", *required)

        def rank(node, load):
            values = []
            if all(k in a and k in b for k in required):
                if (
                    role == "prefill"
                    and "estimated_prefill_queue_ms" in a
                    and "estimated_prefill_queue_ms" in b
                ):
                    values.append(load["estimated_prefill_queue_ms"])
                values.extend(load[k] for k in required)
                if role == "decode":
                    if (
                        a.get("kv_capacity", 0) > 0
                        and b.get("kv_capacity", 0) > 0
                        and "kv_tokens" in a
                        and "kv_tokens" in b
                    ):
                        values.append(load["kv_tokens"] / load["kv_capacity"])
                    if "kv_tokens" in a and "kv_tokens" in b:
                        values.append(load["kv_tokens"])
            return (*values, self.inflight.get(node.worker_id, 0))

        return right if rank(right, b) < rank(left, a) else left

    def _pick(self, nodes, policy, role, bucket, loads):
        if not nodes:
            raise NoHealthyWorker(f"no admissible {role} Worker in bucket {bucket.id}")
        if policy == "round_robin":
            key = (bucket.id, role)
            index = self.cursors.get(key, 0)
            self.cursors[key] = index + 1
            return nodes[index % len(nodes)]
        if policy == "random":
            return self.random.choice(nodes)
        if len(nodes) == 1:
            return nodes[0]
        left, right = self.random.sample(nodes, 2)
        return self._better(left, right, role, loads)

    def select_prefill(self, exclude=(), *, request, bucket, prefix_hits=None):
        with self.lock:
            loads = self._loads()
            nodes = self.candidates("prefill", request, bucket, exclude, loads=loads)
            policy = bucket.prefill.policy or self.config.policy
            if nodes and any(
                n.health.get("transfer_mode") == "shm_dram" or n.health.get("shared_dram")
                for n in nodes
            ):
                decodes = self.candidates("decode", request, bucket, loads=loads)
                nodes = [p for p in nodes if any(self._shm_pair(p, d) for d in decodes)]
                if not nodes:
                    raise NoHealthyWorker("no same-host shared-pool P/D pair is admissible")
                if len({n.health.get("numa_id", 0) for n in nodes + decodes}) > 1:
                    return self._select_shm_prefill(
                        nodes, policy, request, bucket, loads, prefix_hits
                    )
            if policy != "cache_aware":
                return self._pick(nodes, policy, "prefill", bucket, loads)
            detailed = prefix_hits or {}
            hits = {key: getattr(value, "total", value) for key, value in detailed.items()}
            restore_cost = {
                key: getattr(value, "restore_cost_ms", 0) for key, value in detailed.items()
            }
            owners = [
                n
                for n in nodes
                if hits.get(n.worker_id, 0) >= self.config.cache_min_tokens
                and hits[n.worker_id] / len(request.prompt_ids) >= self.config.cache_min_ratio
            ]
            if not owners:
                return self._pick(nodes, "power_of_two", "prefill", bucket, loads)
            best = owners[0]
            for node in owners[1:]:
                if hits[node.worker_id] > hits[best.worker_id] or (
                    hits[node.worker_id] == hits[best.worker_id]
                    and (
                        restore_cost.get(node.worker_id, 0) < restore_cost.get(best.worker_id, 0)
                        or (
                            restore_cost.get(node.worker_id, 0)
                            == restore_cost.get(best.worker_id, 0)
                            and self._better(best, node, "prefill", loads) is node
                        )
                    )
                ):
                    best = node
            alternatives = [n for n in nodes if n is not best]
            if self.config.affinity_mode == "balanced" and alternatives:
                other = self._pick(alternatives, "power_of_two", "prefill", bucket, loads)
                a, b = loads[best.worker_id], loads[other.worker_id]
                if "pending_prefill_tokens" in a and "pending_prefill_tokens" in b:

                    def reusable(node):
                        # HBM may cover all prompt pages, but Prefill still samples logits.
                        page = node.health.get("layout", {}).get("page_size", 1)
                        limit = (len(request.prompt_ids) - 1) // page * page
                        return min(hits.get(node.worker_id, 0), limit)

                    x = (
                        a["pending_prefill_tokens"]
                        + len(request.prompt_ids)
                        - reusable(best)
                        + restore_cost.get(best.worker_id, 0)
                        / self.config.dram_prefill_ms_per_token
                    )
                    y = (
                        b["pending_prefill_tokens"]
                        + len(request.prompt_ids)
                        - reusable(other)
                        + restore_cost.get(other.worker_id, 0)
                        / self.config.dram_prefill_ms_per_token
                    )
                    if (
                        x - y > self.config.affinity_load_gap
                        and x > y * self.config.affinity_load_factor
                    ):
                        return other
            return best

    @staticmethod
    def _shm_pair(prefill, decode):
        p, d = prefill.health, decode.health
        from .shared_dram_routing import compatible_history

        if not compatible_history(p, d):
            return False
        if p.get("transfer_mode") != "shm_dram" and d.get("transfer_mode") != "shm_dram":
            return p.get("transfer_mode") == d.get("transfer_mode")
        return (
            p.get("transfer_mode") == d.get("transfer_mode") == "shm_dram"
            and p.get("host_id")
            and p.get("host_id") == d.get("host_id")
            and p.get("shm_pool")
            and p.get("shm_pool") == d.get("shm_pool")
        )

    def _decode_cost(self, node, prefill, loads):
        load = loads[node.worker_id]
        pressure = (
            load.get("waiting_requests", 0)
            + load.get("running_requests", 0)
            + self.inflight.get(node.worker_id, 0)
        )
        if load.get("kv_capacity", 0):
            pressure += load.get("kv_tokens", 0) / load["kv_capacity"]
        cross = prefill.health.get("numa_id", 0) != node.health.get("numa_id", 0)
        return pressure + (self.config.pd_cross_numa_penalty if cross else 0)

    def _select_shm_prefill(self, nodes, policy, request, bucket, loads, prefix_hits):
        decodes = self.candidates("decode", request, bucket, loads=loads)
        pairs = {p.worker_id: [d for d in decodes if self._shm_pair(p, d)] for p in nodes}
        nodes = [p for p in nodes if pairs[p.worker_id]]
        if not nodes:
            raise NoHealthyWorker("no same-host shared-pool P/D pair is admissible")
        if policy in ("random", "round_robin"):
            return self._pick(nodes, policy, "prefill", bucket, loads)
        detailed = prefix_hits or {}

        def score(node):
            value = detailed.get(node.worker_id, 0)
            hit = getattr(value, "total", value) if policy == "cache_aware" else 0
            if (
                hit < self.config.cache_min_tokens
                or hit / len(request.prompt_ids) < self.config.cache_min_ratio
            ):
                hit = 0
            page = node.health.get("layout", {}).get("page_size", 1)
            hit = min(hit, (len(request.prompt_ids) - 1) // page * page)
            load = loads[node.worker_id]
            queue_ms = load.get(
                "estimated_prefill_queue_ms",
                load.get("pending_prefill_tokens", 0) * self.config.dram_prefill_ms_per_token,
            )
            cost = queue_ms + self.inflight.get(node.worker_id, 0)
            cost += (len(request.prompt_ids) - hit) * self.config.dram_prefill_ms_per_token
            cost += getattr(value, "restore_cost_ms", 0) if hit else 0
            return cost + min(self._decode_cost(d, node, loads) for d in pairs[node.worker_id])

        # The cost includes the best admissible partner: cache savings may outweigh
        # a cross-socket hop, while a busy local D does not trap a request.
        return min(nodes, key=lambda n: (score(n), n.worker_id))

    def select_decode(self, exclude=(), *, request, bucket, prefill=None):
        with self.lock:
            loads = self._loads()
            nodes = self.candidates("decode", request, bucket, exclude, loads=loads)
            if prefill is not None and (
                prefill.health.get("transfer_mode") == "shm_dram"
                or prefill.health.get("shared_dram")
            ):
                nodes = [n for n in nodes if self._shm_pair(prefill, n)]
                if not nodes:
                    raise NoHealthyWorker("no same-host shared-pool Decode Worker")
                if any(
                    n.health.get("numa_id", 0) != prefill.health.get("numa_id", 0) for n in nodes
                ):
                    return min(
                        nodes, key=lambda n: (self._decode_cost(n, prefill, loads), n.worker_id)
                    )
            return self._pick(
                nodes,
                bucket.decode.policy or self.config.decode_policy,
                "decode",
                bucket,
                loads,
            )

    def acquire_prefill(self, **kwargs):
        with self.lock:
            node = self.select_prefill(**kwargs)
            self.inflight[node.worker_id] = self.inflight.get(node.worker_id, 0) + 1
            return node

    def acquire_decode(self, exclude=(), **kwargs):
        with self.lock:
            node = self.select_decode(exclude, **kwargs)
            self.inflight[node.worker_id] = self.inflight.get(node.worker_id, 0) + 1
            return node

    def release(self, node):
        if node is not None:
            with self.lock:
                count = self.inflight.get(node.worker_id, 0)
                if count < 1:
                    raise RuntimeError("routing occupancy released twice")
                if count == 1:
                    self.inflight.pop(node.worker_id)
                else:
                    self.inflight[node.worker_id] = count - 1

    def record(self, request_id, prefill, decode, attempt, event="route", *, bucket=None):
        logger.info(
            "pd_route %s",
            json.dumps(
                {
                    "event": event,
                    "request_id": request_id,
                    "prefill_id": prefill.worker_id,
                    "decode_id": decode.worker_id,
                    "attempt": attempt,
                    "bucket": bucket.id if bucket else None,
                    "prefill_policy": (bucket.prefill.policy if bucket else None)
                    or self.config.policy,
                    "decode_policy": (bucket.decode.policy if bucket else None)
                    or self.config.decode_policy,
                },
                separators=(",", ":"),
            ),
        )


def add_policy_args(parser):
    parser.add_argument("--pd-shared-dram-remote-h2d-factor", type=float, default=1.5)
    parser.add_argument(
        "--pd-cross-numa-penalty",
        type=float,
        default=4.0,
        help="Additional shared-relay cross-socket routing cost (ms equivalent)",
    )
    parser.add_argument("--routing-policy", "--policy", choices=POLICIES, default="power_of_two")
    parser.add_argument(
        "--decode-policy",
        choices=[p for p in POLICIES if p != "cache_aware"],
        default="power_of_two",
    )
    parser.add_argument("--bucket-config")
    parser.add_argument("--routing-load-max-age", type=float, default=6.0)
    parser.add_argument("--cache-affinity-min-matched-tokens", type=int, default=1024)
    parser.add_argument("--cache-affinity-min-match-ratio", type=float, default=0.0)
    parser.add_argument("--affinity-mode", choices=("prefer", "balanced"), default="prefer")
    parser.add_argument("--affinity-load-factor", type=float, default=2.0)
    parser.add_argument("--affinity-load-gap", type=int, default=1024)
    parser.add_argument("--prefix-index-max-entries", type=int, default=262144)
    parser.add_argument("--prefix-index-max-worker-entries", type=int, default=65536)
    parser.add_argument("--prefix-index-max-age", type=float, default=6.0)
    parser.add_argument("--prefix-sync-interval", type=float, default=0.2)
    parser.add_argument("--min-remote-pull-tokens", type=int, default=1024)
    parser.add_argument("--dram-h2d-gbps", type=float, default=12.0)
    parser.add_argument("--dram-rdma-gbps", type=float, default=10.0)
    parser.add_argument("--dram-prefill-ms-per-token", type=float, default=0.01)
    parser.add_argument("--dram-max-pull-bytes", type=int, default=64 * 1024**2)
    for name in (
        "max-worker-inflight-requests",
        "max-waiting-requests",
        "max-pending-prefill-tokens",
    ):
        parser.add_argument("--" + name, type=int)
    for name in ("max-running-usage", "max-kv-usage"):
        parser.add_argument("--" + name, type=float)


def policy_config(args):
    return RoutingConfig(
        policy=args.routing_policy,
        decode_policy=args.decode_policy,
        admission=AdmissionLimits(
            max_inflight_requests=args.max_worker_inflight_requests,
            max_waiting_requests=args.max_waiting_requests,
            max_pending_prefill_tokens=args.max_pending_prefill_tokens,
            max_running_usage=args.max_running_usage,
            max_kv_usage=args.max_kv_usage,
        ),
        buckets=load_buckets(args.bucket_config) if args.bucket_config else (RoutingBucket(),),
        load_max_age=args.routing_load_max_age,
        cache_min_tokens=args.cache_affinity_min_matched_tokens,
        cache_min_ratio=args.cache_affinity_min_match_ratio,
        affinity_mode=args.affinity_mode,
        affinity_load_factor=args.affinity_load_factor,
        affinity_load_gap=args.affinity_load_gap,
        prefix_index_max_entries=args.prefix_index_max_entries,
        prefix_index_max_worker_entries=args.prefix_index_max_worker_entries,
        prefix_index_max_age=args.prefix_index_max_age,
        prefix_sync_interval=args.prefix_sync_interval,
        min_remote_pull_tokens=args.min_remote_pull_tokens,
        dram_h2d_gbps=args.dram_h2d_gbps,
        dram_rdma_gbps=args.dram_rdma_gbps,
        dram_prefill_ms_per_token=args.dram_prefill_ms_per_token,
        dram_max_pull_bytes=args.dram_max_pull_bytes,
        pd_cross_numa_penalty=args.pd_cross_numa_penalty,
        pd_shared_dram_remote_h2d_factor=args.pd_shared_dram_remote_h2d_factor,
    )


def policy_cli_args(args):
    names = (
        "pd_cross_numa_penalty",
        "pd_shared_dram_remote_h2d_factor",
        "routing_policy",
        "decode_policy",
        "bucket_config",
        "routing_load_max_age",
        "cache_affinity_min_matched_tokens",
        "cache_affinity_min_match_ratio",
        "affinity_mode",
        "affinity_load_factor",
        "affinity_load_gap",
        "prefix_index_max_entries",
        "prefix_index_max_worker_entries",
        "prefix_index_max_age",
        "prefix_sync_interval",
        "min_remote_pull_tokens",
        "dram_h2d_gbps",
        "dram_rdma_gbps",
        "dram_prefill_ms_per_token",
        "dram_max_pull_bytes",
        "max_worker_inflight_requests",
        "max_waiting_requests",
        "max_pending_prefill_tokens",
        "max_running_usage",
        "max_kv_usage",
    )
    return [
        part
        for name in names
        if getattr(args, name) is not None
        for part in ("--" + name.replace("_", "-"), str(getattr(args, name)))
    ]
