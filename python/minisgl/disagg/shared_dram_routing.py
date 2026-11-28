"""Metadata-only shared-cache hints; actual reuse always requires Worker pins."""

from __future__ import annotations

import threading
from pathlib import Path

from .numa import host_id
from .prefix_hash import PrefixHashSpec
from .shared_dram_directory import SharedDramClient
from .shared_dram_pool import SharedDramManifest
from .tier_index import TierHit


def shared_identity(health):
    info = health.get("shared_dram")
    if not isinstance(info, dict) or info.get("mode") != "shared_shm":
        return None
    return info.get("host_id"), info.get("pool")


def compatible_history(prefill, decode):
    a, b = shared_identity(prefill), shared_identity(decode)
    return a == b and (a is None or all(isinstance(v, str) and v for v in a))


class SharedHistoryHints:
    def __init__(self):
        self.lock, self.clients = threading.Lock(), {}

    @staticmethod
    def spec_for(nodes):
        for node in nodes:
            info = node.health.get("shared_dram")
            if isinstance(info, dict) and info.get("host_id") == host_id():
                try:
                    return PrefixHashSpec(info["spec"])
                except (KeyError, TypeError, ValueError):
                    continue
        return None

    def hits(self, nodes, prepared, previous, config):
        result, lookups = {}, {}
        with self.lock:
            for node in nodes:
                key = None
                info = node.health.get("shared_dram")
                if not isinstance(info, dict) or info.get("host_id") != host_id():
                    continue
                try:
                    manifest = SharedDramManifest.from_json(Path(info["manifest"]).read_text())
                    if (manifest.host_id, manifest.name) != shared_identity(node.health):
                        continue
                    spec = PrefixHashSpec(info["spec"])
                    if spec.domain_id != prepared.domain_id:
                        continue
                    key = manifest.name, spec.domain_id
                    if key not in self.clients:
                        self.clients[key] = SharedDramClient(manifest, domain_id=spec.domain_id)
                    old = previous.get(node.worker_id, 0)
                    hbm = getattr(old, "hbm_tokens", old)
                    limit = (len(prepared.token_ids) - 1) // manifest.page_size
                    hbm = min(hbm, limit * manifest.page_size)
                    start = hbm // manifest.page_size
                    query = key, start
                    if query not in lookups:
                        hashes = tuple(prepared.digests())[start:limit][:16384]
                        lookups[query] = self.clients[key].lookup_prefix(hashes)
                    pages = lookups[query]
                    local = sum(p["numa_id"] == node.health.get("numa_id", 0) for p in pages)
                    remote = len(pages) - local
                    cost = manifest.page_bytes * (
                        local + remote * config.pd_shared_dram_remote_h2d_factor
                    )
                    cost = cost / (config.dram_h2d_gbps * 1e9) * 1000
                    result[node.worker_id] = TierHit(
                        hbm,
                        local * manifest.page_size,
                        remote * manifest.page_size,
                        restore_cost_ms=cost,
                    )
                except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                    if key in self.clients:
                        # Hints own no DMA leases. A failed optional lookup may
                        # have an unknown RPC sequence; reconnect with a fresh id.
                        self.clients.pop(key).disconnect()
                    continue
        return result

    def close(self):
        with self.lock:
            for client in self.clients.values():
                try:
                    client.close(drained=True)
                except (OSError, RuntimeError, ValueError):
                    client.disconnect()
            self.clients.clear()
