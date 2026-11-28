"""Linux NUMA discovery. Inventory injection does not initialize CUDA."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def host_id():
    # Shared memory is only valid within this host's current boot.
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return hashlib.sha256(boot.encode()).hexdigest()[:32]


def gpu_numa_nodes(gpus, *, override=None, sysfs=Path("/sys/bus/pci/devices")):
    """MINISGL_GPU_NUMA_MAP is a JSON {CUDA-visible-id: OS-numa-node} map."""
    override = override if override is not None else os.environ.get("MINISGL_GPU_NUMA_MAP")
    if override is not None:
        mapping = json.loads(override) if isinstance(override, str) else override
        if (
            not isinstance(mapping, dict)
            or any(type(v) is not int or v < -1 for v in mapping.values())
            or any(g not in mapping for g in gpus)
        ):
            raise ValueError("GPU NUMA override must map every visible GPU to an integer >= -1")
        return {g: max(0, mapping[g]) for g in gpus}
    mapping = dict.fromkeys(gpus, 0)
    if shutil.which("nvidia-smi") is None:
        return mapping
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        for line in output.splitlines():
            index, uuid, bdf = [s.strip() for s in line.split(",")]
            # nvidia-smi can print an eight-digit PCI domain; sysfs uses four.
            domain, bus, dev = bdf.lower().split(":")
            path = sysfs / f"{int(domain, 16):04x}:{bus}:{dev}" / "numa_node"
            try:
                node = max(0, int(path.read_text().strip()))
            except (OSError, ValueError):
                node = 0
            for gpu in gpus:
                if gpu in (index, uuid) or uuid.startswith(gpu):
                    mapping[gpu] = node
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return mapping


def mixed_gpu_layout(gpus, nodes, n, m):
    """Spread P first, then D, across sockets; leave at least one D with each P."""
    groups = {node: [g for g in gpus if nodes[g] == node] for node in sorted(set(nodes.values()))}
    if len(groups) <= 1:
        return gpus[:n], gpus[n : n + m]
    pg, dg = [], []
    for _ in range(n):
        available = [node for node, group in groups.items() if group]
        node = min(available, key=lambda node: (sum(nodes[g] == node for g in pg), node))
        pg.append(groups[node].pop(0))
    for _ in range(m):
        available = [node for node, group in groups.items() if group]
        node = min(
            available,
            key=lambda node: (
                (
                    0
                    if any(nodes[g] == node for g in pg) and not any(nodes[g] == node for g in dg)
                    else 1
                ),
                sum(nodes[g] == node for g in dg),
                node,
            ),
        )
        dg.append(groups[node].pop(0))
    return pg, dg


def host_numa_nodes(*, override=None):
    """Host nodes, including a socket with no selected GPU (overflow still uses it)."""
    override = override if override is not None else os.environ.get("MINISGL_NUMA_NODES")
    if override is not None:
        nodes = (
            [int(n) for n in override.split(",")] if isinstance(override, str) else list(override)
        )
    else:
        nodes = [int(p.name[4:]) for p in Path("/sys/devices/system/node").glob("node[0-9]*")]
    nodes = sorted(set(nodes)) or [0]
    if any(type(n) is not int or n < 0 for n in nodes):
        raise ValueError("NUMA inventory must contain nonnegative OS node ids")
    return nodes


def node_cpus(node):
    cpus = set()
    text = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    for part in text.split(","):
        ends = [int(x) for x in part.split("-")]
        cpus.update(range(ends[0], ends[-1] + 1))
    return cpus & os.sched_getaffinity(0)


def bind_cpu(node):
    cpus = node_cpus(node)
    if not cpus:
        raise RuntimeError(f"NUMA {node} has no CPU available in this process's cpuset")
    os.sched_setaffinity(0, cpus)


def _numa_lib():
    name = ctypes.util.find_library("numa")
    if not name:
        raise RuntimeError("NUMA binding requires Linux libnuma (install libnuma1)")
    return ctypes.CDLL(name, use_errno=True)


def _mask(node):
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    mask = (ctypes.c_ulong * (node // bits + 1))()
    mask[node // bits] = 1 << (node % bits)
    return mask


def bind_memory(address, size, node):
    lib = _numa_lib()
    lib.mbind.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
        ctypes.c_uint,
    ]
    if lib.mbind(address, size, 2, _mask(node), node + 1, 0):  # MPOL_BIND
        raise OSError(ctypes.get_errno(), f"mbind(MPOL_BIND) failed for NUMA {node}")


def bind_worker(node):
    """Run before importing the model or allocating CUDA/host pools."""
    bind_cpu(node)
    lib = _numa_lib()
    lib.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
    if lib.set_mempolicy(2, _mask(node), node + 1):
        raise OSError(ctypes.get_errno(), f"set_mempolicy(MPOL_BIND) failed for NUMA {node}")
