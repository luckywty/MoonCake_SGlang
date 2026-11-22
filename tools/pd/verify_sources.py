"""Verify the fixed upstream Git blobs and line references cited in the Phase 0 audit."""

import argparse
import re
import subprocess
from pathlib import Path

PINS = {
    "sgl-project/mini-sglang": "9a91cfafe754aa85daee49998176275667eb58f2",
    "kvcache-ai/Mooncake": "301261358eef0b13fa55645ccde4a988f1ca7505",
    "sgl-project/sglang": "4b02753381c948c42b7d8e0289bb72519f11e5f7",
}
LINK = re.compile(r"https://github\.com/([^/]+/[^/]+)/blob/(\w+)/([^#)]+)#L(\d+)(?:-L(\d+))?")
ANCHORS = {
    ("sgl-project/mini-sglang", "python/minisgl/engine/engine.py"): "self.num_pages + 1,",
    (
        "sgl-project/mini-sglang",
        "python/minisgl/scheduler/scheduler.py",
    ): "batch.out_loc = self.engine.page_table[input_mapping]",
    (
        "sgl-project/mini-sglang",
        "python/minisgl/scheduler/cache.py",
    ): "self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])",
    (
        "sgl-project/mini-sglang",
        "python/minisgl/kvcache/mha_pool.py",
    ): "(2, num_layers, num_pages, page_size, local_kv_heads, head_dim)",
    ("kvcache-ai/Mooncake", "python/mooncake/engine.pyi"): "def batch_transfer_sync_write(",
    (
        "kvcache-ai/Mooncake",
        "mooncake-integration/store/store_py.cpp",
    ): '"MooncakeDistributedStore"',
    (
        "sgl-project/sglang",
        "python/sglang/srt/disaggregation/decode.py",
    ): "class DecodePreallocQueue",
    (
        "sgl-project/sglang",
        "python/sglang/srt/managers/cache_controller.py",
    ): "class HiCacheController",
}


def verify(report: Path, roots: dict[str, Path]) -> None:
    files: dict[tuple[str, str], list[str]] = {}
    refs = LINK.findall(report.read_text(encoding="utf-8"))
    for repo, ref, path, first, last in refs:
        if ref != PINS.get(repo):
            raise ValueError(f"Unexpected source pin: {repo} {ref}")
        key = (repo, path)
        if key not in files:
            output = subprocess.check_output(["git", "-C", roots[repo], "show", f"{ref}:{path}"])
            files[key] = output.decode("utf-8").splitlines()
        if not 1 <= int(first) <= int(last or first) <= len(files[key]):
            raise ValueError(f"Invalid range: {path}:{first}-{last} (length {len(files[key])})")
    for key, needle in ANCHORS.items():
        if not any(needle in line for line in files.get(key, [])):
            raise ValueError(f"Source anchor missing: {key} {needle}")
    for repo in PINS:
        count = sum(1 for name, _ in files if name == repo)
        print(f"{repo} {PINS[repo]}: {count} files")
    print(f"PASS: {len(refs)} line references, {len(ANCHORS)} source anchors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini", type=Path, required=True)
    parser.add_argument("--moon", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    args = parser.parse_args()
    report = Path(__file__).resolve().parents[2] / "docs/pd/phase0.md"
    verify(report, dict(zip(PINS, (args.mini, args.moon, args.sglang))))


if __name__ == "__main__":
    main()
