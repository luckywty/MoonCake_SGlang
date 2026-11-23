#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 0 ]]; then
    echo "Usage: bash tools/pd/test_phase2.sh" >&2
    exit 2
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$repo_root"
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
test_python="${FLUXINFER_TEST_PYTHON:-python3}"
"$test_python" -m pytest -o addopts= -q \
    tests/disagg tests/integration/test_pd_entrypoints.py \
    tests/integration/test_pd_tensor_cache.py tests/integration/test_pd_pipeline.py \
    tests/integration/test_pd_failures.py \
    tests/integration/test_pd_gateway.py tests/integration/test_pd_gpu.py \
    tests/integration/test_pd_layered.py tests/integration/test_pd_layered_cuda.py \
    tests/integration/test_pd_batching.py tests/integration/test_pd_control_loop.py \
    tests/integration/test_pd_cluster_config.py tests/integration/test_pd_routing.py \
    tests/core/test_cache_allocate.py tests/misc/test_serialize.py
