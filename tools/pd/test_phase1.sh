#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--integration" ) ]]; then
    echo "Usage: bash tools/pd/test_phase1.sh [--integration]" >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$repo_root"
export PYTHONPATH="$repo_root/python${PYTHONPATH:+:$PYTHONPATH}"
test_python="${FLUXINFER_TEST_PYTHON:-python3}"

"$test_python" -m unittest discover -s tests/disagg -v
"$test_python" -m minisgl.disagg

if [[ "${1:-}" == "--integration" ]]; then
    "$test_python" -m pytest -o addopts= -q \
        tests/integration/test_pd_entrypoints.py \
        tests/core/test_cache_allocate.py \
        tests/misc/test_serialize.py
fi
