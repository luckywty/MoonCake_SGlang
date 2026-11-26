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
    tests/disagg tests/integration tests/core tests/misc
