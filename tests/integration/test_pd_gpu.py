"""Opt-in actual GPU acceptance. CPU reference success cannot satisfy this test."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("FLUXINFER_PD_GPU_MODEL"),
    reason="requires CUDA GPU, full runtime dependencies and FLUXINFER_PD_GPU_MODEL",
)
def test_actual_gpu_greedy_pd_continuation(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/pd/verify_phase2_gpu.py"),
            "--model",
            os.environ["FLUXINFER_PD_GPU_MODEL"],
            "--output-dir",
            str(tmp_path),
            "--prefill-device",
            os.environ.get("FLUXINFER_PD_PREFILL_DEVICE", "0"),
            "--decode-device",
            os.environ.get("FLUXINFER_PD_DECODE_DEVICE", "0"),
            "--transfer-backend", os.environ.get("FLUXINFER_PD_TRANSFER_BACKEND", "tcp"),
            "--layers-per-chunk", os.environ.get("FLUXINFER_PD_LAYERS_PER_CHUNK", "1"),
            "--mooncake-host", os.environ.get("FLUXINFER_PD_MOONCAKE_HOST", "127.0.0.1"),
            "--mooncake-device", os.environ.get("FLUXINFER_PD_MOONCAKE_DEVICE", ""),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "python")},
        text=True,
        capture_output=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "gpu_results.json").exists()
