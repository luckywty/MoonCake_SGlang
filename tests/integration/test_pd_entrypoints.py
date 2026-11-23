"""CPU integration tests. Use the real imports and config hierarchy, without starting CUDA."""

import inspect
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from minisgl.core import Req, SamplingParams
from minisgl.disagg.protocol import KVLayout, KVTransferMetadata, RequestHandoff
from minisgl.disagg.role import WorkerRole
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.server.args import parse_args

'''用于验证：新增的 P/D 角色和交接协议，是否正确接入 mini-SGLang 原有的配置、Scheduler、命令行和请求对象。'''
def config(**kwargs):
    return SchedulerConfig("local-model", DistributedInfo(0, 1), torch.bfloat16, **kwargs)


def test_programmatic_configuration_and_replace():
    unified = config()
    assert unified.worker_role is WorkerRole.UNIFIED
    assert config(worker_role="decode").worker_role is WorkerRole.DECODE
    assert replace(unified, worker_role="prefill").worker_role is WorkerRole.PREFILL
    assert (
        inspect.signature(SchedulerConfig).parameters["worker_role"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    with pytest.raises(ValueError):
        config(worker_role="typo")


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_scheduler_rejects_reserved_role_before_engine(role):
    with patch("minisgl.engine.Engine", side_effect=AssertionError("Engine constructed")) as engine:
        with pytest.raises(NotImplementedError, match="Phase 2"):
            Scheduler(config(worker_role=role))
        engine.assert_not_called()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_cli_rejects_unsupported_tp_before_auto_dtype_or_download(role):
    with (
        patch("minisgl.utils.cached_load_hf_config") as load_config,
        pytest.raises(SystemExit) as exit_info,
    ):
        parse_args(
            [
                "--model",
                "/nonexistent/model",
                "--model-source",
                "modelscope",
                "--worker-role",
                role,
                "--tp-size",
                "2",
            ]
        )
    assert exit_info.value.code == 2
    load_config.assert_not_called()


def test_cli_default_and_explicit_unified():
    base = ["--model", "/local/model", "--dtype", "bfloat16"]
    default, default_shell = parse_args(base)
    explicit, explicit_shell = parse_args(base + ["--worker-role", "unified"])
    assert default == explicit
    assert default.worker_role is WorkerRole.UNIFIED
    assert default_shell is explicit_shell is False


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_split_cli_dispatches_to_real_worker_entrypoint(role):
    from minisgl.server.launch import launch_server

    arguments = ["minisgl", "--model", "/local/model", "--dtype", "bfloat16", "--worker-role", role]
    with (
        patch.object(sys, "argv", arguments),
        patch("minisgl.disagg.worker.launch_split_worker") as worker,
        patch("minisgl.server.api_server.run_api_server") as unified,
    ):
        launch_server()
    worker.assert_called_once()
    assert worker.call_args.args[0].worker_role is WorkerRole(role)
    unified.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        ["--pd-timeout", "nan"],
        ["--pd-timeout", "inf"],
        ["--pd-timeout", "0"],
        ["--pd-layers-per-chunk", "0"],
        ["--pd-staging-slots", "1"],
        ["--pd-staging-slots", "5"],
        ["--pd-data-port", "1919"],
        ["--pd-data-port", "1920"],
        ["--pd-data-port", "0"],
        ["--host", "0.0.0.0"],
        ["--shell-mode"],
        ["--dummy-weight"],
    ],
)
def test_split_cli_validation_before_model_loading(extra):
    with patch("minisgl.utils.cached_load_hf_config") as model_config:
        with pytest.raises(SystemExit) as result:
            parse_args(["--model", "/missing", "--worker-role", "decode", *extra])
        assert result.value.code == 2
        model_config.assert_not_called()


def test_prefill_wildcard_does_not_require_decode_advertisement():
    args, _ = parse_args(
        [
            "--model",
            "/local/model",
            "--dtype",
            "bfloat16",
            "--worker-role",
            "prefill",
            "--host",
            "0.0.0.0",
            "--port",
            "65534",
        ]
    )
    assert args.worker_role is WorkerRole.PREFILL
    assert args.pd_advertise_host is None


def test_real_req_first_token_boundary():
    # Same methods and order as Engine.forward_batch and Scheduler._process_last_data.
    req = Req(torch.tensor([1, 2, 3], dtype=torch.int32), 0, 0, 4, 7, SamplingParams(), None)
    req.complete_one()
    req.append_host(torch.tensor([11], dtype=torch.int32))
    assert req.cached_len == 3
    assert req.device_len == 4
    assert req.input_ids.tolist() == [1, 2, 3, 11]
    meta = KVTransferMetadata.from_slots(
        "00000000-0000-0000-0000-000000000001",
        KVLayout("a" * 64, "bfloat16", 2, 1, 2, 1),
        (7, 4, 2),
        num_pool_pages=8,
    )
    incoming = RequestHandoff(req.uid, (1, 2, 3), 11, 4, False, 99, meta)
    assert incoming.remaining_tokens == req.remain_len == 3


def run_module(*args):
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(root / "python"), "HF_HUB_OFFLINE": "1"}
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_actual_help_entrypoint():
    result = run_module("minisgl", "--help")
    assert result.returncode == 0, result.stderr
    assert "--worker-role {unified,prefill,decode}" in result.stdout


def test_actual_invalid_split_entrypoint():
    result = run_module(
        "minisgl", "--model", "/nonexistent/model", "--worker-role", "prefill", "--tp-size", "2"
    )
    assert result.returncode == 2
    assert "TP=1" in result.stderr
    assert "Traceback" not in result.stderr


def test_actual_control_smoke_entrypoint():
    result = run_module("minisgl.disagg")
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["result"] == "PASS"
    assert output["state"] == "decode_ready"
    assert output["kv_tokens"] == 5
    assert output["remaining_tokens"] == 3
