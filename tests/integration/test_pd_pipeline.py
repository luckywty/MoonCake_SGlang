import multiprocessing as mp
import socket
from threading import Event
from uuid import uuid4

import pytest
from minisgl.disagg.client import PDClient, RemoteWorkerError
from minisgl.disagg.control import Channel, StartRequest, endpoint_address
from pd_reference import health, reference_tokens, run_cpu_worker


@pytest.fixture(scope="module") #Fixture 可以理解为：为测试函数提前创建测试所需要的环境，并在测试结束后清理环境。
def workers():
    '''启动两个独立的 CPU Worker 进程，
    分别模拟 Prefill Worker 和 Decode Worker，建立真实的进程间通信环境。'''
    context = mp.get_context("spawn")
    ready = context.Queue()
    stop = context.Event()
    processes = []
    endpoints = []
    try:
        for role in ("prefill", "decode"):
            #创建进程对象
            process = context.Process(target=run_cpu_worker, args=(role, ready, stop, 0.03))
            process.start() #真正启动子进程
            processes.append(process)   #保存进程对象
            endpoints.append(ready.get(timeout=20)) #等待 Worker 返回 endpoint
        yield tuple(endpoints)
    finally:
        stop.set()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        ready.close()


@pytest.mark.parametrize("prompt", [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7]])
@pytest.mark.parametrize("max_tokens", [1, 5])
def test_real_tcp_cpu_attention_continuation(workers, prompt, max_tokens):
    expected = reference_tokens(prompt, max_tokens)
    #这个 generate() 是 stream() 的非流式封装：把逐个产生的 token 编号收集成列表，等生成结束后一次性返回。
    actual = PDClient(*workers, timeout=5).generate(prompt, max_tokens, ignore_eos=True)
    assert actual == expected
    for endpoint in workers:
        state = health(endpoint)
        assert state["free_rows"] == 2
        assert state["free_pages"] == 32


def test_cancel_during_decode_recovers_capacity(workers):
    cancel = Event()
    iterator = PDClient(*workers, timeout=5).stream(
        [1, 2, 3, 4, 5], 30, ignore_eos=True, cancel=cancel
    )
    next(iterator)
    next(iterator)
    cancel.set()
    with pytest.raises(InterruptedError):
        next(iterator)
    iterator.close()
    assert health(workers[1])["free_pages"] == 32
    assert health(workers[1])["free_rows"] == 2
    assert PDClient(*workers, timeout=5).generate([1, 2], 3, ignore_eos=True) == reference_tokens(
        [1, 2], 3
    )


def test_disconnect_while_decode_waits_for_transfer(workers):
    start = StartRequest(str(uuid4()), 0, (1, 2, 3, 4, 5), 5, True)
    with socket.create_connection(endpoint_address(workers[0])) as p_sock:
        p = Channel(p_sock, 5)
        p.send({"op": "prefill", "request": start.to_object()})
        handoff = p.receive()["handoff"]
        with socket.create_connection(endpoint_address(workers[1])) as d_sock:
            d = Channel(d_sock, 5)
            d.send({"op": "decode", "handoff": handoff})
            assert d.receive()["status"] == "ready"
        # P and D now lose their control channels before any payload is sent.
    assert health(workers[1])["free_pages"] == 32
    assert health(workers[1])["free_rows"] == 2


def test_invalid_budget_and_token_fail_without_resource_leak(workers):
    for ids, count in (([1] * 64, 2), ([999], 4)):
        with pytest.raises(RemoteWorkerError):
            PDClient(*workers, timeout=5).generate(ids, count)
        assert health(workers[0])["free_pages"] == 32
        assert health(workers[0])["free_rows"] == 2


def test_pre_cancelled_request_does_not_contact_workers(workers):
    cancelled = Event()
    cancelled.set()
    with pytest.raises(InterruptedError):
        PDClient(*workers, timeout=5).generate([1], 3, cancel=cancelled)


def test_duplicate_attempt_rejected(workers):
    start = StartRequest(str(uuid4()), 0, (1, 2), 1, True)
    for index in range(2):
        with socket.create_connection(endpoint_address(workers[0])) as sock:
            channel = Channel(sock, 5)
            channel.send({"op": "prefill", "request": start.to_object()})
            result = channel.receive()
            assert result["status"] == ("handoff" if index == 0 else "error")


def test_first_token_eos_short_circuits_decode(workers):
    # Find a deterministic prompt whose first greedy token is the reference model EOS (31).
    prompt = next(([token] for token in range(32) if reference_tokens([token], 1) == [31]), None)
    if prompt is None:
        pytest.fail("reference fixture must include a first-token EOS case")
    assert PDClient(workers[0], "127.0.0.1:1", timeout=3).generate(prompt, 5) == [31]
    assert health(workers[1])["free_rows"] == 2
