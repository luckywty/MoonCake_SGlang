"""Bounded prompt preparation without spawning processes or downloading a tokenizer."""

import asyncio
import threading
from concurrent.futures import Future

import pytest
from minisgl.disagg import prompt_prepare
from minisgl.disagg.prompt_prepare import PreparedPrompt, PromptPreparer
from pd_cache_reference import cpu_radix_compare, prefix_owner  # noqa: F401


class ControlledExecutor:
    def __init__(self, **kwargs):
        assert kwargs["max_workers"] == 1
        assert kwargs["mp_context"].get_start_method() == "spawn"
        assert kwargs["initargs"] == ("test-model",)
        assert kwargs["initializer"] is prompt_prepare._init_tokenizer
        self.jobs = []
        self.closed = False
        self.submit_error = None

    def submit(self, fn, *args):
        if self.submit_error is not None:
            raise self.submit_error
        assert fn is prompt_prepare._prepare
        future = Future()
        # A running process cannot be cancelled; its queue slot stays occupied.
        assert future.set_running_or_notify_cancel()
        self.jobs.append((future, args))
        return future

    def shutdown(self, *, wait, cancel_futures):
        assert wait and cancel_futures
        assert all(f.done() for f, _ in self.jobs)
        self.closed = True


async def until(predicate):
    deadline = asyncio.get_running_loop().time() + 2
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "preparation did not make progress"
        await asyncio.sleep(0.001)


@pytest.mark.usefixtures("cpu_radix_compare")
@pytest.mark.parametrize("failure", ["cancel", "timeout"])
def test_cancelled_or_timed_out_preparation_holds_its_slot_until_the_cpu_job_finishes(
    monkeypatch, failure
):
    monkeypatch.setattr(prompt_prepare, "ProcessPoolExecutor", ControlledExecutor)
    spec = prefix_owner().spec

    async def scenario():
        preparer = PromptPreparer("test-model", max_pending=1, timeout=0.05)
        cancel = threading.Event()
        first = asyncio.create_task(preparer.prepare(None, "first", spec, cancel))
        await until(lambda: len(preparer.pool.jobs) == 1)
        old_future, args = preparer.pool.jobs[0]
        assert args == (None, "first", spec.manifest)
        if failure == "cancel":
            cancel.set()
        with pytest.raises(InterruptedError if failure == "cancel" else TimeoutError):
            await first
        assert old_future.running() and preparer.slots.locked()
        # The next caller must wait for real completion, not for the old caller to exit.
        preparer.timeout = 1
        second = asyncio.create_task(preparer.prepare(None, "second", None, threading.Event()))
        await asyncio.sleep(0.01)
        assert len(preparer.pool.jobs) == 1
        old_future.set_result(PreparedPrompt((1,), None))
        await until(lambda: len(preparer.pool.jobs) == 2)
        new_future, args = preparer.pool.jobs[1]
        assert args == (None, "second", None)
        expected = PreparedPrompt((2, 3), None)
        new_future.set_result(expected)
        assert await second == expected
        await until(lambda: not preparer.slots.locked())
        preparer.close()
        assert preparer.pool.closed

    asyncio.run(scenario())


def test_failed_process_submission_releases_queue_capacity_for_the_next_request(monkeypatch):
    monkeypatch.setattr(prompt_prepare, "ProcessPoolExecutor", ControlledExecutor)

    async def scenario():
        preparer = PromptPreparer("test-model", max_pending=1, timeout=1)
        preparer.pool.submit_error = RuntimeError("injected process submission failure")
        with pytest.raises(RuntimeError, match="injected process submission failure"):
            await preparer.prepare(None, "first", None, threading.Event())
        assert not preparer.slots.locked()
        preparer.pool.submit_error = None
        task = asyncio.create_task(preparer.prepare(None, "retry", None, threading.Event()))
        await until(lambda: len(preparer.pool.jobs) == 1)
        expected = PreparedPrompt((4,), None)
        preparer.pool.jobs[0][0].set_result(expected)
        assert await task == expected
        await until(lambda: not preparer.slots.locked())
        preparer.close()

    asyncio.run(scenario())
