"""P/D prompt preparation in a bounded, spawned CPU process (no model engine)."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .prefix_hash import PreparedPrefix, PrefixHashSpec, prepare_prefix

_tokenizer = None


def _init_tokenizer(model):
    global _tokenizer
    from minisgl.utils import load_tokenizer

    _tokenizer = load_tokenizer(model)


@dataclass(frozen=True)
class PreparedPrompt:
    token_ids: tuple[int, ...]
    prefix: PreparedPrefix | None


def prepare_with_tokenizer(tokenizer, messages, prompt, spec):
    # Keep the P/D gateway's existing chat-template/special-token semantics.
    if messages:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    elif prompt is not None:
        ids = tokenizer.encode(prompt)
    else:
        raise ValueError("messages or prompt is required")
    ids = tuple(ids)
    return PreparedPrompt(ids, prepare_prefix(ids, spec) if spec is not None else None)


def _prepare(messages, prompt, manifest):
    spec = PrefixHashSpec(manifest) if manifest is not None else None
    return prepare_with_tokenizer(_tokenizer, messages, prompt, spec)


class PromptPreparer:
    def __init__(self, model, max_pending=128, timeout=120.0):
        if max_pending < 1 or timeout <= 0:
            raise ValueError("prompt preparation bounds must be positive")
        self.pool = ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("spawn"),
            initializer=_init_tokenizer, initargs=(model,),
        )
        self.slots = asyncio.BoundedSemaphore(max_pending)
        self.timeout = timeout

    async def prepare(self, messages, prompt, spec, cancel):
        deadline = time.monotonic() + self.timeout
        while True:
            if cancel.is_set():
                raise InterruptedError("prompt preparation cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("prompt preparation queue timed out")
            try:
                await asyncio.wait_for(self.slots.acquire(), 0.1)
                break
            except asyncio.TimeoutError:
                continue
        loop = asyncio.get_running_loop()
        try:
            future = self.pool.submit(_prepare, messages, prompt, spec.manifest if spec else None)
        except BaseException:
            self.slots.release()
            raise

        def release(_):
            # Cancellation does not release capacity while a CPU job is still running.
            try:
                loop.call_soon_threadsafe(self.slots.release)
            except RuntimeError:
                pass  # The application loop has already shut down.

        future.add_done_callback(release)
        wrapped = asyncio.wrap_future(future)
        # Consume late exceptions when a disconnected request no longer awaits its result.
        wrapped.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        try:
            while True:
                if cancel.is_set():
                    raise InterruptedError("prompt preparation cancelled")
                if time.monotonic() >= deadline:
                    raise TimeoutError("prompt preparation timed out")
                done, _ = await asyncio.wait((wrapped,), timeout=0.1)
                if done:
                    return wrapped.result()
        except BaseException:
            future.cancel()
            raise

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
