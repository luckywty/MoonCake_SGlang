"""Cancellation-aware frontend request/token admission, shared by client streams."""

import math
import threading
import time
from contextlib import contextmanager

from .control import CancelledError


class AdmissionError(RuntimeError):
    pass


class AdmissionBudget:
    def __init__(self, max_requests=128, max_tokens=1048576):
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("max_inflight_requests must be positive")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_inflight_tokens must be positive")
        self.max_requests, self.max_tokens = max_requests, max_tokens
        self.requests = self.tokens = 0
        self.condition = threading.Condition()

    @contextmanager
    def reserve(self, tokens, timeout, cancel=None):
        if tokens > self.max_tokens:
            raise AdmissionError("request exceeds frontend token budget")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("admission timeout must be positive")
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if cancel is not None and cancel.is_set():
                    raise CancelledError("request cancelled during admission")
                if self.requests < self.max_requests and self.tokens + tokens <= self.max_tokens:
                    self.requests += 1
                    self.tokens += tokens
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AdmissionError("frontend admission budget is busy")
                self.condition.wait(min(remaining, 0.1))
        try:
            yield
        finally:
            with self.condition:
                self.requests -= 1
                self.tokens -= tokens
                self.condition.notify_all()
