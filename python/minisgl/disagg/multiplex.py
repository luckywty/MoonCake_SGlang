"""Bounded request channels over one persistent, framed Worker connection.

Only complete frame writes share a lock. Requests never hold a Worker-wide
lock while waiting for prefill, native drain, installation, or tokens.
"""

from __future__ import annotations

import json
import math
import queue
import select
import socket
import threading
import time
from dataclasses import dataclass
from uuid import uuid4

from .control import MAX_CONTROL_BYTES, CancelledError, Channel, endpoint_address
from .protocol import check_transfer_id

MUX_VERSION = 1


def envelope(request_id, message, *, terminal=False):
    check_transfer_id(request_id)
    if not isinstance(message, dict) or type(terminal) is not bool:
        raise ValueError("invalid multiplex message")
    return {"request_id": request_id, "message": message, "terminal": terminal}


def unpack(frame):
    if not isinstance(frame, dict) or set(frame) != {"request_id", "message", "terminal"}:
        raise ValueError("expected a request_id multiplex envelope")
    check_transfer_id(frame["request_id"])
    if not isinstance(frame["message"], dict) or type(frame["terminal"]) is not bool:
        raise ValueError("invalid multiplex message")
    return frame["request_id"], frame["message"], frame["terminal"]


@dataclass
class _Failure:
    error: BaseException


class RequestChannel:
    def __init__(self, connection, request_id, cancel):
        self.connection, self.request_id, self.cancel = connection, request_id, cancel
        self.messages = queue.Queue(connection.output_limit)
        self.finished = threading.Event()
        self.closed = False

    def send(self, message, *, timeout=None):
        if self.closed:
            raise ConnectionError("request channel is closed")
        if self.cancel is not None and self.cancel.is_set():
            raise CancelledError("request cancelled")
        self.connection.send(
            envelope(self.request_id, message), timeout=timeout, cancel=self.cancel
        )

    def receive(self, *, timeout=None):
        deadline = time.monotonic() + (self.connection.timeout if timeout is None else timeout)
        while True:
            if self.cancel is not None and self.cancel.is_set():
                raise CancelledError("request cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("multiplex request timed out")
            try:
                message = self.messages.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                if self.finished.is_set():
                    raise ConnectionError("request channel ended before its expected response")
                continue
            if isinstance(message, _Failure):
                raise message.error
            return message

    def fail(self, error):
        # Make room for a terminal error, including on an overflowing stream.
        while True:
            try:
                self.messages.put_nowait(_Failure(error))
                break
            except queue.Full:
                try:
                    self.messages.get_nowait()
                except queue.Empty:
                    pass
        self.finished.set()

    def close(self, *, cancel=True):
        if self.closed:
            return
        self.closed = True
        if cancel and not self.finished.is_set():
            self.connection.cancel(self.request_id)
        self.connection.detach(self)


class WorkerConnection:
    def __init__(
        self,
        endpoint,
        *,
        timeout=30.0,
        max_requests=144,
        output_limit=128,
        connect=None,
        connect_timeout=None,
    ):
        if not math.isfinite(timeout) or timeout <= 0 or max_requests < 1 or output_limit < 1:
            raise ValueError("connection timeouts and queue limits must be positive")
        self.endpoint, self.timeout = endpoint, timeout
        self.max_requests, self.output_limit = max_requests, output_limit
        self.lock, self.write_lock = threading.Lock(), threading.Lock()
        self.requests = {}
        self.pending_cancels = set()
        self.stopped = threading.Event()
        self.failure = None
        connector = connect or socket.create_connection
        initial_timeout = timeout if connect_timeout is None else connect_timeout
        self.socket = connector(endpoint_address(endpoint), timeout=initial_timeout)
        self.channel = Channel(self.socket, initial_timeout)
        try:
            handshake_id = str(uuid4())
            self.channel.send(
                {"op": "multiplex", "version": MUX_VERSION, "request_id": handshake_id}
            )
            if self.channel.receive() != {
                "status": "multiplex",
                "version": MUX_VERSION,
                "request_id": handshake_id,
            }:
                raise ValueError("Worker does not support multiplex protocol version 1")
        except BaseException:
            self.socket.close()
            raise
        self.channel = Channel(self.socket, timeout)
        self.reader = threading.Thread(target=self._read, name="pd-mux-reader", daemon=True)
        self.reader.start()

    def open(self, request_id, cancel=None):
        check_transfer_id(request_id)
        with self.lock:
            if self.stopped.is_set():
                raise ConnectionError("Worker connection is closed") from self.failure
            if request_id in self.requests:
                raise ValueError("request_id is already active on this connection")
            if len(self.requests) >= self.max_requests:
                raise RuntimeError("multiplex connection request budget is busy")
            request = RequestChannel(self, request_id, cancel)
            self.requests[request_id] = request
            return request

    def detach(self, request):
        with self.lock:
            if self.requests.get(request.request_id) is request:
                self.requests.pop(request.request_id)

    def send(self, message, *, timeout=None, cancel=None):
        # Reject application/size errors before touching shared framing. Such an
        # error belongs to one request, not to every stream on the connection.
        if (
            len(json.dumps(message, separators=(",", ":"), allow_nan=False).encode())
            > MAX_CONTROL_BYTES
        ):
            raise ValueError("control message exceeds size limit")
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            if cancel is not None and cancel.is_set():
                raise CancelledError("request cancelled while awaiting a frame writer")
            if self.stopped.is_set():
                raise ConnectionError("Worker connection is closed") from self.failure
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Worker frame writer is busy")
            if self.write_lock.acquire(timeout=min(remaining, 0.1)):
                break
        try:
            if self.stopped.is_set():
                raise ConnectionError("Worker connection is closed") from self.failure
            if cancel is not None and cancel.is_set():
                raise CancelledError("request cancelled before frame submission")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Worker frame writer deadline expired")
            try:
                # Once a frame starts, finish it atomically. Per-request cancel
                # cannot interrupt its bytes and corrupt peer requests' framing.
                Channel(self.socket, remaining).send(message)
            except (OSError, RuntimeError, ValueError) as exc:
                self._fail(exc)
                raise
        finally:
            self.write_lock.release()

    def cancel(self, request_id):
        with self.lock:
            if self.stopped.is_set():
                return
            overflow = len(self.pending_cancels) >= self.max_requests
            if not overflow:
                self.pending_cancels.add(request_id)
        if overflow:
            self._fail(ConnectionError("pending cancel budget is full"))
            return
        self._flush_cancel()

    def _flush_cancel(self):
        # If another frame owns the writer, keep the cancel queued. The reader
        # retries it between incoming frames, even after the local stream closes.
        if not self.write_lock.acquire(blocking=False):
            return
        try:
            with self.lock:
                if self.stopped.is_set() or not self.pending_cancels:
                    return
                request_id = self.pending_cancels.pop()
            Channel(self.socket, min(self.timeout, 0.2)).send(
                envelope(request_id, {"op": "cancel", "transfer_id": request_id})
            )
        except (OSError, RuntimeError, ValueError) as exc:
            # A partial frame cannot be followed by another request's frame.
            self._fail(exc)
        finally:
            self.write_lock.release()

    def _read(self):
        try:
            while not self.stopped.is_set():
                self._flush_cancel()
                readable, _, _ = select.select([self.socket], [], [], 0.1)
                if not readable:
                    continue
                request_id, message, terminal = unpack(self.channel.receive())
                with self.lock:
                    request = self.requests.get(request_id)
                    if terminal:
                        self.pending_cancels.discard(request_id)
                if request is None:
                    continue  # late response to a closed request; never reroute it
                try:
                    request.messages.put_nowait(message)
                except queue.Full:
                    request.fail(ConnectionError("request output queue is full"))
                    self.cancel(request_id)
                if terminal:
                    request.finished.set()
        except (OSError, RuntimeError, ValueError) as exc:
            if not self.stopped.is_set():
                self._fail(exc)

    def _fail(self, error):
        with self.lock:
            self.failure = error
            self.stopped.set()
            requests = list(self.requests.values())
        for request in requests:
            request.fail(ConnectionError(f"Worker connection failed: {error}"))
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def close(self):
        self._fail(ConnectionError("Worker connection closed"))
        if self.reader is not threading.current_thread():
            self.reader.join(timeout=self.timeout + 0.2)
        self.socket.close()
