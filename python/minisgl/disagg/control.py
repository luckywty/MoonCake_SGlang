"""Bounded, deadline-aware control messages shared by the RPC workers and the TCP backend."""

from __future__ import annotations

import json
import math
import select
import socket
import struct
import time
from dataclasses import dataclass
from threading import Event
from urllib.parse import urlsplit

from .protocol import _array, _int, _load, _object, _token, _wire, check_transfer_id

MAX_CONTROL_BYTES = 8 * 1024 * 1024

'''定义控制消息的收发规则、超时和取消机制'''
class CancelledError(InterruptedError):
    pass


def endpoint_address(endpoint: str) -> tuple[str, int]:
    parsed = urlsplit("//" + endpoint)
    if (
        not parsed.hostname
        or parsed.port is None
        or not 0 < parsed.port < 65536
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        raise ValueError("expected a host:port endpoint")
    return parsed.hostname, parsed.port


def endpoint_name(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _timeout(sock: socket.socket, deadline: float, cancel: Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("request cancelled")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("P/D operation timed out")
    sock.settimeout(min(remaining, 0.2))


def read_bytes(
    sock: socket.socket, length: int, deadline: float, cancel: Event | None = None
) -> bytes:
    data = bytearray(length)
    view = memoryview(data)
    offset = 0
    while offset < length:
        _timeout(sock, deadline, cancel)
        try:
            count = sock.recv_into(view[offset:])
        except socket.timeout:
            continue
        if count == 0:
            raise ConnectionError("peer closed the P/D connection")
        offset += count
    return bytes(data)


def write_bytes(
    sock: socket.socket, data: bytes | bytearray, deadline: float, cancel: Event | None = None
) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        _timeout(sock, deadline, cancel)
        try:
            count = sock.send(view[offset:])
        except socket.timeout:
            continue
        if count == 0:
            raise ConnectionError("peer closed the P/D connection")
        offset += count


class Channel:
    def __init__(self, sock: socket.socket, timeout: float, cancel: Event | None = None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.socket = sock
        self.timeout = timeout
        self.cancel = cancel

    def send(self, message: dict) -> None:
        payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
        if len(payload) > MAX_CONTROL_BYTES:
            raise ValueError("control message exceeds size limit")
        write_bytes(
            self.socket,
            struct.pack("!Q", len(payload)) + payload,
            time.monotonic() + self.timeout,
            self.cancel,
        )

    def receive(self) -> dict:
        deadline = time.monotonic() + self.timeout
        length = struct.unpack("!Q", read_bytes(self.socket, 8, deadline, self.cancel))[0]
        if length > MAX_CONTROL_BYTES:
            raise ValueError("control message exceeds size limit")
        result = _load(read_bytes(self.socket, length, deadline, self.cancel))
        if not isinstance(result, dict):
            raise ValueError("control message must be an object")
        return result

    def check_connected(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise CancelledError("request cancelled")
        readable, _, _ = select.select([self.socket], [], [], 0)
        if readable:
            if not self.socket.recv(1, socket.MSG_PEEK):
                raise CancelledError("request connection closed")
            raise ValueError("unexpected control message while computing")


@dataclass(frozen=True)
class StartRequest:
    transfer_id: str
    uid: int
    prompt_ids: tuple[int, ...]
    max_tokens: int
    ignore_eos: bool

    def __post_init__(self) -> None:
        check_transfer_id(self.transfer_id)
        _int("uid", self.uid)
        _int("max_tokens", self.max_tokens, 1)
        if not isinstance(self.prompt_ids, tuple) or not self.prompt_ids:
            raise ValueError("prompt_ids must be a nonempty tuple")
        for token in self.prompt_ids:
            _token(token)
        if type(self.ignore_eos) is not bool:
            raise ValueError("ignore_eos must be a boolean")

    def to_object(self) -> dict:
        return json.loads(_wire(self))

    @classmethod
    def from_object(cls, obj: dict) -> StartRequest:
        data = _object(cls, obj)
        data["prompt_ids"] = _array(data["prompt_ids"])
        return cls(**data)
