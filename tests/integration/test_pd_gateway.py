from fastapi.testclient import TestClient
from minisgl.disagg.client import RemoteWorkerError
from minisgl.disagg.gateway import create_app


class Tokenizer:
    eos_token_id = 31

    def encode(self, text):
        return [1, 2]

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2]

    def batch_decode(self, sequences):
        return ["".join(chr(65 + token) for token in sequence) for sequence in sequences]


class Client:
    def __init__(self, error=False):
        self.error = error
        self.cancel = None

    def stream(self, ids, max_tokens, **kwargs):
        self.cancel = kwargs["cancel"]
        if self.error:
            raise RemoteWorkerError("injected remote failure")
        yield from [1, 2, 31][:max_tokens]

    def generate(self, *args, **kwargs):
        return list(self.stream(*args, **kwargs))


def test_http_nonstream_and_token_accounting():
    client = Client()
    with TestClient(create_app(client, Tokenizer(), "tiny-model")) as http:
        result = http.post(
            "/v1/chat/completions",
            json={"model": "tiny-model", "temperature": 0, "prompt": "hello", "max_tokens": 5},
        )
        assert result.status_code == 200
        body = result.json()
        assert body["choices"][0]["message"]["content"] == "BC"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["completion_tokens"] == 3
    assert client.cancel.is_set()


def test_http_stream_and_chat_template():
    with TestClient(create_app(Client(), Tokenizer(), "tiny-model")) as http:
        result = http.post(
            "/v1/chat/completions",
            json={
                "model": "tiny-model",
                "temperature": 0,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 5,
                "stream": True,
            },
        )
        assert result.status_code == 200
        assert "[DONE]" in result.text
        assert '"content": "B"' in result.text
        assert '"content": "C"' in result.text


def test_http_rejects_sampling_and_reports_remote_error():
    with TestClient(create_app(Client(), Tokenizer(), "tiny-model")) as http:
        assert (
            http.post(
                "/v1/chat/completions",
                json={"model": "tiny-model", "prompt": "hello", "temperature": 1},
            ).status_code
            == 400
        )
        assert (
            http.post(
                "/v1/chat/completions",
                json={"model": "tiny-model", "temperature": 0, "prompt": "hello", "max_tokens": 0},
            ).status_code
            == 400
        )
    with TestClient(create_app(Client(error=True), Tokenizer(), "tiny-model")) as http:
        result = http.post(
            "/v1/chat/completions",
            json={"model": "tiny-model", "temperature": 0, "prompt": "hello"},
        )
        assert result.status_code == 503
        assert "injected" in result.text


def test_first_token_eos_uses_stop_finish_reason():
    class EosClient(Client):
        def stream(self, *args, **kwargs):
            yield 31

    with TestClient(create_app(EosClient(), Tokenizer(), "tiny-model")) as http:
        result = http.post(
            "/v1/chat/completions",
            json={"model": "tiny-model", "temperature": 0, "prompt": "hello", "max_tokens": 1},
        )
        assert result.status_code == 200
        assert result.json()["choices"][0]["finish_reason"] == "stop"
        assert result.json()["choices"][0]["message"]["content"] == ""


import asyncio
import json
from threading import Event


def test_http_disconnect_closes_inflight_generator():
    closed = Event()

    class CancelClient(Client):
        def stream(self, *args, **kwargs):
            try:
                yield 1
                kwargs["cancel"].wait(timeout=2)
            finally:
                closed.set()

    async def exchange():
        incoming = asyncio.Queue()
        payload = json.dumps(
            {
                "model": "tiny-model",
                "prompt": "hello",
                "temperature": 0,
                "max_tokens": 8,
                "stream": True,
            }
        ).encode()
        await incoming.put({"type": "http.request", "body": payload, "more_body": False})

        async def receive():
            return await incoming.get()

        async def send(message):
            if message["type"] == "http.response.body" and b'"content": "B"' in message.get(
                "body", b""
            ):
                await incoming.put({"type": "http.disconnect"})

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "server": ("test", 80),
            "client": ("client", 1234),
            "headers": [(b"content-type", b"application/json")],
        }
        await create_app(CancelClient(), Tokenizer(), "tiny-model")(scope, receive, send)
        for _ in range(50):
            if closed.is_set():
                return
            await asyncio.sleep(0.05)
        raise AssertionError("streaming generator remained open after disconnect")

    asyncio.run(exchange())
