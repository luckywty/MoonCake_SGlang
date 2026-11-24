"""HTTP frontend for the Phase 2 P/D client. This process needs only the tokenizer."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from threading import Event, Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from minisgl.core import SamplingParams
from minisgl.message import DetokenizeMsg
from minisgl.server.api_server import OpenAICompletionRequest
from minisgl.tokenizer.detokenize import DetokenizeManager

from .admission import AdmissionError
from .client import PDClient, RemoteWorkerError
from .cluster_config import add_routing_args, remote_addresses

"""提供 HTTP 接口，处理分词、文本输出和流式响应
接收用户的 HTTP 请求、Tokenization、调用 P/D 推理客户端，
以及将生成的 Token 转换成 OpenAI API 风格的响应。"""


def create_app(client: PDClient, tokenizer, model: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        try:
            if hasattr(client, "start"):
                await asyncio.to_thread(client.start)
            yield
        finally:
            if hasattr(client, "close"):
                await asyncio.to_thread(client.close)

    app = FastAPI(lifespan=lifespan)

    def finish_reason(tokens, req):
        stopped_by_eos = tokens and not req.ignore_eos and tokens[-1] == tokenizer.eos_token_id
        return "length" if len(tokens) == req.max_tokens and not stopped_by_eos else "stop"

    async def disconnect(request: Request, cancel: Event):
        while not cancel.is_set():
            if await request.is_disconnected():
                cancel.set()
                return
            await asyncio.sleep(0.1)

    @app.post("/v1/chat/completions")
    async def completions(req: OpenAICompletionRequest, request: Request):
        params = SamplingParams(temperature=req.temperature, top_k=req.top_k, top_p=req.top_p)
        if (
            not params.is_greedy
            or req.max_tokens < 1
            or req.n != 1
            or req.stop
            or req.presence_penalty != 0
            or req.frequency_penalty != 0
        ):
            raise HTTPException(400, "Phase 2 supports greedy requests with max_tokens >= 1")
        if req.messages:
            ids = tokenizer.apply_chat_template(
                [message.model_dump() for message in req.messages],
                tokenize=True,
                add_generation_prompt=True,
            )
        elif req.prompt is not None:
            ids = tokenizer.encode(req.prompt)
        else:
            raise HTTPException(400, "messages or prompt is required")
        cancel = Event()
        routing_request_id = str(uuid4())
        request_id = "chatcmpl-" + routing_request_id

        async def chunks():
            watcher = asyncio.create_task(disconnect(request, cancel))
            iterator = client.stream(
                ids,
                req.max_tokens,
                ignore_eos=req.ignore_eos,
                cancel=cancel,
                request_id=routing_request_id,
            )
            tokens = []
            detokenizer = DetokenizeManager(tokenizer)
            iterator_lock = Lock()

            def pull():
                with iterator_lock:
                    return next(iterator, None)

            def close_iterator():
                with iterator_lock:
                    iterator.close()

            try:
                while True:
                    token = await asyncio.to_thread(pull)
                    if token is None:
                        break
                    tokens.append(token)
                    finished = len(tokens) == req.max_tokens or (
                        not req.ignore_eos and token == tokenizer.eos_token_id
                    )
                    delta = detokenizer.detokenize([DetokenizeMsg(0, token, finished)])[0]
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "model": model,
                                "choices": [
                                    {"index": 0, "delta": {"content": delta}, "finish_reason": None}
                                ],
                            }
                        )
                        + "\n\n"
                    )
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason(tokens, req),
                                }
                            ],
                        }
                    )
                    + "\n\ndata: [DONE]\n\n"
                )
            except (OSError, RemoteWorkerError, AdmissionError, ValueError) as exc:
                yield (
                    "data: " + json.dumps({"error": {"message": str(exc)}}) + "\n\ndata: [DONE]\n\n"
                )
            finally:
                cancel.set()
                watcher.cancel()
                cleanup = asyncio.create_task(asyncio.to_thread(close_iterator))
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(cleanup)
                with suppress(asyncio.CancelledError):
                    await watcher

        if req.stream:
            return StreamingResponse(chunks(), media_type="text/event-stream")
        watcher = asyncio.create_task(disconnect(request, cancel))
        try:
            tokens = await asyncio.to_thread(
                client.generate,
                ids,
                req.max_tokens,
                ignore_eos=req.ignore_eos,
                cancel=cancel,
                request_id=routing_request_id,
            )
            detokenizer = DetokenizeManager(tokenizer)
            content = "".join(
                detokenizer.detokenize([DetokenizeMsg(0, token, i + 1 == len(tokens))])[0]
                for i, token in enumerate(tokens)
            )
            return {
                "id": request_id,
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": finish_reason(tokens, req),
                    }
                ],
                "usage": {
                    "prompt_tokens": len(ids),
                    "completion_tokens": len(tokens),
                    "total_tokens": len(ids) + len(tokens),
                },
            }
        except AdmissionError as exc:
            raise HTTPException(429, str(exc)) from exc
        except (OSError, RemoteWorkerError, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            cancel.set()
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": model, "object": "model", "owned_by": "fluxinfer"}],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill")
    parser.add_argument("--decode")
    add_routing_args(parser)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1940)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-inflight-requests", type=int, default=128)
    parser.add_argument("--max-inflight-tokens", type=int, default=1048576)
    args = parser.parse_args()
    try:
        if args.prefill or args.decode:
            if not (args.prefill and args.decode) or args.prefill_addrs or args.decode_addrs:
                raise ValueError("use either --prefill/--decode or both address lists")
            args.prefill_addrs, args.decode_addrs = [args.prefill], [args.decode]
        prefill, decode = remote_addresses(args)
        if not prefill:
            raise ValueError("prefill and decode addresses are required")
        from .routed_client import RoutedPDClient
        from .routing_policy import policy_config

        client = RoutedPDClient(
            prefill,
            decode,
            timeout=args.timeout,
            routing_seed=args.routing_seed,
            max_inflight_requests=args.max_inflight_requests,
            max_inflight_tokens=args.max_inflight_tokens,
            health_interval=args.health_interval,
            health_timeout=args.health_timeout,
            ready_timeout=args.worker_ready_timeout,
            max_quarantined_bytes=args.max_quarantined_bytes,
            routing_config=policy_config(args),
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO)
    import uvicorn
    from minisgl.utils import load_tokenizer

    app = create_app(
        client,
        load_tokenizer(args.model),
        args.model,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
