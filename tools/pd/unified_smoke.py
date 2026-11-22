"""Record or compare greedy token IDs through the original LLM entrypoint on one GPU."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--graph-bs", type=int, choices=(0, 4), default=0)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--record", type=Path)
    output.add_argument("--check", type=Path)
    args = parser.parse_args()

    import torch
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    if not torch.cuda.is_available():
        parser.error(
            "this smoke test requires one CUDA GPU and the mini-SGLang runtime dependencies"
        )
    expected = json.loads(args.check.read_text()) if args.check else None
    prompts = [[1, 2, 3, 4], [1, 2, 3, 5], [5, 6, 7]]
    settings = {
        "model": args.model,
        "dtype": "bfloat16",
        "graph_bs": args.graph_bs,
        "page_size": 1,
        "attention_backend": "fi",
        "prompts": prompts,
        "max_tokens": 8,
        "ignore_eos": True,
    }
    if expected is not None and expected["settings"] != settings:
        parser.error("baseline settings differ; use the same weights, configuration and graph mode")

    llm = LLM(
        model_path=args.model,
        dtype=torch.bfloat16,
        max_running_req=4,
        num_page_override=512,
        max_seq_len_override=256,
        cuda_graph_max_bs=args.graph_bs,
        attention_backend="fi",
    )
    try:
        results = llm.generate(prompts, SamplingParams(max_tokens=8, ignore_eos=True))
    finally:
        llm.shutdown()
    token_ids = [result["token_ids"] for result in results]
    if any(len(tokens) != 8 for tokens in token_ids):
        raise RuntimeError(f"unexpected output lengths: {list(map(len, token_ids))}")
    record = {
        "settings": settings,
        "torch": torch.__version__,
        "token_ids": token_ids,
    }
    if expected is not None and token_ids != expected["token_ids"]:
        raise RuntimeError(
            f"greedy token mismatch: expected {expected['token_ids']}, got {token_ids}"
        )
    if args.record:
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": "PASS", **record}))


if __name__ == "__main__":
    main()
