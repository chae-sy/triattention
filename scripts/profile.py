#!/usr/bin/env python3
"""Measure decode-only end-to-end latency for Full KV and TriAttention.

The prompt prefill is performed before each timed section.  A measurement
therefore includes only iterative decode forwards (and greedy token selection),
including any TriAttention compression triggered during decoding.

Example:
    python scripts/profile.py \
        --model Qwen/Qwen3-8B \
        --stats-path triattention/calibration/for_aime24_experiment/qwen3_8b.pt \
        --prompt-length 2048 \
        --generation-lengths 256,512,1024 \
        --budget 2048 \
        --output profiles/qwen3_decode.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# This file intentionally shares its name with Python's stdlib ``profile``
# module.  Remove ``scripts/`` from module lookup before importing PyTorch:
# PyTorch imports ``cProfile``, which in turn imports the stdlib module.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


RESULT_PREFIX = "PROFILE_RESULT="
DEFAULT_PROMPT = "Solve the following problem carefully and explain your reasoning. "


def parse_lengths(raw: str) -> list[int]:
    lengths = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not lengths or any(value <= 0 for value in lengths):
        raise argparse.ArgumentTypeError(
            "--generation-lengths must be a non-empty comma-separated list of positive integers"
        )
    return lengths


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_prompt_inputs(tokenizer: Any, prompt_length: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Build a deterministic token sequence of exactly ``prompt_length`` tokens."""
    base_ids = tokenizer.encode(DEFAULT_PROMPT, add_special_tokens=False)
    if not base_ids:
        raise ValueError("The tokenizer produced no tokens for the built-in profiling prompt.")

    token_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        token_ids.append(tokenizer.bos_token_id)
    while len(token_ids) < prompt_length:
        token_ids.extend(base_ids)
    token_ids = token_ids[:prompt_length]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, device=device),
    }


def apply_triattention(model: Any, args: argparse.Namespace, device: torch.device) -> None:
    if args.stats_path is None:
        raise ValueError("--stats-path is required when profiling TriAttention.")
    stats_path = Path(args.stats_path).expanduser().resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(f"TriAttention statistics file not found: {stats_path}")

    from triattention.methods.triattention import apply_triattention_patch

    apply_triattention_patch(
        model,
        stats_path=stats_path,
        model_path=Path(args.model),
        kv_budget=args.budget,
        score_aggregation=args.score_aggregation,
        pruning_seed=args.seed,
        normalize_scores=args.normalize_scores,
        # Count prompt tokens so a prompt close to the budget can trigger
        # compression while decoding; prefill itself remains outside timing.
        count_prompt_tokens=True,
        divide_length=args.divide_length,
        per_head_pruning=args.per_head_pruning,
        per_layer_perhead_pruning=args.per_layer_perhead_pruning,
        layer_perhead_aggregation=args.layer_perhead_aggregation,
        disable_mlr=args.disable_mlr,
        disable_trig=args.disable_trig,
    )


@torch.inference_mode()
def decode_once(model: Any, prompt_inputs: dict[str, torch.Tensor], generation_length: int, device: torch.device) -> float:
    """Return host-wall-clock time for decode only, excluding the prefill."""
    prefill = model(**prompt_inputs, use_cache=True, return_dict=True)
    synchronize(device)

    past_key_values = prefill.past_key_values
    next_token = prefill.logits[:, -1:].argmax(dim=-1)
    attention_mask = prompt_inputs["attention_mask"]

    synchronize(device)
    started = time.perf_counter()
    for _ in range(generation_length):
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(next_token, device=device)), dim=-1
        )
        output = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1:].argmax(dim=-1)
    synchronize(device)
    return time.perf_counter() - started


def run_worker(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    if args.method == "triattention":
        apply_triattention(model, args, device)

    prompt_inputs = build_prompt_inputs(tokenizer, args.prompt_length, device)
    # Warmups intentionally include prefill, but none of their time is reported.
    for generation_length in args.generation_lengths:
        for _ in range(args.warmup):
            decode_once(model, prompt_inputs, generation_length, device)

    rows: list[dict[str, object]] = []
    for generation_length in args.generation_lengths:
        for repeat in range(args.repeats):
            latency_s = decode_once(model, prompt_inputs, generation_length, device)
            rows.append(
                {
                    "method": args.method,
                    "repeat": repeat,
                    "prompt_tokens": args.prompt_length,
                    "generation_tokens": generation_length,
                    "latency_s": latency_s,
                    "latency_ms_per_token": latency_s * 1000 / generation_length,
                    "tokens_per_second": generation_length / latency_s,
                    "model": args.model,
                    "budget": args.budget if args.method == "triattention" else "",
                    "device": str(device),
                    "dtype": args.dtype,
                }
            )
    return rows


def worker_command(args: argparse.Namespace, method: str) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--_worker", "--method", method,
        "--model", args.model, "--prompt-length", str(args.prompt_length),
        "--generation-lengths", ",".join(map(str, args.generation_lengths)),
        "--repeats", str(args.repeats), "--warmup", str(args.warmup),
        "--budget", str(args.budget), "--divide-length", str(args.divide_length),
        "--device", args.device, "--dtype", args.dtype,
        "--attn-implementation", args.attn_implementation, "--seed", str(args.seed),
        "--score-aggregation", args.score_aggregation,
    ]
    if args.stats_path is not None:
        command.extend(["--stats-path", str(args.stats_path)])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.normalize_scores:
        command.append("--normalize-scores")
    if args.per_head_pruning:
        command.append("--per-head-pruning")
    if args.per_layer_perhead_pruning:
        command.append("--per-layer-perhead-pruning")
    if args.disable_mlr:
        command.append("--disable-mlr")
    if args.disable_trig:
        command.append("--disable-trig")
    command.extend(["--layer-perhead-aggregation", args.layer_perhead_aggregation])
    return command


def collect_worker_rows(args: argparse.Namespace, method: str) -> list[dict[str, object]]:
    result = subprocess.run(worker_command(args, method), cwd=REPO_ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{method} profiling worker failed (exit {result.returncode}).\n{result.stdout}\n{result.stderr}"
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    raise RuntimeError(f"{method} profiling worker returned no result.\n{result.stdout}\n{result.stderr}")


def write_csv(output: Path, rows: list[dict[str, object]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model directory.")
    parser.add_argument("--stats-path", type=Path, help="TriAttention Q/K statistics .pt file.")
    parser.add_argument("--output", type=Path, default=Path("profiles/decode_latency.csv"))
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--generation-lengths", type=parse_lengths, default=[256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--divide-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "eager"],
        default="sdpa",
        help="Transformers attention backend; neither option requires the flash-attn package.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--score-aggregation", choices=["mean", "max"], default="mean")
    parser.add_argument("--normalize-scores", action="store_true")
    parser.add_argument("--per-head-pruning", action="store_true")
    parser.add_argument("--per-layer-perhead-pruning", action="store_true")
    parser.add_argument("--layer-perhead-aggregation", choices=["mean", "max"], default="max")
    parser.add_argument("--disable-mlr", action="store_true")
    parser.add_argument("--disable-trig", action="store_true")
    parser.add_argument("--method", choices=["fullkv", "triattention"], help=argparse.SUPPRESS)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prompt_length <= 0 or args.repeats <= 0 or args.warmup < 0:
        parser.error("prompt length and repeats must be positive; warmup cannot be negative")
    if args.budget <= 0 or args.divide_length <= 0:
        parser.error("budget and divide length must be positive")
    if args._worker and args.method is None:
        parser.error("internal worker requires --method")
    if not args._worker and args.stats_path is None:
        parser.error("--stats-path is required to compare against TriAttention")
    # The parent launches workers from REPO_ROOT.  Resolve here, while the
    # user's working directory is still authoritative, so relative paths work
    # equally from the repository root and from scripts/.
    if args.stats_path is not None:
        args.stats_path = args.stats_path.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    if args._worker:
        print(RESULT_PREFIX + json.dumps(run_worker(args)))
        return
    rows = collect_worker_rows(args, "fullkv") + collect_worker_rows(args, "triattention")
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} decode-only measurements to {args.output}")


if __name__ == "__main__":
    main()
