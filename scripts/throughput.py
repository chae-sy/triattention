#!/usr/bin/env python3
"""Measure maximum-feasible-batch decode throughput for Full KV and TriAttention.

Each feasibility test and final measurement runs in a fresh subprocess.  Prefill
is completed before timing; reported latency is host-wall-clock decode latency.

Example:
    python scripts/throughput.py \
        --model Qwen/Qwen3-8B \
        --stats-path triattention/calibration/for_aime24_experiment/qwen3_8b.pt \
        --prompt-length 2048 --generation-lengths 8192,16384 \
        --budget 2048 --output profiles/qwen3_throughput.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ``scripts/profile.py`` shadows the stdlib ``profile`` module while this
# script is launched from scripts/.  Remove that directory before importing
# PyTorch, whose cProfile dependency imports the stdlib module by name.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


RESULT_PREFIX = "THROUGHPUT_RESULT="
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


def release_cached_cuda_memory(device: torch.device) -> None:
    """Release allocations left by a completed warmup or measurement."""
    if device.type == "cuda":
        synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()


def build_prompt_inputs(
    tokenizer: Any, prompt_length: int, batch_size: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Build identical deterministic prompts with shape [batch_size, prompt_length].

    Identical sequences are intentional: the current TriAttention scorer reads
    ``key_states[0, ...]`` when choosing shared keep indices, then gathers those
    indices across every batch item.  Greedy decoding keeps these replicas equal.
    """
    base_ids = tokenizer.encode(DEFAULT_PROMPT, add_special_tokens=False)
    if not base_ids:
        raise ValueError("The tokenizer produced no tokens for the built-in profiling prompt.")

    token_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        token_ids.append(tokenizer.bos_token_id)
    while len(token_ids) < prompt_length:
        token_ids.extend(base_ids)
    token_ids = token_ids[:prompt_length]

    one_prompt = torch.tensor(token_ids, dtype=torch.long, device=device)
    input_ids = one_prompt.unsqueeze(0).expand(batch_size, -1).clone()
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, device=device),
    }


def apply_triattention(model: Any, args: argparse.Namespace) -> None:
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
        # Matches profile.py: prompt tokens count toward a per-sequence budget,
        # while prefill execution remains outside the timed region.
        count_prompt_tokens=True,
        divide_length=args.divide_length,
        per_head_pruning=args.per_head_pruning,
        per_layer_perhead_pruning=args.per_layer_perhead_pruning,
        layer_perhead_aggregation=args.layer_perhead_aggregation,
        disable_mlr=args.disable_mlr,
        disable_trig=args.disable_trig,
    )


@torch.inference_mode()
def decode_once(
    model: Any,
    prompt_inputs: dict[str, torch.Tensor],
    generation_length: int,
    device: torch.device,
) -> float:
    """Run one prefill plus decode sequence and return decode-only wall time."""
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


def load_model_and_inputs(
    args: argparse.Namespace, batch_size: int
) -> tuple[Any, dict[str, torch.Tensor], torch.device]:
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
        apply_triattention(model, args)
    return model, build_prompt_inputs(tokenizer, args.prompt_length, batch_size, device), device


def run_feasibility_worker(args: argparse.Namespace) -> dict[str, object]:
    """A batch is feasible only when prefill and all decode steps finish."""
    try:
        model, prompt_inputs, device = load_model_and_inputs(args, args.batch_size)
        decode_once(model, prompt_inputs, args.generation_lengths[0], device)
        return {"feasible": True}
    except torch.cuda.OutOfMemoryError:
        # This process exits immediately after reporting, but clear cached blocks
        # as well to make cleanup explicit for CUDA runtimes that keep the context.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"feasible": False}


def run_measurement_worker(args: argparse.Namespace) -> list[dict[str, object]]:
    model, prompt_inputs, device = load_model_and_inputs(args, args.batch_size)
    generation_length = args.generation_lengths[0]

    for _ in range(args.warmup):
        decode_once(model, prompt_inputs, generation_length, device)
    # ``decode_once`` owns all per-request KV tensors.  Drop its cached blocks
    # before the first measured request so warmup allocator state cannot make a
    # boundary-size batch fail solely through fragmentation.
    release_cached_cuda_memory(device)

    rows: list[dict[str, object]] = []
    for repeat in range(args.repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        latency_s = decode_once(model, prompt_inputs, generation_length, device)
        peak_allocated_mb = ""
        peak_reserved_mb = ""
        if device.type == "cuda":
            peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
        total_tokens = args.batch_size * generation_length
        rows.append(
            {
                "method": args.method,
                "repeat": repeat,
                "model": args.model,
                "prompt_tokens": args.prompt_length,
                "generation_tokens": generation_length,
                "max_batch_size": args.batch_size,
                "decode_latency_s": latency_s,
                "throughput_tokens_per_second": total_tokens / latency_s,
                "per_sequence_tokens_per_second": generation_length / latency_s,
                "peak_memory_allocated_mb": peak_allocated_mb,
                "peak_memory_reserved_mb": peak_reserved_mb,
                "budget": args.budget if args.method == "triattention" else "",
                "device": str(device),
                "dtype": args.dtype,
            }
        )
        release_cached_cuda_memory(device)
    return rows


def worker_command(
    args: argparse.Namespace, method: str, generation_length: int, batch_size: int, mode: str
) -> list[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()), "--_worker", "--worker-mode", mode,
        "--method", method, "--model", args.model,
        "--prompt-length", str(args.prompt_length),
        "--generation-lengths", str(generation_length),
        "--batch-size", str(batch_size), "--repeats", str(args.repeats),
        "--warmup", str(args.warmup), "--budget", str(args.budget),
        "--divide-length", str(args.divide_length), "--device", args.device,
        "--dtype", args.dtype, "--attn-implementation", args.attn_implementation,
        "--seed", str(args.seed), "--score-aggregation", args.score_aggregation,
        "--max-batch-limit", str(args.max_batch_limit),
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


def run_subprocess(
    args: argparse.Namespace, method: str, generation_length: int, batch_size: int, mode: str
) -> object:
    worker_env = os.environ.copy()
    allocator_conf = worker_env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in allocator_conf:
        worker_env["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"{allocator_conf},expandable_segments:True".strip(",")
        )
    result = subprocess.run(
        worker_command(args, method, generation_length, batch_size, mode),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=worker_env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{method} {mode} worker failed (exit {result.returncode}).\n"
            f"{result.stdout}\n{result.stderr}"
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    raise RuntimeError(f"{method} {mode} worker returned no result.\n{result.stdout}\n{result.stderr}")


def is_feasible(args: argparse.Namespace, method: str, generation_length: int, batch_size: int) -> bool:
    result = run_subprocess(args, method, generation_length, batch_size, "feasibility")
    assert isinstance(result, dict)
    return bool(result["feasible"])


def find_max_batch_size(args: argparse.Namespace, method: str, generation_length: int) -> int:
    """Find the largest whole-generation-feasible batch using fresh workers."""
    if args.batch_size is not None:
        if not is_feasible(args, method, generation_length, args.batch_size):
            raise RuntimeError(
                f"Fixed batch size {args.batch_size} is not feasible for {method}, "
                f"generation length {generation_length}."
            )
        return args.batch_size

    largest_success = 0
    candidate = 1
    first_failure: int | None = None
    while candidate <= args.max_batch_limit:
        if is_feasible(args, method, generation_length, candidate):
            largest_success = candidate
            if candidate == args.max_batch_limit:
                return candidate
            next_candidate = min(candidate * 2, args.max_batch_limit)
            if next_candidate == candidate:
                return candidate
            candidate = next_candidate
        else:
            first_failure = candidate
            break

    if largest_success == 0:
        raise RuntimeError(
            f"Batch size 1 is not feasible for {method}, generation length {generation_length}."
        )
    if first_failure is None:
        return largest_success

    low, high = largest_success, first_failure - 1
    while low < high:
        middle = (low + high + 1) // 2
        if is_feasible(args, method, generation_length, middle):
            low = middle
        else:
            high = middle - 1
    return low


def write_csv(output: Path, rows: list[dict[str, object]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model directory.")
    parser.add_argument("--stats-path", type=Path, help="TriAttention Q/K statistics .pt file.")
    parser.add_argument("--output", type=Path, default=Path("profiles/decode_throughput.csv"))
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--generation-lengths", type=parse_lengths, default=[256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--divide-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--attn-implementation", choices=["sdpa", "eager"], default="sdpa",
        help="Transformers attention backend; neither option requires flash-attn.",
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
    parser.add_argument("--max-batch-limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, help="Fixed batch size; skips maximum-batch search.")
    parser.add_argument("--method", choices=["fullkv", "triattention"], help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=["feasibility", "measurement"], help=argparse.SUPPRESS)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prompt_length <= 0 or args.repeats <= 0 or args.warmup < 0:
        parser.error("prompt length and repeats must be positive; warmup cannot be negative")
    if args.budget <= 0 or args.divide_length <= 0:
        parser.error("budget and divide length must be positive")
    if args.max_batch_limit <= 0 or (args.batch_size is not None and args.batch_size <= 0):
        parser.error("batch sizes must be positive")
    if args._worker and (args.method is None or args.worker_mode is None):
        parser.error("internal worker requires --method and --worker-mode")
    if not args._worker and args.stats_path is None:
        parser.error("--stats-path is required to compare against TriAttention")
    if args.stats_path is not None:
        # Parent workers run from REPO_ROOT; retain the user's original CWD here.
        args.stats_path = args.stats_path.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    if args._worker:
        if args.worker_mode == "feasibility":
            result: object = run_feasibility_worker(args)
        else:
            result = run_measurement_worker(args)
        print(RESULT_PREFIX + json.dumps(result))
        return

    rows: list[dict[str, object]] = []
    for generation_length in args.generation_lengths:
        for method in ("fullkv", "triattention"):
            batch_size = find_max_batch_size(args, method, generation_length)
            print(f"{method}: generation_length={generation_length}, batch_size={batch_size}")
            measured_rows = run_subprocess(
                args, method, generation_length, batch_size, "measurement"
            )
            assert isinstance(measured_rows, list)
            rows.extend(measured_rows)
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} throughput measurements to {args.output}")


if __name__ == "__main__":
    main()
