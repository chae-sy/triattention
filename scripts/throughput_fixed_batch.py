#!/usr/bin/env python3
"""Measure fixed-batch decode throughput for FullKV and TriAttention.

This follows scripts/profile.py closely:
  * deterministic prompt construction
  * greedy argmax decoding
  * prompt prefill excluded from decode wall-clock timing
  * FullKV / TriAttention run in separate fresh subprocesses
  * TriAttention patch arguments match profile.py

A fixed batch size is supplied for each generation length. Every warmup and
measured repeat runs in its own fresh subprocess. Each completed measured repeat
is appended to the output CSV immediately, so partial results survive interruption.

Example:
    python scripts/throughput.py \
        --model Qwen/Qwen3-8B \
        --stats-path triattention/calibration/for_aime24_experiment/qwen3_8b.pt \
        --prompt-length 128 \
        --generation-lengths 8192,16384,32768,65536 \
        --budget 2048 \
        --fixed-batch 82,32,17,8 \
        --output profiles/decode_throughput.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Avoid scripts/profile.py shadowing Python's stdlib `profile` during torch import.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


RESULT_PREFIX = "THROUGHPUT_RESULT="
FEASIBLE_PREFIX = "THROUGHPUT_FEASIBLE="
OOM_EXIT_CODE = 42
DEFAULT_PROMPT = "Solve the following problem carefully and explain your reasoning. "


def parse_lengths(raw: str) -> list[int]:
    lengths = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not lengths or any(v <= 0 for v in lengths):
        raise argparse.ArgumentTypeError(
            "--generation-lengths must be a non-empty comma-separated list of positive integers"
        )
    return lengths


def parse_fixed_batches(raw: str) -> list[int]:
    batches = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not batches or any(v <= 0 for v in batches):
        raise argparse.ArgumentTypeError(
            "--fixed-batch must be a non-empty comma-separated list of positive integers"
        )
    return batches



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_prompt_inputs(
    tokenizer: Any,
    prompt_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build the same deterministic prompt for every sequence: [B, prompt_length]."""
    base_ids = tokenizer.encode(DEFAULT_PROMPT, add_special_tokens=False)
    if not base_ids:
        raise ValueError("The tokenizer produced no tokens for the built-in throughput prompt.")

    token_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        token_ids.append(tokenizer.bos_token_id)
    while len(token_ids) < prompt_length:
        token_ids.extend(base_ids)
    token_ids = token_ids[:prompt_length]

    one = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_ids = one.expand(batch_size, -1).contiguous()
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, device=device),
    }


def apply_triattention(model: Any, args: argparse.Namespace) -> None:
    if args.stats_path is None:
        raise ValueError("--stats-path is required for TriAttention throughput.")
    stats_path = Path(args.stats_path).expanduser().resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(f"TriAttention statistics file not found: {stats_path}")

    from triattention.methods.triattention import apply_triattention_patch

    apply_triattention_patch(
        model,
        stats_path=stats_path,
        model_path=Path(args.model),
        kv_budget=args.budget,  # budget remains per sequence; benchmark does not rescale it by B
        score_aggregation=args.score_aggregation,
        pruning_seed=args.seed,
        normalize_scores=args.normalize_scores,
        count_prompt_tokens=True,
        divide_length=args.divide_length,
        per_head_pruning=args.per_head_pruning,
        per_layer_perhead_pruning=args.per_layer_perhead_pruning,
        layer_perhead_aggregation=args.layer_perhead_aggregation,
        disable_mlr=args.disable_mlr,
        disable_trig=args.disable_trig,
    )



def inspect_physical_cache(past_key_values: Any) -> dict[str, object]:
    """Inspect actual materialized KV sequence lengths without trusting budget metadata.

    Returns per-sequence physical token length for every cache layer when the
    installed Transformers cache exposes its tensors. This intentionally reads
    tensor shape[-2], i.e. the physically materialized sequence dimension.
    """
    lengths: list[int] = []

    # Current Transformers DynamicCache / Cache API.
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        for layer in layers:
            keys = getattr(layer, "keys", None)
            if keys is None:
                keys = getattr(layer, "key_cache", None)
            if torch.is_tensor(keys):
                lengths.append(int(keys.shape[-2]))

    # Older Transformers DynamicCache API: key_cache is a list of tensors.
    if not lengths:
        key_cache = getattr(past_key_values, "key_cache", None)
        if isinstance(key_cache, (list, tuple)):
            for keys in key_cache:
                if torch.is_tensor(keys):
                    lengths.append(int(keys.shape[-2]))

    # Legacy tuple-of-(K,V) cache.
    if not lengths and isinstance(past_key_values, (list, tuple)):
        for layer in past_key_values:
            if (
                isinstance(layer, (list, tuple))
                and len(layer) >= 1
                and torch.is_tensor(layer[0])
            ):
                lengths.append(int(layer[0].shape[-2]))

    if not lengths:
        # Last-resort diagnostic only. get_seq_length may be logical for some
        # custom cache implementations, so mark this separately.
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                fallback = int(get_seq_length())
            except TypeError:
                fallback = int(get_seq_length(0))
            return {
                "physical_cache_inspection": "fallback_get_seq_length",
                "physical_cache_tokens_layer0": fallback,
                "physical_cache_tokens_min": fallback,
                "physical_cache_tokens_mean": float(fallback),
                "physical_cache_tokens_max": fallback,
                "physical_cache_tokens_sum_layers": fallback,
                "physical_cache_num_layers": 1,
                "physical_cache_layer_lengths": [fallback],
            }

        return {
            "physical_cache_inspection": "unavailable",
            "physical_cache_tokens_layer0": -1,
            "physical_cache_tokens_min": -1,
            "physical_cache_tokens_mean": -1.0,
            "physical_cache_tokens_max": -1,
            "physical_cache_tokens_sum_layers": -1,
            "physical_cache_num_layers": 0,
            "physical_cache_layer_lengths": [],
        }

    return {
        "physical_cache_inspection": "tensor_shape",
        "physical_cache_tokens_layer0": lengths[0],
        "physical_cache_tokens_min": min(lengths),
        "physical_cache_tokens_mean": sum(lengths) / len(lengths),
        "physical_cache_tokens_max": max(lengths),
        "physical_cache_tokens_sum_layers": sum(lengths),
        "physical_cache_num_layers": len(lengths),
        "physical_cache_layer_lengths": lengths,
    }

@torch.inference_mode()
def decode_once(
    model: Any,
    prompt_inputs: dict[str, torch.Tensor],
    generation_length: int,
    device: torch.device,
) -> tuple[float, dict[str, object]]:
    """Return decode-only host wall time. Prefill is intentionally untimed."""
    prefill = model(**prompt_inputs, use_cache=True, return_dict=True)
    synchronize(device)

    past_key_values = prefill.past_key_values
    next_token = prefill.logits[:, -1:].argmax(dim=-1)
    attention_mask = prompt_inputs["attention_mask"]

    synchronize(device)
    started = time.perf_counter()
    for _ in range(generation_length):
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(next_token, device=device)),
            dim=-1,
        )
        output = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = output.past_key_values
        # Keep greedy token selection inside the timed decode region.
        next_token = output.logits[:, -1:].argmax(dim=-1)
    synchronize(device)
    latency_s = time.perf_counter() - started
    cache_stats = inspect_physical_cache(past_key_values)
    return latency_s, cache_stats


def load_worker_state(args: argparse.Namespace) -> tuple[Any, Any, torch.device]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    if args.method == "triattention":
        apply_triattention(model, args)

    return tokenizer, model, device


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True only for explicit CUDA-memory/allocator failure signatures.

    PyTorch normally raises torch.OutOfMemoryError, but at very large batches
    the CUDA caching allocator can fail first with an NVML/CUDACachingAllocator
    RuntimeError. Those are treated as an infeasible batch as well so the
    search can continue instead of aborting.
    """
    if isinstance(exc, torch.OutOfMemoryError):
        return True

    message = str(exc).lower()
    oom_signatures = (
        "cuda out of memory",
        "cuda error: out of memory",
        "cudacachingallocator",
        "nvml_success == r internal assert failed",
        "outofmemoryerror",
    )
    return any(signature in message for signature in oom_signatures)


class WorkerOOM(RuntimeError):
    """Parent-side signal that a fresh worker hit a CUDA memory limit."""


def feasibility_worker(args: argparse.Namespace) -> None:
    """One fresh process: batch is feasible only after the full generation completes."""
    try:
        tokenizer, model, device = load_worker_state(args)
        prompt_inputs = build_prompt_inputs(
            tokenizer, args.prompt_length, args.batch_size, device
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        latency_s, cache_stats = decode_once(
            model,
            prompt_inputs,
            args.worker_generation_length,
            device,
        )

        peak_allocated_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )
        peak_reserved_mb = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )

        payload = {
            "batch_size": args.batch_size,
            "decode_latency_s": latency_s,
            "peak_memory_allocated_mb": peak_allocated_mb,
            "peak_memory_reserved_mb": peak_reserved_mb,
            **cache_stats,
        }
        print(FEASIBLE_PREFIX + json.dumps(payload), flush=True)
    except BaseException as exc:
        if is_cuda_oom(exc):
            # This process exits immediately, so allocator state cannot contaminate
            # the next candidate process.
            print(
                FEASIBLE_PREFIX
                + json.dumps({"batch_size": args.batch_size, "oom": True}),
                flush=True,
            )
            raise SystemExit(OOM_EXIT_CODE)
        raise


def single_run_worker(args: argparse.Namespace) -> None:
    """Run exactly one full prefill+decode in this fresh worker process."""
    try:
        tokenizer, model, device = load_worker_state(args)
        prompt_inputs = build_prompt_inputs(
            tokenizer, args.prompt_length, args.batch_size, device
        )
        generation_length = args.worker_generation_length

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        latency_s, cache_stats = decode_once(
            model,
            prompt_inputs,
            generation_length,
            device,
        )

        peak_allocated = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )

        if args.worker_mode == "warmup":
            print(RESULT_PREFIX + json.dumps({"warmup": True}), flush=True)
            return

        total_generated_tokens = args.batch_size * generation_length
        row = {
            "method": args.method,
            "repeat": args.worker_repeat,
            "model": args.model,
            "prompt_tokens": args.prompt_length,
            "generation_tokens": generation_length,
            "max_batch_size": args.batch_size,
            "decode_latency_s": latency_s,
            "throughput_tokens_per_second": total_generated_tokens / latency_s,
            "per_sequence_tokens_per_second": generation_length / latency_s,
            "peak_memory_allocated_mb": peak_allocated,
            "peak_memory_reserved_mb": peak_reserved,
            "budget": args.budget if args.method == "triattention" else "",
            "final_logical_tokens": args.prompt_length + generation_length,
            **cache_stats,
            "device": str(device),
            "dtype": args.dtype,
        }
        print(RESULT_PREFIX + json.dumps(row), flush=True)

    except BaseException as exc:
        if is_cuda_oom(exc):
            print(
                RESULT_PREFIX
                + json.dumps(
                    {
                        "oom": True,
                        "batch_size": args.batch_size,
                        "worker_mode": args.worker_mode,
                    }
                ),
                flush=True,
            )
            raise SystemExit(OOM_EXIT_CODE)
        raise


def common_worker_command(
    args: argparse.Namespace,
    *,
    method: str,
    generation_length: int,
    batch_size: int,
    worker_mode: str,
    repeat: int = -1,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--worker-mode", worker_mode,
        "--worker-repeat", str(repeat),
        "--method", method,
        "--worker-generation-length", str(generation_length),
        "--batch-size", str(batch_size),
        "--model", args.model,
        "--prompt-length", str(args.prompt_length),
        "--generation-lengths", str(generation_length),
        "--fixed-batch", str(batch_size),
        "--repeats", str(args.repeats),
        "--warmup", str(args.warmup),
        "--budget", str(args.budget),
        "--divide-length", str(args.divide_length),
        "--device", args.device,
        "--dtype", args.dtype,
        "--attn-implementation", args.attn_implementation,
        "--seed", str(args.seed),
        "--score-aggregation", args.score_aggregation,
    ]
    if args.stats_path is not None:
        cmd.extend(["--stats-path", str(args.stats_path)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.normalize_scores:
        cmd.append("--normalize-scores")
    if args.per_head_pruning:
        cmd.append("--per-head-pruning")
    if args.per_layer_perhead_pruning:
        cmd.append("--per-layer-perhead-pruning")
    if args.disable_mlr:
        cmd.append("--disable-mlr")
    if args.disable_trig:
        cmd.append("--disable-trig")
    cmd.extend(["--layer-perhead-aggregation", args.layer_perhead_aggregation])
    return cmd



def worker_env() -> dict[str, str]:
    """Fresh worker environment with fragmentation-resistant CUDA allocation."""
    env = dict(os.environ)
    alloc_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"{alloc_conf},expandable_segments:True".strip(",")
        )
    return env

def run_feasibility(
    args: argparse.Namespace,
    method: str,
    generation_length: int,
    batch_size: int,
) -> bool:
    """Fresh subprocess per candidate. False means CUDA OOM only."""
    cmd = common_worker_command(
        args,
        method=method,
        generation_length=generation_length,
        batch_size=batch_size,
        worker_mode="feasibility",
    )
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, env=worker_env())

    if result.returncode == 0:
        payload = None
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(FEASIBLE_PREFIX):
                payload = json.loads(line.removeprefix(FEASIBLE_PREFIX))
                break

        if payload is not None:
            print(
                "[feasible] "
                f"batch={batch_size} "
                f"physical_kv_mean={payload.get('physical_cache_tokens_mean')} "
                f"physical_kv_min={payload.get('physical_cache_tokens_min')} "
                f"physical_kv_max={payload.get('physical_cache_tokens_max')} "
                f"peak_allocated_mb={payload.get('peak_memory_allocated_mb', 0.0):.1f} "
                f"peak_reserved_mb={payload.get('peak_memory_reserved_mb', 0.0):.1f}"
            )
        return True
    if result.returncode == OOM_EXIT_CODE:
        return False

    # Any non-OOM error is a correctness/configuration failure and must be visible.
    raise RuntimeError(
        f"{method} batch-feasibility worker failed for "
        f"generation_length={generation_length}, batch_size={batch_size} "
        f"(exit {result.returncode}).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def find_max_batch(
    args: argparse.Namespace,
    method: str,
    generation_length: int,
) -> int:
    limit = args.max_batch_limit

    print(f"[search] method={method} generation={generation_length}: trying batch=1")
    if not run_feasibility(args, method, generation_length, 1):
        raise RuntimeError(
            f"{method} cannot complete generation_length={generation_length} even at batch_size=1."
        )

    low = 1
    if limit == 1:
        return 1

    # Exponential search: 1, 2, 4, 8, ...
    high = 2
    failed_high: int | None = None
    while high <= limit:
        print(f"[search] method={method} generation={generation_length}: trying batch={high}")
        if run_feasibility(args, method, generation_length, high):
            low = high
            if high == limit:
                return high
            high *= 2
        else:
            failed_high = high
            break

    if failed_high is None:
        # We stepped past a non-power-of-two limit. Test the limit exactly.
        if low < limit:
            print(f"[search] method={method} generation={generation_length}: trying batch={limit}")
            if run_feasibility(args, method, generation_length, limit):
                return limit
            failed_high = limit
        else:
            return low

    # Binary search strictly between largest success and failed candidate.
    left, right = low + 1, failed_high - 1
    best = low
    while left <= right:
        mid = (left + right) // 2
        print(f"[search] method={method} generation={generation_length}: trying batch={mid}")
        if run_feasibility(args, method, generation_length, mid):
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best


def run_single_process(
    args: argparse.Namespace,
    method: str,
    generation_length: int,
    batch_size: int,
    *,
    worker_mode: str,
    repeat: int = -1,
) -> dict[str, object]:
    """Launch one fresh worker that performs exactly one full generation."""
    cmd = common_worker_command(
        args,
        method=method,
        generation_length=generation_length,
        batch_size=batch_size,
        worker_mode=worker_mode,
        repeat=repeat,
    )
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, env=worker_env())
    if result.returncode == OOM_EXIT_CODE:
        raise WorkerOOM(
            f"{method} {worker_mode} hit a CUDA memory limit at "
            f"generation_length={generation_length}, batch_size={batch_size}, "
            f"repeat={repeat}."
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"{method} {worker_mode} worker failed for "
            f"generation_length={generation_length}, batch_size={batch_size}, "
            f"repeat={repeat} (exit {result.returncode}).\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))

    raise RuntimeError(
        f"{method} {worker_mode} worker returned no result.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def normalize_csv_row(row: dict[str, object]) -> dict[str, object]:
    row = dict(row)
    if isinstance(row.get("physical_cache_layer_lengths"), list):
        row["physical_cache_layer_lengths"] = json.dumps(
            row["physical_cache_layer_lengths"]
        )
    return row


def csv_fields() -> list[str]:
    return [
        "method",
        "repeat",
        "model",
        "prompt_tokens",
        "generation_tokens",
        "max_batch_size",
        "decode_latency_s",
        "throughput_tokens_per_second",
        "per_sequence_tokens_per_second",
        "peak_memory_allocated_mb",
        "peak_memory_reserved_mb",
        "budget",
        "final_logical_tokens",
        "physical_cache_inspection",
        "physical_cache_tokens_layer0",
        "physical_cache_tokens_min",
        "physical_cache_tokens_mean",
        "physical_cache_tokens_max",
        "physical_cache_tokens_sum_layers",
        "physical_cache_num_layers",
        "physical_cache_layer_lengths",
        "device",
        "dtype",
    ]


def initialize_csv(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields())
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())


def append_csv_row(output: Path, row: dict[str, object]) -> None:
    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields())
        writer.writerow(normalize_csv_row(row))
        f.flush()
        os.fsync(f.fileno())


def run_measurements(
    args: argparse.Namespace,
    method: str,
    generation_length: int,
    batch_size: int,
) -> int:
    """Measure the requested fixed batch and append each repeat immediately."""
    try:
        for warmup_idx in range(args.warmup):
            print(
                f"[warmup] method={method} generation={generation_length} "
                f"batch={batch_size} warmup={warmup_idx + 1}/{args.warmup}"
            )
            run_single_process(
                args,
                method,
                generation_length,
                batch_size,
                worker_mode="warmup",
            )

        completed = 0
        for repeat in range(args.repeats):
            print(
                f"[measure] method={method} generation={generation_length} "
                f"batch={batch_size} repeat={repeat + 1}/{args.repeats}"
            )
            row = run_single_process(
                args,
                method,
                generation_length,
                batch_size,
                worker_mode="measure",
                repeat=repeat,
            )
            print(
                "[result] "
                f"method={method} generation={generation_length} "
                f"batch={batch_size} repeat={repeat} "
                f"physical_kv_mean={row.get('physical_cache_tokens_mean')} "
                f"physical_kv_min={row.get('physical_cache_tokens_min')} "
                f"physical_kv_max={row.get('physical_cache_tokens_max')} "
                f"peak_allocated_mb={row.get('peak_memory_allocated_mb', 0.0):.1f} "
                f"peak_reserved_mb={row.get('peak_memory_reserved_mb', 0.0):.1f}"
            )
            append_csv_row(args.output, row)
            completed += 1
            print(f"[csv] appended row to {args.output}")

        return completed

    except WorkerOOM as exc:
        raise RuntimeError(
            f"Requested fixed batch {batch_size} is not memory-stable for "
            f"method={method}, generation_length={generation_length}."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model directory.")
    parser.add_argument("--stats-path", type=Path, help="TriAttention Q/K statistics .pt file.")
    parser.add_argument("--output", type=Path, default=Path("profiles/decode_throughput.csv"))
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--generation-lengths", type=parse_lengths, default=[256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--divide-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "eager"],
        default="sdpa",
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
    parser.add_argument(
        "--mode",
        choices=["both", "tri_only"],
        default="both",
        help=(
            "Benchmark mode: 'both' measures FullKV and TriAttention; "
            "'tri_only' skips FullKV and measures only TriAttention."
        ),
    )

    parser.add_argument(
        "--fixed-batch",
        type=parse_fixed_batches,
        required=True,
        help=(
            "Comma-separated fixed batch sizes aligned with --generation-lengths. "
            "Example: --generation-lengths 8192,16384 --fixed-batch 82,32"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

    # Internal subprocess protocol.
    parser.add_argument("--method", choices=["fullkv", "triattention"], help=argparse.SUPPRESS)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-mode",
        choices=["feasibility", "warmup", "measure"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-generation-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-repeat", type=int, default=-1, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.prompt_length <= 0 or args.repeats <= 0 or args.warmup < 0:
        parser.error("prompt length and repeats must be positive; warmup cannot be negative")
    if args.budget <= 0 or args.divide_length <= 0:
        parser.error("budget and divide length must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args._worker:
        if args.method is None or args.worker_mode is None or args.worker_generation_length is None:
            parser.error("internal worker requires --method, --worker-mode and --worker-generation-length")
        if args.batch_size is None:
            parser.error("internal worker requires --batch-size")
    else:
        if len(args.fixed_batch) != len(args.generation_lengths):
            parser.error(
                "--fixed-batch must contain exactly one batch size for each "
                "--generation-lengths entry"
            )
        if args.stats_path is None:
            parser.error("--stats-path is required to compare against TriAttention")

    if args.stats_path is not None:
        args.stats_path = args.stats_path.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()

    if args._worker:
        if args.worker_mode == "feasibility":
            feasibility_worker(args)
        else:
            single_run_worker(args)
        return

    initialize_csv(args.output)
    completed_rows = 0

    methods = ("triattention",) if args.mode == "tri_only" else ("fullkv", "triattention")
    print(f"[mode] {args.mode}: methods={list(methods)}")
    print(
        "[fixed-batches] "
        + ", ".join(
            f"{generation_length}->{batch_size}"
            for generation_length, batch_size in zip(
                args.generation_lengths, args.fixed_batch
            )
        )
    )

    for generation_length, batch_size in zip(
        args.generation_lengths, args.fixed_batch
    ):
        for method in methods:
            print(
                f"[fixed-batch] method={method} generation={generation_length} "
                f"batch={batch_size}"
            )
            completed_rows += run_measurements(
                args,
                method,
                generation_length,
                batch_size,
            )

    print(f"Wrote {completed_rows} throughput measurements to {args.output}")


if __name__ == "__main__":
    main()