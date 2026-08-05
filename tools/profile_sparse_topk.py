"""Measure parameter count, profiler FLOPs, and CUDA latency for sparse Top-K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.dwmoespace_sparse_topk import dwmoespace_sparse_topk


def profile_case(top_k: int, inputs: torch.Tensor, iterations: int) -> dict:
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    model = dwmoespace_sparse_topk(
        num_classes=6,
        expert_output_dim=100,
        router_hidden_dim=32,
        expert_kernels=(3, 7, 15),
        top_k=top_k,
        router_temperature=1.0,
        norm_type="local_filter",
    ).cuda().eval()
    with torch.no_grad():
        for _ in range(20):
            model(inputs)
        torch.cuda.synchronize()
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities, with_flops=True
        ) as profiler:
            model(inputs)
            torch.cuda.synchronize()
        flops = sum(int(event.flops or 0) for event in profiler.key_averages())
        samples_ms = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(inputs)
            end.record()
            end.synchronize()
            samples_ms.append(float(start.elapsed_time(end)))
    return {
        "top_k": top_k,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "profiler_flops_per_batch": flops,
        "profiler_flops_per_sample": flops / inputs.shape[0],
        "latency_ms_median": statistics.median(samples_ms),
        "latency_ms_mean": statistics.fmean(samples_ms),
        "latency_ms_stdev": statistics.stdev(samples_ms),
        "routing": model.moe.routing_summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--time_samples", type=int, choices=[1000, 2000], required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen latency protocol")
    torch.manual_seed(2026)
    inputs = torch.randn(args.batch_size, 16, args.time_samples, device="cuda")
    report = {
        "status": "completed",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "time_samples": args.time_samples,
        "iterations": args.iterations,
        "cases": [profile_case(k, inputs, args.iterations) for k in (1, 2, 3)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
