"""Verify that sparse Top-K routing dispatches only selected sample paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.dwmoespace_sparse_topk import dwmoespace_sparse_topk


def finite_nonzero_gradient(parameter: torch.nn.Parameter) -> bool:
    gradient = parameter.grad
    return bool(
        gradient is not None
        and torch.isfinite(gradient).all().item()
        and gradient.abs().sum().item() > 0
    )


def run_case(top_k: int, device: torch.device, batch_size: int) -> dict:
    torch.manual_seed(2026 + top_k)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2026 + top_k)
    model = dwmoespace_sparse_topk(
        num_classes=6,
        expert_output_dim=100,
        router_hidden_dim=32,
        expert_kernels=(3, 7, 15),
        top_k=top_k,
        router_temperature=1.0,
        norm_type="local_filter",
    ).to(device)

    hook_batch_sizes = [[] for _ in model.moe.experts]
    hooks = []
    for expert_index, expert in enumerate(model.moe.experts):
        hooks.append(
            expert.register_forward_pre_hook(
                lambda _module, inputs, i=expert_index: (
                    hook_batch_sizes[i].append(int(inputs[0].shape[0]))
                )
            )
        )

    model.train()
    inputs = torch.randn(batch_size, 16, 1000, device=device)
    labels = torch.arange(batch_size, device=device) % 6
    details = model.forward_with_routing_details(inputs)
    logits = details["logits"]
    sparse_weights = details["sparse_gate_weights"]
    selected = details["selected_experts"]
    executed = details["executed_sample_counts"]

    assert logits.shape == (batch_size, 6)
    assert torch.isfinite(logits).all()
    assert sparse_weights.shape == (batch_size, 3)
    assert selected.shape == (batch_size, top_k)
    assert torch.all(sparse_weights >= 0)
    assert torch.allclose(
        sparse_weights.sum(dim=-1),
        torch.ones(batch_size, device=device),
        rtol=0.0,
        atol=1e-6,
    )
    nonzeros = (sparse_weights > 0).sum(dim=-1)
    assert torch.equal(
        nonzeros, torch.full_like(nonzeros, fill_value=top_k)
    )
    assert torch.all(
        torch.sort(selected, dim=-1).values[:, 1:]
        != torch.sort(selected, dim=-1).values[:, :-1]
    ) if top_k > 1 else True
    assert int(executed.sum().item()) == batch_size * top_k
    observed_hook_sizes = [sum(values) for values in hook_batch_sizes]
    assert observed_hook_sizes == executed.detach().cpu().tolist()
    assert sum(observed_hook_sizes) == batch_size * top_k

    loss = F.cross_entropy(logits, labels) + 0.01 * details[
        "routing_aux_loss"
    ]
    loss.backward()
    router_grad_ok = all(
        finite_nonzero_gradient(parameter)
        for parameter in model.moe.router.parameters()
        if parameter.requires_grad
    )
    classifier_grad_ok = all(
        finite_nonzero_gradient(parameter)
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    )
    assert router_grad_ok
    assert classifier_grad_ok
    expert_grad_ok = []
    for expert_index, expert in enumerate(model.moe.experts):
        gradients_ok = all(
            finite_nonzero_gradient(parameter)
            for parameter in expert.parameters()
            if parameter.requires_grad
        )
        if observed_hook_sizes[expert_index] > 0:
            assert gradients_ok
        expert_grad_ok.append(gradients_ok)

    for hook in hooks:
        hook.remove()

    model.eval()
    model.moe.reset_routing_stats()
    with torch.no_grad():
        first = model.forward_with_routing_details(inputs)
        second = model.forward_with_routing_details(inputs)
    assert torch.equal(first["selected_experts"], second["selected_experts"])
    assert torch.equal(
        first["sparse_gate_weights"], second["sparse_gate_weights"]
    )
    assert torch.equal(first["logits"], second["logits"])
    summary = model.moe.routing_summary()
    assert summary["only_selected_samples_executed"] is True
    assert summary["executed_sample_expert_assignments"] == (
        2 * batch_size * top_k
    )

    return {
        "top_k": top_k,
        "loss": float(loss.detach().cpu()),
        "nonzeros_per_sample": sorted(set(nonzeros.detach().cpu().tolist())),
        "selected_expert_shape": list(selected.shape),
        "executed_sample_counts": executed.detach().cpu().tolist(),
        "hook_sample_counts": observed_hook_sizes,
        "router_gradient_ok": router_grad_ok,
        "classifier_gradient_ok": classifier_grad_ok,
        "expert_gradient_ok": expert_grad_ok,
        "deterministic_eval": True,
        "routing_summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 3:
        raise ValueError("batch_size must be at least 3")
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    report = {
        "status": "passed",
        "device": str(device),
        "torch_version": torch.__version__,
        "cases": [run_case(top_k, device, args.batch_size) for top_k in (1, 2, 3)],
    }
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
