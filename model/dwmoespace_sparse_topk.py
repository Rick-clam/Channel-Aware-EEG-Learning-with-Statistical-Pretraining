"""Genuinely sparse sample-level Top-K routing for temporal EEG experts.

The legacy entry points expose ``--top_k`` but their active models compute every
expert and aggregate all outputs.  This module dispatches sample subsets to only
the selected experts.  It intentionally lives beside the legacy models so their
historical behavior remains reproducible.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FilterAxisLayerNorm2d(nn.Module):
    """Normalize filters without mixing channel or temporal positions."""

    def __init__(self, num_filters: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_filters, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_filters, 1, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        return (
            (x - mean)
            * torch.rsqrt(variance + self.eps)
            * self.weight
            + self.bias
        )


class SparseTemporalConv(nn.Module):
    """Shared frontend returning an explicit [B,C,F,T] latent grid."""

    def __init__(self, norm_type: str = "local_filter") -> None:
        super().__init__()
        if norm_type not in ("legacy_global", "local_filter"):
            raise ValueError(
                "norm_type must be 'legacy_global' or 'local_filter'"
            )
        self.convolutions = nn.ModuleList(
            [
                nn.Conv2d(1, 5, (1, 15), (1, 5), (0, 7)),
                nn.Conv2d(5, 8, (1, 15), (1, 1), (0, 7)),
                nn.Conv2d(8, 10, (1, 15), (1, 1), (0, 7)),
                nn.Conv2d(10, 16, (1, 3), (1, 2), (0, 1)),
                nn.Conv2d(16, 20, (1, 3), (1, 1), (0, 1)),
                nn.Conv2d(20, 25, (1, 3), (1, 1), (0, 1)),
            ]
        )
        filter_counts = (5, 8, 10, 16, 20, 25)
        norm_factory = (
            FilterAxisLayerNorm2d
            if norm_type == "local_filter"
            else lambda count: nn.GroupNorm(1, count)
        )
        self.norms = nn.ModuleList(
            [norm_factory(count) for count in filter_counts]
        )
        self.target_steps = (250, 200, None, 100, 80, None)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, return_grid: bool = False) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 16:
            raise ValueError(
                "Sparse temporal frontend requires [B,16,T], received "
                f"{tuple(x.shape)}"
            )
        latent = x.unsqueeze(1).to(dtype=torch.float32)
        for convolution, norm, target_steps in zip(
            self.convolutions, self.norms, self.target_steps
        ):
            latent = self.activation(norm(convolution(latent)))
            if target_steps is not None:
                latent = F.interpolate(
                    latent, size=(16, target_steps), mode="nearest"
                )
        grid = latent.permute(0, 2, 1, 3).contiguous()
        if return_grid:
            return grid
        return grid.flatten(start_dim=2)


class SparseTemporalScaleExpert(nn.Module):
    """One temporal-scale expert operating only on the latent-time axis."""

    def __init__(
        self,
        *,
        filter_dim: int = 25,
        embed_dim: int = 63,
        output_dim: int = 100,
        temporal_kernel: int = 7,
        pooled_steps: int = 4,
    ) -> None:
        super().__init__()
        if temporal_kernel <= 0 or temporal_kernel % 2 != 1:
            raise ValueError("temporal_kernel must be a positive odd integer")
        self.temporal_kernel = int(temporal_kernel)
        self.output_dim = int(output_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(
                filter_dim,
                embed_dim,
                kernel_size=self.temporal_kernel,
                padding=self.temporal_kernel // 2,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(pooled_steps),
        )
        self.projection = nn.Linear(embed_dim * pooled_steps, output_dim)

    def forward(self, latent_grid: torch.Tensor) -> torch.Tensor:
        if latent_grid.ndim != 4:
            raise ValueError(
                "Sparse temporal experts require [B,C,F,T], received "
                f"{tuple(latent_grid.shape)}"
            )
        batch, channels, filters, steps = latent_grid.shape
        tokens = latent_grid.reshape(batch * channels, filters, steps)
        encoded = self.encoder(tokens).flatten(start_dim=1)
        return self.projection(encoded).view(batch, channels, self.output_dim)


class SparseTopKChannelExpertMoE(nn.Module):
    """Route each sample to K experts and execute only those sample paths."""

    def __init__(
        self,
        *,
        expert_output_dim: int = 100,
        router_hidden_dim: int = 32,
        expert_embed_dim: int = 63,
        expert_kernels: Iterable[int] = (3, 7, 15),
        top_k: int = 1,
        router_temperature: float = 1.0,
        norm_type: str = "local_filter",
    ) -> None:
        super().__init__()
        if isinstance(expert_kernels, str):
            expert_kernels = tuple(
                int(value.strip())
                for value in expert_kernels.split(",")
                if value.strip()
            )
        else:
            expert_kernels = tuple(int(value) for value in expert_kernels)
        if not expert_kernels:
            raise ValueError("expert_kernels must not be empty")
        if any(kernel <= 0 or kernel % 2 != 1 for kernel in expert_kernels):
            raise ValueError("expert_kernels must contain positive odd integers")
        if not 1 <= int(top_k) <= len(expert_kernels):
            raise ValueError(
                f"top_k must be in [1,{len(expert_kernels)}], got {top_k}"
            )
        if router_temperature <= 0:
            raise ValueError("router_temperature must be positive")

        self.num_channels = 16
        self.expert_output_dim = int(expert_output_dim)
        self.expert_kernels: Tuple[int, ...] = expert_kernels
        self.num_experts = len(expert_kernels)
        self.top_k = int(top_k)
        self.router_temperature = float(router_temperature)
        self.temporal_conv = SparseTemporalConv(norm_type=norm_type)
        self.experts = nn.ModuleList(
            [
                SparseTemporalScaleExpert(
                    output_dim=self.expert_output_dim,
                    embed_dim=expert_embed_dim,
                    temporal_kernel=kernel,
                )
                for kernel in expert_kernels
            ]
        )
        descriptor_dim = self.num_channels * 3
        self.router = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, self.num_experts),
        )
        nn.init.normal_(self.router[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.router[-1].bias)

        self.routing_aux_loss = None
        self.last_gate_weights = None
        self.last_router_probabilities = None
        self.last_selected_experts = None
        self.last_executed_sample_counts = None
        self.reset_routing_stats()

    def reset_routing_stats(self) -> None:
        self._routing_sample_count = 0
        self._routing_assignment_count = 0
        self._routing_expert_counts = torch.zeros(
            self.num_experts, dtype=torch.long
        )
        self._routing_forward_count = 0

    @staticmethod
    def _raw_descriptors(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 16:
            raise ValueError(
                "Sparse Top-K routing requires [B,16,T] EEG input, received "
                f"{tuple(x.shape)}"
            )
        descriptors = torch.stack(
            (
                x.mean(dim=-1),
                x.std(dim=-1, unbiased=False),
                x.abs().mean(dim=-1),
            ),
            dim=-1,
        )
        return descriptors.flatten(start_dim=1)

    def _route(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.router(self._raw_descriptors(x))
        probabilities = torch.softmax(
            logits / self.router_temperature, dim=-1
        )
        _, selected = torch.topk(
            logits, k=self.top_k, dim=-1, largest=True, sorted=True
        )
        hard_mask = torch.zeros_like(probabilities).scatter(
            1, selected, 1.0
        )
        sparse_probabilities = probabilities * hard_mask
        sparse_probabilities = sparse_probabilities / sparse_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)

        # With K=1, ordinary renormalization is identically one and supplies no
        # task gradient to the router.  The forward value stays exactly hard and
        # sparse, while this straight-through term provides a probability
        # surrogate for backward propagation.
        dispatch_weights = sparse_probabilities
        if self.top_k == 1 and self.training:
            dispatch_weights = (
                sparse_probabilities
                + probabilities
                - probabilities.detach()
            )

        importance = probabilities.mean(dim=0)
        hard_load = hard_mask.mean(dim=0).detach() / self.top_k
        # Switch-style differentiable load-balancing objective.  The hard load
        # records actual dispatched paths, while importance supplies gradients.
        self.routing_aux_loss = self.num_experts * torch.sum(
            importance * hard_load
        )
        return {
            "logits": logits,
            "probabilities": probabilities,
            "selected": selected,
            "hard_mask": hard_mask,
            "sparse_probabilities": sparse_probabilities,
            "dispatch_weights": dispatch_weights,
        }

    def forward(
        self, x: torch.Tensor, *, return_details: bool = False
    ):
        route = self._route(x)
        latent_grid = self.temporal_conv(x, return_grid=True)
        batch = latent_grid.shape[0]
        mixture = latent_grid.new_zeros(
            batch, self.num_channels, self.expert_output_dim
        )
        executed_counts = torch.zeros(
            self.num_experts, dtype=torch.long, device=latent_grid.device
        )

        # Sparse dispatch: each expert sees only its selected sample subset.
        for expert_index, expert in enumerate(self.experts):
            sample_indices = torch.nonzero(
                route["hard_mask"][:, expert_index], as_tuple=False
            ).flatten()
            if sample_indices.numel() == 0:
                continue
            selected_latents = latent_grid.index_select(0, sample_indices)
            expert_output = expert(selected_latents)
            weights = route["dispatch_weights"].index_select(
                0, sample_indices
            )[:, expert_index].view(-1, 1, 1)
            mixture = mixture.index_add(
                0, sample_indices, expert_output * weights
            )
            executed_counts[expert_index] = sample_indices.numel()

        expected_assignments = batch * self.top_k
        actual_assignments = int(executed_counts.sum().item())
        if actual_assignments != expected_assignments:
            raise RuntimeError(
                "Sparse dispatch count mismatch: "
                f"expected {expected_assignments}, got {actual_assignments}"
            )

        diagnostic_weights = route["sparse_probabilities"].detach()
        self.last_gate_weights = (
            diagnostic_weights.unsqueeze(1)
            .expand(-1, self.num_channels, -1)
            .cpu()
        )
        self.last_router_probabilities = route["probabilities"].detach().cpu()
        self.last_selected_experts = route["selected"].detach().cpu()
        self.last_executed_sample_counts = executed_counts.detach().cpu()
        self._routing_sample_count += int(batch)
        self._routing_assignment_count += actual_assignments
        self._routing_expert_counts += executed_counts.detach().cpu()
        self._routing_forward_count += 1

        if return_details:
            return mixture, {
                "router_logits": route["logits"],
                "router_probabilities": route["probabilities"],
                "sparse_gate_weights": route["sparse_probabilities"],
                "selected_experts": route["selected"],
                "executed_sample_counts": executed_counts,
                "routing_aux_loss": self.routing_aux_loss,
            }
        return mixture

    def routing_summary(self) -> Dict[str, object]:
        expected = self._routing_sample_count * self.top_k
        counts = self._routing_expert_counts.tolist()
        denominator = max(expected, 1)
        return {
            "routing_scope": "sample",
            "num_experts": self.num_experts,
            "expert_kernels": list(self.expert_kernels),
            "top_k": self.top_k,
            "active_expert_fraction": self.top_k / self.num_experts,
            "num_forward_calls": self._routing_forward_count,
            "num_samples": self._routing_sample_count,
            "expected_sample_expert_assignments": expected,
            "executed_sample_expert_assignments": (
                self._routing_assignment_count
            ),
            "expert_assignment_counts": counts,
            "expert_assignment_rates": [count / denominator for count in counts],
            "only_selected_samples_executed": (
                self._routing_assignment_count == expected
            ),
        }


class dwmoespace_sparse_topk(nn.Module):
    """Classifier wrapper matching the existing run_moe entry points."""

    def __init__(
        self,
        num_classes: int,
        *,
        expert_output_dim: int = 100,
        router_hidden_dim: int = 32,
        expert_kernels: Iterable[int] = (3, 7, 15),
        top_k: int = 1,
        router_temperature: float = 1.0,
        norm_type: str = "local_filter",
    ) -> None:
        super().__init__()
        self.moe = SparseTopKChannelExpertMoE(
            expert_output_dim=expert_output_dim,
            router_hidden_dim=router_hidden_dim,
            expert_kernels=expert_kernels,
            top_k=top_k,
            router_temperature=router_temperature,
            norm_type=norm_type,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(expert_output_dim, num_classes),
        )

    @property
    def routing_aux_loss(self):
        return self.moe.routing_aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.moe(x).transpose(1, 2)
        return self.classifier(features)

    def forward_with_routing_details(self, x: torch.Tensor):
        mixture, routing = self.moe(x, return_details=True)
        return {
            "logits": self.classifier(mixture.transpose(1, 2)),
            **routing,
        }
