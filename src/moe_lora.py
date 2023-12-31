"""
Will probably only work for Mistral for now
"""

from typing import List

import torch
from torch import nn


class LoraExpert(nn.Module):
    def __init__(self, lora_rank: int, lora_alpha: float):
        super().__init__()
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self.A = nn.Parameter(torch.zeros(self.lora_rank))
        self.B = nn.Parameter(torch.zeros(self.lora_rank))

        self.merged = False

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        return weights + self.A @ self.B * self.lora_alpha


class MoLoraLayer(nn.Module):
    """
    Modified from Mistral's MoE implementation
    """
    def __init__(self, feed_forward: nn.Module, experts: List[nn.Module], gate: nn.Module, num_experts_per_tok: int):
        super().__init__()
        assert len(experts) > 0
        self.feed_forward = feed_forward
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, inputs: torch.Tensor):
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.gate(inputs_squashed)
        expert_weights, selected_experts = torch.topk(
            gate_logits, self.args.num_experts_per_tok
        )
        expert_weights = nn.functional.softmax(
            expert_weights,
            dim=1,
            dtype=torch.float,
        ).type_as(inputs)
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += expert_weights[batch_idx, nth_expert, None] * expert(
                inputs_squashed[batch_idx]
            )
        return results.view_as(inputs)