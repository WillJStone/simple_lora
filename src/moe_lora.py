"""
Will probably only work for Mistral for now
"""

from typing import List

import torch
from torch import nn
from torch.nn import functional as F


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


class MoeLayer(nn.Module):
    def __init__(self, feed_forward: nn.Module, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.feed_forward = feed_forward
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        return results