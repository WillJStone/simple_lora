"""
Will probably only work for Mistral for now
"""
import math
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class MoLoRAArgs:
    num_experts: int
    num_experts_per_tok: int
    lora_rank: int
    lora_alpha: float


class MistralMoeLayer(nn.Module):
    def __init__(self, feed_forward: nn.Module, gate: nn.Module, args: MoLoRAArgs):
        super().__init__()
        self.feed_forward = feed_forward
        self.gate = gate
        self.num_experts = args.num_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        self.lora_rank = args.lora_rank
        self.lora_alpha = args.lora_alpha

        in_dim = feed_forward.w1.weight.shape[1]
        hidden_dim = feed_forward.w1.weight.shape[0]
        self.A1 = nn.Parameter(torch.zeros(self.num_experts, in_dim, self.lora_rank))
        self.B1 = nn.Parameter(torch.zeros(self.num_experts, self.lora_rank, hidden_dim))
        self.A2 = nn.Parameter(torch.zeros(self.num_experts, in_dim, self.lora_rank))
        self.B2 = nn.Parameter(torch.zeros(self.num_experts, self.lora_rank, hidden_dim))
        self.A3 = nn.Parameter(torch.zeros(self.num_experts, hidden_dim, self.lora_rank))
        self.B3 = nn.Parameter(torch.zeros(self.num_experts, self.lora_rank, in_dim))

        nn.init.kaiming_uniform_(self.A1, a=math.sqrt(5))
        nn.init.zeros_(self.B1)
        nn.init.kaiming_uniform_(self.A2, a=math.sqrt(5))
        nn.init.zeros_(self.B2)
        nn.init.kaiming_uniform_(self.A3, a=math.sqrt(5))
        nn.init.zeros_(self.B3)

    def expert_forward(self, inputs: torch.Tensor, expert_idx: int):
        """
        Does the same forward pass as FeedForward, but with the adapted weights
        """
        w1_prime = self.feed_forward.w1.weight.data + self.lora_alpha * self.A1[expert_idx] @ self.B1[expert_idx]
        w2_prime = self.feed_forward.w2.weight.data + self.lora_alpha * self.A2[expert_idx] @ self.B2[expert_idx]
        w3_prime = self.feed_forward.w3.weight.data + self.lora_alpha * self.A3[expert_idx] @ self.B3[expert_idx]

        hidden = F.silu(F.linear(inputs, w1_prime))
        hidden = hidden * F.linear(inputs, w3_prime)

        return F.linear(hidden, w2_prime)

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for expert_idx in range(self.num_experts):
            batch_idx, nth_expert = torch.where(selected_experts == expert_idx)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * self.expert_forward(
                inputs[batch_idx], expert_idx
            )
            
        return results