"""
Will probably only work for Mistral for now
"""
import dataclasses
import math
from typing import List

import torch
from torch import nn
from torch.nn import functional as F


@dataclasses.dataclass
class MoLoRAArgs:
    num_experts: int
    num_experts_per_tok: int
    lora_rank: int
    lora_alpha: float


class MistralMoLoraLayer(nn.Module):
    def __init__(self, feed_forward: nn.Module, gate: nn.Module, args: MoLoRAArgs):
        super().__init__()
        self.feed_forward = feed_forward
        self.gate = gate
        self.num_experts = args.num_experts
        self.num_experts_per_tok = args.num_experts_per_tok
        self.lora_rank = args.lora_rank
        self.lora_alpha = args.lora_alpha

        in_dim = feed_forward.up_proj.weight.shape[1]
        hidden_dim = feed_forward.up_proj.weight.shape[0]
        dtype = feed_forward.up_proj.weight.dtype
        self.A1 = nn.Parameter(torch.zeros(self.num_experts, in_dim, self.lora_rank, dtype=dtype))
        self.B1 = nn.Parameter(torch.zeros(self.num_experts, self.lora_rank, hidden_dim, dtype=dtype))
        self.A2 = nn.Parameter(torch.zeros(self.num_experts, in_dim, self.lora_rank, dtype=dtype))
        self.B2 = nn.Parameter(torch.zeros(self.num_experts, self.lora_rank, hidden_dim, dtype=dtype))
        self.A3 = nn.Parameter(torch.zeros(self.num_experts, hidden_dim, self.lora_rank, dtype=dtype))
        self.B3 = nn.Parameter(torch.zeros(self.num_experts, self.lora_rank, in_dim, dtype=dtype))

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
        w1_prime = self.feed_forward.up_proj.weight.data + self.lora_alpha * self.A1[expert_idx] @ self.B1[expert_idx]
        w2_prime = self.feed_forward.down_proj.weight.data + self.lora_alpha * self.A2[expert_idx] @ self.B2[expert_idx]
        w3_prime = self.feed_forward.gat_proj.weight.data + self.lora_alpha * self.A3[expert_idx] @ self.B3[expert_idx]

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


def replace_layers(model, layer_to_replace, new_layer, args: MoLoRAArgs):
    """
    Recursively replaces all instances of a given layer type in a PyTorch model, in-place.

    Parameters:
    model (nn.Module): The PyTorch model to modify.
    layer_to_replace (class): The type of the layer to replace.
    new_layer (nn.Module): The new layer to insert in place of the old layer.

    Returns:
    None: The model is modified in-place, so nothing is returned.
    """
    for name, child in model.named_children():
        if isinstance(child, layer_to_replace):
            # Replace the layer in-place
            gate = torch.nn.Linear(child.up_proj.weight.shape[1], args.num_experts, bias=False, dtype=child.up_proj.weight.dtype)
            setattr(model, name, new_layer(child, gate, args))
        else:
            # Recursively apply the function to child modules
            replace_layers(child, layer_to_replace, new_layer, args)