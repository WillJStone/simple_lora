import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


class LinearLora(nn.Module):
    def __init__(self, linear_layer: nn.Module, lora_rank: int, lora_alpha: float):
        super().__init__()
        self.linear_layer = linear_layer
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        self.A = nn.Parameter(torch.zeros(self.linear_layer.weight.shape[0], self.lora_rank))
        self.B = nn.Parameter(torch.zeros(self.lora_rank, self.linear_layer.weight.shape[1]))

        self.merged = False
        self.linear_layer.weight.requires_grad = False

    def merge(self):
        if self.merged:
            return

        self.linear_layer.weight += nn.Parameter(self.A @ self.B) * self.lora_alpha
        self.merged = True

    def unmerge(self):
        if not self.merged:
            return

        self.linear_layer.weight -= nn.Parameter(self.A @ self.B) * self.lora_alpha
        self.merged = False

    def forward(self, x: Tensor) -> Tensor:
        if self.training and not self.merged:
            self.merge()
        elif not self.training and self.merged:
            self.unmerge()

        return self.linear_layer(x)
