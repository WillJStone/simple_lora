import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


DEFAULT_LORA_CONFIG = {
    "lora_rank": 10,
    "lora_alpha": 0.1,
    "lora_layers": ["wq", "wk"],
}


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


class LoraModel(nn.Module):
    def __init__(self, model: nn.Module, lora_config: dict):
        super().__init__()
        self.model = model
        self.lora_config = lora_config

    def inject_lora_layers(self):
        def apply_lora_recursively(layer, parent, name):
            # If the layer is a leaf node and its name is in the lora_layers list, apply LinearLora
            if len(list(layer.children())) == 0 and name in self.lora_config["lora_layers"]:
                lora_layer = LinearLora(
                    layer,
                    self.lora_config["lora_rank"],
                    self.lora_config["lora_alpha"],
                )
                setattr(parent, name, lora_layer)
            else:
                # Recursively apply to all children
                for child_name, child_layer in layer.named_children():
                    apply_lora_recursively(child_layer, layer, child_name)

        # Start recursion from the top model layer
        for name, layer in self.model.named_children():
            apply_lora_recursively(layer, self.model, name)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)