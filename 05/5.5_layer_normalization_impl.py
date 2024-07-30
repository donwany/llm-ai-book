import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# Example usage
input_data = torch.randn(4, 3, 5)  # Batch size 4, sequence length 3, input dimension 5
layer_norm = LayerNormalization()
output = layer_norm(input_data)
print("Output shape:", output.shape)  # Should be the same shape as input_data: (4, 3, 5)
