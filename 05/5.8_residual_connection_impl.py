import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        sublayer_output = sublayer(self.norm(x))
        sublayer_output_with_dropout = self.dropout(sublayer_output)
        return x + sublayer_output_with_dropout


# Example sublayer operation: FeedForwardBlock
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


# Example usage
batch_size = 4
seq_len = 3
d_model = 5
d_ff = 10
dropout = 0.1

# Initialize layers
ff_block = FeedForwardBlock(d_model, d_ff, dropout)
residual_connection = ResidualConnection(dropout)

# Example input tensor
input_tensor = torch.randn(batch_size, seq_len, d_model)

# Apply residual connection
output_tensor = residual_connection(input_tensor, ff_block)
print("Output shape:", output_tensor.shape)  # Should be (batch_size, seq_len, d_model)
