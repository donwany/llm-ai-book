import torch
import torch.nn as nn


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

# Initialize the FeedForwardBlock
ff_block = FeedForwardBlock(d_model, d_ff, dropout)

# Example input tensor of shape (batch_size, seq_len, d_model)
input_tensor = torch.randn(batch_size, seq_len, d_model)

# Forward pass
output_tensor = ff_block(input_tensor)
print("Output shape:", output_tensor.shape)  # Should be (batch_size, seq_len, d_model)
