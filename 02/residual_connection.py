import torch
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()  # Normalization layer

    def forward(self, x, sublayer):
        # Normalize the input 'x' using LayerNormalization and apply the sublayer operation
        sublayer_output = sublayer(self.norm(x))
        # Apply dropout to the output of the sublayer
        sublayer_output_with_dropout = self.dropout(sublayer_output)
        # Add the original input 'x' to the output of the sublayer
        return x + sublayer_output_with_dropout
