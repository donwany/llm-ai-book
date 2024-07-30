import torch
import torch.nn as nn
import math


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, 'd_model is not divisible by h'
        self.d_k = d_model // h

        # Define weight matrices for query, key, value, and output transformations
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Calculate the attention scores as per the formula
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # Apply the mask if provided
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        # Apply dropout if provided
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # Calculate the weighted sum of values and attention scores
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Transform query, key, and value using linear transformations
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Split the query, key, and value tensors into multiple heads
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Apply attention mechanism to each head
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Concatenate the outputs of different heads
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Apply linear transformation to the concatenated output
        return self.w_o(x)
