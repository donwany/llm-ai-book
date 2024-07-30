import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model  # Dimension of the embedding vectors
        self.vocab_size = vocab_size  # Size of the vocabulary

        # nn.Embedding is a PyTorch layer that converts integer indices to dense embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Multiply the embeddings by sqrt(d_model) to normalize the variance of the embeddings
        return self.embedding(x) * math.sqrt(self.d_model)
