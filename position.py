"""
positional.py - Sinusoidal Positional Encoding and Learned Positional Embedding
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from "Attention Is All You Need".

    PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

    The encoding is registered as a non-trainable buffer.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the encoding matrix once
        pe = torch.zeros(max_len, d_model)                     # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                          # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  indices

        pe = pe.unsqueeze(0)                            # (1, max_len, d_model)

        # Register as buffer (not a parameter → not updated by optimizer)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x : (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embedding (used in W&B experiment 2.4).
    torch.nn.Embedding with max_len positions.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout   = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        x : (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        x = x + self.embedding(positions)
        return self.dropout(x)