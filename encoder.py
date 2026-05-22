"""
encoder.py - Transformer Encoder Layer and Encoder Stack
"""

import torch.nn as nn
from attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Point-wise Feed-Forward Network:
        FFN(x) = max(0, x W1 + b1) W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """
    One Encoder block:
        x → Self-Attention → Add & Norm → FFN → Add & Norm

    We use Pre-LayerNorm (also called Pre-LN):
        x = x + SubLayer( LayerNorm(x) )

    Justification (required by the assignment):
        Pre-LN stabilises training by normalising inputs before each sub-layer,
        preventing gradient explosion/vanishing in early warm-up steps.
        It makes training less sensitive to the learning rate schedule, and several
        ablation studies show it converges faster and more stably than Post-LN,
        especially for small datasets like Multi30k.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff         = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        x        : (batch, src_len, d_model)
        src_mask : (batch, 1, 1, src_len)
        Returns  : x_out (same shape), attn_weights
        """
        # ── Self-Attention sub-layer (Pre-LN) ─────────────────────────
        residual = x
        x_norm   = self.norm1(x)
        attn_out, attn_w = self.self_attn(x_norm, x_norm, x_norm, src_mask)
        x = residual + self.dropout(attn_out)

        # ── Feed-Forward sub-layer (Pre-LN) ───────────────────────────
        residual = x
        x = residual + self.dropout(self.ff(self.norm2(x)))

        return x, attn_w


class Encoder(nn.Module):
    """Stack of N encoder layers."""

    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers  = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm    = nn.LayerNorm(d_model)   # final norm (Pre-LN convention)

    def forward(self, x, src_mask=None):
        """
        x        : (batch, src_len, d_model)
        Returns  : (batch, src_len, d_model), list of attn_weights per layer
        """
        attn_weights_all = []
        for layer in self.layers:
            x, attn_w = layer(x, src_mask)
            attn_weights_all.append(attn_w)
        return self.norm(x), attn_weights_all