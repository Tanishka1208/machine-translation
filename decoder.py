"""
decoder.py - Transformer Decoder Layer and Decoder Stack
"""

import torch.nn as nn
from attention import MultiHeadAttention
from encoder import FeedForward


class DecoderLayer(nn.Module):
    """
    One Decoder block (Pre-LN):
        x → Masked Self-Attention → Add & Norm
          → Cross-Attention        → Add & Norm
          → FFN                    → Add & Norm
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff         = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, enc_out, trg_mask, src_mask):
        """
        x        : (batch, trg_len, d_model)  – decoder input
        enc_out  : (batch, src_len, d_model)  – encoder output
        trg_mask : (batch, 1, trg_len, trg_len)  – causal + pad mask
        src_mask : (batch, 1, 1,       src_len)  – encoder pad mask
        Returns  : (batch, trg_len, d_model), self_attn_w, cross_attn_w
        """
        # ── Masked Self-Attention ──────────────────────────────────────
        residual = x
        x_norm   = self.norm1(x)
        self_out, self_attn_w = self.self_attn(x_norm, x_norm, x_norm, trg_mask)
        x = residual + self.dropout(self_out)

        # ── Cross-Attention ────────────────────────────────────────────
        residual  = x
        x_norm    = self.norm2(x)
        cross_out, cross_attn_w = self.cross_attn(x_norm, enc_out, enc_out, src_mask)
        x = residual + self.dropout(cross_out)

        # ── Feed-Forward ───────────────────────────────────────────────
        residual = x
        x = residual + self.dropout(self.ff(self.norm3(x)))

        return x, self_attn_w, cross_attn_w


class Decoder(nn.Module):
    """Stack of N decoder layers."""

    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, trg_mask, src_mask):
        """
        Returns: (batch, trg_len, d_model), list of (self_attn, cross_attn) per layer
        """
        attn_weights_all = []
        for layer in self.layers:
            x, self_attn_w, cross_attn_w = layer(x, enc_out, trg_mask, src_mask)
            attn_weights_all.append((self_attn_w, cross_attn_w))
        return self.norm(x), attn_weights_all