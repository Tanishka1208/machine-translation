"""
attention.py - Scaled Dot-Product Attention and Multi-Head Attention
(nn.MultiheadAttention is NOT used, as per assignment rules)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q : (batch, heads, seq_q, d_k)
            K : (batch, heads, seq_k, d_k)
            V : (batch, heads, seq_v, d_v)   seq_v == seq_k
            mask: (batch, 1, 1, seq_k)  or  (batch, 1, seq_q, seq_k)
                  positions to MASK OUT should be True (will be set to -inf)
        Returns:
            output : (batch, heads, seq_q, d_v)
            attn_w : (batch, heads, seq_q, seq_k)  – attention weights (for visualisation)
        """
        d_k = Q.size(-1)
        # ── scores: (batch, heads, seq_q, seq_k) ──────────────────────
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_w = F.softmax(scores, dim=-1)

        # Replace NaN that appears when an entire row is -inf (padding-only token)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)

        attn_w = self.dropout(attn_w)
        output = torch.matmul(attn_w, V)
        return output, attn_w


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need".
    Projects Q, K, V with h separate linear layers, computes attention in parallel,
    concatenates and projects back.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        # Projection matrices
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout   = nn.Dropout(dropout)

    def split_heads(self, x):
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        batch, seq, _ = x.size()
        x = x.view(batch, seq, self.n_heads, self.d_k)
        return x.transpose(1, 2)   # (batch, heads, seq, d_k)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query : (batch, seq_q, d_model)
            key   : (batch, seq_k, d_model)
            value : (batch, seq_v, d_model)
            mask  : broadcastable mask (True = masked)
        Returns:
            output  : (batch, seq_q, d_model)
            attn_w  : (batch, heads, seq_q, seq_k)
        """
        Q = self.split_heads(self.W_Q(query))   # (b, h, seq_q, d_k)
        K = self.split_heads(self.W_K(key))     # (b, h, seq_k, d_k)
        V = self.split_heads(self.W_V(value))   # (b, h, seq_v, d_k)

        x, attn_w = self.attention(Q, K, V, mask)

        # Concatenate heads: (b, h, seq_q, d_k) → (b, seq_q, d_model)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.size(0), x.size(1), self.d_model)

        output = self.W_O(x)
        return output, attn_w


# ── Mask helpers ───────────────────────────────────────────────────────────────

def make_pad_mask(seq, pad_idx=0):
    """
    Padding mask: True where token == pad_idx.
    seq : (batch, seq_len)
    Returns : (batch, 1, 1, seq_len)  – broadcastable over heads and query positions
    """
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(seq_len, device):
    """
    Causal (look-ahead) mask for the decoder.
    Upper-triangular True means 'cannot attend to future positions'.
    Returns : (1, 1, seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)


def make_decoder_mask(trg, pad_idx=0):
    """
    Combined decoder mask = padding mask OR causal mask.
    trg : (batch, trg_len)
    Returns : (batch, 1, trg_len, trg_len)
    """
    trg_len = trg.size(1)
    pad_mask    = make_pad_mask(trg, pad_idx)                          # (b,1,1,T)
    causal_mask = make_causal_mask(trg_len, trg.device)                # (1,1,T,T)
    return pad_mask | causal_mask                                       # (b,1,T,T)