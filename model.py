"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
#  Exposed at module level so the autograder can import and test it
#  independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)

    # ── scores: (..., seq_q, seq_k) ───────────────────────────────────
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(scores, dim=-1)

    # Replace NaN that appears when an entire row is -inf (all-padding token)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)

    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
#  Exposed at module level so they can be tested independently and
#  reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # (batch, src_len) → (batch, 1, 1, src_len)
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    tgt_len = tgt.size(1)

    # Padding mask: (batch, 1, 1, tgt_len)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal mask: (1, 1, tgt_len, tgt_len) — upper triangle = True
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1
    ).bool().unsqueeze(0).unsqueeze(0)

    # Combine: (batch, 1, tgt_len, tgt_len)
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head

        # Projection matrices (no bias as in the paper)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        B, seq, _ = x.size()
        return x.view(B, seq, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        Q = self._split_heads(self.W_Q(query))   # (B, h, seq_q, d_k)
        K = self._split_heads(self.W_K(key))     # (B, h, seq_k, d_k)
        V = self._split_heads(self.W_V(value))   # (B, h, seq_k, d_k)

        # Scaled dot-product attention across all heads simultaneously
        x, _ = scaled_dot_product_attention(Q, K, V, mask)  # (B, h, seq_q, d_k)

        # Merge heads: (B, h, seq_q, d_k) → (B, seq_q, d_model)
        B, _, seq_q, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(B, seq_q, self.d_model)

        return self.W_O(x)   # (B, seq_q, d_model)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build encoding matrix
        pe       = torch.zeros(max_len, d_model)                        # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()        # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                                # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)   # even-indexed dims
        pe[:, 1::2] = torch.cos(position * div_term)   # odd-indexed  dims

        pe = pe.unsqueeze(0)   # (1, max_len, d_model)

        # Register as buffer — NOT a trainable parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Uses Pre-LayerNorm (Pre-LN):
        x = x + SubLayer( LayerNorm(x) )

    Justification: Pre-LN normalises inputs before each sub-layer,
    preventing gradient explosion/vanishing in early warm-up steps.
    It makes training less sensitive to the learning rate and converges
    faster than Post-LN, especially for small datasets like Multi30k.

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]
        """
        # Pre-LN self-attention sub-layer
        _x = self.norm1(x)
        x  = x + self.dropout(self.self_attn(_x, _x, _x, src_mask))

        # Pre-LN feed-forward sub-layer
        x  = x + self.dropout(self.ff(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Masked self-attention (Pre-LN)
        _x = self.norm1(x)
        x  = x + self.dropout(self.self_attn(_x, _x, _x, tgt_mask))

        # Cross-attention (Pre-LN)
        _x = self.norm2(x)
        x  = x + self.dropout(self.cross_attn(_x, memory, memory, src_mask))

        # Feed-forward (Pre-LN)
        x  = x + self.dropout(self.ff(self.norm3(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 256).
        N              (int)  : Number of encoder/decoder layers (default 3).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 512).
        dropout        (float): Dropout probability (default 0.1).
        checkpoint_path(str)  : If provided, download & load weights from GDrive.
    """

    # ── Fill these in before Gradescope submission ─────────────────────
    GDRIVE_WEIGHT_ID = "1rUR_rMA8mqZQ_DS02DE3WqmrhTVS0Lzw"   # ← best_model.pt GDrive ID
    GDRIVE_VOCAB_ID  = "1B3nkGw8rdKx_w2Ur8Qi1PiqFCHC6aB5D"    # ← vocab.pt      GDrive ID

    # Special token indices (must match dataset.py)
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(
        self,
        src_vocab_size: int = 1,     # will be overridden after vocab is loaded
        tgt_vocab_size: int = 1,     # will be overridden after vocab is loaded
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
        max_len:   int   = 100,
        checkpoint_path: str = None,
    ) -> None:
        super().__init__()

        # Store hyperparameters
        self.d_model  = d_model
        self.max_len  = max_len

        # ── 1. Load spacy tokenizers ───────────────────────────────────
        import spacy
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        # ── 2. Download vocab.pt if not present, then load ─────────────
        # ── 2. Load vocab.pt if it exists, else use passed-in vocab sizes ──
        vocab_path = "vocab.pt"
        if os.path.exists(vocab_path):
            saved = torch.load(vocab_path, map_location="cpu", weights_only=False)
            self.src_vocab = saved["src_vocab"]
            self.trg_vocab = saved["trg_vocab"]
            src_vocab_size = len(self.src_vocab)
            tgt_vocab_size = len(self.trg_vocab)
        elif "YOUR_VOCAB" not in self.GDRIVE_VOCAB_ID:
            print("[Transformer] Downloading vocab.pt from Google Drive …")
            gdown.download(
                f"https://drive.google.com/uc?id={self.GDRIVE_VOCAB_ID}",
                vocab_path,
                quiet=False,
            )
            saved = torch.load(vocab_path, map_location="cpu", weights_only=False)
            self.src_vocab = saved["src_vocab"]
            self.trg_vocab = saved["trg_vocab"]
            src_vocab_size = len(self.src_vocab)
            tgt_vocab_size = len(self.trg_vocab)
        else:
            # During training — vocab injected after __init__ via model.src_vocab = ...
            self.src_vocab = None
            self.trg_vocab = None
            # src_vocab_size and tgt_vocab_size already passed as arguments, use them

        # ── 3. Build model layers ──────────────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=self.PAD_IDX)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=self.PAD_IDX)

        self.src_pe = PositionalEncoding(d_model, dropout, max_len)
        self.tgt_pe = PositionalEncoding(d_model, dropout, max_len)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

        # Xavier uniform initialisation
        self._init_parameters()

        # ── 4. Download weights if not present, then load ──────────────
        weight_path = checkpoint_path if checkpoint_path else "best_model.pt"
        if not os.path.exists(weight_path):
            # Only download if a real GDrive ID has been set (i.e. after training)
            if "YOUR_WEIGHT" not in self.GDRIVE_WEIGHT_ID:
                print("[Transformer] Downloading best_model.pt from Google Drive …")
                gdown.download(
                    f"https://drive.google.com/uc?id={self.GDRIVE_WEIGHT_ID}",
                    weight_path,
                    quiet=False,
                )
        if os.path.exists(weight_path):
            state = torch.load(weight_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            self.load_state_dict(state, strict=False)
            print(f"[Transformer] Weights loaded from {weight_path}")
        else:
            print("[Transformer] WARNING: no weights found, using random init.")

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS — keep these signatures exactly ──────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ── infer() ───────────────────────────────────────────────────────

    @torch.no_grad()
    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.

        Args:
            src_sentence: The raw German text.

        Returns:
            The fully translated English string, detokenized and clean.

        Called by autograder:
            model = Transformer().to(device)
            model.eval()
            english = model.infer(german_sentence)
        """
        self.eval()
        device = next(self.parameters()).device

        # ── Tokenize & numericalize ────────────────────────────────────
        tokens = [tok.text.lower() for tok in self.spacy_de.tokenizer(src_sentence)]
        tokens = tokens[: self.max_len - 2]
        src_ids = (
            [self.SOS_IDX]
            + [self.src_vocab.stoi.get(t, self.UNK_IDX) for t in tokens]
            + [self.EOS_IDX]
        )
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

        # ── Build source mask ──────────────────────────────────────────
        src_mask = make_src_mask(src, self.PAD_IDX)   # (1,1,1,src_len) all False

        # ── Encode ────────────────────────────────────────────────────
        memory = self.encode(src, src_mask)

        # ── Autoregressive greedy decoding ────────────────────────────
        ys = torch.tensor([[self.SOS_IDX]], dtype=torch.long, device=device)

        for _ in range(self.max_len):
            tgt_mask = make_tgt_mask(ys, self.PAD_IDX)
            logits   = self.decode(memory, src_mask, ys, tgt_mask)  # (1,t,vocab)
            next_tok = logits[:, -1, :].argmax(-1).item()
            if next_tok == self.EOS_IDX:
                break
            ys = torch.cat(
                [ys, torch.tensor([[next_tok]], dtype=torch.long, device=device)],
                dim=1,
            )

        # ── Detokenize ─────────────────────────────────────────────────
        token_ids = ys[0, 1:].tolist()   # strip leading <sos>
        translation = " ".join(
            self.trg_vocab.itos[i]
            for i in token_ids
            if i not in (self.SOS_IDX, self.EOS_IDX, self.PAD_IDX)
        )
        return translation