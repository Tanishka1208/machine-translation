"""
model.py - Full Transformer Model for German→English Translation

Includes:
  • Transformer.forward()  – standard training forward pass
  • Transformer.infer()    – end-to-end inference: German string → English string
  • Transformer.__init__() – loads vocab, tokenizers, and pretrained weights
                             (all inside __init__ as required by the autograder)
"""

import os
import math
import torch
import torch.nn as nn
import spacy
import gdown

from config import cfg
from dataset import Vocabulary, tokenize_de, tokenize_en, get_data
from positional import PositionalEncoding, LearnedPositionalEncoding
from encoder import Encoder
from decoder import Decoder
from attention import make_pad_mask, make_decoder_mask


# ── Generator (final linear + log-softmax) ────────────────────────────────────

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)   # logits; loss fn applies log_softmax internally


# ── Transformer ────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """
    Full sequence-to-sequence Transformer.

    All initialisation (vocab, tokenizers, weight loading) happens inside
    __init__() so the autograder can do:
        model = Transformer().to(device)
        model.eval()
        english = model.infer(german_sentence)
    """

    # ── Google Drive file ID for the saved checkpoint ──────────────────────────
    # Replace this with your own file ID after uploading to Google Drive.
    GDRIVE_FILE_ID   = "1F8wrJB_kyQlTdUiBmllnZPl-KQzTJJuu"   # best_model.pt ID (you already have this)
    GDRIVE_VOCAB_ID  = "1Ho9oCZtXwo6gvfMgpMfr4Gf9idBRS4fh"            # vocab.pt ID (get this from drive)
    WEIGHT_FILENAME  = "best_model.pt"
    VOCAB_FILENAME   = "vocab.pt"

    def __init__(
        self,
        d_model       = cfg.D_MODEL,
        n_heads       = cfg.N_HEADS,
        n_encoder     = cfg.N_ENCODER,
        n_decoder     = cfg.N_DECODER,
        d_ff          = cfg.D_FF,
        dropout       = cfg.DROPOUT,
        max_seq_len   = cfg.MAX_SEQ_LEN,
        pos_encoding  = "sinusoidal",   # "sinusoidal" | "learned"
        load_weights  = True,           # set False during training init
    ):
        super().__init__()

        # ── 1. Load tokenizers ─────────────────────────────────────────
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        # ── 2. Load / build vocabularies ──────────────────────────────
        vocab_path = self.VOCAB_FILENAME
        if os.path.exists(vocab_path):
            saved = torch.load(vocab_path, map_location="cpu")
            self.src_vocab = saved["src_vocab"]
            self.trg_vocab = saved["trg_vocab"]
        else:
            # Build from scratch (first run / training)
            _, _, _, self.src_vocab, self.trg_vocab, _, _ = get_data()
            torch.save(
                {"src_vocab": self.src_vocab, "trg_vocab": self.trg_vocab},
                vocab_path,
            )

        src_vocab_size = len(self.src_vocab)
        trg_vocab_size = len(self.trg_vocab)

        # ── 3. Store hyper-parameters ──────────────────────────────────
        self.d_model     = d_model
        self.max_seq_len = max_seq_len

        # ── 4. Build model layers ──────────────────────────────────────
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=cfg.PAD_IDX)
        self.trg_embed = nn.Embedding(trg_vocab_size, d_model, padding_idx=cfg.PAD_IDX)

        if pos_encoding == "sinusoidal":
            self.src_pe = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
            self.trg_pe = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        else:
            self.src_pe = LearnedPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
            self.trg_pe = LearnedPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        self.encoder   = Encoder(d_model, n_heads, d_ff, n_encoder, dropout)
        self.decoder   = Decoder(d_model, n_heads, d_ff, n_decoder, dropout)
        self.generator = Generator(d_model, trg_vocab_size)

        self._init_parameters()

        # ── 5. Load pretrained weights (for inference / autograder) ────
        if load_weights:
            weight_path = self.WEIGHT_FILENAME
            if not os.path.exists(weight_path):
                print(f"[Transformer] Downloading weights from Google Drive …")
                gdown.download(
                    f"https://drive.google.com/uc?id={self.GDRIVE_FILE_ID}",
                    weight_path,
                    quiet=False,
                )
            if os.path.exists(weight_path):
                state = torch.load(weight_path, map_location="cpu")
                self.load_state_dict(state, strict=False)
                print(f"[Transformer] Weights loaded from {weight_path}")
            else:
                print("[Transformer] WARNING: no weight file found; using random init.")

    # ── Parameter initialisation ───────────────────────────────────────────────

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Mask helpers ──────────────────────────────────────────────────────────

    def make_src_mask(self, src):
        """(batch, src_len) → (batch, 1, 1, src_len), True = masked (pad)"""
        return make_pad_mask(src, cfg.PAD_IDX)

    def make_trg_mask(self, trg):
        """(batch, trg_len) → (batch, 1, trg_len, trg_len), causal + pad"""
        return make_decoder_mask(trg, cfg.PAD_IDX)

    # ── Encode / Decode helpers (used by greedy_decode in utils.py) ───────────

    def encode(self, src, src_mask):
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        enc_out, _ = self.encoder(x, src_mask)
        return enc_out

    def decode(self, trg, enc_out, trg_mask, src_mask):
        x = self.trg_pe(self.trg_embed(trg) * math.sqrt(self.d_model))
        dec_out, _ = self.decoder(x, enc_out, trg_mask, src_mask)
        return dec_out

    # ── Forward pass (training) ───────────────────────────────────────────────

    def forward(self, src, trg):
        """
        src : (batch, src_len)
        trg : (batch, trg_len)   ← teacher-forced, includes <sos> but NOT <eos>
        Returns logits : (batch, trg_len, trg_vocab_size)
                also returns encoder/decoder attention weights for W&B logging
        """
        src_mask = self.make_src_mask(src)   # (b, 1, 1, S)
        trg_mask = self.make_trg_mask(trg)   # (b, 1, T, T)

        # Encoder
        src_emb = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        enc_out, enc_attn = self.encoder(src_emb, src_mask)

        # Decoder
        trg_emb = self.trg_pe(self.trg_embed(trg) * math.sqrt(self.d_model))
        dec_out, dec_attn = self.decoder(trg_emb, enc_out, trg_mask, src_mask)

        logits = self.generator(dec_out)   # (b, T, vocab)
        return logits, enc_attn, dec_attn

    # ── infer() ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def infer(self, german_sentence: str) -> str:
        """
        End-to-end German → English translation.

        Accepts a raw German sentence string, tokenizes it using spacy,
        runs the encoder, then performs autoregressive greedy decoding,
        and returns the English translation as a string.

        This method is called directly by the autograder:
            model = Transformer().to(device)
            model.eval()
            english = model.infer(german_sentence)
        """
        self.eval()
        device = next(self.parameters()).device

        # ── Tokenize & numericalize ────────────────────────────────────
        tokens = [tok.text.lower() for tok in self.spacy_de.tokenizer(german_sentence)]
        tokens = tokens[: self.max_seq_len - 2]
        src_ids = (
            [cfg.SOS_IDX]
            + [self.src_vocab.stoi.get(t, cfg.UNK_IDX) for t in tokens]
            + [cfg.EOS_IDX]
        )
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

        # ── Source mask (no padding here) ─────────────────────────────
        # make_src_mask returns True for PAD positions; src has no pads
        src_mask = self.make_src_mask(src)   # all False

        # ── Encode ────────────────────────────────────────────────────
        src_emb = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        enc_out, _ = self.encoder(src_emb, src_mask)

        # ── Autoregressive greedy decoding ────────────────────────────
        ys = torch.tensor([[cfg.SOS_IDX]], dtype=torch.long, device=device)

        for _ in range(self.max_seq_len):
            trg_mask = self.make_trg_mask(ys)
            trg_emb  = self.trg_pe(self.trg_embed(ys) * math.sqrt(self.d_model))
            dec_out, _ = self.decoder(trg_emb, enc_out, trg_mask, src_mask)
            logits   = self.generator(dec_out[:, -1, :])   # (1, vocab)
            next_tok = logits.argmax(-1).item()
            if next_tok == cfg.EOS_IDX:
                break
            ys = torch.cat(
                [ys, torch.tensor([[next_tok]], dtype=torch.long, device=device)],
                dim=1,
            )

        # ── Detokenize ─────────────────────────────────────────────────
        token_ids   = ys[0, 1:].tolist()   # strip <sos>
        translation = " ".join(
            self.trg_vocab.itos[i]
            for i in token_ids
            if i not in (cfg.SOS_IDX, cfg.EOS_IDX, cfg.PAD_IDX)
        )
        return translation