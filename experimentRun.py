"""
wandb_experiments.py — All 5 W&B Experiments for DA6401 Assignment 3

Experiment 2.1 — Noam Scheduler vs Fixed LR
Experiment 2.2 — Scaling Factor 1/sqrt(dk) Ablation + Gradient Norms
Experiment 2.3 — Attention Rollout & Head Specialisation
Experiment 2.4 — Sinusoidal vs Learned Positional Encoding
Experiment 2.5 — Label Smoothing eps=0.1 vs eps=0.0 + Prediction Confidence

Run individual experiments:
    python wandb_experiments.py --exp 2.1
    python wandb_experiments.py --exp 2.2
    python wandb_experiments.py --exp 2.3
    python wandb_experiments.py --exp 2.4
    python wandb_experiments.py --exp 2.5
    python wandb_experiments.py --exp all
"""

import os
import math
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Local imports ──────────────────────────────────────────────────────────────
from dataset import get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX
from model import (
    Transformer, make_src_mask, make_tgt_mask,
    PositionalEncoding, MultiHeadAttention,
    EncoderLayer, Encoder, DecoderLayer, Decoder,
    PositionwiseFeedForward, scaled_dot_product_attention,
)
from lr_scheduler import NoamScheduler
from train import LabelSmoothingLoss, run_epoch, greedy_decode, evaluate_bleu


# ══════════════════════════════════════════════════════════════════════════════
# SHARED CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_CONFIG = dict(
    d_model      = 256,
    N            = 3,
    num_heads    = 8,
    d_ff         = 512,
    dropout      = 0.1,
    batch_size   = 128,
    epochs       = 15,
    warmup_steps = 400,
    smoothing    = 0.1,
    max_len      = 100,
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT = "da6401-assignment3"


# ══════════════════════════════════════════════════════════════════════════════
# LEARNED POSITIONAL ENCODING  (used in Experiment 2.4)
# ══════════════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """torch.nn.Embedding-based learned positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout    = nn.Dropout(dropout)
        self.embedding  = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len   = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — build a fresh Transformer with optional PE type
# ══════════════════════════════════════════════════════════════════════════════

def build_model(src_vocab_size, tgt_vocab_size, cfg, pe_type="sinusoidal"):
    """
    Build a Transformer from scratch (no weight download) for experiments.
    pe_type: "sinusoidal" | "learned"
    """
    model = Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model        = cfg["d_model"],
        N              = cfg["N"],
        num_heads      = cfg["num_heads"],
        d_ff           = cfg["d_ff"],
        dropout        = cfg["dropout"],
        max_len        = cfg["max_len"],
        checkpoint_path= None,   # no download during experiments
    )
    # Swap positional encoding if learned
    if pe_type == "learned":
        model.src_pe = LearnedPositionalEncoding(cfg["d_model"], cfg["dropout"], cfg["max_len"])
        model.tgt_pe = LearnedPositionalEncoding(cfg["d_model"], cfg["dropout"], cfg["max_len"])
    return model.to(DEVICE)


def build_optimizer_and_scheduler(model, cfg, use_noam=True, fixed_lr=1e-4):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr    = 1.0,
        betas = (0.9, 0.98),
        eps   = 1e-9,
    )
    if use_noam:
        scheduler = NoamScheduler(optimizer, d_model=cfg["d_model"],
                                  warmup_steps=cfg["warmup_steps"])
    else:
        # Fixed LR: set it directly, wrap in a no-op scheduler
        for pg in optimizer.param_groups:
            pg["lr"] = fixed_lr
        scheduler = _FixedLRWrapper(optimizer, fixed_lr)
    return optimizer, scheduler


class _FixedLRWrapper:
    """Mimics scheduler interface but keeps LR constant."""
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self._lr       = lr
        self._step     = 0
    def step(self):
        self._step += 1
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._lr
    def state_dict(self):
        return {"lr": self._lr, "step": self._step}
    def load_state_dict(self, d):
        self._lr   = d["lr"]
        self._step = d["step"]


def get_prediction_confidence(logits, targets, pad_idx=PAD_IDX):
    """
    Mean softmax probability assigned to the CORRECT token (excluding pads).
    logits  : (B*T, vocab)
    targets : (B*T,)
    """
    with torch.no_grad():
        probs    = F.softmax(logits, dim=-1)
        correct  = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        non_pad  = targets.ne(pad_idx)
        return correct[non_pad].mean().item() if non_pad.sum() > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# CORE TRAINING LOOP WITH DETAILED W&B LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def train_with_wandb(
    run_name,
    cfg,
    src_vocab,
    tgt_vocab,
    train_loader,
    val_loader,
    test_loader,
    spacy_de,
    use_noam          = True,
    fixed_lr          = 1e-4,
    pe_type           = "sinusoidal",
    label_smoothing   = 0.1,
    log_grad_norms    = False,   # Experiment 2.2
    log_confidence    = False,   # Experiment 2.5
    extra_config      = None,
    max_grad_steps    = None,    # stop after N steps (Experiment 2.2)
):
    """
    Full training run with W&B logging. Returns trained model.
    """
    run_cfg = {**cfg,
               "use_noam":       use_noam,
               "fixed_lr":       fixed_lr if not use_noam else None,
               "pe_type":        pe_type,
               "label_smoothing":label_smoothing}
    if extra_config:
        run_cfg.update(extra_config)

    wandb.init(project=WANDB_PROJECT, name=run_name, config=run_cfg, reinit=True)

    model     = build_model(len(src_vocab), len(tgt_vocab), cfg, pe_type)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg, use_noam=use_noam, fixed_lr=fixed_lr
    )
    loss_fn   = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=label_smoothing)

    # References to Q, K weight matrices in first encoder layer (Exp 2.2)
    enc0_W_Q = model.encoder.layers[0].self_attn.W_Q.weight
    enc0_W_K = model.encoder.layers[0].self_attn.W_K.weight

    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss   = 0.0
        total_tokens = 0

        pbar = tqdm(train_loader, desc=f"[{run_name}] Epoch {epoch}", leave=False)

        for src, tgt in pbar:
            if max_grad_steps and global_step >= max_grad_steps:
                break

            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input  = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            src_mask = make_src_mask(src, PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_input, PAD_IDX)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            B, T, V = logits.shape

            loss = loss_fn(logits.reshape(B * T, V), tgt_target.reshape(B * T))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if hasattr(scheduler, "step"):
                scheduler.step()

            global_step  += 1
            n_tok         = tgt_target.ne(PAD_IDX).sum().item()
            total_loss   += loss.item() * n_tok
            total_tokens += n_tok

            step_log = {
                "train/step_loss": loss.item(),
                "train/lr":        optimizer.param_groups[0]["lr"],
                "global_step":     global_step,
            }

            # ── Gradient norm logging (Experiment 2.2) ────────────────
            if log_grad_norms and enc0_W_Q.grad is not None:
                step_log["grad_norm/W_Q"] = enc0_W_Q.grad.norm().item()
                step_log["grad_norm/W_K"] = enc0_W_K.grad.norm().item()

            # ── Prediction confidence (Experiment 2.5) ────────────────
            if log_confidence:
                conf = get_prediction_confidence(
                    logits.reshape(B * T, V).detach(),
                    tgt_target.reshape(B * T),
                )
                step_log["train/prediction_confidence"] = conf

            wandb.log(step_log)

        if max_grad_steps and global_step >= max_grad_steps:
            break

        # ── Epoch-level metrics ───────────────────────────────────────
        avg_train_loss = total_loss / max(total_tokens, 1)

        # Validation loss
        model.eval()
        val_loss_total, val_tokens = 0.0, 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                tgt_in  = tgt[:, :-1]
                tgt_tgt = tgt[:, 1:]
                sm = make_src_mask(src, PAD_IDX)
                tm = make_tgt_mask(tgt_in, PAD_IDX)
                logits = model(src, tgt_in, sm, tm)
                B, T, V = logits.shape
                l = loss_fn(logits.reshape(B * T, V), tgt_tgt.reshape(B * T))
                n = tgt_tgt.ne(PAD_IDX).sum().item()
                val_loss_total += l.item() * n
                val_tokens     += n

        avg_val_loss = val_loss_total / max(val_tokens, 1)
        val_ppl      = math.exp(min(avg_val_loss, 20))

        epoch_log = {
            "epoch":            epoch,
            "train/epoch_loss": avg_train_loss,
            "val/loss":         avg_val_loss,
            "val/ppl":          val_ppl,
        }

        # Validation BLEU every 5 epochs
        if epoch % 5 == 0 or epoch == cfg["epochs"]:
            val_bleu = evaluate_bleu(model, val_loader, tgt_vocab,
                                     device=str(DEVICE), max_len=cfg["max_len"])
            epoch_log["val/bleu"] = val_bleu
            print(f"  [{run_name}] Epoch {epoch} | "
                  f"train_loss={avg_train_loss:.4f} | "
                  f"val_loss={avg_val_loss:.4f} | "
                  f"val_bleu={val_bleu:.2f}")

        wandb.log(epoch_log)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_{run_name}.pt")

    wandb.finish()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2.1 — Noam Scheduler vs Fixed Learning Rate
# ══════════════════════════════════════════════════════════════════════════════

def experiment_2_1(train_loader, val_loader, test_loader, src_vocab, tgt_vocab, spacy_de):
    """
    Train two models:
      Run A: Noam scheduler (warmup + inverse sqrt decay)
      Run B: Fixed LR = 1e-4

    Logs: train loss, val loss, val PPL, val BLEU, LR per step.
    In W&B report: overlay the two runs' loss curves using the Groups feature.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2.1 — Noam Scheduler vs Fixed LR")
    print("="*60)

    cfg = {**BASE_CONFIG, "epochs": 15}

    # ── Run A: Noam ───────────────────────────────────────────────────
    train_with_wandb(
        run_name      = "exp2.1_noam",
        cfg           = cfg,
        src_vocab     = src_vocab,
        tgt_vocab     = tgt_vocab,
        train_loader  = train_loader,
        val_loader    = val_loader,
        test_loader   = test_loader,
        spacy_de      = spacy_de,
        use_noam      = True,
        extra_config  = {"experiment": "2.1", "scheduler": "noam"},
    )

    # ── Run B: Fixed LR = 1e-4 ────────────────────────────────────────
    train_with_wandb(
        run_name      = "exp2.1_fixed_lr",
        cfg           = cfg,
        src_vocab     = src_vocab,
        tgt_vocab     = tgt_vocab,
        train_loader  = train_loader,
        val_loader    = val_loader,
        test_loader   = test_loader,
        spacy_de      = spacy_de,
        use_noam      = False,
        fixed_lr      = 1e-4,
        extra_config  = {"experiment": "2.1", "scheduler": "fixed_1e-4"},
    )

    # ── Standalone LR curve plot ──────────────────────────────────────
    wandb.init(project=WANDB_PROJECT, name="exp2.1_lr_curve", reinit=True)
    d_model, warmup = cfg["d_model"], cfg["warmup_steps"]
    steps = list(range(1, 6001))
    lrs   = [(d_model**-0.5) * min(s**-0.5, s*(warmup**-1.5)) for s in steps]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, lrs, label="Noam LR")
    ax.axhline(1e-4, color="orange", linestyle="--", label="Fixed LR = 1e-4")
    ax.axvline(warmup, color="red",    linestyle="--", label=f"warmup={warmup}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Noam vs Fixed LR Schedule")
    ax.legend()
    plt.tight_layout()
    wandb.log({"lr_schedule_comparison": wandb.Image(fig)})
    plt.close(fig)
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2.2 — Scaling Factor 1/sqrt(dk)
# ══════════════════════════════════════════════════════════════════════════════

class _UnscaledMHA(nn.Module):
    """
    Multi-Head Attention WITHOUT the 1/sqrt(dk) scaling factor.
    Identical to MultiHeadAttention but removes the division.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        B, seq, _ = x.size()
        return x.view(B, seq, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        Q = self._split(self.W_Q(query))
        K = self._split(self.W_K(key))
        V = self._split(self.W_V(value))

        # NO sqrt(d_k) scaling
        scores = torch.matmul(Q, K.transpose(-2, -1))   # ← no / sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn_w = F.softmax(scores, dim=-1)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        x = torch.matmul(attn_w, V)
        B, _, seq_q, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(B, seq_q, self.d_model)
        return self.W_O(x)


def _swap_mha_in_model(model, use_scale=True):
    """Replace all MultiHeadAttention in encoder/decoder with scaled or unscaled."""
    if use_scale:
        return   # default model already uses scaled attention
    cfg_d   = model.encoder.layers[0].self_attn.d_model
    cfg_h   = model.encoder.layers[0].self_attn.num_heads
    cfg_do  = model.encoder.layers[0].dropout.p
    for layer in model.encoder.layers:
        layer.self_attn = _UnscaledMHA(cfg_d, cfg_h, cfg_do).to(DEVICE)
    for layer in model.decoder.layers:
        layer.self_attn  = _UnscaledMHA(cfg_d, cfg_h, cfg_do).to(DEVICE)
        layer.cross_attn = _UnscaledMHA(cfg_d, cfg_h, cfg_do).to(DEVICE)


def experiment_2_2(train_loader, val_loader, test_loader, src_vocab, tgt_vocab, spacy_de):
    """
    Train for 1000 steps each:
      Run A: with    1/sqrt(dk) scaling
      Run B: without 1/sqrt(dk) scaling

    Logs: W_Q and W_K gradient norms every step.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2.2 — Scaling Factor 1/sqrt(dk)")
    print("="*60)

    cfg = {**BASE_CONFIG, "epochs": 999}   # will be cut by max_grad_steps

    for use_scale in [True, False]:
        label    = "with_scale" if use_scale else "no_scale"
        run_name = f"exp2.2_{label}"
        print(f"\n  Running: {run_name}")

        wandb.init(project=WANDB_PROJECT, name=run_name, reinit=True,
                   config={**cfg, "experiment": "2.2", "use_scale": use_scale})

        model     = build_model(len(src_vocab), len(tgt_vocab), cfg)
        _swap_mha_in_model(model, use_scale=use_scale)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                     betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, cfg["d_model"], cfg["warmup_steps"])
        loss_fn   = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, 0.1)

        # References to Q/K weights in first encoder layer
        W_Q_ref = model.encoder.layers[0].self_attn.W_Q.weight
        W_K_ref = model.encoder.layers[0].self_attn.W_K.weight

        model.train()
        step = 0
        done = False

        for src, tgt in tqdm(train_loader, desc=run_name):
            if done:
                break
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in  = tgt[:, :-1]
            tgt_tgt = tgt[:, 1:]
            sm = make_src_mask(src, PAD_IDX)
            tm = make_tgt_mask(tgt_in, PAD_IDX)

            logits = model(src, tgt_in, sm, tm)
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B*T, V), tgt_tgt.reshape(B*T))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            log = {"step": step, "loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
            if W_Q_ref.grad is not None:
                log["grad_norm/W_Q"] = W_Q_ref.grad.norm().item()
                log["grad_norm/W_K"] = W_K_ref.grad.norm().item()
            wandb.log(log)

            if step >= 1000:
                done = True

        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2.3 — Attention Rollout & Head Specialisation
# ══════════════════════════════════════════════════════════════════════════════

def _get_encoder_attention_weights(model, sentence, src_vocab, device):
    """
    Forward pass a single sentence and return per-layer encoder attention.
    Returns:
        all_attn  : list of (n_heads, seq, seq) numpy arrays, one per layer
        tokens    : list of token strings
    """
    model.eval()
    tokens  = [tok.text.lower() for tok in model.spacy_de.tokenizer(sentence)]
    tokens  = tokens[:BASE_CONFIG["max_len"] - 2]
    ids     = ([SOS_IDX]
               + [src_vocab.stoi.get(t, 3) for t in tokens]
               + [EOS_IDX])
    src     = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = make_src_mask(src, PAD_IDX)

    display_tokens = ["<sos>"] + tokens + ["<eos>"]

    # Monkey-patch encoder layers to capture attention weights
    attn_store = []

    def make_hook(layer_idx):
        orig_forward = model.encoder.layers[layer_idx].forward
        def hooked_forward(x, mask):
            _x  = model.encoder.layers[layer_idx].norm1(x)
            out, attn_w = _hooked_mha(
                model.encoder.layers[layer_idx].self_attn, _x, _x, _x, mask
            )
            attn_store.append(attn_w[0].detach().cpu().numpy())   # (heads,seq,seq)
            x = x + model.encoder.layers[layer_idx].dropout(out)
            x = x + model.encoder.layers[layer_idx].dropout(
                model.encoder.layers[layer_idx].ff(
                    model.encoder.layers[layer_idx].norm2(x)
                )
            )
            return x
        return hooked_forward

    def _hooked_mha(mha, query, key, value, mask):
        """Call MHA internals and also return attn_weights."""
        d_model   = mha.d_model
        num_heads = mha.num_heads
        d_k       = mha.d_k

        def split(x):
            B, seq, _ = x.size()
            return x.view(B, seq, num_heads, d_k).transpose(1, 2)

        Q = split(mha.W_Q(query))
        K = split(mha.W_K(key))
        V = split(mha.W_V(value))
        x, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        B, _, seq_q, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(B, seq_q, d_model)
        return mha.W_O(x), attn_w

    # Temporarily patch each layer's forward
    original_forwards = []
    for i in range(len(model.encoder.layers)):
        original_forwards.append(model.encoder.layers[i].forward)
        model.encoder.layers[i].forward = make_hook(i)

    with torch.no_grad():
        src_emb = model.src_pe(model.src_embed(src) * math.sqrt(model.d_model))
        model.encoder(src_emb, src_mask)

    # Restore original forwards
    for i, orig in enumerate(original_forwards):
        model.encoder.layers[i].forward = orig

    return attn_store, display_tokens


def experiment_2_3(src_vocab, tgt_vocab):
    """
    Load the best trained model, extract last-encoder-layer attention weights
    for sample German sentences, and log:
      - Per-head heatmaps for the last encoder layer
      - Attention rollout heatmap
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2.3 — Attention Rollout & Head Specialisation")
    print("="*60)

    wandb.init(project=WANDB_PROJECT, name="exp2.3_attention_rollout",
               config={"experiment": "2.3"}, reinit=True)

    # Load best model weights
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = BASE_CONFIG["d_model"],
        N              = BASE_CONFIG["N"],
        num_heads      = BASE_CONFIG["num_heads"],
        d_ff           = BASE_CONFIG["d_ff"],
        dropout        = 0.0,   # no dropout during viz
        max_len        = BASE_CONFIG["max_len"],
        checkpoint_path= None,
    ).to(DEVICE)
    model.src_vocab = src_vocab
    model.trg_vocab = tgt_vocab

    # Try loading best checkpoint from exp 2.1 noam run
    ckpt = "best_exp2.1_noam.pt"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False),
                              strict=False)
        print(f"  Loaded weights from {ckpt}")
    else:
        print("  WARNING: no checkpoint found, using random weights for attention viz.")

    sentences = [
        "Ein Mann schaut aus dem Fenster.",
        "Zwei Kinder spielen im Park.",
        "Eine Frau liest ein Buch auf der Bank.",
    ]

    for sentence in sentences:
        all_attn, tokens = _get_encoder_attention_weights(model, sentence, src_vocab, DEVICE)
        seq_len = len(tokens)

        # ── Per-head heatmaps for LAST encoder layer ───────────────────
        last_attn = all_attn[-1][:, :seq_len, :seq_len]   # (heads, seq, seq)
        n_heads   = last_attn.shape[0]

        fig, axes = plt.subplots(2, n_heads // 2, figsize=(4 * n_heads // 2, 8))
        axes      = axes.flatten()

        for h in range(n_heads):
            ax = axes[h]
            im = ax.imshow(last_attn[h], cmap="Blues", aspect="auto",
                           vmin=0, vmax=last_attn[h].max() + 1e-9)
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=90, fontsize=7)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(tokens, fontsize=7)
            ax.set_title(f"Head {h+1}", fontsize=9)
            plt.colorbar(im, ax=ax)

        plt.suptitle(f"Last Encoder Layer — All Heads\n'{sentence[:40]}'", fontsize=11)
        plt.tight_layout()
        key = sentence[:20].replace(" ", "_")
        wandb.log({f"attention/per_head/{key}": wandb.Image(fig)})
        plt.close(fig)

        # ── Attention Rollout ──────────────────────────────────────────
        rollout = np.eye(seq_len)
        for layer_attn in all_attn:
            a    = layer_attn[:, :seq_len, :seq_len]   # (heads, seq, seq)
            a    = a.mean(axis=0)                      # mean over heads
            a    = a + np.eye(seq_len)                 # add residual
            a    = a / (a.sum(axis=-1, keepdims=True) + 1e-9)
            rollout = rollout @ a

        fig2, ax2 = plt.subplots(figsize=(8, 7))
        im2 = ax2.imshow(rollout, cmap="Blues", aspect="auto")
        ax2.set_xticks(range(seq_len))
        ax2.set_xticklabels(tokens, rotation=90, fontsize=8)
        ax2.set_yticks(range(seq_len))
        ax2.set_yticklabels(tokens, fontsize=8)
        ax2.set_title(f"Attention Rollout\n'{sentence[:40]}'")
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        wandb.log({f"attention/rollout/{key}": wandb.Image(fig2)})
        plt.close(fig2)

        # ── Log last-layer mean attention as a table ───────────────────
        mean_attn = last_attn.mean(axis=0)   # (seq, seq)
        table     = wandb.Table(
            columns = tokens,
            data    = [[float(mean_attn[i, j]) for j in range(seq_len)]
                       for i in range(seq_len)],
        )
        wandb.log({f"attention/mean_table/{key}": table})

    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2.4 — Sinusoidal vs Learned Positional Encoding
# ══════════════════════════════════════════════════════════════════════════════

def experiment_2_4(train_loader, val_loader, test_loader, src_vocab, tgt_vocab, spacy_de):
    """
    Train two models:
      Run A: sinusoidal PE (fixed, non-trainable buffer)
      Run B: learned PE   (nn.Embedding, trainable)

    Logs: val BLEU, val loss per epoch for comparison.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2.4 — Sinusoidal vs Learned Positional Encoding")
    print("="*60)

    cfg = {**BASE_CONFIG, "epochs": 15}

    for pe_type in ["sinusoidal", "learned"]:
        train_with_wandb(
            run_name      = f"exp2.4_{pe_type}_pe",
            cfg           = cfg,
            src_vocab     = src_vocab,
            tgt_vocab     = tgt_vocab,
            train_loader  = train_loader,
            val_loader    = val_loader,
            test_loader   = test_loader,
            spacy_de      = spacy_de,
            use_noam      = True,
            pe_type       = pe_type,
            extra_config  = {"experiment": "2.4", "pe_type": pe_type},
        )


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2.5 — Label Smoothing: eps=0.1 vs eps=0.0
# ══════════════════════════════════════════════════════════════════════════════

def experiment_2_5(train_loader, val_loader, test_loader, src_vocab, tgt_vocab, spacy_de):
    """
    Train two models:
      Run A: label smoothing eps = 0.1
      Run B: no smoothing   eps = 0.0  (standard cross-entropy)

    Logs: train loss, val loss, prediction_confidence per step.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2.5 — Label Smoothing eps=0.1 vs eps=0.0")
    print("="*60)

    cfg = {**BASE_CONFIG, "epochs": 15}

    for eps in [0.1, 0.0]:
        label = "smoothed" if eps > 0 else "no_smoothing"
        train_with_wandb(
            run_name        = f"exp2.5_label_{label}",
            cfg             = cfg,
            src_vocab       = src_vocab,
            tgt_vocab       = tgt_vocab,
            train_loader    = train_loader,
            val_loader      = val_loader,
            test_loader     = test_loader,
            spacy_de        = spacy_de,
            use_noam        = True,
            label_smoothing = eps,
            log_confidence  = True,   # ← logs softmax prob of correct token
            extra_config    = {"experiment": "2.5", "label_smoothing": eps},
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", type=str, default="all",
        choices=["all", "2.1", "2.2", "2.3", "2.4", "2.5"],
        help="Which experiment to run",
    )
    args = parser.parse_args()

    # ── Load data once, share across experiments ───────────────────────
    print("Loading data …")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size = BASE_CONFIG["batch_size"],
        min_freq   = 2,
        max_len    = BASE_CONFIG["max_len"] - 2,
    )

    # Save vocab for Transformer.__init__ to find
    import torch as _t
    _t.save({"src_vocab": src_vocab, "trg_vocab": tgt_vocab}, "vocab.pt")

    # spacy_de needed for Experiment 2.3
    import spacy
    spacy_de = spacy.load("de_core_news_sm")

    print(f"src_vocab={len(src_vocab):,}  tgt_vocab={len(tgt_vocab):,}")
    print(f"Device: {DEVICE}\n")

    if args.exp in ("all", "2.1"):
        experiment_2_1(train_loader, val_loader, test_loader,
                       src_vocab, tgt_vocab, spacy_de)

    if args.exp in ("all", "2.2"):
        experiment_2_2(train_loader, val_loader, test_loader,
                       src_vocab, tgt_vocab, spacy_de)

    if args.exp in ("all", "2.3"):
        experiment_2_3(src_vocab, tgt_vocab)

    if args.exp in ("all", "2.4"):
        experiment_2_4(train_loader, val_loader, test_loader,
                       src_vocab, tgt_vocab, spacy_de)

    if args.exp in ("all", "2.5"):
        experiment_2_5(train_loader, val_loader, test_loader,
                       src_vocab, tgt_vocab, spacy_de)

    print("\nAll experiments complete.")



if __name__ == "__main__":
    main()