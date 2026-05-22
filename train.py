"""
train.py - Main training pipeline for DA6401 Assignment 3

Trains the Transformer on Multi30k (DE→EN) with:
  • Noam LR scheduler
  • Label smoothing (eps=0.1)
  • W&B logging (loss, lr, BLEU, attention maps, prediction confidence)
  • Best checkpoint saving
"""

import os
import math
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from config import cfg
from dataset import get_data
from model import Transformer
from scheduler import NoamScheduler
from loss import LabelSmoothingLoss
from utils import (
    evaluate_bleu_loader,
    plot_attention_heads,
    get_prediction_confidence,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: one training epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, scheduler, device, epoch, log_interval=50):
    model.train()
    total_loss   = 0.0
    total_tokens = 0
    conf_values  = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for step, (src, trg) in enumerate(pbar):
        src = src.to(device)
        trg = trg.to(device)

        # Teacher forcing: feed trg[:-1] as input, predict trg[1:] as target
        trg_input  = trg[:, :-1]   # (B, T-1) — drop <eos>
        trg_target = trg[:, 1:]    # (B, T-1) — drop <sos>

        scheduler.zero_grad()

        logits, enc_attn, dec_attn = model(src, trg_input)
        # logits: (B, T-1, vocab)

        # Flatten for loss
        B, T, V = logits.shape
        loss = criterion(
            logits.reshape(B * T, V),
            trg_target.reshape(B * T),
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP)
        scheduler.step()   # updates lr AND calls optimizer.step()

        n_tokens = trg_target.ne(cfg.PAD_IDX).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

        # Prediction confidence (for W&B experiment 2.5)
        with torch.no_grad():
            conf = get_prediction_confidence(logits.reshape(B * T, V))
            conf_values.append(conf)

        if (step + 1) % log_interval == 0:
            avg_loss = total_loss / max(total_tokens, 1)
            wandb.log({
                "train/loss":       avg_loss,
                "train/ppl":        math.exp(min(avg_loss, 20)),
                "train/lr":         scheduler.current_lr,
                "train/step":       scheduler.current_step,
                "train/confidence": sum(conf_values) / len(conf_values),
            })
            conf_values = []

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.current_lr:.6f}")

    return total_loss / max(total_tokens, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: one validation epoch
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for src, trg in tqdm(loader, desc="Validation", leave=False):
        src = src.to(device)
        trg = trg.to(device)

        trg_input  = trg[:, :-1]
        trg_target = trg[:, 1:]

        logits, _, _ = model(src, trg_input)
        B, T, V = logits.shape

        loss = criterion(
            logits.reshape(B * T, V),
            trg_target.reshape(B * T),
        )
        n_tokens = trg_target.ne(cfg.PAD_IDX).sum().item()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Attention map logging (last encoder layer, first sample in val batch)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def log_attention_maps(model, val_loader, src_vocab, trg_vocab, device):
    model.eval()
    src, trg = next(iter(val_loader))
    src = src[:1].to(device)   # single sample
    trg = trg[:1].to(device)

    trg_input = trg[:, :-1]
    _, enc_attn, _ = model(src, trg_input)

    # enc_attn is a list of (batch, heads, seq_q, seq_k) tensors, one per layer
    last_layer_attn = enc_attn[-1][0].cpu().numpy()   # (heads, seq_q, seq_k)

    src_ids   = src[0].tolist()
    src_tokens = [src_vocab.itos[i] for i in src_ids
                  if i not in (cfg.PAD_IDX,)]

    images = plot_attention_heads(
        last_layer_attn[:, :len(src_tokens), :len(src_tokens)],
        src_tokens, src_tokens,
        layer_name="encoder_last",
    )
    wandb.log({"attention/encoder_last_layer": images})


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(
    run_name        = "transformer_baseline",
    label_smooth    = cfg.LABEL_SMOOTH,
    warmup_steps    = cfg.WARMUP_STEPS,
    epochs          = cfg.EPOCHS,
    pos_encoding    = "sinusoidal",
    use_noam        = True,
    fixed_lr        = 1e-4,
    log_attn_every  = 5,      # log attention maps every N epochs
    log_bleu_every  = 2,      # compute val BLEU every N epochs
    extra_config    = None,   # dict of extra keys to log to W&B
):
    device = cfg.DEVICE
    print(f"Using device: {device}")

    # ── W&B init ──────────────────────────────────────────────────────
    wandb_cfg = dict(
        d_model       = cfg.D_MODEL,
        n_heads       = cfg.N_HEADS,
        n_encoder     = cfg.N_ENCODER,
        n_decoder     = cfg.N_DECODER,
        d_ff          = cfg.D_FF,
        dropout       = cfg.DROPOUT,
        batch_size    = cfg.BATCH_SIZE,
        epochs        = epochs,
        warmup_steps  = warmup_steps,
        label_smooth  = label_smooth,
        pos_encoding  = pos_encoding,
        use_noam      = use_noam,
    )
    if extra_config:
        wandb_cfg.update(extra_config)

    wandb.init(
        project = cfg.WANDB_PROJECT,
        entity  = cfg.WANDB_ENTITY,
        name    = run_name,
        config  = wandb_cfg,
        reinit  = True,
    )

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading data …")
    train_loader, val_loader, test_loader, src_vocab, trg_vocab, spacy_de, spacy_en \
        = get_data()

    # ── Model (load_weights=False so we start fresh) ─────────────────
    print("Building model …")
    model = Transformer(
        d_model      = cfg.D_MODEL,
        n_heads      = cfg.N_HEADS,
        n_encoder    = cfg.N_ENCODER,
        n_decoder    = cfg.N_DECODER,
        d_ff         = cfg.D_FF,
        dropout      = cfg.DROPOUT,
        max_seq_len  = cfg.MAX_SEQ_LEN,
        pos_encoding = pos_encoding,
        load_weights = False,
    ).to(device)

    # Sync vocabs so the data-loaded vocabs are stored in the model
    model.src_vocab = src_vocab
    model.trg_vocab = trg_vocab

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    # ── Loss ──────────────────────────────────────────────────────────
    criterion = LabelSmoothingLoss(
        vocab_size = len(trg_vocab),
        pad_idx    = cfg.PAD_IDX,
        eps        = label_smooth,
    )

    # ── Optimiser + Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr    = 1.0,          # actual lr is controlled by scheduler
        betas = (0.9, 0.98),
        eps   = 1e-9,
    )

    if use_noam:
        scheduler = NoamScheduler(optimizer, cfg.D_MODEL, warmup_steps)
    else:
        # Fixed LR: set lr directly, wrap in a dummy object that mimics NoamScheduler
        for pg in optimizer.param_groups:
            pg["lr"] = fixed_lr
        scheduler = _FixedLRScheduler(optimizer, fixed_lr)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_bleu     = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\n── Epoch {epoch}/{epochs} ──")

        train_loss = train_epoch(model, train_loader, criterion, scheduler,
                                 device, epoch)
        val_loss   = val_epoch(model, val_loader, criterion, device)
        val_ppl    = math.exp(min(val_loss, 20))

        log_dict = {
            "epoch":          epoch,
            "train/epoch_loss": train_loss,
            "val/loss":       val_loss,
            "val/ppl":        val_ppl,
        }

        # BLEU (expensive, do every log_bleu_every epochs)
        if epoch % log_bleu_every == 0 or epoch == epochs:
            val_bleu = evaluate_bleu_loader(
                model, val_loader, src_vocab, trg_vocab, spacy_de, device,
                max_samples=300,
            )
            log_dict["val/bleu"] = val_bleu
            print(f"  Val BLEU: {val_bleu:.2f}")
            if val_bleu > best_bleu:
                best_bleu = val_bleu

        # Attention maps
        if epoch % log_attn_every == 0 or epoch == epochs:
            log_attention_maps(model, val_loader, src_vocab, trg_vocab, device)

        wandb.log(log_dict)
        print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.MODEL_SAVE)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    # ── Final test BLEU ───────────────────────────────────────────────
    print("\nEvaluating on test set …")
    model.load_state_dict(torch.load(cfg.MODEL_SAVE, map_location=device))
    test_bleu = evaluate_bleu_loader(
        model, test_loader, src_vocab, trg_vocab, spacy_de, device,
        max_samples=1000,
    )
    wandb.log({"test/bleu": test_bleu, "best_val_bleu": best_bleu})
    print(f"Test BLEU: {test_bleu:.2f}")

    wandb.finish()
    return model, src_vocab, trg_vocab


# ──────────────────────────────────────────────────────────────────────────────
# Dummy fixed-LR scheduler (mirrors NoamScheduler interface)
# ──────────────────────────────────────────────────────────────────────────────

class _FixedLRScheduler:
    def __init__(self, optimizer, lr):
        self.optimizer   = optimizer
        self._lr         = lr
        self._step       = 0

    def step(self):
        self._step += 1
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def current_lr(self):
        return self._lr

    @property
    def current_step(self):
        return self._step


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Save vocab after first data load so Transformer.__init__ can find it
    import torch as _torch
    from dataset import get_data as _get_data

    print("Pre-building vocab cache …")
    _, _, _, sv, tv, _, _ = _get_data()
    _torch.save({"src_vocab": sv, "trg_vocab": tv}, cfg.VOCAB_FILENAME)
    print(f"Vocab saved to {cfg.VOCAB_FILENAME}")

    train(run_name="transformer_baseline")