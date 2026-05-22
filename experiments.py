"""
experiments.py - All 5 W&B ablation experiments for DA6401 Assignment 3

Experiment 2.1 – Noam vs Fixed LR
Experiment 2.2 – Scaling factor 1/sqrt(dk) vs no scaling
Experiment 2.3 – Attention rollout & head specialisation
Experiment 2.4 – Sinusoidal vs Learned positional encoding
Experiment 2.5 – Label smoothing eps=0.1 vs eps=0.0
"""

import math
import torch
import torch.nn as nn
import wandb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from train import train, _FixedLRScheduler


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2.1 – Noam Scheduler vs Fixed Learning Rate
# ══════════════════════════════════════════════════════════════════════════════

def experiment_noam_vs_fixed():
    """
    Train two models side-by-side and overlay their loss/accuracy curves in W&B.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2.1 – Noam Scheduler vs Fixed LR")
    print("="*60)

    # Run 1: Noam
    train(
        run_name       = "exp2.1_noam_scheduler",
        use_noam       = True,
        warmup_steps   = cfg.WARMUP_STEPS,
        epochs         = cfg.EPOCHS,
        extra_config   = {"experiment": "2.1_noam_vs_fixed", "scheduler": "noam"},
    )

    # Run 2: Fixed LR = 1e-4
    train(
        run_name       = "exp2.1_fixed_lr_1e-4",
        use_noam       = False,
        fixed_lr       = 1e-4,
        epochs         = cfg.EPOCHS,
        extra_config   = {"experiment": "2.1_noam_vs_fixed", "scheduler": "fixed_1e-4"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2.2 – Scaling Factor 1/sqrt(dk)
# ══════════════════════════════════════════════════════════════════════════════

def _make_model_no_scale(src_vocab_size, trg_vocab_size):
    """
    Monkey-patch ScaledDotProductAttention to remove the sqrt(d_k) scaling,
    then build a normal Transformer.
    """
    import attention as attn_module

    # Save original forward
    OrigForward = attn_module.ScaledDotProductAttention.forward

    def unscaled_forward(self, Q, K, V, mask=None):
        # No division by sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        import torch.nn.functional as F
        attn_w = F.softmax(scores, dim=-1)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        attn_w = self.dropout(attn_w)
        output = torch.matmul(attn_w, V)
        return output, attn_w

    attn_module.ScaledDotProductAttention.forward = unscaled_forward
    return OrigForward   # return so we can restore


def experiment_scaling_factor():
    """
    Train with and without 1/sqrt(dk) scaling.
    Log gradient norms of Q, K weight matrices during first 1000 steps.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2.2 – Scaling Factor 1/sqrt(dk)")
    print("="*60)

    device = cfg.DEVICE

    train_loader, val_loader, _, src_vocab, trg_vocab, spacy_de, _ = get_data()

    for use_scale in [True, False]:
        run_name = "exp2.2_with_scale" if use_scale else "exp2.2_no_scale"
        print(f"\n  Running: {run_name}")

        import attention as attn_module
        from attention import ScaledDotProductAttention
        import torch.nn.functional as F

        # Patch if needed
        if not use_scale:
            def unscaled_forward(self, Q, K, V, mask=None):
                scores = torch.matmul(Q, K.transpose(-2, -1))
                if mask is not None:
                    scores = scores.masked_fill(mask, float("-inf"))
                w = F.softmax(scores, dim=-1)
                w = torch.nan_to_num(w, nan=0.0)
                w = self.dropout(w)
                return torch.matmul(w, V), w
            ScaledDotProductAttention.forward = unscaled_forward
        else:
            # Restore original (reload module)
            import importlib, attention as _a
            importlib.reload(_a)
            from attention import ScaledDotProductAttention as SDP
            attn_module.ScaledDotProductAttention = SDP

        model = Transformer(
            load_weights = False,
        ).to(device)
        model.src_vocab = src_vocab
        model.trg_vocab = trg_vocab

        criterion = LabelSmoothingLoss(len(trg_vocab), cfg.PAD_IDX, cfg.LABEL_SMOOTH)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                     betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, cfg.D_MODEL, cfg.WARMUP_STEPS)

        wandb.init(
            project = cfg.WANDB_PROJECT,
            entity  = cfg.WANDB_ENTITY,
            name    = run_name,
            config  = {"experiment": "2.2_scaling", "use_scale": use_scale},
            reinit  = True,
        )

        # Collect Q and K weight references from first encoder layer
        enc0     = model.encoder.layers[0].self_attn
        W_Q_ref  = enc0.W_Q.weight
        W_K_ref  = enc0.W_K.weight

        model.train()
        global_step = 0
        MAX_STEPS   = 1000

        for src, trg in tqdm(train_loader, desc=run_name):
            if global_step >= MAX_STEPS:
                break
            src, trg = src.to(device), trg.to(device)
            trg_in  = trg[:, :-1]
            trg_tgt = trg[:, 1:]

            scheduler.zero_grad()
            logits, _, _ = model(src, trg_in)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B*T, V), trg_tgt.reshape(B*T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP)
            scheduler.step()

            global_step += 1
            if global_step % 10 == 0:
                q_grad_norm = W_Q_ref.grad.norm().item() if W_Q_ref.grad is not None else 0
                k_grad_norm = W_K_ref.grad.norm().item() if W_K_ref.grad is not None else 0
                wandb.log({
                    "step":            global_step,
                    "loss":            loss.item(),
                    "grad_norm/W_Q":   q_grad_norm,
                    "grad_norm/W_K":   k_grad_norm,
                    "lr":              scheduler.current_lr,
                })

        wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2.3 – Attention Rollout & Head Specialisation
# ══════════════════════════════════════════════════════════════════════════════

def experiment_attention_rollout():
    """
    Load the best trained model, extract last-encoder-layer attention weights
    for a fixed German sentence, and log per-head heatmaps to W&B.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2.3 – Attention Rollout & Head Specialisation")
    print("="*60)

    device = cfg.DEVICE

    wandb.init(
        project = cfg.WANDB_PROJECT,
        entity  = cfg.WANDB_ENTITY,
        name    = "exp2.3_attention_rollout",
        config  = {"experiment": "2.3_attention_rollout"},
        reinit  = True,
    )

    # Load model with saved weights
    model = Transformer(load_weights=True).to(device)
    model.eval()

    # Sample sentence for visualisation
    sentences = [
        "Ein Mann schaut aus dem Fenster.",
        "Zwei Kinder spielen im Park.",
        "Eine Frau liest ein Buch.",
    ]

    for sentence in sentences:
        tokens = [tok.text.lower() for tok in model.spacy_de.tokenizer(sentence)]
        tokens = tokens[: cfg.MAX_SEQ_LEN - 2]

        src_ids = (
            [cfg.SOS_IDX]
            + [model.src_vocab.stoi.get(t, cfg.UNK_IDX) for t in tokens]
            + [cfg.EOS_IDX]
        )
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = model.make_src_mask(src)

        with torch.no_grad():
            src_emb = model.src_pe(model.src_embed(src) * math.sqrt(model.d_model))
            _, enc_attn_all = model.encoder(src_emb, src_mask)

        # Last encoder layer: shape (1, n_heads, seq, seq)
        last_attn = enc_attn_all[-1][0].cpu().numpy()   # (n_heads, seq, seq)

        display_tokens = ["<sos>"] + tokens + ["<eos>"]
        seq_len = len(display_tokens)
        last_attn_trimmed = last_attn[:, :seq_len, :seq_len]

        images = plot_attention_heads(
            last_attn_trimmed,
            display_tokens,
            display_tokens,
            layer_name=f"encoder_last | '{sentence[:30]}'",
        )
        wandb.log({f"attention/{sentence[:20]}": images})

        # ── Attention Rollout ──────────────────────────────────────────
        # Multiply attention matrices across all encoder layers (mean over heads)
        rollout = np.eye(seq_len)
        for layer_attn in enc_attn_all:
            a = layer_attn[0].cpu().numpy()           # (heads, seq, seq)
            a_mean = a.mean(axis=0)[:seq_len, :seq_len]
            # Add residual
            a_aug = a_mean + np.eye(seq_len)
            a_aug = a_aug / a_aug.sum(axis=-1, keepdims=True)
            rollout = np.matmul(rollout, a_aug)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rollout, cmap="Blues")
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(display_tokens, rotation=90, fontsize=8)
        ax.set_yticks(range(seq_len))
        ax.set_yticklabels(display_tokens, fontsize=8)
        ax.set_title(f"Attention Rollout | '{sentence[:30]}'")
        plt.tight_layout()
        wandb.log({f"rollout/{sentence[:20]}": wandb.Image(fig)})
        plt.close(fig)

    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2.4 – Sinusoidal vs Learned Positional Encoding
# ══════════════════════════════════════════════════════════════════════════════

def experiment_positional_encoding():
    print("\n" + "="*60)
    print("EXPERIMENT 2.4 – Sinusoidal vs Learned Positional Encoding")
    print("="*60)

    # Run 1: sinusoidal (reuse baseline if already trained)
    train(
        run_name     = "exp2.4_sinusoidal_pe",
        pos_encoding = "sinusoidal",
        epochs       = cfg.EPOCHS,
        extra_config = {"experiment": "2.4_positional", "pe_type": "sinusoidal"},
    )

    # Run 2: learned embeddings
    train(
        run_name     = "exp2.4_learned_pe",
        pos_encoding = "learned",
        epochs       = cfg.EPOCHS,
        extra_config = {"experiment": "2.4_positional", "pe_type": "learned"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2.5 – Label Smoothing eps=0.1 vs eps=0.0
# ══════════════════════════════════════════════════════════════════════════════

def experiment_label_smoothing():
    print("\n" + "="*60)
    print("EXPERIMENT 2.5 – Label Smoothing eps=0.1 vs eps=0.0")
    print("="*60)

    for eps in [0.1, 0.0]:
        label = "smoothed" if eps > 0 else "no_smoothing"
        train(
            run_name      = f"exp2.5_label_smooth_{label}",
            label_smooth  = eps,
            epochs        = cfg.EPOCHS,
            extra_config  = {"experiment": "2.5_label_smoothing", "eps": eps},
        )


# ══════════════════════════════════════════════════════════════════════════════
# Noam LR curve visualisation (standalone W&B plot, no training needed)
# ══════════════════════════════════════════════════════════════════════════════

def log_noam_lr_curve():
    """
    Log the theoretical Noam LR curve to W&B (used in experiment 2.1 report).
    """
    wandb.init(
        project = cfg.WANDB_PROJECT,
        entity  = cfg.WANDB_ENTITY,
        name    = "noam_lr_curve",
        reinit  = True,
    )
    d_model      = cfg.D_MODEL
    warmup_steps = cfg.WARMUP_STEPS
    steps        = list(range(1, 8001))
    lrs = [
        (d_model ** -0.5) * min(s ** -0.5, s * (warmup_steps ** -1.5))
        for s in steps
    ]
    for s, lr in zip(steps, lrs):
        wandb.log({"noam_lr": lr, "step": s})

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, lrs)
    ax.axvline(warmup_steps, color="red", linestyle="--", label=f"warmup={warmup_steps}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"Noam LR Schedule (d_model={d_model}, warmup={warmup_steps})")
    ax.legend()
    plt.tight_layout()
    wandb.log({"noam_lr_curve": wandb.Image(fig)})
    plt.close(fig)
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run W&B ablation experiments")
    parser.add_argument(
        "--exp", type=str, default="all",
        choices=["all", "2.1", "2.2", "2.3", "2.4", "2.5", "lr_curve"],
        help="Which experiment to run",
    )
    args = parser.parse_args()

    if args.exp in ("all", "lr_curve"):
        log_noam_lr_curve()

    if args.exp in ("all", "2.1"):
        experiment_noam_vs_fixed()

    if args.exp in ("all", "2.2"):
        experiment_scaling_factor()

    if args.exp in ("all", "2.3"):
        experiment_attention_rollout()

    if args.exp in ("all", "2.4"):
        experiment_positional_encoding()

    if args.exp in ("all", "2.5"):
        experiment_label_smoothing()

    print("\nAll experiments complete.")