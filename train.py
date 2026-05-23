"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.criterion  = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value (normalised by number of non-pad tokens).
        """
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            # Start with uniform smoothing
            smooth = torch.full_like(log_probs,
                                     self.smoothing / (self.vocab_size - 2))
            # Zero out pad index
            smooth[:, self.pad_idx] = 0.0
            # Assign 1 - smoothing to the correct token
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            # Zero out rows corresponding to pad targets
            pad_mask = target.eq(self.pad_idx)
            smooth[pad_mask] = 0.0

        loss     = self.criterion(log_probs, smooth)
        n_tokens = (~pad_mask).sum().float()
        return loss / n_tokens.clamp(min=1)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        pbar = tqdm(data_iter,
                    desc=f"Epoch {epoch_num} [{'train' if is_train else 'val'}]",
                    leave=False)

        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing:
            #   input  = tgt[:, :-1]  (drop <eos>)
            #   target = tgt[:, 1:]   (drop <sos>)
            tgt_input  = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            src_mask = make_src_mask(src, PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_input, PAD_IDX)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits: (B, T-1, vocab)

            B, T, V = logits.shape
            loss = loss_fn(
                logits.reshape(B * T, V),
                tgt_target.reshape(B * T),
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n_tokens      = tgt_target.ne(PAD_IDX).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and excludes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = make_tgt_mask(ys, PAD_IDX)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(-1).item()
            if next_tok == end_symbol:
                break
            ys = torch.cat(
                [ys, torch.tensor([[next_tok]], dtype=torch.long, device=device)],
                dim=1,
            )

    return ys   # shape [1, out_len]


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with itos list.
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    from evaluate import load as load_metric
    bleu_metric = load_metric("bleu")

    model.eval()
    predictions = []
    references  = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)                     # (1, src_len)
                src_mask = make_src_mask(src_i, PAD_IDX)

                ys = greedy_decode(
                    model, src_i, src_mask, max_len,
                    SOS_IDX, EOS_IDX, device,
                )
                pred_ids = ys[0, 1:].tolist()   # strip <sos>

                pred = " ".join(
                    tgt_vocab.itos[idx]
                    for idx in pred_ids
                    if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)
                )

                ref_ids = tgt[i].tolist()
                ref = " ".join(
                    tgt_vocab.itos[idx]
                    for idx in ref_ids
                    if idx not in (SOS_IDX, EOS_IDX, PAD_IDX)
                )

                predictions.append(pred)
                references.append([ref])

    result = bleu_metric.compute(predictions=predictions, references=references)
    return result["bleu"] * 100


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": {
                "src_vocab_size": len(model.src_embed.weight),
                "tgt_vocab_size": len(model.fc_out.weight),
                "d_model":        model.d_model,
                "N":              len(model.encoder.layers),
                "num_heads":      model.encoder.layers[0].self_attn.num_heads,
                "d_ff":           model.encoder.layers[0].ff.linear1.out_features,
                "dropout":        model.encoder.layers[0].dropout.p,
            },
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("epoch", 0)


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment with W&B logging.
    """
    # ── Hyperparameters ───────────────────────────────────────────────
    D_MODEL      = 256
    N_LAYERS     = 3
    N_HEADS      = 8
    D_FF         = 512
    DROPOUT      = 0.1
    BATCH_SIZE   = 128
    EPOCHS       = 20
    WARMUP_STEPS = 400
    SMOOTHING    = 0.1
    MAX_LEN      = 100
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    # ── 1. Init W&B ───────────────────────────────────────────────────
    wandb.init(
        project = "da6401-a3",
        config  = dict(
            d_model      = D_MODEL,
            n_layers     = N_LAYERS,
            n_heads      = N_HEADS,
            d_ff         = D_FF,
            dropout      = DROPOUT,
            batch_size   = BATCH_SIZE,
            epochs       = EPOCHS,
            warmup_steps = WARMUP_STEPS,
            smoothing    = SMOOTHING,
        ),
    )

    # ── 2. Build dataset / vocabs ─────────────────────────────────────
    print("Loading data …")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size = BATCH_SIZE,
        min_freq   = 2,
        max_len    = MAX_LEN - 2,
    )
    print(f"  src vocab: {len(src_vocab):,}  |  tgt vocab: {len(tgt_vocab):,}")

    # Save vocab for inference
    torch.save(
        {"src_vocab": src_vocab, "trg_vocab": tgt_vocab},
        "vocab.pt",
    )

    # ── 4. Instantiate Transformer (load_weights=False for training) ──
    print("Building model …")
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = D_MODEL,
        N              = N_LAYERS,
        num_heads      = N_HEADS,
        d_ff           = D_FF,
        dropout        = DROPOUT,
        max_len        = MAX_LEN,
        #checkpoint_path= None,        # don't download weights, train fresh
    ).to(DEVICE)
    # Inject freshly-built vocabs so infer() works
    model.src_vocab = src_vocab
    model.trg_vocab = tgt_vocab

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    # ── 5. Adam optimizer ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr    = 1.0,          # actual lr driven by scheduler
        betas = (0.9, 0.98),
        eps   = 1e-9,
    )

    # ── 6. Noam scheduler ─────────────────────────────────────────────
    scheduler = NoamScheduler(optimizer, d_model=D_MODEL, warmup_steps=WARMUP_STEPS)

    # ── 7. Label smoothing loss ───────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = PAD_IDX,
        smoothing  = SMOOTHING,
    )

    # ── 8. Training loop ──────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{EPOCHS} ──")

        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num = epoch,
            is_train  = True,
            device    = str(DEVICE),
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None,
            epoch_num = epoch,
            is_train  = False,
            device    = str(DEVICE),
        )
        val_ppl = math.exp(min(val_loss, 20))

        wandb.log({
            "epoch":            epoch,
            "train/loss":       train_loss,
            "val/loss":         val_loss,
            "val/ppl":          val_ppl,
            "lr":               optimizer.param_groups[0]["lr"],
        })
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ppl={val_ppl:.2f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, "best_model.pt")
            print("  ✓ Saved best_model.pt")

    # ── 9. Final BLEU on test set ─────────────────────────────────────
    print("\nLoading best checkpoint for BLEU evaluation …")
    load_checkpoint("best_model.pt", model)
    model.to(DEVICE)

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=str(DEVICE))
    wandb.log({"test/bleu": bleu})
    print(f"Test BLEU: {bleu:.2f}")

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()