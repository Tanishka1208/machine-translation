"""
utils.py - Utility functions: BLEU evaluation, greedy decode, attention plots
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from evaluate import load as load_metric

from config import cfg


# ── BLEU ───────────────────────────────────────────────────────────────────────

_bleu_metric = None

def get_bleu_metric():
    global _bleu_metric
    if _bleu_metric is None:
        _bleu_metric = load_metric("bleu")
    return _bleu_metric


def compute_bleu(predictions, references):
    """
    predictions : list of strings
    references  : list of strings
    Returns     : corpus BLEU score (0-100)
    """
    metric = get_bleu_metric()
    refs   = [[r] for r in references]
    result = metric.compute(predictions=predictions, references=refs)
    return result["bleu"] * 100


# ── Greedy decoding ────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, sos_idx, eos_idx, device):
    """
    Autoregressive greedy decoding.
    src      : (1, src_len)
    src_mask : (1, 1, 1, src_len)
    Returns  : list of token indices (excluding <sos>)
    """
    enc_out = model.encode(src, src_mask)
    ys = torch.tensor([[sos_idx]], device=device)   # (1, 1)

    for _ in range(max_len):
        trg_mask = model.make_trg_mask(ys)
        dec_out  = model.decode(ys, enc_out, trg_mask, src_mask)
        logits   = model.generator(dec_out[:, -1, :])   # (1, vocab)
        pred     = logits.argmax(-1).item()
        if pred == eos_idx:
            break
        ys = torch.cat([ys, torch.tensor([[pred]], device=device)], dim=1)

    return ys[0, 1:].tolist()   # strip <sos>


# ── Translation helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def translate_sentence(sentence, model, src_vocab, trg_vocab, spacy_de, device,
                        max_len=100):
    """
    Translate a single German sentence string → English string.
    Used by the infer() method and evaluation.
    """
    model.eval()
    tokens = [tok.text.lower() for tok in spacy_de.tokenizer(sentence)]
    tokens = tokens[: max_len - 2]
    src_ids = [cfg.SOS_IDX] + \
              [src_vocab.stoi.get(t, cfg.UNK_IDX) for t in tokens] + \
              [cfg.EOS_IDX]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_mask = (src != cfg.PAD_IDX).unsqueeze(1).unsqueeze(2)   # True = attend

    # Invert: our mask convention is True = MASKED (ignored); here src has no pads
    src_mask = ~src_mask   # (1, 1, 1, src_len) — all False for non-padded input

    token_ids = greedy_decode(model, src, src_mask, max_len,
                              cfg.SOS_IDX, cfg.EOS_IDX, device)
    translation = " ".join(trg_vocab.itos[i] for i in token_ids
                           if i not in (cfg.SOS_IDX, cfg.EOS_IDX, cfg.PAD_IDX))
    return translation


# ── Evaluation on DataLoader ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_bleu_loader(model, loader, src_vocab, trg_vocab, spacy_de, device,
                          max_samples=500):
    """
    Compute corpus BLEU over the first max_samples examples in a DataLoader.
    """
    model.eval()
    predictions, references = [], []

    for i, (src, trg) in enumerate(loader):
        if i * cfg.BATCH_SIZE >= max_samples:
            break
        for j in range(src.size(0)):
            # Decode German tokens back to string for translate_sentence
            src_ids = src[j].tolist()
            de_tokens = [src_vocab.itos[idx] for idx in src_ids
                         if idx not in (cfg.PAD_IDX, cfg.SOS_IDX, cfg.EOS_IDX)]
            de_sentence = " ".join(de_tokens)
            pred = translate_sentence(de_sentence, model, src_vocab, trg_vocab,
                                      spacy_de, device)
            ref_ids = trg[j].tolist()
            ref_tokens = [trg_vocab.itos[idx] for idx in ref_ids
                          if idx not in (cfg.PAD_IDX, cfg.SOS_IDX, cfg.EOS_IDX)]
            ref = " ".join(ref_tokens)
            predictions.append(pred)
            references.append(ref)

    return compute_bleu(predictions, references)


# ── Attention visualisation ────────────────────────────────────────────────────

def plot_attention_heads(attn_weights, src_tokens, trg_tokens, layer_name="encoder"):
    """
    attn_weights : (n_heads, seq_q, seq_k) – numpy array
    Returns a wandb.Image list, one per head.
    """
    n_heads = attn_weights.shape[0]
    images  = []
    for h in range(n_heads):
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(attn_weights[h], cmap="Blues", aspect="auto",
                       vmin=0, vmax=attn_weights[h].max())
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=90, fontsize=8)
        ax.set_yticks(range(len(trg_tokens)))
        ax.set_yticklabels(trg_tokens, fontsize=8)
        ax.set_title(f"{layer_name} | Head {h+1}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        images.append(wandb.Image(fig, caption=f"{layer_name} Head {h+1}"))
        plt.close(fig)
    return images


def get_prediction_confidence(logits):
    """
    logits : (batch * trg_len, vocab_size)
    Returns mean softmax probability assigned to the correct (greedy) token.
    """
    probs   = torch.softmax(logits, dim=-1)
    max_p   = probs.max(dim=-1).values
    return max_p.mean().item()