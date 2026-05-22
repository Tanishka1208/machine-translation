"""
loss.py - Label Smoothing Cross-Entropy Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as described in Section 5.4 of the paper.

    Instead of hard 0/1 targets the model is trained with:
        y_smooth = (1 - eps) * y_one_hot  +  eps / (V - 1)   (excluding pad)

    eps = 0.0 reduces to standard cross-entropy.
    """

    def __init__(self, vocab_size, pad_idx=0, eps=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.eps        = eps
        self.criterion  = nn.KLDivLoss(reduction="sum")

    def forward(self, logits, targets):
        """
        logits  : (batch * trg_len, vocab_size)  – raw model output (pre-softmax)
        targets : (batch * trg_len,)
        Returns : scalar loss
        """
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.eps / (self.vocab_size - 2))
            smooth[:, self.pad_idx] = 0.0
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.eps)
            # Zero out padding positions completely
            pad_mask = targets.eq(self.pad_idx)
            smooth[pad_mask] = 0.0

        loss = self.criterion(log_probs, smooth)
        n_tokens = (~pad_mask).sum().float()
        return loss / n_tokens