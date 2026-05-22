"""
scheduler.py - Noam Learning Rate Scheduler from "Attention Is All You Need"

lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""

import torch.optim as optim


class NoamScheduler:
    """
    Wraps an Adam optimiser and adjusts the learning rate each step
    according to the Noam schedule.

    Usage:
        optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model, warmup_steps)

        for batch in data_loader:
            loss = ...
            loss.backward()
            optimizer.step()
            scheduler.step()
    """

    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer     = optimizer
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        self._step         = 0
        self._rate         = 0.0

    def step(self):
        self._step += 1
        rate = self._compute_lr(self._step)
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _compute_lr(self, step):
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )

    @property
    def current_lr(self):
        return self._rate

    @property
    def current_step(self):
        return self._step