"""
config.py - All hyperparameters and configuration for DA6401 Assignment 3
"""

import torch

class Config:
    # ── Model ──────────────────────────────────────────────────────────
    D_MODEL       = 256          # embedding / model dimension
    N_HEADS       = 8            # number of attention heads
    N_ENCODER     = 3            # encoder layers
    N_DECODER     = 3            # decoder layers
    D_FF          = 512          # feed-forward inner dimension
    DROPOUT       = 0.1
    MAX_SEQ_LEN   = 100          # max tokens per sequence

    # ── Training ───────────────────────────────────────────────────────
    BATCH_SIZE    = 128
    EPOCHS        = 20
    WARMUP_STEPS  = 400
    LABEL_SMOOTH  = 0.1
    CLIP          = 1.0          # gradient clipping

    # ── Data ───────────────────────────────────────────────────────────
    SRC_LANG      = "de"
    TRG_LANG      = "en"
    MIN_FREQ      = 2            # minimum token frequency for vocab

    # ── Special tokens ─────────────────────────────────────────────────
    PAD_IDX       = 0
    SOS_IDX       = 1
    EOS_IDX       = 2
    UNK_IDX       = 3

    # ── Paths ──────────────────────────────────────────────────────────
    MODEL_SAVE    = "/kaggle/working/machine_translation/best_model.pt"
    VOCAB_SAVE    = "/kaggle/working/machine_translation/vocab.pt"
    VOCAB_FILENAME = "/kaggle/working/machine_translation/vocab.pt"


    # ── Device ─────────────────────────────────────────────────────────
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── W&B ────────────────────────────────────────────────────────────
    WANDB_PROJECT = "da6401-assignment3-machine-translation"  # set your W&B project name here if needed
    WANDB_ENTITY  = None         # set your W&B username here if needed


cfg = Config()