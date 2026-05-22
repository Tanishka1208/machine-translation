"""
dataset.py - Data loading, tokenization, vocabulary building for Multi30k
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import spacy
from datasets import load_dataset

from config import cfg


# ── Tokenizers ─────────────────────────────────────────────────────────────────

def load_tokenizers():
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")
    return spacy_de, spacy_en


def tokenize_de(text, spacy_de):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]


def tokenize_en(text, spacy_en):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]


# ── Vocabulary ─────────────────────────────────────────────────────────────────

class Vocabulary:
    """Simple vocabulary with stoi / itos mappings."""

    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.itos = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def build(self, token_lists):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
        return self

    def numericalize(self, tokens):
        return [cfg.SOS_IDX] + \
               [self.stoi.get(t, cfg.UNK_IDX) for t in tokens] + \
               [cfg.EOS_IDX]

    def __len__(self):
        return len(self.itos)


# ── Dataset ────────────────────────────────────────────────────────────────────

class Multi30kDataset(Dataset):
    def __init__(self, data, src_vocab, trg_vocab, spacy_de, spacy_en, max_len=None):
        self.data      = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.spacy_de  = spacy_de
        self.spacy_en  = spacy_en
        self.max_len   = max_len or cfg.MAX_SEQ_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        src_tokens = tokenize_de(sample["de"], self.spacy_de)[: self.max_len - 2]
        trg_tokens = tokenize_en(sample["en"], self.spacy_en)[: self.max_len - 2]
        src = torch.tensor(self.src_vocab.numericalize(src_tokens), dtype=torch.long)
        trg = torch.tensor(self.trg_vocab.numericalize(trg_tokens), dtype=torch.long)
        return src, trg


def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=cfg.PAD_IDX)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=cfg.PAD_IDX)
    return src_padded, trg_padded


# ── Main loader ────────────────────────────────────────────────────────────────

def get_data():
    """Load Multi30k, build vocabularies, return DataLoaders + vocabs."""
    raw = load_dataset("bentrevett/multi30k")
    spacy_de, spacy_en = load_tokenizers()

    train_data = raw["train"]
    val_data   = raw["validation"]
    test_data  = raw["test"]

    # Tokenize all training sentences for vocab building
    src_tokens_train = [tokenize_de(s["de"], spacy_de) for s in train_data]
    trg_tokens_train = [tokenize_en(s["en"], spacy_en) for s in train_data]

    src_vocab = Vocabulary(min_freq=cfg.MIN_FREQ).build(src_tokens_train)
    trg_vocab = Vocabulary(min_freq=cfg.MIN_FREQ).build(trg_tokens_train)

    train_ds = Multi30kDataset(train_data, src_vocab, trg_vocab, spacy_de, spacy_en)
    val_ds   = Multi30kDataset(val_data,   src_vocab, trg_vocab, spacy_de, spacy_en)
    test_ds  = Multi30kDataset(test_data,  src_vocab, trg_vocab, spacy_de, spacy_en)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, src_vocab, trg_vocab, spacy_de, spacy_en