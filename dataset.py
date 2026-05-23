from collections import Counter
from datasets import load_dataset
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ── Special token constants ────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class Vocabulary:
    """Simple vocabulary with stoi / itos mappings."""

    def __init__(self):
        self.itos = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def build(self, token_lists, min_freq=2):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
        return self

    def numericalize(self, tokens):
        return (
            [SOS_IDX]
            + [self.stoi.get(t, UNK_IDX) for t in tokens]
            + [EOS_IDX]
        )

    def __len__(self):
        return len(self.itos)


class Multi30kDataset(Dataset):
    def __init__(self, split="train"):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split

        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        raw = load_dataset("bentrevett/multi30k")
        self.data = raw[split]

        # Load spacy tokenizers for de and en
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        # Vocabularies — built from training split regardless of current split
        self.src_vocab = None
        self.tgt_vocab = None

        # Raw token lists (filled by process_data)
        self.src_data = []   # list of token-id lists
        self.tgt_data = []   # list of token-id lists

    # ── Tokenizer helpers ──────────────────────────────────────────────────────

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    # ── build_vocab ────────────────────────────────────────────────────────────

    def build_vocab(self, min_freq=2):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        Uses the TRAINING split to build vocab regardless of self.split.
        """
        raw = load_dataset("bentrevett/multi30k")
        train_data = raw["train"]

        src_tokens_all = [self.tokenize_de(s["de"]) for s in train_data]
        tgt_tokens_all = [self.tokenize_en(s["en"]) for s in train_data]

        self.src_vocab = Vocabulary().build(src_tokens_all, min_freq)
        self.tgt_vocab = Vocabulary().build(tgt_tokens_all, min_freq)

        return self.src_vocab, self.tgt_vocab

    # ── process_data ───────────────────────────────────────────────────────────

    def process_data(self, max_len=98):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.
        Populates self.src_data and self.tgt_data.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError("Call build_vocab() before process_data().")

        self.src_data = []
        self.tgt_data = []

        for sample in self.data:
            src_tokens = self.tokenize_de(sample["de"])[:max_len]
            tgt_tokens = self.tokenize_en(sample["en"])[:max_len]
            self.src_data.append(
                torch.tensor(self.src_vocab.numericalize(src_tokens), dtype=torch.long)
            )
            self.tgt_data.append(
                torch.tensor(self.tgt_vocab.numericalize(tgt_tokens), dtype=torch.long)
            )

    # ── PyTorch Dataset interface ──────────────────────────────────────────────

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


# ── Collate function ───────────────────────────────────────────────────────────

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


# ── Convenience loader ─────────────────────────────────────────────────────────

def get_dataloaders(batch_size=128, min_freq=2, max_len=98, num_workers=2):
    """
    Build train / val / test DataLoaders and return shared vocabularies.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    # Build all three splits
    train_ds = Multi30kDataset(split="train")
    val_ds   = Multi30kDataset(split="validation")
    test_ds  = Multi30kDataset(split="test")

    # Build vocab once from training data, share across splits
    src_vocab, tgt_vocab = train_ds.build_vocab(min_freq=min_freq)

    val_ds.src_vocab  = src_vocab
    val_ds.tgt_vocab  = tgt_vocab
    test_ds.src_vocab = src_vocab
    test_ds.tgt_vocab = tgt_vocab

    train_ds.process_data(max_len)
    val_ds.process_data(max_len)
    test_ds.process_data(max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab