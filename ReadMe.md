# DA6401 Assignment 3 — Neural Machine Translation with Transformer

Implementation of "Attention Is All You Need" (Vaswani et al., 2017) for
German→English translation on the Multi30k dataset.

## Files
- `model.py` — Transformer architecture with `infer()` method
- `lr_scheduler.py` — Noam learning rate scheduler
- `dataset.py` — Multi30k data loading and vocabulary
- `train.py` — Training loop, BLEU evaluation, checkpointing
- `experimentRun.py` — All 5 W&B ablation experiments

## Setup
pip install torch spacy datasets wandb evaluate sacrebleu tqdm gdown
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

## Train
python train.py

## Model Config
d_model=256 | heads=8 | layers=3 | d_ff=512 | dropout=0.1 | warmup=400 | epochs=20

## Inference
model = Transformer().to(device)
model.eval()
model.infer("Ein Mann läuft durch den Park.")

## W&B Report
[link]https://api.wandb.ai/links/na22b076-indian-institute-of-technology-madras/sllexjrg

