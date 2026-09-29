# Assignment 2: Building a Small-Scale Foundation Model from Scratch

A mini-GPT (decoder-only transformer) implemented in PyTorch and trained from
scratch for next-token prediction on the tokenized dataset produced in
Assignment 1 (GPT-2 BPE tokens from Wikipedia, CC-News, and OpenWebText).

## Project structure

```
.
├── model.py            # MiniGPT: embeddings, causal multi-head attention, MLP, LayerNorm
├── data.py             # loads Assignment 1 tokens, train/val split, (input, target) windows
├── train.py            # training loop, perplexity evaluation, checkpoint save/load
├── run_experiments.py  # hyperparameter experiments (one change at a time from a baseline)
├── plot_results.py     # loss and perplexity curves
├── evaluate.py         # reload a checkpoint, report perplexity, generate sample text
├── requirements.txt
├── results/            # per-run metrics (JSON) + summary.csv       (created by training)
├── plots/              # training curves (PNG)                      (created by plot_results.py)
└── mini_gpt_checkpoint.pt   # best model checkpoint                 (created by run_experiments.py)
```

## Model

| Component | Setting (baseline) |
|---|---|
| Transformer blocks | 2 (pre-LayerNorm, residual connections) |
| Embedding dimension | 128 |
| Attention heads | 4 (causal multi-head self-attention, implemented from scratch) |
| Positional encoding | learned positional embeddings |
| Feed-forward | Linear(128 -> 512) -> GELU -> Linear(512 -> 128) |
| Output | final LayerNorm -> linear head (weights tied to the token embedding) -> logits over 50,257 GPT-2 tokens |
| Loss | cross-entropy for next-token prediction |
| Optimizer | AdamW, gradient clipping at 1.0 |

## Data

`data.py` reads `tokenized_blocks.pt` from Assignment 1 (656,119 blocks x 512
tokens) using memory mapping, so only the blocks that are needed are read from
disk. By default it uses 4,000 blocks (~2M tokens), sampled as 40 evenly spaced
segments of consecutive blocks so that all three domains are represented.
EOS padding is removed (one EOS is kept as a document separator), every 10th
segment is held out for validation, and the token stream is cut into
non-overlapping windows of `seq_len` tokens, where the target is the input
shifted by one position.

## Setup

The Assignment 1 virtual environment already has PyTorch. From this folder:

```bash
source ../CS7374-Assignment-1-liwei/venv/bin/activate   # or create a new venv
pip install -r requirements.txt
```

Set the path to the Assignment 1 dataset once (adjust to your machine):

```bash
DATA=../CS7374-Assignment-1-liwei/data/tokenized_blocks.pt
```

## Running

```bash
# 1. Smoke test (200 blocks, 1 epoch) to check that everything runs
python train.py --data_path $DATA --quick

# 2. All hyperparameter experiments (8 runs, 4 epochs each)
python run_experiments.py --data_path $DATA

# 3. Loss and perplexity curves
python plot_results.py

# 4. Reload the best checkpoint, report perplexity, and generate text
python evaluate.py
```

A single run with custom settings:

```bash
python train.py --data_path $DATA --run_name my_run --lr 5e-4 --batch_size 64 --n_layer 1
```

If a run is interrupted, `python run_experiments.py --data_path $DATA --skip_existing`
continues with the runs that have not finished yet. On Apple Silicon the
scripts use the GPU (MPS) automatically; add `--device cpu` if MPS causes errors.

## Hyperparameter experiments

Each run changes one setting from the baseline (lr 1e-3, batch 32, 2 layers,
128-dim embeddings, 4 heads, seq_len 128):

| Run | Change |
|---|---|
| baseline | - |
| lr_5e-4 | learning rate 5e-4 |
| batch_16 / batch_64 | batch size 16 / 64 |
| layers_1 | 1 transformer layer |
| embd_64 / embd_256 | embedding size 64 / 256 |
| seq_64 | sequence length 64 |

## Metrics

- Average training loss per epoch (token-weighted cross-entropy)
- Validation loss and perplexity, where perplexity = exp(average cross-entropy per token)
- Epoch 0 is the untrained model; its loss should be close to ln(50257) = 10.8

## Reproducibility

All runs use seed 42 for weight initialization and data shuffling, and the
exact configuration of every run is stored in `results/<run>.json` and inside
each checkpoint.

## Deliverables

| Deliverable | File |
|---|---|
| Model implementation | `model.py` |
| Training loop | `train.py` (`train_one_epoch`, `run_training`) |
| Checkpoint saving/loading | `train.py` (`save_checkpoint`, `load_checkpoint`), `evaluate.py` |
| Loss and perplexity logging | `train.py`, `results/*.json`, `results/summary.csv` |
| Model checkpoint | `mini_gpt_checkpoint.pt` |
| Visualizations | `plots/*.png` |
