"""Train a mini-GPT on the Assignment 1 dataset (next-token prediction).

Examples:
    python train.py --data_path ../CS7374-Assignment-1-liwei/data/tokenized_blocks.pt
    python train.py --quick          # tiny smoke test (a few minutes at most)
    python train.py --lr 5e-4 --n_layer 1 --run_name my_run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F

from data import GPT2_VOCAB_SIZE, make_dataloaders, prepare_token_streams
from model import GPTConfig, MiniGPT


@dataclass
class TrainConfig:
    """Everything needed to reproduce one training run."""

    run_name: str = "baseline"
    data_path: str = "data/tokenized_blocks.pt"
    num_blocks: int = 4000  # 512-token blocks read from Assignment 1 (~2M tokens)
    n_segments: int = 40  # blocks are sampled as evenly spaced segments
    val_every: int = 10  # every 10th segment is validation (10%)
    seq_len: int = 128  # tokens per input block (assignment: 32-128)
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 4
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    log_interval: int = 50  # log the training loss every N optimizer steps
    device: str = "auto"  # auto | cpu | mps | cuda
    results_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    cache_dir: str = "data/cache"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def perplexity(loss: float) -> float:
    """Perplexity = exp(average cross-entropy per token)."""
    return math.exp(loss) if loss < 700 else float("inf")


@torch.no_grad()
def evaluate(model: MiniGPT, loader, device: torch.device) -> tuple[float, float]:
    """Return (average loss per token, perplexity) on a dataset."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return avg_loss, perplexity(avg_loss)


def train_one_epoch(
    model: MiniGPT,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    global_step: int,
    step_log: list,
) -> tuple[float, int]:
    """One pass over the training data. Returns (average loss, global_step)."""
    model.train()
    total_loss, total_tokens = 0.0, 0
    start = time.time()

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)  # 1. forward pass + cross-entropy loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # 2. backward pass (backpropagation)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()  # 3. optimizer step

        n_tokens = y.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
        global_step += 1

        if global_step % cfg.log_interval == 0:
            step_log.append({"step": global_step, "epoch": epoch, "loss": loss.item()})
            tokens_per_sec = total_tokens / max(time.time() - start, 1e-9)
            print(
                f"  epoch {epoch} step {global_step:5d} | loss {loss.item():.4f} "
                f"| {tokens_per_sec:,.0f} tokens/s"
            )

    return total_loss / total_tokens, global_step


def save_checkpoint(
    path: str | Path,
    model: MiniGPT,
    train_cfg: TrainConfig,
    history: dict,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save weights plus everything needed to rebuild and evaluate the model."""
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "train_config": asdict(train_cfg),
        "history": history,
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[MiniGPT, dict]:
    """Rebuild a MiniGPT from a checkpoint file and load its weights."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MiniGPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def build_optimizer(model: MiniGPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW; weight decay applies to weight matrices, not biases/LayerNorm."""
    decay = [p for _, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )


def run_training(cfg: TrainConfig) -> dict:
    """Train one model, log metrics, save a checkpoint, and return a summary."""
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"\n=== Run '{cfg.run_name}' on {device} ===")

    train_stream, val_stream, data_info = prepare_token_streams(
        cfg.data_path, cfg.num_blocks, cfg.n_segments, cfg.val_every, cache_dir=cfg.cache_dir
    )
    train_loader, val_loader = make_dataloaders(
        train_stream, val_stream, cfg.seq_len, cfg.batch_size, cfg.seed
    )
    if len(train_loader) == 0:
        raise ValueError("Not enough training data for one batch; increase --num_blocks.")
    print(
        f"Data: {data_info['train_tokens']:,} train tokens, {data_info['val_tokens']:,} val "
        f"tokens | {len(train_loader)} train batches/epoch"
    )

    model_cfg = GPTConfig(
        vocab_size=max(GPT2_VOCAB_SIZE, data_info["max_token_id"] + 1),
        block_size=cfg.seq_len,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = MiniGPT(model_cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    n_params = model.num_parameters()
    n_params_non_emb = model.num_parameters(exclude_embeddings=True)
    print(f"Model: {n_params:,} parameters ({n_params_non_emb:,} excluding embeddings)")

    history = {
        "epoch": [], "train_loss": [], "train_ppl": [],
        "val_loss": [], "val_ppl": [], "epoch_time_sec": [],
    }
    step_log: list = []

    # Epoch 0 = the untrained model (loss should be close to ln(vocab) ~ 10.8).
    val_loss, val_ppl = evaluate(model, val_loader, device)
    for key, value in (("epoch", 0), ("train_loss", None), ("train_ppl", None),
                       ("val_loss", val_loss), ("val_ppl", val_ppl), ("epoch_time_sec", 0.0)):
        history[key].append(value)
    print(f"epoch 0 (untrained) | val loss {val_loss:.4f} | val ppl {val_ppl:,.1f}")

    checkpoint_path = Path(cfg.checkpoint_dir) / f"{cfg.run_name}.pt"
    global_step, run_start = 0, time.time()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, device, cfg, epoch, global_step, step_log
        )
        val_loss, val_ppl = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_ppl"].append(perplexity(train_loss))
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["epoch_time_sec"].append(epoch_time)
        print(
            f"epoch {epoch} | train loss {train_loss:.4f} (ppl {perplexity(train_loss):,.1f}) "
            f"| val loss {val_loss:.4f} (ppl {val_ppl:,.1f}) | {epoch_time:.0f}s"
        )
        save_checkpoint(checkpoint_path, model, cfg, history, epoch)  # overwrite each epoch

    total_time = time.time() - run_start

    # Round-trip check: reload the saved checkpoint and confirm the same val loss.
    reloaded, _ = load_checkpoint(checkpoint_path, device)
    reload_loss, _ = evaluate(reloaded, val_loader, device)
    print(f"Checkpoint saved to {checkpoint_path} (reloaded val loss {reload_loss:.4f})")

    summary = {
        "run_name": cfg.run_name,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "n_embd": cfg.n_embd,
        "seq_len": cfg.seq_len,
        "epochs": cfg.epochs,
        "params": n_params,
        "params_non_embedding": n_params_non_emb,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_ppl": history["val_ppl"][-1],
        "best_val_ppl": min(history["val_ppl"]),
        "total_time_sec": total_time,
        "checkpoint": str(checkpoint_path),
    }

    results_path = Path(cfg.results_dir) / f"{cfg.run_name}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({
        "summary": summary,
        "train_config": asdict(cfg),
        "model_config": model_cfg.to_dict(),
        "data_info": data_info,
        "device": str(device),
        "batches_per_epoch": len(train_loader),
        "history": history,
        "step_log": step_log,
    }, indent=2))
    print(f"Metrics saved to {results_path}")
    return summary


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a mini-GPT from scratch.")
    for f in fields(TrainConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    parser.add_argument(
        "--quick", action="store_true", help="Tiny smoke test: 200 blocks, 1 epoch."
    )
    args = vars(parser.parse_args())
    quick = args.pop("quick")
    cfg = TrainConfig(**args)
    if quick:
        cfg.num_blocks, cfg.n_segments, cfg.epochs, cfg.log_interval = 200, 10, 1, 10
        cfg.run_name = f"{cfg.run_name}_quick"
    return cfg


if __name__ == "__main__":
    run_training(parse_args())
