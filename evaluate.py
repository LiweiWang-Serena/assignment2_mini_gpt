"""Load a saved checkpoint, report validation perplexity, and generate text.

Usage:
    python evaluate.py                                   # uses mini_gpt_checkpoint.pt
    python evaluate.py --checkpoint checkpoints/baseline.pt --prompt "the city of"
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from data import NextTokenDataset, prepare_token_streams
from train import evaluate, get_device, load_checkpoint


def load_tokenizer():
    """GPT-2 tokenizer from Assignment 1 (optional; only used to show text)."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("gpt2")
    except (ImportError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="mini_gpt_checkpoint.pt")
    parser.add_argument("--data_path", default=None, help="Override the dataset path.")
    parser.add_argument("--prompt", default="the history of")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    cfg = checkpoint["train_config"]
    print(f"Loaded '{cfg['run_name']}' from {args.checkpoint} (epoch {checkpoint['epoch']})")
    print(f"Architecture: {checkpoint['model_config']}")
    print(f"Parameters: {model.num_parameters():,}")

    # Rebuild the same validation split that was used during training.
    _, val_stream, _ = prepare_token_streams(
        args.data_path or cfg["data_path"], cfg["num_blocks"], cfg["n_segments"],
        cfg["val_every"], cache_dir=cfg["cache_dir"],
    )
    val_loader = DataLoader(NextTokenDataset(val_stream, cfg["seq_len"]),
                            batch_size=cfg["batch_size"])
    val_loss, val_ppl = evaluate(model, val_loader, device)
    print(f"Validation loss {val_loss:.4f} | perplexity {val_ppl:,.1f}")

    tokenizer = load_tokenizer()
    if tokenizer is None:
        print("Install 'transformers' to decode generated text.")
        return
    torch.manual_seed(0)
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt.lower())], device=device)
    output = model.generate(prompt_ids, args.max_new_tokens, args.temperature, args.top_k)
    print(f"\nPrompt: {args.prompt!r}\nGenerated: {tokenizer.decode(output[0].tolist())}")


if __name__ == "__main__":
    main()
