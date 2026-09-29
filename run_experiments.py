"""Hyperparameter experiments: change one setting at a time from a baseline.

Examples:
    python run_experiments.py --data_path ../CS7374-Assignment-1-liwei/data/tokenized_blocks.pt
    python run_experiments.py --quick                 # smoke test of every run
    python run_experiments.py --only baseline lr_5e-4 # run a subset
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from train import TrainConfig, load_checkpoint, run_training

# Baseline: lr=1e-3, batch=32, 2 layers, 128-dim embeddings, 4 heads, seq_len=128.
EXPERIMENTS: dict[str, dict] = {
    "baseline": {},
    "lr_5e-4": {"lr": 5e-4},
    "batch_16": {"batch_size": 16},
    "batch_64": {"batch_size": 64},
    "layers_1": {"n_layer": 1},
    "embd_64": {"n_embd": 64},
    "embd_256": {"n_embd": 256},
    "seq_64": {"seq_len": 64},
}

DELIVERABLE_CHECKPOINT = "mini_gpt_checkpoint.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hyperparameter experiments.")
    parser.add_argument("--data_path", default=TrainConfig.data_path)
    parser.add_argument("--num_blocks", type=int, default=TrainConfig.num_blocks)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--only", nargs="*", choices=list(EXPERIMENTS), help="Runs to execute.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip runs whose results/<run>.json already exists.")
    parser.add_argument("--quick", action="store_true", help="Smoke test: 200 blocks, 1 epoch.")
    args = parser.parse_args()

    names = args.only or list(EXPERIMENTS)
    summaries = []
    for name in names:
        cfg = TrainConfig(run_name=name, data_path=args.data_path, num_blocks=args.num_blocks,
                          epochs=args.epochs, device=args.device, **EXPERIMENTS[name])
        if args.quick:
            cfg.num_blocks, cfg.n_segments, cfg.epochs, cfg.log_interval = 200, 10, 1, 10
            cfg.run_name, cfg.results_dir = f"{name}_quick", "results_quick"

        result_file = Path(cfg.results_dir) / f"{cfg.run_name}.json"
        if args.skip_existing and result_file.exists():
            print(f"Skipping '{cfg.run_name}' (found {result_file})")
            summaries.append(json.loads(result_file.read_text())["summary"])
            continue
        summaries.append(run_training(cfg))

    results_dir = Path("results_quick" if args.quick else "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print("\n=== Summary (sorted by final validation perplexity) ===")
    print(f"{'run':<12}{'params':>12}{'train loss':>12}{'val loss':>10}{'val ppl':>10}{'time':>8}")
    for s in sorted(summaries, key=lambda s: s["final_val_ppl"]):
        print(f"{s['run_name']:<12}{s['params']:>12,}{s['final_train_loss']:>12.4f}"
              f"{s['final_val_loss']:>10.4f}{s['final_val_ppl']:>10.1f}{s['total_time_sec']:>7.0f}s")

    if not args.quick:
        best = min(summaries, key=lambda s: s["final_val_ppl"])
        shutil.copy(best["checkpoint"], DELIVERABLE_CHECKPOINT)
        load_checkpoint(DELIVERABLE_CHECKPOINT)  # confirm the deliverable loads
        print(f"\nBest run: '{best['run_name']}' (val ppl {best['final_val_ppl']:.1f}) "
              f"-> copied to {DELIVERABLE_CHECKPOINT}")


if __name__ == "__main__":
    main()
