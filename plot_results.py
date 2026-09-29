"""Plot training loss and perplexity curves from results/*.json.

Usage:
    python plot_results.py                  # reads results/, writes plots/
    python plot_results.py --results_dir results_quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files without opening windows
import matplotlib.pyplot as plt  # noqa: E402


def load_runs(results_dir: Path) -> dict[str, dict]:
    runs = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if "history" in data:
            runs[data["summary"]["run_name"]] = data
    if not runs:
        raise SystemExit(f"No run results found in {results_dir}/")
    return runs


def plot_baseline(run: dict, name: str, out_dir: Path) -> None:
    """Loss (per step and per epoch) and perplexity curves for one run."""
    hist, steps = run["history"], run["step_log"]
    train_epochs = [e for e, v in zip(hist["epoch"], hist["train_loss"]) if v is not None]
    train_loss = [v for v in hist["train_loss"] if v is not None]
    train_ppl = [v for v in hist["train_ppl"] if v is not None]
    batches_per_epoch = run.get("batches_per_epoch") or 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    if steps:
        ax1.plot([s["step"] / batches_per_epoch for s in steps], [s["loss"] for s in steps],
                 alpha=0.35, label="train loss (per step)")
    ax1.plot(train_epochs, train_loss, "o-", label="train loss (epoch avg)")
    ax1.plot(hist["epoch"], hist["val_loss"], "s-", label="validation loss")
    ax1.set(xlabel="epoch", ylabel="cross-entropy loss", title=f"Loss - {name}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(train_epochs, train_ppl, "o-", label="train perplexity")
    ax2.plot(hist["epoch"], hist["val_ppl"], "s-", label="validation perplexity")
    ax2.set_yscale("log")
    ax2.set(xlabel="epoch", ylabel="perplexity (log scale)", title=f"Perplexity - {name}")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_training_curves.png", dpi=150)
    plt.close(fig)


def plot_comparison(runs: dict[str, dict], key: str, ylabel: str, filename: str,
                    out_dir: Path, log_scale: bool = False) -> None:
    """Validation metric per epoch for every run (epoch 0 = untrained model skipped)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, run in runs.items():
        hist = run["history"]
        ax.plot(hist["epoch"][1:], hist[key][1:], "o-", label=name)
    if log_scale:
        ax.set_yscale("log")
    ax.set(xlabel="epoch", ylabel=ylabel, title=f"{ylabel} by run")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_final_bar(runs: dict[str, dict], out_dir: Path) -> None:
    names = sorted(runs, key=lambda n: runs[n]["summary"]["final_val_ppl"])
    values = [runs[n]["summary"]["final_val_ppl"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, values, color="tab:blue")
    ax.bar_label(bars, fmt="%.0f")
    ax.set(ylabel="final validation perplexity", title="Final validation perplexity (lower is better)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "final_val_perplexity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_dir", default="plots")
    parser.add_argument("--baseline", default="baseline", help="Run to plot in detail.")
    args = parser.parse_args()

    runs = load_runs(Path(args.results_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = next((n for n in runs if n.startswith(args.baseline)), next(iter(runs)))
    plot_baseline(runs[detail], detail, out_dir)
    plot_comparison(runs, "val_loss", "validation loss", "val_loss_all_runs.png", out_dir)
    plot_comparison(runs, "val_ppl", "validation perplexity", "val_perplexity_all_runs.png",
                    out_dir, log_scale=True)
    plot_comparison(runs, "train_loss", "training loss", "train_loss_all_runs.png", out_dir)
    plot_final_bar(runs, out_dir)
    print(f"Saved plots for {len(runs)} run(s) to {out_dir}/")


if __name__ == "__main__":
    main()
