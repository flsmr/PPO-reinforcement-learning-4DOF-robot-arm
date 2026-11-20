"""Compare experiments across seeds: learning curves, solve episodes and diagnostics.

Runs are grouped by name, `<experiment>_seed<N>`, e.g. runs/gae_1000_seed3.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from visualize import ROOT, load_history, pyplot, rolling_mean

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
THRESHOLD = 30


def load_experiments(runs_dir, complete_only=True):
    """Return {experiment: [run data, ...]}, keeping only runs with a full 100-episode window."""
    experiments = {}
    for config in sorted(Path(runs_dir).glob("*/config.json")):
        match = re.fullmatch(r"(.+)_seed(\d+)", config.parent.name)
        metadata = json.loads(config.read_text())
        if not match or (complete_only and metadata["status"] != "complete"):
            continue
        _, scores = load_history(run_dir=config.parent)
        rolling = rolling_mean(values=scores)
        if not rolling.size:
            continue
        with (config.parent / "updates.csv").open() as file:
            updates = list(csv.DictReader(file))
        solved = np.flatnonzero(rolling >= THRESHOLD)
        experiments.setdefault(match.group(1), []).append({
            "seed": int(match.group(2)), "scores": scores, "rolling": rolling,
            "solved": int(solved[0]) + 100 if solved.size else None,
            "std": np.array([float(row["latent_std_mean"]) for row in updates]),
            "value_loss": np.array([float(row["value_loss"]) for row in updates]),
        })
    return experiments


def stack(arrays):
    """Stack equal-length prefixes so seeds are averaged over the same episodes."""
    length = min(len(array) for array in arrays)
    return np.stack([array[:length] for array in arrays])


def summary(experiments):
    lines = ["| Experiment | Seeds | Solved | Episodes to solve | Final 100-ep mean | Best 100-ep mean |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, runs in experiments.items():
        solved = [run["solved"] for run in runs if run["solved"] is not None]
        final = [run["rolling"][-1] for run in runs]
        best = [run["rolling"].max() for run in runs]
        solve_text = f"{np.mean(solved):.0f} ± {np.std(solved):.0f}" if solved else "-"
        lines.append(f"| {name} | {len(runs)} | {len(solved)}/{len(runs)} | {solve_text} | "
                     f"{np.mean(final):.2f} ± {np.std(final):.2f} | {np.mean(best):.2f} ± {np.std(best):.2f} |")
    return "\n".join(lines)


def plot(experiments, output_dir, colors, prefix="experiment"):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    colors = [colors[name] for name in experiments]

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for color, (name, runs) in zip(colors, experiments.items()):
        curves = stack([run["rolling"] for run in runs])
        episodes = np.arange(100, 100 + curves.shape[1])
        mean, spread = curves.mean(axis=0), curves.std(axis=0)
        ax.fill_between(episodes, mean - spread, mean + spread, color=color, alpha=0.18, linewidth=0)
        ax.plot(episodes, mean, color=color, linewidth=2, label=f"{name} ({len(runs)} seeds)")
    ax.axhline(y=THRESHOLD, color="#5c5c5c", linestyle="--", linewidth=1.2, label="Solved threshold: 30")
    ax.set(xlabel="Episode", ylabel="100-episode mean return", title="Learning curves, mean ± 1 std over seeds")
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(output_dir / f"{prefix}_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(ncols=2, figsize=(9, 3.6), layout="constrained")
    rng = np.random.default_rng(0)
    for index, (color, (name, runs)) in enumerate(zip(colors, experiments.items())):
        for ax, values in zip(axes, [[run["solved"] for run in runs if run["solved"] is not None],
                                     [run["rolling"][-1] for run in runs]]):
            if values:
                jitter = index + rng.uniform(-0.12, 0.12, size=len(values))
                ax.scatter(jitter, values, s=48, color=color, edgecolors="white", linewidths=1.5, zorder=3)
                ax.plot([index - 0.25, index + 0.25], [np.mean(values)] * 2, color="#1a1a1a", linewidth=2)
    for ax, label in zip(axes, ["First episode with 100-episode mean ≥ 30", "Final 100-episode mean return"]):
        ax.set_xticks(range(len(experiments)), list(experiments), fontsize=9)
        ax.set(ylabel=label)
        ax.grid(alpha=0.18, axis="y")
    fig.savefig(output_dir / f"{prefix}_seeds.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(nrows=2, figsize=(9, 5.5), sharex=True, layout="constrained")
    for color, (name, runs) in zip(colors, experiments.items()):
        for ax, key in zip(axes, ["value_loss", "std"]):
            values = stack([run[key] for run in runs])
            ax.plot(np.arange(1, values.shape[1] + 1), values.mean(axis=0), color=color, linewidth=2, label=name)
    axes[0].set(ylabel="Value loss", yscale="log", title="Training diagnostics, mean over seeds")
    axes[1].set(xlabel="PPO iteration (1,000 steps × 20 arms)", ylabel="Policy std (latent)")
    for ax in axes:
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=9)
    fig.savefig(output_dir / f"{prefix}_diagnostics.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plots")
    parser.add_argument("--include-running", action="store_true", help="Also use unfinished runs")
    parser.add_argument("--experiments", nargs="+", help="Experiments to compare, in plot order (default: all)")
    parser.add_argument("--prefix", default="experiment", help="File name prefix of the figures")
    args = parser.parse_args()
    experiments = load_experiments(runs_dir=args.runs_dir, complete_only=not args.include_running)
    colors = dict(zip(experiments, COLORS * len(experiments)))  # Same color per experiment in every figure.
    if args.experiments:
        missing = set(args.experiments) - set(experiments)
        if missing:
            parser.error(f"Unknown experiments: {', '.join(sorted(missing))}")
        experiments = {name: experiments[name] for name in args.experiments}
    if not experiments:
        parser.error("No runs with at least 100 episodes found")
    print(summary(experiments))
    plot(experiments=experiments, output_dir=args.output_dir, colors=colors, prefix=args.prefix)
    print(f"Plots saved in {args.output_dir}")


if __name__ == "__main__":
    main()
