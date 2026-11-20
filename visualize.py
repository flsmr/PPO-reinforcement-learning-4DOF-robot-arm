"""Store run data and regenerate a small set of comparable Reacher plots."""
import argparse
import csv
import json
import os
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent
EPISODE_FIELDS = ["episode", "score", "rolling100", "transitions", "elapsed_seconds"]
UPDATE_FIELDS = ["iteration", "transitions", "elapsed_seconds", "policy_loss",
                 "value_loss", "optimizer_steps", "latent_std_mean"]


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rolling_mean(values, window=100):
    """Only full windows count; the first rolling100 value belongs to episode 100."""
    values = np.asarray(a=values, dtype=np.float64)
    if window < 1:
        raise ValueError("The rolling window must be positive.")
    if values.size < window:
        return np.array(object=[], dtype=np.float64)
    # Match training's np.mean directly, including an exact threshold window of 30.
    windows = np.lib.stride_tricks.sliding_window_view(x=values, window_shape=window)
    return windows.mean(axis=-1)


def create_run(name, runs_dir=ROOT / "runs"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Use a run name containing letters, numbers, underscores or hyphens.")
    folder = Path(runs_dir) / name
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()  # Fail instead of overwriting an earlier experiment.
    return folder


def finish_run(run_dir, checkpoint, imported=False, checkpoint_path=None):
    scores = np.asarray(a=checkpoint["episode_scores"], dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Episode scores must be a finite one-dimensional sequence.")
    rolling = rolling_mean(values=scores)
    solved = np.flatnonzero(rolling >= 30)
    solved_episode = int(solved[0]) + 100 if solved.size else None
    if checkpoint.get("solved_episode") != solved_episode:
        raise ValueError("Checkpoint success metadata disagrees with its episode scores.")

    episode_path = run_dir / "episodes.csv"
    if not episode_path.exists():
        with episode_path.open(mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=EPISODE_FIELDS, lineterminator="\n")
            writer.writeheader()
            for index, score in enumerate(scores):
                writer.writerow({
                    "episode": index + 1, "score": float(score),
                    "rolling100": float(rolling[index - 99]) if index >= 99 else "",
                    # Legacy checkpoints did not record these per-episode quantities.
                    "transitions": "", "elapsed_seconds": "",
                })
    else:
        with episode_path.open() as file:
            rows = list(csv.DictReader(file))
        if len(rows) != len(scores) or not np.allclose(
            a=[float(row["score"]) for row in rows], b=scores, rtol=0, atol=1e-10
        ):
            raise ValueError("Episode CSV disagrees with the checkpoint.")

    metadata = {key: checkpoint[key] for key in [
        "seed", "worker_id", "state_size", "action_size", "hidden_size", "num_agents", "iterations",
        "rollout_steps", "transitions", "hyperparameters", "features", "software",
    ] if key in checkpoint}
    metadata.update(
        name=run_dir.name, status="complete", episodes=len(scores),
        final_rolling100=float(rolling[-1]) if rolling.size else None,
        best_rolling100=float(rolling.max()) if rolling.size else None,
        recent10=float(scores[-10:].mean()) if scores.size else None,
        solved_episode=solved_episode, imported=imported,
    )
    if checkpoint_path is not None:
        metadata["checkpoint"] = str(checkpoint_path)
    if imported:
        metadata["notes"] = (
            "Imported from a completed checkpoint. Episode scores are original; "
            "per-episode timing, transition stamps and update history were not recorded and are "
            "left unavailable."
        )
    write_json(path=run_dir / "config.json", data=metadata)
    return metadata


def load_history(run_dir):
    run_dir = Path(run_dir)
    if not (run_dir / "episodes.csv").exists():
        metadata = json.loads((run_dir / "config.json").read_text())
        if metadata["status"] != "complete":
            return np.array(object=[], dtype=int), np.array(object=[], dtype=np.float64)
        raise FileNotFoundError(f"Completed run is missing episodes.csv: {run_dir}")
    with (run_dir / "episodes.csv").open() as file:
        rows = list(csv.DictReader(file))
    episodes = np.asarray(a=[int(row["episode"]) for row in rows])
    scores = np.asarray(a=[float(row["score"]) for row in rows])
    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite rewards in {run_dir}")
    return episodes, scores


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    return plt


def plot_run(run_dir, output_dir=None):
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "plots" / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run_dir / "config.json").read_text())
    episodes, scores = load_history(run_dir=run_dir)
    if not scores.size:
        print(f"No complete episodes yet: {run_dir.name}")
        return
    plt = pyplot()
    average = rolling_mean(values=scores)
    suffix = f' ({metadata["status"]})' if metadata["status"] != "complete" else ""
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.plot(episodes, scores, color="#94A3B8", alpha=0.7, linewidth=1,
            label="Episode mean across arms")
    if average.size:
        ax.plot(episodes[99:], average, color="#2563EB", linewidth=2.2, label="100-episode mean")
    ax.axhline(y=30, color="#BE3448", linestyle="--", linewidth=1.2, label="Solved threshold: 30")
    ax.set(title=f"{run_dir.name}{suffix}", xlabel="Complete episode", ylabel="Undiscounted return")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(output_dir / "learning_curve.png", dpi=180)
    plt.close(fig)

    update_path = run_dir / "updates.csv"
    if update_path.exists():
        with update_path.open() as file:
            rows = list(csv.DictReader(file))
        if rows:
            transitions = [int(row["transitions"]) for row in rows]
            fig, axes = plt.subplots(nrows=2, figsize=(9, 6), sharex=True, layout="constrained")
            axes[0].plot(transitions, [float(row["value_loss"]) for row in rows], color="#2563EB")
            axes[0].set(ylabel="Value loss", title=f"{run_dir.name}: training diagnostics")
            axes[1].plot(transitions, [float(row["latent_std_mean"]) for row in rows], color="#0D9488")
            axes[1].set(xlabel="Arm transitions", ylabel="Mean latent Gaussian std")
            for ax in axes:
                ax.grid(alpha=0.18)
            fig.savefig(output_dir / "diagnostics.png", dpi=180)
            plt.close(fig)
    print(f"Plots saved in {output_dir}")


def plot_comparison(run_dirs, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    count = 0
    for run_dir in run_dirs:
        metadata = json.loads((run_dir / "config.json").read_text())
        if metadata["status"] != "complete":
            continue
        episodes, scores = load_history(run_dir=run_dir)
        average = rolling_mean(values=scores)
        if not average.size:
            continue
        ax.plot(episodes[99:], average, linewidth=2, label=f'{run_dir.name} (seed {metadata.get("seed", "?")})')
        count += 1
    if count >= 2:
        ax.axhline(y=30, color="#BE3448", linestyle="--", linewidth=1.2)
        ax.set(title="Run comparison", xlabel="Complete episode", ylabel="100-episode mean return")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9)
        fig.savefig(output, dpi=180)
        print(f"Comparison saved to {output}")
    elif output.exists():
        output.unlink()  # Do not leave a comparison from a previous selection on disk.
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--run", type=Path, nargs="+", help="Plot selected run directories")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plots")
    args = parser.parse_args()
    run_dirs = args.run if args.run else sorted(path.parent for path in args.runs_dir.glob("*/config.json"))
    if not run_dirs:
        parser.error("No saved runs found")
    for run_dir in run_dirs:
        plot_run(run_dir=run_dir, output_dir=args.output_dir / run_dir.name)
    plot_comparison(run_dirs=run_dirs, output=args.output_dir / "comparison.png")


if __name__ == "__main__":
    main()
