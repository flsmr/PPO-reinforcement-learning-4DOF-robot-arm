"""Checks that experiment bookkeeping cannot silently change the comparison."""
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from visualize import create_run, finish_run, load_history, plot_comparison, plot_run, rolling_mean


class RunDataTests(unittest.TestCase):
    def test_training_records_launch_failure_without_touching_completed_runs(self):
        import train
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(train, "make_env", side_effect=RuntimeError("fixture launch failure")):
                with self.assertRaisesRegex(RuntimeError, "fixture launch failure"):
                    train.train(iterations=1, rollout_steps=1, run_name="failed", runs_dir=Path(temporary))
            folder = Path(temporary) / "failed"
            metadata = json.loads((folder / "config.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertFalse((folder / "checkpoint.pt").exists())
            self.assertEqual(load_history(run_dir=folder)[1].size, 0)

    def test_comparison_requires_two_complete_runs_with_full_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folders = []
            for index, length in enumerate([100, 100, 99]):
                folder = root / f"fixture_{index}"
                folder.mkdir()
                (folder / "config.json").write_text(json.dumps({"status": "complete", "seed": index}))
                with (folder / "episodes.csv").open(mode="w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=["episode", "score"])
                    writer.writeheader()
                    writer.writerows({"episode": i+1, "score": 10+index} for i in range(length))
                folders.append(folder)
            output = root / "comparison.png"
            plot_comparison(run_dirs=folders, output=output)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))
            plot_comparison(run_dirs=[folders[0], folders[2]], output=output)
            self.assertFalse(output.exists())

    def test_failed_launch_without_history_does_not_break_plotting(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text('{"status": "failed"}')
            episodes, scores = load_history(run_dir=folder)
            self.assertEqual(episodes.size, 0)
            self.assertEqual(scores.size, 0)
            (folder / "config.json").write_text('{"status": "complete"}')
            with self.assertRaises(FileNotFoundError):
                load_history(run_dir=folder)

    def test_default_cli_rollout_matches_baseline_but_random_episode_is_complete(self):
        import train
        with patch("sys.argv", ["train.py"]), patch.object(train, "train") as training:
            train.main()
            self.assertEqual(training.call_args.kwargs["rollout_steps"], 1000)
        with patch("sys.argv", ["train.py", "--random"]), patch.object(train, "random_episode") as random:
            train.main()
            self.assertEqual(random.call_args.kwargs["steps"], 1001)

    def test_rolling_mean_requires_complete_windows(self):
        self.assertEqual(rolling_mean(values=[40.0] * 99).size, 0)
        self.assertEqual(rolling_mean(values=[30.0] * 100)[0], 30.0)
        np.testing.assert_allclose(actual=rolling_mean(values=[40.0] * 100), desired=[40.0])
        np.testing.assert_allclose(actual=rolling_mean(values=[1, 2, 3, 4], window=2), desired=[1.5, 2.5, 3.5])

    def test_run_names_cannot_overwrite_or_escape_the_runs_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = create_run(name="baseline_seed0", runs_dir=root)
            self.assertEqual(folder, root / "baseline_seed0")
            with self.assertRaises(FileExistsError):
                create_run(name="baseline_seed0", runs_dir=root)
            for name in ["..", "../baseline", "a/b", "/tmp/outside"]:
                with self.assertRaises(ValueError):
                    create_run(name=name, runs_dir=root)

    def test_finishing_run_saves_only_config_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "baseline"
            folder.mkdir()
            checkpoint = {
                "episode_scores": [31.0] * 100,
                "solved_episode": 100, "seed": 0,
                "hyperparameters": {"gamma": 0.99},
            }
            finish_run(run_dir=folder, checkpoint=checkpoint, imported=True)
            self.assertEqual({p.name for p in folder.iterdir()}, {"config.json", "episodes.csv"})
            with (folder / "episodes.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[98]["rolling100"], "")
            self.assertEqual(float(rows[99]["rolling100"]), 31.0)
            self.assertEqual(rows[0]["elapsed_seconds"], "")
            plot_run(run_dir=folder, output_dir=Path(temporary) / "plots")
            self.assertEqual({p.name for p in folder.iterdir()}, {"config.json", "episodes.csv"})
            self.assertTrue((Path(temporary) / "plots/learning_curve.png").exists())


if __name__ == "__main__":
    unittest.main()
