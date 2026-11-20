"""Prepared experiments must launch with their saved settings and protect results."""
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train


def prepared_config(folder):
    folder.mkdir()
    config = {
        'name': folder.name, 'status': 'prepared', 'seed': 3, 'worker_id': 43,
        'iterations': 1000, 'rollout_steps': 1000, 'hidden_size': 128,
        'features': {'use_gae': True, 'normalize_advantages': True},
        'hyperparameters': {'learning_rate': 0.0003, 'gamma': 0.99, 'clip_epsilon': 0.2,
                            'update_epochs': 10, 'minibatch_size': 512,
                            'value_loss_coef': 0.5, 'gae_lambda': 0.95},
    }
    path = folder / 'config.json'
    path.write_text(json.dumps(config))
    return path


class RunConfigTests(unittest.TestCase):
    def test_config_launch_passes_all_settings_to_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = prepared_config(Path(temporary) / 'fixture')
            with patch.object(train, 'train') as training:
                train.train_from_config(config_path=path)
            settings = training.call_args.kwargs
            self.assertEqual(settings['iterations'], 1000)
            self.assertEqual(settings['seed'], 3)
            self.assertEqual(settings['worker_id'], 43)
            self.assertEqual(settings['runs_dir'], path.parent.parent)
            self.assertTrue(settings['use_gae'])
            self.assertTrue(settings['normalize_advantages'])
            self.assertEqual(settings['agent_options']['minibatch_size'], 512)
            self.assertEqual(settings['agent_options']['hidden_size'], 128)

    def test_used_run_and_unknown_feature_are_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = prepared_config(Path(temporary) / 'fixture')
            config = json.loads(path.read_text())
            for status in ['running', 'complete', 'failed', 'interrupted']:
                config['status'] = status
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError), patch.object(train, 'make_env') as environment:
                    train.train_from_config(config_path=path)
                environment.assert_not_called()
            config['status'] = 'prepared'
            config['features']['unknown_switch'] = True
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                train.train_from_config(config_path=path)

    def test_launch_failure_marks_config_failed_and_cannot_restart_over_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = prepared_config(Path(temporary) / 'fixture')
            with patch.object(train, 'make_env', side_effect=RuntimeError('fixture failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    train.train_from_config(config_path=path)
            self.assertEqual(json.loads(path.read_text())['status'], 'failed')
            with self.assertRaises(ValueError):
                train.train_from_config(config_path=path)

    def test_directory_lock_prevents_duplicate_launch_without_extra_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = prepared_config(Path(temporary) / 'fixture')
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, 'already running'):
                    train.train_from_config(config_path=path)
            finally:
                os.close(descriptor)
            self.assertEqual({p.name for p in path.parent.iterdir()}, {'config.json'})

    def test_new_runs_directory_cannot_overwrite_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "checkpoints").mkdir()
            checkpoint = root / "checkpoints" / "protected.pt"
            checkpoint.write_bytes(b"original checkpoint")
            with patch.object(train, "ROOT", root), patch.object(train, "make_env") as environment:
                with self.assertRaises(FileExistsError):
                    train.train(run_name="protected", runs_dir=root / "other_runs")
            environment.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b"original checkpoint")

    def test_cli_config_cannot_silently_override_saved_settings(self):
        with patch('sys.argv', ['train.py', '--config', 'fixture/config.json']), patch.object(train, 'train_from_config') as launch:
            train.main()
            self.assertEqual(launch.call_args.kwargs['config_path'], Path('fixture/config.json'))
        with patch('sys.argv', ['train.py', '--config', 'fixture/config.json', '--seed', '9']):
            with self.assertRaises(SystemExit):
                train.main()


if __name__ == '__main__':
    unittest.main()
