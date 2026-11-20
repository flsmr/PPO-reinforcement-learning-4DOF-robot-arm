"""Check the normalization switch from CLI through optimization and saved settings."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ppo_agent import PPOAgent
from test_gae import example_rollout
import train


class AdvantageNormalizationTests(unittest.TestCase):
    def test_switch_changes_actor_advantages_but_preserves_critic_targets(self):
        for enabled in [False, True]:
            with self.subTest(enabled=enabled):
                rollout = example_rollout()
                raw_advantages, raw_returns = rollout.compute_advantages(gamma=0.5)
                expected = torch.as_tensor(data=raw_advantages).flatten()
                if enabled:
                    expected = (expected - expected.mean()) / (expected.std(unbiased=False) + 1e-8)
                agent = PPOAgent(state_size=1, action_size=1, gamma=0.5,
                                 normalize_advantages=enabled, update_epochs=1, minibatch_size=6)
                with patch.object(agent, 'compute_loss', wraps=agent.compute_loss) as loss:
                    agent.update(rollout=rollout)
                batch = loss.call_args.kwargs
                torch.testing.assert_close(actual=batch['advantages'].sort().values,
                                           expected=expected.sort().values)
                torch.testing.assert_close(actual=batch['returns'].sort().values,
                                           expected=torch.as_tensor(data=raw_returns).flatten().sort().values)

    def test_cli_passes_independent_switch_with_default_disabled(self):
        for flags, enabled, gae in [([], False, False), (['--normalize-advantages'], True, False),
                                    (['--gae', '--normalize-advantages'], True, True)]:
            with self.subTest(flags=flags):
                with patch('sys.argv', ['train.py', *flags]), patch.object(train, 'train') as training:
                    train.main()
                self.assertEqual(training.call_args.kwargs['normalize_advantages'], enabled)
                self.assertEqual(training.call_args.kwargs['use_gae'], gae)

    def test_training_saves_both_switch_states_in_config_and_checkpoint(self):
        info = SimpleNamespace(vector_observations=np.zeros(shape=(20, 33), dtype=np.float32),
                               rewards=[0.1] * 20, local_done=[False] * 20, max_reached=[False] * 20)
        class Environment:
            brain_names = ['brain']
            brains = {'brain': SimpleNamespace(vector_action_space_size=4)}
            def reset(self, train_mode):
                return {'brain': info}
            def step(self, vector_action):
                return {'brain': info}
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for enabled in [False, True]:
                with patch.object(train, 'ROOT', root), patch.object(train, 'make_env', return_value=Environment()):
                    train.train(iterations=1, rollout_steps=2, run_name=f'normalized_{enabled}',
                                runs_dir=root / 'runs', normalize_advantages=enabled)
                folder = root / 'runs' / f'normalized_{enabled}'
                config = json.loads((folder / 'config.json').read_text())
                checkpoint = torch.load(f=config['checkpoint'], map_location='cpu', weights_only=True)
                self.assertEqual(config['features']['normalize_advantages'], enabled)
                self.assertEqual(checkpoint['features']['normalize_advantages'], enabled)
                self.assertEqual({p.name for p in folder.iterdir()}, {'config.json', 'episodes.csv', 'updates.csv'})


if __name__ == '__main__':
    unittest.main()
