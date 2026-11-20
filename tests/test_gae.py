"""Numerical checks for GAE across terminal, timeout and rollout boundaries."""
import unittest

import numpy as np

from ppo_agent import PPOAgent, RolloutBuffer


def example_rollout():
    buffer = RolloutBuffer(rollout_steps=3, num_agents=2, state_size=1, action_size=1)
    rewards = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32)
    next_values = np.array([[1, 1], [1, 8], [0, 4]], dtype=np.float32)
    terminated = np.array([[False, False], [False, False], [True, False]])
    ended = np.array([[False, False], [False, True], [True, False]])
    for t in range(3):
        buffer.add(
            states=np.zeros(shape=(2, 1), dtype=np.float32),
            latent_actions=np.zeros(shape=(2, 1), dtype=np.float32),
            rewards=rewards[t], values=np.ones(shape=2, dtype=np.float32),
            log_probs=np.zeros(shape=2, dtype=np.float32),
            next_values=next_values[t], terminated=terminated[t], ended=ended[t],
        )
    return buffer


class GAETests(unittest.TestCase):
    def test_lambda_one_matches_baseline_with_all_boundary_types(self):
        buffer = example_rollout()
        baseline = buffer.compute_advantages(gamma=0.5)
        gae = buffer.compute_advantages(gamma=0.5, use_gae=True, gae_lambda=1.0)
        np.testing.assert_allclose(actual=gae[0], desired=baseline[0])
        np.testing.assert_allclose(actual=gae[1], desired=[[2.75, 22], [3.5, 24], [3, 32]])

    def test_lambda_zero_is_one_step_td_error(self):
        advantages, returns = example_rollout().compute_advantages(
            gamma=0.5, use_gae=True, gae_lambda=0.0,
        )
        np.testing.assert_allclose(actual=advantages, desired=[[0.5, 9.5], [1.5, 23], [2, 31]])
        np.testing.assert_allclose(actual=returns, desired=advantages + 1)

    def test_intermediate_lambda_does_not_cross_episode_reset(self):
        advantages, returns = example_rollout().compute_advantages(
            gamma=0.5, use_gae=True, gae_lambda=0.8,
        )
        np.testing.assert_allclose(actual=advantages, desired=[[1.42, 18.7], [2.3, 23], [2, 31]], rtol=1e-6)
        np.testing.assert_allclose(actual=returns, desired=[[2.42, 19.7], [3.3, 24], [3, 32]], rtol=1e-6)

    def test_agent_gae_update_produces_finite_parameters(self):
        agent = PPOAgent(state_size=1, action_size=1, use_gae=True,
                         gae_lambda=0.8, update_epochs=1, minibatch_size=4)
        metrics = agent.update(rollout=example_rollout())
        self.assertEqual(metrics['optimizer_steps'], 2)
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        for parameter in agent.model.parameters():
            self.assertTrue(np.isfinite(parameter.detach().numpy()).all())


if __name__ == '__main__':
    unittest.main()
