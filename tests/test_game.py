# -*- coding: utf-8 -*-
"""
Test game module for the MazeGame class.
"""

from unittest import TestCase, main
from unittest.mock import patch

import tensorflow as tf
from gymnasium.spaces import Discrete
from numpy import array, ndarray, zeros

from reinforce.game import MazeGame


class TestMazeGame(TestCase):
    """Test cases for the MazeGame class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.game = MazeGame()
        self.initial_observation = self.game.reset()

    def test_observation_shape_and_type(self):
        """Test that reset returns observation with correct shape and type."""
        observation = self.game.reset()
        self.assertIsInstance(observation, ndarray)
        self.assertEqual(len(observation.shape), 3)

    def test_action_space(self):
        """Test that the action space has the expected number of actions."""
        self.assertEqual(7, self.game.environment.action_space.n)
        self.assertIsInstance(self.game.environment.action_space, Discrete)

    def test_env_step_range_of_actions(self):
        """Test env_step with different actions."""
        self.game.reset()

        for action in range(7):
            self.game.reset()
            observation, reward, done = self.game.env_step(action=array(action))
            self.assertIsInstance(observation, ndarray)
            self.assertIsInstance(reward, ndarray)
            self.assertIsInstance(done, ndarray)

    def test_step_tensorflow_conversion(self):
        """Test that step method converts inputs/outputs to TensorFlow tensors."""
        self.game.reset()
        observation, reward, done = self.game.step(action=tf.constant(0))

        self.assertIsInstance(observation, tf.Tensor)
        self.assertIsInstance(reward, tf.Tensor)
        self.assertIsInstance(done, tf.Tensor)

        self.assertEqual(reward.shape, ())
        self.assertEqual(done.shape, ())

    def test_step_with_different_actions(self):
        """Test step method with different TensorFlow action values."""
        for action in range(7):
            self.game.reset()
            observation, reward, done = self.game.step(action=tf.constant(action))
            self.assertIsInstance(observation, tf.Tensor)
            self.assertIsInstance(reward, tf.Tensor)
            self.assertIsInstance(done, tf.Tensor)

    def test_episode_termination(self):
        """Test that the game can reach terminal state."""
        with patch.object(self.game, "env_step", return_value=(zeros((8, 8, 3)), 1.0, 1)):
            _, _, done = self.game.step(tf.constant(0))
            self.assertEqual(done.numpy(), 1)

    def test_reward_range(self):
        """Test that rewards are within expected range."""
        self.game.reset()
        _, reward, _ = self.game.step(tf.constant(0))
        self.assertGreaterEqual(reward.numpy(), -100)
        self.assertLessEqual(reward.numpy(), 100)


if __name__ == "__main__":
    main()
