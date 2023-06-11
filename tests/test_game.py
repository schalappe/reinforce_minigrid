# -*- coding: utf-8 -*-
"""Test game."""
import unittest
from reinforce.game import MazeGame
from numpy import ndarray
import tensorflow as tf


class TestMazeGame(unittest.TestCase):

    def test_observation(self):
        game = MazeGame()
        observation = game.reset()
        print(observation.shape)
        self.assertIsInstance(observation, ndarray)

    def test_action_space(self):
        game = MazeGame()
        self.assertEqual(7, game.environment.action_space.n)

    def test_env_step(self):
        game = MazeGame()
        game.reset()
        _, reward, done = game.env_step(action=0)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, int)

    def test_step(self):
        game = MazeGame()
        game.reset()
        _, reward, done = game.step(action=tf.constant(0))
        self.assertIsInstance(reward, tf.Tensor)
        self.assertIsInstance(done, tf.Tensor)


if __name__ == '__main__':
    unittest.main()
