# -*- coding: utf-8 -*-
"""Test model."""
import unittest
from reinforce.model import A2CModel
import numpy as np
import tensorflow as tf


class TestA2CModel(unittest.TestCase):
    def test_creation(self):
        # ##: Model.
        model = A2CModel(action_space=7)
        model.build(input_shape=(1, 176, 176, 3))
        model.summary()

    def test_outputs(self):
        # ##: Model.
        model = A2CModel(action_space=7)

        # ##: Inputs.
        inputs = np.zeros((1, 176, 176, 3))
        logit, value = model(inputs)
        action = tf.random.categorical(logit, 1)[0, 0]

        # ##: Assert.
        self.assertIn(action.numpy(), list(range(7)))
        self.assertIsInstance(value.numpy()[0, 0], np.float32)


if __name__ == '__main__':
    unittest.main()
