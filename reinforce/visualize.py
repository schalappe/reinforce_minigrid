# -*- coding: utf-8 -*-
"""Visualize play."""
from typing import List

import numpy as np
import tensorflow as tf
from game import MazeGame
from PIL import Image
from reinforce.addons import preprocess_input


def render_episode(env: MazeGame, model: tf.keras.Model, max_steps: int = 100) -> List:
    """Play and render episode game."""
    state = env.reset()
    images = [Image.fromarray(state.astype(np.uint8))]
    state = tf.constant(state, dtype=tf.float32)

    for _ in range(1, max_steps + 1):
        state = tf.expand_dims(preprocess_input(state), 0)
        action_probs, _ = model(state)
        action = np.argmax(np.squeeze(action_probs))

        state, _, done = env.step(action)
        state = tf.constant(state, dtype=tf.float32)
        images.append(Image.fromarray(state.numpy().astype(np.uint8)))

        if done:
            break

    return images


if __name__ == "__main__":
    from os.path import abspath, dirname, join

    # ##: Necessary.
    game = MazeGame()
    a2c_model = tf.keras.models.load_model(join(dirname(dirname(abspath(__file__))), "zoo", "a2c_model_1686462486"))

    # ##: Save GIF image.
    all_images = render_episode(game, a2c_model)
    image_file = join(dirname(dirname(abspath(__file__))), "zoo", "BabyAI-GoToObjMaze-v0.tif")
    all_images[0].save(image_file, save_all=True, append_images=all_images[1:], loop=0, duration=1)
