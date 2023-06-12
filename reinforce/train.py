# -*- coding: utf-8 -*-
"""Train Actor-Critic model."""
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from os.path import abspath, dirname, join
from statistics import mean
from typing import Tuple

import hydra
import tensorflow as tf
from omegaconf import DictConfig
from tqdm import trange

from reinforce.game import MazeGame
from reinforce.losses import compute_loss
from reinforce.model import A2CModel
from reinforce.replay import get_expected_return
from reinforce.addons import preprocess_input


@dataclass
class A2CModelConfig:
    """Configuration for model."""

    gamma: float
    learning_rate: float


@dataclass
class EpisodeConfig:
    """Information for episode."""

    max_episode: int
    steps_per_episode: int


class TrainA2CModel:
    """Train an Actor-Critic model."""

    def __init__(self, model_config: A2CModelConfig, episode_config: EpisodeConfig):
        self.game = MazeGame()
        self.model = A2CModel(action_space=self.game.environment.action_space.n)
        self.episodes = episode_config.max_episode
        self.steps = episode_config.steps_per_episode
        self.gamma = model_config.gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=model_config.learning_rate)

    def run_episode(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Runs a single episode to collect training data.

        Parameters
        ----------
        initial_state: tf.Tensor
            Initial state

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            Action probabilities, values, rewards
        """
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for step in tf.range(self.steps):
            state = tf.expand_dims(preprocess_input(state), 0)

            # ##: Run the model and to get action probabilities and critic value.
            action_logits_t, value = self.model(state)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # ##: Store critic values and log probability of the action chosen.
            values = values.write(step, tf.squeeze(value))
            action_probs = action_probs.write(step, action_probs_t[0, action])

            # ##: Apply action to the environment to get next state and reward
            state, reward, done = self.game.step(action)
            state.set_shape(initial_state_shape)

            # ##: Store reward.
            rewards = rewards.write(step, reward)

            if tf.cast(done, tf.bool):
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards

    @tf.function
    def train_step(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Runs a model training step.

        Parameters
        ----------
        initial_state: tf.Tensor
            Initial state

        Returns
        -------
        tf.Tensor
            Episode reward
        """
        with tf.GradientTape() as tape:
            # ##: Run the model for one episode to collect training data.
            action_probs, values, rewards = self.run_episode(initial_state)

            # ##: Calculate the expected returns.
            returns = get_expected_return(rewards, self.gamma)

            # ##: Convert training data to appropriate TF tensor shapes.
            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

            # ##: Calculate the loss values to update our network.
            loss = compute_loss(action_probs, values, returns)

        # ##: Compute the gradients from the loss.
        grads = tape.gradient(loss, self.model.trainable_variables)

        # ##: Apply the gradients to the model's parameters.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward, loss

    def train(self):
        """Train model."""
        # ##: Keep the last episodes reward
        episodes_reward = deque(maxlen=100)

        step = trange(self.episodes)
        for _ in step:
            # ##: Initial state.
            initial_state = self.game.reset()
            initial_state = tf.constant(initial_state, dtype=tf.float32)

            # ##: One step training.
            episode_reward, loss = self.train_step(initial_state)
            episode_reward = float(episode_reward)
            loss = float(loss)

            episodes_reward.append(episode_reward)
            running_reward = mean(episodes_reward)

            step.set_postfix(current_loss=loss, episode_reward=episode_reward, running_reward=running_reward)

        # ##: Save model.
        self.model.save(
            join(dirname(dirname(abspath(__file__))), "zoo", f"a2c_model_{int(datetime.now().timestamp())}")
        )


@hydra.main(version_base=None, config_path="config", config_name="a2c_model")
def train_model(config: DictConfig):
    """
    Train model.

    Parameters
    ----------
    config: DictConfig
        Configuration for training
    """
    # ##: Get configuration.
    model_config = A2CModelConfig(gamma=config.model.gamma, learning_rate=config.model.learning_rate)
    episode_config = EpisodeConfig(
        max_episode=config.episode.max_episode, steps_per_episode=config.episode.steps_per_episode
    )

    # ##: Model.
    a2c_model = TrainA2CModel(model_config=model_config, episode_config=episode_config)

    # ##: Train.
    a2c_model.train()


if __name__ == "__main__":
    train_model()
