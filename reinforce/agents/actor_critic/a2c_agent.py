# -*- coding: utf-8 -*-
"""
A2C Agent implementation.
"""

from typing import Any, Dict, Tuple

import tensorflow as tf
from numpy import ndarray

from reinforce.agents.actor_critic.actor_critic_agent import ActorCriticAgent
from reinforce.configs.models import A2CConfig
from reinforce.utils.preprocessing import preprocess_observation


class A2CAgent(ActorCriticAgent):  # Inherit from the new base class
    """
    A2C (Advantage Actor-Critic) agent implementation.

    This class implements the Actor-Critic interface for the A2C algorithm.
    """

    def __init__(self, action_space: int, hyperparameters: A2CConfig):
        """
        Initialize the A2C agent.

        Parameters
        ----------
        action_space : int
            Number of possible actions. Should match `hyperparameters.action_space`.
        hyperparameters : A2CConfig
            Pydantic model containing A2C hyperparameters.
        """
        super().__init__(action_space=action_space, hyperparameters=hyperparameters, agent_name="A2CAgent")

        # ##: The base class init already assigns self.hyperparameters, but we refine the type hint here.
        self.hyperparameters: A2CConfig = hyperparameters

    # ##: act method remains specific to A2C's action selection logic.
    def act(self, observation: ndarray, training: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action based on the current observation.

        Parameters
        ----------
        observation : np.ndarray
            The current observation from the environment.
        training : bool, optional
            Whether the agent is in training mode, by default ``True``.

        Returns
        -------
        Tuple[int, Dict[str, Any]]
            The selected action and additional information about the decision.
        """
        observation = preprocess_observation(observation)

        # ##: Add batch dimension if necessary and pass through model.
        if len(observation.shape) == 3:
            observation = tf.expand_dims(observation, axis=0)
        action_logits, value = self._model(observation)

        if not training:
            action = tf.argmax(action_logits[0]).numpy()
        else:
            action = tf.random.categorical(action_logits, 1)[0, 0].numpy()

        action_probs = tf.nn.softmax(action_logits)

        return action, {
            "value": value[0, 0].numpy(),
            "action_probs": action_probs[0].numpy(),
            "action_logits": action_logits[0].numpy(),
        }

    def learn(self, experience_batch: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]) -> Dict[str, Any]:
        """
        Update the agent based on a batch of experiences provided as tensors.

        Parameters
        ----------
        experience_batch : Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
            A tuple containing batches of tensors:
            (observations, actions, rewards, next_observations, dones).
            Observations and next_observations are expected to be preprocessed.

        Returns
        -------
        Dict[str, Any]
            Dictionary of learning metrics.
        """
        # ##: Unpack tensors from the dataset batch.
        # ##: Data is already preprocessed and in tensor format.
        observations, actions, rewards, next_observations, dones = experience_batch

        # ##: Compute returns and advantages using the input tensors directly.
        returns, advantages = self._compute_returns_and_advantages(rewards, dones, observations, next_observations)

        # ##: Perform one training step.
        metrics = self._train_step(observations, actions, returns, advantages)

        return metrics

    def _compute_returns_and_advantages(
        self, rewards: tf.Tensor, dones: tf.Tensor, observations: tf.Tensor, next_observations: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute returns and advantages for the given experiences.

        Parameters
        ----------
        rewards : tf.Tensor
            Batch of rewards.
        dones : tf.Tensor
            Batch of episode termination flags.
        observations : tf.Tensor
            Batch of observations.
        next_observations : tf.Tensor
            Batch of next observations.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Tuple of (returns, advantages).
        """
        # ##: Get value estimates for current and next observations.
        _, values = self._model(observations)
        _, next_values = self._model(next_observations)

        # ##: Reshape values for easier calculations.
        values = tf.squeeze(values)
        next_values = tf.squeeze(next_values)  # Should be float16 if mixed precision

        # ##: Ensure rewards and dones match the model's compute dtype (e.g., float16)
        compute_dtype = self._model.compute_dtype
        rewards = tf.cast(rewards, compute_dtype)
        dones = tf.cast(dones, compute_dtype)

        # ##: Calculate returns using TD(lambda) with lambda=0 (i.e., TD(0)).
        # ##: Ensure discount factor is also cast if necessary (though usually float32 is fine here)
        discount_factor = tf.cast(self.hyperparameters.discount_factor, compute_dtype)
        returns = rewards + discount_factor * next_values * (tf.cast(1.0, compute_dtype) - dones)

        # ##: Calculate advantages.
        advantages = returns - values  # Now returns and values should both be compute_dtype

        return returns, advantages

    def _calculate_losses(
        self,
        action_logits: tf.Tensor,
        values: tf.Tensor,
        actions: tf.Tensor,
        returns: tf.Tensor,
        advantages: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Calculate policy, value, entropy, and total losses."""
        # ##: Select the action probabilities for the actions that were taken.
        indices = tf.range(tf.shape(actions)[0])
        action_indices = tf.stack([indices, actions], axis=1)
        action_log_probs = tf.math.log(tf.nn.softmax(action_logits) + 1e-10)
        selected_action_log_probs = tf.gather_nd(action_log_probs, action_indices)

        # ##: Calculate policy loss (negative for gradient ascent).
        policy_loss = -tf.reduce_mean(selected_action_log_probs * advantages)

        # ##: Calculate value loss.
        value_loss = self.hyperparameters.value_coef * tf.reduce_mean(tf.square(returns - values))

        # ##: Calculate entropy loss (for exploration).
        action_probs = tf.nn.softmax(action_logits)
        entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
        entropy_loss = -self.hyperparameters.entropy_coef * tf.reduce_mean(entropy)

        # ##: Calculate total loss.
        total_loss = policy_loss + value_loss + entropy_loss

        return total_loss, policy_loss, value_loss, entropy_loss, tf.reduce_mean(entropy)

    @tf.function
    def _train_step(
        self, observations: tf.Tensor, actions: tf.Tensor, returns: tf.Tensor, advantages: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Perform one training step.

        Parameters
        ----------
        observations : tf.Tensor
            Batch of observations.
        actions : tf.Tensor
            Batch of actions taken.
        returns : tf.Tensor
            Batch of returns.
        advantages : tf.Tensor
            Batch of advantages.

        Returns
        -------
        Dict[str, tf.Tensor]
            Dictionary of training metrics.
        """
        with tf.GradientTape() as tape:
            # ##: Forward pass.
            action_logits, values = self._model(observations)
            values = tf.squeeze(values)

            # ##: Calculate losses using the helper method.
            total_loss, policy_loss, value_loss, entropy_loss, entropy = self._calculate_losses(
                action_logits, values, actions, returns, advantages
            )

        # ##: Calculate gradients and apply them.
        gradients = tape.gradient(total_loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "entropy": tf.reduce_mean(entropy),
        }

    def _load_specific_hyperparameters(self, config: Dict[str, Any]) -> A2CConfig:
        """
        Load specific hyperparameters for the A2C agent.
        """
        return A2CConfig(**config)
