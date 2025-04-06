# -*- coding: utf-8 -*-
"""
PPO Agent implementation.
"""

from typing import Any, Dict, Tuple

import tensorflow as tf
from numpy import ndarray

from reinforce.agents.actor_critic.actor_critic_agent import (
    ActorCriticAgent,
    HyperparameterConfig,
)
from reinforce.configs.models import PPOConfig
from reinforce.utils.preprocessing import preprocess_observation


class PPOAgent(ActorCriticAgent):  # Inherit from the new base class
    """
    PPO (Proximal Policy Optimization) agent implementation.

    This class implements the Actor-Critic interface for the PPO algorithm.
    It uses a clipped surrogate objective and optionally an adaptive KL penalty.
    """

    def __init__(self, action_space: int, hyperparameters: PPOConfig):
        """
        Initialize the PPO agent.

        Parameters
        ----------
        action_space : int
            Number of possible actions.
        hyperparameters : PPOConfig
            Pydantic model containing hyperparameters for PPO.
        """
        # ##: Call the base class initializer.
        super().__init__(action_space=action_space, hyperparameters=hyperparameters, agent_name="PPOAgent")

        # ##: Ensure type hint reflects the specific config for this agent.
        self.hyperparameters: PPOConfig = hyperparameters

        # ##: PPO specific: Adaptive KL penalty coefficient (if used).
        self.kl_coeff = tf.Variable(self.hyperparameters.initial_kl_coeff, trainable=False, dtype=tf.float32)

    def act(self, observation: ndarray, training: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action, get value estimate, and log probability for the current observation.

        Parameters
        ----------
        observation : np.ndarray
            The current observation from the environment.
        training : bool, optional
            Whether the agent is in training mode, by default ``True``.

        Returns
        -------
        Tuple[int, Dict[str, Any]]
            Tuple containing:
            - action (int): The selected action.
            - agent_info (Dict[str, Any]): Dictionary containing:
                - value (float): The value estimate V(s).
                - log_prob (float): The log probability log(pi(a|s)).
                - action_probs (np.ndarray): Full action probability distribution.
                - action_logits (np.ndarray): Raw action logits.
        """
        observation = preprocess_observation(observation)
        if len(observation.shape) == 3:
            observation = tf.expand_dims(observation, axis=0)

        # ##: Get policy logits and value from the model.
        action_logits, value = self._model(observation, training=training)  # Pass training flag
        action_probs = tf.nn.softmax(action_logits)

        # ##: Sample action from the distribution.
        action_dist = tf.compat.v1.distributions.Categorical(logits=action_logits)
        action_tensor = action_dist.sample()
        action = action_tensor[0].numpy()

        # ##: Calculate log probability of the sampled action.
        log_prob = action_dist.log_prob(action_tensor)[0].numpy()

        return action, {
            "value": value[0, 0].numpy(),
            "log_prob": log_prob,
            "action_probs": action_probs[0].numpy(),
            "action_logits": action_logits[0].numpy(),
        }

    def learn(self, experience_batch: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        """
        Update the agent based on a batch of experiences sampled from the RolloutBuffer.

        This method performs one gradient update step using the PPO objective.

        Parameters
        ----------
        experience_batch : Dict[str, tf.Tensor]
            A dictionary containing batches of tensors from the RolloutBuffer:
            'observations', 'actions', 'advantages', 'returns', 'log_probs_old', 'values_old'.

        Returns
        -------
        Dict[str, Any]
            Dictionary of learning metrics (losses, KL divergence, etc.).
        """
        observations = experience_batch["observations"]
        actions = experience_batch["actions"]
        advantages = experience_batch["advantages"]
        returns = experience_batch["returns"]
        log_probs_old = experience_batch["log_probs_old"]
        # values_old = experience_batch['values_old'] # May not be needed directly in loss

        metrics = self._train_step(observations, actions, advantages, returns, log_probs_old)

        # ##: Adjust KL coefficient if adaptive KL penalty is enabled.
        if self.hyperparameters.use_adaptive_kl:
            kl_div = metrics.get("kl_divergence", 0.0)
            if kl_div > 1.5 * self.hyperparameters.target_kl:
                self.kl_coeff.assign(self.kl_coeff.read_value() * 2.0)
            elif kl_div < self.hyperparameters.target_kl / 1.5:
                self.kl_coeff.assign(self.kl_coeff.read_value() / 2.0)
            metrics["kl_coeff"] = self.kl_coeff.read_value().numpy()

        return metrics

    @tf.function
    def _train_step(
        self,
        observations: tf.Tensor,
        actions: tf.Tensor,
        advantages: tf.Tensor,
        returns: tf.Tensor,
        log_probs_old: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """
        Perform one PPO training step (gradient update).

        Parameters
        ----------
        observations : tf.Tensor
            Batch of observations.
        actions : tf.Tensor
            Batch of actions taken.
        advantages : tf.Tensor
            Batch of advantages (e.g., GAE).
        returns : tf.Tensor
            Batch of returns (e.g., GAE returns).
        log_probs_old : tf.Tensor
            Batch of log probabilities from the policy used during rollout collection.

        Returns
        -------
        Dict[str, tf.Tensor]
            Dictionary of training metrics for this step.
        """
        with tf.GradientTape() as tape:
            # ##: Forward pass to get current policy logits and value estimates.
            action_logits, values = self._model(observations, training=True)
            values = tf.squeeze(values)  # Remove last dim

            # ##: Calculate current log probabilities and entropy.
            action_dist = tf.compat.v1.distributions.Categorical(logits=action_logits)
            log_probs_new = action_dist.log_prob(actions)
            entropy = tf.reduce_mean(action_dist.entropy())

            # ##: Calculate the probability ratio r_t(theta) = exp(log_prob_new - log_prob_old).
            ratio = tf.exp(log_probs_new - log_probs_old)

            # ##: Calculate the clipped surrogate objective.
            clip_range = self.hyperparameters.clip_range
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss_unclipped = ratio * advantages
            policy_loss_clipped = clipped_ratio * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(policy_loss_unclipped, policy_loss_clipped))

            # ##: Calculate value loss (clipped value loss optional, standard MSE here).
            value_loss = 0.5 * tf.reduce_mean(tf.square(returns - values))

            # ##: Calculate entropy bonus (negative for gradient ascent).
            entropy_loss = -self.hyperparameters.entropy_coef * entropy

            # ##: Calculate KL divergence (for adaptive penalty or logging).
            # ##: Ensure log_probs_old is detached if it came from tape context previously.
            kl_divergence = tf.reduce_mean(log_probs_old - log_probs_new)

            # ##: Calculate total loss.
            total_loss = policy_loss + self.hyperparameters.value_coef * value_loss + entropy_loss

            # ##: Add adaptive KL penalty if enabled.
            if self.hyperparameters.use_adaptive_kl:
                kl_penalty = self.kl_coeff.read_value() * tf.maximum(
                    0.0, kl_divergence - self.hyperparameters.target_kl
                )
                total_loss += kl_penalty

        # ##: Calculate gradients and apply them.
        gradients = tape.gradient(total_loss, self._model.trainable_variables)
        if self.hyperparameters.max_grad_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.hyperparameters.max_grad_norm)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy,
            "kl_divergence": kl_divergence,
            "advantages": tf.reduce_mean(advantages),
            "returns": tf.reduce_mean(returns),
            "ratio": tf.reduce_mean(ratio),
        }

    def load(self, path: str) -> None:
        """
        Load the PPO agent from the specified path.

        Overrides the base load method to provide the specific config class,
        custom objects for model loading, and reinitialize PPO-specific state.

        Parameters
        ----------
        path : str
            Directory path to load the agent from.
        """
        super().load(path)
        self.kl_coeff = tf.Variable(self.hyperparameters.initial_kl_coeff, trainable=False, dtype=tf.float32)

    def _load_specific_hyperparameters(self, config: Dict[str, Any]) -> HyperparameterConfig:
        return PPOConfig(**config)
