# -*- coding: utf-8 -*-
"""
PPO Agent implementation.
"""

from typing import Any, Dict, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from keras import Model, optimizers
from numpy import ndarray

from reinforce.agents.actor_critic.ac_agent import ActorCriticAgent
from reinforce.configs.models.agent import PPOConfig
from reinforce.utils.preprocessing import preprocess_observation


class PPOAgent(ActorCriticAgent):
    """
    PPO (Proximal Policy Optimization) agent implementation.

    This class implements the Actor-Critic interface for the PPO algorithm. It uses a clipped surrogate objective
    and optionally an adaptive KL penalty. Requires the model to be injected during initialization.
    """

    def __init__(self, model: Model, hyperparameters: PPOConfig):
        """
        Initialize the PPO agent.

        Parameters
        ----------
        model : Model
            The Keras model instance (actor-critic network).
        hyperparameters : PPOConfig
            Pydantic model containing hyperparameters for PPO.
        """
        super().__init__(model=model, hyperparameters=hyperparameters, agent_name="PPOAgent")

        # ##: Refine the type hint for hyperparameters specific to PPO.
        self.hyperparameters: PPOConfig = hyperparameters

        # ##: PPO specific: Adaptive KL penalty coefficient.
        self._kl_coefficient = tf.Variable(self.hyperparameters.initial_kl_coeff, trainable=False, dtype=tf.float32)

        # ##: Initialize optimizer with optional LR scheduling.
        if self.hyperparameters.lr_schedule_enabled:
            initial_learning_rate = self.hyperparameters.learning_rate
            lr_schedule = optimizers.schedules.PolynomialDecay(
                initial_learning_rate,
                decay_steps=self.hyperparameters.max_total_steps,
                end_learning_rate=initial_learning_rate * self.hyperparameters.lr_decay_factor,
                power=1.0,
            )
            self._optimizer = optimizers.Adam(learning_rate=lr_schedule)

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
        action_logits, value = self._model(observation, training=training)
        action_probs = tf.nn.softmax(action_logits)

        # ##: Sample action from the distribution.
        action_dist = tfp.distributions.Categorical(logits=action_logits)
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

    def learn(self, experience_batch: Dict[str, tf.Tensor]) -> Tuple[Dict[str, Any], bool]:
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
        Tuple[Dict[str, Any], bool]
            Tuple containing:
            - Dictionary of learning metrics (losses, KL divergence, etc.).
            - Boolean flag indicating if training should continue for this epoch (True) or stop early (False).
        """
        observations = experience_batch["observations"]
        actions = experience_batch["actions"]
        advantages = experience_batch["advantages"]
        returns = experience_batch["returns"]
        log_probs_old = experience_batch["log_probs_old"]
        values_old = experience_batch["values_old"]

        # ##: Perform a single training step.
        metrics, should_continue = self._train_step(
            observations, actions, advantages, returns, log_probs_old, values_old
        )

        # ##: Adjust KL coefficient if adaptive KL penalty is enabled and we are continuing.
        if should_continue and self.hyperparameters.use_adaptive_kl:
            kl_div = metrics.get("kl_divergence", tf.constant(0.0, dtype=tf.float32))
            target_kl = tf.cast(self.hyperparameters.target_kl, dtype=tf.float32)

            if tf.greater(kl_div, 1.5 * target_kl):
                self._kl_coefficient.assign(self._kl_coefficient.read_value() * 2.0)
            elif tf.less(kl_div, target_kl / 1.5):
                self._kl_coefficient.assign(self._kl_coefficient.read_value() / 2.0)
            metrics["kl_coeff"] = self._kl_coefficient.read_value()

        return metrics, should_continue

    @tf.function
    def _train_step(
        self,
        observations: tf.Tensor,
        actions: tf.Tensor,
        advantages: tf.Tensor,
        returns: tf.Tensor,
        log_probs_old: tf.Tensor,
        values_old: tf.Tensor,
    ) -> Tuple[Dict[str, tf.Tensor], bool]:
        """
        Perform one PPO training step (gradient update).
        Returns metrics and a flag indicating if training should continue.

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
        values_old : tf.Tensor
            Batch of value estimates from the policy used during rollout collection.

        Returns
        -------
        Tuple[Dict[str, tf.Tensor], bool]
            Tuple containing:
            - Dictionary of training metrics for this step.
            - Boolean flag indicating if training should continue (True) or stop early (False).
        """
        should_continue = True
        with tf.GradientTape() as tape:
            # ##: Forward pass to get current policy logits and value estimates.
            action_logits, values = self._model(observations, training=True)
            values = tf.squeeze(values)  # Remove last dim

            # ##: Calculate current log probabilities and entropy.
            action_dist = tfp.distributions.Categorical(logits=action_logits)
            log_probs_new = action_dist.log_prob(actions)
            entropy = tf.reduce_mean(action_dist.entropy())

            # ##: Calculate the probability ratio r_t(theta) = exp(log_prob_new - log_prob_old).
            log_ratio = tf.clip_by_value(log_probs_new - log_probs_old, -10.0, 10.0)
            ratio = tf.clip_by_value(tf.exp(log_ratio), 1e-10, 1e10)

            # ##: Stabilize advantages.
            advantages_safe = tf.clip_by_value(advantages, -10.0, 10.0)

            # ##: Calculate the clipped surrogate objective.
            clip_range = self.hyperparameters.clip_range
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - clip_range, 1.0 + clip_range)

            # ##: Calculate policy loss.
            policy_loss_unclipped = ratio * advantages_safe
            policy_loss_clipped = clipped_ratio * advantages_safe
            policy_loss = -tf.reduce_mean(tf.minimum(policy_loss_unclipped, policy_loss_clipped))

            # ##: Calculate clipped value loss (PPO2-style).
            clip_range = self.hyperparameters.clip_range
            values_clipped = values_old + tf.clip_by_value(values - values_old, -clip_range, clip_range)
            value_loss_unclipped = tf.square(returns - values)
            value_loss_clipped = tf.square(returns - values_clipped)
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_unclipped, value_loss_clipped))

            # ##: Ensure entropy loss is well-behaved.
            entropy_bounded = tf.maximum(entropy, -5.0)
            entropy_loss = -self.hyperparameters.entropy_coef * entropy_bounded

            # ##: Calculate KL divergence between old and new policy.
            kl_divergence = tf.reduce_mean(log_probs_old - log_probs_new)

            # ##: Calculate total loss *before* the early stopping check.
            total_loss = policy_loss + self.hyperparameters.value_coef * value_loss + entropy_loss

            # ##: Add adaptive KL penalty if enabled.
            kl_penalty = tf.constant(0.0, dtype=tf.float32)
            if self.hyperparameters.use_adaptive_kl:
                kl_penalty = self._kl_coefficient.read_value() * tf.maximum(
                    0.0, kl_divergence - self.hyperparameters.target_kl
                )
                total_loss += kl_penalty

            # ##: Early stopping check based on KL divergence.
            target_kl = tf.cast(self.hyperparameters.target_kl, dtype=tf.float32)
            if tf.greater(kl_divergence, 2.0 * target_kl):
                should_continue = False

        # ##: Calculate and apply gradients only if we should continue.
        if should_continue:
            gradients = tape.gradient(total_loss, self._model.trainable_variables)
            if self.hyperparameters.max_grad_norm is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, self.hyperparameters.max_grad_norm)
            self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))

        # ##: Construct metrics dictionary regardless of early stopping.
        metrics = {
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

        if self.hyperparameters.use_adaptive_kl:
            metrics["kl_coeff"] = self._kl_coefficient.read_value()
            metrics["kl_penalty"] = kl_penalty

        return metrics, should_continue

    def load(self, path: str) -> None:
        """
        Load the PPO agent from the specified path.

        Overrides the base load method to reinitialize PPO-specific state after the base class handles
        loading the model and hyperparameters.

        Parameters
        ----------
        path : str
            Directory path to load the agent from.
        """
        super().load(path)
        self._kl_coefficient = tf.Variable(self.hyperparameters.initial_kl_coeff, trainable=False, dtype=tf.float32)

    def _load_specific_hyperparameters(self, config: Dict[str, Any]) -> PPOConfig:
        """
        Load PPO-specific hyperparameters from the configuration dictionary.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing the configuration parameters loaded from file.

        Returns
        -------
        PPOConfig
            PPO-specific Pydantic config model instance.
        """
        return PPOConfig(**config)
