"""
Random Network Distillation (RND) for intrinsic motivation.

RND provides exploration bonus by measuring prediction error between:
- A fixed random target network (never trained)
- A predictor network that learns to match the target

Novel states produce high prediction error = high intrinsic reward.
This encourages exploration without explicit state counting.

Reference: Burda et al., "Exploration by Random Network Distillation" (2018)
"""

import numpy as np
import tensorflow as tf
from keras import Model, layers

from reinforce.core.network_utils import get_orthogonal_initializer


def build_rnd_networks(input_shape: tuple, feature_dim: int = 512) -> tuple[Model, Model]:
    """
    Build the RND target and predictor networks.

    Both networks map observations to fixed-size feature vectors.
    The target network is frozen; the predictor learns to match its outputs.
    The predictor has additional capacity (extra dense layers) to learn the mapping.

    Parameters
    ----------
    input_shape : tuple
        Shape of input observations (height, width, channels).
    feature_dim : int, optional
        Dimension of the output feature vector. Default is 512.

    Returns
    -------
    tuple[Model, Model]
        Target network (fixed) and predictor network (trainable).
    """

    def build_cnn_base(inputs: layers.Input, name_prefix: str) -> layers.Layer:
        """Shared CNN architecture for feature extraction."""
        x = layers.Rescaling(scale=1.0 / 255.0, name=f"{name_prefix}_normalize")(inputs)

        x = layers.Conv2D(
            32,
            kernel_size=8,
            strides=4,
            padding="same",
            activation="leaky_relu",
            kernel_initializer=get_orthogonal_initializer(scale=2**0.5),
            name=f"{name_prefix}_conv1",
        )(x)

        x = layers.Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            activation="leaky_relu",
            kernel_initializer=get_orthogonal_initializer(scale=2**0.5),
            name=f"{name_prefix}_conv2",
        )(x)

        x = layers.Conv2D(
            64,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="leaky_relu",
            kernel_initializer=get_orthogonal_initializer(scale=2**0.5),
            name=f"{name_prefix}_conv3",
        )(x)

        return layers.Flatten(name=f"{name_prefix}_flatten")(x)

    # ##>: Build target network (simple: CNN -> single dense layer).
    target_inputs = layers.Input(shape=input_shape, name="target_input")
    target_features = build_cnn_base(target_inputs, "target")
    target_output = layers.Dense(
        feature_dim,
        kernel_initializer=get_orthogonal_initializer(scale=2**0.5),
        name="target_features",
    )(target_features)
    target_network = Model(inputs=target_inputs, outputs=target_output, name="RNDTarget")

    # ##>: Build predictor network (CNN -> hidden layer -> output layer for more capacity).
    predictor_inputs = layers.Input(shape=input_shape, name="predictor_input")
    predictor_features = build_cnn_base(predictor_inputs, "predictor")
    x = layers.Dense(
        feature_dim,
        activation="relu",
        kernel_initializer=get_orthogonal_initializer(scale=2**0.5),
        name="predictor_hidden",
    )(predictor_features)
    predictor_output = layers.Dense(
        feature_dim,
        kernel_initializer=get_orthogonal_initializer(scale=1.0),
        name="predictor_output",
    )(x)
    predictor_network = Model(inputs=predictor_inputs, outputs=predictor_output, name="RNDPredictor")

    # ##>: Freeze target network weights.
    target_network.trainable = False

    return target_network, predictor_network


class RNDModule:
    """
    Random Network Distillation module for computing intrinsic rewards.

    Maintains running statistics for reward normalization and provides
    methods for computing intrinsic rewards and training the predictor.
    """

    def __init__(
        self,
        input_shape: tuple,
        feature_dim: int = 512,
        learning_rate: float = 1e-4,
        intrinsic_reward_scale: float = 1.0,
        update_proportion: float = 0.25,
    ):
        """
        Initialize the RND module.

        Parameters
        ----------
        input_shape : tuple
            Shape of input observations.
        feature_dim : int, optional
            RND feature dimension. Default is 512.
        learning_rate : float, optional
            Learning rate for predictor network. Default is 1e-4.
        intrinsic_reward_scale : float, optional
            Scale factor for intrinsic rewards. Default is 1.0.
        update_proportion : float, optional
            Proportion of batch to use for RND updates (to save compute). Default is 0.25.
        """
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.update_proportion = update_proportion

        # ##>: Build networks.
        self.target_network, self.predictor_network = build_rnd_networks(input_shape, feature_dim)

        # ##>: Optimizer for predictor network only.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-5)

        # ##>: Running statistics for reward normalization (Welford's algorithm).
        self.reward_mean = 0.0
        self.reward_m2 = 0.0
        self.reward_count = 1e-4

    def compute_intrinsic_reward(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute intrinsic reward as prediction error.

        Parameters
        ----------
        observations : np.ndarray
            Batch of observations. Shape: (batch_size, height, width, channels).

        Returns
        -------
        np.ndarray
            Intrinsic rewards (normalized). Shape: (batch_size,).
        """
        obs_tensor = tf.convert_to_tensor(observations, dtype=tf.float32)

        # ##>: Get target and predictor features.
        target_features = self.target_network(obs_tensor, training=False)
        predictor_features = self.predictor_network(obs_tensor, training=False)

        # ##>: Prediction error as intrinsic reward.
        prediction_error = tf.reduce_mean(tf.square(target_features - predictor_features), axis=1)
        intrinsic_reward = prediction_error.numpy()

        # ##>: Update running statistics and normalize.
        self._update_reward_stats(intrinsic_reward)
        normalized_reward = self._normalize_reward(intrinsic_reward)

        return normalized_reward * self.intrinsic_reward_scale

    def _update_reward_stats(self, rewards: np.ndarray) -> None:
        """Update running mean and variance using parallel Welford's algorithm."""
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)

        delta = batch_mean - self.reward_mean
        total_count = self.reward_count + batch_count

        self.reward_mean += delta * batch_count / total_count
        batch_m2 = batch_var * batch_count
        self.reward_m2 += batch_m2 + delta**2 * self.reward_count * batch_count / total_count
        self.reward_count = total_count

    def _normalize_reward(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards using running statistics."""
        variance = self.reward_m2 / self.reward_count if self.reward_count > 0 else 1.0
        return (rewards - self.reward_mean) / (np.sqrt(variance) + 1e-8)

    def train_step(self, observations: np.ndarray) -> float:
        """
        Train the predictor network on a batch of observations.

        Parameters
        ----------
        observations : np.ndarray
            Batch of observations.

        Returns
        -------
        float
            Mean prediction loss.
        """
        # ##>: Subsample batch for efficiency.
        batch_size = len(observations)
        num_samples = max(1, int(batch_size * self.update_proportion))
        indices = np.random.choice(batch_size, num_samples, replace=False)
        obs_subset = observations[indices]

        obs_tensor = tf.convert_to_tensor(obs_subset, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_features = self.target_network(obs_tensor, training=False)
            predictor_features = self.predictor_network(obs_tensor, training=True)
            loss = tf.reduce_mean(tf.square(target_features - predictor_features))

        gradients = tape.gradient(loss, self.predictor_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.predictor_network.trainable_variables))

        return float(loss.numpy())

    def get_stats(self) -> dict:
        """Return current reward normalization statistics."""
        variance = self.reward_m2 / self.reward_count if self.reward_count > 0 else 1.0
        return {
            "rnd_reward_mean": self.reward_mean,
            "rnd_reward_std": np.sqrt(variance),
            "rnd_sample_count": self.reward_count,
        }
