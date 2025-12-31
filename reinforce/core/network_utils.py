"""
Shared neural network utilities for reinforcement learning.

Provides common building blocks used by both PPO and DQN:
- Orthogonal weight initialization (per RL best practices)
- IMPALA-style residual CNN blocks
- Noisy Dense layers for exploration (Rainbow DQN)
"""

import keras
import numpy as np
import tensorflow as tf
from keras import initializers, layers


def get_orthogonal_initializer(scale: float = 1.0) -> initializers.Orthogonal:
    """
    Create an orthogonal initializer with the specified gain.

    Per the "37 Implementation Details of PPO" paper:
    - Hidden layers use sqrt(2) gain
    - Policy output uses 0.01 gain
    - Value output uses 1.0 gain

    Parameters
    ----------
    scale : float
        The gain/scale for the orthogonal initialization.

    Returns
    -------
    initializers.Orthogonal
        Keras orthogonal initializer with specified gain.
    """
    return initializers.Orthogonal(gain=scale)


def _residual_block(x: layers.Layer, filters: int, name_prefix: str) -> layers.Layer:
    """
    Create an IMPALA-style residual block.

    IMPALA introduced residual blocks that significantly improve feature extraction for visual RL.

    Parameters
    ----------
    x : layers.Layer
        Input tensor.
    filters : int
        Number of convolutional filters.
    name_prefix : str
        Prefix for layer names.

    Returns
    -------
    layers.Layer
        Output tensor after residual connection.
    """
    residual = x

    out = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)
    out = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_initializer=get_orthogonal_initializer(scale=2**0.5),
        bias_initializer="zeros",
        name=f"{name_prefix}_conv1",
    )(out)

    out = layers.Activation("relu", name=f"{name_prefix}_relu2")(out)
    out = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_initializer=get_orthogonal_initializer(scale=2**0.5),
        bias_initializer="zeros",
        name=f"{name_prefix}_conv2",
    )(out)

    return layers.Add(name=f"{name_prefix}_add")([residual, out])


def _impala_cnn_block(x: layers.Layer, filters: int, name_prefix: str) -> layers.Layer:
    """
    Create an IMPALA CNN block with downsampling and residual connections.

    Parameters
    ----------
    x : layers.Layer
        Input tensor.
    filters : int
        Number of convolutional filters.
    name_prefix : str
        Prefix for layer names.

    Returns
    -------
    layers.Layer
        Output tensor.
    """
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        kernel_initializer=get_orthogonal_initializer(scale=2**0.5),
        bias_initializer="zeros",
        name=f"{name_prefix}_conv_init",
    )(x)

    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name=f"{name_prefix}_pool")(x)
    x = _residual_block(x, filters, f"{name_prefix}_res1")
    x = _residual_block(x, filters, f"{name_prefix}_res2")

    return x


def build_impala_cnn(inputs: layers.Input, name_prefix: str = "impala") -> layers.Layer:
    """
    Build an IMPALA-style CNN feature extractor.

    This architecture was shown to outperform Nature CNN on visual RL tasks.

    Parameters
    ----------
    inputs : layers.Input
        Input layer (expects image observations).
    name_prefix : str, optional
        Prefix for layer names. Default is "impala".

    Returns
    -------
    layers.Layer
        Flattened feature tensor ready for dense layers.
    """
    x = layers.Rescaling(scale=1.0 / 255.0, name=f"{name_prefix}_normalize")(inputs)
    x = _impala_cnn_block(x, filters=16, name_prefix=f"{name_prefix}_block1")
    x = _impala_cnn_block(x, filters=32, name_prefix=f"{name_prefix}_block2")
    x = _impala_cnn_block(x, filters=32, name_prefix=f"{name_prefix}_block3")
    x = layers.Activation("relu", name=f"{name_prefix}_final_relu")(x)
    x = layers.Flatten(name=f"{name_prefix}_flatten")(x)

    return x


@keras.saving.register_keras_serializable(package="reinforce")
class NoisyDense(layers.Layer):
    """
    Factorized Noisy Dense layer for exploration in Rainbow DQN.

    Instead of epsilon-greedy exploration, noisy networks add learnable noise to network weights.
    The network learns when and how much to explore through backpropagation.

    Reference: Fortunato et al., "Noisy Networks for Exploration" (2018)
    """

    def __init__(
        self,
        units: int,
        sigma_init: float = 0.5,
        activation: str | None = None,
        use_bias: bool = True,
        **kwargs,
    ):
        """
        Initialize noisy dense layer.

        Parameters
        ----------
        units : int
            Number of output units.
        sigma_init : float, optional
            Initial noise standard deviation. Default is 0.5.
        activation : str | None, optional
            Activation function. Default is None.
        use_bias : bool, optional
            Whether to use bias. Default is True.
        """
        super().__init__(**kwargs)
        self.units = units
        self.sigma_init = sigma_init
        self.activation_fn = keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape: tuple) -> None:
        """Build layer weights."""
        input_dim = int(input_shape[-1])

        # ##>: Factorized noise initialization (more parameter efficient than independent noise).
        mu_range = 1.0 / np.sqrt(input_dim)
        sigma_value = self.sigma_init / np.sqrt(input_dim)

        # ##>: Mean weights (learnable).
        # ##&: add_weight is inherited from keras.layers.Layer (dynamic binding).
        self.w_mu = self.add_weight(  # type: ignore[missing-attribute]
            name="w_mu",
            shape=(input_dim, self.units),
            initializer=initializers.RandomUniform(-mu_range, mu_range),
            trainable=True,
        )
        self.w_sigma = self.add_weight(  # type: ignore[missing-attribute]
            name="w_sigma",
            shape=(input_dim, self.units),
            initializer=initializers.Constant(sigma_value),
            trainable=True,
        )

        if self.use_bias:
            self.b_mu = self.add_weight(  # type: ignore[missing-attribute]
                name="b_mu",
                shape=(self.units,),
                initializer=initializers.RandomUniform(-mu_range, mu_range),
                trainable=True,
            )
            self.b_sigma = self.add_weight(  # type: ignore[missing-attribute]
                name="b_sigma",
                shape=(self.units,),
                initializer=initializers.Constant(sigma_value),
                trainable=True,
            )

        # ##>: Store input dim for factorized noise.
        self.input_dim = input_dim
        super().build(input_shape)

    def _factorized_noise(self, size: int) -> tf.Tensor:
        """Generate factorized Gaussian noise."""
        x = tf.random.normal((size,))
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """Forward pass with noisy weights."""
        if training:
            # ##>: Generate factorized noise for efficiency.
            epsilon_i = self._factorized_noise(self.input_dim)
            epsilon_j = self._factorized_noise(self.units)
            epsilon_w = tf.tensordot(epsilon_i, epsilon_j, axes=0)

            # ##>: Noisy weights: w = w_mu + w_sigma * noise.
            w = self.w_mu + self.w_sigma * epsilon_w
            output = tf.matmul(inputs, w)

            if self.use_bias:
                epsilon_b = epsilon_j
                b = self.b_mu + self.b_sigma * epsilon_b
                output = output + b
        else:
            # ##>: Inference: use mean weights only.
            output = tf.matmul(inputs, self.w_mu)
            if self.use_bias:
                output = output + self.b_mu

        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def get_config(self) -> dict:
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "sigma_init": self.sigma_init,
                "activation": keras.activations.serialize(self.activation_fn),
                "use_bias": self.use_bias,
            }
        )
        return config
