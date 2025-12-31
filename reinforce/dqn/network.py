"""
Rainbow DQN Network Architecture.

Implements:
- Dueling DQN: Separate value and advantage streams
- Noisy Networks: Parametric noise for exploration
- Categorical DQN (C51): Distributional value estimation
"""

import tensorflow as tf
from keras import Model, layers

from reinforce.core.network_utils import NoisyDense, build_impala_cnn, get_orthogonal_initializer


def build_rainbow_network(
    input_shape: tuple,
    num_actions: int,
    num_atoms: int = 51,
    v_min: float = -10.0,
    v_max: float = 10.0,
    use_noisy: bool = True,
    use_dueling: bool = True,
) -> Model:
    """
    Build Rainbow DQN network combining all enhancements.

    Parameters
    ----------
    input_shape : tuple
        Observation shape (height, width, channels).
    num_actions : int
        Number of discrete actions.
    num_atoms : int, optional
        Number of atoms for distributional RL (C51). Default is 51.
    v_min : float, optional
        Minimum value support. Default is -10.0.
    v_max : float, optional
        Maximum value support. Default is 10.0.
    use_noisy : bool, optional
        Whether to use noisy layers. Default is True.
    use_dueling : bool, optional
        Whether to use dueling architecture. Default is True.

    Returns
    -------
    Model
        Rainbow DQN model outputting action-value distributions.
    """
    inputs = layers.Input(shape=input_shape, name="observation")

    # ##>: Shared CNN backbone (IMPALA-style).
    features = build_impala_cnn(inputs, name_prefix="rainbow")

    if use_dueling:
        # ##>: Value stream: V(s) distribution over atoms.
        if use_noisy:
            value_hidden = NoisyDense(256, activation="relu", name="value_hidden")(features)  # type: ignore[not-callable]
            value_atoms = NoisyDense(num_atoms, name="value_atoms")(value_hidden)  # type: ignore[not-callable]
        else:
            value_hidden = layers.Dense(
                256,
                activation="relu",
                kernel_initializer=get_orthogonal_initializer(scale=2**0.5),
                name="value_hidden",
            )(features)
            value_atoms = layers.Dense(
                num_atoms,
                kernel_initializer=get_orthogonal_initializer(scale=1.0),
                name="value_atoms",
            )(value_hidden)

        # ##>: Advantage stream: A(s, a) distribution for each action.
        if use_noisy:
            adv_hidden = NoisyDense(256, activation="relu", name="adv_hidden")(features)  # type: ignore[not-callable]
            adv_atoms_flat = NoisyDense(num_actions * num_atoms, name="adv_atoms")(adv_hidden)  # type: ignore[not-callable]
        else:
            adv_hidden = layers.Dense(
                256,
                activation="relu",
                kernel_initializer=get_orthogonal_initializer(scale=2**0.5),
                name="adv_hidden",
            )(features)
            adv_atoms_flat = layers.Dense(
                num_actions * num_atoms,
                kernel_initializer=get_orthogonal_initializer(scale=1.0),
                name="adv_atoms",
            )(adv_hidden)

        adv_atoms = layers.Reshape((num_actions, num_atoms), name="adv_reshape")(adv_atoms_flat)

        # ##>: Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,:)).
        value_atoms = layers.Reshape((1, num_atoms), name="value_reshape")(value_atoms)
        adv_mean = layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True), name="adv_mean")(adv_atoms)
        q_atoms = layers.Add(name="q_combine")([value_atoms, adv_atoms])
        q_atoms = layers.Subtract(name="q_subtract")([q_atoms, adv_mean])
    else:
        # ##>: Standard Q-network without dueling.
        if use_noisy:
            hidden = NoisyDense(512, activation="relu", name="hidden")(features)  # type: ignore[not-callable]
            q_atoms_flat = NoisyDense(num_actions * num_atoms, name="q_atoms")(hidden)  # type: ignore[not-callable]
        else:
            hidden = layers.Dense(
                512,
                activation="relu",
                kernel_initializer=get_orthogonal_initializer(scale=2**0.5),
                name="hidden",
            )(features)
            q_atoms_flat = layers.Dense(
                num_actions * num_atoms,
                kernel_initializer=get_orthogonal_initializer(scale=1.0),
                name="q_atoms",
            )(hidden)

        q_atoms = layers.Reshape((num_actions, num_atoms), name="q_reshape")(q_atoms_flat)

    # ##>: Softmax over atoms to get probability distribution.
    q_dist = layers.Softmax(axis=-1, name="q_distribution")(q_atoms)

    return Model(inputs=inputs, outputs=q_dist, name="RainbowNetwork")
