"""Factory functions for creating algorithm-specific components."""

import gymnasium as gym

from reinforce.config.training_config import Algorithm, MainConfig
from reinforce.core.base_agent import BaseAgent


def create_agent(
    config: MainConfig,
    observation_space: gym.Space,
    action_space: gym.Space,
) -> BaseAgent:
    """
    Create agent based on algorithm configuration.

    Parameters
    ----------
    config : MainConfig
        Full training configuration.
    observation_space : gym.Space
        Environment observation space.
    action_space : gym.Space
        Environment action space.

    Returns
    -------
    BaseAgent
        Configured agent instance (PPOAgent or RainbowAgent).

    Raises
    ------
    ValueError
        If algorithm is not recognized.
    """
    if config.algorithm == Algorithm.PPO:
        from reinforce.ppo.agent import PPOAgent

        return PPOAgent(
            observation_space=observation_space,
            action_space=action_space,
            learning_rate=config.ppo.learning_rate,
            gamma=config.ppo.gamma,
            lam=config.ppo.lambda_gae,
            clip_param=config.ppo.clip_param,
            entropy_coef=config.ppo.entropy_coef,
            vf_coef=config.ppo.value_coef,
            epochs=config.ppo.epochs,
            batch_size=config.ppo.batch_size,
            num_envs=config.training.num_envs,
            steps_per_update=config.training.steps_per_update,
            max_grad_norm=config.ppo.max_grad_norm,
            total_timesteps=config.training.total_timesteps,
            use_lr_annealing=config.ppo.use_lr_annealing,
            use_value_clipping=config.ppo.use_value_clipping,
        )

    elif config.algorithm == Algorithm.DQN:
        from reinforce.dqn.agent import RainbowAgent

        return RainbowAgent(
            observation_space=observation_space,
            action_space=action_space,
            learning_rate=config.dqn.learning_rate,
            gamma=config.dqn.gamma,
            n_step=config.dqn.n_step,
            num_atoms=config.dqn.num_atoms,
            v_min=config.dqn.v_min,
            v_max=config.dqn.v_max,
            buffer_size=config.dqn.buffer_size,
            batch_size=config.dqn.batch_size,
            target_update_freq=config.dqn.target_update_freq,
            learning_starts=config.dqn.learning_starts,
            train_freq=config.dqn.train_freq,
            priority_alpha=config.dqn.priority_alpha,
            priority_beta_start=config.dqn.priority_beta_start,
            priority_beta_frames=config.dqn.priority_beta_frames,
            use_noisy=config.dqn.use_noisy,
            use_dueling=config.dqn.use_dueling,
            use_double=config.dqn.use_double,
            use_per=config.dqn.use_per,
        )

    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
