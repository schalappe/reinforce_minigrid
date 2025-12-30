"""Pydantic models for training configuration supporting multiple RL algorithms."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Algorithm(str, Enum):
    """Supported RL algorithms."""

    PPO = "ppo"
    DQN = "dqn"


class EnvironmentConfig(BaseModel):
    """Configuration for the training environment."""

    seed: int = 42


class PPOConfig(BaseModel):
    """Hyperparameters for the PPO algorithm."""

    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    lambda_gae: float = Field(default=0.95, alias="lambda")
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = Field(default=0.5, alias="vf_coef")
    epochs: int = 4
    batch_size: int = 256
    max_grad_norm: float = 0.5
    use_lr_annealing: bool = True
    use_value_clipping: bool = False

    model_config = {"populate_by_name": True}

    @field_validator("learning_rate", "max_grad_norm")
    @classmethod
    def must_be_positive(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    @field_validator("gamma", "lambda_gae")
    @classmethod
    def must_be_in_unit_interval(cls, v: float, info) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"{info.field_name} must be in (0, 1]")
        return v


class DQNConfig(BaseModel):
    """Hyperparameters for the Rainbow DQN algorithm."""

    learning_rate: float = 6.25e-5
    gamma: float = 0.99
    # ##>: Multi-step learning parameters.
    n_step: int = 3
    # ##>: Categorical DQN (C51) parameters.
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    # ##>: Replay buffer parameters.
    buffer_size: int = 100_000
    batch_size: int = 32
    # ##>: Training schedule parameters.
    target_update_freq: int = 8_000
    learning_starts: int = 20_000
    train_freq: int = 4
    # ##>: Prioritized Experience Replay parameters.
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_frames: int = 100_000
    # ##>: Rainbow component toggles.
    use_noisy: bool = True
    use_dueling: bool = True
    use_double: bool = True
    use_per: bool = True
    use_multistep: bool = True
    use_categorical: bool = True

    model_config = {"populate_by_name": True}

    @field_validator("learning_rate")
    @classmethod
    def lr_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("learning_rate must be positive")
        return v

    @field_validator("gamma")
    @classmethod
    def gamma_in_unit_interval(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("gamma must be in (0, 1]")
        return v


class RNDConfig(BaseModel):
    """Configuration for Random Network Distillation (intrinsic motivation)."""

    enabled: bool = True
    feature_dim: int = 512
    learning_rate: float = 1e-4
    intrinsic_reward_scale: float = 1.0
    update_proportion: float = 0.25
    intrinsic_reward_coef: float = 0.5


class ExplorationConfig(BaseModel):
    """Configuration for hybrid exploration strategies."""

    # ##>: Epsilon-greedy parameters.
    use_epsilon_greedy: bool = True
    epsilon_start: float = 0.3
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 500_000

    # ##>: UCB parameters.
    use_ucb: bool = True
    ucb_coefficient: float = 0.5

    # ##>: Adaptive entropy parameters.
    use_adaptive_entropy: bool = True
    target_entropy_ratio: float = 0.5
    entropy_lr: float = 0.01
    min_entropy_coef: float = 0.001
    max_entropy_coef: float = 0.1


class TrainingConfig(BaseModel):
    """Configuration for the training process."""

    total_timesteps: int = 3_000_000
    steps_per_update: int = 128
    num_envs: int = 8


class LoggingConfig(BaseModel):
    """Configuration for logging and model saving."""

    log_interval: int = 1
    save_interval: int = 10
    save_path: str = "models/ppo_maze"
    load_path: str | None = None


class MainConfig(BaseModel):
    """Root configuration aggregating all sub-configurations."""

    algorithm: Algorithm = Algorithm.PPO
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    dqn: DQNConfig = Field(default_factory=DQNConfig)
    rnd: RNDConfig = Field(default_factory=RNDConfig)
    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
