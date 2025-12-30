"""Pydantic models for PPO training configuration."""

from pydantic import BaseModel, Field, field_validator


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

    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
