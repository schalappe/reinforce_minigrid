# -*- coding: utf-8 -*-
"""
Base Pydantic model for trainer configurations.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class TrainerConfig(BaseModel):
    """
    Base model for trainer configurations.

    Attributes
    ----------
    trainer_type : str
        Type of trainer to use.
    max_steps_per_episode : int, default=100
        Maximum number of steps per episode. Must be ≥1.
    eval_frequency : int, default=100
        Number of episodes between evaluations. Must be ≥1.
    num_eval_episodes : int, default=5
        Number of episodes to evaluate for. Must be ≥1.
    gamma : float, default=0.99
        Discount factor for future rewards. Must be between 0 and 1 (inclusive).
    log_frequency : int, default=10
        Number of episodes between logging. Must be ≥1.
    save_frequency : int, default=500
        Number of episodes between saving the model. Must be ≥1.
    save_dir : Path, default=Path("outputs") / "models"
        Directory to save models and logs.
    trial_info : Optional[dict], optional
        Internal field for Optuna integration (excluded from serialization).
    pruning_callback : Optional[callable], optional
        Internal field for Optuna pruning callback (excluded from serialization).

    Notes
    -----
    The `trial_info` and `pruning_callback` fields are added by ExperimentRunner when using Optuna integration.
    """

    trainer_type: str = Field(..., description="Type of trainer to use")
    max_steps_per_episode: int = Field(100, ge=1, description="Maximum number of steps per episode")
    eval_frequency: int = Field(100, ge=1, description="Number of episodes between evaluations")
    num_eval_episodes: int = Field(5, ge=1, description="Number of episodes to evaluate for")
    gamma: float = Field(0.99, ge=0, le=1, description="Discount factor for future rewards")
    log_frequency: int = Field(10, ge=1, description="Number of episodes between logging")
    save_frequency: int = Field(500, ge=1, description="Number of episodes between saving the model")
    save_dir: Path = Field(Path("outputs") / "models", description="Directory to save models and logs")
    save_path: Optional[str] = Field(None, description="Path to save the final trained agent model")

    # ##: Fields for Optuna integration (optional, added by ExperimentRunner if needed).
    trial_info: Optional[dict] = Field(None, exclude=True)
    pruning_callback: Optional[callable] = Field(None, exclude=True)

    class Config:
        """
        Pydantic config settings.

        Notes
        -----
        - `arbitrary_types_allowed=True` enables support for callable types in `pruning_callback`.
        """

        arbitrary_types_allowed = True
