# -*- coding: utf-8 -*-
"""
Concrete implementation for managing training checkpoints.
"""

from pathlib import Path
from typing import Any, Dict, Union

from loguru import logger

from reinforce.agents import BaseAgent
from reinforce.utils.logger import AimTracker, setup_logger

setup_logger()


def save_checkpoint(
    agent: BaseAgent, save_path_base: Union[str, Path], trainer_state: Dict[str, Any], tracker: AimTracker
):
    """
    Save the agent's state and the trainer's state.

    Creates a directory based on `save_path_base`, saves the agent model within an 'agent' subdirectory,
    and logs artifact information

    Parameters
    ----------
    agent : BaseAgent
        The agent whose state needs to be saved.
    save_path_base : Union[str, Path]
        The base directory path for saving the checkpoint. The directory will be created if it doesn't exist.
    trainer_state : Dict[str, Any]
        A dictionary containing the current state of the trainer
    tracker : AimTracker
        A tracker instance to log artifact information about the checkpoint.
    """
    path_obj = Path(save_path_base)
    path_obj.mkdir(parents=True, exist_ok=True)
    agent_path = path_obj / "agent"

    # ##: Save the agent.
    agent.save(str(agent_path))
    logger.info(f"Agent state saved successfully to: {agent_path}")

    # ##: Log checkpoint artifact info using the provided logger instance.
    episode = trainer_state.get("episode", "unknown")
    total_steps = trainer_state.get("total_steps", "unknown")
    artifact_name = f"checkpoint_episode_{episode}"
    meta = {
        "episode": episode,
        "total_steps": total_steps,
        "agent_save_path": str(agent_path.resolve()),
        **{k: v for k, v in trainer_state.items() if k not in ["episode", "total_steps"]},
    }

    tracker.log_artifact(artifact_data=trainer_state, name=artifact_name, path=str(path_obj.resolve()), meta=meta)
    logger.info(f"Checkpoint saved successfully to: {path_obj}")
