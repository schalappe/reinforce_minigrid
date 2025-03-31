# -*- coding: utf-8 -*-
"""
JSON schema for trainer configurations.
"""

from typing import Any, Dict

from jsonschema import validate

# ##: Schema for episode trainer configuration.
EPISODE_TRAINER_SCHEMA = {
    "type": "object",
    "properties": {
        "trainer_type": {"type": "string", "enum": ["EpisodeTrainer"], "description": "Type of trainer to use"},
        "max_episodes": {"type": "integer", "minimum": 1, "description": "Maximum number of episodes to train for"},
        "max_steps_per_episode": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum number of steps per episode",
        },
        "update_frequency": {"type": "integer", "minimum": 1, "description": "Number of steps between agent updates"},
        "eval_frequency": {"type": "integer", "minimum": 1, "description": "Number of episodes between evaluations"},
        "num_eval_episodes": {"type": "integer", "minimum": 1, "description": "Number of episodes to evaluate for"},
        "gamma": {"type": "number", "minimum": 0, "maximum": 1, "description": "Discount factor for future rewards"},
        "log_frequency": {"type": "integer", "minimum": 1, "description": "Number of episodes between logging"},
        "save_frequency": {
            "type": "integer",
            "minimum": 1,
            "description": "Number of episodes between saving the model",
        },
        "save_dir": {"type": "string", "description": "Directory to save models and logs"},
    },
    "required": ["trainer_type"],
    "additionalProperties": False,
}


# ##: Schema for distributed trainer configuration.
DISTRIBUTED_TRAINER_SCHEMA = {
    "type": "object",
    "properties": {
        "trainer_type": {"type": "string", "enum": ["DistributedTrainer"], "description": "Type of trainer to use"},
        "num_workers": {"type": "integer", "minimum": 1, "description": "Number of worker processes to use"},
        "max_episodes_per_worker": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum number of episodes per worker",
        },
        "max_steps_per_episode": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum number of steps per episode",
        },
        "update_frequency": {"type": "integer", "minimum": 1, "description": "Number of steps between agent updates"},
        "eval_frequency": {"type": "integer", "minimum": 1, "description": "Number of episodes between evaluations"},
        "num_eval_episodes": {"type": "integer", "minimum": 1, "description": "Number of episodes to evaluate for"},
        "gamma": {"type": "number", "minimum": 0, "maximum": 1, "description": "Discount factor for future rewards"},
        "log_frequency": {"type": "integer", "minimum": 1, "description": "Number of episodes between logging"},
        "save_frequency": {
            "type": "integer",
            "minimum": 1,
            "description": "Number of episodes between saving the model",
        },
        "save_dir": {"type": "string", "description": "Directory to save models and logs"},
    },
    "required": ["trainer_type", "num_workers"],
    "additionalProperties": False,
}


# ##: Combined schema for all trainer types.
TRAINER_SCHEMA = {"type": "object", "oneOf": [EPISODE_TRAINER_SCHEMA, DISTRIBUTED_TRAINER_SCHEMA]}


def validate_trainer_config(config: Dict[str, Any]) -> None:
    """
    Validate a trainer configuration against the schema.

    Parameters
    ----------
    config: Dict[str, Any]
        Trainer configuration to validate.

    Raises
    ------
    ValidationError
        If the configuration is invalid.
    """
    validate(config, TRAINER_SCHEMA)
