# -*- coding: utf-8 -*-
"""
JSON schema for agent configurations.
"""

from typing import Any, Dict

from jsonschema import validate

# ##: Schema for A2C agent configuration.
A2C_SCHEMA = {
    "type": "object",
    "properties": {
        "agent_type": {"type": "string", "enum": ["A2C"], "description": "Type of agent to use"},
        "action_space": {"type": "integer", "minimum": 1, "description": "Number of possible actions"},
        "embedding_size": {"type": "integer", "minimum": 1, "description": "Size of the embedding layer"},
        "learning_rate": {"type": "number", "minimum": 0, "description": "Learning rate for the optimizer"},
        "discount_factor": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Discount factor for future rewards",
        },
        "entropy_coef": {"type": "number", "minimum": 0, "description": "Entropy regularization coefficient"},
        "value_coef": {"type": "number", "minimum": 0, "description": "Value loss coefficient"},
    },
    "required": ["agent_type", "action_space"],
    "additionalProperties": False,
}


# ##: Combined schema for all agent types.
AGENT_SCHEMA = {"type": "object", "oneOf": [A2C_SCHEMA]}


def validate_agent_config(config: Dict[str, Any]) -> None:
    """
    Validate an agent configuration against the schema.

    Parameters
    ----------
    config: Dict[str, Any]
       The Agent configuration to validate.

    Raises
    ------
    ValidationError
        If the configuration is invalid.
    """
    validate(config, AGENT_SCHEMA)
