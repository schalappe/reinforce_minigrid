# -*- coding: utf-8 -*-
"""
Concrete persistence handler using Keras model saving and JSON for hyperparameters.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from keras import Model, models

from reinforce.agents.persistence.base_persistence import AgentPersistence
from reinforce.configs import ConfigManager


class KerasFilePersistence(AgentPersistence):
    """
    Concrete persistence handler using Keras model saving and JSON for hyperparameters.
    """

    def save(self, path: str, agent_name: str, model: Model, hyperparameters: Dict[str, Any]) -> None:
        """
        Save the agent's model and hyperparameters using Keras and JSON.

        Parameters
        ----------
        path : str
            Directory path to save the agent.
        agent_name : str
            Name of the agent (used for file naming).
        model : Model
            The Keras model to save.
        hyperparameters : Dict[str, Any]
            The agent's hyperparameters.
        """
        save_directory = Path(path)
        save_directory.mkdir(parents=True, exist_ok=True)

        # ##: Save the Keras model.
        model_path = save_directory / f"{agent_name}_model.keras"
        model.save(model_path)

        # ##: Save the hyperparameters to JSON.
        hyperparams_path = save_directory / "hyperparams.json"
        ConfigManager.save_config(hyperparameters, str(hyperparams_path))

    def load(self, path: str, agent_name: str) -> Tuple[Model, Dict[str, Any]]:
        """
        Load the agent's model and hyperparameters from Keras and JSON files.

        Parameters
        ----------
        path : str
            Directory path to load the agent from.
        agent_name : str
            Name of the agent (used for file naming).

        Returns
        -------
        Tuple[Model, Dict[str, Any]]
            A tuple containing the loaded model, hyperparameters.

        Raises
        ------
        FileNotFoundError
            If the model or hyperparameter file is not found.
        """
        load_directory = Path(path)

        # ##: Load the Keras model.
        model_path = load_directory / f"{agent_name}_model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        loaded_model = models.load_model(model_path)

        # ##: Load the hyperparameters.
        hyperparams_path = load_directory / "hyperparams.json"
        if not hyperparams_path.exists():
            raise FileNotFoundError(f"Hyperparameters file not found at {hyperparams_path}")
        loaded_hyperparams_dict = ConfigManager.load_config(str(hyperparams_path))

        return loaded_model, loaded_hyperparams_dict
