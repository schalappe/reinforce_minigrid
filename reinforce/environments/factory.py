# -*- coding: utf-8 -*-
"""
Factory for creating environment instances.
"""

from reinforce.configs.models import EnvironmentConfig
from reinforce.environments.minigrid import MazeEnvironment


class EnvironmentFactory:
    """Factory class for creating environment instances based on configuration."""

    @staticmethod
    def create(env_config: EnvironmentConfig) -> MazeEnvironment:
        """
        Create an environment based on the Pydantic configuration model.

        This method creates and returns a `MazeEnvironment` instance based on the provided config model.

        Parameters
        ----------
        env_config : EnvironmentConfig
            Environment configuration.

        Returns
        -------
        MazeEnvironment
            Created environment instance.

        Raises
        ------
        ValueError
            If the environment type specified in the config is not supported.

        Notes
        -----
        Currently only supports `MazeEnvironment`. Needs update if more env types are added.
        """
        if env_config.env_type == "MazeEnvironment":
            return MazeEnvironment(use_image_obs=env_config.use_image_obs)
        raise ValueError(f"Unsupported environment type: {env_config.env_type}")
