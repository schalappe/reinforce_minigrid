# -*- coding: utf-8 -*-
"""
Pydantic model for environment configuration.
"""

from typing import Literal

from pydantic import BaseModel, Field


class EnvironmentConfig(BaseModel):
    """
    Pydantic model for environment configuration.

    Attributes
    ----------
    env_type : Literal["MazeEnvironment"]
        Type of environment. Defaults to "MazeEnvironment".
    use_image_obs : bool
        Whether to use image observations. Defaults to True.
    """

    env_type: Literal["MazeEnvironment"] = Field("MazeEnvironment", description="Type of environment")
    use_image_obs: bool = Field(True, description="Whether to use image observations")
