# -*- coding: utf-8 -*-
"""
Models for all actors and critics algorithms.
"""

from .resnet import ResNetACModel
from .separate_net import SeparateNetACModel

__all__ = ["ResNetACModel", "SeparateNetACModel"]
