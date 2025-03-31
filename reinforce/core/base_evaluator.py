# -*- coding: utf-8 -*-
"""
Base interface for all evaluators.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from reinforce.core.base_agent import BaseAgent
from reinforce.core.base_environment import BaseEnvironment


class BaseEvaluator(ABC):
    """
    Base interface for all evaluators.

    This abstract class defines the interface that all evaluator implementations must adhere to.
    It provides a standard API for evaluating agents in environments.
    """

    @abstractmethod
    def __init__(
        self,
        agent: BaseAgent,
        environment: BaseEnvironment,
        config: Dict[str, Any],
        callbacks: Optional[List[Callable]] = None,
    ):
        """
        Initialize the evaluator.

        Parameters
        ----------
        agent : BaseAgent
            The agent to evaluate
        environment : BaseEnvironment
            The environment to evaluate in
        config : dict
            Configuration parameters for evaluation
        callbacks : list of callable, optional
            Callbacks to invoke during evaluation
        """

    @abstractmethod
    def evaluate(self, num_episodes: int = 1) -> Dict[str, Any]:
        """
        Evaluate the agent in the environment.

        Parameters
        ----------
        num_episodes : int, default=1
            Number of episodes to evaluate

        Returns
        -------
        Dict[str, Any]
            Dictionary of evaluation metrics.
        """

    @abstractmethod
    def visualize(self, path: Optional[str] = None) -> None:
        """Visualize the agent's performance."""

    @abstractmethod
    def benchmark(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Benchmark the agent's performance.

        Parameters
        ----------
        num_episodes : int, default=10
            Number of episodes to benchmark.

        Returns
        -------
        Dict[str, Any]
            Dictionary of benchmark metrics.
        """
