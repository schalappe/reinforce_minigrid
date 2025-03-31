# -*- coding: utf-8 -*-
"""
Metrics for evaluating reinforcement learning agents.
"""

from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class MetricsLogger:
    """
    Logger for tracking and visualizing training metrics.

    This class provides functionality for logging, aggregating, and visualizing metrics during training and evaluation.
    """

    def __init__(self):
        """Initialize the metrics logger."""
        self.metrics = {}

    def log_metric(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """
        Log a metric.

        Parameters
        ----------
        name : str
            Name of the metric.
        value : Any
            Value of the metric.
        step : int, optional
            Optional step number.
        """
        if name not in self.metrics:
            self.metrics[name] = []

        metric_value = value
        if hasattr(value, "numpy"):
            metric_value = value.numpy()

        if step is not None:
            self.metrics[name].append((step, metric_value))
        else:
            self.metrics[name].append(metric_value)

    def get_metric(self, name: str) -> List[Any]:
        """
        Get a metric by name.

        Parameters
        ----------
        name : str
            Name of the metric.

        Returns
        -------
        List[Any]
            List of values for the metric.

        Raises
        ------
        KeyError
            If the metric is not found.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found")

        return self.metrics[name]

    def get_latest(self, name: str) -> Any:
        """
        Get the latest value of a metric.

        Parameters
        ----------
        name : str
            Name of the metric.

        Returns
        -------
        Any
            Latest value of the metric.

        Raises
        ------
        KeyError
            If the metric is not found.
        IndexError
            If the metric has no values.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found")

        if not self.metrics[name]:
            raise IndexError(f"Metric '{name}' has no values")

        return self.metrics[name][-1]

    def get_mean(self, name: str, window: Optional[int] = None) -> float:
        """
        Get the mean of a metric.

        Parameters
        ----------
        name : str
            Name of the metric
        window : int, optional
            Optional window size for moving average.

        Returns
        -------
        float
            Mean of the metric.

        Raises
        ------
        KeyError
            If the metric is not found.
        IndexError
            If the metric has no values.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found")

        if not self.metrics[name]:
            raise IndexError(f"Metric '{name}' has no values")

        values = self.metrics[name]
        if window is not None:
            values = values[-window:]

        if isinstance(values[0], tuple):
            return np.mean([v[1] for v in values])
        return np.mean(values)

    def plot_metric(self, name: str, title: Optional[str] = None, window: Optional[int] = None) -> plt.Figure:
        """
        Plot a metric.

        Parameters
        ----------
        name : str
            Name of the metric.
        title : str, optional
            Optional title for the plot.
        window : int, optional
            Optional window size for moving average.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure.

        Raises
        ------
        KeyError
            If the metric is not found.
        IndexError
            If the metric has no values.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found")

        if not self.metrics[name]:
            raise IndexError(f"Metric '{name}' has no values")

        fig, ax = plt.subplots(figsize=(10, 5))
        values = self.metrics[name]

        if isinstance(values[0], tuple):
            steps = [v[0] for v in values]
        values = [v[1] for v in values]
        ax.plot(steps, values, label=name)

        if window is not None:
            moving_avg = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1 :], moving_avg, label=f"{name} (MA-{window})")
        else:
            ax.plot(values, label=name)

        if window is not None:
            moving_avg = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(values)), moving_avg, label=f"{name} (MA-{window})")
        ax.set_xlabel("Step")
        ax.set_ylabel(name)

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(f"{name} over time")

        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        return fig

    def plot_metrics(self, names: List[str], title: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple metrics on the same figure.

        Parameters
        ----------
        names : list of str
            List of metric names.
        title : str, optional
            Optional title for the plot.

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure.

        Raises
        ------
        KeyError
            If any metric is not found.
        IndexError
            If any metric has no values.
        """
        for name in names:
            if name not in self.metrics:
                raise KeyError(f"Metric '{name}' not found")

            if not self.metrics[name]:
                raise IndexError(f"Metric '{name}' has no values")

        fig, ax = plt.subplots(figsize=(10, 5))

        for name in names:
            values = self.metrics[name]

            if isinstance(values[0], tuple):
                steps = [v[0] for v in values]
                plot_values = [v[1] for v in values]
                ax.plot(steps, plot_values, label=name)
            else:
                ax.plot(values, label=name)

        ax.set_xlabel("Step")
        ax.set_ylabel("Value")

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("Metrics over time")

        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        return fig

    def save_plot(self, name: str, path: str, title: Optional[str] = None, window: Optional[int] = None) -> None:
        """
        Save a plot of a metric to a file.

        Parameters
        ----------
        name : str
            Name of the metric.
        path : str
            Path to save the plot to.
        title : str, optional
            Optional title for the plot.
        window : int, optional
            Optional window size for moving average.

        Raises
        ------
        KeyError
            If the metric is not found.
        IndexError
            If the metric has no values.
        """
        fig = self.plot_metric(name, title, window)
        fig.savefig(path)
        plt.close(fig)

    def save_metrics(self, path: str) -> None:
        """
        Save metrics to a file.

        Parameters
        ----------
        path : str
            Path to save the metrics to.
        """
        np.save(path, self.metrics)

    def load_metrics(self, path: str) -> None:
        """
        Load metrics from a file.

        Parameters
        ----------
        path : str
            Path to load the metrics from.
        """
        self.metrics = np.load(path, allow_pickle=True).item()

    def clear(self) -> None:
        """Clear all metrics."""
        self.metrics = {}
