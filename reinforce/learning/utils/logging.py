# -*- coding: utf-8 -*-
"""Logging utilities for training metrics."""

import csv
import time
from pathlib import Path
from typing import Union

from loguru import logger


class MetricsLogger:
    """
    Handles logging of training metrics to a CSV file.

    Attributes
    ----------
    log_filepath : Path
        Path to the CSV log file.
    log_file : object
        File object for the opened log file.
    log_writer : object
        CSV writer object.
    start_time : float
        Timestamp when the logger was initialized (used for total duration).
    """

    def __init__(self, save_dir: Union[str, Path], filename: str = "training_log.csv"):
        """
        Initializes the MetricsLogger.

        Creates the save directory if it doesn't exist and opens the log file, writing the header row.

        Parameters
        ----------
        save_dir : Union[str, Path]
            Directory where the log file will be saved.
        filename : str, optional
            Name of the log file, by default "training_log.csv".
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.log_filepath = save_dir / filename
        self.start_time = time.time()

        # ##: Open file in append mode, create if doesn't exist.
        self.log_file = open(self.log_filepath, "a", newline="")
        self.log_writer = csv.writer(self.log_file)

        # ##: Write header only if the file is newly created (empty).
        if self.log_file.tell() == 0:
            self.log_writer.writerow(
                ["Epoch", "Mean Return", "Mean Length", "Num Episodes", "Epoch Duration (s)", "Total Duration (s)"]
            )
            self.log_file.flush()

        logger.info(f"Logging training metrics to: {self.log_filepath}")

    def log_epoch(
        self,
        epoch: int,
        mean_return: float,
        mean_length: float,
        num_episodes: int,
        epoch_duration: float,
    ):
        """
        Logs the metrics for a completed training epoch.

        Calculates the total duration since initialization and writes a new row to the CSV file.

        Parameters
        ----------
        epoch : int
            The epoch number (1-based).
        mean_return : float
            Average return achieved during the epoch.
        mean_length : float
            Average length of episodes completed during the epoch.
        num_episodes : int
            Number of episodes completed during the epoch.
        epoch_duration : float
            Duration of the epoch in seconds.
        """
        current_total_duration = time.time() - self.start_time
        self.log_writer.writerow(
            [epoch, mean_return, mean_length, num_episodes, epoch_duration, current_total_duration]
        )
        self.log_file.flush()

    def close(self):
        """Closes the log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            print(f"Closed log file: {self.log_filepath}")

    def __del__(self):
        """Ensures the file is closed when the object is destroyed."""
        self.close()
