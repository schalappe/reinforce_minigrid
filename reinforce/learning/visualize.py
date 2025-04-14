# -*- coding: utf-8 -*-
"""Script to visualize training progress from the log file."""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Configuration ---
DEFAULT_VIS_CONFIG = {
    "log_dir": "checkpoints",  # Directory containing the log file
    "log_filename": "training_log.csv",  # Name of the log file
    "output_dir": "plots",  # Directory to save the plots
    "plot_filename": "training_progress.png",  # Name for the output plot file
}


def visualize(config):
    """
    Generates plots from the training log file.

    Args:
        config (dict): Dictionary containing visualization configuration.
    """
    print("Initializing visualization...")
    print(f"Configuration: {config}")

    log_filepath = os.path.join(config["log_dir"], config["log_filename"])

    # Check if log file exists
    if not os.path.isfile(log_filepath):
        print(f"Error: Log file not found at {log_filepath}")
        print("Please run the training script first (train.py) to generate the log file.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)
    output_filepath = os.path.join(config["output_dir"], config["plot_filename"])

    # Load data using pandas
    try:
        df = pd.read_csv(log_filepath)
        print(f"Loaded log data with {len(df)} epochs.")
        # Basic validation of columns
        required_columns = ["Epoch", "Mean Return", "Mean Length"]
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Log file {log_filepath} is missing required columns.")
            print(f"Expected columns: {required_columns}")
            print(f"Found columns: {list(df.columns)}")
            return

    except Exception as e:
        print(f"Error reading log file {log_filepath}: {e}")
        return

    # --- Generate Plots ---
    print("Generating plots...")
    sns.set_theme(style="darkgrid")

    plt.figure(figsize=(12, 5))

    # Plot Mean Return vs Epoch
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    sns.lineplot(data=df, x="Epoch", y="Mean Return")
    plt.title("Mean Episodic Return per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Return")

    # Plot Mean Episode Length vs Epoch
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    sns.lineplot(data=df, x="Epoch", y="Mean Length")
    plt.title("Mean Episode Length per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Length (Steps)")

    plt.tight_layout()  # Adjust layout to prevent overlap

    # Save the plot
    try:
        plt.savefig(output_filepath)
        print(f"Plot saved successfully to {output_filepath}")
    except Exception as e:
        print(f"Error saving plot to {output_filepath}: {e}")

    # Optionally display the plot
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PPO training progress")
    parser.add_argument(
        "--log-dir", type=str, default=DEFAULT_VIS_CONFIG["log_dir"], help="Directory containing the training log file"
    )
    parser.add_argument(
        "--log-filename",
        type=str,
        default=DEFAULT_VIS_CONFIG["log_filename"],
        help="Name of the training log CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_VIS_CONFIG["output_dir"],
        help="Directory to save the generated plots",
    )
    parser.add_argument(
        "--plot-filename",
        type=str,
        default=DEFAULT_VIS_CONFIG["plot_filename"],
        help="Filename for the output plot image",
    )

    args = parser.parse_args()
    config = vars(args)

    visualize(config)
