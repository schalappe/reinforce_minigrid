# -*- coding: utf-8 -*-
"""
Hyperparameter optimization script for PPO using Optuna.

This script defines an objective function to train the PPO agent with hyperparameters suggested by Optuna,
runs an optimization study, and provides utilities for visualizing and exporting the results.
"""

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import optuna
import yaml
from loguru import logger
from optuna.pruners import MedianPruner
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)

from reinforce.learning.train import train
from reinforce.learning.utils.config import DEFAULT_TRAIN_CONFIG

# --- Constants ---
STUDY_NAME_PREFIX = "ppo_maze_optimization"
DEFAULT_N_TRIALS = 50
DEFAULT_N_JOBS = 1
DEFAULT_TIMEOUT = None
DEFAULT_STORAGE_DB = "sqlite:///optuna_ppo_study.db"
DEFAULT_OUTPUT_DIR = "optuna_results"


def objective(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """
    Objective function for Optuna optimization.

    Trains the PPO agent with hyperparameters suggested by the trial and returns a performance metric.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object providing hyperparameter suggestions.
    base_config : Dict[str, Any]
        Base configuration dictionary to be updated with trial parameters.

    Returns
    -------
    float
        The performance metric to be minimized (e.g., negative mean return over last N epochs).
    """
    # --- Hyperparameter Sampling ---
    trial_config = base_config.copy()
    trial_config["seed"] = trial.suggest_int("seed", 1, 10000)

    # ##: PPO Agent Hyperparameters.
    trial_config["gamma"] = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    trial_config["lam"] = trial.suggest_float("lam", 0.9, 1.0)
    trial_config["clip_ratio"] = trial.suggest_float("clip_ratio", 0.1, 0.3)
    trial_config["policy_learning_rate"] = trial.suggest_float("policy_learning_rate", 1e-5, 1e-3, log=True)
    trial_config["value_function_learning_rate"] = trial.suggest_float(
        "value_function_learning_rate", 1e-4, 1e-2, log=True
    )
    trial_config["train_policy_iterations"] = trial.suggest_categorical("train_policy_iterations", [40, 80, 120])
    trial_config["train_value_iterations"] = trial.suggest_categorical("train_value_iterations", [40, 80, 120])
    trial_config["target_kl"] = trial.suggest_float("target_kl", 0.005, 0.05, log=True)

    # ##: Network Parameters.
    trial_config["conv_filters"] = trial.suggest_categorical("conv_filters", [16, 32, 64])
    trial_config["conv_kernel_size"] = trial.suggest_categorical("conv_kernel_size", [3, 5])
    trial_config["dense_units"] = trial.suggest_categorical("dense_units", [64, 128, 256])

    # --- Setup Trial Directory ---
    trial_save_dir = Path(base_config["save_dir"]) / f"trial_{trial.number}"
    trial_save_dir.mkdir(parents=True, exist_ok=True)
    trial_config["save_dir"] = str(trial_save_dir)

    logger.info(f"\n--- Starting Trial {trial.number} ---")
    logger.info(f"Sampled Config: {trial_config}")

    try:
        # --- Run Training ---
        final_avg_reward = train(trial_config, trial=trial)

        # ##: Check if training was pruned or failed (indicated by -inf).
        if final_avg_reward == -float("inf"):
            logger.error(f"Trial {trial.number} resulted in an invalid metric (-inf). Treating as failure.")
            return float("inf")

        logger.info(f"Trial {trial.number} finished. Final Avg Reward (Metric): {final_avg_reward:.2f}")

        # ##: Optuna minimizes by default. Since higher reward is better, return the negative reward.
        metric_to_optimize = -final_avg_reward
        return metric_to_optimize

    except optuna.TrialPruned:
        logger.warning(f"Trial {trial.number} pruned.")
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return float("inf")


def run_optimization(args: argparse.Namespace):
    """
    Sets up and runs the Optuna optimization study.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing optimization settings.
    """
    logger.info("--- Starting Hyperparameter Optimization ---")
    logger.info(f"Arguments: {args}")

    # --- Load Base Config ---
    base_config = DEFAULT_TRAIN_CONFIG.copy()
    if args.base_config_yaml:
        try:
            with open(args.base_config_yaml, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    base_config.update(yaml_config)
                    logger.info(f"Loaded base configuration from: {args.base_config_yaml}")
        except FileNotFoundError:
            logger.error(f"Base config file not found: {args.base_config_yaml}")
            return
        except yaml.YAMLError as e:
            logger.error(f"Error parsing base config YAML {args.base_config_yaml}: {e}")
            return

    # --- Setup Output Directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config["save_dir"] = str(output_dir / "checkpoints")

    # --- Configure Study ---
    study_name = f"{STUDY_NAME_PREFIX}_{int(time.time())}"
    storage_name = args.storage_db
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,
    )

    logger.info(f"Study Name: {study.study_name}")
    logger.info(f"Storage: {storage_name}")
    logger.info(f"Direction: {study.direction}")
    logger.info(f"Pruner: {study.pruner.__class__.__name__}")
    logger.info(f"Number of Trials: {args.n_trials}")
    logger.info(f"Parallel Jobs (Workers): {args.n_jobs}")
    logger.info(f"Timeout: {args.timeout} seconds" if args.timeout else "No timeout")

    # --- Run Optimization ---
    try:
        study.optimize(
            lambda trial: objective(trial, base_config),
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
        )
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}", exc_info=True)

    # --- Results ---
    logger.info("\n--- Optimization Finished ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")

    if study.best_trial:
        logger.info(f"Best trial number: {study.best_trial.number}")
        logger.info(f"Best value (metric): {study.best_trial.value}")
        logger.info("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"  {key}: {value}")

        # --- Save Best Parameters ---
        best_params_path = output_dir / "best_params.yaml"
        try:
            with open(best_params_path, "w", encoding="utf-8") as f:
                yaml.dump(study.best_trial.params, f, default_flow_style=False)
            logger.info(f"Best parameters saved to: {best_params_path}")
        except Exception as e:
            logger.error(f"Failed to save best parameters: {e}")

        # --- Generate Visualizations ---
        logger.info("Generating visualizations...")
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        try:
            fig_history = plot_optimization_history(study)
            fig_history.write_image(viz_dir / "optimization_history.png")

            # ##: Plot importance only if trials completed successfully.
            if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
                try:
                    fig_importance = plot_param_importances(study)
                    fig_importance.write_image(viz_dir / "param_importances.png")
                except Exception as e:
                    logger.warning(f"Could not generate param importances plot: {e}")

                # ##: Contour plot requires at least 2 params and completed trials.
                if len(study.best_trial.params) >= 2:
                    try:
                        importances = optuna.importance.get_param_importances(study)
                        param_names = [p[0] for p in importances][:2]
                        if len(param_names) == 2:
                            fig_contour = plot_contour(study, params=param_names)
                            fig_contour.write_image(viz_dir / "contour_plot.png")
                    except Exception as e:
                        logger.warning(f"Could not generate contour plot: {e}")

            # ##: Parallel coordinate plot requires completed trials.
            if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
                fig_parallel = plot_parallel_coordinate(study)
                fig_parallel.write_image(viz_dir / "parallel_coordinate.png")

            # ##: Slice plot requires completed trials.
            if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
                try:
                    importances_slice = optuna.importance.get_param_importances(study)
                    param_names_slice = [p[0] for p in importances_slice]
                    if param_names_slice:
                        fig_slice = plot_slice(study, params=param_names_slice)
                        fig_slice.write_image(viz_dir / "slice_plot.png")
                except Exception as e:
                    logger.warning(f"Could not generate slice plot: {e}")

            logger.info(f"Visualizations saved to: {viz_dir}")

        except ImportError:
            logger.error("Failed to generate visualizations. Please install plotly and kaleido:")
            logger.error("  pip install plotly kaleido")
        except Exception as e:
            logger.error(f"An unexpected error occurred during visualization generation: {e}")

    else:
        logger.warning("No trials were completed successfully.")


def main():
    """Parses command-line arguments and starts the optimization."""
    parser = argparse.ArgumentParser(description="Run PPO Hyperparameter Optimization using Optuna.")

    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Number of optimization trials to run.")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS, help="Number of parallel jobs (workers).")
    parser.add_argument(
        "--storage-db",
        type=str,
        default=DEFAULT_STORAGE_DB,
        help="Optuna storage database URL (e.g., sqlite:///study.db).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save results (best params, visualizations).",
    )
    parser.add_argument(
        "--base-config-yaml",
        type=str,
        default=None,
        help="Path to a base YAML config file to load default parameters.",
    )

    args = parser.parse_args()
    run_optimization(args)


if __name__ == "__main__":
    main()
