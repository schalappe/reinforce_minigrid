# -*- coding: utf-8 -*-
"""
Hyperparameter search for reinforcement learning experiments using Optuna.

This module provides a class for running hyperparameter optimization studies using the Optuna framework.
It supports parallel execution, pruning of underperforming trials, and visualization of results.
"""

import json
from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path
from traceback import format_exc
from typing import Any, Dict, Optional, Union

from aim import Image
from optuna import Study, create_study
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.trial import Trial, TrialState
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)

from reinforce.configs import ConfigManager
from reinforce.experiments import ExperimentRunner
from reinforce.utils import AimLogger

logger = getLogger(__name__)


class HyperparameterSearch:
    """
    Search over hyperparameters for reinforcement learning experiments using Optuna.

    This class implements an efficient hyperparameter optimization using Optuna framework
    with support for parallel execution, pruning, and visualization.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the hyperparameter search.

        Parameters
        ----------
        config_dir : str, optional
            Directory containing configuration files. If None, default paths are used.
        """
        self.config_manager = ConfigManager(config_dir)
        self.experiment_runner = ExperimentRunner(config_dir)
        self.results_dir = Path("outputs/hyperparameter_search")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.study: Optional[Study] = None
        self.search_config: Optional[Dict[str, Any]] = None
        self.base_config: Optional[Dict[str, Any]] = None
        self.search_name: Optional[str] = None
        self.search_aim_logger: Optional[AimLogger] = None

    def run_search(
        self, search_config_path: Union[str, Path], n_trials: int = 20, n_jobs: int = -1, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a hyperparameter search using Optuna.

        Parameters
        ----------
        search_config_path : str | Path
            Path to the search configuration file.
        n_trials : int, optional
            Number of trials to run, by default 20.
        n_jobs : int, optional
            Number of parallel jobs. -1 means using all available cores, by default -1.
        timeout : int, optional
            Timeout in seconds. None means no timeout, by default None.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing search results including:
            - num_completed_trials: Number of successfully completed trials
            - best_trial_number: Index of the best performing trial
            - best_mean_reward: Best mean reward achieved
            - best_hyperparameters: Parameters of the best trial
            - all_completed_trials: List of all completed trials
            - study_name: Name of the Optuna study
            - direction: Optimization direction ('maximize' or 'minimize')

        Raises
        ------
        FileNotFoundError
            If the search configuration file is not found.
        ValueError
            If the search configuration is invalid.
        RuntimeError
            If the Optuna study fails to initialize or optimize.
        """
        # ##: Initialize AIM logger for the overall search process.
        search_config_name = Path(search_config_path).stem
        self.search_aim_logger = AimLogger(
            experiment_name=f"HyperparameterSearch_{search_config_name}",
            tags=["hyperparameter-search", "summary", search_config_name],
        )

        # ##: Log search setup parameters.
        if self.search_aim_logger.run:
            self.search_aim_logger.log_params(
                {
                    "search_config_path": str(search_config_path),
                    "n_trials": n_trials,
                    "n_jobs": n_jobs,
                    "timeout": timeout,
                    "search_name": search_config_name,
                },
                prefix="search_setup",
            )
        else:
            logger.warning("Failed to initialize AIM logger for search summary.")

        # ##: Load search configuration.
        try:
            self.search_config = self.config_manager.load_config(str(search_config_path))
            self.search_name = search_config_name

            if self.search_aim_logger.run and self.search_config:
                self.search_aim_logger.log_params(self.search_config.get("hyperparameters", {}), prefix="search_space")

        except Exception as exc:
            logger.error("Error loading search config: %s", exc)
            if self.search_aim_logger:
                self.search_aim_logger.close()
            raise

        # ##: Load base configuration.
        try:
            base_config_source = self.search_config.get("base_config", {})
            if isinstance(base_config_source, str):
                self.base_config = self.config_manager.load_config(str(base_config_source))
            else:
                self.base_config = base_config_source if base_config_source else {}

            if self.search_aim_logger.run:
                log_val = str(base_config_source) if isinstance(base_config_source, str) else "inline_dict"
                self.search_aim_logger.log_params({"base_config_source": log_val}, prefix="search_setup")

        except Exception as exc:
            logger.error("Error loading base config: %s", exc)
            if self.search_aim_logger:
                self.search_aim_logger.close()
            raise

        # ##: Configure storage for study persistence.
        self.results_dir.mkdir(parents=True, exist_ok=True)
        storage_name = f"sqlite:///{self.results_dir}/{self.search_name}.db"

        # ##: Setup pruning.
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)

        # ##: Create study.
        try:
            self.study = create_study(
                study_name=self.search_name,
                storage=storage_name,
                load_if_exists=True,
                direction="maximize",
                pruner=pruner,
            )
        except Exception as exc:
            logger.error("Could not create study with persistent storage: %s", exc)
            logger.error("Using in-memory storage instead")
            self.study = create_study(study_name=self.search_name, direction="maximize", pruner=pruner)

        # ##: Run optimization.
        try:
            self.study.optimize(
                lambda trial: self._objective(trial),
                n_trials=n_trials,
                n_jobs=n_jobs,
                timeout=timeout,
                show_progress_bar=True,
            )
        except Exception as exc:
            logger.error("Optuna optimization failed: %s", exc)
            if self.search_aim_logger and self.search_aim_logger.run:
                self.search_aim_logger.log_text(f"Optimization failed: {exc}\n{format_exc()}", name="error_log")
            if self.search_aim_logger:
                self.search_aim_logger.close()
            raise

        # ##: Prepare summary.
        best_params = {}
        best_value = None
        if self.study.best_trial:
            best_params = self.study.best_trial.params
            best_value = self.study.best_trial.value
        else:
            logger.info("No best trial found (optimization might have failed or yielded no results).")

        all_trials = []
        for trial in self.study.trials:
            if trial.state == TrialState.COMPLETE:
                all_trials.append(
                    {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                    }
                )

        summary = {
            "num_completed_trials": len(all_trials),
            "best_trial_number": self.study.best_trial.number if self.study.best_trial else None,
            "best_mean_reward": best_value,
            "best_hyperparameters": best_params,
            "all_completed_trials": all_trials,
            "study_name": self.study.study_name,
            "direction": str(self.study.direction),
        }

        # ##: Save summary locally.
        summary_path = self.results_dir / f"{self.search_name}_summary.json"
        self.config_manager.save_config(summary, str(summary_path))

        # ##: Log summary to the search-level AIM run.
        if self.search_aim_logger and self.search_aim_logger.run:
            self.search_aim_logger.log_params(summary, prefix="search_summary")

            if best_value is not None:
                self.search_aim_logger.log_metric("best_mean_reward", best_value)

        # ##: Generate, save, and log visualizations.
        self._save_and_log_visualizations()

        # ##: Print results.
        logger.info("Hyperparameter search complete!")
        logger.info("Number of trials: %s", summary["num_completed_trials"])
        logger.info("Best trial: %s", summary["best_trial_number"])
        logger.info("Best mean reward: %f:.4f", summary["best_mean_reward"])
        logger.info("Best hyperparameters: %s", json.dumps(summary["best_hyperparameters"], indent=2))
        logger.info("Visualization plots saved to: %s", self.results_dir / "visualizations")

        # ##: Close the search-level AIM logger.
        if self.search_aim_logger:
            self.search_aim_logger.close()

        return summary

    def _objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.

        This method is called by Optuna for each trial to evaluate a set of hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial object containing hyperparameter sampling methods.

        Returns
        -------
        float
            Mean reward achieved in this trial (the optimization objective).

        Raises
        ------
        TrialPruned
            If the trial is pruned due to poor performance.
        RuntimeError
            If the experiment fails to run.
        """
        # ##: Sample hyperparameters from search space.
        params = {}
        for param_path, param_config in self.search_config["hyperparameters"].items():
            param_type = param_config.get("type", "categorical")

            if param_type == "categorical":
                params[param_path] = trial.suggest_categorical(param_path, param_config["values"])
            elif param_type == "float":
                log_scale = param_config.get("log_scale", False)
                params[param_path] = trial.suggest_float(
                    param_path, param_config["low"], param_config["high"], log=log_scale
                )
            elif param_type == "int":
                log_scale = param_config.get("log_scale", False)
                params[param_path] = trial.suggest_int(
                    param_path, param_config["low"], param_config["high"], log=log_scale
                )

        # ##: Create experiment configuration.
        if not self.base_config:
            logger.warning("No base configuration specified, using empty configuration")
            self.base_config = {}

        # ##: Create experiment config from base config and parameters.
        experiment_config = self._create_experiment_config(self.base_config, params)
        if not isinstance(experiment_config, dict):
            experiment_config = {}
            logger.warning(
                "Got invalid experiment configuration of type %s, using empty configuration", type(experiment_config)
            )

        # ##: Generate unique ID and AIM details for this trial.
        experiment_id = f"trial_{trial.number}"
        aim_run_name = f"{self.search_name}_{experiment_id}"  # e.g., a2c_search_trial_0
        aim_tags = ["hyperparameter-search", "trial", self.search_name, f"trial_{trial.number}"]
        # Add agent type tag if available in base config
        agent_type = self.base_config.get("agent", {}).get("agent_type", "unknown_agent")
        aim_tags.append(agent_type)

        # ##: Add trial info and sampled params to experiment config for logging by ExperimentRunner.
        if experiment_config is None:
            experiment_config = {}
            logger.warning("experiment_config is None, using empty dictionary")
        experiment_config["_trial_info"] = {"number": trial.number, "optuna_params": trial.params}
        experiment_config["sampled_hyperparameters"] = params

        # ##: Save trial-specific configuration locally.
        config_path = self.results_dir / f"{experiment_id}_config.yaml"
        self.config_manager.save_config(experiment_config, str(config_path))

        logger.info("\n--- Running Trial %d ---", trial.number)
        logger.info("Parameters: %s", params)
        logger.info("AIM Run Name: %s", aim_run_name)

        # ##: Setup pruning callback.
        def pruning_callback(step, value):
            try:
                trial.report(value, step)
                if trial.should_prune():
                    logger.info("Trial %d pruned at step %s with value %s.", trial.number, step, value)
                    raise TrialPruned()
            except TrialPruned:
                raise
            except Exception as exc:
                logger.error("Error in pruning callback for trial %s: %s", trial.number, exc)

        # ##: Run the experiment for this trial, passing AIM details.
        try:
            experiment_results = self.experiment_runner.run_experiment(
                config_path, pruning_callback=pruning_callback, aim_tags=list(set(aim_tags))
            )
        except TrialPruned:
            logger.info("Trial %d was pruned.", trial.number)
            raise
        except Exception as exc:
            logger.error("Error running experiment for trial %d: %s", trial.number, exc)
            return -float("inf")

        # ##: Return the final mean reward (or relevant metric) for Optuna.
        objective_value = experiment_results.get("final_mean_reward", experiment_results.get("mean_reward", 0))
        logger.info("--- Trial %d Completed ---", trial.number)
        logger.info("Final Objective Value: %f:.4f", objective_value)
        return objective_value

    @staticmethod
    def _pruning_callback(trial: Trial, step: int, value: float) -> None:
        """
        Callback for pruning under-performing trials.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial being evaluated.
        step : int
            Current training step.
        value : float
            Current observed value (typically reward).

        Raises
        ------
        TrialPruned
            If the trial should be pruned based on intermediate results.
        """
        trial.report(value, step)

        if trial.should_prune():
            raise TrialPruned()

    def _save_and_log_visualizations(self) -> None:
        """
        Generate and save optimization visualizations.

        Creates several plots showing the optimization progress and saves them to disk.
        Also logs the visualizations to AIM if available.

        The following visualizations are generated:
        - Optimization history plot
        - Parameter importance plot
        - Slice plot
        - Contour plot

        Notes
        -----
        Requires plotly for visualization generation.s
        """
        if self.study is None:
            logger.warning("No study available for visualization")
            return

        # ##: Create visualization directory locally.
        vis_dir = self.results_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        plot_functions = {
            "optimization_history": plot_optimization_history,
            "param_importances": plot_param_importances,
            "slice": plot_slice,
            "contour": plot_contour,
        }

        for name, plot_func in plot_functions.items():
            try:
                fig = plot_func(self.study)
                img_path = vis_dir / f"{name}.png"
                fig.write_image(str(img_path))
                logger.info("Saved visualization: %s", img_path)

                # ##: Log the saved image to the search-level AIM run.
                if self.search_aim_logger and self.search_aim_logger.run:
                    try:
                        with open(img_path, "rb") as file:
                            aim_image = Image(file.read(), format="png", caption=f"Optuna {name} plot")
                            self.search_aim_logger.log_image(aim_image, name=f"optuna_{name}_plot")
                    except Exception as exc:
                        logger.error("Could not log visualization '%s' to AIM: %s", name, exc)

            except (ValueError, ImportError, RuntimeError) as exc:
                logger.error("Could not generate or save visualization '%s': %s", name, exc)
            except Exception as exc:
                logger.error("An unexpected error occurred during visualization '%s': %s", name, exc)

    @staticmethod
    def _create_experiment_config(base_config: Dict[str, Any], hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an experiment configuration from a base configuration and hyperparameters.

        Parameters
        ----------
        base_config : Dict[str, Any]
            Base configuration dictionary.
        hyperparameters : Dict[str, Any]
            Dictionary of hyperparameters to merge, with dot notation paths.

        Returns
        -------
        Dict[str, Any]
            Combined experiment configuration with hyperparameters merged into the base config.

        Examples
        --------
        >>> base = {"agent": {"lr": 0.01}}
        >>> params = {"agent.lr": 0.001, "env.name": "CartPole"}
        >>> _create_experiment_config(base, params)
        {'agent': {'lr': 0.001}, 'env': {'name': 'CartPole'}}
        """
        experiment_config = base_config.copy()

        for param_path, value in hyperparameters.items():
            components = param_path.split(".")

            current = experiment_config
            for component in components[:-1]:
                if component not in current:
                    current[component] = {}
                current = current[component]

            current[components[-1]] = value

        return experiment_config


def main():
    """
    Create an experiment configuration from a base configuration and hyperparameters.

    Parameters
    ----------
    base_config : Dict[str, Any]
        Base configuration dictionary.
    hyperparameters : Dict[str, Any]
        Dictionary of hyperparameters to merge, with dot notation paths.

    Returns
    -------
    Dict[str, Any]
        Combined experiment configuration with hyperparameters merged into the base config.

    Examples
    --------
    >>> base = {"agent": {"lr": 0.01}}
    >>> params = {"agent.lr": 0.001, "env.name": "CartPole"}
    >>> _create_experiment_config(base, params)
    {'agent': {'lr': 0.001}, 'env': {'name': 'CartPole'}}
    """
    # ##: Parse command line arguments.
    parser = ArgumentParser(description="Run a hyperparameter search for reinforcement learning experiments")
    parser.add_argument("config", help="Path to the search configuration file")
    parser.add_argument("--config-dir", help="Directory containing configuration files")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel jobs (-1 uses all cores)")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    args = parser.parse_args()

    # ##: Create hyperparameter search.
    search = HyperparameterSearch(args.config_dir)

    # ##: Run the search.
    search.run_search(args.config, n_trials=args.trials, n_jobs=args.jobs, timeout=args.timeout)


if __name__ == "__main__":
    main()
