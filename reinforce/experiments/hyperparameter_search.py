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
from reinforce.experiments.experiment_runner import ExperimentRunner
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

    def _setup_search_logging(self, search_config_path: Union[str, Path], n_trials: int) -> AimLogger:
        """
        Initialize AIM logger for the search and log setup parameters.

        Sets up an AimLogger instance to track the hyperparameter search process, logging search configuration
        details and setup parameters. The logger is configured with an experiment name derived from the search
        configuration file's stem.

        Parameters
        ----------
        search_config_path : Union[str, Path]
            Path to the hyperparameter search configuration file.
        n_trials : int
            Number of trials to run in the hyperparameter search.

        Returns
        -------
        AimLogger
            An AimLogger instance if the initialization was successful.

        Raises
        ------
        Exception
            If there are issues with initializing the AimLogger.
        """
        search_config_name = Path(search_config_path).stem
        search_aim_logger = AimLogger(
            experiment_name=f"HyperparameterSearch_{search_config_name}",
            tags=["hyperparameter-search", "summary", search_config_name],
        )

        if search_aim_logger.run:
            search_aim_logger.log_params(
                {
                    "search_config_path": str(search_config_path),
                    "n_trials": n_trials,
                    "search_name": search_config_name,
                },
                prefix="search_setup",
            )
            self.search_name = search_config_name
        else:
            logger.warning("Failed to initialize AIM logger for search summary.")
            self.search_name = search_config_name

        return search_aim_logger

    def _load_configs(self, search_config_path: Union[str, Path], search_aim_logger: Optional[AimLogger]) -> None:
        """
        Load search and base configurations.

        Loads the hyperparameter search configuration and the base experiment configuration. The base configuration
        can be specified as a file path in the search configuration, or provided inline as a dictionary. If an AimLogger
        instance is provided, the configurations are logged to AIM for tracking purposes.

        Parameters
        ----------
        search_config_path : Union[str, Path]
            Path to the hyperparameter search configuration file.
        search_aim_logger : Optional[AimLogger]
            Optional AimLogger instance for logging.

        Raises
        ------
        FileNotFoundError
            If the search or base configuration file is not found.
        Exception
            If there are issues loading the configurations.
        """
        try:
            self.search_config = self.config_manager.load_config(str(search_config_path))
            if search_aim_logger and search_aim_logger.run and self.search_config:
                search_aim_logger.log_params(self.search_config.get("hyperparameters", {}), prefix="search_space")
        except FileNotFoundError as exc:
            logger.error("Search config file not found: %s", exc)
            raise
        except Exception as exc:
            logger.error("Error loading search config: %s", exc)
            raise

        try:
            base_config_source = self.search_config.get("base_config", {})
            if isinstance(base_config_source, str):
                self.base_config = self.config_manager.load_config(str(base_config_source))
            else:
                self.base_config = base_config_source if isinstance(base_config_source, dict) else {}

            if search_aim_logger and search_aim_logger.run:
                log_val = str(base_config_source) if isinstance(base_config_source, str) else "inline_dict"
                search_aim_logger.log_params({"base_config_source": log_val}, prefix="search_setup")
        except FileNotFoundError as exc:
            logger.error("Base config file not found: %s", exc)
            raise
        except Exception as exc:
            logger.error("Error loading base config: %s", exc)
            raise

    def _create_or_load_study(self) -> None:
        """
        Create or load the Optuna study with storage and pruner.

        Creates an Optuna study for hyperparameter optimization, using persistent storage if available and
        a MedianPruner for early stopping of unpromising trials. The study is loaded from a SQLite database
        if it exists, allowing the optimization process to be resumed.

        Raises
        ------
        RuntimeError
            If the search name is not set before creating the study.
        ImportError
            If the database backend for Optuna storage is not installed.
        Exception
            If there are issues creating the study.
        """
        if not self.search_name:
            raise RuntimeError("Search name not set before creating study.")

        self.results_dir.mkdir(parents=True, exist_ok=True)
        storage_name = f"sqlite:///{self.results_dir}/{self.search_name}.db"
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)

        try:
            self.study = create_study(
                study_name=self.search_name,
                storage=storage_name,
                load_if_exists=True,
                direction="maximize",
                pruner=pruner,
            )
            logger.info("Using persistent storage for study: %s", storage_name)
        except ImportError as exc:
            logger.error("Database backend for Optuna storage not installed: %s", exc)
            logger.warning("Using in-memory storage instead.")
            self.study = create_study(study_name=self.search_name, direction="maximize", pruner=pruner)
        except Exception as exc:
            logger.error("Could not create study with persistent storage: %s", exc)
            logger.warning("Using in-memory storage instead.")
            self.study = create_study(study_name=self.search_name, direction="maximize", pruner=pruner)

    def _run_optimization(self, n_trials: int, search_aim_logger: Optional[AimLogger]) -> None:
        """
        Run the Optuna optimization process.

        Executes the Optuna optimization process, sampling hyperparameters and evaluating the objective function
        for each trial. The optimization process is monitored and any exceptions are logged to AIM, if available.

        Parameters
        ----------
        n_trials : int
            Number of trials to run in the hyperparameter search.
        search_aim_logger : Optional[AimLogger]
            Optional AimLogger instance for logging.

        Raises
        ------
        RuntimeError
            If the study is not initialized before running optimization.
        Exception
            If the Optuna optimization process fails.
        """
        if not self.study:
            raise RuntimeError("Study not initialized before running optimization.")

        try:
            self.study.optimize(self._objective, n_trials=n_trials, show_progress_bar=True)
        except Exception as exc:
            logger.error("Optuna optimization failed: %s", exc)
            if search_aim_logger and search_aim_logger.run:
                search_aim_logger.log_text(f"Optimization failed: {exc}\n{format_exc()}", name="error_log")
            raise

    def _summarize_and_save_results(self, search_aim_logger: Optional[AimLogger]) -> Dict[str, Any]:
        """
        Prepare, save, and log the search summary and visualizations.

        Collects the results of the hyperparameter search, including the best trial and all completed trials.
        The results are saved to a JSON file and logged to AIM, if available. Additionally, visualizations
        of the optimization process are generated and saved.

        Parameters
        ----------
        search_aim_logger : Optional[AimLogger]
            Optional AimLogger instance for logging.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the search summary.

        Raises
        ------
        RuntimeError
            If the study is not available for summarization.
        """
        if not self.study:
            raise RuntimeError("Study not available for summarization.")

        best_params = {}
        best_value = None
        best_trial_num = None

        if self.study.best_trial:
            best_params = self.study.best_trial.params
            best_value = self.study.best_trial.value
            best_trial_num = self.study.best_trial.number
        else:
            logger.info("No best trial found (optimization might have failed or yielded no results).")

        all_trials = [
            {"number": trial.number, "value": trial.value, "params": trial.params}
            for trial in self.study.trials
            if trial.state == TrialState.COMPLETE
        ]

        summary = {
            "num_completed_trials": len(all_trials),
            "best_trial_number": best_trial_num,
            "best_mean_reward": best_value,
            "best_hyperparameters": best_params,
            "all_completed_trials": all_trials,
            "study_name": self.study.study_name,
            "direction": str(self.study.direction),
        }

        # ##: Save summary locally.
        summary_path = self.results_dir / f"{self.search_name}_summary.json"
        self.config_manager.save_config(summary, str(summary_path))

        # ##: Log summary to AIM.
        if search_aim_logger and search_aim_logger.run:
            search_aim_logger.log_params(summary, prefix="search_summary")
            if best_value is not None:
                search_aim_logger.log_metric("best_mean_reward", best_value)

        # ##: Generate, save, and log visualizations.
        self._save_and_log_visualizations(search_aim_logger)

        # ##: Print results.
        logger.info("Hyperparameter search complete!")
        logger.info("Number of trials: %s", summary["num_completed_trials"])
        logger.info("Best trial: %s", summary["best_trial_number"])
        if summary["best_mean_reward"] is not None:
            logger.info("Best mean reward: %.4f", summary["best_mean_reward"])
        else:
            logger.info("Best mean reward: N/A")
        logger.info("Best hyperparameters: %s", json.dumps(summary["best_hyperparameters"], indent=2))
        logger.info("Visualization plots saved to: %s", self.results_dir / "visualizations")

        return summary

    def run_search(self, search_config_path: Union[str, Path], n_trials: int = 20) -> Dict[str, Any]:
        """
        Run a hyperparameter search using Optuna.

        Orchestrates the search process by calling helper methods for logging, config loading,
        study creation, optimization, and result summarization.

        Parameters
        ----------
        search_config_path : Union[str, Path]
            Path to the search configuration file.
        n_trials : int, default=20
            Number of trials to run.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the search summary.

        Raises
        ------
        FileNotFoundError
            If configuration files are not found.
        ValueError
            If configurations are invalid.
        RuntimeError
            If Optuna study or optimization encounters critical errors.
        """
        search_aim_logger = None
        try:
            search_aim_logger = self._setup_search_logging(search_config_path, n_trials)
            self._load_configs(search_config_path, search_aim_logger)

            # ##: Create or load study.
            self._create_or_load_study()
            self._run_optimization(n_trials, search_aim_logger)

            return self._summarize_and_save_results(search_aim_logger)

        except Exception as exc:
            if not isinstance(exc, RuntimeError) or "Optuna optimization failed" not in str(exc):
                logger.error("Hyperparameter search failed: %s\n%s", exc, format_exc())

            if search_aim_logger and search_aim_logger.run:
                search_aim_logger.log_text(f"Search failed: {exc}\n{format_exc()}", name="error_log")
            raise
        finally:
            if search_aim_logger:
                search_aim_logger.close()

    def _sample_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters for the current trial based on search config.

        Samples hyperparameters from the search space defined in the search configuration ile. The hyperparameters
        can be categorical, float, or integer, and can be sampled on a log scale.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial object.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the sampled hyperparameters.

        Raises
        ------
        ValueError
            If the configuration for a parameter is invalid.
        Exception
            If there are errors during the sampling process.
        """
        params = {}
        if not self.search_config or "hyperparameters" not in self.search_config:
            logger.warning("No hyperparameters defined in search config.")
            return params

        for param_path, param_config in self.search_config["hyperparameters"].items():
            param_type = param_config.get("type", "categorical")
            try:
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
                else:
                    logger.warning("Unsupported parameter type '%s' for %s", param_type, param_path)
            except KeyError as exc:
                logger.error("Missing required key '%s' for parameter %s in search config.", exc, param_path)
                raise ValueError(f"Invalid config for parameter {param_path}") from exc
            except Exception as exc:
                logger.error("Error suggesting parameter %s: %s", param_path, exc)
                raise

        return params

    def _prepare_trial_config(self, params: Dict[str, Any], trial: Trial) -> tuple[Path, list[str]]:
        """
        Prepare and save the configuration for a specific trial.

        Creates a trial-specific configuration by merging the sampled hyperparameters into the base configuration.
        The resulting configuration is saved to a YAML file. Additionally, metadata about the trial is added
        to the configuration.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of sampled hyperparameters for the current trial.
        trial : optuna.Trial
            Current trial object.

        Returns
        -------
        tuple[Path, list[str]]
            A tuple containing the path to the saved configuration file and a list of AIM tags.

        Raises
        ------
        TypeError
            If the experiment configuration is not a dictionary.
        IOError
            If there are issues saving the configuration file.
        Exception
            If there are unexpected errors during the configuration preparation.
        """
        if not self.base_config:
            logger.warning("No base configuration specified, using empty configuration")
            self.base_config = {}

        experiment_config = self._create_experiment_config(self.base_config, params)
        if not isinstance(experiment_config, dict):
            logger.error(
                "Failed to create a valid dictionary for experiment config. Base: %s, Params: %s",
                self.base_config,
                params,
            )
            raise TypeError(f"Expected experiment_config to be a dict, got {type(experiment_config)}")

        # ##: Generate unique ID and AIM details.
        experiment_id = f"trial_{trial.number}"
        aim_run_name = f"{self.search_name}_{experiment_id}"
        aim_tags = ["hyperparameter-search", "trial", str(self.search_name), f"trial_{trial.number}"]
        agent_type = self.base_config.get("agent", {}).get("agent_type", "unknown_agent")
        aim_tags.append(agent_type)

        # ##: Add trial info and sampled params to config.
        experiment_config["_trial_info"] = {"number": trial.number, "optuna_params": trial.params}
        experiment_config["sampled_hyperparameters"] = params

        # ##: Save trial-specific configuration.
        config_path = self.results_dir / f"{experiment_id}_config.yaml"
        try:
            self.config_manager.save_config(experiment_config, str(config_path))
        except IOError as exc:
            logger.error("Failed to save trial config %s: %s", config_path, exc)
            raise
        except Exception as exc:
            logger.error("Unexpected error saving trial config %s: %s", config_path, exc)
            raise

        logger.info("\n--- Running Trial %d ---", trial.number)
        logger.info("Parameters: %s", params)
        logger.info("AIM Run Name: %s", aim_run_name)

        return config_path, list(set(aim_tags))

    def _execute_trial(self, trial: Trial, config_path: Path, aim_tags: list[str]) -> float:
        """
        Define pruning callback and execute the experiment for the trial.

        Executes a single trial of the reinforcement learning experiment, using the trial-specific configuration.
        A pruning callback is defined to allow Optuna to prune unpromising trials early.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial object.
        config_path : Path
            Path to the trial-specific configuration file.
        aim_tags : list[str]
            List of AIM tags for the experiment.

        Returns
        -------
        float
            The objective value (mean reward) for the trial.

        Raises
        ------
        TrialPruned
            If the trial is pruned by the pruning callback.
        FileNotFoundError
            If the experiment configuration file is not found.
        Exception
            If there are errors during the experiment execution.
        """

        def pruning_callback(step: int, value: float):
            """Inner function for pruning callback."""
            try:
                trial.report(value, step)
                if trial.should_prune():
                    logger.info("Trial %d pruned at step %d with value %.4f.", trial.number, step, value)
                    raise TrialPruned()
            except TrialPruned:
                raise
            except Exception as exc:
                logger.error("Error in pruning callback for trial %d (step %d): %s", trial.number, step, exc)

        try:
            experiment_results = self.experiment_runner.run_experiment(
                config_path, pruning_callback=pruning_callback, aim_tags=aim_tags
            )

            objective_value = experiment_results.get("final_mean_reward", experiment_results.get("mean_reward", 0.0))
            logger.info("--- Trial %d Completed ---", trial.number)
            logger.info("Final Objective Value: %.4f", objective_value)
            return objective_value
        except TrialPruned:
            logger.info("Trial %d was pruned.", trial.number)
            raise
        except FileNotFoundError as exc:
            logger.error("Experiment config file not found for trial %d: %s", trial.number, exc)
            return -float("inf")
        except Exception as exc:
            logger.error("Error running experiment for trial %d: %s\n%s", trial.number, exc, format_exc())
            return -float("inf")

    def _objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.

        Orchestrates sampling, config preparation, and execution for a single trial.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial object.

        Returns
        -------
        float
            Mean reward (objective value) for the trial.
        """
        try:
            params = self._sample_hyperparameters(trial)
            config_path, aim_tags = self._prepare_trial_config(params, trial)

            return self._execute_trial(trial, config_path, aim_tags)

        except TrialPruned:
            raise
        except Exception as exc:
            logger.error("Critical error in objective function for trial %d: %s\n%s", trial.number, exc, format_exc())
            return -float("inf")

    def _save_and_log_visualizations(self, search_aim_logger: Optional[AimLogger]) -> None:
        """
        Generate and save optimization visualizations. Also logs them to AIM.

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
                if search_aim_logger and search_aim_logger.run:
                    try:
                        with open(img_path, "rb") as f_img:
                            aim_image = Image(f_img.read(), format="png", caption=f"Optuna {name} plot")
                            search_aim_logger.log_image(aim_image, name=f"optuna_{name}_plot")
                    except FileNotFoundError:
                        logger.error("Could not find visualization file '%s' to log to AIM.", img_path)
                    except IOError as io_err:
                        logger.error("Could not read visualization file '%s' for AIM logging: %s", img_path, io_err)
                    except Exception as aim_exc:
                        logger.error("Could not log visualization '%s' to AIM: %s", name, aim_exc)

            except ImportError:
                logger.warning("Plotly not installed. Skipping visualization '%s'.", name)
            except ValueError as ve:
                logger.warning("Could not generate visualization '%s' (possibly insufficient data): %s", name, ve)
            except RuntimeError as rte:
                logger.error("Runtime error generating visualization '%s': %s", name, rte)
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
        >>> HyperparameterSearch._create_experiment_config(base, params)
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
    Entry point for running hyperparameter search from the command line.

    Parses arguments, initializes the HyperparameterSearch class, and runs the search.
    """
    # ##: Parse command line arguments.
    parser = ArgumentParser(description="Run a hyperparameter search for reinforcement learning experiments")
    parser.add_argument("config", help="Path to the search configuration file")
    parser.add_argument("--config-dir", help="Directory containing configuration files")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to run")
    args = parser.parse_args()

    # ##: Create hyperparameter search.
    search = HyperparameterSearch(args.config_dir)

    # ##: Run the search.
    search.run_search(args.config, n_trials=args.trials)


if __name__ == "__main__":
    main()
