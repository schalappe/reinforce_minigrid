# -*- coding: utf-8 -*-
"""
Hyperparameter search for reinforcement learning experiments using Optuna.
"""

import json
import multiprocessing
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional, Union

import optuna
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)

from reinforce.configs import ConfigManager
from reinforce.experiments.experiment_runner import ExperimentRunner


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
            Directory containing configuration files.
        """
        self.config_manager = ConfigManager(config_dir)
        self.experiment_runner = ExperimentRunner(config_dir)
        self.results_dir = Path("outputs/hyperparameter_search")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.study = None
        self.search_config = None
        self.base_config = None
        self.search_name = None

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
            Timeout in seconds. None means no timeout, by default ``None``.

        Returns
        -------
        Dict[str, Any]
            Dictionary of search results.
        """
        # ##: Load search configuration.
        self.search_config = self.config_manager.load_config(str(search_config_path))
        self.search_name = Path(search_config_path).stem

        # ##: Load base configuration.
        base_config = self.search_config.get("base_config", {})
        if isinstance(base_config, str):
            self.base_config = self.config_manager.load_config(str(base_config))
        else:
            self.base_config = base_config if base_config else {}

        # ##: Set actual number of jobs (handle -1 case).
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        # ##: Configure storage for study persistence.
        self.results_dir.mkdir(parents=True, exist_ok=True)
        storage_name = f"sqlite:///{self.results_dir}/{self.search_name}.db"

        # ##: Setup pruning.
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)

        # ##: Create study.
        try:
            self.study = optuna.create_study(
                study_name=self.search_name,
                storage=storage_name,
                load_if_exists=True,
                direction="maximize",
                pruner=pruner,
            )
        except Exception as exc:
            print(f"Could not create study with persistent storage: {exc}")
            print("Using in-memory storage instead")
            self.study = optuna.create_study(study_name=self.search_name, direction="maximize", pruner=pruner)

        # ##: Run optimization.
        self.study.optimize(
            lambda trial: self._objective(trial),
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=True,
        )

        # ##: Prepare summary.
        best_params = self.study.best_params
        best_value = self.study.best_value

        all_trials = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                all_trials.append(
                    {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                    }
                )

        summary = {
            "num_experiments": len(all_trials),
            "best_experiment": f"trial_{self.study.best_trial.number}",
            "best_mean_reward": best_value,
            "best_hyperparameters": best_params,
            "all_experiments": all_trials,
        }

        # ##: Save summary.
        summary_path = self.results_dir / f"{self.search_name}_summary.json"
        self.config_manager.save_config(summary, str(summary_path))

        # ##: Generate and save visualizations.
        self._save_visualizations()

        # ##: Print results.
        print("Hyperparameter search complete!")
        print(f"Number of trials: {summary['num_experiments']}")
        print(f"Best trial: {summary['best_experiment']}")
        print(f"Best mean reward: {summary['best_mean_reward']:.4f}")
        print(f"Best hyperparameters: {json.dumps(summary['best_hyperparameters'], indent=2)}")
        print(f"Visualization plots saved to: {self.results_dir}")

        return summary

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial object.

        Returns
        -------
        float
            Mean reward achieved in this trial.
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
            print("Warning: No base configuration specified, using empty configuration")
            self.base_config = {}

        # ##: Create experiment config from base config and parameters.
        experiment_config = self._create_experiment_config(self.base_config, params)
        if not isinstance(experiment_config, dict):
            experiment_config = {}
            print(
                f"Warning: Got invalid experiment configuration of type {type(experiment_config)}, using empty configuration"
            )

        # ##: Generate unique ID for this trial.
        experiment_id = f"trial_{trial.number}"

        # ##: Add trial info to experiment config - ensure we have a valid config first.
        if experiment_config is None:
            experiment_config = {}
            print("Warning: experiment_config is None, using empty dictionary")
        experiment_config["_trial_info"] = {"number": trial.number}

        # ##: Save configuration.
        config_path = self.results_dir / f"{experiment_id}_config.yaml"
        self.config_manager.save_config(experiment_config, str(config_path))

        print(f"Running trial {trial.number} with parameters: {params}")

        # ##: Setup pruning callback.
        def pruning_callback(step, value):
            try:
                trial.report(value, step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            except Exception as e:
                print(f"Pruning error: {e}")

        # ##: Get intermediate values for pruning during experiment run.
        experiment_results = self.experiment_runner.run_experiment(config_path, pruning_callback=pruning_callback)

        # ##: Check if the trial was pruned.
        if experiment_results.get("pruned", False):
            raise optuna.exceptions.TrialPruned()

        # ##: Save results.
        experiment_results["hyperparameters"] = params
        experiment_results["trial_number"] = trial.number
        results_path = self.results_dir / f"{experiment_id}_results.json"
        self.config_manager.save_config(experiment_results, str(results_path))

        # ##: Return mean reward.
        return experiment_results.get("mean_reward", 0)

    @staticmethod
    def _pruning_callback(trial: optuna.Trial, step: int, value: float) -> None:
        """
        Callback for pruning under-performing trials.

        Parameters
        ----------
        trial : optuna.Trial
            Current trial.
        step : int
            Current training step.
        value : float
            Current observed value (reward).
        """
        trial.report(value, step)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    def _save_visualizations(self) -> None:
        """
        Generate and save optimization visualizations.
        """
        if self.study is None:
            print("No study available for visualization")
            return

        # ##: Create visualization directory.
        vis_dir = self.results_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        # ##: Plot optimization history.
        try:
            fig = plot_optimization_history(self.study)
            fig.write_image(str(vis_dir / "optimization_history.png"))
        except Exception as e:
            print(f"Could not generate optimization history plot: {e}")

        # ##: Plot parameter importances if we have completed trials.
        try:
            fig = plot_param_importances(self.study)
            fig.write_image(str(vis_dir / "param_importances.png"))
        except Exception as e:
            print(f"Could not generate parameter importance plot: {e}")

        # ##: Plot slice plot for key parameters.
        try:
            fig = plot_slice(self.study)
            fig.write_image(str(vis_dir / "slice_plot.png"))
        except Exception as e:
            print(f"Could not generate slice plot: {e}")

        # ##: Plot contour plot for parameter interactions.
        try:
            fig = plot_contour(self.study)
            fig.write_image(str(vis_dir / "contour_plot.png"))
        except Exception as e:
            print(f"Could not generate contour plot: {e}")

    @staticmethod
    def _create_experiment_config(base_config: Dict[str, Any], hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an experiment configuration from a base configuration and hyperparameters.

        Parameters
        ----------
        base_config : Dict[str, Any]
            Base configuration.
        hyperparameters : Dict[str, Any]
            Hyperparameters to apply.

        Returns
        -------
        Dict[str, Any]
            Combined experiment configuration.
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
    Run a hyperparameter search from the command line.
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
