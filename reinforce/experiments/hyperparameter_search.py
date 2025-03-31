# -*- coding: utf-8 -*-
"""
Hyperparameter search for reinforcement learning experiments.
"""

import argparse
import itertools
import json
import os
from typing import Any, Dict, Iterator, List, Optional

from reinforce.configs import ConfigManager
from reinforce.experiments.experiment_runner import ExperimentRunner

# ##: TODO: Use pathlib for path manipulations.
# ##: TODO: Use optuna for hyperparameter search.


class HyperparameterSearch:
    """
    Search over hyperparameters for reinforcement learning experiments.

    This class implements a grid search over hyperparameters for reinforcement learning experiments.
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
        self.results_dir = "outputs/hyperparameter_search"
        os.makedirs(self.results_dir, exist_ok=True)

    def run_search(self, search_config_path: str) -> Dict[str, Any]:
        """
        Run a hyperparameter search.

        Parameters
        ----------
        search_config_path : str
            Path to the search configuration file.

        Returns
        -------
        Dict[str, Any]
            Dictionary of search results.
        """
        search_config = self.config_manager.load_config(search_config_path)

        param_grid = self._generate_param_grid(search_config["hyperparameters"])

        base_config = search_config["base_config"]
        if isinstance(base_config, str):
            base_config = self.config_manager.load_config(base_config)

        results = []
        for params in param_grid:
            experiment_config = self._create_experiment_config(base_config, params)

            experiment_id = self._get_experiment_id(params)
            config_path = os.path.join(self.results_dir, f"{experiment_id}_config.yaml")
            self.config_manager.save_config(experiment_config, config_path)

            print(f"Running experiment {experiment_id}...")
            experiment_results = self.experiment_runner.run_experiment(config_path)

            experiment_results["hyperparameters"] = params

            results.append(experiment_results)
            results_path = os.path.join(self.results_dir, f"{experiment_id}_results.json")
            self.config_manager.save_config(experiment_results, results_path)

        best_result = max(results, key=lambda r: r.get("mean_reward", 0))

        summary = {
            "num_experiments": len(results),
            "best_experiment": self._get_experiment_id(best_result["hyperparameters"]),
            "best_mean_reward": best_result.get("mean_reward", 0),
            "best_hyperparameters": best_result["hyperparameters"],
            "all_experiments": [
                {
                    "id": self._get_experiment_id(r["hyperparameters"]),
                    "mean_reward": r.get("mean_reward", 0),
                    "hyperparameters": r["hyperparameters"],
                }
                for r in results
            ],
        }

        search_name = os.path.splitext(os.path.basename(search_config_path))[0]
        summary_path = os.path.join(self.results_dir, f"{search_name}_summary.json")
        self.config_manager.save_config(summary, summary_path)

        print("Hyperparameter search complete!")
        print(f"Number of experiments: {summary['num_experiments']}")
        print(f"Best experiment: {summary['best_experiment']}")
        print(f"Best mean reward: {summary['best_mean_reward']:.2f}")
        print(f"Best hyperparameters: {json.dumps(summary['best_hyperparameters'], indent=2)}")

        return summary

    @staticmethod
    def _generate_param_grid(hyperparameters: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
        """
        Generate a grid of hyperparameters.

        Parameters
        ----------
        hyperparameters : Dict[str, List[Any]]
            Dictionary mapping parameter names to lists of values.

        Returns
        -------
        Iterator[Dict[str, Any]]
            Iterator over dictionaries of hyperparameter combinations.
        """
        param_names = list(hyperparameters.keys())
        param_values = list(hyperparameters.values())

        for values in itertools.product(*param_values):
            yield dict(zip(param_names, values))

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

    @staticmethod
    def _get_experiment_id(hyperparameters: Dict[str, Any]) -> str:
        """
        Generate a unique ID for an experiment.

        Parameters
        ----------
        hyperparameters : Dict[str, Any]
            Hyperparameters for the experiment.

        Returns
        -------
        str
            Unique experiment ID.
        """
        param_str = "_".join(f"{k}={v}" for k, v in sorted(hyperparameters.items()))
        return f"experiment_{hash(param_str) % 10000:04d}"


def main():
    """
    Run a hyperparameter search from the command line.
    """
    # ##: Parse command line arguments.
    parser = argparse.ArgumentParser(description="Run a hyperparameter search for reinforcement learning experiments")
    parser.add_argument("config", help="Path to the search configuration file")
    parser.add_argument("--config-dir", help="Directory containing configuration files")
    args = parser.parse_args()

    # #: Create hyperparameter search.
    search = HyperparameterSearch(args.config_dir)

    # ##: Run the search.
    search.run_search(args.config)


if __name__ == "__main__":
    main()
