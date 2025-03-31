# -*- coding: utf-8 -*-
"""
Experiment runner for reinforcement learning experiments.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional, Union

from reinforce.agents.a2c import A2CAgent
from reinforce.configs import ConfigManager
from reinforce.environments import MazeEnvironment
from reinforce.trainers.episode_trainer import EpisodeTrainer



class ExperimentRunner:
    """
    Runner for reinforcement learning experiments.

    This class sets up and runs reinforcement learning experiments based on configuration files.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the experiment runner.

        Parameters
        ----------
        config_dir : str, optional
            Directory containing configuration files.
        """
        self.config_manager = ConfigManager(config_dir)

    def run_experiment(self, experiment_config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Run an experiment.

        Parameters
        ----------
        experiment_config_path : str | Path
            Path to the experiment configuration file.

        Returns
        -------
        Dict[str, Any]
            Dictionary of experiment results.
        """
        # ##: Load experiment configuration.
        config = self.config_manager.load_experiment_config(str(experiment_config_path))

        # ##: Set up the environment, agent, and trainer.
        environment = self._create_environment(config.get("environment", {}))
        agent = self._create_agent(config.get("agent", {}), environment)
        trainer = self._create_trainer(config.get("trainer", {}), agent, environment)

        # ##: Run the experiment.
        results = trainer.train()

        # ##: Save results if specified.
        if "save_results" in config and config["save_results"]:
            results_dir = Path(config.get("results_dir", "outputs/results"))
            results_dir.mkdir(parents=True, exist_ok=True)

            experiment_name = Path(experiment_config_path).stem
            results_path = results_dir / f"{experiment_name}_results.json"

            self.config_manager.save_config(results, str(results_path))

        return results

    @staticmethod
    def _create_environment(env_config: Dict[str, Any]) -> MazeEnvironment:
        """
        Create an environment based on the configuration.

        Parameters
        ----------
        env_config : Dict[str, Any]
            Environment configuration.

        Returns
        -------
        MazeEnvironment
            Created environment.
        """
        use_image_obs = env_config.get("use_image_obs", True)
        return MazeEnvironment(use_image_obs=use_image_obs)

    @staticmethod
    def _create_agent(agent_config: Dict[str, Any], environment) -> A2CAgent:
        """
        Create an agent based on the configuration.

        Parameters
        ----------
        agent_config : Dict[str, Any]
            Agent configuration.
        environment : Environment
            Environment to use.

        Returns
        -------
        A2CAgent
            Created agent.

        Raises
        ------
        ValueError
            If the agent type is not supported.
        """
        agent_type = agent_config.get("agent_type", "A2C")

        if agent_type == "A2C":
            return A2CAgent(
                action_space=agent_config.get("action_space", environment.action_space.n),
                embedding_size=agent_config.get("embedding_size", 128),
                learning_rate=agent_config.get("learning_rate", 0.001),
                discount_factor=agent_config.get("discount_factor", 0.99),
                entropy_coef=agent_config.get("entropy_coef", 0.01),
                value_coef=agent_config.get("value_coef", 0.5),
            )

        raise ValueError(f"Unsupported agent type: {agent_type}")

    @staticmethod
    def _create_trainer(trainer_config: Dict[str, Any], agent, environment) -> EpisodeTrainer:
        """
        Create a trainer based on the configuration.

        Parameters
        ----------
        trainer_config : Dict[str, Any]
            Trainer configuration.
        agent : Agent
            Agent to train.
        environment : Environment
            Environment to train in.

        Returns
        -------
        EpisodeTrainer
            Created trainer.

        Raises
        ------
        ValueError
            If the trainer type is not supported.
        """
        trainer_type = trainer_config.get("trainer_type", "EpisodeTrainer")

        if trainer_type == "EpisodeTrainer":
            return EpisodeTrainer(agent=agent, environment=environment, config=trainer_config)

        raise ValueError(f"Unsupported trainer type: {trainer_type}")


def main():
    """
    Run an experiment from the command line.
    """
    # ##: Parse command line arguments.
    parser = ArgumentParser(description="Run a reinforcement learning experiment")
    parser.add_argument("config", help="Path to the experiment configuration file")
    parser.add_argument("--config-dir", help="Directory containing configuration files")
    args = parser.parse_args()

    # ##: Create experiment runner.
    runner = ExperimentRunner(args.config_dir)

    # ##: Run the experiment.
    results = runner.run_experiment(args.config)

    # ##: Print summary of results.
    print("Experiment complete!")
    print(f"Episodes: {results.get('episodes', 0)}")
    print(f"Total steps: {results.get('total_steps', 0)}")
    print(f"Mean reward: {results.get('mean_reward', 0):.2f}")
    print(f"Max reward: {results.get('max_reward', 0):.2f}")


if __name__ == "__main__":
    main()
