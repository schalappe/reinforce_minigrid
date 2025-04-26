[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Reinforce MiniGrid Maze

This project implements a reinforcement learning agent using Proximal Policy Optimization (PPO) to solve custom maze environments built upon the [MiniGrid](https://github.com/Farama-Foundation/Minigrid) framework. The agent learns to navigate increasingly complex mazes through a curriculum learning approach.

This work is inspired by the concepts and environments found in [MiniGrid](https://github.com/Farama-Foundation/Minigrid) and [BabyAI](https://github.com/mila-iqia/babyai).

## Features

*   **Custom Maze Environments:** Includes several maze environments (`BaseMaze`, `EasyMaze`, `MediumMaze`, `HardMaze`) with varying difficulties located in the `maze/` directory.
*   **PPO Agent:** A PPO agent implemented using TensorFlow/Keras, located in the `reinforce/` directory.
    *   Actor-Critic architecture with CNN + LSTM layers (`reinforce/network.py`).
    *   Generalized Advantage Estimation (GAE) buffer (`reinforce/buffer.py`).
    *   Core PPO algorithm logic (`reinforce/ppo.py`).
*   **Curriculum Learning:** The training script (`reinforce/train.py`) employs a curriculum strategy, starting with simpler mazes and progressing to harder ones based on agent performance.
*   **Configuration:** Training parameters are managed via YAML configuration files (see `configs/default_training.yaml`).
*   **Training & Evaluation:**
    *   Train the agent using `make train`.
    *   Visualize a trained agent's performance and save it as a GIF using `make visualize` (`reinforce/visualize.py`).
*   **Manual Control:** Interactively control the agent in the environment using `make manual` (`manual.py`).
*   **Project Management:** A `Makefile` provides convenient commands for installation, training, visualization, linting, formatting, and cleaning.

## Project Structure

```
.
├── Makefile          # Convenience commands for project tasks
├── manual.py         # Script for manual environment control
├── poetry.lock       # Poetry lock file
├── pyproject.toml    # Project metadata and dependencies (Poetry)
├── README.md         # This file
├── configs/          # Configuration files (e.g., training parameters)
│   └── default_training.yaml
├── maze/             # Custom MiniGrid maze environment definitions
│   ├── __init__.py
│   └── envs/
│       ├── __init__.py
│       ├── base_maze.py
│       ├── easy_maze.py
│       ├── hard_maze.py
│       ├── maze.py
│       └── medium_maze.py
├── models/           # Directory for saving trained models (created during training)
├── reinforce/        # PPO agent implementation and training logic
│   ├── __init__.py
│   ├── agent.py      # PPO Agent class
│   ├── buffer.py     # Experience replay buffer (GAE)
│   ├── network.py    # Actor-Critic network definition
│   ├── ppo.py        # Core PPO algorithm functions
│   ├── train.py      # Main training script with curriculum learning
│   ├── visualize.py  # Script for evaluating and rendering agent performance
│   └── config/       # Configuration loading utilities
│       ├── __init__.py
│       ├── config_loader.py
│       └── training_config.py
└── tests/            # Unit tests (optional)
    ├── test_game.py
    └── test_model.py
```

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Install Poetry:** Follow the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).
2.  **Install Dependencies:** Navigate to the project root directory and run:
    ```bash
    make install-deps
    # or directly using poetry
    # poetry install
    ```

## Usage

### Training

To train the PPO agent using the default configuration and curriculum:

```bash
make train
```

You can override default parameters via the Makefile or command line arguments. See `reinforce/train.py --help` and the `Makefile` for details. Training progress is logged to the console, and models are saved periodically to the `models/` directory.

### Visualization

To visualize a trained agent's performance in a specific maze level (e.g., 'easy') and save it as a GIF:

```bash
make visualize LEVEL=easy MODEL_PREFIX=./models/ppo_maze_final_ts<timestamp>_stage<stage_index>
```

Replace `<timestamp>` and `<stage_index>` with the appropriate values from your saved model filenames. The output GIF will be saved as `evaluation_render.gif` by default. See `reinforce/visualize.py --help` and the `Makefile` for more options.

### Manual Control

To manually control the agent in the `HardMaze` environment using keyboard inputs:

```bash
make manual
```

Run `python manual.py --action` to see the available keyboard controls.

### Configuration

Training hyperparameters and settings can be modified in `configs/default_training.yaml` or by providing command-line overrides during training (see `make train` command and `reinforce/train.py`).

## License

This project is licensed under the [MIT License](LICENSE).
