[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Reinforce MiniGrid Maze

A reinforcement learning project implementing multiple algorithms to solve custom maze environments built on the [MiniGrid](https://github.com/Farama-Foundation/Minigrid) framework. Agents learn to navigate increasingly complex mazes through curriculum learning.

**Implemented Algorithms:**
- **PPO** (Proximal Policy Optimization) with RND intrinsic motivation and hybrid exploration
- **Rainbow DQN** with all six enhancements (Dueling, Double DQN, PER, N-step, Noisy Networks, C51)

Inspired by [MiniGrid](https://github.com/Farama-Foundation/Minigrid) and [BabyAI](https://github.com/mila-iqia/babyai).

## Features

*   **Multi-Algorithm Support:** Factory-based architecture supporting both PPO and Rainbow DQN
*   **Custom Maze Environments:** Four difficulty levels (`BaseMaze`, `EasyMaze`, `MediumMaze`, `HardMaze`) with adaptive reward shaping
*   **PPO Implementation:**
    *   IMPALA CNN architecture with residual blocks
    *   RND (Random Network Distillation) for intrinsic curiosity rewards
    *   Hybrid exploration (ε-greedy + UCB action tracking + adaptive entropy)
    *   GAE buffer with vectorized parallel environments
*   **Rainbow DQN Implementation:**
    *   Dueling architecture, Double Q-learning, Prioritized Experience Replay
    *   N-step returns, Noisy Networks, Categorical DQN (C51)
*   **Curriculum Learning:** Progressive difficulty (Base → Easy → Medium → Hard) based on performance thresholds
*   **GPU Optimized:** Large batch sizes (PPO: 512, DQN: 256) with dataset prefetching for maximum GPU utilization
*   **Configuration:** YAML-based config with Pydantic models and CLI overrides

## Project Structure

```bash
.
├── Makefile               # Convenience commands (train, visualize, lint, format)
├── manual.py              # Script for manual environment control
├── pyproject.toml         # Project metadata and dependencies (uv)
├── uv.lock                # uv lock file
├── configs/               # Configuration files
│   ├── default_training.yaml
│   └── kaggle_training.yaml
├── maze/                  # Custom MiniGrid maze environments
│   └── envs/              # Base, Easy, Medium, Hard maze definitions
├── models/                # Saved model checkpoints (created during training)
└── reinforce/             # Multi-algorithm RL implementation
    ├── core/              # Shared components (BaseAgent, BaseBuffer, schedules)
    ├── ppo/               # PPO: agent, buffer, network, RND, exploration
    ├── dqn/               # Rainbow DQN: agent, buffer, network, losses
    ├── config/            # Pydantic config models + YAML loader
    ├── factory.py         # Algorithm-agnostic agent creation
    ├── train.py           # Unified training script with curriculum
    └── visualize.py       # Evaluation and GIF rendering
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

1.  **Install uv:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
2.  **Install Dependencies:**
    ```bash
    make install-deps
    # or: uv sync
    ```

## Usage

### Training

```bash
# Train with PPO (default)
make train
make train-ppo TOTAL_TIMESTEPS=5000000 NUM_ENVS=8

# Train with Rainbow DQN
make train-dqn

# Train with specific algorithm
make train ALGORITHM=dqn
```

Training progress is logged to the console. Models are saved periodically to `models/` with performance metrics.

### Visualization

```bash
make visualize LEVEL=easy MODEL_PREFIX=./models/ppo_maze_final ALGORITHM=ppo
```

Generates `evaluation_render.gif` showing the trained agent's performance.

### Manual Control

```bash
make manual  # Test environments with keyboard controls
```

### Configuration

Edit `configs/default_training.yaml` or use CLI overrides:

```bash
python -m reinforce.train --algorithm dqn --dqn-lr 1e-4 --num-envs 16
```

Key parameters: `total_timesteps`, `num_envs`, `learning_rate`, `gamma`, `batch_size`. See `reinforce/train.py --help` for all options.

## License

This project is licensed under the [MIT License](LICENSE).
