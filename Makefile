# Training parameters
TOTAL_TIMESTEPS ?= 3000000
STEPS_PER_UPDATE ?= 2048
NUM_ENVS ?= 5
CONFIG ?= configs/default_training.yaml
ALGORITHM ?= ppo

# Visualization parameters
MODEL_PREFIX ?= ./models/ppo_maze_final
OUTPUT_GIF ?= evaluation_render.gif
LEVEL ?= easy

.PHONY: all help train train-ppo train-dqn visualize manual install-deps lint format typecheck clean sync quality

all: help

help:
	@echo "Available targets:"
	@echo "  train       - Launch RL training (use ALGORITHM=ppo|dqn)"
	@echo "  train-ppo   - Launch PPO training with curriculum learning"
	@echo "  train-dqn   - Launch Rainbow DQN training"
	@echo "  visualize   - Generate evaluation GIF from trained model"
	@echo "  manual      - Start manual control interface"
	@echo "  install-deps- Install Python dependencies"
	@echo "  sync        - Sync dependencies (install + update lockfile)"
	@echo "  lint        - Run static code analysis (ruff)"
	@echo "  format      - Format source code (ruff)"
	@echo "  typecheck   - Run type checking (pyrefly)"
	@echo "  clean       - Remove generated files"

train:
	uv run python -m reinforce.train \
		--config $(CONFIG) \
		--algorithm $(ALGORITHM) \
		--total-timesteps $(TOTAL_TIMESTEPS) \
		--steps-per-update $(STEPS_PER_UPDATE) \
		--num-envs $(NUM_ENVS)

train-ppo:
	uv run python -m reinforce.train \
		--config $(CONFIG) \
		--algorithm ppo \
		--total-timesteps $(TOTAL_TIMESTEPS) \
		--steps-per-update $(STEPS_PER_UPDATE) \
		--num-envs $(NUM_ENVS)

train-dqn:
	uv run python -m reinforce.train \
		--config $(CONFIG) \
		--algorithm dqn \
		--total-timesteps $(TOTAL_TIMESTEPS) \
		--num-envs $(NUM_ENVS)

visualize:
	uv run python -m reinforce.visualize \
		--model-prefix $(MODEL_PREFIX) \
		--output-gif $(OUTPUT_GIF) \
		--level $(LEVEL) \
		--algorithm $(ALGORITHM)

manual:
	uv run python manual.py

install-deps:
	uv sync --all-groups

sync:
	uv sync --all-groups

lint:
	uv run ruff check maze/ reinforce/ tests/

format:
	uv run ruff format maze/ reinforce/ tests/
	uv run ruff check --fix maze/ reinforce/ tests/

typecheck:
	uv run pyrefly check maze/ reinforce/ tests/

quality: format lint typecheck

clean:
	rm -rf models/*.keras *.gif
