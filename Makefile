# Training parameters
TOTAL_TIMESTEPS ?= 3000000
STEPS_PER_UPDATE ?= 2048
NUM_ENVS ?= 5
CONFIG ?= configs/default_training.yaml

# Visualization parameters
MODEL_PREFIX ?= ./models/ppo_maze_final
OUTPUT_GIF ?= evaluation_render.gif
LEVEL ?= easy

.PHONY: all help train visualize manual install-deps lint format clean

all: help

help:
	@echo "Available targets:"
	@echo "  train       - Launch PPO training with curriculum learning"
	@echo "  visualize   - Generate evaluation GIF from trained model"
	@echo "  manual      - Start manual control interface"
	@echo "  install-deps- Install Python dependencies"
	@echo "  lint        - Run static code analysis"
	@echo "  format      - Format source code"
	@echo "  clean       - Remove generated files"

train:
	python -m reinforce.train \
		--config $(CONFIG) \
		--total-timesteps $(TOTAL_TIMESTEPS) \
		--steps-per-update $(STEPS_PER_UPDATE) \
		--num-envs $(NUM_ENVS)

visualize:
	python -m reinforce.visualize \
		--model-prefix $(MODEL_PREFIX) \
		--output-gif $(OUTPUT_GIF) \
		--level $(LEVEL)

manual:
	python manual.py

install-deps:
	poetry install

lint:
	pylint maze/ reinforce/ tests/

format:
	black maze/ reinforce/ tests/

clean:
	rm -rf models/*.keras *.gif
